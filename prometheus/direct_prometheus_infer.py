from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
import torch
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_DIR)
load_dotenv(Path(PROJECT_ROOT) / ".env")

from config import DatasetConfig, GPTConfig, RESULTS_DIR
from evaluate import StreamingFiLMAnomalyDetector
from gpt_adjudicator import (
    RECOMMENDED_ACTIONS,
    RESPONSE_JSON_SCHEMA,
    VALID_LABELS,
    VALID_SEVERITIES,
    build_window_summary,
    load_prompt_template,
)
from model import build_model_from_checkpoint
from utils import choose_device, ensure_directory

ENABLE_GPT_LOGGING = True
TELEGRAM_SEND_ALL_RESULTS = True
TELEGRAM_SKIP_NORMAL_RESULTS = True
DYNAMIC_THRESHOLD_BUFFER_SIZE = 300
DYNAMIC_THRESHOLD_PERCENTILE = 99.0
DYNAMIC_THRESHOLD_MIN_BUFFER = 50


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def compact_reason(text: str, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)]}..."


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class CategoryEncoder:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.columns = list(metadata.get("columns", []))
        categories = metadata.get("categories", {})
        self.mappings: dict[str, dict[str, int]] = {
            column: {value: index for index, value in enumerate(categories.get(column, []))}
            for column in self.columns
        }

    def encode(self, values: dict[str, Any]) -> np.ndarray:
        encoded: list[float] = []
        for column in self.columns:
            raw = str(values.get(column, "unknown") or "unknown")
            encoded.append(float(self.mappings.get(column, {}).get(raw, -1)))
        return np.asarray(encoded, dtype=np.float32)


class EntityStatePreprocessor:
    def __init__(
        self,
        feature_columns: list[str],
        numeric_context_columns: list[str],
        categorical_context_columns: list[str],
        category_encoder: CategoryEncoder,
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_context_columns = numeric_context_columns
        self.categorical_context_columns = categorical_context_columns
        self.category_encoder = category_encoder
        self.last_feature_values: dict[str, np.ndarray] = {}

    def process(
        self,
        entity_id: str,
        features: dict[str, Any],
        context_numeric: dict[str, Any],
        context_categorical: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        previous = self.last_feature_values.get(entity_id)
        if previous is None:
            previous = np.zeros(len(self.feature_columns), dtype=np.float32)

        raw_features = np.asarray(
            [to_float(features.get(column), np.nan) for column in self.feature_columns],
            dtype=np.float32,
        )
        cleaned_features = np.where(np.isfinite(raw_features), raw_features, previous)
        cleaned_features = np.clip(cleaned_features, a_min=0.0, a_max=None).astype(np.float32)
        self.last_feature_values[entity_id] = cleaned_features

        numeric_vector = np.asarray(
            [to_float(context_numeric.get(column), 0.0) for column in self.numeric_context_columns],
            dtype=np.float32,
        )
        categorical_payload = {
            column: str(context_categorical.get(column, "unknown") or "unknown")
            for column in self.categorical_context_columns
        }
        categorical_vector = self.category_encoder.encode(categorical_payload)
        context_vector = np.concatenate([numeric_vector, categorical_vector], axis=0).astype(np.float32)
        return cleaned_features, context_vector


class DynamicThreshold:
    def __init__(self, buffer_size: int, percentile: float, min_buffer: int) -> None:
        self.buffer = deque(maxlen=max(1, int(buffer_size)))
        self.percentile = float(percentile)
        self.min_buffer = max(1, int(min_buffer))

    def current_threshold(self) -> float | None:
        if len(self.buffer) < self.min_buffer:
            return None
        scores = np.asarray(self.buffer, dtype=np.float32)
        return float(np.percentile(scores, self.percentile))

    def observe(self, score: float) -> None:
        self.buffer.append(float(score))


@dataclass
class RuntimeConfig:
    model_dir: Path
    prom_url: str
    poll_interval: int = 5
    request_timeout: int = 30
    verify_ssl: bool = True
    device: str = "cuda"
    top_k_features: int = 5
    cooldown_seconds: int = 300
    skip_log_cooldown_seconds: int = 300
    results_dir: Path = RESULTS_DIR / "live_prometheus"


class DirectPrometheusAnomalyRunner:
    """
    Poll Prometheus and run live detector -> GPT -> Telegram.
    This path intentionally does not use offline evaluation outputs.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.resolved_device = choose_device(config.device)
        self.query_variants = {
            "cpu_util": [
                """
                sum by (
                    container_label_io_kubernetes_pod_namespace,
                    container_label_io_kubernetes_pod_name,
                    container_label_io_kubernetes_container_name,
                    node,
                    instance
                ) (
                    rate(container_cpu_usage_seconds_total{
                        job="cadvisor",
                        image!="",
                        container_label_io_kubernetes_pod_name!="",
                        container_label_io_kubernetes_container_name!=""
                    }[5m])
                )
                """.strip(),
                """
                sum by (namespace, pod, container, node, instance) (
                    rate(container_cpu_usage_seconds_total{container!="",pod!=""}[1m])
                )
                """.strip(),
            ],
            "mem_used": [
                """
                sum by (
                    container_label_io_kubernetes_pod_namespace,
                    container_label_io_kubernetes_pod_name,
                    container_label_io_kubernetes_container_name,
                    node,
                    instance
                ) (
                    container_memory_working_set_bytes{
                        job="cadvisor",
                        image!="",
                        container_label_io_kubernetes_pod_name!="",
                        container_label_io_kubernetes_container_name!=""
                    }
                )
                """.strip(),
                """
                sum by (namespace, pod, container, node, instance) (
                    container_memory_working_set_bytes{container!="",pod!=""}
                )
                """.strip(),
            ],
            "mem_limit": [
                """
                sum by (
                    container_label_io_kubernetes_pod_namespace,
                    container_label_io_kubernetes_pod_name,
                    container_label_io_kubernetes_container_name,
                    node,
                    instance
                ) (
                    container_spec_memory_limit_bytes{
                        job="cadvisor",
                        image!="",
                        container_label_io_kubernetes_pod_name!="",
                        container_label_io_kubernetes_container_name!=""
                    }
                )
                """.strip(),
                """
                sum by (namespace, pod, container) (
                    kube_pod_container_resource_limits{
                        resource="memory",
                        unit="byte",
                        container!=""
                    }
                )
                """.strip(),
                """
                sum by (namespace, pod, container) (
                    kube_pod_container_resource_limits_memory_bytes{container!=""}
                )
                """.strip(),
            ],
            "net_in": [
                """
                sum by (
                    container_label_io_kubernetes_pod_namespace,
                    container_label_io_kubernetes_pod_name,
                    node,
                    instance
                ) (
                    rate(container_network_receive_bytes_total{
                        job="cadvisor",
                        container_label_io_kubernetes_pod_name!="",
                        interface="eth0"
                    }[5m])
                )
                """.strip(),
                """
                sum by (namespace, pod, node, instance) (
                    rate(container_network_receive_bytes_total{pod!=""}[1m])
                )
                """.strip(),
            ],
            "net_out": [
                """
                sum by (
                    container_label_io_kubernetes_pod_namespace,
                    container_label_io_kubernetes_pod_name,
                    node,
                    instance
                ) (
                    rate(container_network_transmit_bytes_total{
                        job="cadvisor",
                        container_label_io_kubernetes_pod_name!="",
                        interface="eth0"
                    }[5m])
                )
                """.strip(),
                """
                sum by (namespace, pod, node, instance) (
                    rate(container_network_transmit_bytes_total{pod!=""}[1m])
                )
                """.strip(),
            ],
            "disk_io": [
                """
                sum by (
                    container_label_io_kubernetes_pod_namespace,
                    container_label_io_kubernetes_pod_name,
                    container_label_io_kubernetes_container_name,
                    node,
                    instance
                ) (
                    rate(container_fs_reads_bytes_total{
                        job="cadvisor",
                        image!="",
                        container_label_io_kubernetes_pod_name!="",
                        container_label_io_kubernetes_container_name!=""
                    }[5m])
                    +
                    rate(container_fs_writes_bytes_total{
                        job="cadvisor",
                        image!="",
                        container_label_io_kubernetes_pod_name!="",
                        container_label_io_kubernetes_container_name!=""
                    }[5m])
                )
                """.strip(),
                """
                sum by (namespace, pod, container, node, instance) (
                    rate(container_fs_reads_bytes_total{container!="",pod!=""}[1m]) +
                    rate(container_fs_writes_bytes_total{container!="",pod!=""}[1m])
                )
                """.strip(),
            ],
        }
        self.last_query_stats: dict[str, dict[str, Any]] = {}
        self.bundle = self._load_bundle(config.model_dir, self.resolved_device)
        self.preprocessor = EntityStatePreprocessor(
            feature_columns=self.bundle["feature_columns"],
            numeric_context_columns=self.bundle["numeric_context_columns"],
            categorical_context_columns=self.bundle["categorical_context_columns"],
            category_encoder=self.bundle["category_encoder"],
        )
        self.detector = StreamingFiLMAnomalyDetector(
            model=self.bundle["model"],
            x_scaler=self.bundle["x_scaler"],
            c_scaler=self.bundle["c_scaler"],
            detector_meta=self.bundle["detector_meta"],
            feature_names=self.bundle["feature_columns"],
            top_k_features=config.top_k_features,
            device=self.resolved_device,
        )

        self.gpt_config = GPTConfig(
            evaluation_dir=(config.results_dir / config.model_dir.name).resolve(),
            output_dir=(config.results_dir / config.model_dir.name).resolve(),
        )
        self.gpt_prompt = load_prompt_template(self.gpt_config.prompt_template_path)
        self.gpt_unavailable_reason: str | None = None
        self.openai_client = self._build_openai_client()
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        self.dynamic_threshold = DynamicThreshold(
            buffer_size=DYNAMIC_THRESHOLD_BUFFER_SIZE,
            percentile=DYNAMIC_THRESHOLD_PERCENTILE,
            min_buffer=DYNAMIC_THRESHOLD_MIN_BUFFER,
        )
        self.last_notified_at: dict[str, int] = {}
        self.last_skip_logged_at: dict[tuple[str, str], int] = {}
        self.ready_window_counts: dict[str, int] = {}

        log_dir = ensure_directory((config.results_dir / config.model_dir.name).resolve())
        self.decision_log_path = log_dir / "live_gpt_decisions.jsonl"
        self.gpt_response_log_path = log_dir / "live_gpt_responses.jsonl"
        self.telegram_log_path = log_dir / "live_telegram_messages.jsonl"
        self.threshold_log_path = log_dir / "live_threshold_decisions.jsonl"

        if not DOTENV_AVAILABLE:
            print("[WARN] python-dotenv is not installed. .env loading may be skipped.")
        print(f"OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"TELEGRAM_BOT_TOKEN loaded: {bool(self.telegram_bot_token)}")
        print(f"TELEGRAM_CHAT_ID loaded: {bool(self.telegram_chat_id)}")

        if not os.getenv("OPENAI_API_KEY"):
            self.gpt_unavailable_reason = "missing_openai_api_key"
            print("[WARN] OPENAI_API_KEY not found. GPT adjudication will be skipped.")
        elif self.openai_client is None:
            self.gpt_unavailable_reason = "openai_package_not_installed"
            print("[WARN] Python package `openai` is not installed in this venv. GPT adjudication and Telegram alerts are disabled.")

        if not self.telegram_enabled:
            print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing. Telegram alerts are disabled.")

    def _build_openai_client(self) -> Any | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        if importlib.util.find_spec("openai") is None:
            return None
        from openai import OpenAI
        return OpenAI(api_key=api_key, timeout=self.gpt_config.request_timeout_seconds)

    def _default_category_metadata(self, context_columns: list[str]) -> dict[str, Any]:
        default_categorical = set(
            DatasetConfig().container_context_categorical + DatasetConfig().machine_context_categorical
        )
        columns = [column for column in context_columns if column in default_categorical]
        return {
            "columns": columns,
            "categories": {column: [] for column in columns},
        }

    def _load_category_metadata(
        self,
        model_dir: Path,
        context_columns: list[str],
    ) -> dict[str, Any]:
        candidate_paths: list[Path] = [model_dir / "context_encoder.joblib"]

        training_summary_path = model_dir / "training_summary.json"
        if training_summary_path.exists():
            try:
                training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
                dataset_dir = (
                    training_summary.get("train_config", {}).get("dataset_dir")
                    or training_summary.get("dataset_dir")
                )
                if dataset_dir:
                    candidate_paths.append(Path(dataset_dir) / "context_encoder.joblib")
            except Exception:
                pass

        for path in candidate_paths:
            if path.exists():
                return joblib.load(path)

        print(
            "[WARN] context_encoder.joblib was not found in the model or dataset directory. "
            "Falling back to empty category mappings."
        )
        return self._default_category_metadata(context_columns)

    def _load_bundle(self, model_dir: Path, device: str) -> dict[str, Any]:
        checkpoint = torch.load(model_dir / "film_ae.pt", map_location=device)
        model = build_model_from_checkpoint(checkpoint, device=device)

        x_scaler = joblib.load(model_dir / "x_scaler.joblib")
        c_scaler = joblib.load(model_dir / "c_scaler.joblib")
        detector_meta = joblib.load(model_dir / "detector_meta.joblib")
        feature_columns = list(checkpoint["feature_columns"])
        context_columns = list(checkpoint["context_columns"])
        category_metadata = self._load_category_metadata(model_dir, context_columns)
        categorical_columns = list(category_metadata.get("columns", []))
        numeric_columns = [column for column in context_columns if column not in categorical_columns]

        return {
            "model": model,
            "x_scaler": x_scaler,
            "c_scaler": c_scaler,
            "detector_meta": detector_meta,
            "feature_columns": feature_columns,
            "numeric_context_columns": numeric_columns,
            "categorical_context_columns": categorical_columns,
            "category_encoder": CategoryEncoder(category_metadata),
        }

    def run_query(self, query: str) -> list[dict[str, Any]]:
        response = requests.get(
            self.config.prom_url,
            params={"query": query},
            timeout=self.config.request_timeout,
            verify=self.config.verify_ssl,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {body}")
        return body.get("data", {}).get("result", [])

    @staticmethod
    def normalize_metric_labels(metric: dict[str, str]) -> dict[str, str]:
        namespace = str(
            metric.get("namespace")
            or metric.get("container_label_io_kubernetes_pod_namespace")
            or "default"
        )
        pod = str(
            metric.get("pod")
            or metric.get("container_label_io_kubernetes_pod_name")
            or "unknown-pod"
        )
        container = str(
            metric.get("container")
            or metric.get("container_label_io_kubernetes_container_name")
            or ""
        )
        return {
            "namespace": namespace,
            "pod": pod,
            "container": container,
        }

    @classmethod
    def container_key(cls, metric: dict[str, str]) -> tuple[str, str, str]:
        labels = cls.normalize_metric_labels(metric)
        namespace = labels["namespace"]
        pod = labels["pod"]
        container = labels["container"] or "unknown-container"
        return namespace, pod, container

    @classmethod
    def pod_key(cls, metric: dict[str, str]) -> tuple[str, str]:
        labels = cls.normalize_metric_labels(metric)
        return labels["namespace"], labels["pod"]

    @staticmethod
    def infer_machine_id(metric: dict[str, str]) -> str:
        for key in (
            "node",
            "instance",
            "host",
            "hostname",
            "kubernetes_io_hostname",
            "nodename",
        ):
            value = str(metric.get(key, "") or "").strip()
            if value:
                return value
        return str(metric.get("pod", "unknown-machine"))

    def _entry_for(
        self,
        store: dict[tuple[str, str, str], dict[str, Any]],
        metric: dict[str, str],
    ) -> dict[str, Any]:
        labels = self.normalize_metric_labels(metric)
        key = (
            labels["namespace"],
            labels["pod"],
            labels["container"] or "unknown-container",
        )
        namespace, pod, container = key
        entry = store.setdefault(
            key,
            {
                "entity_id": f"{namespace}/{pod}/{container}",
                "namespace": namespace,
                "pod": pod,
                "container": container,
                "machine_id": self.infer_machine_id(metric),
                "features": {},
            },
        )
        candidate_machine_id = self.infer_machine_id(metric)
        if candidate_machine_id and entry.get("machine_id", "").startswith("unknown"):
            entry["machine_id"] = candidate_machine_id
        return entry

    def query_first_nonempty(self, metric_name: str) -> list[dict[str, Any]]:
        variants = self.query_variants[metric_name]
        errors: list[str] = []
        counts: list[int] = []

        for index, query in enumerate(variants, start=1):
            try:
                result = self.run_query(query)
                count = int(len(result))
                counts.append(count)
                if count > 0:
                    self.last_query_stats[metric_name] = {
                        "variant_index": index,
                        "rows": count,
                    }
                    return result
            except Exception as exc:
                errors.append(f"variant_{index}:{compact_reason(str(exc), limit=120)}")

        self.last_query_stats[metric_name] = {
            "variant_index": None,
            "rows": 0,
            "errors": errors,
            "counts": counts,
        }
        return []

    def collect_snapshot(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        store: dict[tuple[str, str, str], dict[str, Any]] = {}

        for row in self.query_first_nonempty("cpu_util"):
            metric = row.get("metric", {})
            value = to_float(row.get("value", [None, 0.0])[1], 0.0)
            entry = self._entry_for(store, metric)
            entry["features"]["cpu_util"] = value

        mem_limit_map: dict[tuple[str, str, str], float] = {}
        for row in self.query_first_nonempty("mem_limit"):
            metric = row.get("metric", {})
            value = to_float(row.get("value", [None, 0.0])[1], 0.0)
            mem_limit_map[self.container_key(metric)] = value
            self._entry_for(store, metric)

        for row in self.query_first_nonempty("mem_used"):
            metric = row.get("metric", {})
            mem_used = to_float(row.get("value", [None, 0.0])[1], 0.0)
            key = self.container_key(metric)
            entry = self._entry_for(store, metric)
            mem_limit = mem_limit_map.get(key, 0.0)
            entry["features"]["mem_util"] = (mem_used / mem_limit) if mem_limit > 0 else 0.0

        pod_net_in: dict[tuple[str, str], float] = {}
        pod_machine_ids: dict[tuple[str, str], str] = {}
        for row in self.query_first_nonempty("net_in"):
            metric = row.get("metric", {})
            value = to_float(row.get("value", [None, 0.0])[1], 0.0)
            key = self.pod_key(metric)
            pod_net_in[key] = value
            pod_machine_ids[key] = self.infer_machine_id(metric)

        pod_net_out: dict[tuple[str, str], float] = {}
        for row in self.query_first_nonempty("net_out"):
            metric = row.get("metric", {})
            value = to_float(row.get("value", [None, 0.0])[1], 0.0)
            key = self.pod_key(metric)
            pod_net_out[key] = value
            pod_machine_ids.setdefault(key, self.infer_machine_id(metric))

        for row in self.query_first_nonempty("disk_io"):
            metric = row.get("metric", {})
            value = to_float(row.get("value", [None, 0.0])[1], 0.0)
            entry = self._entry_for(store, metric)
            entry["features"]["disk_io"] = value

        for key, entry in store.items():
            namespace, pod, _ = key
            pod_key = (namespace, pod)
            entry["features"]["net_in"] = pod_net_in.get(pod_key, 0.0)
            entry["features"]["net_out"] = pod_net_out.get(pod_key, 0.0)
            if (
                entry.get("machine_id", "").startswith("unknown")
                and pod_key in pod_machine_ids
            ):
                entry["machine_id"] = pod_machine_ids[pod_key]

        if not store:
            stats = []
            for metric_name in ("cpu_util", "mem_used", "mem_limit", "net_in", "net_out", "disk_io"):
                metric_stats = self.last_query_stats.get(metric_name, {})
                rows = metric_stats.get("rows", 0)
                variant_index = metric_stats.get("variant_index")
                stats.append(f"{metric_name}=rows:{rows},variant:{variant_index}")
            print(f"[SKIP] snapshot | reason=no_matching_metrics | {' | '.join(stats)}")

        return store

    def process_entity(
        self,
        entity: dict[str, Any],
        time_stamp: int,
    ) -> dict[str, Any]:
        entity_id = str(entity["entity_id"])
        namespace = str(entity["namespace"])
        pod = str(entity["pod"])
        container = str(entity["container"])
        machine_id = str(entity.get("machine_id", "unknown-machine"))
        features = dict(entity.get("features", {}))

        context_numeric = {}
        context_categorical = {
            "container_app_du": namespace,
            "container_status": "running",
            "machine_failure_domain_1": "unknown",
            "machine_failure_domain_2": "unknown",
            "machine_status": "active",
        }

        full_features = {
            "cpu_util": features.get("cpu_util", 0.0),
            "mem_util": features.get("mem_util", 0.0),
            "cpi": 0.0,
            "mem_gps": 0.0,
            "mpki": 0.0,
            "net_in": features.get("net_in", 0.0),
            "net_out": features.get("net_out", 0.0),
            "disk_io": features.get("disk_io", 0.0),
        }

        feature_row, context_vector = self.preprocessor.process(
            entity_id=entity_id,
            features=full_features,
            context_numeric=context_numeric,
            context_categorical=context_categorical,
        )

        result = self.detector.update(
            entity_id=entity_id,
            feature_row=feature_row,
            context_vector=context_vector,
            metadata={
                "machine_id": machine_id,
                "time_stamp": time_stamp,
                "namespace": namespace,
                "pod": pod,
                "container": container,
                "source": "prometheus_direct",
            },
        )

        window_id = int(self.ready_window_counts.get(entity_id, 0))
        fixed_threshold = float(self.detector.threshold)
        response = {
            "entity_id": entity_id,
            "container_id": container,
            "machine_id": machine_id,
            "window_id": window_id,
            "window_ready": bool(result.ready),
            "threshold": fixed_threshold,
            "fixed_threshold": fixed_threshold,
            "dynamic_threshold": None,
            "final_threshold": fixed_threshold,
            "window_size": int(self.detector.window_size),
            "namespace": namespace,
            "pod": pod,
            "container_app_du": namespace,
            "container_status": "running",
            "machine_status": "active",
            "machine_failure_domain_1": "unknown",
            "machine_failure_domain_2": "unknown",
            "start_time": time_stamp,
            "end_time": time_stamp,
            "features": full_features,
        }

        if not result.ready:
            response["status"] = "buffering"
            return response

        self.ready_window_counts[entity_id] = window_id + 1
        anomaly_score = float(result.anomaly_score or 0.0)
        dynamic_threshold = self.dynamic_threshold.current_threshold()
        final_threshold = max(fixed_threshold, dynamic_threshold) if dynamic_threshold is not None else fixed_threshold
        predicted_label = int(anomaly_score > final_threshold)
        self.dynamic_threshold.observe(anomaly_score)
        response.update(
            {
                "status": "anomaly" if predicted_label == 1 else "normal",
                "predicted_label": predicted_label,
                "anomaly_score": anomaly_score,
                "threshold": final_threshold,
                "fixed_threshold": fixed_threshold,
                "dynamic_threshold": dynamic_threshold,
                "final_threshold": final_threshold,
                "score_over_threshold": anomaly_score - final_threshold,
                "top_k_features": result.top_k_features or [],
                "top_k_feature_errors": result.top_k_feature_errors or [],
                "feature_error_vector": result.feature_error_vector or [],
                "metadata": result.metadata or {},
            }
        )
        self.log_threshold_decision(response)
        return response

    def adjudicate_with_gpt(self, candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if self.openai_client is None:
            return None, self.gpt_unavailable_reason or "gpt_unavailable"

        summary = build_window_summary(candidate)
        try:
            response = self.openai_client.responses.create(
                model=self.gpt_config.model,
                instructions=self.gpt_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(summary, ensure_ascii=True),
                            }
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "container_anomaly_decision",
                        "schema": RESPONSE_JSON_SCHEMA,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            return None, f"openai_request_failed:{compact_reason(str(exc), limit=120)}"

        response_text = getattr(response, "output_text", "")
        if not response_text:
            return None, "empty_response_text"

        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError:
            return None, "invalid_json_response"

        normalized = {
            "label": str(decision.get("label", "")).strip().lower(),
            "severity": str(decision.get("severity", "")).strip().lower(),
            "explanation": str(decision.get("explanation", "")).strip(),
            "recommended_action": str(decision.get("recommended_action", "")).strip(),
        }
        if normalized["label"] not in VALID_LABELS:
            return None, "invalid_gpt_label"
        if normalized["severity"] not in VALID_SEVERITIES:
            return None, "invalid_gpt_severity"
        if normalized["recommended_action"] not in RECOMMENDED_ACTIONS:
            return None, "invalid_gpt_recommended_action"
        if not normalized["explanation"]:
            return None, "empty_gpt_explanation"

        return (
            {
                **candidate,
                "gpt_input_summary": summary,
                "structured_json": normalized,
                "label": normalized["label"],
                "severity": normalized["severity"],
                "explanation": normalized["explanation"],
                "recommended_action": normalized["recommended_action"],
                "used_fallback": False,
                "gpt_model": self.gpt_config.model,
                "response_id": getattr(response, "id", None),
            },
            None,
        )

    def in_cooldown(self, entity_id: str, time_stamp: int) -> tuple[bool, int]:
        last_sent = self.last_notified_at.get(entity_id)
        if last_sent is None:
            return False, 0
        elapsed = int(time_stamp - last_sent)
        if elapsed >= int(self.config.cooldown_seconds):
            return False, 0
        return True, int(self.config.cooldown_seconds - elapsed)

    def format_telegram_alert(self, decision: dict[str, Any]) -> str:
        top_features = decision.get("top_k_features", [])
        top_text = ", ".join(str(value) for value in top_features[:5]) if top_features else "n/a"
        lines = [
            "Live container anomaly alert",
            f"Entity: {decision.get('entity_id', 'unknown')}",
            f"Container: {decision.get('container_id', 'unknown')}",
            f"Machine: {decision.get('machine_id', 'unknown')}",
            f"Anomaly score: {float(decision.get('anomaly_score', 0.0)):.6f}",
            f"Threshold: {float(decision.get('threshold', 0.0)):.6f}",
            f"Top features: {top_text}",
            f"GPT label: {decision.get('label', 'unknown')}",
            f"GPT severity: {decision.get('severity', 'unknown')}",
            f"Explanation: {decision.get('explanation', '')}",
            f"Recommended action: {decision.get('recommended_action', '')}",
        ]
        return "\n".join(lines)

    def send_telegram_alert(self, decision: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.telegram_enabled:
            return False, "missing_telegram_credentials"

        message_text = self.format_telegram_alert(decision)
        endpoint = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        try:
            response = requests.post(
                endpoint,
                json={
                    "chat_id": self.telegram_chat_id,
                    "text": message_text,
                    "disable_web_page_preview": True,
                },
                timeout=max(1, int(self.config.request_timeout)),
            )
            response.raise_for_status()
            body = response.json()
            if not body.get("ok", False):
                self.log_telegram_message(
                    decision=decision,
                    message_text=message_text,
                    sent=False,
                    error=f"telegram_send_failed:{body.get('description', 'unknown_error')}",
                )
                return False, f"telegram_send_failed:{body.get('description', 'unknown_error')}"
            self.log_telegram_message(
                decision=decision,
                message_text=message_text,
                sent=True,
                error=None,
            )
            return True, None
        except Exception as exc:
            error_text = f"telegram_send_failed:{compact_reason(str(exc), limit=120)}"
            self.log_telegram_message(
                decision=decision,
                message_text=message_text,
                sent=False,
                error=error_text,
            )
            return False, error_text

    def log_gpt_decision(self, decision: dict[str, Any], telegram_sent: bool) -> None:
        payload = {
            "timestamp": now_iso(),
            "entity_id": decision.get("entity_id"),
            "anomaly_score": decision.get("anomaly_score"),
            "threshold": decision.get("threshold"),
            "top_features": decision.get("top_k_features", []),
            "gpt_label": decision.get("label"),
            "gpt_severity": decision.get("severity"),
            "gpt_explanation": decision.get("explanation"),
            "recommended_action": decision.get("recommended_action"),
            "telegram_sent": bool(telegram_sent),
        }
        append_jsonl(self.decision_log_path, payload)

    def log_gpt_response(self, decision: dict[str, Any], status: str) -> None:
        payload = {
            "timestamp": now_iso(),
            "entity_id": decision.get("entity_id"),
            "container_id": decision.get("container_id"),
            "machine_id": decision.get("machine_id"),
            "anomaly_score": decision.get("anomaly_score"),
            "threshold": decision.get("threshold"),
            "status": status,
            "response_id": decision.get("response_id"),
            "gpt_model": decision.get("gpt_model"),
            "label": decision.get("label"),
            "severity": decision.get("severity"),
            "explanation": decision.get("explanation"),
            "recommended_action": decision.get("recommended_action"),
            "structured_json": decision.get("structured_json"),
            "gpt_input_summary": decision.get("gpt_input_summary"),
        }
        append_jsonl(self.gpt_response_log_path, payload)

    def log_telegram_message(
        self,
        decision: dict[str, Any],
        message_text: str,
        sent: bool,
        error: str | None,
    ) -> None:
        payload = {
            "timestamp": now_iso(),
            "entity_id": decision.get("entity_id"),
            "container_id": decision.get("container_id"),
            "machine_id": decision.get("machine_id"),
            "anomaly_score": decision.get("anomaly_score"),
            "threshold": decision.get("threshold"),
            "label": decision.get("label"),
            "severity": decision.get("severity"),
            "recommended_action": decision.get("recommended_action"),
            "telegram_sent": bool(sent),
            "error": error,
            "message_text": message_text,
        }
        append_jsonl(self.telegram_log_path, payload)

    def log_threshold_decision(self, window_result: dict[str, Any]) -> None:
        payload = {
            "timestamp": now_iso(),
            "entity_id": window_result.get("entity_id"),
            "score": window_result.get("anomaly_score"),
            "fixed_threshold": window_result.get("fixed_threshold"),
            "dynamic_threshold": window_result.get("dynamic_threshold"),
            "final_threshold": window_result.get("final_threshold"),
            "decision": window_result.get("status"),
        }
        append_jsonl(self.threshold_log_path, payload)

    def should_print_skip(self, entity_id: str, reason: str, time_stamp: int) -> bool:
        key = (entity_id, reason)
        last_logged = self.last_skip_logged_at.get(key)
        if last_logged is not None:
            elapsed = int(time_stamp - last_logged)
            if elapsed < int(self.config.skip_log_cooldown_seconds):
                return False
        self.last_skip_logged_at[key] = int(time_stamp)
        return True

    def should_send_telegram_result(self, decision: dict[str, Any]) -> tuple[bool, str | None]:
        label = str(decision.get("label", "")).strip().lower()
        if TELEGRAM_SEND_ALL_RESULTS:
            return True, None
        if TELEGRAM_SKIP_NORMAL_RESULTS and label == "normal":
            return False, "gpt_label_normal"
        return True, None

    def handle_ready_result(self, result: dict[str, Any], time_stamp: int) -> None:
        entity_id = str(result["entity_id"])

        if int(result.get("predicted_label", 0)) != 1:
            print(f"[NORMAL] {entity_id} | score={float(result.get('anomaly_score', 0.0)):.6f}")
            return

        decision, gpt_error = self.adjudicate_with_gpt(result)
        if decision is None:
            if self.should_print_skip(entity_id, str(gpt_error), time_stamp):
                print(f"[SKIP] {entity_id} | reason={gpt_error}")
            return

        if str(decision["label"]).strip().lower() == "normal":
            print(f"[GPT-NORMAL] {entity_id} | skipped")
            self.log_gpt_response(decision, status="normal_skipped")
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=False)
            return

        print(
            f"[GPT-ALERT] {entity_id} | "
            f"label={decision['label']} | severity={decision['severity']}"
        )
        self.log_gpt_response(decision, status="alert_candidate")

        should_send, skip_reason = self.should_send_telegram_result(decision)
        if not should_send:
            if self.should_print_skip(entity_id, str(skip_reason), time_stamp):
                print(f"[SKIP] {entity_id} | reason={skip_reason}")
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=False)
            return

        cooldown_active, remaining = self.in_cooldown(entity_id, time_stamp)
        if cooldown_active:
            reason = f"cooldown_active:{remaining}s_remaining"
            if self.should_print_skip(entity_id, reason, time_stamp):
                print(f"[SKIP] {entity_id} | reason={reason}")
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=False)
            return

        sent, telegram_error = self.send_telegram_alert(decision)
        if sent:
            self.last_notified_at[entity_id] = int(time_stamp)
            print(f"[TELEGRAM] {entity_id} | sent")
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=True)
            return

        if self.should_print_skip(entity_id, str(telegram_error), time_stamp):
            print(f"[SKIP] {entity_id} | reason={telegram_error}")
        if ENABLE_GPT_LOGGING:
            self.log_gpt_decision(decision, telegram_sent=False)

    def run_once(self) -> None:
        snapshot = self.collect_snapshot()
        now_ts = int(time.time())

        if not snapshot:
            return

        for entity in snapshot.values():
            entity_id = str(entity["entity_id"])
            try:
                result = self.process_entity(entity=entity, time_stamp=now_ts)
                if not result["window_ready"]:
                    print(f"[BUFFERING] {entity_id}")
                    continue
                self.handle_ready_result(result, time_stamp=now_ts)
            except Exception as exc:
                print(f"[SKIP] {entity_id} | reason={compact_reason(str(exc), limit=160)}")

    def run_forever(self) -> None:
        print("Starting direct Prometheus -> FiLM detector -> GPT -> Telegram loop...")
        print(f"Prometheus: {self.config.prom_url}")
        print(f"Model dir : {self.config.model_dir}")
        print(f"Device    : requested={self.config.device} resolved={self.resolved_device}")
        print(f"Cooldown  : {self.config.cooldown_seconds}s")
        print(f"Log file  : {self.decision_log_path}")
        print(f"GPT log   : {self.gpt_response_log_path}")
        print(f"Telegram log: {self.telegram_log_path}")
        print(f"Threshold log: {self.threshold_log_path}")
        print(
            "Adaptive threshold: "
            f"buffer={DYNAMIC_THRESHOLD_BUFFER_SIZE}, "
            f"percentile={DYNAMIC_THRESHOLD_PERCENTILE}, "
            f"min_buffer={DYNAMIC_THRESHOLD_MIN_BUFFER}"
        )
        print(f"Interval  : {self.config.poll_interval}s")
        while True:
            try:
                self.run_once()
            except Exception as exc:
                print(f"[SKIP] poll | reason={compact_reason(str(exc), limit=160)}")
            time.sleep(self.config.poll_interval)


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description="Run live Prometheus -> FiLM AE -> GPT adjudication -> Telegram alerts."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model artifacts.")
    parser.add_argument(
        "--prom-url",
        default="http://35.206.92.147:9090/api/v1/query",
        help="Prometheus instant query endpoint.",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k-features", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--skip-log-cooldown-seconds", type=int, default=300)
    parser.add_argument(
        "--results-dir",
        default=str((RESULTS_DIR / "live_prometheus").resolve()),
        help="Directory for live GPT decision logs.",
    )
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    return RuntimeConfig(
        model_dir=Path(args.model_dir).expanduser().resolve(),
        prom_url=args.prom_url,
        poll_interval=args.poll_interval,
        request_timeout=args.request_timeout,
        verify_ssl=not args.insecure,
        device=args.device,
        top_k_features=args.top_k_features,
        cooldown_seconds=max(0, int(args.cooldown_seconds)),
        skip_log_cooldown_seconds=max(0, int(args.skip_log_cooldown_seconds)),
        results_dir=Path(args.results_dir).expanduser().resolve(),
    )


def main() -> None:
    config = parse_args()
    runner = DirectPrometheusAnomalyRunner(config)
    runner.run_forever()


if __name__ == "__main__":
    main()
