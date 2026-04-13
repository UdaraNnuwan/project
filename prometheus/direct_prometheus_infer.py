from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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
    adjudicate_anomaly,
)
from live_infer import load_model_artifacts
from telegram_utils import format_telegram_alert, send_telegram_message
from utils import choose_device, ensure_directory

ENABLE_GPT_LOGGING = True
TELEGRAM_NOTIFICATIONS_ENABLED = True
TELEGRAM_STARTUP_TEST_ENABLED = True
TELEGRAM_STARTUP_TEST_MESSAGE = "DevOps Alert Bot startup test: Telegram notifications are working."
TELEGRAM_SEND_ALL_RESULTS = False
TELEGRAM_SKIP_NORMAL_RESULTS = True
ROLLING_SCORE_BUFFER_SIZE = 500
ROLLING_SCORE_PERCENTILE = 99.0
ROLLING_SCORE_MIN_BUFFER = 100
ENABLE_ZSCORE_CANDIDATE = True
ZSCORE_THRESHOLD = 3.0
ENABLE_FEATURE_SHIFT_CANDIDATE = True
FEATURE_SHIFT_ZSCORE_THRESHOLD = 3.0
FEATURE_SHIFT_MIN_BUFFER = 100
STATS_STD_EPS = 1e-8
ADAPTIVE_STATE_DEBUG_LOGS = True


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


def mask_secret(value: str | None, prefix: int = 7, suffix: int = 4) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= prefix + suffix:
        return "*" * len(text)
    return f"{text[:prefix]}...{text[-suffix:]}"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def env_csv(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def format_metric_value(value: Any, precision: int = 6, default: str = "n/a") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return default


VOLATILITY_LOOKBACK = max(5, env_int("LIVE_VOLATILITY_LOOKBACK", 30))
VOLATILITY_MIN_BUFFER = max(5, env_int("LIVE_VOLATILITY_MIN_BUFFER", 15))
VOLATILITY_LOW_MAX = env_float("LIVE_VOLATILITY_LOW_MAX", 0.15)
VOLATILITY_HIGH_MIN = env_float("LIVE_VOLATILITY_HIGH_MIN", 0.35)


@dataclass(frozen=True)
class AdaptiveDecisionProfile:
    band: str
    threshold_percentile: float
    consecutive_windows: int
    smoothing_window: int
    cooldown_windows: int


LOW_VOLATILITY_PROFILE = AdaptiveDecisionProfile(
    band="low",
    threshold_percentile=env_float("LIVE_LOW_VOLATILITY_PERCENTILE", 99.0),
    consecutive_windows=max(1, env_int("LIVE_LOW_VOLATILITY_CONSECUTIVE_WINDOWS", 2)),
    smoothing_window=max(1, env_int("LIVE_LOW_VOLATILITY_SMOOTHING_WINDOW", 3)),
    cooldown_windows=max(0, env_int("LIVE_LOW_VOLATILITY_COOLDOWN_WINDOWS", 2)),
)
MEDIUM_VOLATILITY_PROFILE = AdaptiveDecisionProfile(
    band="medium",
    threshold_percentile=env_float("LIVE_MEDIUM_VOLATILITY_PERCENTILE", 99.3),
    consecutive_windows=max(1, env_int("LIVE_MEDIUM_VOLATILITY_CONSECUTIVE_WINDOWS", 3)),
    smoothing_window=max(1, env_int("LIVE_MEDIUM_VOLATILITY_SMOOTHING_WINDOW", 5)),
    cooldown_windows=max(0, env_int("LIVE_MEDIUM_VOLATILITY_COOLDOWN_WINDOWS", 3)),
)
HIGH_VOLATILITY_PROFILE = AdaptiveDecisionProfile(
    band="high",
    threshold_percentile=env_float("LIVE_HIGH_VOLATILITY_PERCENTILE", 99.7),
    consecutive_windows=max(1, env_int("LIVE_HIGH_VOLATILITY_CONSECUTIVE_WINDOWS", 5)),
    smoothing_window=max(1, env_int("LIVE_HIGH_VOLATILITY_SMOOTHING_WINDOW", 7)),
    cooldown_windows=max(0, env_int("LIVE_HIGH_VOLATILITY_COOLDOWN_WINDOWS", 5)),
)
DEFAULT_VOLATILITY_PROFILE = MEDIUM_VOLATILITY_PROFILE


def trailing_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32)
    if series.size == 0:
        return series
    width = max(1, int(window))
    if width == 1:
        return series.astype(np.float32, copy=True)

    cumulative = np.cumsum(series, dtype=np.float64)
    smoothed = np.empty(series.size, dtype=np.float64)
    for index in range(series.size):
        start = max(0, index - width + 1)
        total = cumulative[index] - (cumulative[start - 1] if start > 0 else 0.0)
        smoothed[index] = total / float(index - start + 1)
    return smoothed.astype(np.float32)


def compute_percentile_threshold(values: np.ndarray, percentile: float) -> float | None:
    series = np.asarray(values, dtype=np.float32)
    if series.size == 0:
        return None
    return float(np.percentile(series, float(percentile)))


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


class RollingStatsBuffer:
    def __init__(self, buffer_size: int, percentile: float, min_buffer: int) -> None:
        self.buffer = deque(maxlen=max(1, int(buffer_size)))
        self.percentile = float(percentile)
        self.min_buffer = max(1, int(min_buffer))

    def values(self) -> np.ndarray:
        if not self.buffer:
            return np.asarray([], dtype=np.float32)
        return np.asarray(self.buffer, dtype=np.float32)

    def summary(self) -> dict[str, float] | None:
        if len(self.buffer) < self.min_buffer:
            return None
        scores = self.values()
        return {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "percentile_threshold": float(np.percentile(scores, self.percentile)),
        }

    def current_threshold(self) -> float | None:
        stats = self.summary()
        if stats is None:
            return None
        return float(stats["percentile_threshold"])

    def size(self) -> int:
        return int(len(self.buffer))

    def z_score_details(self, value: float) -> tuple[float | None, str | None]:
        stats = self.summary()
        if stats is None:
            return None, f"warmup<{self.min_buffer}"
        std = float(stats["std"])
        if std <= STATS_STD_EPS:
            return None, "std_too_small"
        return float((float(value) - float(stats["mean"])) / std), None

    def observe(self, score: float) -> None:
        self.buffer.append(float(score))


@dataclass
class EntityAdaptiveState:
    score_buffer: RollingStatsBuffer
    feature_error_buffers: dict[str, RollingStatsBuffer]
    restored_score_count: int = 0
    consecutive_breach_count: int = 0
    last_confirmed_time_stamp: int | None = None
    last_confirmed_window_id: int | None = None
    last_confirmed_cooldown_seconds: int = 0
    last_confirmed_cooldown_windows: int = 0


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
    cooldown_windows: int = field(default_factory=lambda: max(0, env_int("LIVE_ALERT_COOLDOWN_WINDOWS", 3)))
    skip_log_cooldown_seconds: int = 300
    include_namespaces: list[str] = field(default_factory=lambda: env_csv("LIVE_ALERT_INCLUDE_NAMESPACES"))
    exclude_namespaces: list[str] = field(
        default_factory=lambda: env_csv(
            "LIVE_ALERT_EXCLUDE_NAMESPACES",
            "kube-system,kube-flannel,monitoring",
        )
    )
    include_pods: list[str] = field(default_factory=lambda: env_csv("LIVE_ALERT_INCLUDE_PODS"))
    exclude_pods: list[str] = field(default_factory=lambda: env_csv("LIVE_ALERT_EXCLUDE_PODS"))
    include_containers: list[str] = field(default_factory=lambda: env_csv("LIVE_ALERT_INCLUDE_CONTAINERS"))
    exclude_containers: list[str] = field(default_factory=lambda: env_csv("LIVE_ALERT_EXCLUDE_CONTAINERS"))
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
            reconstruction_model=self.bundle["reconstruction_model"],
            forecasting_model=self.bundle["forecasting_model"],
            x_scaler=self.bundle["x_scaler"],
            c_scaler=self.bundle["c_scaler"],
            detector_meta=self.bundle["detector_meta"],
            feature_names=self.bundle["feature_columns"],
            top_k_features=config.top_k_features,
            device=self.resolved_device,
            model_mode=self.bundle["mode"],
        )

        self.gpt_config = GPTConfig(
            evaluation_dir=(config.results_dir / config.model_dir.name).resolve(),
            output_dir=(config.results_dir / config.model_dir.name).resolve(),
        )
        self.gpt_unavailable_reason: str | None = None
        self.openai_client = self._build_openai_client()
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(
            TELEGRAM_NOTIFICATIONS_ENABLED and self.telegram_bot_token and self.telegram_chat_id
        )
        self.entity_adaptive_state: dict[str, EntityAdaptiveState] = {}
        self.entity_state_creation_counts: dict[str, int] = {}
        self.last_skip_logged_at: dict[tuple[str, str], int] = {}
        self.ready_window_counts: dict[str, int] = {}

        log_dir = ensure_directory((config.results_dir / config.model_dir.name).resolve())
        self.decision_log_path = log_dir / "live_gpt_decisions.jsonl"
        self.gpt_response_log_path = log_dir / "live_gpt_responses.jsonl"
        self.telegram_log_path = log_dir / "live_telegram_messages.jsonl"
        self.threshold_log_path = log_dir / "live_threshold_decisions.jsonl"
        self.restore_score_buffers_from_threshold_log()

        if not DOTENV_AVAILABLE:
            print("[WARN] python-dotenv is not installed. .env loading may be skipped.")
        print(f"OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
        if os.getenv("OPENAI_API_KEY"):
            print(f"OPENAI_API_KEY masked: {mask_secret(os.getenv('OPENAI_API_KEY'))}")
        print(f"TELEGRAM_BOT_TOKEN loaded: {bool(self.telegram_bot_token)}")
        print(f"TELEGRAM_CHAT_ID loaded: {bool(self.telegram_chat_id)}")

        if not os.getenv("OPENAI_API_KEY"):
            self.gpt_unavailable_reason = "missing_openai_api_key"
            print("[WARN] OPENAI_API_KEY not found. GPT adjudication will be skipped.")
        elif self.openai_client is None:
            self.gpt_unavailable_reason = "openai_package_not_installed"
            print("[WARN] Python package `openai` is not installed in this venv. GPT adjudication and Telegram alerts are disabled.")

        if not TELEGRAM_NOTIFICATIONS_ENABLED:
            print("[WARN] Telegram notifications are disabled in direct_prometheus_infer.py.")
        elif not self.telegram_enabled:
            print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing. Telegram alerts are disabled.")

        self.send_startup_telegram_test()

    def _new_score_buffer(self) -> RollingStatsBuffer:
        return RollingStatsBuffer(
            buffer_size=ROLLING_SCORE_BUFFER_SIZE,
            percentile=ROLLING_SCORE_PERCENTILE,
            min_buffer=ROLLING_SCORE_MIN_BUFFER,
        )

    def _new_feature_error_buffer(self) -> RollingStatsBuffer:
        return RollingStatsBuffer(
            buffer_size=ROLLING_SCORE_BUFFER_SIZE,
            percentile=ROLLING_SCORE_PERCENTILE,
            min_buffer=FEATURE_SHIFT_MIN_BUFFER,
        )

    def get_entity_adaptive_state(
        self,
        entity_id: str,
    ) -> tuple[EntityAdaptiveState, bool, bool]:
        entity_key = str(entity_id)
        state = self.entity_adaptive_state.get(entity_key)
        history_already_existed = state is not None
        if state is None:
            state = EntityAdaptiveState(
                score_buffer=self._new_score_buffer(),
                feature_error_buffers={},
            )
            self.entity_adaptive_state[entity_key] = state
            self.entity_state_creation_counts[entity_key] = self.entity_state_creation_counts.get(entity_key, 0) + 1
        creation_count = int(self.entity_state_creation_counts.get(entity_key, 1))
        history_reinitialized = bool(not history_already_existed and creation_count > 1)
        return state, history_already_existed, history_reinitialized

    def get_feature_error_buffer(
        self,
        entity_id: str,
        feature_name: str,
    ) -> RollingStatsBuffer:
        state, _, _ = self.get_entity_adaptive_state(entity_id)
        buffer = state.feature_error_buffers.get(feature_name)
        if buffer is None:
            buffer = self._new_feature_error_buffer()
            state.feature_error_buffers[feature_name] = buffer
        return buffer

    def restore_score_buffers_from_threshold_log(self) -> None:
        if not self.threshold_log_path.exists():
            return

        restored_scores = 0
        restored_entities: set[str] = set()
        try:
            with self.threshold_log_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    entity_id = str(payload.get("entity_id", "") or "").strip()
                    if not entity_id:
                        continue
                    score_value = payload.get("score")
                    if score_value is None:
                        score_value = payload.get("anomaly_score")
                    if score_value is None:
                        continue
                    state, _, _ = self.get_entity_adaptive_state(entity_id)
                    state.score_buffer.observe(float(score_value))
                    state.restored_score_count += 1
                    restored_scores += 1
                    restored_entities.add(entity_id)
        except Exception as exc:
            print(f"[WARN] Adaptive score history restore failed: {compact_reason(str(exc), limit=160)}")
            return

        if restored_scores > 0:
            print(
                "[STATE] restored adaptive score history | "
                f"entities={len(restored_entities)} | scores={restored_scores} | "
                f"source={self.threshold_log_path}"
            )

    def emit_adaptive_state_debug(
        self,
        *,
        entity_id: str,
        history_already_existed: bool,
        history_reinitialized: bool,
        previous_buffer_length: int,
        buffer_length_after_append: int,
        score_appended: bool,
        window_ready: bool,
    ) -> None:
        if not ADAPTIVE_STATE_DEBUG_LOGS:
            return
        print(
            f"[STATE] {entity_id} | "
            f"history_exists={history_already_existed} | "
            f"prev_buffer={int(previous_buffer_length)} | "
            f"after_append={int(buffer_length_after_append)} | "
            f"appended={bool(score_appended)} | "
            f"window_ready={bool(window_ready)} | "
            f"history_reinitialized={bool(history_reinitialized)}"
        )

    @staticmethod
    def _matches_any(value: str, patterns: list[str]) -> bool:
        text = str(value or "")
        return any(fnmatch.fnmatchcase(text, pattern) for pattern in patterns)

    def alert_filter_reason(self, record: dict[str, Any]) -> str | None:
        namespace = str(record.get("namespace", "") or record.get("container_app_du", ""))
        pod = str(record.get("pod", "") or "")
        container = str(record.get("container_id", "") or record.get("container", ""))

        if self.config.include_namespaces and not self._matches_any(namespace, self.config.include_namespaces):
            return f"namespace_not_allowlisted:{namespace}"
        if self.config.include_pods and not self._matches_any(pod, self.config.include_pods):
            return f"pod_not_allowlisted:{pod}"
        if self.config.include_containers and not self._matches_any(container, self.config.include_containers):
            return f"container_not_allowlisted:{container}"

        if self.config.exclude_namespaces and self._matches_any(namespace, self.config.exclude_namespaces):
            return f"namespace_excluded:{namespace}"
        if self.config.exclude_pods and self._matches_any(pod, self.config.exclude_pods):
            return f"pod_excluded:{pod}"
        if self.config.exclude_containers and self._matches_any(container, self.config.exclude_containers):
            return f"container_excluded:{container}"
        return None

    def recent_volatility(
        self,
        score_buffer: RollingStatsBuffer,
    ) -> tuple[float | None, float | None, float | None, int]:
        recent_scores = score_buffer.values()
        if recent_scores.size == 0:
            return None, None, None, 0

        if recent_scores.size > VOLATILITY_LOOKBACK:
            recent_scores = recent_scores[-VOLATILITY_LOOKBACK:]

        sample_size = int(recent_scores.size)
        mean_value = float(np.mean(recent_scores))
        std_value = float(np.std(recent_scores))
        if sample_size < VOLATILITY_MIN_BUFFER:
            return None, mean_value, std_value, sample_size

        denominator = max(abs(mean_value), STATS_STD_EPS)
        volatility = float(std_value / denominator)
        return volatility, mean_value, std_value, sample_size

    def adaptive_decision_profile(
        self,
        score_buffer: RollingStatsBuffer,
    ) -> tuple[AdaptiveDecisionProfile, float | None, float | None, float | None, int]:
        volatility, mean_value, std_value, sample_size = self.recent_volatility(score_buffer)
        if volatility is None:
            return DEFAULT_VOLATILITY_PROFILE, volatility, mean_value, std_value, sample_size
        if volatility < VOLATILITY_LOW_MAX:
            return LOW_VOLATILITY_PROFILE, volatility, mean_value, std_value, sample_size
        if volatility >= VOLATILITY_HIGH_MIN:
            return HIGH_VOLATILITY_PROFILE, volatility, mean_value, std_value, sample_size
        return MEDIUM_VOLATILITY_PROFILE, volatility, mean_value, std_value, sample_size

    def adaptive_cooldown_seconds(self, cooldown_windows: int) -> int:
        configured_windows = max(1, int(self.config.cooldown_windows))
        configured_seconds = max(0, int(self.config.cooldown_seconds))
        scaled_seconds = int(round(configured_seconds * (float(cooldown_windows) / float(configured_windows))))
        window_seconds = max(0, int(self.config.poll_interval)) * max(0, int(cooldown_windows))
        return max(scaled_seconds, window_seconds)

    def cooldown_status(
        self,
        adaptive_state: EntityAdaptiveState,
        time_stamp: int,
        window_id: int,
        cooldown_seconds: int,
        cooldown_windows: int,
    ) -> tuple[bool, str | None]:
        if adaptive_state.last_confirmed_time_stamp is None or adaptive_state.last_confirmed_window_id is None:
            return False, None

        effective_cooldown_seconds = max(
            0,
            int(adaptive_state.last_confirmed_cooldown_seconds or cooldown_seconds),
        )
        effective_cooldown_windows = max(
            0,
            int(adaptive_state.last_confirmed_cooldown_windows or cooldown_windows),
        )
        elapsed_seconds = int(time_stamp - int(adaptive_state.last_confirmed_time_stamp))
        elapsed_windows = int(window_id - int(adaptive_state.last_confirmed_window_id))
        seconds_active = elapsed_seconds < effective_cooldown_seconds
        windows_active = elapsed_windows < effective_cooldown_windows
        if not seconds_active and not windows_active:
            return False, None

        remaining_seconds = max(0, effective_cooldown_seconds - max(0, elapsed_seconds))
        remaining_windows = max(0, effective_cooldown_windows - max(0, elapsed_windows))
        return True, f"cooldown:{remaining_seconds}s,{remaining_windows}w"

    def record_alert_state(
        self,
        adaptive_state: EntityAdaptiveState,
        time_stamp: int,
        window_id: int,
        cooldown_seconds: int,
        cooldown_windows: int,
    ) -> None:
        adaptive_state.last_confirmed_time_stamp = int(time_stamp)
        adaptive_state.last_confirmed_window_id = int(window_id)
        adaptive_state.last_confirmed_cooldown_seconds = max(0, int(cooldown_seconds))
        adaptive_state.last_confirmed_cooldown_windows = max(0, int(cooldown_windows))

    def _build_openai_client(self) -> Any | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        if importlib.util.find_spec("openai") is None:
            return None
        from openai import OpenAI
        return OpenAI(api_key=api_key, timeout=self.gpt_config.request_timeout_seconds)

    def send_startup_telegram_test(self) -> None:
        if not TELEGRAM_STARTUP_TEST_ENABLED:
            return
        if not self.telegram_bot_token or not self.telegram_chat_id:
            print("[WARN] Telegram startup test skipped: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
            return

        try:
            sent, body, error = send_telegram_message(
                bot_token=str(self.telegram_bot_token),
                chat_id=str(self.telegram_chat_id),
                message_text=TELEGRAM_STARTUP_TEST_MESSAGE,
                timeout_seconds=int(self.config.request_timeout),
            )
            append_jsonl(
                self.telegram_log_path,
                {
                    "timestamp": now_iso(),
                    "startup_test": True,
                    "telegram_sent": bool(sent),
                    "error": None if sent else compact_reason(str(error or "telegram_startup_test_failed"), limit=120),
                    "message_text": TELEGRAM_STARTUP_TEST_MESSAGE,
                    "message_id": None if body is None else body.get("result", {}).get("message_id"),
                },
            )
            if sent:
                print("[TELEGRAM] startup test sent")
            else:
                print(f"[WARN] Telegram startup test failed: {compact_reason(str(error or 'unknown_error'), limit=160)}")
        except Exception as exc:
            error_text = compact_reason(str(exc), limit=160)
            append_jsonl(
                self.telegram_log_path,
                {
                    "timestamp": now_iso(),
                    "startup_test": True,
                    "telegram_sent": False,
                    "error": error_text,
                    "message_text": TELEGRAM_STARTUP_TEST_MESSAGE,
                },
            )
            print(f"[WARN] Telegram startup test failed: {error_text}")

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
        bundle = load_model_artifacts(model_dir=model_dir, device=device)
        feature_columns = list(bundle["feature_columns"])
        context_columns = list(bundle["context_columns"])
        category_metadata = self._load_category_metadata(model_dir, context_columns)
        categorical_columns = list(category_metadata.get("columns", []))
        numeric_columns = [column for column in context_columns if column not in categorical_columns]
        bundle.update(
            {
                "feature_columns": feature_columns,
                "numeric_context_columns": numeric_columns,
                "categorical_context_columns": categorical_columns,
                "category_encoder": CategoryEncoder(category_metadata),
            }
        )
        return bundle

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

    def evaluate_feature_shift(
        self,
        entity_id: str,
        top_k_features: list[str],
        top_k_feature_errors: list[float],
    ) -> tuple[bool, str | None]:
        if not ENABLE_FEATURE_SHIFT_CANDIDATE:
            return False, None
        if not top_k_features or not top_k_feature_errors:
            return False, None

        top_feature = str(top_k_features[0])
        top_error = float(top_k_feature_errors[0])
        buffer = self.get_feature_error_buffer(entity_id, top_feature)
        stats = buffer.summary()
        if stats is None:
            return False, None

        percentile_threshold = float(stats["percentile_threshold"])
        if top_error > percentile_threshold:
            return True, f"feature_shift:{top_feature}>p{ROLLING_SCORE_PERCENTILE:.0f}"

        std = float(stats["std"])
        if std > STATS_STD_EPS:
            z_score = (top_error - float(stats["mean"])) / std
            if z_score > FEATURE_SHIFT_ZSCORE_THRESHOLD:
                return True, f"feature_shift:{top_feature}_z>{FEATURE_SHIFT_ZSCORE_THRESHOLD:.1f}"

        return False, None

    def observe_feature_errors(
        self,
        entity_id: str,
        feature_error_vector: list[float],
    ) -> None:
        for feature_name, feature_error in zip(self.bundle["feature_columns"], feature_error_vector):
            self.get_feature_error_buffer(entity_id, feature_name).observe(float(feature_error))

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
        adaptive_state, history_already_existed, history_reinitialized = self.get_entity_adaptive_state(entity_id)
        existing_score_history_length = adaptive_state.score_buffer.size()
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
        fallback_threshold = float(self.detector.threshold)
        response = {
            "entity_id": entity_id,
            "container_id": container,
            "machine_id": machine_id,
            "mode": str(result.mode),
            "window_id": window_id,
            "window_ready": bool(result.ready),
            "threshold": fallback_threshold,
            "fallback_threshold": fallback_threshold,
            "static_threshold": fallback_threshold,
            "fixed_threshold": fallback_threshold,
            "dynamic_threshold": None,
            "final_threshold": fallback_threshold,
            "threshold_mode": "warmup",
            "smoothed_score": None,
            "score_buffer_size": existing_score_history_length,
            "score_buffer_size_before_decision": existing_score_history_length,
            "score_buffer_size_after_append": existing_score_history_length,
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
            "z_score": None,
            "z_score_reason": f"warmup<{ROLLING_SCORE_MIN_BUFFER}",
            "recent_volatility": None,
            "recent_volatility_mean": None,
            "recent_volatility_std": None,
            "recent_volatility_sample_size": 0,
            "volatility_band": DEFAULT_VOLATILITY_PROFILE.band,
            "adaptive_threshold_percentile": float(DEFAULT_VOLATILITY_PROFILE.threshold_percentile),
            "adaptive_consecutive_windows": int(DEFAULT_VOLATILITY_PROFILE.consecutive_windows),
            "adaptive_smoothing_window": int(DEFAULT_VOLATILITY_PROFILE.smoothing_window),
            "adaptive_cooldown_windows": int(DEFAULT_VOLATILITY_PROFILE.cooldown_windows),
            "adaptive_cooldown_seconds": int(self.adaptive_cooldown_seconds(DEFAULT_VOLATILITY_PROFILE.cooldown_windows)),
            "current_consecutive_breach_count": int(adaptive_state.consecutive_breach_count),
            "confirmed_anomaly": False,
            "significant_feature_shift": False,
            "decision_reason": "buffering",
            "alert_suppressed_reason": None,
            "history_already_existed": history_already_existed,
            "history_reinitialized": history_reinitialized,
            "history_restored_score_count": int(adaptive_state.restored_score_count),
            "recon_score": None,
            "forecast_score": None,
            "final_score": None,
        }

        if not result.ready:
            self.emit_adaptive_state_debug(
                entity_id=entity_id,
                history_already_existed=history_already_existed,
                history_reinitialized=history_reinitialized,
                previous_buffer_length=existing_score_history_length,
                buffer_length_after_append=existing_score_history_length,
                score_appended=False,
                window_ready=False,
            )
            response["status"] = "buffering"
            return response

        self.ready_window_counts[entity_id] = window_id + 1
        anomaly_score = float(result.anomaly_score or 0.0)
        recon_score = float(result.recon_score) if result.recon_score is not None else None
        forecast_score = float(result.forecast_score) if result.forecast_score is not None else None
        top_k_features = [str(value) for value in (result.top_k_features or [])]
        top_k_feature_errors = [float(value) for value in (result.top_k_feature_errors or [])]
        feature_error_vector = [float(value) for value in (result.feature_error_vector or [])]

        history_size = adaptive_state.score_buffer.size()
        score_history = adaptive_state.score_buffer.values()
        score_stats = adaptive_state.score_buffer.summary()
        adaptive_profile, recent_volatility, volatility_mean, volatility_std, volatility_sample_size = self.adaptive_decision_profile(
            adaptive_state.score_buffer
        )
        smoothed_history = trailing_moving_average(score_history, adaptive_profile.smoothing_window)
        smoothed_score = float(
            trailing_moving_average(
                np.append(score_history, np.asarray([anomaly_score], dtype=np.float32)),
                adaptive_profile.smoothing_window,
            )[-1]
        )
        dynamic_threshold = compute_percentile_threshold(smoothed_history, adaptive_profile.threshold_percentile)
        adaptive_threshold_ready = bool(history_size >= ROLLING_SCORE_MIN_BUFFER and dynamic_threshold is not None)
        threshold_mode = "dynamic" if adaptive_threshold_ready else "warmup"
        final_threshold = dynamic_threshold if adaptive_threshold_ready and dynamic_threshold is not None else fallback_threshold
        if ENABLE_ZSCORE_CANDIDATE:
            z_score, z_score_reason = adaptive_state.score_buffer.z_score_details(anomaly_score)
        else:
            z_score, z_score_reason = None, "zscore_disabled"
        score_threshold_hit = smoothed_score > final_threshold
        z_score_hit = bool(z_score is not None and z_score > ZSCORE_THRESHOLD)
        significant_feature_shift, feature_shift_reason = self.evaluate_feature_shift(
            entity_id=entity_id,
            top_k_features=top_k_features,
            top_k_feature_errors=top_k_feature_errors,
        )
        adaptive_cooldown_seconds = self.adaptive_cooldown_seconds(adaptive_profile.cooldown_windows)
        if score_threshold_hit:
            adaptive_state.consecutive_breach_count += 1
        else:
            adaptive_state.consecutive_breach_count = 0
        consecutive_breach_count = int(adaptive_state.consecutive_breach_count)
        candidate_signal = bool(score_threshold_hit or z_score_hit or significant_feature_shift)
        cooldown_active, cooldown_reason = self.cooldown_status(
            adaptive_state=adaptive_state,
            time_stamp=time_stamp,
            window_id=window_id,
            cooldown_seconds=adaptive_cooldown_seconds,
            cooldown_windows=adaptive_profile.cooldown_windows,
        )

        decision_reasons: list[str] = []
        if score_threshold_hit:
            if adaptive_threshold_ready:
                decision_reasons.append("smoothed_score_above_dynamic_threshold")
            else:
                decision_reasons.append("smoothed_score_above_fallback_threshold")
        if z_score_hit:
            decision_reasons.append(f"z_score>{ZSCORE_THRESHOLD:.1f}")
        if significant_feature_shift and feature_shift_reason:
            decision_reasons.append(feature_shift_reason)
        if score_threshold_hit:
            decision_reasons.append(
                f"consecutive_breach={consecutive_breach_count}/{adaptive_profile.consecutive_windows}"
            )

        confirmed_anomaly = False
        if candidate_signal and cooldown_active:
            status = "suppressed"
            adaptive_state.consecutive_breach_count = 0
            if cooldown_reason:
                decision_reasons.append(cooldown_reason)
        elif score_threshold_hit and consecutive_breach_count >= adaptive_profile.consecutive_windows:
            status = "confirmed_anomaly"
            confirmed_anomaly = True
            self.record_alert_state(
                adaptive_state=adaptive_state,
                time_stamp=time_stamp,
                window_id=window_id,
                cooldown_seconds=adaptive_cooldown_seconds,
                cooldown_windows=adaptive_profile.cooldown_windows,
            )
            adaptive_state.consecutive_breach_count = 0
            decision_reasons.append("confirmed_after_consecutive_breaches")
        elif candidate_signal:
            status = "candidate"
        else:
            status = "normal"

        anomaly_candidate = bool(candidate_signal)
        decision_reason = ",".join(decision_reasons) if decision_reasons else "below_all_candidate_rules"

        adaptive_state.score_buffer.observe(anomaly_score)
        score_buffer_size_after_append = adaptive_state.score_buffer.size()
        self.emit_adaptive_state_debug(
            entity_id=entity_id,
            history_already_existed=history_already_existed,
            history_reinitialized=history_reinitialized,
            previous_buffer_length=history_size,
            buffer_length_after_append=score_buffer_size_after_append,
            score_appended=True,
            window_ready=True,
        )
        self.observe_feature_errors(entity_id, feature_error_vector)
        response.update(
            {
                "status": status,
                "predicted_label": int(confirmed_anomaly),
                "anomaly_candidate": bool(anomaly_candidate),
                "confirmed_anomaly": bool(confirmed_anomaly),
                "anomaly_score": anomaly_score,
                "recon_score": recon_score,
                "forecast_score": forecast_score,
                "final_score": anomaly_score,
                "smoothed_score": smoothed_score,
                "threshold": final_threshold,
                "fallback_threshold": fallback_threshold,
                "static_threshold": fallback_threshold,
                "fixed_threshold": fallback_threshold,
                "dynamic_threshold": dynamic_threshold,
                "final_threshold": final_threshold,
                "threshold_mode": threshold_mode,
                "score_buffer_size": score_buffer_size_after_append,
                "score_buffer_size_before_decision": history_size,
                "score_buffer_size_after_append": score_buffer_size_after_append,
                "score_over_threshold": smoothed_score - final_threshold,
                "z_score": z_score,
                "z_score_reason": z_score_reason,
                "significant_feature_shift": bool(significant_feature_shift),
                "recent_volatility": recent_volatility,
                "recent_volatility_mean": volatility_mean,
                "recent_volatility_std": volatility_std,
                "recent_volatility_sample_size": int(volatility_sample_size),
                "volatility_band": adaptive_profile.band,
                "adaptive_threshold_percentile": float(adaptive_profile.threshold_percentile),
                "adaptive_consecutive_windows": int(adaptive_profile.consecutive_windows),
                "adaptive_smoothing_window": int(adaptive_profile.smoothing_window),
                "adaptive_cooldown_windows": int(adaptive_profile.cooldown_windows),
                "adaptive_cooldown_seconds": int(adaptive_cooldown_seconds),
                "current_consecutive_breach_count": int(consecutive_breach_count),
                "decision_reason": decision_reason,
                "alert_suppressed_reason": cooldown_reason if status == "suppressed" else None,
                "top_k_features": top_k_features,
                "top_k_feature_errors": top_k_feature_errors,
                "feature_error_vector": feature_error_vector,
                "adaptive_threshold_ready": adaptive_threshold_ready,
                "rolling_score_min": None if score_stats is None else float(score_stats["min"]),
                "rolling_score_max": None if score_stats is None else float(score_stats["max"]),
                "rolling_score_mean": None if score_stats is None else float(score_stats["mean"]),
                "rolling_score_percentile_threshold": dynamic_threshold,
                "history_already_existed": history_already_existed,
                "history_reinitialized": history_reinitialized,
                "history_restored_score_count": int(adaptive_state.restored_score_count),
                "metadata": result.metadata or {},
            }
        )
        self.log_threshold_decision(response)
        return response

    def adjudicate_with_gpt(self, candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        decision = adjudicate_anomaly(candidate, config=self.gpt_config)
        if not bool(decision.get("used_fallback", False)):
            return decision, None

        call_reason = str(decision.get("call_reason", "") or self.gpt_unavailable_reason or "gpt_unavailable")
        call_error = str(decision.get("call_error", "") or "").strip()
        if call_error:
            return None, f"{call_reason}:{compact_reason(call_error, limit=120)}"
        return None, call_reason

    def build_gpt_failure_telegram_payload(
        self,
        result: dict[str, Any],
        gpt_error: str,
    ) -> dict[str, Any]:
        return {
            **result,
            "gpt_failed": True,
            "gpt_failure_reason": str(gpt_error),
            "label": "model_candidate",
            "severity": "unknown",
            "explanation": "",
            "model_explanation": (
                f"{str(result.get('mode', 'reconstruction')).title()} detector flagged this window before GPT adjudication. "
                f"Candidate rule: {result.get('decision_reason', 'unknown')}"
            ),
            "recommended_action": (
                "Review container metrics manually and fix GPT quota or API availability "
                "before relying on GPT adjudication."
            ),
        }

    def build_live_result_telegram_payload(
        self,
        result: dict[str, Any],
        *,
        label: str,
        severity: str,
        explanation: str,
        recommended_action: str,
    ) -> dict[str, Any]:
        return {
            **result,
            "label": str(label),
            "severity": str(severity),
            "explanation": str(explanation),
            "recommended_action": str(recommended_action),
        }

    def send_telegram_alert(self, decision: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.telegram_enabled:
            return False, "telegram_disabled"

        message_text = format_telegram_alert(decision)
        try:
            sent, body, error = send_telegram_message(
                bot_token=str(self.telegram_bot_token),
                chat_id=str(self.telegram_chat_id),
                message_text=message_text,
                timeout_seconds=int(self.config.request_timeout),
            )
            if not sent:
                error_text = compact_reason(str(error or "telegram_send_failed"), limit=120)
                self.log_telegram_message(
                    decision=decision,
                    message_text=message_text,
                    sent=False,
                    error=error_text,
                )
                return False, error_text
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

    def deliver_telegram_result(
        self,
        decision: dict[str, Any],
        *,
        entity_id: str,
        time_stamp: int,
        success_message: str,
    ) -> bool:
        if not self.telegram_enabled:
            decision["alert_suppressed_reason"] = "telegram_disabled"
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=False)
            return False

        sent, telegram_error = self.send_telegram_alert(decision)
        if sent:
            print(success_message)
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=True)
            return True

        decision["alert_suppressed_reason"] = telegram_error
        if self.should_print_skip(entity_id, str(telegram_error), time_stamp):
            print(f"[SKIP] {entity_id} | reason={telegram_error}")
        if ENABLE_GPT_LOGGING:
            self.log_gpt_decision(decision, telegram_sent=False)
        return False

    def log_gpt_decision(self, decision: dict[str, Any], telegram_sent: bool) -> None:
        payload = {
            "timestamp": now_iso(),
            "entity_id": decision.get("entity_id"),
            "mode": decision.get("mode"),
            "recon_score": decision.get("recon_score"),
            "forecast_score": decision.get("forecast_score"),
            "final_score": decision.get("final_score", decision.get("anomaly_score")),
            "anomaly_score": decision.get("anomaly_score"),
            "smoothed_score": decision.get("smoothed_score"),
            "threshold": decision.get("threshold"),
            "fallback_threshold": decision.get("fallback_threshold"),
            "static_threshold": decision.get("static_threshold"),
            "fixed_threshold": decision.get("fixed_threshold"),
            "dynamic_threshold": decision.get("dynamic_threshold"),
            "final_threshold": decision.get("final_threshold"),
            "threshold_mode": decision.get("threshold_mode"),
            "score_buffer_size": decision.get("score_buffer_size"),
            "z_score": decision.get("z_score"),
            "z_score_reason": decision.get("z_score_reason"),
            "recent_volatility": decision.get("recent_volatility"),
            "volatility_band": decision.get("volatility_band"),
            "adaptive_threshold_percentile": decision.get("adaptive_threshold_percentile"),
            "adaptive_consecutive_windows": decision.get("adaptive_consecutive_windows"),
            "adaptive_smoothing_window": decision.get("adaptive_smoothing_window"),
            "adaptive_cooldown_windows": decision.get("adaptive_cooldown_windows"),
            "current_consecutive_breach_count": decision.get("current_consecutive_breach_count"),
            "decision_reason": decision.get("decision_reason"),
            "alert_suppressed_reason": decision.get("alert_suppressed_reason"),
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
            "mode": decision.get("mode"),
            "recon_score": decision.get("recon_score"),
            "forecast_score": decision.get("forecast_score"),
            "final_score": decision.get("final_score", decision.get("anomaly_score")),
            "container_id": decision.get("container_id"),
            "machine_id": decision.get("machine_id"),
            "anomaly_score": decision.get("anomaly_score"),
            "smoothed_score": decision.get("smoothed_score"),
            "threshold": decision.get("threshold"),
            "fallback_threshold": decision.get("fallback_threshold"),
            "static_threshold": decision.get("static_threshold"),
            "fixed_threshold": decision.get("fixed_threshold"),
            "dynamic_threshold": decision.get("dynamic_threshold"),
            "final_threshold": decision.get("final_threshold"),
            "threshold_mode": decision.get("threshold_mode"),
            "score_buffer_size": decision.get("score_buffer_size"),
            "z_score": decision.get("z_score"),
            "z_score_reason": decision.get("z_score_reason"),
            "recent_volatility": decision.get("recent_volatility"),
            "volatility_band": decision.get("volatility_band"),
            "adaptive_threshold_percentile": decision.get("adaptive_threshold_percentile"),
            "adaptive_consecutive_windows": decision.get("adaptive_consecutive_windows"),
            "adaptive_smoothing_window": decision.get("adaptive_smoothing_window"),
            "adaptive_cooldown_windows": decision.get("adaptive_cooldown_windows"),
            "current_consecutive_breach_count": decision.get("current_consecutive_breach_count"),
            "decision_reason": decision.get("decision_reason"),
            "alert_suppressed_reason": decision.get("alert_suppressed_reason"),
            "status": status,
            "response_id": decision.get("response_id"),
            "gpt_model": decision.get("gpt_model"),
            "label": decision.get("label"),
            "severity": decision.get("severity"),
            "explanation": decision.get("explanation"),
            "recommended_action": decision.get("recommended_action"),
            "gpt_failed": decision.get("gpt_failed", False),
            "gpt_failure_reason": decision.get("gpt_failure_reason"),
            "call_reason": decision.get("call_reason"),
            "call_error": decision.get("call_error"),
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
            "mode": decision.get("mode"),
            "recon_score": decision.get("recon_score"),
            "forecast_score": decision.get("forecast_score"),
            "final_score": decision.get("final_score", decision.get("anomaly_score")),
            "container_id": decision.get("container_id"),
            "machine_id": decision.get("machine_id"),
            "anomaly_score": decision.get("anomaly_score"),
            "smoothed_score": decision.get("smoothed_score"),
            "threshold": decision.get("threshold"),
            "fallback_threshold": decision.get("fallback_threshold"),
            "static_threshold": decision.get("static_threshold"),
            "fixed_threshold": decision.get("fixed_threshold"),
            "dynamic_threshold": decision.get("dynamic_threshold"),
            "final_threshold": decision.get("final_threshold"),
            "threshold_mode": decision.get("threshold_mode"),
            "score_buffer_size": decision.get("score_buffer_size"),
            "z_score": decision.get("z_score"),
            "z_score_reason": decision.get("z_score_reason"),
            "recent_volatility": decision.get("recent_volatility"),
            "volatility_band": decision.get("volatility_band"),
            "adaptive_threshold_percentile": decision.get("adaptive_threshold_percentile"),
            "adaptive_consecutive_windows": decision.get("adaptive_consecutive_windows"),
            "adaptive_smoothing_window": decision.get("adaptive_smoothing_window"),
            "adaptive_cooldown_windows": decision.get("adaptive_cooldown_windows"),
            "current_consecutive_breach_count": decision.get("current_consecutive_breach_count"),
            "decision_reason": decision.get("decision_reason"),
            "alert_suppressed_reason": decision.get("alert_suppressed_reason"),
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
            "mode": window_result.get("mode"),
            "recon_score": window_result.get("recon_score"),
            "forecast_score": window_result.get("forecast_score"),
            "final_score": window_result.get("final_score", window_result.get("anomaly_score")),
            "score": window_result.get("anomaly_score"),
            "smoothed_score": window_result.get("smoothed_score"),
            "fallback_threshold": window_result.get("fallback_threshold"),
            "static_threshold": window_result.get("static_threshold"),
            "fixed_threshold": window_result.get("fixed_threshold"),
            "dynamic_threshold": window_result.get("dynamic_threshold"),
            "final_threshold": window_result.get("final_threshold"),
            "threshold_mode": window_result.get("threshold_mode"),
            "score_buffer_size": window_result.get("score_buffer_size"),
            "score_buffer_size_before_decision": window_result.get("score_buffer_size_before_decision"),
            "score_buffer_size_after_append": window_result.get("score_buffer_size_after_append"),
            "adaptive_threshold_ready": window_result.get("adaptive_threshold_ready"),
            "z_score": window_result.get("z_score"),
            "z_score_reason": window_result.get("z_score_reason"),
            "recent_volatility": window_result.get("recent_volatility"),
            "recent_volatility_mean": window_result.get("recent_volatility_mean"),
            "recent_volatility_std": window_result.get("recent_volatility_std"),
            "recent_volatility_sample_size": window_result.get("recent_volatility_sample_size"),
            "volatility_band": window_result.get("volatility_band"),
            "adaptive_threshold_percentile": window_result.get("adaptive_threshold_percentile"),
            "adaptive_consecutive_windows": window_result.get("adaptive_consecutive_windows"),
            "adaptive_smoothing_window": window_result.get("adaptive_smoothing_window"),
            "adaptive_cooldown_windows": window_result.get("adaptive_cooldown_windows"),
            "adaptive_cooldown_seconds": window_result.get("adaptive_cooldown_seconds"),
            "current_consecutive_breach_count": window_result.get("current_consecutive_breach_count"),
            "top_k_features": window_result.get("top_k_features", []),
            "top_k_feature_errors": window_result.get("top_k_feature_errors", []),
            "rolling_score_min": window_result.get("rolling_score_min"),
            "rolling_score_max": window_result.get("rolling_score_max"),
            "rolling_score_mean": window_result.get("rolling_score_mean"),
            "rolling_score_percentile_threshold": window_result.get("rolling_score_percentile_threshold"),
            "history_already_existed": window_result.get("history_already_existed"),
            "history_reinitialized": window_result.get("history_reinitialized"),
            "history_restored_score_count": window_result.get("history_restored_score_count"),
            "decision_reason": window_result.get("decision_reason"),
            "alert_suppressed_reason": window_result.get("alert_suppressed_reason"),
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

    def emit_result_line(self, prefix: str, result: dict[str, Any]) -> None:
        z_display = format_metric_value(result.get("z_score"), precision=3)
        z_reason = str(result.get("z_score_reason", "") or "").strip()
        z_fragment = f"z={z_display}" if result.get("z_score") is not None else f"z={z_display}:{z_reason}"
        print(
            f"[{prefix}] {result.get('entity_id', 'unknown')} | "
            f"score={format_metric_value(result.get('anomaly_score'))} | "
            f"smoothed={format_metric_value(result.get('smoothed_score'))} | "
            f"static={format_metric_value(result.get('static_threshold', result.get('fallback_threshold')))} | "
            f"dynamic={format_metric_value(result.get('dynamic_threshold'))} | "
            f"final={format_metric_value(result.get('final_threshold'))} | "
            f"{z_fragment} | "
            f"vol={format_metric_value(result.get('recent_volatility'), precision=4)} | "
            f"band={result.get('volatility_band', 'unknown')} | "
            f"pctl={format_metric_value(result.get('adaptive_threshold_percentile'), precision=2)} | "
            f"hits={int(result.get('current_consecutive_breach_count', 0))}/{int(result.get('adaptive_consecutive_windows', 0))} | "
            f"smooth_w={int(result.get('adaptive_smoothing_window', 0))} | "
            f"cooldown_w={int(result.get('adaptive_cooldown_windows', 0))} | "
            f"buffer={int(result.get('score_buffer_size', 0))} | "
            f"mode={result.get('threshold_mode', 'unknown')} | "
            f"reason={result.get('decision_reason', 'unknown')}"
        )

    def handle_ready_result(self, result: dict[str, Any], time_stamp: int) -> None:
        entity_id = str(result["entity_id"])
        status = str(result.get("status", "normal") or "normal")

        if status == "normal":
            self.emit_result_line("NORMAL", result)
            return
        if status == "candidate":
            self.emit_result_line("CANDIDATE", result)
            return
        if status == "suppressed":
            self.emit_result_line("SUPPRESSED", result)
            if result.get("alert_suppressed_reason") and self.should_print_skip(
                entity_id,
                str(result.get("alert_suppressed_reason")),
                time_stamp,
            ):
                print(f"[SUPPRESS] {entity_id} | reason={result.get('alert_suppressed_reason')}")
            return
        if status != "confirmed_anomaly" or int(result.get("predicted_label", 0)) != 1:
            self.emit_result_line("NORMAL", result)
            return

        filter_reason = self.alert_filter_reason(result)
        if filter_reason:
            result["alert_suppressed_reason"] = filter_reason
            self.emit_result_line("FILTERED", result)
            print(f"[SUPPRESS] {entity_id} | reason={filter_reason}")
            filtered_decision = self.build_live_result_telegram_payload(
                result,
                label="warning",
                severity="low",
                explanation=f"Matched configured filter: {filter_reason}. Forwarded to Telegram because all results are enabled.",
                recommended_action="monitor",
            )
            self.deliver_telegram_result(
                filtered_decision,
                entity_id=entity_id,
                time_stamp=time_stamp,
                success_message=f"[TELEGRAM] {entity_id} | sent | filtered",
            )
            return

        self.emit_result_line("CONFIRMED", result)

        decision, gpt_error = self.adjudicate_with_gpt(result)
        if decision is None:
            fallback_decision = self.build_gpt_failure_telegram_payload(result, str(gpt_error))
            self.log_gpt_response(fallback_decision, status="gpt_failed_fallback")
            self.deliver_telegram_result(
                fallback_decision,
                entity_id=entity_id,
                time_stamp=time_stamp,
                success_message=f"[TELEGRAM] {entity_id} | sent | gpt_failed={gpt_error}",
            )
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
            decision["alert_suppressed_reason"] = skip_reason
            if self.should_print_skip(entity_id, str(skip_reason), time_stamp):
                print(f"[SKIP] {entity_id} | reason={skip_reason}")
            if ENABLE_GPT_LOGGING:
                self.log_gpt_decision(decision, telegram_sent=False)
            return

        self.deliver_telegram_result(
            decision,
            entity_id=entity_id,
            time_stamp=time_stamp,
            success_message=f"[TELEGRAM] {entity_id} | sent",
        )

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
        print(f"Cooldown  : {self.config.cooldown_seconds}s / {self.config.cooldown_windows} windows")
        print(f"Log file  : {self.decision_log_path}")
        print(f"GPT log   : {self.gpt_response_log_path}")
        print(f"Telegram log: {self.telegram_log_path}")
        print(f"Threshold log: {self.threshold_log_path}")
        print(
            "Alert filters: "
            f"include_ns={self.config.include_namespaces or ['*']} | "
            f"exclude_ns={self.config.exclude_namespaces or []} | "
            f"include_pods={self.config.include_pods or ['*']} | "
            f"exclude_pods={self.config.exclude_pods or []} | "
            f"include_containers={self.config.include_containers or ['*']} | "
            f"exclude_containers={self.config.exclude_containers or []}"
        )
        print(
            "Adaptive threshold: "
            f"buffer={ROLLING_SCORE_BUFFER_SIZE}, "
            f"percentile={ROLLING_SCORE_PERCENTILE}, "
            f"min_buffer={ROLLING_SCORE_MIN_BUFFER}, "
            f"zscore={ENABLE_ZSCORE_CANDIDATE}, "
            f"feature_shift={ENABLE_FEATURE_SHIFT_CANDIDATE}"
        )
        print(
            "Adaptive decision bands: "
            f"lookback={VOLATILITY_LOOKBACK}, "
            f"low<{VOLATILITY_LOW_MAX:.4f}, "
            f"high>={VOLATILITY_HIGH_MIN:.4f}, "
            f"low=(p{LOW_VOLATILITY_PROFILE.threshold_percentile:.1f},hits={LOW_VOLATILITY_PROFILE.consecutive_windows},"
            f"smooth={LOW_VOLATILITY_PROFILE.smoothing_window},cooldown={LOW_VOLATILITY_PROFILE.cooldown_windows}w), "
            f"medium=(p{MEDIUM_VOLATILITY_PROFILE.threshold_percentile:.1f},hits={MEDIUM_VOLATILITY_PROFILE.consecutive_windows},"
            f"smooth={MEDIUM_VOLATILITY_PROFILE.smoothing_window},cooldown={MEDIUM_VOLATILITY_PROFILE.cooldown_windows}w), "
            f"high=(p{HIGH_VOLATILITY_PROFILE.threshold_percentile:.1f},hits={HIGH_VOLATILITY_PROFILE.consecutive_windows},"
            f"smooth={HIGH_VOLATILITY_PROFILE.smoothing_window},cooldown={HIGH_VOLATILITY_PROFILE.cooldown_windows}w)"
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
    parser.add_argument("--cooldown-windows", type=int, default=env_int("LIVE_ALERT_COOLDOWN_WINDOWS", 3))
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
        cooldown_windows=max(0, int(args.cooldown_windows)),
        skip_log_cooldown_seconds=max(0, int(args.skip_log_cooldown_seconds)),
        results_dir=Path(args.results_dir).expanduser().resolve(),
    )


def main() -> None:
    config = parse_args()
    runner = DirectPrometheusAnomalyRunner(config)
    runner.run_forever()


if __name__ == "__main__":
    main()
