from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import torch

from .config import PipelineConfig
from .gpt_adjudicator import GPTAdjudicator
from .model import FiLMAutoencoder


@dataclass
class RuntimeAssets:
    model: FiLMAutoencoder
    x_scaler: Any
    c_scaler: Any
    detector_meta: dict[str, Any]
    ae_meta: dict[str, Any]
    feature_columns: list[str]
    base_threshold: float
    window_size: int
    context_dim: int
    device: torch.device


@dataclass
class WindowAnomalyScores:
    all_score: float
    top_score: float
    feature_error_map: dict[str, float]
    ranked_feature_errors: list[tuple[str, float]]
    top_features: list[str]


@dataclass
class RealtimeDecision:
    target_key: tuple[str, str, str]
    all_score: float
    all_threshold: float
    all_status: str
    top_score: float
    top_threshold: float
    top_status: str
    top_features: list[str]
    ranked_feature_errors: list[tuple[str, float]]
    reason: str
    gpt_decision: dict[str, Any] | None = None


def setup_logger(name: str, log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def safe_load_joblib(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    return default


def extract_threshold(detector_meta: dict[str, Any] | None) -> float:
    if isinstance(detector_meta, dict):
        for key in ["threshold", "thresh", "best_threshold", "anomaly_threshold"]:
            if key in detector_meta:
                return float(detector_meta[key])
    return 1.0


def extract_window_size(ae_meta: dict[str, Any] | None, detector_meta: dict[str, Any] | None, fallback: int) -> int:
    for meta in [ae_meta, detector_meta]:
        if isinstance(meta, dict):
            for key in ["window_size", "seq_len", "sequence_length", "lookback"]:
                if key in meta:
                    return int(meta[key])
    return int(fallback)


def infer_context_dim(c_scaler: Any) -> int:
    if hasattr(c_scaler, "n_features_in_"):
        return int(c_scaler.n_features_in_)
    return 2


def build_label_maps(seen_keys: set[tuple[str, str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    namespaces = sorted({key[0] for key in seen_keys})
    containers = sorted({key[2] for key in seen_keys})
    return {value: idx for idx, value in enumerate(namespaces)}, {value: idx for idx, value in enumerate(containers)}


def build_context_vector(namespace: str, container: str, ns_map: dict[str, int], ct_map: dict[str, int], context_dim: int) -> np.ndarray:
    vector = np.asarray([float(ns_map.get(namespace, -1)), float(ct_map.get(container, -1))], dtype=np.float32)
    if len(vector) < context_dim:
        vector = np.pad(vector, (0, context_dim - len(vector)), mode="constant", constant_values=0.0)
    elif len(vector) > context_dim:
        vector = vector[:context_dim]
    return vector.reshape(1, -1)


def _select_runtime_asset_paths(config: PipelineConfig) -> dict[str, Path]:
    if config.realtime.legacy_model_path.exists():
        return {
        "model_path": config.realtime.legacy_model_path,
        "x_scaler_path": config.realtime.legacy_x_scaler_path,
        "c_scaler_path": config.realtime.legacy_c_scaler_path,
        "detector_meta_path": config.realtime.legacy_detector_meta_path,
        "ae_meta_path": config.realtime.legacy_ae_meta_path,
    }
    return {
        "model_path": config.paths.checkpoint_path,
        "x_scaler_path": config.paths.x_scaler_path,
        "c_scaler_path": config.paths.c_scaler_path,
        "detector_meta_path": config.paths.detector_meta_joblib,
        "ae_meta_path": config.paths.detector_meta_joblib,
    }


def load_runtime_assets(config: PipelineConfig, logger: logging.Logger | None = None) -> RuntimeAssets:
    logger = logger or logging.getLogger(__name__)
    paths = _select_runtime_asset_paths(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_meta = safe_load_joblib(paths["ae_meta_path"], default={}) or {}
    detector_meta = safe_load_joblib(paths["detector_meta_path"], default={}) or {}
    x_scaler = joblib.load(paths["x_scaler_path"])
    c_scaler = joblib.load(paths["c_scaler_path"])

    base_threshold = extract_threshold(detector_meta)
    context_dim = infer_context_dim(c_scaler)
    window_size = extract_window_size(ae_meta, detector_meta, config.dataset.window_size)

    checkpoint = torch.load(paths["model_path"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        window_size = int(checkpoint.get("window_size", window_size))
        n_features = int(checkpoint.get("n_features", len(config.dataset.feature_columns)))
        context_dim = int(checkpoint.get("context_dim", context_dim))
        units = int(checkpoint.get("units", config.train.units))
        latent = int(checkpoint.get("latent", config.train.latent))
        feature_columns = list(checkpoint.get("feature_columns", config.realtime.feature_columns))
    else:
        state_dict = checkpoint
        n_features = len(config.realtime.feature_columns)
        units = config.train.units
        latent = config.train.latent
        feature_columns = list(config.realtime.feature_columns)

    model = FiLMAutoencoder(
        window_size=window_size,
        n_features=n_features,
        context_dim=context_dim,
        units=units,
        latent=latent,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded runtime assets | model=%s window_size=%s features=%s context_dim=%s threshold=%.6f",
        Path(paths["model_path"]).name,
        window_size,
        n_features,
        context_dim,
        base_threshold,
    )

    return RuntimeAssets(
        model=model,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        detector_meta=detector_meta,
        ae_meta=ae_meta,
        feature_columns=feature_columns,
        base_threshold=base_threshold,
        window_size=window_size,
        context_dim=context_dim,
        device=device,
    )


def query_prometheus(prom_url: str, query: str, timeout: int, logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    logger = logger or logging.getLogger(__name__)
    try:
        response = requests.get(prom_url, params={"query": query}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload["data"]["result"]
    except Exception as exc:
        logger.error("Prometheus query failed: %s", exc)
        return []


def normalize_metric_labels(metric: dict[str, Any]) -> dict[str, str]:
    return {
        "namespace": metric.get("namespace") or metric.get("container_label_io_kubernetes_pod_namespace") or "",
        "pod": metric.get("pod") or metric.get("container_label_io_kubernetes_pod_name") or "",
        "container": metric.get("container") or metric.get("container_label_io_kubernetes_container_name") or "",
    }


def result_to_df(results: list[dict[str, Any]], metric_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        metric = item.get("metric", {})
        value = item.get("value", [None, None])
        labels = normalize_metric_labels(metric)
        if metric_name in ("net_in", "net_out") and not labels["container"]:
            labels["container"] = "__pod__"
        rows.append(
            {
                "namespace": labels["namespace"],
                "pod": labels["pod"],
                "container": labels["container"],
                metric_name: float(value[1]) if value[1] is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)


def expand_pod_level_network_to_containers(df_metric: pd.DataFrame, container_index: pd.DataFrame) -> pd.DataFrame:
    if df_metric.empty:
        return df_metric
    pod_level = df_metric[df_metric["container"] == "__pod__"].copy()
    normal = df_metric[df_metric["container"] != "__pod__"].copy()
    if pod_level.empty or container_index.empty:
        return normal if not container_index.empty else df_metric
    expanded = pod_level.merge(
        container_index[["namespace", "pod", "container"]].drop_duplicates(),
        on=["namespace", "pod"],
        how="left",
        suffixes=("", "_real"),
    )
    expanded["container"] = expanded["container_real"].fillna("__pod__")
    expanded = expanded.drop(columns=["container_real"])
    return pd.concat([normal, expanded], ignore_index=True)


def prometheus_queries(feature_columns: list[str]) -> dict[str, str]:
    queries = {
        "cpu_util": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
            (rate(container_cpu_usage_seconds_total{job="cadvisor",image!="",container_label_io_kubernetes_pod_name!="",container_label_io_kubernetes_container_name!=""}[5m]))
        """,
        "mem_util": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
            (container_memory_working_set_bytes{job="cadvisor",image!="",container_label_io_kubernetes_pod_name!="",container_label_io_kubernetes_container_name!=""})
        """,
        "net_in": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name)
            (rate(container_network_receive_bytes_total{job="cadvisor",interface="eth0",container_label_io_kubernetes_pod_name!=""}[5m]))
        """,
        "net_out": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name)
            (rate(container_network_transmit_bytes_total{job="cadvisor",interface="eth0",container_label_io_kubernetes_pod_name!=""}[5m]))
        """,
        "disk_read": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
            (rate(container_fs_reads_bytes_total{job="cadvisor",image!="",container_label_io_kubernetes_pod_name!="",container_label_io_kubernetes_container_name!=""}[5m]))
        """,
        "disk_write": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
            (rate(container_fs_writes_bytes_total{job="cadvisor",image!="",container_label_io_kubernetes_pod_name!="",container_label_io_kubernetes_container_name!=""}[5m]))
        """,
        "mem_rss": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
            (container_memory_rss{job="cadvisor",image!="",container_label_io_kubernetes_pod_name!="",container_label_io_kubernetes_container_name!=""})
        """,
        "mem_cache": """
            sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
            (container_memory_cache{job="cadvisor",image!="",container_label_io_kubernetes_pod_name!="",container_label_io_kubernetes_container_name!=""})
        """,
    }
    return {key: value for key, value in queries.items() if key in feature_columns}


def collect_snapshot(config: PipelineConfig, logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    metric_dfs: dict[str, pd.DataFrame] = {}

    for metric_name, query in prometheus_queries(config.realtime.feature_columns).items():
        results = query_prometheus(config.realtime.prom_url, query, config.realtime.query_timeout_seconds, logger)
        metric_dfs[metric_name] = result_to_df(results, metric_name)

    container_index_parts: list[pd.DataFrame] = []
    for key in ["cpu_util", "mem_util", "disk_read", "disk_write", "mem_rss", "mem_cache"]:
        df_metric = metric_dfs.get(key, pd.DataFrame())
        if not df_metric.empty:
            container_index_parts.append(df_metric[["namespace", "pod", "container"]])

    if container_index_parts:
        container_index = pd.concat(container_index_parts, ignore_index=True).drop_duplicates()
    else:
        container_index = pd.DataFrame(columns=["namespace", "pod", "container"])

    for key in ["net_in", "net_out"]:
        if key in metric_dfs:
            metric_dfs[key] = expand_pod_level_network_to_containers(metric_dfs[key], container_index)

    dfs = [df_metric for df_metric in metric_dfs.values() if not df_metric.empty]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df_metric in dfs[1:]:
        merged = pd.merge(merged, df_metric, on=["namespace", "pod", "container"], how="outer")

    merged = merged.fillna(0.0)
    merged["timestamp"] = pd.Timestamp.utcnow().isoformat()

    final_cols = ["timestamp", "namespace", "pod", "container", *config.realtime.feature_columns]
    for column in final_cols:
        if column not in merged.columns:
            merged[column] = 0.0 if column in config.realtime.feature_columns else ""
    merged = merged[final_cols]

    if config.realtime.target_namespace:
        merged = merged[merged["namespace"] == config.realtime.target_namespace]
    if config.realtime.target_pod:
        merged = merged[merged["pod"] == config.realtime.target_pod]
    if config.realtime.target_container:
        merged = merged[merged["container"] == config.realtime.target_container]

    merged = merged[(merged[config.realtime.feature_columns].sum(axis=1) > 0)]
    return merged.sort_values(["namespace", "pod", "container"]).reset_index(drop=True)


def append_raw_snapshot(df: pd.DataFrame, output_path: str | Path) -> None:
    if df.empty:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    df.to_csv(output_path, mode="a", header=not exists, index=False)


def compute_window_anomaly_scores(
    window_rows: pd.DataFrame,
    assets: RuntimeAssets,
    ns_map: dict[str, int],
    ct_map: dict[str, int],
    top_k_features: int,
) -> WindowAnomalyScores:
    if window_rows.empty:
        raise ValueError("window_rows must contain at least one row")

    x_raw = window_rows[assets.feature_columns].to_numpy(dtype=np.float32, copy=True)
    x_raw = np.clip(x_raw, a_min=0.0, a_max=None)
    x_log = np.log1p(x_raw)
    x_scaled = assets.x_scaler.transform(x_log)

    namespace = str(window_rows.iloc[-1].get("namespace", ""))
    container = str(window_rows.iloc[-1].get("container", ""))
    context_raw = build_context_vector(namespace, container, ns_map, ct_map, assets.context_dim)
    c_scaled = assets.c_scaler.transform(context_raw)

    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=assets.device).unsqueeze(0)
    c_tensor = torch.as_tensor(c_scaled, dtype=torch.float32, device=assets.device)

    with torch.no_grad():
        reconstructed = assets.model(x_tensor, c_tensor)

    x_pred_scaled = reconstructed.detach().cpu().numpy()[0]
    errors = (x_scaled - x_pred_scaled) ** 2
    mse_per_feature = np.mean(errors, axis=0)
    feature_error_map = {feature: float(score) for feature, score in zip(assets.feature_columns, mse_per_feature.tolist())}

    ranked = sorted(feature_error_map.items(), key=lambda item: item[1], reverse=True)
    top_features = [name for name, _ in ranked[:top_k_features]]
    top_feature_scores = [score for _, score in ranked[:top_k_features]]

    return WindowAnomalyScores(
        all_score=float(np.mean(mse_per_feature)),
        top_score=float(np.mean(top_feature_scores)) if top_feature_scores else 0.0,
        feature_error_map=feature_error_map,
        ranked_feature_errors=ranked,
        top_features=top_features,
    )


def reason_from_top_features(top_features: list[str]) -> str:
    if any(feature in top_features for feature in ["mem_rss", "mem_util", "mem_cache"]):
        return "abnormal memory behavior detected"
    if "cpu_util" in top_features:
        return "abnormal CPU behavior detected"
    if any(feature in top_features for feature in ["net_in", "net_out"]):
        return "abnormal network behavior detected"
    if any(feature in top_features for feature in ["disk_read", "disk_write"]):
        return "abnormal disk I/O behavior detected"
    return "abnormal multivariate behavior detected"


def init_container_state(config: PipelineConfig) -> dict[str, Any]:
    return {
        "window_buffer": deque(),
        "all_score_history": deque(maxlen=config.realtime.all_score_history_size),
        "top_score_history": deque(maxlen=config.realtime.top_score_history_size),
        "inference_count": 0,
        "all_anomaly_active": False,
        "all_anomaly_hits": 0,
        "all_normal_hits": 0,
        "top_anomaly_active": False,
        "top_anomaly_hits": 0,
        "top_normal_hits": 0,
    }


def compute_dynamic_threshold(score_history: deque[float], base_threshold: float, multiplier: float, min_factor: float) -> float:
    history = np.asarray(score_history, dtype=np.float64)
    if len(history) <= 1:
        return float(base_threshold)
    return max(float(base_threshold) * float(min_factor), float(history.mean() + multiplier * history.std()))


def update_status(
    score: float,
    threshold: float,
    active_flag_name: str,
    hit_name: str,
    normal_name: str,
    state: dict[str, Any],
    anomaly_hits_needed: int,
    normal_hits_needed: int,
) -> tuple[bool, str, str | None]:
    is_anomaly_now = score > threshold
    if is_anomaly_now:
        state[hit_name] += 1
        state[normal_name] = 0
    else:
        state[normal_name] += 1
        state[hit_name] = 0

    status_changed = None
    if (not state[active_flag_name]) and state[hit_name] >= anomaly_hits_needed:
        state[active_flag_name] = True
        status_changed = "STARTED"
    elif state[active_flag_name] and state[normal_name] >= normal_hits_needed:
        state[active_flag_name] = False
        status_changed = "CLEARED"

    current_status = "ANOMALY_ACTIVE" if state[active_flag_name] else "NORMAL"
    return is_anomaly_now, current_status, status_changed


def format_feature_errors(ranked: list[tuple[str, float]]) -> str:
    return ", ".join(f"{name}={score:.6f}" for name, score in ranked)


def log_realtime_decision(decision: RealtimeDecision, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("TARGET            : %s", decision.target_key)
    logger.info("ALL_SCORE         : %.6f", decision.all_score)
    logger.info("ALL_THRESHOLD     : %.6f", decision.all_threshold)
    logger.info("ALL_STATUS        : %s", decision.all_status)
    logger.info("TOP_SCORE         : %.6f", decision.top_score)
    logger.info("TOP_THRESHOLD     : %.6f", decision.top_threshold)
    logger.info("TOP_FEATS         : %s", decision.top_features)
    logger.info("TOP_REASON        : %s", decision.reason)
    logger.info("TOP_STATUS        : %s", decision.top_status)
    logger.info("ALL_FEATURE_ERRORS: %s", format_feature_errors(decision.ranked_feature_errors))
    if decision.gpt_decision:
        logger.info("GPT_LABEL         : %s", decision.gpt_decision.get("label"))
        logger.info("GPT_SEVERITY      : %s", decision.gpt_decision.get("severity"))
        logger.info("GPT_EXPLANATION   : %s", decision.gpt_decision.get("explanation"))
        logger.info("GPT_ACTION        : %s", decision.gpt_decision.get("recommended_action"))
    logger.info("=" * 60)


class RealtimeMonitor:
    def __init__(self, config: PipelineConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or setup_logger("live_realtime_dual", config.realtime.log_file)
        self.assets = load_runtime_assets(config, self.logger)
        self.states = defaultdict(lambda: init_container_state(config))
        self.seen_keys: set[tuple[str, str, str]] = set()
        self.gpt_adjudicator = GPTAdjudicator(config.gpt) if config.realtime.enable_gpt_adjudication else None

    def maybe_adjudicate(self, target_key: tuple[str, str, str], scores: WindowAnomalyScores, threshold: float, window_rows: pd.DataFrame) -> dict[str, Any] | None:
        if self.gpt_adjudicator is None:
            return None
        payload = {
            "anomaly_score": scores.all_score,
            "threshold": threshold,
            "normalized_score": float(scores.all_score / max(threshold, 1e-6)),
            "top_k_anomalous_features": scores.top_features,
            "top_k_feature_errors": [score for _, score in scores.ranked_feature_errors[: self.config.realtime.top_k_features]],
            "context_metadata": {
                "namespace": target_key[0],
                "pod": target_key[1],
                "container": target_key[2],
                "start_time": str(window_rows.iloc[0].get("timestamp", "")),
                "end_time": str(window_rows.iloc[-1].get("timestamp", "")),
            },
            "recent_logs": [],
            "recent_events": [],
        }
        return self.gpt_adjudicator.adjudicate(payload)

    def process_snapshot(self, snapshot: pd.DataFrame) -> list[RealtimeDecision]:
        decisions: list[RealtimeDecision] = []
        if snapshot.empty:
            return decisions

        append_raw_snapshot(snapshot, self.config.realtime.raw_snapshot_csv)

        for _, row in snapshot.iterrows():
            key = (str(row["namespace"]), str(row["pod"]), str(row["container"]))
            self.seen_keys.add(key)

            state = self.states[key]
            state["window_buffer"].append(row.to_dict())
            if len(state["window_buffer"]) > self.assets.window_size:
                state["window_buffer"].popleft()
            if len(state["window_buffer"]) < self.assets.window_size:
                continue

            ns_map, ct_map = build_label_maps(self.seen_keys)
            window_df = pd.DataFrame(list(state["window_buffer"]))
            scores = compute_window_anomaly_scores(
                window_rows=window_df,
                assets=self.assets,
                ns_map=ns_map,
                ct_map=ct_map,
                top_k_features=self.config.realtime.top_k_features,
            )
            reason = reason_from_top_features(scores.top_features)

            state["all_score_history"].append(scores.all_score)
            state["top_score_history"].append(scores.top_score)
            state["inference_count"] += 1

            all_threshold = compute_dynamic_threshold(
                state["all_score_history"],
                self.assets.base_threshold,
                self.config.realtime.all_dynamic_threshold_std_multiplier,
                self.config.realtime.all_min_threshold_factor,
            )
            top_base_threshold = float(self.assets.base_threshold) * (self.config.realtime.top_k_features / len(self.assets.feature_columns))
            top_threshold = compute_dynamic_threshold(
                state["top_score_history"],
                top_base_threshold,
                self.config.realtime.top_dynamic_threshold_std_multiplier,
                self.config.realtime.top_min_threshold_factor,
            )

            if state["inference_count"] <= self.config.realtime.warmup_windows:
                decision = RealtimeDecision(
                    target_key=key,
                    all_score=scores.all_score,
                    all_threshold=all_threshold,
                    all_status=f"WARMUP ({state['inference_count']}/{self.config.realtime.warmup_windows})",
                    top_score=scores.top_score,
                    top_threshold=top_threshold,
                    top_status="WARMUP",
                    top_features=scores.top_features,
                    ranked_feature_errors=scores.ranked_feature_errors,
                    reason=reason,
                )
                decisions.append(decision)
                log_realtime_decision(decision, self.logger)
                continue

            _, all_status, all_changed = update_status(
                scores.all_score,
                all_threshold,
                "all_anomaly_active",
                "all_anomaly_hits",
                "all_normal_hits",
                state,
                self.config.realtime.all_anomaly_consecutive_hits,
                self.config.realtime.all_clear_consecutive_normals,
            )
            _, top_status, top_changed = update_status(
                scores.top_score,
                top_threshold,
                "top_anomaly_active",
                "top_anomaly_hits",
                "top_normal_hits",
                state,
                self.config.realtime.top_anomaly_consecutive_hits,
                self.config.realtime.top_clear_consecutive_normals,
            )

            if all_changed == "STARTED":
                all_status = "ANOMALY_STARTED"
            elif all_changed == "CLEARED":
                all_status = "ANOMALY_CLEARED"
            if top_changed == "STARTED":
                top_status = "ANOMALY_STARTED"
            elif top_changed == "CLEARED":
                top_status = "ANOMALY_CLEARED"

            gpt_decision = None
            if all_changed == "STARTED" or top_changed == "STARTED":
                gpt_decision = self.maybe_adjudicate(key, scores, all_threshold, window_df)

            decision = RealtimeDecision(
                target_key=key,
                all_score=scores.all_score,
                all_threshold=all_threshold,
                all_status=all_status,
                top_score=scores.top_score,
                top_threshold=top_threshold,
                top_status=top_status,
                top_features=scores.top_features,
                ranked_feature_errors=scores.ranked_feature_errors,
                reason=reason,
                gpt_decision=gpt_decision,
            )
            decisions.append(decision)
            log_realtime_decision(decision, self.logger)

        return decisions

    def run_forever(self) -> None:
        self.logger.info("Starting continuous dual-status anomaly detection loop...")
        while True:
            try:
                snapshot = collect_snapshot(self.config, self.logger)
                if snapshot.empty:
                    self.logger.info("No matching rows from Prometheus for current filter.")
                else:
                    self.process_snapshot(snapshot)
            except KeyboardInterrupt:
                self.logger.info("Stopped by user.")
                break
            except Exception as exc:
                self.logger.exception("Loop error: %s", exc)
            time.sleep(self.config.realtime.poll_interval_seconds)
