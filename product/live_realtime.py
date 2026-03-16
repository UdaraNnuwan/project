import os
import sys
import time
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import torch

# =========================================================
# PATHS / IMPORTS
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.model import FiLMAutoencoder


# =========================================================
# CONFIG
# =========================================================
PROM_URL = "http://35.206.92.147:9090/api/v1/query"

MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "ae_model.pt")
X_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
C_SCALER_PATH = os.path.join(MODEL_DIR, "ctx_scaler.joblib")
DETECTOR_META_PATH = os.path.join(MODEL_DIR, "detector_meta.joblib")
AE_META_PATH = os.path.join(MODEL_DIR, "ae_model_meta.joblib")

RAW_SNAPSHOT_CSV = os.path.join(CURRENT_DIR, "raw_snapshots.csv")
LOG_FILE = os.path.join(CURRENT_DIR, "result_dual_status.log")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# TARGET FILTERS
# None = monitor all
# ---------------------------------------------------------
TARGET_NAMESPACE = None
TARGET_POD = None
TARGET_CONTAINER = None

# ---------------------------------------------------------
# LOOP SETTINGS
# ---------------------------------------------------------
POLL_INTERVAL_SECONDS = 30
WINDOW_SIZE_FALLBACK = 24

# Warmup
WARMUP_WINDOWS = 20

# ALL feature anomaly logic
ALL_ANOMALY_CONSECUTIVE_HITS = 3
ALL_CLEAR_CONSECUTIVE_NORMALS = 3
ALL_SCORE_HISTORY_SIZE = 100
ALL_DYNAMIC_THRESHOLD_STD_MULTIPLIER = 4.0
ALL_MIN_THRESHOLD_FACTOR = 1.0

# TOP feature anomaly logic
TOPK = 3
TOP_ANOMALY_CONSECUTIVE_HITS = 2
TOP_CLEAR_CONSECUTIVE_NORMALS = 2
TOP_SCORE_HISTORY_SIZE = 100
TOP_DYNAMIC_THRESHOLD_STD_MULTIPLIER = 4.0
TOP_MIN_THRESHOLD_FACTOR = 1.0

# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------
FEATURE_COLS = [
    "cpu_util",
    "mem_util",
    "net_in",
    "net_out",
    "disk_read",
    "disk_write",
    "mem_rss",
    "mem_cache",
]

# =========================================================
# LOGGING
# =========================================================
LOGGER = logging.getLogger("live_realtime_dual")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
LOGGER.addHandler(fh)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
LOGGER.addHandler(sh)

# =========================================================
# PROMETHEUS QUERIES
# =========================================================
QUERIES = {
    "cpu_util": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            rate(container_cpu_usage_seconds_total{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }[5m])
        )
    """,
    "mem_util": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            container_memory_working_set_bytes{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }
        )
    """,
    "net_in": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name
        ) (
            rate(container_network_receive_bytes_total{
                job="cadvisor",
                interface="eth0",
                container_label_io_kubernetes_pod_name!=""
            }[5m])
        )
    """,
    "net_out": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name
        ) (
            rate(container_network_transmit_bytes_total{
                job="cadvisor",
                interface="eth0",
                container_label_io_kubernetes_pod_name!=""
            }[5m])
        )
    """,
    "disk_read": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            rate(container_fs_reads_bytes_total{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }[5m])
        )
    """,
    "disk_write": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            rate(container_fs_writes_bytes_total{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }[5m])
        )
    """,
    "mem_rss": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            container_memory_rss{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }
        )
    """,
    "mem_cache": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            container_memory_cache{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }
        )
    """,
}

# =========================================================
# HELPERS
# =========================================================
def safe_load_joblib(path, default=None):
    if os.path.exists(path):
        return joblib.load(path)
    return default


def query_prometheus(query: str):
    try:
        response = requests.get(PROM_URL, params={"query": query}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload["data"]["result"]
    except Exception as e:
        LOGGER.error(f"Prometheus query failed: {e}")
        return []


def normalize_metric_labels(metric: dict):
    namespace = (
        metric.get("namespace")
        or metric.get("container_label_io_kubernetes_pod_namespace")
        or ""
    )
    pod = (
        metric.get("pod")
        or metric.get("container_label_io_kubernetes_pod_name")
        or ""
    )
    container = (
        metric.get("container")
        or metric.get("container_label_io_kubernetes_container_name")
        or ""
    )

    return {
        "namespace": namespace,
        "pod": pod,
        "container": container,
    }


def result_to_df(results, metric_name: str) -> pd.DataFrame:
    rows = []

    for item in results:
        metric = item.get("metric", {})
        value = item.get("value", [None, None])
        labels = normalize_metric_labels(metric)

        if metric_name in ("net_in", "net_out") and not labels["container"]:
            labels["container"] = "__pod__"

        rows.append({
            "namespace": labels["namespace"],
            "pod": labels["pod"],
            "container": labels["container"],
            metric_name: float(value[1]) if value[1] is not None else 0.0
        })

    return pd.DataFrame(rows)


def expand_pod_level_network_to_containers(df_metric: pd.DataFrame, container_index: pd.DataFrame) -> pd.DataFrame:
    if df_metric.empty:
        return df_metric

    pod_level = df_metric[df_metric["container"] == "__pod__"].copy()
    normal = df_metric[df_metric["container"] != "__pod__"].copy()

    if pod_level.empty:
        return df_metric

    if container_index.empty:
        return normal

    expanded = pod_level.merge(
        container_index[["namespace", "pod", "container"]].drop_duplicates(),
        on=["namespace", "pod"],
        how="left",
        suffixes=("", "_real")
    )
    expanded["container"] = expanded["container_real"].fillna("__pod__")
    expanded = expanded.drop(columns=["container_real"])

    return pd.concat([normal, expanded], ignore_index=True)


def collect_snapshot() -> pd.DataFrame:
    metric_dfs = {}

    for metric_name, query in QUERIES.items():
        results = query_prometheus(query)
        metric_dfs[metric_name] = result_to_df(results, metric_name)

    container_index_parts = []
    for key in ["cpu_util", "mem_util", "disk_read", "disk_write", "mem_rss", "mem_cache"]:
        df = metric_dfs.get(key, pd.DataFrame())
        if not df.empty:
            container_index_parts.append(df[["namespace", "pod", "container"]])

    if container_index_parts:
        container_index = pd.concat(container_index_parts, ignore_index=True).drop_duplicates()
    else:
        container_index = pd.DataFrame(columns=["namespace", "pod", "container"])

    for key in ["net_in", "net_out"]:
        metric_dfs[key] = expand_pod_level_network_to_containers(metric_dfs[key], container_index)

    dfs = [df for df in metric_dfs.values() if not df.empty]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(
            merged,
            df,
            on=["namespace", "pod", "container"],
            how="outer"
        )

    merged = merged.fillna(0.0)

    merged["timestamp"] = datetime.now(timezone.utc).isoformat()

    final_cols = ["timestamp", "namespace", "pod", "container"] + FEATURE_COLS
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = 0.0 if c not in ("timestamp", "namespace", "pod", "container") else ""

    merged = merged[final_cols]

    if TARGET_NAMESPACE:
        merged = merged[merged["namespace"] == TARGET_NAMESPACE]
    if TARGET_POD:
        merged = merged[merged["pod"] == TARGET_POD]
    if TARGET_CONTAINER:
        merged = merged[merged["container"] == TARGET_CONTAINER]

    merged = merged[(merged[FEATURE_COLS].sum(axis=1) > 0)]

    return merged.sort_values(["namespace", "pod", "container"]).reset_index(drop=True)


def append_raw_snapshot(df: pd.DataFrame):
    if df.empty:
        return
    exists = os.path.exists(RAW_SNAPSHOT_CSV)
    df.to_csv(RAW_SNAPSHOT_CSV, mode="a", header=not exists, index=False)


def extract_threshold(detector_meta) -> float:
    if isinstance(detector_meta, dict):
        for key in ["threshold", "thresh", "best_threshold", "anomaly_threshold"]:
            if key in detector_meta:
                return float(detector_meta[key])
    return 1.0


def extract_window_size(ae_meta, detector_meta) -> int:
    for meta in [ae_meta, detector_meta]:
        if isinstance(meta, dict):
            for key in ["window_size", "seq_len", "sequence_length", "lookback"]:
                if key in meta:
                    return int(meta[key])
    return WINDOW_SIZE_FALLBACK


def build_label_maps(seen_keys):
    namespaces = sorted(list({k[0] for k in seen_keys}))
    containers = sorted(list({k[2] for k in seen_keys}))

    ns_map = {v: i for i, v in enumerate(namespaces)}
    ct_map = {v: i for i, v in enumerate(containers)}
    return ns_map, ct_map


def build_context_vector(namespace, container, ns_map, ct_map, c_dim):
    base = [
        float(ns_map.get(namespace, -1)),
        float(ct_map.get(container, -1)),
    ]
    vec = np.array(base, dtype=np.float32)

    if len(vec) < c_dim:
        vec = np.pad(vec, (0, c_dim - len(vec)), mode="constant", constant_values=0.0)
    elif len(vec) > c_dim:
        vec = vec[:c_dim]

    return vec.reshape(1, -1)


def infer_context_dim(c_scaler):
    if hasattr(c_scaler, "n_features_in_"):
        return int(c_scaler.n_features_in_)
    return 2


def build_model(ae_meta, x_dim, c_dim):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        window_size = int(checkpoint.get("window_size", extract_window_size(ae_meta, {})))
        n_features = int(checkpoint.get("n_features", x_dim))
        context_dim = int(checkpoint.get("context_dim", c_dim))
        units = int(checkpoint.get("units", 64))
        latent = int(checkpoint.get("latent", 16))
    else:
        state_dict = checkpoint
        window_size = extract_window_size(ae_meta, {})
        n_features = x_dim
        context_dim = c_dim
        units = 64
        latent = 16

    LOGGER.info(
        f"Checkpoint params | window_size={window_size}, "
        f"n_features={n_features}, context_dim={context_dim}, "
        f"units={units}, latent={latent}"
    )

    model = FiLMAutoencoder(
        window_size=window_size,
        n_features=n_features,
        context_dim=context_dim,
        units=units,
        latent=latent,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def compute_anomaly_score(
    window_rows: pd.DataFrame,
    x_scaler,
    c_scaler,
    model,
    ns_map,
    ct_map,
):
    if window_rows.empty:
        raise ValueError("window_rows must contain at least one row")

    x_raw = window_rows[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    x_raw = np.clip(x_raw, a_min=0.0, a_max=None)

    x_log = np.log1p(x_raw)
    x_scaled = x_scaler.transform(x_log)

    c_dim = infer_context_dim(c_scaler)
    namespace = str(window_rows.iloc[-1].get("namespace", ""))
    container = str(window_rows.iloc[-1].get("container", ""))

    context_raw = build_context_vector(
        namespace=namespace,
        container=container,
        ns_map=ns_map,
        ct_map=ct_map,
        c_dim=c_dim,
    )
    c_scaled = c_scaler.transform(context_raw)

    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    c_tensor = torch.as_tensor(c_scaled, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        reconstructed = model(x_tensor, c_tensor)

    x_pred_scaled = reconstructed.detach().cpu().numpy()[0]

    errors = (x_scaled - x_pred_scaled) ** 2
    mse_per_feature = np.mean(errors, axis=0)

    feature_error_map = {
        feature: float(score)
        for feature, score in zip(FEATURE_COLS, mse_per_feature.tolist())
    }

    # all feature score
    all_score = float(np.mean(mse_per_feature))

    # top-k feature score
    ranked = sorted(feature_error_map.items(), key=lambda kv: kv[1], reverse=True)
    top_features = [k for k, _ in ranked[:TOPK]]
    top_feature_scores = [v for _, v in ranked[:TOPK]]
    top_score = float(np.mean(top_feature_scores)) if top_feature_scores else 0.0

    return all_score, top_score, feature_error_map, ranked, top_features


def reason_from_top_features(top_features):
    if "mem_rss" in top_features or "mem_util" in top_features or "mem_cache" in top_features:
        return "abnormal memory behavior detected"
    if "cpu_util" in top_features:
        return "abnormal CPU behavior detected"
    if "net_in" in top_features or "net_out" in top_features:
        return "abnormal network behavior detected"
    if "disk_read" in top_features or "disk_write" in top_features:
        return "abnormal disk I/O behavior detected"
    return "abnormal multivariate behavior detected"


def init_container_state():
    return {
        "window_buffer": deque(),

        "all_score_history": deque(maxlen=ALL_SCORE_HISTORY_SIZE),
        "top_score_history": deque(maxlen=TOP_SCORE_HISTORY_SIZE),

        "inference_count": 0,

        "all_anomaly_active": False,
        "all_anomaly_hits": 0,
        "all_normal_hits": 0,

        "top_anomaly_active": False,
        "top_anomaly_hits": 0,
        "top_normal_hits": 0,
    }


def compute_dynamic_threshold(score_history, base_threshold, multiplier, min_factor):
    hist = np.array(score_history, dtype=np.float64)
    if len(hist) <= 1:
        return float(base_threshold)

    return max(
        float(base_threshold) * float(min_factor),
        float(hist.mean() + multiplier * hist.std())
    )


def format_all_feature_errors(ranked):
    return ", ".join([f"{k}={v:.6f}" for k, v in ranked])


def update_status(
    score,
    threshold,
    active_flag_name,
    hit_name,
    normal_name,
    state,
    anomaly_hits_needed,
    normal_hits_needed,
):
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


def log_dual_status_block(
    key,
    all_score,
    all_threshold,
    all_status,
    top_score,
    top_threshold,
    top_status,
    top_features,
    reason,
    ranked,
):
    LOGGER.info("=" * 60)
    LOGGER.info(f"TARGET            : {key}")

    LOGGER.info(f"ALL_SCORE         : {all_score:.6f}")
    LOGGER.info(f"ALL_THRESHOLD     : {all_threshold:.6f}")
    LOGGER.info(f"ALL_STATUS        : {all_status}")

    LOGGER.info(f"TOP_SCORE         : {top_score:.6f}")
    LOGGER.info(f"TOP_THRESHOLD     : {top_threshold:.6f}")
    LOGGER.info(f"TOP_FEATS         : {top_features}")
    LOGGER.info(f"TOP_REASON        : {reason}")
    LOGGER.info(f"TOP_STATUS        : {top_status}")

    LOGGER.info(f"ALL_FEATURE_ERRORS: {format_all_feature_errors(ranked)}")
    LOGGER.info("=" * 60)


# =========================================================
# MAIN
# =========================================================
def main():
    LOGGER.info("Loading model artifacts...")

    ae_meta = safe_load_joblib(AE_META_PATH, default={})
    detector_meta = safe_load_joblib(DETECTOR_META_PATH, default={})
    x_scaler = joblib.load(X_SCALER_PATH)
    c_scaler = joblib.load(C_SCALER_PATH)

    base_threshold = extract_threshold(detector_meta)
    window_size = extract_window_size(ae_meta, detector_meta)
    c_dim = infer_context_dim(c_scaler)

    LOGGER.info(f"Base threshold : {base_threshold}")
    LOGGER.info(f"Window size    : {window_size}")
    LOGGER.info(f"Context dim    : {c_dim}")

    model = build_model(ae_meta, len(FEATURE_COLS), c_dim)

    states = defaultdict(init_container_state)
    seen_keys = set()

    LOGGER.info("Starting continuous dual-status anomaly detection loop...")

    while True:
        try:
            snapshot = collect_snapshot()

            if snapshot.empty:
                LOGGER.info("No matching rows from Prometheus for current filter.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            append_raw_snapshot(snapshot)

            for _, row in snapshot.iterrows():
                key = (row["namespace"], row["pod"], row["container"])
                seen_keys.add(key)

                state = states[key]
                state["window_buffer"].append(row.to_dict())

                if len(state["window_buffer"]) > window_size:
                    state["window_buffer"].popleft()

                if len(state["window_buffer"]) < window_size:
                    continue

                ns_map, ct_map = build_label_maps(seen_keys)
                window_df = pd.DataFrame(list(state["window_buffer"]))

                all_score, top_score, feature_error_map, ranked, top_features = compute_anomaly_score(
                    window_rows=window_df,
                    x_scaler=x_scaler,
                    c_scaler=c_scaler,
                    model=model,
                    ns_map=ns_map,
                    ct_map=ct_map,
                )

                reason = reason_from_top_features(top_features)

                state["all_score_history"].append(all_score)
                state["top_score_history"].append(top_score)
                state["inference_count"] += 1

                all_threshold = compute_dynamic_threshold(
                    state["all_score_history"],
                    base_threshold=base_threshold,
                    multiplier=ALL_DYNAMIC_THRESHOLD_STD_MULTIPLIER,
                    min_factor=ALL_MIN_THRESHOLD_FACTOR,
                )

                # top threshold derived from current top score history
                # use base threshold scaled down to top-k proportion
                top_base_threshold = float(base_threshold) * (TOPK / len(FEATURE_COLS))
                top_threshold = compute_dynamic_threshold(
                    state["top_score_history"],
                    base_threshold=top_base_threshold,
                    multiplier=TOP_DYNAMIC_THRESHOLD_STD_MULTIPLIER,
                    min_factor=TOP_MIN_THRESHOLD_FACTOR,
                )

                if state["inference_count"] <= WARMUP_WINDOWS:
                    LOGGER.info("=" * 60)
                    LOGGER.info(f"TARGET            : {key}")
                    LOGGER.info(f"STATUS            : WARMUP ({state['inference_count']}/{WARMUP_WINDOWS})")
                    LOGGER.info(f"ALL_SCORE         : {all_score:.6f}")
                    LOGGER.info(f"ALL_THRESHOLD     : {all_threshold:.6f}")
                    LOGGER.info(f"TOP_SCORE         : {top_score:.6f}")
                    LOGGER.info(f"TOP_THRESHOLD     : {top_threshold:.6f}")
                    LOGGER.info(f"TOP_FEATS         : {top_features}")
                    LOGGER.info(f"TOP_REASON        : {reason}")
                    LOGGER.info(f"ALL_FEATURE_ERRORS: {format_all_feature_errors(ranked)}")
                    LOGGER.info("=" * 60)
                    continue

                _, all_status, all_changed = update_status(
                    score=all_score,
                    threshold=all_threshold,
                    active_flag_name="all_anomaly_active",
                    hit_name="all_anomaly_hits",
                    normal_name="all_normal_hits",
                    state=state,
                    anomaly_hits_needed=ALL_ANOMALY_CONSECUTIVE_HITS,
                    normal_hits_needed=ALL_CLEAR_CONSECUTIVE_NORMALS,
                )

                _, top_status, top_changed = update_status(
                    score=top_score,
                    threshold=top_threshold,
                    active_flag_name="top_anomaly_active",
                    hit_name="top_anomaly_hits",
                    normal_name="top_normal_hits",
                    state=state,
                    anomaly_hits_needed=TOP_ANOMALY_CONSECUTIVE_HITS,
                    normal_hits_needed=TOP_CLEAR_CONSECUTIVE_NORMALS,
                )

                if all_changed == "STARTED":
                    all_status = "ANOMALY_STARTED"
                elif all_changed == "CLEARED":
                    all_status = "ANOMALY_CLEARED"

                if top_changed == "STARTED":
                    top_status = "ANOMALY_STARTED"
                elif top_changed == "CLEARED":
                    top_status = "ANOMALY_CLEARED"

                log_dual_status_block(
                    key=key,
                    all_score=all_score,
                    all_threshold=all_threshold,
                    all_status=all_status,
                    top_score=top_score,
                    top_threshold=top_threshold,
                    top_status=top_status,
                    top_features=top_features,
                    reason=reason,
                    ranked=ranked,
                )

        except KeyboardInterrupt:
            LOGGER.info("Stopped by user.")
            break
        except Exception as e:
            LOGGER.exception(f"Loop error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()