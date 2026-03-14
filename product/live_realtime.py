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

from src.model import FiLMAutoencoder  # adjust only if needed

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
LOG_FILE = os.path.join(CURRENT_DIR, "result.log")

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

# continuous detection tuning
WARMUP_WINDOWS = 20
ANOMALY_CONSECUTIVE_HITS = 3
CLEAR_CONSECUTIVE_NORMALS = 3
SCORE_HISTORY_SIZE = 100
DYNAMIC_THRESHOLD_STD_MULTIPLIER = 4.0
MIN_THRESHOLD_FACTOR = 1.0

# ---------------------------------------------------------
# LIVE FEATURES (must match training count/order)
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
LOGGER = logging.getLogger("live_realtime")
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

    now = datetime.now(timezone.utc).isoformat()
    merged["timestamp"] = now

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


def compute_anomaly_score(model, x_scaler, c_scaler, window_rows, ns_map, ct_map):
    x = window_rows[FEATURE_COLS].values.astype(np.float32)
    x_scaled = x_scaler.transform(x)

    namespace = str(window_rows["namespace"].iloc[-1])
    container = str(window_rows["container"].iloc[-1])

    c_dim = infer_context_dim(c_scaler)
    c_raw = build_context_vector(namespace, container, ns_map, ct_map, c_dim)
    c_scaled = c_scaler.transform(c_raw)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    c_tensor = torch.tensor(c_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        x_hat = model(x_tensor, c_tensor)

    x_true_scaled = x_tensor.squeeze(0).cpu().numpy()
    x_pred_scaled = x_hat.squeeze(0).cpu().numpy()

    sq_err = (x_true_scaled - x_pred_scaled) ** 2
    total_score = float(np.mean(sq_err))
    per_feature = sq_err.mean(axis=0)

    feature_error_map = {
        FEATURE_COLS[i]: float(per_feature[i])
        for i in range(len(FEATURE_COLS))
    }

    return total_score, feature_error_map


def top_reason(feature_error_map):
    ranked = sorted(feature_error_map.items(), key=lambda kv: kv[1], reverse=True)
    top_features = [k for k, _ in ranked[:3]]

    if "mem_rss" in top_features or "mem_util" in top_features or "mem_cache" in top_features:
        reason = "abnormal memory behavior detected"
    elif "cpu_util" in top_features:
        reason = "abnormal CPU behavior detected"
    elif "net_in" in top_features or "net_out" in top_features:
        reason = "abnormal network behavior detected"
    elif "disk_read" in top_features or "disk_write" in top_features:
        reason = "abnormal disk I/O behavior detected"
    else:
        reason = "abnormal multivariate behavior detected"

    return ranked, top_features, reason


def init_container_state():
    return {
        "window_buffer": deque(),
        "score_history": deque(maxlen=SCORE_HISTORY_SIZE),
        "inference_count": 0,
        "anomaly_active": False,
        "anomaly_hits": 0,
        "normal_hits": 0,
    }


def log_status_block(key, score, threshold, top_features, reason, status):
    LOGGER.info("=" * 60)
    LOGGER.info(f"TARGET    : {key}")
    LOGGER.info(f"SCORE     : {score:.6f}")
    LOGGER.info(f"THRESHOLD : {threshold:.6f}")
    LOGGER.info(f"TOP FEATS : {top_features}")
    LOGGER.info(f"REASON    : {reason}")
    LOGGER.info(f"STATUS    : {status}")
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

    LOGGER.info(f"Base threshold: {base_threshold}")
    LOGGER.info(f"Window size   : {window_size}")
    LOGGER.info(f"Context dim   : {c_dim}")

    model = build_model(ae_meta, len(FEATURE_COLS), c_dim)

    states = defaultdict(init_container_state)
    seen_keys = set()

    LOGGER.info("Starting continuous live anomaly detection loop...")

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

                total_score, feature_error_map = compute_anomaly_score(
                    model=model,
                    x_scaler=x_scaler,
                    c_scaler=c_scaler,
                    window_rows=window_df,
                    ns_map=ns_map,
                    ct_map=ct_map
                )

                state["score_history"].append(total_score)
                state["inference_count"] += 1

                hist = np.array(state["score_history"], dtype=np.float64)

                dynamic_threshold = base_threshold
                if len(hist) > 1:
                    dynamic_threshold = max(
                        base_threshold * MIN_THRESHOLD_FACTOR,
                        float(hist.mean() + DYNAMIC_THRESHOLD_STD_MULTIPLIER * hist.std())
                    )

                ranked, top_features, reason = top_reason(feature_error_map)

                if state["inference_count"] <= WARMUP_WINDOWS:
                    LOGGER.info("=" * 60)
                    LOGGER.info(f"TARGET    : {key}")
                    LOGGER.info(f"STATUS    : WARMUP ({state['inference_count']}/{WARMUP_WINDOWS})")
                    LOGGER.info(f"SCORE     : {total_score:.6f}")
                    LOGGER.info(f"THRESHOLD : {dynamic_threshold:.6f}")
                    LOGGER.info("=" * 60)
                    continue

                is_anomaly_now = total_score > dynamic_threshold

                if is_anomaly_now:
                    state["anomaly_hits"] += 1
                    state["normal_hits"] = 0
                else:
                    state["normal_hits"] += 1
                    state["anomaly_hits"] = 0

                if (not state["anomaly_active"]) and state["anomaly_hits"] >= ANOMALY_CONSECUTIVE_HITS:
                    state["anomaly_active"] = True
                    log_status_block(key, total_score, dynamic_threshold, top_features, reason, "ANOMALY_STARTED")
                    continue

                if state["anomaly_active"] and state["normal_hits"] >= CLEAR_CONSECUTIVE_NORMALS:
                    state["anomaly_active"] = False
                    log_status_block(key, total_score, dynamic_threshold, top_features, reason, "ANOMALY_CLEARED")
                    continue

                current_status = "ANOMALY_ACTIVE" if state["anomaly_active"] else "NORMAL"
                log_status_block(key, total_score, dynamic_threshold, top_features, reason, current_status)

        except KeyboardInterrupt:
            LOGGER.info("Stopped by user.")
            break
        except Exception as e:
            LOGGER.exception(f"Loop error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()