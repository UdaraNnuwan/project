import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from functools import reduce

import joblib
import numpy as np
import pandas as pd
import requests
import torch

# =========================================================
# PATH SETUP
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.model import FiLMAutoencoder  # noqa: E402

# =========================================================
# CONFIGURATION
# =========================================================
PROM_URL = "http://35.206.92.147:9090/api/v1/query"
LOG_PATH = os.path.join(CURRENT_DIR, "result.log")
RAW_DATA_LOG_PATH = os.path.join(CURRENT_DIR, "raw_snapshots.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "ae_model.pt")
X_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
C_SCALER_PATH = os.path.join(MODEL_DIR, "ctx_scaler.joblib")
DETECTOR_META_PATH = os.path.join(MODEL_DIR, "detector_meta.joblib")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Poll interval
COLLECT_INTERVAL_SECONDS = 5
STALENESS_THRESHOLD_SECONDS = 10
ALERT_COOLDOWN_SECONDS = 10
ENABLE_DYNAMIC_THRESHOLD = True
THRESHOLD_HISTORY_SIZE = 120
MIN_HISTORY_FOR_DYNAMIC_THRESHOLD = 20
DYNAMIC_THRESHOLD_STD_MULTIPLIER = 3.0

# Target selector
TARGET_NAMESPACE = "default"
TARGET_POD = "chcekone"      # pod name contains this text
TARGET_CONTAINER = None

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

FINAL_COLUMNS = [
    "timestamp",
    "namespace",
    "pod",
    "container",
    "cpu_util",
    "mem_util",
    "net_in",
    "net_out",
    "disk_read",
    "disk_write",
    "mem_rss",
    "mem_cache",
]

SKIP_ALL_ZERO_ROWS = True

# =========================================================
# PROMETHEUS QUERIES
# =========================================================
QUERIES = {
    "cpu_util": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
        (rate(container_cpu_usage_seconds_total{image!="", container_label_io_kubernetes_container_name!=""}[5m]))
    """,
    "mem_util": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
        (container_memory_working_set_bytes{image!="", container_label_io_kubernetes_container_name!=""})
    """,
    "net_in": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name)
        (rate(container_network_receive_bytes_total{image!="", interface="eth0"}[5m]))
    """,
    "net_out": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name)
        (rate(container_network_transmit_bytes_total{image!="", interface="eth0"}[5m]))
    """,
    "disk_read": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
        (rate(container_fs_reads_bytes_total{image!="", container_label_io_kubernetes_container_name!=""}[5m]))
    """,
    "disk_write": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
        (rate(container_fs_writes_bytes_total{image!="", container_label_io_kubernetes_container_name!=""}[5m]))
    """,
    "mem_rss": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
        (container_memory_rss{image!="", container_label_io_kubernetes_container_name!=""})
    """,
    "mem_cache": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name)
        (container_memory_cache{image!="", container_label_io_kubernetes_container_name!=""})
    """,
}

# =========================================================
# LOGGING
# =========================================================
def setup_logger():
    logger = logging.getLogger("realtime_film_monitor")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logger()

# =========================================================
# MODEL / SCALER LOAD
# =========================================================
def load_assets():
    LOGGER.info("[INFO] Loading model, scalers, and metadata...")

    x_scaler = joblib.load(X_SCALER_PATH)
    c_scaler = joblib.load(C_SCALER_PATH)
    det_meta = joblib.load(DETECTOR_META_PATH)

    threshold = det_meta["threshold"]
    window_size = det_meta["window_size"]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = FiLMAutoencoder(
        window_size=window_size,
        n_features=len(FEATURE_COLS),
        context_dim=c_scaler.n_features_in_,
    ).to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return x_scaler, c_scaler, threshold, window_size, model

# =========================================================
# PROMETHEUS COLLECTION
# =========================================================
def query_prometheus(query: str):
    try:
        response = requests.get(PROM_URL, params={"query": query}, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "success":
            return []

        return payload["data"]["result"]
    except Exception as e:
        LOGGER.error(f"[ERROR] Query failed: {e}")
        return []


def normalize_labels(metric: dict) -> dict:
    return {
        "namespace": metric.get("container_label_io_kubernetes_pod_namespace", ""),
        "pod": metric.get("container_label_io_kubernetes_pod_name", ""),
        "container": metric.get("container_label_io_kubernetes_container_name", ""),
    }


def collect_all_metrics_once() -> pd.DataFrame:
    metric_dfs = {}

    for name, query in QUERIES.items():
        results = query_prometheus(query)
        rows = []

        for item in results:
            labels = normalize_labels(item["metric"])

            # Network metrics have no container label in some cases
            if name in ["net_in", "net_out"] and not labels["container"]:
                labels["container"] = "__pod__"

            rows.append({
                "namespace": labels["namespace"],
                "pod": labels["pod"],
                "container": labels["container"],
                name: float(item["value"][1]),
            })

        metric_dfs[name] = pd.DataFrame(rows)

    # Build container index from non-network metrics
    container_parts = [
        df for key, df in metric_dfs.items()
        if key not in ["net_in", "net_out"] and not df.empty
    ]

    if container_parts:
        container_index = (
            pd.concat(container_parts, ignore_index=True)[["namespace", "pod", "container"]]
            .drop_duplicates()
        )
    else:
        container_index = pd.DataFrame(columns=["namespace", "pod", "container"])

    # Spread pod-level network metrics to container level
    for net_key in ["net_in", "net_out"]:
        df = metric_dfs[net_key]
        if not df.empty and not container_index.empty:
            pod_level = df[df["container"] == "__pod__"]
            if not pod_level.empty:
                merged = pod_level.merge(
                    container_index,
                    on=["namespace", "pod"],
                    suffixes=("_old", "")
                )
                metric_dfs[net_key] = merged[["namespace", "pod", "container", net_key]]

    dfs = [df for df in metric_dfs.values() if not df.empty]
    if not dfs:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    final_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["namespace", "pod", "container"], how="outer"
        ),
        dfs
    )

    final_df = final_df.fillna(0.0)
    final_df["timestamp"] = datetime.now(timezone.utc).isoformat()

    for col in FINAL_COLUMNS:
        if col not in final_df.columns:
            if col in ["timestamp", "namespace", "pod", "container"]:
                final_df[col] = ""
            else:
                final_df[col] = 0.0

    final_df = final_df[FINAL_COLUMNS]

    if SKIP_ALL_ZERO_ROWS:
        final_df = final_df[final_df[FEATURE_COLS].sum(axis=1) > 0]

    return final_df

# =========================================================
# INFERENCE HELPERS
# =========================================================
def compute_anomaly_score(x_window, x_scaler, c_scaler, model):
    x_raw = np.asarray(x_window, dtype=np.float32)
    x_log = np.log1p(np.clip(x_raw, a_min=0, a_max=None))
    x_scaled = x_scaler.transform(x_log)

    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).to(DEVICE)
    c_scaled = np.zeros((1, c_scaler.n_features_in_), dtype=np.float32)
    c_tensor = torch.FloatTensor(c_scaled).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(x_tensor, c_tensor)

    x_pred_scaled = reconstructed.cpu().numpy().squeeze()

    errors = (x_scaled - x_pred_scaled) ** 2
    mse_per_feature = np.mean(errors, axis=0)
    total_score = float(np.mean(mse_per_feature))

    return total_score, mse_per_feature


def compute_dynamic_threshold(score_history, static_threshold):
    if not ENABLE_DYNAMIC_THRESHOLD or len(score_history) < MIN_HISTORY_FOR_DYNAMIC_THRESHOLD:
        return float(static_threshold)

    history = np.asarray(score_history, dtype=np.float32)
    adaptive_threshold = float(
        np.mean(history) + DYNAMIC_THRESHOLD_STD_MULTIPLIER * np.std(history)
    )
    return max(float(static_threshold), adaptive_threshold)


def append_raw_snapshots(raw_df):
    raw_df = raw_df[FINAL_COLUMNS].copy()
    raw_df.to_csv(
        RAW_DATA_LOG_PATH,
        mode="a",
        header=not os.path.exists(RAW_DATA_LOG_PATH),
        index=False,
    )


def init_container_state(window_size):
    return {
        "window_buffer": deque(maxlen=window_size),
        "score_history": deque(maxlen=THRESHOLD_HISTORY_SIZE),
        "last_processed_timestamp": None,
        "last_alert_time": 0.0,
        "anomaly_active": False,
    }


def print_top_metrics(mse_per_feature, latest_row):
    LOGGER.info("-" * 50)
    LOGGER.info("REASON (TOP CONTRIBUTING METRICS):")

    top_indices = np.argsort(mse_per_feature)[::-1][:3]
    for idx in top_indices:
        f_name = FEATURE_COLS[idx]
        f_error = mse_per_feature[idx]
        actual_val = latest_row[f_name]
        LOGGER.info(
            f" -> {f_name.upper()}: Error={f_error:.6f} "
            f"(Actual Val={actual_val:.2f})"
        )


def is_stale(ts_str: str, threshold_seconds: int) -> bool:
    try:
        ts = pd.to_datetime(ts_str, utc=True)
        now = datetime.now(timezone.utc)
        age = (now - ts.to_pydatetime()).total_seconds()
        return age > threshold_seconds
    except Exception:
        return False

# =========================================================
# MAIN
# =========================================================
def main():
    x_scaler, c_scaler, static_threshold, window_size, model = load_assets()

    LOGGER.info("")
    LOGGER.info("[INFO] Starting unified real-time monitor...")
    LOGGER.info(f"[INFO] Target namespace : {TARGET_NAMESPACE}")
    LOGGER.info(f"[INFO] Target pod match : {TARGET_POD}")
    LOGGER.info(f"[INFO] Target container : {TARGET_CONTAINER or 'ALL'}")
    LOGGER.info(f"[INFO] Window size      : {window_size}")
    LOGGER.info(f"[INFO] Static threshold : {static_threshold:.6f}")
    LOGGER.info(f"[INFO] Dynamic threshold: {'ON' if ENABLE_DYNAMIC_THRESHOLD else 'OFF'}")
    LOGGER.info(f"[INFO] Raw data file    : {RAW_DATA_LOG_PATH}")
    LOGGER.info("")

    container_states = {}

    while True:
        try:
            df = collect_all_metrics_once()

            if df.empty:
                LOGGER.warning("[WARN] No data returned from Prometheus.")
                time.sleep(COLLECT_INTERVAL_SECONDS)
                continue

            mask = (
                (df["namespace"] == TARGET_NAMESPACE)
                & (df["pod"].astype(str).str.contains(TARGET_POD, na=False))
            )
            target_df = df[mask].copy()

            if TARGET_CONTAINER:
                target_df = target_df[target_df["container"] == TARGET_CONTAINER].copy()

            if target_df.empty:
                LOGGER.warning("[WARN] Target pod/container not found in current snapshot.")
                time.sleep(COLLECT_INTERVAL_SECONDS)
                continue

            processed_rows = []

            for _, latest_row in target_df.sort_values(["pod", "container"]).iterrows():
                container_key = (
                    latest_row["namespace"],
                    latest_row["pod"],
                    latest_row["container"],
                )
                state = container_states.setdefault(
                    container_key,
                    init_container_state(window_size),
                )
                latest_ts = latest_row["timestamp"]

                if latest_ts == state["last_processed_timestamp"]:
                    continue

                state["last_processed_timestamp"] = latest_ts

                if is_stale(latest_ts, STALENESS_THRESHOLD_SECONDS):
                    LOGGER.warning(
                        f"[WARN] Data is stale | pod={latest_row['pod']} | "
                        f"container={latest_row['container']} | timestamp={latest_ts}"
                    )
                    continue

                feature_values = latest_row[FEATURE_COLS].astype(float).values
                state["window_buffer"].append(feature_values)
                processed_rows.append(latest_row[FINAL_COLUMNS].to_dict())

                LOGGER.info(
                    f"[INFO] Snapshot received | pod={latest_row['pod']} | "
                    f"container={latest_row['container']} | "
                    f"buffer={len(state['window_buffer'])}/{window_size}"
                )

                if len(state["window_buffer"]) < window_size:
                    LOGGER.info(
                        f"[INFO] Waiting for enough data to fill the window for "
                        f"container={latest_row['container']}..."
                    )
                    continue

                total_score, mse_per_feature = compute_anomaly_score(
                    x_window=np.array(state["window_buffer"]),
                    x_scaler=x_scaler,
                    c_scaler=c_scaler,
                    model=model,
                )
                active_threshold = compute_dynamic_threshold(
                    state["score_history"],
                    static_threshold,
                )

                LOGGER.info("")
                LOGGER.info("=" * 60)
                LOGGER.info(f"TIMESTAMP : {latest_ts}")
                LOGGER.info(f"POD       : {latest_row['pod']}")
                LOGGER.info(f"CONTAINER : {latest_row['container']}")
                LOGGER.info(f"SCORE     : {total_score:.6f}")
                LOGGER.info(f"THRESHOLD : {active_threshold:.6f}")
                LOGGER.info(f"BASELINE  : {static_threshold:.6f}")
                LOGGER.info(
                    f"HISTORY   : {len(state['score_history'])}/{THRESHOLD_HISTORY_SIZE}"
                )

                if total_score > active_threshold:
                    now = time.time()

                    if not state["anomaly_active"]:
                        LOGGER.info("STATUS    : ANOMALY STARTED")
                        state["anomaly_active"] = True
                        state["last_alert_time"] = now
                        print_top_metrics(mse_per_feature, latest_row)

                    else:
                        if now - state["last_alert_time"] >= ALERT_COOLDOWN_SECONDS:
                            LOGGER.info("STATUS    : ANOMALY STILL ACTIVE")
                            state["last_alert_time"] = now
                            print_top_metrics(mse_per_feature, latest_row)
                        else:
                            LOGGER.info("STATUS    : ANOMALY ACTIVE (cooldown)")

                    LOGGER.info("-" * 50)
                    LOGGER.info("LATEST SNAPSHOT:")
                    LOGGER.info("\n%s", latest_row[FEATURE_COLS].to_frame().T.to_string(index=False))

                else:
                    state["score_history"].append(total_score)
                    if state["anomaly_active"]:
                        LOGGER.info("STATUS    : ANOMALY CLEARED")
                        state["anomaly_active"] = False
                    else:
                        LOGGER.info("STATUS    : NORMAL")

                LOGGER.info("=" * 60)

            if processed_rows:
                append_raw_snapshots(pd.DataFrame(processed_rows, columns=FINAL_COLUMNS))
            else:
                LOGGER.info("[INFO] No new data. Skipping inference.")

        except KeyboardInterrupt:
            LOGGER.info("[INFO] Stopped by user.")
            break
        except Exception as e:
            LOGGER.exception(f"[ERROR] Main loop failed: {e}")

        time.sleep(COLLECT_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
