import os
import sys
import time
import logging

# Force real-time log flushing for production live environment
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from collections import deque, defaultdict
from datetime import datetime, timezone

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

from alibaba_trace.model_architecture import DualHeadBiLSTMFiLM
from alibaba_trace.data.scaling import StreamingMinMaxScaler
from alibaba_trace.data.dataset import _encode_metadata, META_COLS
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
from alibaba_trace.incident.pipeline import IncidentPipeline, SynchronousAlertDispatcher
from alibaba_trace.incident.rca_genai import GenAIRCAEngine
from alibaba_trace.incident.models import ContainerContext

# ── Notification / AI helpers (imported once at module level) ──────────────
from alibaba_trace.utils.notifications import send_telegram_alert
from alibaba_trace.utils.ai_helper import get_gpt_explanation


# =========================================================
# CONFIG
# =========================================================
PROM_URL = "http://35.206.92.147:9090/api/v1/query"

MODEL_DIR = os.path.join(PROJECT_ROOT, "alibaba_trace", "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "dual_head_model.pt")
X_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.json")
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
POLL_INTERVAL_SECONDS = 10
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

# ---------------------------------------------------------
# ANSI terminal colors (no external libs)
# ---------------------------------------------------------
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def terminal_log(key, cpu, mem, net, all_score, top_score, all_status, top_status, warmup_info=None):
    """Emit a single colored status line to stdout."""
    ts   = datetime.now().strftime("%H:%M:%S")
    ns, pod, container = key
    label = f"{ns}/{pod}/{container}"

    if warmup_info:
        tag   = f"{_YELLOW}⏳ WARMUP {warmup_info}{_RESET}"
    elif "ANOMALY" in all_status or "ANOMALY" in top_status:
        tag   = f"{_RED}🔴 ANOMALY{_RESET}"
    else:
        tag   = f"{_GREEN}🟢 NORMAL {_RESET}"

    line = (
        f"[{ts}] {tag} {label:<60} "
        f"| CPU={cpu:>8.4f} MEM={mem:>10.0f} NET={net:>10.0f} "
        f"ALL={all_score:.6f} TOP={top_score:.6f}"
    )
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

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


def build_label_maps(seen_keys):
    namespaces = sorted(list({k[0] for k in seen_keys}))
    containers = sorted(list({k[2] for k in seen_keys}))

    ns_map = {v: i for i, v in enumerate(namespaces)}
    ct_map = {v: i for i, v in enumerate(containers)}
    return ns_map, ct_map


def build_model(x_dim, c_dim):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    window_size = 50
    units = 128
    latent = 64

    model = DualHeadBiLSTMFiLM(
        window_size=window_size,
        n_ts_features=x_dim,
        n_meta_features=c_dim,
        latent_dim=latent,
        lstm_units=(units, max(32, units // 2)),
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model, window_size


def compute_anomaly_score(
    window_rows: pd.DataFrame,
    x_scaler,
    model,
):
    if window_rows.empty:
        raise ValueError("window_rows must contain at least one row")

    alibaba_cols = ["cpu_util_percent", "mem_util_percent", "cpu_request", "mem_request", "net_in", "net_out", "disk_io_percent"]
    df = window_rows.copy()
    df["cpu_util_percent"] = df["cpu_util"]
    df["mem_util_percent"] = df["mem_util"]
    df["cpu_request"] = df["cpu_util"] # dummy
    df["mem_request"] = df["mem_util"] # dummy
    df["net_in"] = df["net_in"]
    df["net_out"] = df["net_out"]
    df["disk_io_percent"] = df.get("disk_read", 0) + df.get("disk_write", 0)
    
    x_raw = df[alibaba_cols].to_numpy(dtype=np.float32)
    min_ = np.asarray(x_scaler.min_, dtype=np.float32)
    max_ = np.asarray(x_scaler.max_, dtype=np.float32)
    range_ = (max_ - min_) + 1e-8
    x_scaled = (x_raw - min_) / range_
    x_scaled = np.clip(x_scaled, 0.0, 1.0)

    # Encode metadata
    df["container_id"] = df["container"]
    df["machine_id"] = df["pod"]
    c_scaled = _encode_metadata(df, META_COLS)[0:1]

    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    c_tensor = torch.as_tensor(c_scaled, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        reconstructed, future_pred = model(x_tensor, c_tensor) # DualHead output

    x_pred_scaled = reconstructed.detach().cpu().numpy()[0]

    errors = (x_scaled - x_pred_scaled) ** 2
    mse_per_feature = np.mean(errors, axis=0)

    feature_error_map = {
        feature: float(score)
        for feature, score in zip(["cpu_util_percent", "mem_util_percent", "cpu_request", "mem_request", "net_in", "net_out", "disk_io_percent"], mse_per_feature.tolist())
    }

    # all feature score
    all_score = float(np.mean(mse_per_feature))

    # top-k feature score
    ranked = sorted(feature_error_map.items(), key=lambda kv: kv[1], reverse=True)
    top_features = [k for k, _ in ranked[:TOPK]]
    top_feature_scores = [v for _, v in ranked[:TOPK]]
    top_score = float(np.mean(top_feature_scores)) if top_feature_scores else 0.0

    return all_score, top_score, feature_error_map, ranked, top_features, x_scaled[0], x_pred_scaled


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
    row=None,
):
    # Compact terminal line
    cpu = float(row["cpu_util"]) if row is not None else 0.0
    mem = float(row["mem_util"]) if row is not None else 0.0
    net = float(row["net_in"])   if row is not None else 0.0
    terminal_log(key, cpu, mem, net, all_score, top_score, all_status, top_status)

    # Structured detail to file log only
    LOGGER.info(
        "DUAL | %s | ALL=%s(%.6f/%.6f) TOP=%s(%.6f/%.6f) feats=%s reason=%s",
        "/".join(key), all_status, all_score, all_threshold,
        top_status, top_score, top_threshold,
        top_features, reason,
    )


# =========================================================
# MAIN
# =========================================================
def main():
    LOGGER.info("Loading model artifacts...")

    try:
        send_telegram_alert(
            "\U0001F680 Live Inference Agent Started!\n\nDual-Head Model loaded. Monitoring active.",
            skip_dedup=True,   # always send the startup banner
        )
        LOGGER.info("Startup Telegram message sent successfully.")
    except Exception as e:
        LOGGER.error(f"Startup Telegram message failed: {e}")

    try:
        from alibaba_trace.utils.ai_helper import get_configured_engine
        engine = get_configured_engine(temperature=0.1)
        LOGGER.info(f"OpenAI test initialized: {engine}")
    except Exception as e:
        LOGGER.error(f"OpenAI test initialization failed: {e}")

    with open(X_SCALER_PATH, 'r') as f:
        scaler_data = json.load(f)
    x_scaler = StreamingMinMaxScaler(["cpu_util_percent", "mem_util_percent", "cpu_request", "mem_request", "net_in", "net_out", "disk_io_percent"])
    x_scaler.set_params(np.array(scaler_data['min_']), np.array(scaler_data['max_']))

    base_threshold = 0.05
    c_dim = 2

    LOGGER.info(f"Base threshold : {base_threshold}")
    LOGGER.info(f"Context dim    : {c_dim}")

    model, window_size = build_model(7, c_dim)



    states = defaultdict(init_container_state)
    seen_keys = set()

    LOGGER.info("Starting continuous dual-status anomaly detection loop...")

    while True:
        try:
            snapshot = collect_snapshot()

            if not snapshot.empty:
                with open('incoming_k8s_metrics.jsonl', 'a', encoding='utf-8') as f:
                    js_data = snapshot.to_json(orient='records', lines=True)
                    f.write(js_data + ('\n' if not js_data.endswith('\n') else ''))

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

                all_score, top_score, feature_error_map, ranked, top_features, x_recent, x_recon = compute_anomaly_score(
                    window_rows=window_df,
                    x_scaler=x_scaler,
                    model=model,
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
                    warmup_info = f"{state['inference_count']}/{WARMUP_WINDOWS}"
                    terminal_log(
                        key,
                        cpu=float(row["cpu_util"]),
                        mem=float(row["mem_util"]),
                        net=float(row["net_in"]),
                        all_score=all_score,
                        top_score=top_score,
                        all_status="NORMAL",
                        top_status="NORMAL",
                        warmup_info=warmup_info,
                    )
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

                if all_changed == "STARTED" or top_changed == "STARTED":
                    all_status = "ANOMALY_STARTED" if all_changed == "STARTED" else all_status
                    top_status = "ANOMALY_STARTED" if top_changed == "STARTED" else top_status

                    # Log raw Prometheus snapshot for post-mortem
                    try:
                        with open('prometheus_anomaly_query_data.log', 'a', encoding='utf-8') as f:
                            f.write(str(window_df.iloc[-1].to_dict()) + '\n')
                            f.flush()
                    except Exception as _log_err:
                        LOGGER.warning("Failed to write anomaly snapshot log: %s", _log_err)

                    namespace, pod, container = key
                    entity_key = f"{namespace}/{pod}/{container}"

                    # ── STEP 1: GPT Root Cause Analysis ──────────────────────
                    gpt_insight = get_gpt_explanation(
                        top_features=top_features,
                        reason=reason,
                        scores={"all_score": all_score, "top_score": top_score},
                    )

                    # ── STEP 2: Format alert message ──────────────────────────
                    top_feat_errors = ", ".join(
                        f"{k}={v:.6f}"
                        for k, v in ranked[:len(top_features)]
                    )
                    triggered_by = []
                    if all_changed == "STARTED":
                        triggered_by.append("ALL")
                    if top_changed == "STARTED":
                        triggered_by.append("TOP")

                    alert_message = (
                        f"\U0001F6A8 ANOMALY DETECTED\n\n"
                        f"Entity      : {entity_key}\n"
                        f"Container   : {container}\n"
                        f"Triggered by: {', '.join(triggered_by)} score\n\n"
                        f"All Score   : {all_score:.6f}  (thr {all_threshold:.6f})\n"
                        f"Top Score   : {top_score:.6f}  (thr {top_threshold:.6f})\n\n"
                        f"Top features: {', '.join(top_features)}\n"
                        f"Feature MSE : {top_feat_errors}\n\n"
                        f"Reason      : {reason}\n"
                        f"GPT Insight : {gpt_insight}"
                    )

                    # ── STEP 3: Send Telegram alert ───────────────────────────
                    result = send_telegram_alert(
                        message=alert_message,
                        entity_key=entity_key,   # used for dedup fingerprint
                    )
                    if not result.get("success"):
                        LOGGER.warning(
                            "Telegram alert not delivered for %s: %s",
                            entity_key, result.get("error"),
                        )
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
                    row=row,
                )

            sys.stdout.write(f"Wait {POLL_INTERVAL_SECONDS}s ")
            sys.stdout.flush()
            for _ in range(POLL_INTERVAL_SECONDS):
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\n")

        except KeyboardInterrupt:
            LOGGER.info("Stopped by user.")
            break
        except Exception as e:
            LOGGER.exception(f"Loop error: {e}")
            time.sleep(10) # wait before retrying on general error

if __name__ == "__main__":
    main()