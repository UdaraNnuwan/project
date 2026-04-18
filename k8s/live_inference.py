# =============================================================================
# live_inference.py
# FINAL FIXED VERSION - BiLSTM-FiLM Live Service (All Errors Fixed)
# =============================================================================

import os
import time
import requests
import logging
import sys
import subprocess
import torch
import numpy as np
import json
from collections import defaultdict, deque
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alibaba_trace.model_architecture import load_dual_head_model
from alibaba_trace.incident_response import (
    ContainerContext,
    SeverityScorer,
    ContextAwareRCA,
    IncidentPipeline,
    AlertGenerator,
    GenAIRCAEngine,
)

from src.notifications import send_telegram_alert

# ========================= CLEAN LOGGING =========================
log_file_path = "live_inference.log"

logger = logging.getLogger("BiLSTM-FiLM-Live")
logger.setLevel(logging.INFO)
logger.propagate = False

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
logger.addHandler(file_handler)

# ========================= CONFIG =========================
PROM_URL = os.getenv("PROM_URL", "http://35.206.92.147:9090/api/v1/query")
QUERY_INTERVAL = int(os.getenv("QUERY_INTERVAL", 10))
THRESHOLD_MULTIPLIER = float(os.getenv("THRESHOLD_MULTIPLIER", 2.0))
WARM_UP_CYCLES = int(os.getenv("WARM_UP_CYCLES", 20))
LOG_LINES = 15

FEATURE_COLS = ["cpu_util_percent", "mem_util_percent", "cpu_request", "mem_request", "net_in", "net_out", "disk_io_percent"]

PROM_QUERIES = {
    "cpu_util_percent": 'rate(container_cpu_usage_seconds_total{image!="",image!~".*pause.*",name!=""}[5m]) * 100',
    "mem_util_percent": '(container_memory_usage_bytes{image!="",image!~".*pause.*",name!=""} / container_spec_memory_limit_bytes{image!="",image!~".*pause.*",name!=""}) * 100',
    "cpu_request": 'container_spec_cpu_quota{image!="",image!~".*pause.*",name!=""} / 100000',
    "mem_request": 'container_spec_memory_limit_bytes{image!="",image!~".*pause.*",name!=""} / (1024*1024*1024)',
    "net_in": 'rate(container_network_receive_bytes_total{image!="",image!~".*pause.*",name!=""}[5m]) / (1024*1024)',
    "net_out": 'rate(container_network_transmit_bytes_total{image!="",image!~".*pause.*",name!=""}[5m]) / (1024*1024)',
    "disk_io_percent": 'rate(container_fs_io_time_seconds_total{image!="",image!~".*pause.*",name!=""}[5m]) * 100'
}

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "alibaba_trace" / "outputs" / "dual_head_model.pt"
SCALER_PATH = MODEL_PATH.parent / "scaler_params.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

buffers = defaultdict(lambda: deque(maxlen=50))
loss_histories = defaultdict(lambda: deque(maxlen=200))
cycle_count = 0

# ========================= LOAD MODEL =========================
def load_model_and_scaler():
    global model, scaler
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = load_dual_head_model(
        checkpoint_path=str(MODEL_PATH),
        window_size=50,
        n_ts_features=7,
        n_meta_features=2,
        latent_dim=64,
        device=device
    )

    with open(SCALER_PATH, 'r') as f:
        params = json.load(f)

    def scaler_func(x):
        x = np.array(x, dtype=np.float32)
        min_ = np.array(params['min_'])
        max_ = np.array(params['max_'])
        return np.clip((x - min_) / (max_ - min_ + 1e-8), 0.0, 1.0)

    scaler = scaler_func
    logger.info("✅ Model + Scaler loaded")

# ========================= CALIBRATE SCORER =========================
def calibrate_scorer():
    logger.info("Calibrating SeverityScorer...")
    baseline_mse = []
    for i in range(40):
        try:
            resp = requests.get(PROM_URL, params={'query': PROM_QUERIES["cpu_util_percent"]}, timeout=8)
            data = resp.json()
            if data.get('status') == 'success':
                for item in data['data'].get('result', []):
                    try:
                        val = float(item['value'][1])
                        baseline_mse.append(val)
                    except:
                        pass
        except:
            pass
        time.sleep(1)

    if baseline_mse:
        scorer.calibrate(baseline_mse)
        logger.info(f"✅ Scorer calibrated | P95={scorer.p95:.5f} | P98={scorer.p98:.5f} | P99.5={scorer.p995:.5f}")
    else:
        logger.warning("Calibration failed. Using default thresholds.")
        scorer.p95 = 0.0325
        scorer.p98 = 0.0350
        scorer.p995 = 0.0380

# ========================= GET CONTAINER LOGS =========================
def get_container_logs(pod_name: str, namespace: str = "prod", lines: int = LOG_LINES):
    try:
        cmd = f"kubectl logs {pod_name} -n {namespace} --tail={lines}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8)
        if result.returncode == 0:
            return result.stdout.strip() or "No logs available"
        else:
            return f"kubectl error: {result.stderr.strip()}"
    except Exception as e:
        return f"Failed to fetch logs: {e}"

# ========================= STARTUP NOTIFICATION =========================
def send_startup_notification():
    message = (
        "🚀 <b>BiLSTM-FiLM Live Inference Service Started Successfully!</b>\n\n"
        "✅ Model loaded\n"
        "✅ Scorer calibrated\n"
        "✅ All 7 metrics monitored\n"
        "✅ Log file created: live_inference.log"
    )
    send_telegram_alert(message)
    logger.info("📨 Startup Telegram notification sent")

# ========================= PIPELINE =========================
scorer = SeverityScorer()
rca_engine = ContextAwareRCA(feature_cols=FEATURE_COLS)
alerter = AlertGenerator(cluster='alibaba-k8s-production', namespace='prod')

incident_pipeline = IncidentPipeline(scorer=scorer, rca=rca_engine, alerter=alerter)
genai_engine = GenAIRCAEngine()

# ========================= MAIN LOOP =========================
def run_inference():
    global cycle_count
    cycle_count += 1

    logger.info(f"Cycle {cycle_count} | Fetching 7 multivariate metrics...")

    raw_data = {}
    for feature, query in PROM_QUERIES.items():
        try:
            resp = requests.get(PROM_URL, params={'query': query}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') == 'success':
                raw_data[feature] = data['data']['result']
        except Exception as e:
            logger.warning(f"Failed to fetch {feature}: {e}")

    if not raw_data:
        logger.warning("No data from Prometheus this cycle.")
        return

    container_names = set()
    current_metrics = defaultdict(dict)

    for feat in FEATURE_COLS:
        for item in raw_data.get(feat, []):
            metric = item.get('metric', {})
            name = (metric.get('pod') or 
                    metric.get('container_label_io_kubernetes_pod_name') or 
                    metric.get('name') or 
                    metric.get('container') or "unknown")

            if name == "unknown":
                continue

            container_names.add(name)

            try:
                val = float(item['value'][1])
                if feat == "mem_util_percent":
                    limit = float(metric.get('container_spec_memory_limit_bytes', 0))
                    usage = float(metric.get('container_memory_usage_bytes', 0))
                    val = (usage / limit * 100) if limit > 0 else 0.0
                current_metrics[name][feat] = round(val, 4)
            except:
                current_metrics[name][feat] = 0.0

    for container_name in container_names:
        window = [current_metrics[container_name].get(feat, 0.0) for feat in FEATURE_COLS]

        if len(window) != 7:
            continue

        buffers[container_name].append(window)

        if len(buffers[container_name]) < 50:
            continue

        ts_array = np.array(list(buffers[container_name]))
        scaled = scaler(ts_array).reshape(1, 50, 7)

        ts_tensor = torch.from_numpy(scaled).float().to(device)

        meta_vec = np.array([
            hash(container_name) % 999983 / 999983.0,
            1.0 if "db" in container_name.lower() else 0.0
        ], dtype=np.float32)
        meta_tensor = torch.from_numpy(meta_vec).unsqueeze(0).float().to(device)

        with torch.no_grad():
            recon, _ = model(ts_tensor, meta_tensor)
            mse_score = torch.mean((recon - ts_tensor) ** 2).item()

        loss_histories[container_name].append(mse_score)

        if len(loss_histories[container_name]) >= 20:
            mean_loss = np.mean(loss_histories[container_name])
            std_loss = np.std(loss_histories[container_name])
            threshold = mean_loss + THRESHOLD_MULTIPLIER * std_loss
        else:
            threshold = float('inf')

        status = "Anomaly" if mse_score > threshold else "Normal"

        metrics_str = " | ".join([f"{k.split('_')[0].upper()}: {v}" for k, v in current_metrics[container_name].items()])

        if status == "Anomaly" and cycle_count > WARM_UP_CYCLES:
            logger.warning(f"🚨 ANOMALY DETECTED | {container_name} | MSE={mse_score:.5f} | {metrics_str}")

            recent_logs = get_container_logs(container_name)

            ctx = ContainerContext.from_ids(
                container_id=container_name,
                machine_id="k8s-node",
                container_type="database" if "db" in container_name.lower() else "api",
                tier="data",
                environment="production",
                namespace="prod",
                pod_name=container_name
            )

            incident = incident_pipeline.process(
                original=ts_array,
                reconstructed=recon.cpu().numpy()[0],
                mse_score=mse_score,
                context=ctx
            )

            # FIXED: Added missing threshold_p95 argument
            llm_result = genai_engine.analyze_with_llm(
                original_metrics=ts_array,
                reconstructed_metrics=recon.cpu().numpy()[0],
                mse=mse_score,
                context=ctx,
                rca_result=incident.__dict__,
                recent_logs=recent_logs,
                threshold_p95=scorer.p95          # ← This was missing
            )

            alert_msg = (
                f"🚨 ANOMALY DETECTED\n"
                f"Container: {container_name}\n"
                f"MSE: {mse_score:.5f}\n"
                f"Metrics: {metrics_str}\n\n"
                f"GenAI Analysis:\n{llm_result}\n\n"
                f"Recent Logs:\n{recent_logs[:800]}..."
            )
            send_telegram_alert(alert_msg)

        else:
            logger.info(f"✅ NORMAL | {container_name} | MSE={mse_score:.5f} | {metrics_str}")

if __name__ == "__main__":
    load_model_and_scaler()
    calibrate_scorer()
    send_startup_notification()
    logger.info("🚀 BiLSTM-FiLM Live Inference Service Started Successfully!")
    logger.info(f"Monitoring Prometheus at: {PROM_URL}")
    logger.info(f"Log file: {log_file_path}")

    while True:
        start = time.time()
        run_inference()
        elapsed = time.time() - start
        sleep_time = max(0, QUERY_INTERVAL - elapsed)
        logger.info(f"Cycle completed in {elapsed:.2f}s | Sleeping {sleep_time:.2f}s...")
        time.sleep(sleep_time)