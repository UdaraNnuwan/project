import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
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
# Adjust these values to match your target workload.
# =========================================================
CSV_PATH = os.path.join(CURRENT_DIR, "../prometheus/prometheus_timeseries_metrics.csv")
LOG_PATH = os.path.join(CURRENT_DIR, "result.log")

MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "ae_model.pt")
X_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
C_SCALER_PATH = os.path.join(MODEL_DIR, "ctx_scaler.joblib")
DETECTOR_META_PATH = os.path.join(MODEL_DIR, "detector_meta.joblib")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target selector for the live Prometheus CSV data.
TARGET_NAMESPACE = "default"
TARGET_POD = "chcekone"  # Matches pods whose name contains this value.
TARGET_CONTAINER = "nginx"

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


def setup_logger():
    logger = logging.getLogger("live_film_inference")
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


def print(*args, sep=" ", end="\n", **kwargs):
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message = f"{message}{end}"
    LOGGER.info(message.rstrip("\n"))


def load_assets():
    """Load the trained model, scalers, and detector metadata."""
    LOGGER.info("[INFO] Loading scalers and metadata...")
    x_scaler = joblib.load(X_SCALER_PATH)
    c_scaler = joblib.load(C_SCALER_PATH)
    det_meta = joblib.load(DETECTOR_META_PATH)

    threshold = det_meta["threshold"]
    window_size = det_meta["window_size"]

    # Support checkpoints saved either as a plain state dict or a wrapped dict.
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


def main():
    x_scaler, c_scaler, threshold, window_size, model = load_assets()

    LOGGER.info("")
    LOGGER.info(f"[INFO] Starting Real-Time Monitoring for {TARGET_CONTAINER}...")

    while True:
        try:
            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
                mask = (
                    (df["namespace"] == TARGET_NAMESPACE)
                    & (df["pod"].str.contains(TARGET_POD, na=False))
                    & (df["container"] == TARGET_CONTAINER)
                )

                target_df = df[mask].tail(window_size)

                if len(target_df) >= window_size:
                    x_raw = target_df[FEATURE_COLS].values
                    x_log = np.log1p(np.clip(x_raw, a_min=0, a_max=None))
                    x_scaled = x_scaler.transform(x_log)

                    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).to(DEVICE)
                    c_scaled = np.zeros((1, c_scaler.n_features_in_))
                    c_tensor = torch.FloatTensor(c_scaled).to(DEVICE)

                    with torch.no_grad():
                        reconstructed = model(x_tensor, c_tensor)

                    x_pred_scaled = reconstructed.cpu().numpy().squeeze()

                    # Compute reconstruction error per feature and overall anomaly score.
                    errors = (x_scaled - x_pred_scaled) ** 2
                    mse_per_feature = np.mean(errors, axis=0)
                    total_score = np.mean(mse_per_feature)

                    # Write the latest inference summary to the console and log file.
                    LOGGER.info("")
                    LOGGER.info("=" * 50)
                    LOGGER.info(f"POD       : {target_df['pod'].iloc[-1]}")
                    LOGGER.info(f"CONTAINER : {TARGET_CONTAINER}")
                    LOGGER.info(f"SCORE     : {total_score:.6f}")
                    LOGGER.info(f"THRESHOLD : {threshold:.6f}")

                    if total_score > threshold:
                        LOGGER.info("STATUS    : ANOMALY DETECTED!")
                        LOGGER.info("-" * 50)
                        LOGGER.info("REASON (TOP CONTRIBUTING METRICS):")

                        # Report the three metrics with the highest reconstruction error.
                        top_indices = np.argsort(mse_per_feature)[::-1][:3]
                        for idx in top_indices:
                            f_name = FEATURE_COLS[idx]
                            f_error = mse_per_feature[idx]
                            # Show the latest raw metric value beside the anomaly contribution.
                            actual_val = x_raw[-1, idx]
                            LOGGER.info(
                                f" -> {f_name.upper()}: Error={f_error:.4f} "
                                f"(Actual Val={actual_val:.2f})"
                            )

                        LOGGER.info("-" * 50)
                        LOGGER.info("LATEST LOG DATA (Snapshot):")
                        LOGGER.info("\n%s", target_df[FEATURE_COLS].tail(1).to_string(index=False))
                    else:
                        LOGGER.info("STATUS    : NORMAL")

                    LOGGER.info("=" * 50)

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(5)


if __name__ == "__main__":
    main()
