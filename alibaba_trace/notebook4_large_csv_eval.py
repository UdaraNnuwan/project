from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alibaba_trace.data_pipeline import FEATURE_COLS, META_COLS, WINDOW_SIZE
from alibaba_trace.model_architecture import load_dual_head_model, load_model


OUTPUTS_DIR = PACKAGE_DIR / "outputs"
SINGLE_HEAD_MODEL_PATH = OUTPUTS_DIR / "model.pt"
DUAL_HEAD_MODEL_PATH = OUTPUTS_DIR / "dual_head_model.pt"
DEFAULT_DATASET_PATH = OUTPUTS_DIR / "notebook4_large_test_dataset_100k.csv"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "notebook4_large_eval_metrics.json"

WINDOW = int(WINDOW_SIZE)
N_TS_FEAT = len(FEATURE_COLS)
N_META_FEAT = len(META_COLS)
LATENT_DIM = 64
LSTM_UNITS = (128, 64)
DROPOUT = 0.2
LARGE_PRIME = 999_983

I_CPU = 0
I_MEM = 1
I_CPUREQ = 2
I_MEMREQ = 3
I_NETIN = 4
I_NETOUT = 5
I_DISK = 6

ARCHETYPES = {
    "api": {
        "container_id": "api-gateway-001",
        "machine_id": "node-frontend-01",
        "base": np.array([0.42, 0.36, 0.45, 0.38, 0.70, 0.56, 0.20], dtype=np.float32),
    },
    "db": {
        "container_id": "db-postgres-001",
        "machine_id": "node-data-01",
        "base": np.array([0.30, 0.70, 0.33, 0.74, 0.14, 0.12, 0.44], dtype=np.float32),
    },
    "worker": {
        "container_id": "worker-batch-001",
        "machine_id": "node-infra-01",
        "base": np.array([0.64, 0.40, 0.67, 0.42, 0.08, 0.08, 0.22], dtype=np.float32),
    },
}


class ReconOnlyModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, ts_t: torch.Tensor, meta_t: torch.Tensor) -> torch.Tensor:
        out = self.model(ts_t, meta_t)
        if isinstance(out, tuple):
            return out[0]
        return out


def compute_film_vector(container_id: str, machine_id: str) -> np.ndarray:
    return np.array(
        [
            float(hash(container_id) % LARGE_PRIME) / LARGE_PRIME,
            float(hash(machine_id) % LARGE_PRIME) / LARGE_PRIME,
        ],
        dtype=np.float32,
    )


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.01, 0.99).astype(np.float32)


def load_reconstruction_model(device: torch.device) -> ReconOnlyModel:
    if SINGLE_HEAD_MODEL_PATH.exists():
        model = load_model(
            checkpoint_path=str(SINGLE_HEAD_MODEL_PATH),
            window_size=WINDOW,
            n_ts_features=N_TS_FEAT,
            n_meta_features=N_META_FEAT,
            latent_dim=LATENT_DIM,
            lstm_units=LSTM_UNITS,
            dropout_rate=DROPOUT,
            device=device,
        )
        source = SINGLE_HEAD_MODEL_PATH
    elif DUAL_HEAD_MODEL_PATH.exists():
        model = load_dual_head_model(
            checkpoint_path=str(DUAL_HEAD_MODEL_PATH),
            window_size=WINDOW,
            n_ts_features=N_TS_FEAT,
            n_meta_features=N_META_FEAT,
            latent_dim=LATENT_DIM,
            lstm_units=LSTM_UNITS,
            dropout_rate=DROPOUT,
            device=device,
        )
        source = DUAL_HEAD_MODEL_PATH
    else:
        raise FileNotFoundError(
            f"No model checkpoint found. Expected either {SINGLE_HEAD_MODEL_PATH} or {DUAL_HEAD_MODEL_PATH}."
        )
    wrapped = ReconOnlyModel(model).eval()
    print(f"Loaded reconstruction scoring model from: {source}")
    return wrapped


def build_normal_windows(container_type: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    spec = ARCHETYPES[container_type]
    base = spec["base"]

    windows = np.tile(base, (n, WINDOW, 1)).astype(np.float32)
    time_axis = np.linspace(0.0, 2.0 * np.pi, WINDOW, dtype=np.float32)
    sin_wave = np.sin(time_axis)[None, :]
    cos_wave = np.cos(time_axis)[None, :]

    windows += rng.normal(0.0, 0.015, size=windows.shape).astype(np.float32)
    windows[:, :, I_CPU] += 0.020 * sin_wave
    windows[:, :, I_MEM] += 0.015 * cos_wave
    windows[:, :, I_NETIN] += 0.020 * sin_wave
    windows[:, :, I_NETOUT] += 0.015 * cos_wave
    windows[:, :, I_CPUREQ] = windows[:, :, I_CPU] + rng.normal(0.02, 0.008, size=(n, WINDOW)).astype(np.float32)
    windows[:, :, I_MEMREQ] = windows[:, :, I_MEM] + rng.normal(0.02, 0.008, size=(n, WINDOW)).astype(np.float32)

    windows = _clip(windows)
    meta = np.tile(
        compute_film_vector(spec["container_id"], spec["machine_id"]),
        (n, 1),
    ).astype(np.float32)
    return windows, meta


def inject_strong_anomaly(container_type: str, windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    windows = windows.copy()
    if container_type == "api":
        windows[:, -20:, I_CPU] = _clip(0.96 + rng.normal(0.0, 0.015, size=(len(windows), 20)).astype(np.float32))
        windows[:, -20:, I_NETIN] = _clip(0.93 + rng.normal(0.0, 0.02, size=(len(windows), 20)).astype(np.float32))
        windows[:, -20:, I_NETOUT] = _clip(0.88 + rng.normal(0.0, 0.02, size=(len(windows), 20)).astype(np.float32))
    elif container_type == "db":
        ramp = np.linspace(0.0, 0.30, 25, dtype=np.float32)[None, :]
        windows[:, -25:, I_MEM] = _clip(0.82 + ramp + rng.normal(0.0, 0.015, size=(len(windows), 25)).astype(np.float32))
        windows[:, -25:, I_DISK] = _clip(0.80 + ramp + rng.normal(0.0, 0.015, size=(len(windows), 25)).astype(np.float32))
        windows[:, -25:, I_MEMREQ] = _clip(windows[:, -25:, I_MEM] + 0.03)
    elif container_type == "worker":
        windows[:, -20:, I_NETOUT] = _clip(0.95 + rng.normal(0.0, 0.015, size=(len(windows), 20)).astype(np.float32))
        windows[:, -20:, I_CPU] = _clip(0.86 + rng.normal(0.0, 0.02, size=(len(windows), 20)).astype(np.float32))
        windows[:, -20:, I_NETIN] = _clip(0.82 + rng.normal(0.0, 0.02, size=(len(windows), 20)).astype(np.float32))
    else:
        raise ValueError(f"Unknown container_type: {container_type}")
    return windows


def build_split(total_samples: int, anomalous: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    container_types = list(ARCHETYPES.keys())
    counts = [total_samples // len(container_types)] * len(container_types)
    for idx in range(total_samples % len(container_types)):
        counts[idx] += 1

    xs = []
    metas = []
    ys = []
    for container_type, count in zip(container_types, counts):
        ts_np, meta_np = build_normal_windows(container_type, count, rng)
        if anomalous:
            ts_np = inject_strong_anomaly(container_type, ts_np, rng)
            labels = np.ones(count, dtype=np.int64)
        else:
            labels = np.zeros(count, dtype=np.int64)
        xs.append(ts_np)
        metas.append(meta_np)
        ys.append(labels)

    x = np.concatenate(xs, axis=0)
    meta = np.concatenate(metas, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(x))
    return x[perm], meta[perm], y[perm]


def save_dataset_csv(dataset_path: Path, calib_x: np.ndarray, calib_meta: np.ndarray, test_x: np.ndarray, test_meta: np.ndarray, test_y: np.ndarray) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def _pack(split_name: str, x_arr: np.ndarray, meta_arr: np.ndarray, y_arr: np.ndarray) -> pd.DataFrame:
        rows: dict[str, np.ndarray] = {
            "split": np.full(len(x_arr), split_name, dtype=object),
            "label": y_arr.astype(np.int64),
        }
        for t in range(WINDOW):
            for f_idx, feature_name in enumerate(FEATURE_COLS):
                rows[f"ts_{t:02d}_{feature_name}"] = x_arr[:, t, f_idx]
        for m_idx, meta_name in enumerate(META_COLS):
            rows[f"meta_{meta_name}"] = meta_arr[:, m_idx]
        return pd.DataFrame(rows)

    calib_y = np.zeros(len(calib_x), dtype=np.int64)
    df = pd.concat(
        [
            _pack("calib", calib_x, calib_meta, calib_y),
            _pack("test", test_x, test_meta, test_y),
        ],
        ignore_index=True,
    )
    df.to_csv(dataset_path, index=False)
    return dataset_path


def load_dataset_csv(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset_path)
    ts_cols = [f"ts_{t:02d}_{feature_name}" for t in range(WINDOW) for feature_name in FEATURE_COLS]
    meta_cols = [f"meta_{meta_name}" for meta_name in META_COLS]

    calib_df = df[df["split"] == "calib"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    calib_x = calib_df[ts_cols].to_numpy(dtype=np.float32).reshape(len(calib_df), WINDOW, N_TS_FEAT)
    calib_meta = calib_df[meta_cols].to_numpy(dtype=np.float32)
    test_x = test_df[ts_cols].to_numpy(dtype=np.float32).reshape(len(test_df), WINDOW, N_TS_FEAT)
    test_meta = test_df[meta_cols].to_numpy(dtype=np.float32)
    test_y = test_df["label"].to_numpy(dtype=np.int64)
    return calib_x, calib_meta, test_x, test_meta, test_y


def batched_recon_mse(model: ReconOnlyModel, x_np: np.ndarray, meta_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            end = min(start + batch_size, len(x_np))
            ts_t = torch.from_numpy(x_np[start:end]).to(device)
            meta_t = torch.from_numpy(meta_np[start:end]).to(device)
            recon_t = model(ts_t, meta_t)
            mse = ((ts_t - recon_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
            scores.append(mse)
    return np.concatenate(scores).astype(np.float64)


def evaluate_csv_dataset(dataset_path: Path, batch_size: int) -> dict[str, float | int | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_reconstruction_model(device)
    calib_x, calib_meta, test_x, test_meta, test_y = load_dataset_csv(dataset_path)

    calib_scores = batched_recon_mse(model, calib_x, calib_meta, device, batch_size)
    threshold_p95 = float(np.percentile(calib_scores, 95.0))
    threshold_p99 = float(np.percentile(calib_scores, 99.0))

    test_scores = batched_recon_mse(model, test_x, test_meta, device, batch_size)
    y_pred = (test_scores >= threshold_p95).astype(np.int64)

    normal_scores = test_scores[test_y == 0]
    anomaly_scores = test_scores[test_y == 1]
    accuracy = float(accuracy_score(test_y, y_pred))
    precision = float(precision_score(test_y, y_pred, zero_division=0))
    recall = float(recall_score(test_y, y_pred, zero_division=0))
    f1 = float(f1_score(test_y, y_pred, zero_division=0))
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(test_y, y_pred, labels=[0, 1]).ravel()]

    return {
        "note": "Notebook 4 style large synthetic benchmark using reconstruction-MSE thresholding. Synthetic only.",
        "dataset_path": str(dataset_path),
        "threshold_p95": threshold_p95,
        "threshold_p99": threshold_p99,
        "n_calibration_normal": int(len(calib_x)),
        "n_test_total": int(len(test_x)),
        "n_test_normal": int((test_y == 0).sum()),
        "n_test_anomalous": int((test_y == 1).sum()),
        "normal_score_mean": float(normal_scores.mean()),
        "normal_score_std": float(normal_scores.std()),
        "anomaly_score_mean": float(anomaly_scores.mean()),
        "anomaly_score_std": float(anomaly_scores.std()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": float(fp / max(1, tn + fp)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 4 style large CSV benchmark.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--n-calib", type=int, default=20000)
    parser.add_argument("--n-normal", type=int, default=50000)
    parser.add_argument("--n-anom", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args()

    if not args.reuse_dataset:
        rng = np.random.default_rng(args.seed)
        calib_x, calib_meta, _ = build_split(args.n_calib, anomalous=False, rng=rng)
        normal_x, normal_meta, normal_y = build_split(args.n_normal, anomalous=False, rng=rng)
        anom_x, anom_meta, anom_y = build_split(args.n_anom, anomalous=True, rng=rng)

        test_x = np.concatenate([normal_x, anom_x], axis=0)
        test_meta = np.concatenate([normal_meta, anom_meta], axis=0)
        test_y = np.concatenate([normal_y, anom_y], axis=0)
        perm = rng.permutation(len(test_x))
        test_x = test_x[perm]
        test_meta = test_meta[perm]
        test_y = test_y[perm]
        save_dataset_csv(args.dataset_path, calib_x, calib_meta, test_x, test_meta, test_y)

    metrics = evaluate_csv_dataset(args.dataset_path, args.batch_size)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Notebook 4 Large CSV Evaluation")
    print("=" * 72)
    print(metrics["note"])
    print(f"Calibration normal windows : {metrics['n_calibration_normal']:,}")
    print(f"Test normal windows        : {metrics['n_test_normal']:,}")
    print(f"Test anomalous windows     : {metrics['n_test_anomalous']:,}")
    print(f"Test total windows         : {metrics['n_test_total']:,}")
    print("-" * 72)
    print(f"P95 threshold              : {metrics['threshold_p95']:.6f}")
    print(f"P99 threshold              : {metrics['threshold_p99']:.6f}")
    print(f"Accuracy                   : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision                  : {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall                     : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-score                   : {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
    print(f"TN / FP / FN / TP          : {metrics['tn']} / {metrics['fp']} / {metrics['fn']} / {metrics['tp']}")
    print("-" * 72)
    print(f"Saved dataset CSV          : {args.dataset_path}")
    print(f"Saved metrics JSON         : {args.metrics_path}")


if __name__ == "__main__":
    main()
