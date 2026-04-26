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
from alibaba_trace.model_architecture import load_dual_head_model


OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUTPUTS_DIR / "dual_head_model.pt"
DEFAULT_DATASET_PATH = OUTPUTS_DIR / "large_scale_test_dataset_100k.csv"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "large_scale_eval_metrics.json"

WINDOW = int(WINDOW_SIZE)
N_TS_FEAT = len(FEATURE_COLS)
N_META_FEAT = len(META_COLS)
LATENT_DIM = 64
LSTM_UNITS = (128, 64)
DROPOUT = 0.2
ALPHA = 0.5
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
        "base": np.array([0.42, 0.38, 0.45, 0.40, 0.72, 0.60, 0.22], dtype=np.float32),
    },
    "db": {
        "container_id": "db-postgres-001",
        "machine_id": "node-data-01",
        "base": np.array([0.34, 0.72, 0.35, 0.76, 0.18, 0.16, 0.48], dtype=np.float32),
    },
    "worker": {
        "container_id": "worker-batch-001",
        "machine_id": "node-infra-01",
        "base": np.array([0.68, 0.42, 0.72, 0.45, 0.10, 0.11, 0.26], dtype=np.float32),
    },
}


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


def build_normal_windows(container_type: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = ARCHETYPES[container_type]
    base = spec["base"]

    windows = np.tile(base, (n, WINDOW, 1)).astype(np.float32)
    time_axis = np.linspace(0.0, 2.0 * np.pi, WINDOW, dtype=np.float32)
    sin_wave = np.sin(time_axis)[None, :]
    cos_wave = np.cos(time_axis)[None, :]

    windows += rng.normal(0.0, 0.025, size=windows.shape).astype(np.float32)
    windows[:, :, I_CPU] += 0.03 * sin_wave
    windows[:, :, I_MEM] += 0.02 * cos_wave
    windows[:, :, I_NETIN] += 0.03 * sin_wave
    windows[:, :, I_NETOUT] += 0.02 * cos_wave
    windows[:, :, I_CPUREQ] = windows[:, :, I_CPU] + rng.normal(0.02, 0.01, size=(n, WINDOW)).astype(np.float32)
    windows[:, :, I_MEMREQ] = windows[:, :, I_MEM] + rng.normal(0.02, 0.01, size=(n, WINDOW)).astype(np.float32)

    windows = _clip(windows)
    next_step = _clip(windows[:, -1, :] + rng.normal(0.0, 0.02, size=(n, N_TS_FEAT)).astype(np.float32))
    meta = np.tile(
        compute_film_vector(spec["container_id"], spec["machine_id"]),
        (n, 1),
    ).astype(np.float32)
    return windows, next_step, meta


def inject_anomaly(
    container_type: str,
    windows: np.ndarray,
    next_step: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    windows = windows.copy()
    next_step = next_step.copy()

    if container_type == "api":
        windows[:, -20:, I_CPU] = _clip(0.92 + rng.normal(0.0, 0.025, size=(len(windows), 20)).astype(np.float32))
        windows[:, -15:, I_NETIN] = _clip(0.88 + rng.normal(0.0, 0.03, size=(len(windows), 15)).astype(np.float32))
        next_step[:, I_CPU] = _clip(0.96 + rng.normal(0.0, 0.015, size=len(windows)).astype(np.float32))
        next_step[:, I_NETIN] = _clip(0.90 + rng.normal(0.0, 0.02, size=len(windows)).astype(np.float32))
    elif container_type == "db":
        ramp = np.linspace(0.0, 0.25, 25, dtype=np.float32)[None, :]
        windows[:, -25:, I_MEM] = _clip(0.78 + ramp + rng.normal(0.0, 0.02, size=(len(windows), 25)).astype(np.float32))
        windows[:, -25:, I_DISK] = _clip(0.74 + ramp + rng.normal(0.0, 0.02, size=(len(windows), 25)).astype(np.float32))
        next_step[:, I_MEM] = _clip(0.97 + rng.normal(0.0, 0.01, size=len(windows)).astype(np.float32))
        next_step[:, I_DISK] = _clip(0.90 + rng.normal(0.0, 0.02, size=len(windows)).astype(np.float32))
    elif container_type == "worker":
        windows[:, -20:, I_NETOUT] = _clip(0.93 + rng.normal(0.0, 0.02, size=(len(windows), 20)).astype(np.float32))
        windows[:, -20:, I_CPU] = _clip(windows[:, -20:, I_CPU] + 0.08)
        next_step[:, I_NETOUT] = _clip(0.96 + rng.normal(0.0, 0.015, size=len(windows)).astype(np.float32))
    else:
        raise ValueError(f"Unknown container_type: {container_type}")

    return windows, _clip(next_step)


def build_balanced_dataset(
    total_samples: int,
    anomalous: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    container_types = list(ARCHETYPES.keys())
    counts = [total_samples // len(container_types)] * len(container_types)
    for idx in range(total_samples % len(container_types)):
        counts[idx] += 1

    all_ts = []
    all_next = []
    all_meta = []
    all_labels = []

    for container_type, count in zip(container_types, counts):
        ts_np, next_np, meta_np = build_normal_windows(container_type, count, rng)
        if anomalous:
            ts_np, next_np = inject_anomaly(container_type, ts_np, next_np, rng)
            labels = np.ones(count, dtype=np.int64)
        else:
            labels = np.zeros(count, dtype=np.int64)

        all_ts.append(ts_np)
        all_next.append(next_np)
        all_meta.append(meta_np)
        all_labels.append(labels)

    ts = np.concatenate(all_ts, axis=0)
    nxt = np.concatenate(all_next, axis=0)
    meta = np.concatenate(all_meta, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    perm = rng.permutation(len(ts))
    return ts[perm], nxt[perm], meta[perm], labels[perm]


def create_large_scale_dataset(
    n_calib: int,
    n_normal: int,
    n_anom: int,
    seed: int,
) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(seed)

    calib_ts, calib_next, calib_meta, _ = build_balanced_dataset(n_calib, anomalous=False, rng=rng)
    normal_ts, normal_next, normal_meta, y_normal = build_balanced_dataset(n_normal, anomalous=False, rng=rng)
    anom_ts, anom_next, anom_meta, y_anom = build_balanced_dataset(n_anom, anomalous=True, rng=rng)

    x_test = np.concatenate([normal_ts, anom_ts], axis=0)
    next_test = np.concatenate([normal_next, anom_next], axis=0)
    c_test = np.concatenate([normal_meta, anom_meta], axis=0)
    y_test = np.concatenate([y_normal, y_anom], axis=0)

    perm = rng.permutation(len(x_test))
    x_test = x_test[perm]
    next_test = next_test[perm]
    c_test = c_test[perm]
    y_test = y_test[perm]

    return {
        "calib_ts": calib_ts.astype(np.float32),
        "calib_next": calib_next.astype(np.float32),
        "calib_meta": calib_meta.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "next_test": next_test.astype(np.float32),
        "c_test": c_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
        "seed": np.array(seed, dtype=np.int64),
        "n_calib": np.array(n_calib, dtype=np.int64),
        "n_normal": np.array(n_normal, dtype=np.int64),
        "n_anom": np.array(n_anom, dtype=np.int64),
    }


def dataset_to_dataframe(dataset: dict[str, np.ndarray | int]) -> pd.DataFrame:
    def _pack_split(split_name: str, ts_arr: np.ndarray, next_arr: np.ndarray, meta_arr: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        rows: dict[str, np.ndarray] = {
            "split": np.full(len(ts_arr), split_name, dtype=object),
            "label": labels.astype(np.int64),
        }

        for t in range(WINDOW):
            for f_idx, feature_name in enumerate(FEATURE_COLS):
                rows[f"ts_{t:02d}_{feature_name}"] = ts_arr[:, t, f_idx]

        for f_idx, feature_name in enumerate(FEATURE_COLS):
            rows[f"next_{feature_name}"] = next_arr[:, f_idx]

        for m_idx, meta_name in enumerate(META_COLS):
            rows[f"meta_{meta_name}"] = meta_arr[:, m_idx]

        return pd.DataFrame(rows)

    calib_labels = np.zeros(len(dataset["calib_ts"]), dtype=np.int64)
    frames = [
        _pack_split("calib", dataset["calib_ts"], dataset["calib_next"], dataset["calib_meta"], calib_labels),
        _pack_split("test", dataset["x_test"], dataset["next_test"], dataset["c_test"], dataset["y_test"]),
    ]
    return pd.concat(frames, ignore_index=True)


def dataframe_to_dataset(
    df: pd.DataFrame,
    seed: int,
    n_calib: int,
    n_normal: int,
    n_anom: int,
) -> dict[str, np.ndarray]:
    ts_cols = [f"ts_{t:02d}_{feature_name}" for t in range(WINDOW) for feature_name in FEATURE_COLS]
    next_cols = [f"next_{feature_name}" for feature_name in FEATURE_COLS]
    meta_cols = [f"meta_{meta_name}" for meta_name in META_COLS]

    calib_df = df[df["split"] == "calib"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    calib_ts = calib_df[ts_cols].to_numpy(dtype=np.float32).reshape(len(calib_df), WINDOW, N_TS_FEAT)
    calib_next = calib_df[next_cols].to_numpy(dtype=np.float32)
    calib_meta = calib_df[meta_cols].to_numpy(dtype=np.float32)
    x_test = test_df[ts_cols].to_numpy(dtype=np.float32).reshape(len(test_df), WINDOW, N_TS_FEAT)
    next_test = test_df[next_cols].to_numpy(dtype=np.float32)
    c_test = test_df[meta_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    return {
        "calib_ts": calib_ts,
        "calib_next": calib_next,
        "calib_meta": calib_meta,
        "x_test": x_test,
        "next_test": next_test,
        "c_test": c_test,
        "y_test": y_test,
        "seed": np.array(seed, dtype=np.int64),
        "n_calib": np.array(n_calib, dtype=np.int64),
        "n_normal": np.array(n_normal, dtype=np.int64),
        "n_anom": np.array(n_anom, dtype=np.int64),
    }


def save_large_scale_dataset(dataset: dict[str, np.ndarray | int], dataset_path: Path) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset_to_dataframe(dataset)
    df.to_csv(dataset_path, index=False)
    return dataset_path


def load_large_scale_dataset(
    dataset_path: Path,
    seed: int,
    n_calib: int,
    n_normal: int,
    n_anom: int,
) -> dict[str, np.ndarray]:
    df = pd.read_csv(dataset_path)
    return dataframe_to_dataset(
        df=df,
        seed=seed,
        n_calib=n_calib,
        n_normal=n_normal,
        n_anom=n_anom,
    )


def batched_scores(
    model: torch.nn.Module,
    ts_np: np.ndarray,
    next_np: np.ndarray,
    meta_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(ts_np), batch_size):
            end = min(start + batch_size, len(ts_np))
            ts_t = torch.from_numpy(ts_np[start:end]).to(device)
            next_t = torch.from_numpy(next_np[start:end]).to(device)
            meta_t = torch.from_numpy(meta_np[start:end]).to(device)

            recon, fore = model(ts_t, meta_t)
            mse_r = ((ts_t - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            mse_f = ((next_t - fore) ** 2).mean(dim=1).cpu().numpy()
            scores.append(ALPHA * mse_r + (1.0 - ALPHA) * mse_f)

    return np.concatenate(scores).astype(np.float64)


def evaluate_saved_dataset(
    dataset: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, float | int | str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dual_head_model(
        checkpoint_path=str(MODEL_PATH),
        window_size=WINDOW,
        n_ts_features=N_TS_FEAT,
        n_meta_features=N_META_FEAT,
        latent_dim=LATENT_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT,
        device=device,
    )

    calib_scores = batched_scores(
        model=model,
        ts_np=dataset["calib_ts"],
        next_np=dataset["calib_next"],
        meta_np=dataset["calib_meta"],
        device=device,
        batch_size=batch_size,
    )
    threshold_p95 = float(np.percentile(calib_scores, 95.0))
    threshold_p99 = float(np.percentile(calib_scores, 99.0))

    scores = batched_scores(
        model=model,
        ts_np=dataset["x_test"],
        next_np=dataset["next_test"],
        meta_np=dataset["c_test"],
        device=device,
        batch_size=batch_size,
    )
    y_true = dataset["y_test"].astype(np.int64)
    y_pred = (scores >= threshold_p95).astype(np.int64)

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    metrics = {
        "note": "Large-scale synthetic FiLM-aware saved test dataset. This is not a real labeled production dataset.",
        "model_path": str(MODEL_PATH),
        "device": str(device),
        "window_size": WINDOW,
        "n_ts_features": N_TS_FEAT,
        "n_meta_features": N_META_FEAT,
        "alpha": ALPHA,
        "seed": int(dataset["seed"]),
        "n_calibration_normal": int(dataset["n_calib"]),
        "n_test_normal": int(dataset["n_normal"]),
        "n_test_anomalous": int(dataset["n_anom"]),
        "n_test_total": int(len(y_true)),
        "threshold_p95": threshold_p95,
        "threshold_p99": threshold_p99,
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
    return metrics


def print_metrics(metrics: dict[str, float | int | str], dataset_path: Path, metrics_path: Path) -> None:
    print("=" * 72)
    print("Large-Scale Trained Model Evaluation")
    print("=" * 72)
    print(metrics["note"])
    print(f"Calibration normal windows : {int(metrics['n_calibration_normal']):,}")
    print(f"Test normal windows        : {int(metrics['n_test_normal']):,}")
    print(f"Test anomalous windows     : {int(metrics['n_test_anomalous']):,}")
    print(f"Test total windows         : {int(metrics['n_test_total']):,}")
    print("-" * 72)
    print(f"P95 threshold              : {float(metrics['threshold_p95']):.6f}")
    print(f"P99 threshold              : {float(metrics['threshold_p99']):.6f}")
    print(f"Normal score mean/std      : {float(metrics['normal_score_mean']):.6f} / {float(metrics['normal_score_std']):.6f}")
    print(f"Anomaly score mean/std     : {float(metrics['anomaly_score_mean']):.6f} / {float(metrics['anomaly_score_std']):.6f}")
    print("-" * 72)
    print(f"Accuracy                   : {float(metrics['accuracy']):.4f} ({float(metrics['accuracy']) * 100:.2f}%)")
    print(f"Precision                  : {float(metrics['precision']):.4f} ({float(metrics['precision']) * 100:.2f}%)")
    print(f"Recall                     : {float(metrics['recall']):.4f} ({float(metrics['recall']) * 100:.2f}%)")
    print(f"F1-score                   : {float(metrics['f1_score']):.4f} ({float(metrics['f1_score']) * 100:.2f}%)")
    print(
        "TN / FP / FN / TP          : "
        f"{int(metrics['tn'])} / {int(metrics['fp'])} / {int(metrics['fn'])} / {int(metrics['tp'])}"
    )
    print(f"False positive rate        : {float(metrics['false_positive_rate']) * 100:.2f}%")
    print(f"False negative rate        : {float(metrics['false_negative_rate']) * 100:.2f}%")
    print("-" * 72)
    print(f"Saved dataset CSV          : {dataset_path}")
    print(f"Saved metrics JSON         : {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a saved ~100k CSV test dataset and evaluate the trained dual-head anomaly detector."
    )
    parser.add_argument("--n-calib", type=int, default=20000, help="Number of normal windows for threshold calibration.")
    parser.add_argument("--n-normal", type=int, default=50000, help="Number of normal test windows.")
    parser.add_argument("--n-anom", type=int, default=50000, help="Number of anomalous test windows.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Inference batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Path to save/load the CSV dataset.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON.")
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Reuse an existing saved CSV dataset instead of generating a new one.",
    )
    args = parser.parse_args()

    if args.reuse_dataset:
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"--reuse-dataset was set, but dataset file does not exist: {args.dataset_path}")
        dataset = load_large_scale_dataset(
            dataset_path=args.dataset_path,
            seed=args.seed,
            n_calib=args.n_calib,
            n_normal=args.n_normal,
            n_anom=args.n_anom,
        )
    else:
        dataset = create_large_scale_dataset(
            n_calib=args.n_calib,
            n_normal=args.n_normal,
            n_anom=args.n_anom,
            seed=args.seed,
        )
        save_large_scale_dataset(dataset, args.dataset_path)

    metrics = evaluate_saved_dataset(dataset=dataset, batch_size=args.batch_size)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_metrics(metrics, dataset_path=args.dataset_path, metrics_path=args.metrics_path)


if __name__ == "__main__":
    main()
