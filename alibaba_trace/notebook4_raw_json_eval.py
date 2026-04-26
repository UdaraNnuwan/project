from __future__ import annotations

import argparse
import gzip
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

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
from alibaba_trace.data.dataset import ALIBABA_V2018_COLS
from alibaba_trace.data.utils import count_csv_rows_in_tar_gz
from alibaba_trace.model_architecture import load_dual_head_model, load_model


OUTPUTS_DIR = PACKAGE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_HEAD_MODEL_PATH = OUTPUTS_DIR / "model.pt"
DUAL_HEAD_MODEL_PATH = OUTPUTS_DIR / "dual_head_model.pt"
SCALER_PATH = OUTPUTS_DIR / "scaler_params.json"
ROW_COUNT_CACHE_PATH = OUTPUTS_DIR / "raw_tar_row_count_cache.json"

DEFAULT_DATASET_PATH = OUTPUTS_DIR / "notebook4_raw_tar_dataset_for_100k_eval.json.gz"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "notebook4_raw_tar_eval_metrics.json"

WINDOW = int(WINDOW_SIZE)
N_TS_FEAT = len(FEATURE_COLS)
N_META_FEAT = len(META_COLS)
LATENT_DIM = 64
LSTM_UNITS = (128, 64)
DROPOUT = 0.2
CHUNK_SIZE = 5000
TRAIN_RATIO = 0.85
SENTINEL_TOTAL_ROWS = 1_000_000_000
TRAIN_MAX_CHUNKS = 50
ROW_START_AFTER_TRAIN = TRAIN_MAX_CHUNKS * CHUNK_SIZE
LARGE_PRIME = 999_983

I_CPU = FEATURE_COLS.index("cpu_util_percent")
I_MEM = FEATURE_COLS.index("mem_util_percent")
I_CPUREQ = FEATURE_COLS.index("cpu_request")
I_MEMREQ = FEATURE_COLS.index("mem_request")
I_NETIN = FEATURE_COLS.index("net_in")
I_NETOUT = FEATURE_COLS.index("net_out")
I_DISK = FEATURE_COLS.index("disk_io_percent")


class ReconOnlyModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def eval(self) -> "ReconOnlyModel":
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


def _clip_scaled(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def default_tar_path() -> Path:
    candidates = [
        Path(r"C:\Users\kaspe\Desktop\Project\dataset\data\container_usage.tar.gz"),
        PROJECT_ROOT.parent / "dataset" / "data" / "container_usage.tar.gz",
        PROJECT_ROOT / "data" / "container_usage.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "container_usage.tar.gz not found in the expected locations. "
        "Pass --tar-path explicitly."
    )


def load_scaler_params() -> tuple[np.ndarray, np.ndarray]:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler params not found: {SCALER_PATH}. "
            "Run the training notebook first so evaluation uses the same scaling."
        )
    params = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    return (
        np.asarray(params["min_"], dtype=np.float32),
        np.asarray(params["max_"], dtype=np.float32),
    )


def scale_raw_windows(windows_raw: np.ndarray, min_arr: np.ndarray, max_arr: np.ndarray) -> np.ndarray:
    eps = np.float32(1e-8)
    range_arr = (max_arr - min_arr).astype(np.float32) + eps
    scaled = (windows_raw.astype(np.float32) - min_arr[None, None, :]) / range_arr[None, None, :]
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


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
            f"No checkpoint found. Expected {SINGLE_HEAD_MODEL_PATH} or {DUAL_HEAD_MODEL_PATH}."
        )
    print(f"Loaded reconstruction scoring model from: {source}")
    return ReconOnlyModel(model).eval()


def load_cached_row_count(tar_path: Path, csv_member_name: str) -> int | None:
    if not ROW_COUNT_CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(ROW_COUNT_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    key = f"{tar_path.resolve()}::{csv_member_name}"
    value = cache.get(key)
    return int(value) if value is not None else None


def save_cached_row_count(tar_path: Path, csv_member_name: str, row_count: int) -> None:
    if ROW_COUNT_CACHE_PATH.exists():
        try:
            cache = json.loads(ROW_COUNT_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    else:
        cache = {}
    key = f"{tar_path.resolve()}::{csv_member_name}"
    cache[key] = int(row_count)
    ROW_COUNT_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_total_rows(tar_path: Path, csv_member_name: str) -> int:
    cached = load_cached_row_count(tar_path, csv_member_name)
    if cached is not None:
        print(f"Using cached row count for raw archive: {cached:,}")
        return cached

    total_rows = count_csv_rows_in_tar_gz(str(tar_path), csv_member_name)
    save_cached_row_count(tar_path, csv_member_name, total_rows)
    return total_rows


def build_raw_test_source_dataset(
    dataset_path: Path,
    tar_path: Path,
    csv_member_name: str,
    n_calib: int,
    n_test_normal: int,
    n_test_anom_source: int,
    split_mode: str,
) -> Path:
    total_needed = n_calib + n_test_normal + n_test_anom_source
    min_arr, max_arr = load_scaler_params()
    total_rows = None
    if split_mode == "exact":
        total_rows = get_total_rows(tar_path, csv_member_name)
        train_row_limit = int(total_rows * TRAIN_RATIO)
        split_note = (
            "Exact chronological split using the full tar.gz row count. "
            "This is slower but uses the real archive boundary."
        )
    elif split_mode == "heuristic":
        train_row_limit = int(SENTINEL_TOTAL_ROWS * TRAIN_RATIO)
        split_note = (
            "Training-notebook-compatible heuristic split. "
            "Held-out windows start when row_cursor reaches 850,000,000 "
            "(train_ratio=0.85 over sentinel_total_rows=1,000,000,000)."
        )
    elif split_mode == "post_train_chunks":
        train_row_limit = ROW_START_AFTER_TRAIN
        split_note = (
            "Post-training raw split. The current checkpoint was trained with MAX_CHUNKS=50, "
            "so rows 0..249,999 were used for optimization. This benchmark starts immediately "
            "after that boundary to avoid overlap with the actual training windows."
        )
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    windows_raw: list[list[list[float]]] = []
    film_meta: list[list[float]] = []
    container_ids: list[str] = []
    machine_ids: list[str] = []
    time_stamp_end: list[str] = []

    raw_buffer: list[np.ndarray] = []
    meta_buffer: list[np.ndarray] = []
    container_buffer: list[np.ndarray] = []
    machine_buffer: list[np.ndarray] = []
    time_buffer: list[np.ndarray] = []

    row_cursor = 0
    with tarfile.open(tar_path, mode="r:gz") as tar:
        member = tar.getmember(csv_member_name)
        fobj = tar.extractfile(member)
        if fobj is None:
            raise RuntimeError(f"Could not open '{csv_member_name}' inside '{tar_path}'.")

        for chunk in pd.read_csv(
            fobj,
            chunksize=CHUNK_SIZE,
            names=ALIBABA_V2018_COLS,
            header=None,
            low_memory=True,
            na_values=["", "NA", "N/A", "nan", "NaN", "null", "NULL", "-1"],
        ):
            n_rows = len(chunk)
            chunk_start = row_cursor
            chunk_end = row_cursor + n_rows
            row_cursor = chunk_end

            if chunk_end <= train_row_limit:
                continue

            offset = max(0, train_row_limit - chunk_start)
            if offset > 0:
                chunk = chunk.iloc[offset:].copy()

            if chunk.empty:
                continue

            chunk.ffill(inplace=True)
            chunk.bfill(inplace=True)
            fill_dict = {c: 0.0 for c in FEATURE_COLS if c in chunk.columns}
            for mc in META_COLS:
                if mc in chunk.columns:
                    fill_dict[mc] = "missing"
            if "time_stamp" in chunk.columns:
                fill_dict["time_stamp"] = "missing"
            chunk.fillna(value=fill_dict, inplace=True)

            raw_arr = chunk.reindex(columns=FEATURE_COLS, fill_value=0.0).to_numpy(dtype=np.float32)
            container_arr = chunk["container_id"].astype(str).to_numpy(dtype=object)
            machine_arr = chunk["machine_id"].astype(str).to_numpy(dtype=object)
            time_arr = chunk["time_stamp"].astype(str).to_numpy(dtype=object)

            meta_arr = np.stack(
                [compute_film_vector(container_id, machine_id) for container_id, machine_id in zip(container_arr, machine_arr)],
                axis=0,
            ).astype(np.float32)

            raw_buffer.append(raw_arr)
            meta_buffer.append(meta_arr)
            container_buffer.append(container_arr)
            machine_buffer.append(machine_arr)
            time_buffer.append(time_arr)

            raw_full = np.concatenate(raw_buffer, axis=0)
            meta_full = np.concatenate(meta_buffer, axis=0)
            container_full = np.concatenate(container_buffer, axis=0)
            machine_full = np.concatenate(machine_buffer, axis=0)
            time_full = np.concatenate(time_buffer, axis=0)

            i = 0
            while i + WINDOW <= len(raw_full) and len(windows_raw) < total_needed:
                end_idx = i + WINDOW - 1
                windows_raw.append(np.round(raw_full[i : i + WINDOW], 6).tolist())
                film_meta.append(np.round(meta_full[end_idx], 8).tolist())
                container_ids.append(str(container_full[end_idx]))
                machine_ids.append(str(machine_full[end_idx]))
                time_stamp_end.append(str(time_full[end_idx]))
                i += 10

            tail_start = max(0, i - (WINDOW - 1))
            raw_buffer = [raw_full[tail_start:]]
            meta_buffer = [meta_full[tail_start:]]
            container_buffer = [container_full[tail_start:]]
            machine_buffer = [machine_full[tail_start:]]
            time_buffer = [time_full[tail_start:]]

            if len(windows_raw) >= total_needed:
                break

    if len(windows_raw) < total_needed:
        raise RuntimeError(
            f"Only collected {len(windows_raw):,} held-out windows but {total_needed:,} were required."
        )

    payload = {
        "meta": {
            "note": "Held-out raw windows from Alibaba container_usage.tar.gz test split only. "
                    "Notebook 4 injects anomalies at evaluation time. No train-split windows included.",
            "split_note": split_note,
            "split_mode": split_mode,
            "tar_path": str(tar_path),
            "csv_member_name": csv_member_name,
            "train_ratio_excluded": TRAIN_RATIO,
            "total_rows": int(total_rows) if total_rows is not None else None,
            "train_row_limit": int(train_row_limit),
            "sentinel_total_rows": SENTINEL_TOTAL_ROWS if split_mode == "heuristic" else None,
            "train_max_chunks": TRAIN_MAX_CHUNKS if split_mode == "post_train_chunks" else None,
            "window_size": WINDOW,
            "stride": 10,
            "feature_columns": FEATURE_COLS,
            "meta_columns": META_COLS,
            "scaler_min": np.round(min_arr, 8).tolist(),
            "scaler_max": np.round(max_arr, 8).tolist(),
            "n_calibration": int(n_calib),
            "n_test_normal": int(n_test_normal),
            "n_test_anomaly_source": int(n_test_anom_source),
            "n_source_windows_total": int(total_needed),
            "n_evaluation_windows_total": int(n_test_normal + n_test_anom_source),
        },
        "windows_raw": windows_raw,
        "film_meta": film_meta,
        "container_ids": container_ids,
        "machine_ids": machine_ids,
        "time_stamp_end": time_stamp_end,
    }

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.suffix == ".gz":
        with gzip.open(dataset_path, mode="wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
    else:
        dataset_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    return dataset_path


def load_raw_json_dataset(dataset_path: Path) -> dict[str, Any]:
    if dataset_path.suffix == ".gz":
        with gzip.open(dataset_path, mode="rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def inject_scaled_anomaly(window_scaled: np.ndarray, idx: int) -> tuple[np.ndarray, str, list[str]]:
    window_scaled = window_scaled.copy()
    pattern = idx % 3

    if pattern == 0:
        cpu_tail = np.linspace(0.92, 0.99, 20, dtype=np.float32)
        net_tail = np.linspace(0.88, 0.97, 20, dtype=np.float32)
        window_scaled[-20:, I_CPU] = _clip_scaled(np.maximum(window_scaled[-20:, I_CPU], cpu_tail))
        window_scaled[-20:, I_NETIN] = _clip_scaled(np.maximum(window_scaled[-20:, I_NETIN], net_tail))
        window_scaled[-20:, I_NETOUT] = _clip_scaled(np.maximum(window_scaled[-20:, I_NETOUT], net_tail[::-1]))
        window_scaled[-20:, I_CPUREQ] = _clip_scaled(np.maximum(window_scaled[-20:, I_CPUREQ], window_scaled[-20:, I_CPU] + 0.03))
        return window_scaled, "cpu_net_spike", ["cpu_util_percent", "net_in", "net_out", "cpu_request"]

    if pattern == 1:
        ramp = np.linspace(0.0, 0.20, 25, dtype=np.float32)
        window_scaled[-25:, I_MEM] = _clip_scaled(np.maximum(window_scaled[-25:, I_MEM], 0.78 + ramp))
        window_scaled[-25:, I_DISK] = _clip_scaled(np.maximum(window_scaled[-25:, I_DISK], 0.74 + ramp))
        window_scaled[-25:, I_MEMREQ] = _clip_scaled(np.maximum(window_scaled[-25:, I_MEMREQ], window_scaled[-25:, I_MEM] + 0.04))
        return window_scaled, "memory_disk_ramp", ["mem_util_percent", "disk_io_percent", "mem_request"]

    net_out_tail = np.linspace(0.90, 0.99, 20, dtype=np.float32)
    cpu_tail = np.linspace(0.82, 0.93, 20, dtype=np.float32)
    window_scaled[-20:, I_NETOUT] = _clip_scaled(np.maximum(window_scaled[-20:, I_NETOUT], net_out_tail))
    window_scaled[-20:, I_NETIN] = _clip_scaled(np.maximum(window_scaled[-20:, I_NETIN], 0.75 + np.linspace(0.0, 0.12, 20, dtype=np.float32)))
    window_scaled[-20:, I_CPU] = _clip_scaled(np.maximum(window_scaled[-20:, I_CPU], cpu_tail))
    return window_scaled, "network_exfiltration", ["net_out", "net_in", "cpu_util_percent"]


def batched_recon_mse(
    model: ReconOnlyModel,
    x_np: np.ndarray,
    meta_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
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


def evaluate_raw_json_dataset(dataset_path: Path, batch_size: int) -> dict[str, Any]:
    payload = load_raw_json_dataset(dataset_path)
    meta = payload["meta"]

    windows_raw = np.asarray(payload["windows_raw"], dtype=np.float32)
    film_meta = np.asarray(payload["film_meta"], dtype=np.float32)
    container_ids = np.asarray(payload["container_ids"], dtype=object)
    machine_ids = np.asarray(payload["machine_ids"], dtype=object)
    time_stamp_end = np.asarray(payload["time_stamp_end"], dtype=object)

    n_calib = int(meta["n_calibration"])
    n_test_normal = int(meta["n_test_normal"])
    n_test_anom_source = int(meta["n_test_anomaly_source"])

    min_arr = np.asarray(meta["scaler_min"], dtype=np.float32)
    max_arr = np.asarray(meta["scaler_max"], dtype=np.float32)
    windows_scaled = scale_raw_windows(windows_raw, min_arr, max_arr)

    calib_x = windows_scaled[:n_calib]
    calib_meta = film_meta[:n_calib]

    normal_x = windows_scaled[n_calib : n_calib + n_test_normal]
    normal_meta = film_meta[n_calib : n_calib + n_test_normal]

    anom_src_start = n_calib + n_test_normal
    anom_src_end = anom_src_start + n_test_anom_source
    anom_source_x = windows_scaled[anom_src_start:anom_src_end]
    anom_source_meta = film_meta[anom_src_start:anom_src_end]

    anomaly_x = np.empty_like(anom_source_x)
    anomaly_types: list[str] = []
    anomaly_feature_tags: list[list[str]] = []
    for idx in range(len(anom_source_x)):
        injected, inj_name, inj_features = inject_scaled_anomaly(anom_source_x[idx], idx)
        anomaly_x[idx] = injected
        anomaly_types.append(inj_name)
        anomaly_feature_tags.append(inj_features)

    x_test = np.concatenate([normal_x, anomaly_x], axis=0)
    c_test = np.concatenate([normal_meta, anom_source_meta], axis=0)
    y_test = np.concatenate(
        [
            np.zeros(len(normal_x), dtype=np.int64),
            np.ones(len(anomaly_x), dtype=np.int64),
        ],
        axis=0,
    )

    normal_src_container_ids = container_ids[n_calib : n_calib + n_test_normal]
    normal_src_machine_ids = machine_ids[n_calib : n_calib + n_test_normal]
    normal_src_timestamps = time_stamp_end[n_calib : n_calib + n_test_normal]

    anom_src_container_ids = container_ids[anom_src_start:anom_src_end]
    anom_src_machine_ids = machine_ids[anom_src_start:anom_src_end]
    anom_src_timestamps = time_stamp_end[anom_src_start:anom_src_end]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_reconstruction_model(device)

    calib_scores = batched_recon_mse(model, calib_x, calib_meta, device, batch_size)
    threshold_p95 = float(np.percentile(calib_scores, 95.0))
    threshold_p99 = float(np.percentile(calib_scores, 99.0))

    test_scores = batched_recon_mse(model, x_test, c_test, device, batch_size)
    y_pred = (test_scores >= threshold_p95).astype(np.int64)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()]

    normal_scores = test_scores[y_test == 0]
    anomaly_scores = test_scores[y_test == 1]

    false_positive_examples = []
    for idx in np.where((y_test == 0) & (y_pred == 1))[0][:5]:
        false_positive_examples.append(
            {
                "container_id": str(normal_src_container_ids[idx]),
                "machine_id": str(normal_src_machine_ids[idx]),
                "time_stamp_end": str(normal_src_timestamps[idx]),
                "score": float(test_scores[idx]),
            }
        )

    false_negative_examples = []
    anom_offset = len(normal_x)
    for local_idx in np.where((y_test[anom_offset:] == 1) & (y_pred[anom_offset:] == 0))[0][:5]:
        global_idx = anom_offset + int(local_idx)
        false_negative_examples.append(
            {
                "container_id": str(anom_src_container_ids[local_idx]),
                "machine_id": str(anom_src_machine_ids[local_idx]),
                "time_stamp_end": str(anom_src_timestamps[local_idx]),
                "injection_type": anomaly_types[local_idx],
                "injected_features": anomaly_feature_tags[local_idx],
                "score": float(test_scores[global_idx]),
            }
        )

    return {
        "note": "Notebook 4 raw tar.gz benchmark using held-out test-split windows only. "
                "Calibration windows are untouched normals from the held-out split. "
                "Anomalies are injected at evaluation time into disjoint held-out source windows.",
        "dataset_path": str(dataset_path),
        "tar_path": str(meta["tar_path"]),
        "csv_member_name": meta["csv_member_name"],
        "split_mode": meta.get("split_mode"),
        "split_note": meta.get("split_note"),
        "train_ratio_excluded": float(meta["train_ratio_excluded"]),
        "n_source_windows_total": int(meta["n_source_windows_total"]),
        "n_calibration_normal": n_calib,
        "n_test_total": int(len(x_test)),
        "n_test_normal": int(len(normal_x)),
        "n_test_anomalous": int(len(anomaly_x)),
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
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Notebook 4 benchmark from held-out raw Alibaba tar.gz test windows and evaluate it."
    )
    parser.add_argument("--tar-path", type=Path, default=None)
    parser.add_argument("--csv-member", type=str, default="container_usage.csv")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--n-calib", type=int, default=20_000)
    parser.add_argument("--n-normal", type=int, default=50_000)
    parser.add_argument("--n-anom", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--split-mode",
        choices=["post_train_chunks", "heuristic", "exact"],
        default="post_train_chunks",
    )
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args()

    tar_path = args.tar_path.resolve() if args.tar_path is not None else default_tar_path()

    if not args.reuse_dataset:
        built_path = build_raw_test_source_dataset(
            dataset_path=args.dataset_path,
            tar_path=tar_path,
            csv_member_name=args.csv_member,
            n_calib=args.n_calib,
            n_test_normal=args.n_normal,
            n_test_anom_source=args.n_anom,
            split_mode=args.split_mode,
        )
        print(f"Saved raw held-out JSON benchmark source -> {built_path}")
    else:
        print(f"Reusing existing raw held-out JSON benchmark source -> {args.dataset_path}")

    metrics = evaluate_raw_json_dataset(args.dataset_path, batch_size=args.batch_size)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Notebook 4 Raw tar.gz Held-Out Benchmark")
    print("=" * 72)
    print(metrics["note"])
    print(f"Source windows : {metrics['n_source_windows_total']:,}")
    print(f"Test total     : {metrics['n_test_total']:,}")
    print(f"Normal         : {metrics['n_test_normal']:,}")
    print(f"Anomalous      : {metrics['n_test_anomalous']:,}")
    print(f"Threshold P95  : {metrics['threshold_p95']:.6f}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision      : {metrics['precision']:.4f}  ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall         : {metrics['recall']:.4f}  ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-score       : {metrics['f1_score']:.4f}  ({metrics['f1_score'] * 100:.2f}%)")
    print(f"TN / FP / FN / TP : {metrics['tn']} / {metrics['fp']} / {metrics['fn']} / {metrics['tp']}")
    print(f"Saved dataset  : {args.dataset_path}")
    print(f"Saved metrics  : {args.metrics_path}")


if __name__ == "__main__":
    main()
