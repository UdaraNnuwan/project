from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

try:
    from config import RESULTS_DIR, TrainConfig
    from model import FiLMAutoencoder
    from streaming_csv_train import StreamingCSVTrainConfig, train_streaming_csv_model
    from streaming_train import StreamingTrainConfig, train_streaming_model
    from utils import (
        apply_3d_scaler,
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        save_loss_curve,
        set_random_seed,
        write_json,
    )
except ImportError:
    from .config import RESULTS_DIR, TrainConfig
    from .model import FiLMAutoencoder
    from .streaming_csv_train import StreamingCSVTrainConfig, train_streaming_csv_model
    from .streaming_train import StreamingTrainConfig, train_streaming_model
    from .utils import (
        apply_3d_scaler,
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        save_loss_curve,
        set_random_seed,
        write_json,
    )


class ConcatenatedArrayView:
    def __init__(self, *arrays: np.ndarray) -> None:
        self.arrays = tuple(array for array in arrays if array is not None and int(array.shape[0]) > 0)
        if not self.arrays:
            self.shape = (0,)
            self.dtype = np.float32
            self._lengths = np.zeros(0, dtype=np.int64)
            return
        self._lengths = np.asarray([int(array.shape[0]) for array in self.arrays], dtype=np.int64)
        self.shape = (int(self._lengths.sum()), *self.arrays[0].shape[1:])
        self.dtype = self.arrays[0].dtype

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, item: Any) -> np.ndarray:
        if isinstance(item, slice):
            indices = np.arange(*item.indices(len(self)), dtype=np.int64)
            return self[indices]

        if isinstance(item, (list, tuple, np.ndarray)):
            indices = np.asarray(item)
            if indices.dtype == bool:
                indices = np.flatnonzero(indices)
            if indices.size == 0:
                return np.empty((0, *self.shape[1:]), dtype=self.dtype)
            return np.stack([np.asarray(self[int(index)]) for index in indices], axis=0)

        index = int(item)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")

        offset = index
        for array, length in zip(self.arrays, self._lengths):
            if offset < int(length):
                return np.asarray(array[offset])
            offset -= int(length)
        raise IndexError("index out of range")

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        materialized = np.concatenate([np.asarray(array) for array in self.arrays], axis=0)
        if dtype is not None:
            return materialized.astype(dtype)
        return materialized


def _split_metadata_frame(train_rows: int, test_rows: int) -> pd.DataFrame:
    total_rows = int(train_rows) + int(test_rows)
    split = np.concatenate(
        [
            np.full(int(train_rows), "train", dtype=object),
            np.full(int(test_rows), "test", dtype=object),
        ]
    )
    return pd.DataFrame(
        {
            "window_id": np.arange(total_rows, dtype=np.int64),
            "split": split,
        }
    )


def load_processed_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    split_paths = {
        "X_train": dataset_path / "X_train.npy",
        "X_test": dataset_path / "X_test.npy",
        "C_train": dataset_path / "C_train.npy",
        "C_test": dataset_path / "C_test.npy",
    }
    if all(path.exists() for path in split_paths.values()):
        x_train = np.load(split_paths["X_train"], mmap_mode="r")
        x_test = np.load(split_paths["X_test"], mmap_mode="r")
        c_train = np.load(split_paths["C_train"], mmap_mode="r")
        c_test = np.load(split_paths["C_test"], mmap_mode="r")
        return {
            "X_train": x_train,
            "X_test": x_test,
            "C_train": c_train,
            "C_test": c_test,
            "X": ConcatenatedArrayView(x_train, x_test),
            "C": ConcatenatedArrayView(c_train, c_test),
            "metadata": _split_metadata_frame(
                train_rows=int(x_train.shape[0]),
                test_rows=int(x_test.shape[0]),
            ),
            "feature_meta": joblib.load(dataset_path / "feature_meta.joblib"),
            "dataset_meta": joblib.load(dataset_path / "dataset_meta.joblib")
            if (dataset_path / "dataset_meta.joblib").exists()
            else None,
        }
    return {
        "X": np.load(dataset_path / "X_all.npy", allow_pickle=False),
        "C": np.load(dataset_path / "C_all.npy", allow_pickle=False),
        "metadata": pd.read_csv(dataset_path / "window_metadata.csv"),
        "feature_meta": joblib.load(dataset_path / "feature_meta.joblib"),
        "dataset_meta": joblib.load(dataset_path / "dataset_meta.joblib")
        if (dataset_path / "dataset_meta.joblib").exists()
        else None,
    }


def split_by_metadata(
    x_all: np.ndarray,
    c_all: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    splits: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split_name in ("train", "val", "test"):
        mask = metadata["split"].astype(str) == split_name
        indices = np.flatnonzero(mask.to_numpy())
        splits[split_name] = (
            x_all[indices],
            c_all[indices],
            metadata.iloc[indices].reset_index(drop=True),
        )
    return splits


class WindowMemmapDataset(Dataset):
    def __init__(
        self,
        x_array: np.ndarray,
        c_array: np.ndarray,
        indices: np.ndarray,
        x_scaler: StandardScaler,
        c_scaler: StandardScaler,
    ) -> None:
        self.x_array = x_array
        self.c_array = c_array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_scaler = x_scaler
        self.c_scaler = c_scaler

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        x_window = np.asarray(self.x_array[index], dtype=np.float32)
        c_vector = np.asarray(self.c_array[index], dtype=np.float32)
        x_scaled = self.x_scaler.transform(x_window)
        c_scaled = self.c_scaler.transform(c_vector.reshape(1, -1))[0]
        return (
            torch.as_tensor(x_scaled, dtype=torch.float32),
            torch.as_tensor(c_scaled, dtype=torch.float32),
        )


def fit_scalers(
    train_x: np.ndarray,
    train_c: np.ndarray,
) -> tuple[StandardScaler, StandardScaler]:
    num_windows, window_size, num_features = train_x.shape
    x_scaler = StandardScaler()
    x_scaler.fit(train_x.reshape(num_windows * window_size, num_features))

    c_scaler = StandardScaler()
    c_scaler.fit(train_c)
    return x_scaler, c_scaler


def fit_scalers_memmap(
    train_x: np.ndarray,
    train_c: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    c_scaler = StandardScaler()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        x_batch = np.asarray(train_x[batch_indices], dtype=np.float32)
        c_batch = np.asarray(train_c[batch_indices], dtype=np.float32)
        x_scaler.partial_fit(x_batch.reshape(-1, x_batch.shape[-1]))
        c_scaler.partial_fit(c_batch)
    return x_scaler, c_scaler


def make_dataloader(
    x_array: np.ndarray,
    c_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x_array, dtype=torch.float32),
        torch.as_tensor(c_array, dtype=torch.float32),
    )
    use_cuda = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


def make_memmap_dataloader(
    x_array: np.ndarray,
    c_array: np.ndarray,
    indices: np.ndarray,
    x_scaler: StandardScaler,
    c_scaler: StandardScaler,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = WindowMemmapDataset(
        x_array=x_array,
        c_array=c_array,
        indices=indices,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
    )
    use_cuda = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: FiLMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
    phase: str,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    progress = tqdm(
        loader,
        desc=f"{phase.title()} Epoch {epoch_index}/{total_epochs}",
        leave=False,
    )

    for x_batch, c_batch in progress:
        x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
        c_batch = c_batch.to(device, non_blocking=device.type == "cuda")

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            reconstructed = model(x_batch, c_batch)
            loss = criterion(reconstructed, x_batch)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = int(x_batch.size(0))
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{(total_loss / max(1, total_examples)):.4f}")

    progress.close()
    return total_loss / max(1, total_examples)


def reconstruct_dataset(
    model: FiLMAutoencoder,
    x_array: np.ndarray,
    c_array: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = make_dataloader(
        x_array=x_array,
        c_array=c_array,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        device=device,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            c_batch = c_batch.to(device, non_blocking=device.type == "cuda")
            predictions.append(model(x_batch, c_batch).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def fit_detection_threshold(
    model: FiLMAutoencoder,
    x_val_scaled: np.ndarray,
    c_val_scaled: np.ndarray,
    batch_size: int,
    quantile: float,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    val_reconstruction = reconstruct_dataset(
        model=model,
        x_array=x_val_scaled,
        c_array=c_val_scaled,
        device=device,
        batch_size=batch_size,
    )
    val_feature_errors = compute_feature_error_matrix(x_val_scaled, val_reconstruction)
    val_scores = compute_window_scores(val_feature_errors)
    threshold = float(np.quantile(val_scores, quantile))
    return threshold, val_scores


def fit_detection_threshold_from_loader(
    model: FiLMAutoencoder,
    loader: DataLoader,
    quantile: float,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            c_batch = c_batch.to(device, non_blocking=device.type == "cuda")
            reconstructed = model(x_batch, c_batch)
            feature_errors = compute_feature_error_matrix(
                x_batch.cpu().numpy(),
                reconstructed.cpu().numpy(),
            )
            scores.append(compute_window_scores(feature_errors))

    if not scores:
        return 0.0, np.zeros(0, dtype=np.float32)
    val_scores = np.concatenate(scores, axis=0).astype(np.float32)
    threshold = float(np.quantile(val_scores, quantile))
    return threshold, val_scores


def train_model(config: TrainConfig | None = None) -> dict[str, Any]:
    config = config or TrainConfig()
    set_random_seed(config.random_seed)

    model_dir = ensure_directory(config.model_dir)

    dataset_bundle = load_processed_dataset(config.dataset_dir)
    feature_meta = dataset_bundle["feature_meta"]

    device = torch.device(choose_device(config.device))
    if "X_train" in dataset_bundle:
        x_train_full = dataset_bundle["X_train"]
        c_train_full = dataset_bundle["C_train"]
        x_test = dataset_bundle["X_test"]
        c_test = dataset_bundle["C_test"]

        num_train_windows = int(x_train_full.shape[0])
        val_size = 0
        if num_train_windows > 1:
            val_size = max(1, int(round(num_train_windows * 0.1)))
            val_size = min(val_size, num_train_windows - 1)
        train_size = num_train_windows - val_size

        train_indices = np.arange(0, train_size, dtype=np.int64)
        val_indices = np.arange(train_size, num_train_windows, dtype=np.int64)
        test_indices = np.arange(0, int(x_test.shape[0]), dtype=np.int64)

        x_scaler, c_scaler = fit_scalers_memmap(
            train_x=x_train_full,
            train_c=c_train_full,
            indices=train_indices,
            batch_size=max(1, config.batch_size),
        )

        train_loader = make_memmap_dataloader(
            x_array=x_train_full,
            c_array=c_train_full,
            indices=train_indices,
            x_scaler=x_scaler,
            c_scaler=c_scaler,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            device=device,
        )
        val_loader = make_memmap_dataloader(
            x_array=x_train_full,
            c_array=c_train_full,
            indices=val_indices if len(val_indices) > 0 else train_indices,
            x_scaler=x_scaler,
            c_scaler=c_scaler,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            device=device,
        )

        model = FiLMAutoencoder(
            window_size=int(x_train_full.shape[1]),
            n_features=int(x_train_full.shape[2]),
            context_dim=int(c_train_full.shape[1]),
            units=config.units,
            latent=config.latent,
        ).to(device)
    else:
        x_all = dataset_bundle["X"]
        c_all = dataset_bundle["C"]
        metadata = dataset_bundle["metadata"]

        splits = split_by_metadata(x_all, c_all, metadata)
        x_train, c_train, train_meta = splits["train"]
        x_val, c_val, val_meta = splits["val"]
        x_test, c_test, test_meta = splits["test"]

        x_scaler, c_scaler = fit_scalers(x_train, c_train)
        x_train_scaled = apply_3d_scaler(x_scaler, x_train)
        x_val_scaled = apply_3d_scaler(x_scaler, x_val)
        x_test_scaled = apply_3d_scaler(x_scaler, x_test)
        c_train_scaled = c_scaler.transform(c_train).astype(np.float32)
        c_val_scaled = c_scaler.transform(c_val).astype(np.float32)
        c_test_scaled = c_scaler.transform(c_test).astype(np.float32)

        train_loader = make_dataloader(
            x_train_scaled,
            c_train_scaled,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            device=device,
        )
        val_loader = make_dataloader(
            x_val_scaled,
            c_val_scaled,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            device=device,
        )

        model = FiLMAutoencoder(
            window_size=x_train_scaled.shape[1],
            n_features=x_train_scaled.shape[2],
            context_dim=c_train_scaled.shape[1],
            units=config.units,
            latent=config.latent,
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_plateau_patience,
        factor=config.lr_plateau_factor,
    )

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history_rows: list[dict[str, Any]] = []
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        started = perf_counter()
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch,
            total_epochs=config.epochs,
            phase="train",
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch_index=epoch,
            total_epochs=config.epochs,
            phase="val",
        )
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        elapsed = perf_counter() - started
        improved = val_loss < best_val_loss
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "epoch_seconds": elapsed,
                "improved": improved,
            }
        )
        print(
            f"Epoch [{epoch}/{config.epochs}] - Loss: {train_loss:.4f}"
            + (f" - Val Loss: {val_loss:.4f}" if val_loss == val_loss else "")
        )

        if improved:
            best_val_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if "X_train" in dataset_bundle:
        threshold, val_scores = fit_detection_threshold_from_loader(
            model=model,
            loader=val_loader,
            quantile=config.threshold_quantile,
            device=device,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "window_size": int(x_train_full.shape[1]),
            "n_features": int(x_train_full.shape[2]),
            "context_dim": int(c_train_full.shape[1]),
            "units": int(config.units),
            "latent": int(config.latent),
            "feature_columns": feature_meta["feature_columns"],
            "context_columns": feature_meta["context_columns"],
        }
        train_rows = int(len(train_indices))
        val_rows = int(len(val_indices))
        test_rows = int(len(test_indices))
    else:
        threshold, val_scores = fit_detection_threshold(
            model=model,
            x_val_scaled=x_val_scaled,
            c_val_scaled=c_val_scaled,
            batch_size=config.batch_size,
            quantile=config.threshold_quantile,
            device=device,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "window_size": int(x_train_scaled.shape[1]),
            "n_features": int(x_train_scaled.shape[2]),
            "context_dim": int(c_train_scaled.shape[1]),
            "units": int(config.units),
            "latent": int(config.latent),
            "feature_columns": feature_meta["feature_columns"],
            "context_columns": feature_meta["context_columns"],
        }
        train_rows = int(len(train_meta))
        val_rows = int(len(val_meta))
        test_rows = int(len(test_meta))

    history_df = pd.DataFrame(history_rows)
    history_path = model_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    loss_curve_path = save_loss_curve(RESULTS_DIR / "loss_curve.png", history_df)

    detector_meta = {
        "threshold": threshold,
        "threshold_quantile": float(config.threshold_quantile),
        "window_size": int(checkpoint["window_size"]),
        "n_features": int(checkpoint["n_features"]),
        "context_dim": int(checkpoint["context_dim"]),
        "score_mode": config.score_mode,
        "num_train_windows": train_rows,
        "num_val_windows": val_rows,
        "num_test_windows": test_rows,
        "best_val_loss": float(best_val_loss),
        "val_score_mean": float(np.mean(val_scores)),
        "val_score_std": float(np.std(val_scores)),
        "test_split_rows": test_rows,
    }

    torch.save(checkpoint, model_dir / "film_ae.pt")
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(c_scaler, model_dir / "c_scaler.joblib")
    joblib.dump(detector_meta, model_dir / "detector_meta.joblib")
    write_json(model_dir / "detector_meta.json", detector_meta)

    summary = {
        "model_dir": str(model_dir.resolve()),
        "history_path": str(history_path.resolve()),
        "loss_curve_path": str(loss_curve_path.resolve()),
        "model_path": str((model_dir / "film_ae.pt").resolve()),
        "detector_meta": detector_meta,
        "train_config": config.to_dict(),
        "train_split_rows": train_rows,
        "val_split_rows": val_rows,
        "test_split_rows": test_rows,
    }
    write_json(model_dir / "training_summary.json", summary)
    if not history_df.empty:
        final_train_loss = float(history_df["train_loss"].iloc[-1])
        print(f"Final training loss: {final_train_loss:.4f}")
        print(f"Saved loss curve to: {loss_curve_path.resolve()}")
    return summary


train_research_pipeline = train_model
train_streaming_csv_pipeline = train_streaming_csv_model
train_streaming_pipeline = train_streaming_model
