from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from config import RESULTS_DIR, TrainConfig
    from hybrid_scoring import combine_mode_scores, normalize_hybrid_weights, normalize_model_mode
    from model import FiLMAutoencoder
    from model_forecasting import ContextGRUForecaster
    from streaming_csv_train import StreamingCSVTrainConfig, train_streaming_csv_model
    from streaming_train import StreamingTrainConfig, train_streaming_model
    from utils import (
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
    from .hybrid_scoring import combine_mode_scores, normalize_hybrid_weights, normalize_model_mode
    from .model import FiLMAutoencoder
    from .model_forecasting import ContextGRUForecaster
    from .streaming_csv_train import StreamingCSVTrainConfig, train_streaming_csv_model
    from .streaming_train import StreamingTrainConfig, train_streaming_model
    from .utils import (
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
        bundle = {
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
        for name in ("X_forecast_train", "X_forecast_test", "y_forecast_train", "y_forecast_test"):
            path = dataset_path / f"{name}.npy"
            if path.exists():
                bundle[name] = np.load(path, mmap_mode="r")
        return bundle

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


class WindowTaskDataset(Dataset):
    def __init__(
        self,
        x_array: np.ndarray,
        c_array: np.ndarray,
        indices: np.ndarray,
        x_scaler: StandardScaler,
        c_scaler: StandardScaler,
        task: str,
        forecast_horizon: int,
    ) -> None:
        self.x_array = x_array
        self.c_array = c_array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_scaler = x_scaler
        self.c_scaler = c_scaler
        self.task = normalize_model_mode(task)
        self.forecast_horizon = max(1, int(forecast_horizon))

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        x_window = np.asarray(self.x_array[index], dtype=np.float32)
        c_vector = np.asarray(self.c_array[index], dtype=np.float32)
        x_scaled = self.x_scaler.transform(x_window)
        c_scaled = self.c_scaler.transform(c_vector.reshape(1, -1))[0].astype(np.float32)
        if self.task == "forecasting":
            if self.forecast_horizon >= int(x_scaled.shape[0]):
                raise ValueError(
                    f"forecast_horizon={self.forecast_horizon} must be smaller than window size={int(x_scaled.shape[0])}."
                )
            x_input = x_scaled[:-self.forecast_horizon]
            y_target = x_scaled[-self.forecast_horizon:]
        else:
            x_input = x_scaled
            y_target = x_scaled
        return (
            torch.as_tensor(x_input, dtype=torch.float32),
            torch.as_tensor(c_scaled, dtype=torch.float32),
            torch.as_tensor(y_target, dtype=torch.float32),
        )


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


def fit_scalers(
    train_x: np.ndarray,
    train_c: np.ndarray,
) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    x_scaler.fit(train_x.reshape(train_x.shape[0] * train_x.shape[1], train_x.shape[2]))

    c_scaler = StandardScaler()
    c_scaler.fit(train_c)
    return x_scaler, c_scaler


def make_task_dataloader(
    x_array: np.ndarray,
    c_array: np.ndarray,
    indices: np.ndarray,
    x_scaler: StandardScaler,
    c_scaler: StandardScaler,
    task: str,
    forecast_horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = WindowTaskDataset(
        x_array=x_array,
        c_array=c_array,
        indices=indices,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        task=task,
        forecast_horizon=forecast_horizon,
    )
    use_cuda = device.type == "cuda"
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
    model: torch.nn.Module,
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

    for x_batch, c_batch, y_batch in progress:
        x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
        c_batch = c_batch.to(device, non_blocking=device.type == "cuda")
        y_batch = y_batch.to(device, non_blocking=device.type == "cuda")

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predicted = model(x_batch, c_batch)
            loss = criterion(predicted, y_batch)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = int(x_batch.size(0))
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{(total_loss / max(1, total_examples)):.4f}")

    progress.close()
    return total_loss / max(1, total_examples)


def collect_scores_from_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    task_mode = normalize_model_mode(task)
    score_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, c_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            c_batch = c_batch.to(device, non_blocking=device.type == "cuda")
            y_batch = y_batch.to(device, non_blocking=device.type == "cuda")
            predicted = model(x_batch, c_batch).detach().cpu().numpy()
            y_true = y_batch.detach().cpu().numpy()
            if task_mode == "forecasting":
                feature_errors = np.mean((y_true - predicted) ** 2, axis=1)
            else:
                feature_errors = compute_feature_error_matrix(y_true, predicted)
            feature_rows.append(feature_errors.astype(np.float32, copy=False))
            score_rows.append(compute_window_scores(feature_errors).astype(np.float32, copy=False))

    if not score_rows:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    return (
        np.concatenate(score_rows, axis=0).astype(np.float32, copy=False),
        np.concatenate(feature_rows, axis=0).astype(np.float32, copy=False),
    )


@dataclass
class SingleModelTrainResult:
    checkpoint: dict[str, Any]
    history: pd.DataFrame
    best_val_loss: float
    val_scores: np.ndarray
    val_feature_errors: np.ndarray
    threshold: float
    history_path: Path
    loss_curve_path: Path


def train_single_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    checkpoint_meta: dict[str, Any],
) -> SingleModelTrainResult:
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
            phase=f"{model_name}_train",
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch_index=epoch,
            total_epochs=config.epochs,
            phase=f"{model_name}_val",
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
                "model_name": model_name,
            }
        )
        print(f"[{model_name}] Epoch [{epoch}/{config.epochs}] train={train_loss:.4f} val={val_loss:.4f}")

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

    val_scores, val_feature_errors = collect_scores_from_loader(
        model=model,
        loader=val_loader,
        task=model_name,
        device=device,
    )
    threshold = float(np.quantile(val_scores, config.threshold_quantile)) if len(val_scores) > 0 else 0.0

    history_df = pd.DataFrame(history_rows)
    history_path = ensure_directory(config.model_dir) / f"{model_name}_training_history.csv"
    history_df.to_csv(history_path, index=False)
    loss_curve_path = save_loss_curve(
        ensure_directory(RESULTS_DIR) / f"{model_name}_loss_curve.png",
        history_df,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        **checkpoint_meta,
    }
    return SingleModelTrainResult(
        checkpoint=checkpoint,
        history=history_df,
        best_val_loss=float(best_val_loss),
        val_scores=val_scores,
        val_feature_errors=val_feature_errors,
        threshold=threshold,
        history_path=history_path,
        loss_curve_path=loss_curve_path,
    )


def _prepare_training_views(dataset_bundle: dict[str, Any]) -> dict[str, Any]:
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
        if len(val_indices) == 0:
            val_indices = train_indices
        test_indices = np.arange(0, int(x_test.shape[0]), dtype=np.int64)
        return {
            "x_train_full": x_train_full,
            "c_train_full": c_train_full,
            "x_test": x_test,
            "c_test": c_test,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "train_rows": int(len(train_indices)),
            "val_rows": int(len(val_indices)),
            "test_rows": int(len(test_indices)),
        }

    x_all = dataset_bundle["X"]
    c_all = dataset_bundle["C"]
    metadata = dataset_bundle["metadata"]
    splits = split_by_metadata(x_all, c_all, metadata)
    x_train, c_train, train_meta = splits["train"]
    x_val, c_val, val_meta = splits["val"]
    x_test, c_test, test_meta = splits["test"]
    if len(x_val) == 0:
        x_val = x_train
        c_val = c_train
        val_meta = train_meta
    return {
        "x_train": np.asarray(x_train, dtype=np.float32),
        "c_train": np.asarray(c_train, dtype=np.float32),
        "x_val": np.asarray(x_val, dtype=np.float32),
        "c_val": np.asarray(c_val, dtype=np.float32),
        "x_test": np.asarray(x_test, dtype=np.float32),
        "c_test": np.asarray(c_test, dtype=np.float32),
        "train_rows": int(len(train_meta)),
        "val_rows": int(len(val_meta)),
        "test_rows": int(len(test_meta)),
    }


def train_model(config: TrainConfig | None = None) -> dict[str, Any]:
    config = config or TrainConfig()
    requested_mode = normalize_model_mode(config.model_mode)
    set_random_seed(config.random_seed)

    model_dir = ensure_directory(config.model_dir)
    dataset_bundle = load_processed_dataset(config.dataset_dir)
    feature_meta = dataset_bundle["feature_meta"]
    feature_columns = list(feature_meta["feature_columns"])
    context_columns = list(feature_meta["context_columns"])
    dataset_meta = dataset_bundle.get("dataset_meta") or {}

    device = torch.device(choose_device(config.device))
    train_views = _prepare_training_views(dataset_bundle)
    reference_windows = train_views.get("x_train_full", train_views.get("x_train"))
    window_size = int(dataset_meta.get("window_size", reference_windows.shape[1]))
    forecast_horizon = int(
        dataset_meta.get("forecast_horizon", feature_meta.get("forecast_horizon", config.forecast_horizon))
    )
    if forecast_horizon >= window_size:
        raise ValueError("forecast_horizon must be smaller than the dataset window size.")

    if "x_train_full" in train_views:
        x_scaler, c_scaler = fit_scalers_memmap(
            train_x=train_views["x_train_full"],
            train_c=train_views["c_train_full"],
            indices=train_views["train_indices"],
            batch_size=max(1, config.batch_size),
        )
    else:
        x_scaler, c_scaler = fit_scalers(
            train_x=train_views["x_train"],
            train_c=train_views["c_train"],
        )

    def build_loader(task: str, phase: str) -> DataLoader:
        if "x_train_full" in train_views:
            indices = train_views["train_indices"] if phase == "train" else train_views["val_indices"]
            return make_task_dataloader(
                x_array=train_views["x_train_full"],
                c_array=train_views["c_train_full"],
                indices=indices,
                x_scaler=x_scaler,
                c_scaler=c_scaler,
                task=task,
                forecast_horizon=forecast_horizon,
                batch_size=config.batch_size,
                shuffle=phase == "train",
                num_workers=config.num_workers,
                device=device,
            )

        x_array = train_views["x_train"] if phase == "train" else train_views["x_val"]
        c_array = train_views["c_train"] if phase == "train" else train_views["c_val"]
        indices = np.arange(0, int(x_array.shape[0]), dtype=np.int64)
        return make_task_dataloader(
            x_array=x_array,
            c_array=c_array,
            indices=indices,
            x_scaler=x_scaler,
            c_scaler=c_scaler,
            task=task,
            forecast_horizon=forecast_horizon,
            batch_size=config.batch_size,
            shuffle=phase == "train",
            num_workers=config.num_workers,
            device=device,
        )

    alpha, beta = normalize_hybrid_weights(config.alpha, config.beta)
    results: dict[str, SingleModelTrainResult] = {}
    train_reconstruction = requested_mode in {"reconstruction", "hybrid"}
    train_forecasting = requested_mode in {"forecasting", "hybrid"}

    if train_reconstruction:
        reconstruction_model = FiLMAutoencoder(
            window_size=window_size,
            n_features=int(len(feature_columns)),
            context_dim=int(len(context_columns)),
            units=config.units,
            latent=config.latent,
        ).to(device)
        results["reconstruction"] = train_single_model(
            model_name="reconstruction",
            model=reconstruction_model,
            train_loader=build_loader("reconstruction", "train"),
            val_loader=build_loader("reconstruction", "val"),
            config=config,
            device=device,
            checkpoint_meta={
                "model_type": "reconstruction",
                "window_size": window_size,
                "n_features": int(len(feature_columns)),
                "context_dim": int(len(context_columns)),
                "units": int(config.units),
                "latent": int(config.latent),
                "feature_columns": feature_columns,
                "context_columns": context_columns,
            },
        )

    if train_forecasting:
        forecasting_model = ContextGRUForecaster(
            input_window_size=int(dataset_meta.get("forecast_input_window_size", window_size - forecast_horizon)),
            n_features=int(len(feature_columns)),
            context_dim=int(len(context_columns)),
            hidden_size=int(config.forecast_hidden_size),
            num_layers=int(config.forecast_num_layers),
            forecast_horizon=int(forecast_horizon),
            dropout=float(config.forecast_dropout),
        ).to(device)
        results["forecasting"] = train_single_model(
            model_name="forecasting",
            model=forecasting_model,
            train_loader=build_loader("forecasting", "train"),
            val_loader=build_loader("forecasting", "val"),
            config=config,
            device=device,
            checkpoint_meta={
                "model_type": "forecasting",
                "window_size": window_size,
                "forecast_input_window_size": int(window_size - forecast_horizon),
                "forecast_horizon": int(forecast_horizon),
                "n_features": int(len(feature_columns)),
                "context_dim": int(len(context_columns)),
                "hidden_size": int(config.forecast_hidden_size),
                "num_layers": int(config.forecast_num_layers),
                "dropout": float(config.forecast_dropout),
                "feature_columns": feature_columns,
                "context_columns": context_columns,
            },
        )

    hybrid_threshold = None
    hybrid_val_scores = np.zeros((0,), dtype=np.float32)
    if "reconstruction" in results and "forecasting" in results:
        combined = combine_mode_scores(
            mode="hybrid",
            recon_feature_errors=results["reconstruction"].val_feature_errors,
            recon_scores=results["reconstruction"].val_scores,
            forecast_feature_errors=results["forecasting"].val_feature_errors,
            forecast_scores=results["forecasting"].val_scores,
            alpha=alpha,
            beta=beta,
        )
        hybrid_val_scores = combined.final_scores
        hybrid_threshold = float(np.quantile(hybrid_val_scores, config.threshold_quantile)) if len(hybrid_val_scores) > 0 else 0.0

    histories: list[pd.DataFrame] = []
    artifact_paths: dict[str, str] = {}
    if "reconstruction" in results:
        recon_result = results["reconstruction"]
        torch.save(recon_result.checkpoint, model_dir / "reconstruction_model.pt")
        torch.save(recon_result.checkpoint, model_dir / "film_ae.pt")
        recon_result.history.to_csv(model_dir / "training_history.csv", index=False)
        histories.append(recon_result.history)
        artifact_paths["reconstruction_model"] = str((model_dir / "reconstruction_model.pt").resolve())
        artifact_paths["legacy_reconstruction_model"] = str((model_dir / "film_ae.pt").resolve())
    if "forecasting" in results:
        forecast_result = results["forecasting"]
        torch.save(forecast_result.checkpoint, model_dir / "forecasting_model.pt")
        histories.append(forecast_result.history)
        artifact_paths["forecasting_model"] = str((model_dir / "forecasting_model.pt").resolve())
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(model_dir / "training_history_all.csv", index=False)

    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(c_scaler, model_dir / "c_scaler.joblib")

    selected_threshold = 0.0
    if requested_mode == "reconstruction" and "reconstruction" in results:
        selected_threshold = float(results["reconstruction"].threshold)
    elif requested_mode == "forecasting" and "forecasting" in results:
        selected_threshold = float(results["forecasting"].threshold)
    elif requested_mode == "hybrid" and hybrid_threshold is not None:
        selected_threshold = float(hybrid_threshold)
    elif "reconstruction" in results:
        selected_threshold = float(results["reconstruction"].threshold)
    elif "forecasting" in results:
        selected_threshold = float(results["forecasting"].threshold)

    detector_meta = {
        "default_mode": requested_mode,
        "available_modes": [mode for mode in ("reconstruction", "forecasting", "hybrid") if mode in results or (mode == "hybrid" and hybrid_threshold is not None)],
        "threshold": float(results["reconstruction"].threshold) if "reconstruction" in results else float(selected_threshold),
        "forecast_threshold": float(results["forecasting"].threshold) if "forecasting" in results else None,
        "hybrid_threshold": float(hybrid_threshold) if hybrid_threshold is not None else None,
        "threshold_quantile": float(config.threshold_quantile),
        "window_size": window_size,
        "forecast_horizon": int(forecast_horizon),
        "forecast_input_window_size": int(window_size - forecast_horizon),
        "n_features": int(len(feature_columns)),
        "context_dim": int(len(context_columns)),
        "score_mode": config.score_mode,
        "num_train_windows": int(train_views["train_rows"]),
        "num_val_windows": int(train_views["val_rows"]),
        "num_test_windows": int(train_views["test_rows"]),
        "alpha": float(alpha),
        "beta": float(beta),
        "modes": {
            "reconstruction": {
                "threshold": float(results["reconstruction"].threshold),
                "artifact_path": "reconstruction_model.pt",
                "best_val_loss": float(results["reconstruction"].best_val_loss),
                "val_score_mean": float(np.mean(results["reconstruction"].val_scores)),
                "val_score_std": float(np.std(results["reconstruction"].val_scores)),
            }
            if "reconstruction" in results
            else {},
            "forecasting": {
                "threshold": float(results["forecasting"].threshold),
                "artifact_path": "forecasting_model.pt",
                "best_val_loss": float(results["forecasting"].best_val_loss),
                "val_score_mean": float(np.mean(results["forecasting"].val_scores)),
                "val_score_std": float(np.std(results["forecasting"].val_scores)),
            }
            if "forecasting" in results
            else {},
            "hybrid": {
                "threshold": float(hybrid_threshold),
                "val_score_mean": float(np.mean(hybrid_val_scores)),
                "val_score_std": float(np.std(hybrid_val_scores)),
                "alpha": float(alpha),
                "beta": float(beta),
            }
            if hybrid_threshold is not None
            else {},
        },
    }
    joblib.dump(detector_meta, model_dir / "detector_meta.joblib")
    write_json(model_dir / "detector_meta.json", detector_meta)
    write_json(
        model_dir / "threshold_config.json",
        {
            "default_mode": requested_mode,
            "selected_threshold": float(selected_threshold),
            "alpha": float(alpha),
            "beta": float(beta),
            "modes": detector_meta["modes"],
        },
    )

    summary = {
        "model_dir": str(model_dir.resolve()),
        "artifact_paths": artifact_paths,
        "detector_meta": detector_meta,
        "train_config": config.to_dict(),
        "train_split_rows": int(train_views["train_rows"]),
        "val_split_rows": int(train_views["val_rows"]),
        "test_split_rows": int(train_views["test_rows"]),
        "history_paths": {
            mode: str(result.history_path.resolve())
            for mode, result in results.items()
        },
        "loss_curve_paths": {
            mode: str(result.loss_curve_path.resolve())
            for mode, result in results.items()
        },
    }
    write_json(model_dir / "training_summary.json", summary)

    if "reconstruction" in results and not results["reconstruction"].history.empty:
        print(
            "Final reconstruction training loss: "
            f"{float(results['reconstruction'].history['train_loss'].iloc[-1]):.4f}"
        )
    if "forecasting" in results and not results["forecasting"].history.empty:
        print(
            "Final forecasting training loss: "
            f"{float(results['forecasting'].history['train_loss'].iloc[-1]):.4f}"
        )
    return summary


train_research_pipeline = train_model
train_streaming_csv_pipeline = train_streaming_csv_model
train_streaming_pipeline = train_streaming_model
