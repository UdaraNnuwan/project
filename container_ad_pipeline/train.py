from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .dataset import DatasetBundle
from .model import FiLMAutoencoder, build_model_from_checkpoint
from .utils import flatten_windows, reshape_windows, save_json, set_random_seed


@dataclass
class TrainedArtifacts:
    model: FiLMAutoencoder
    checkpoint: dict[str, Any]
    history: pd.DataFrame
    detector_meta: dict[str, Any]
    x_scaler: RobustScaler
    c_scaler: StandardScaler


def select_split(bundle: DatasetBundle, split_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mask = bundle.metadata["split"].to_numpy() == split_name
    return bundle.X[mask], bundle.C[mask], bundle.metadata.loc[mask].reset_index(drop=True)


def fit_scalers(X_train: np.ndarray, C_train: np.ndarray) -> tuple[RobustScaler, StandardScaler]:
    x_scaler = RobustScaler(quantile_range=(25, 75))
    c_scaler = StandardScaler()

    x_scaler.fit(flatten_windows(X_train))
    c_scaler.fit(C_train)
    return x_scaler, c_scaler


def transform_inputs(X: np.ndarray, C: np.ndarray, x_scaler: RobustScaler, c_scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    n_samples, window_size, n_features = X.shape
    X_scaled = reshape_windows(x_scaler.transform(flatten_windows(X)), window_size, n_features).astype(np.float32)
    C_scaled = c_scaler.transform(C).astype(np.float32)
    return X_scaled, C_scaled


def create_data_loader(X: np.ndarray, C: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(C))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def infer_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reconstruction_feature_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return np.mean((x_true - x_pred) ** 2, axis=1)


def reconstruction_scores(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return reconstruction_feature_errors(x_true, x_pred).mean(axis=1)


def fit_threshold_on_validation(scores: np.ndarray, quantile: float = 0.995) -> float:
    return float(np.quantile(scores, quantile))


def train_film_autoencoder(
    bundle: DatasetBundle,
    train_config: TrainConfig,
    checkpoint_path: str | Path,
    x_scaler_path: str | Path,
    c_scaler_path: str | Path,
    detector_meta_json: str | Path,
    detector_meta_joblib: str | Path,
) -> TrainedArtifacts:
    set_random_seed(train_config.random_seed)
    device = infer_device(train_config.device)

    X_train, C_train, train_meta = select_split(bundle, "train")
    X_val, C_val, val_meta = select_split(bundle, "val")

    x_scaler, c_scaler = fit_scalers(X_train, C_train)
    X_train_scaled, C_train_scaled = transform_inputs(X_train, C_train, x_scaler, c_scaler)
    X_val_scaled, C_val_scaled = transform_inputs(X_val, C_val, x_scaler, c_scaler)

    train_loader = create_data_loader(X_train_scaled, C_train_scaled, train_config.batch_size, True, train_config.num_workers)
    val_loader = create_data_loader(X_val_scaled, C_val_scaled, train_config.batch_size, False, train_config.num_workers)

    model = FiLMAutoencoder(
        window_size=bundle.dataset_meta["window_size"],
        n_features=bundle.dataset_meta["num_features"],
        context_dim=bundle.dataset_meta["context_dim"],
        units=train_config.units,
        latent=train_config.latent,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    criterion = torch.nn.MSELoss()

    history_rows: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_losses = []
        for x_batch, c_batch in train_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(x_batch, c_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                reconstructed = model(x_batch, c_batch)
                val_losses.append(float(criterion(reconstructed, x_batch).detach().cpu().item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    val_predictions = predict_reconstructions(model, X_val_scaled, C_val_scaled, device=device)
    val_scores = reconstruction_scores(X_val_scaled, val_predictions)
    threshold = fit_threshold_on_validation(val_scores, train_config.threshold_quantile)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "window_size": int(bundle.dataset_meta["window_size"]),
        "n_features": int(bundle.dataset_meta["num_features"]),
        "context_dim": int(bundle.dataset_meta["context_dim"]),
        "units": int(train_config.units),
        "latent": int(train_config.latent),
        "feature_columns": bundle.feature_meta["feature_columns"],
        "context_columns": bundle.feature_meta["context_columns"],
    }

    detector_meta = {
        "threshold": threshold,
        "threshold_quantile": train_config.threshold_quantile,
        "window_size": int(bundle.dataset_meta["window_size"]),
        "n_features": int(bundle.dataset_meta["num_features"]),
        "context_dim": int(bundle.dataset_meta["context_dim"]),
        "score_mode": "mean_feature_mse",
        "num_train_windows": int(len(train_meta)),
        "num_val_windows": int(len(val_meta)),
        "best_val_loss": best_val_loss,
    }

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(c_scaler, c_scaler_path)
    joblib.dump(detector_meta, detector_meta_joblib)
    save_json(Path(detector_meta_json), detector_meta)

    return TrainedArtifacts(
        model=model,
        checkpoint=checkpoint,
        history=pd.DataFrame(history_rows),
        detector_meta=detector_meta,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
    )


def predict_reconstructions(
    model: FiLMAutoencoder,
    X_scaled: np.ndarray,
    C_scaled: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    loader = create_data_loader(X_scaled, C_scaled, batch_size, False, 0)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, c_batch in loader:
            reconstructed = model(x_batch.to(device), c_batch.to(device))
            outputs.append(reconstructed.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def load_trained_model(checkpoint_path: str | Path, device: str = "cpu") -> FiLMAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = build_model_from_checkpoint(checkpoint)
        model.to(device)
        model.eval()
        return model
    raise ValueError("Unsupported checkpoint format.")
