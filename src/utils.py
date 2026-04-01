from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    path_obj = Path(path)
    ensure_directory(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, ensure_ascii=False)
    return path_obj


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_random_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def apply_3d_scaler(scaler: Any, array: np.ndarray) -> np.ndarray:
    n_samples, window_size, n_features = array.shape
    transformed = scaler.transform(array.reshape(n_samples * window_size, n_features))
    return transformed.reshape(n_samples, window_size, n_features).astype(np.float32)


def compute_feature_error_matrix(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return np.mean((x_true - x_pred) ** 2, axis=1)


def compute_window_scores(feature_error_matrix: np.ndarray) -> np.ndarray:
    return np.mean(feature_error_matrix, axis=1)


def top_feature_summary(
    feature_errors: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 5,
) -> tuple[list[list[str]], list[list[float]], list[list[int]]]:
    top_k = max(1, min(int(top_k), len(feature_names)))
    ranked_indices = np.argsort(feature_errors, axis=1)[:, ::-1]

    top_names: list[list[str]] = []
    top_scores: list[list[float]] = []
    full_ranks: list[list[int]] = []
    for row_idx in range(feature_errors.shape[0]):
        order = ranked_indices[row_idx]
        full_ranks.append(order.astype(int).tolist())
        keep = order[:top_k]
        top_names.append([str(feature_names[i]) for i in keep])
        top_scores.append([float(feature_errors[row_idx, i]) for i in keep])

    return top_names, top_scores, full_ranks


def build_prediction_frame(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    feature_errors: np.ndarray,
    feature_names: Sequence[str],
    top_k: int,
    labels: np.ndarray | None = None,
    predicted_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    frame["anomaly_score"] = scores.astype(float)
    frame["threshold"] = float(threshold)
    frame["score_over_threshold"] = frame["anomaly_score"] - float(threshold)

    top_names, top_scores, full_ranks = top_feature_summary(
        feature_errors=feature_errors,
        feature_names=feature_names,
        top_k=top_k,
    )

    frame["feature_error_vector"] = [row.tolist() for row in feature_errors.astype(float)]
    frame["top_feature_rank"] = full_ranks
    frame["top_k_features"] = top_names
    frame["top_k_feature_errors"] = top_scores

    if labels is not None:
        frame["label"] = labels.astype(int)
    if predicted_labels is not None:
        frame["predicted_label"] = predicted_labels.astype(int)
    return frame


def choose_device(preferred: str = "cuda") -> str:
    try:
        import torch

        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def safe_literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (SyntaxError, ValueError):
                pass
    return []


def save_loss_curve(
    path: str | Path,
    history: pd.DataFrame,
    train_column: str = "train_loss",
    val_column: str = "val_loss",
) -> Path:
    path_obj = Path(path)
    ensure_directory(path_obj.parent)

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history[train_column], label="train")
    if val_column in history.columns and history[val_column].notna().any():
        plt.plot(history["epoch"], history[val_column], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_obj, dpi=150)
    plt.close()
    return path_obj
