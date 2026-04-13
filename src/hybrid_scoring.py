from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from utils import compute_feature_error_matrix, compute_window_scores
except ImportError:
    from .utils import compute_feature_error_matrix, compute_window_scores


VALID_MODEL_MODES = ("reconstruction", "forecasting", "hybrid")


@dataclass
class ModeScoreArtifacts:
    recon_scores: np.ndarray
    forecast_scores: np.ndarray
    final_scores: np.ndarray
    recon_feature_errors: np.ndarray
    forecast_feature_errors: np.ndarray
    final_feature_errors: np.ndarray


def normalize_model_mode(mode: str | None) -> str:
    value = str(mode or "reconstruction").strip().lower()
    if value not in VALID_MODEL_MODES:
        raise ValueError(
            f"Unsupported MODEL_MODE '{mode}'. Expected one of {', '.join(VALID_MODEL_MODES)}."
        )
    return value


def normalize_hybrid_weights(alpha: float, beta: float) -> tuple[float, float]:
    alpha = float(alpha)
    beta = float(beta)
    total = alpha + beta
    if total <= 0:
        return 0.5, 0.5
    return alpha / total, beta / total


def threshold_for_mode(detector_meta: dict[str, Any], mode: str) -> float:
    normalized_mode = normalize_model_mode(mode)
    mode_meta = detector_meta.get("modes", {}).get(normalized_mode, {})
    if "threshold" in mode_meta:
        return float(mode_meta["threshold"])
    if normalized_mode == "reconstruction":
        return float(detector_meta.get("threshold", 0.0))
    if normalized_mode == "forecasting":
        return float(detector_meta.get("forecast_threshold", detector_meta.get("threshold", 0.0)))
    return float(detector_meta.get("hybrid_threshold", detector_meta.get("threshold", 0.0)))


def available_modes(detector_meta: dict[str, Any]) -> list[str]:
    modes = detector_meta.get("available_modes")
    if isinstance(modes, list) and modes:
        return [normalize_model_mode(mode) for mode in modes]
    found: list[str] = []
    if detector_meta.get("threshold") is not None:
        found.append("reconstruction")
    if detector_meta.get("forecast_threshold") is not None:
        found.append("forecasting")
    if detector_meta.get("hybrid_threshold") is not None:
        found.append("hybrid")
    return found or ["reconstruction"]


def _run_model_batches(
    model: torch.nn.Module,
    x_array: np.ndarray,
    c_array: np.ndarray,
    batch_size: int,
    device: torch.device,
    desc: str,
    show_progress: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    total_batches = max(1, (len(x_array) + batch_size - 1) // batch_size)
    iterator = tqdm(
        range(0, len(x_array), batch_size),
        total=total_batches,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    model.eval()
    with torch.no_grad():
        for start in iterator:
            end = start + batch_size
            x_batch = torch.as_tensor(x_array[start:end], dtype=torch.float32, device=device)
            c_batch = torch.as_tensor(c_array[start:end], dtype=torch.float32, device=device)
            outputs.append(model(x_batch, c_batch).detach().cpu().numpy())
            iterator.set_postfix(windows=min(end, len(x_array)))
    iterator.close()
    if not outputs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def compute_reconstruction_outputs(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    c_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructed = _run_model_batches(
        model=model,
        x_array=x_scaled,
        c_array=c_scaled,
        batch_size=batch_size,
        device=device,
        desc="Reconstruction scoring",
        show_progress=show_progress,
    )
    feature_errors = compute_feature_error_matrix(x_scaled, reconstructed).astype(np.float32, copy=False)
    scores = compute_window_scores(feature_errors).astype(np.float32, copy=False)
    return feature_errors, scores


def compute_forecasting_outputs(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    c_scaled: np.ndarray,
    forecast_horizon: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = max(1, int(forecast_horizon))
    if horizon >= int(x_scaled.shape[1]):
        raise ValueError(
            f"forecast_horizon must be smaller than window length (got {horizon} vs {int(x_scaled.shape[1])})."
        )

    forecast_inputs = np.asarray(x_scaled[:, :-horizon, :], dtype=np.float32)
    forecast_targets = np.asarray(x_scaled[:, -horizon:, :], dtype=np.float32)
    predictions = _run_model_batches(
        model=model,
        x_array=forecast_inputs,
        c_array=c_scaled,
        batch_size=batch_size,
        device=device,
        desc="Forecast scoring",
        show_progress=show_progress,
    )
    feature_errors = np.mean((forecast_targets - predictions) ** 2, axis=1).astype(np.float32, copy=False)
    scores = compute_window_scores(feature_errors).astype(np.float32, copy=False)
    return feature_errors, scores


def combine_mode_scores(
    mode: str,
    recon_feature_errors: np.ndarray | None,
    recon_scores: np.ndarray | None,
    forecast_feature_errors: np.ndarray | None,
    forecast_scores: np.ndarray | None,
    alpha: float,
    beta: float,
) -> ModeScoreArtifacts:
    normalized_mode = normalize_model_mode(mode)
    alpha, beta = normalize_hybrid_weights(alpha, beta)

    if recon_scores is None and forecast_scores is None:
        raise ValueError("At least one score stream must be available.")

    template_length = 0
    feature_count = 0
    if recon_scores is not None:
        template_length = int(len(recon_scores))
    elif forecast_scores is not None:
        template_length = int(len(forecast_scores))
    if recon_feature_errors is not None:
        feature_count = int(recon_feature_errors.shape[1])
    elif forecast_feature_errors is not None:
        feature_count = int(forecast_feature_errors.shape[1])

    recon_scores_arr = (
        np.asarray(recon_scores, dtype=np.float32)
        if recon_scores is not None
        else np.zeros(template_length, dtype=np.float32)
    )
    forecast_scores_arr = (
        np.asarray(forecast_scores, dtype=np.float32)
        if forecast_scores is not None
        else np.zeros(template_length, dtype=np.float32)
    )
    recon_errors_arr = (
        np.asarray(recon_feature_errors, dtype=np.float32)
        if recon_feature_errors is not None
        else np.zeros((template_length, feature_count), dtype=np.float32)
    )
    forecast_errors_arr = (
        np.asarray(forecast_feature_errors, dtype=np.float32)
        if forecast_feature_errors is not None
        else np.zeros((template_length, feature_count), dtype=np.float32)
    )

    if normalized_mode == "reconstruction":
        final_scores = recon_scores_arr
        final_errors = recon_errors_arr
    elif normalized_mode == "forecasting":
        final_scores = forecast_scores_arr
        final_errors = forecast_errors_arr
    else:
        final_scores = (alpha * recon_scores_arr + beta * forecast_scores_arr).astype(np.float32, copy=False)
        final_errors = (alpha * recon_errors_arr + beta * forecast_errors_arr).astype(np.float32, copy=False)

    return ModeScoreArtifacts(
        recon_scores=recon_scores_arr,
        forecast_scores=forecast_scores_arr,
        final_scores=final_scores,
        recon_feature_errors=recon_errors_arr,
        forecast_feature_errors=forecast_errors_arr,
        final_feature_errors=final_errors,
    )
