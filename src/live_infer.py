from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

try:
    from hybrid_scoring import (
        available_modes,
        combine_mode_scores,
        compute_forecasting_outputs,
        compute_reconstruction_outputs,
        normalize_hybrid_weights,
        normalize_model_mode,
        threshold_for_mode,
    )
    from model import build_model_from_checkpoint
    from model_forecasting import build_forecasting_model_from_checkpoint
    from utils import apply_3d_scaler, choose_device, top_feature_summary
except ImportError:
    from .hybrid_scoring import (
        available_modes,
        combine_mode_scores,
        compute_forecasting_outputs,
        compute_reconstruction_outputs,
        normalize_hybrid_weights,
        normalize_model_mode,
        threshold_for_mode,
    )
    from .model import build_model_from_checkpoint
    from .model_forecasting import build_forecasting_model_from_checkpoint
    from .utils import apply_3d_scaler, choose_device, top_feature_summary


@dataclass
class StreamingInferenceResult:
    entity_id: str
    ready: bool
    mode: str
    anomaly_score: float | None = None
    final_score: float | None = None
    recon_score: float | None = None
    forecast_score: float | None = None
    threshold: float | None = None
    top_k_features: list[str] | None = None
    top_k_feature_errors: list[float] | None = None
    feature_error_vector: list[float] | None = None
    recon_feature_error_vector: list[float] | None = None
    forecast_feature_error_vector: list[float] | None = None
    predicted_label: int | None = None
    metadata: dict[str, Any] | None = None


def _load_first_checkpoint(paths: list[Path], device: str) -> dict[str, Any] | None:
    for path in paths:
        if path.exists():
            return torch.load(path, map_location=device)
    return None


def load_model_artifacts(
    model_dir: str | Path,
    device: str = "cpu",
    model_mode: str | None = None,
) -> dict[str, Any]:
    resolved_model_dir = Path(model_dir).expanduser().resolve()
    resolved_device = choose_device(device)
    detector_meta = joblib.load(resolved_model_dir / "detector_meta.joblib")

    requested_mode = normalize_model_mode(model_mode or detector_meta.get("default_mode", "reconstruction"))
    mode_list = available_modes(detector_meta)
    if requested_mode not in mode_list and requested_mode != "hybrid":
        raise ValueError(
            f"Requested mode '{requested_mode}' is not available in {resolved_model_dir}. "
            f"Available modes: {', '.join(mode_list)}"
        )

    reconstruction_checkpoint = _load_first_checkpoint(
        [
            resolved_model_dir / "reconstruction_model.pt",
            resolved_model_dir / "film_ae.pt",
        ],
        device=resolved_device,
    )
    forecasting_checkpoint = _load_first_checkpoint(
        [
            resolved_model_dir / "forecasting_model.pt",
            resolved_model_dir / "forecast_model.pt",
        ],
        device=resolved_device,
    )

    reconstruction_model = (
        build_model_from_checkpoint(reconstruction_checkpoint, device=resolved_device)
        if reconstruction_checkpoint is not None
        else None
    )
    forecasting_model = (
        build_forecasting_model_from_checkpoint(forecasting_checkpoint, device=resolved_device)
        if forecasting_checkpoint is not None
        else None
    )

    source_checkpoint = reconstruction_checkpoint or forecasting_checkpoint
    if source_checkpoint is None:
        raise FileNotFoundError(
            f"No model checkpoint was found in {resolved_model_dir}. "
            "Expected at least film_ae.pt or forecasting_model.pt."
        )

    if requested_mode == "hybrid" and (reconstruction_model is None or forecasting_model is None):
        raise ValueError(
            f"Hybrid mode requires both reconstruction and forecasting checkpoints in {resolved_model_dir}."
        )
    if requested_mode == "forecasting" and forecasting_model is None:
        raise ValueError(f"Forecasting mode requires forecasting_model.pt in {resolved_model_dir}.")

    alpha, beta = normalize_hybrid_weights(
        detector_meta.get("alpha", 0.6),
        detector_meta.get("beta", 0.4),
    )
    feature_columns = list(source_checkpoint.get("feature_columns", []))
    context_columns = list(source_checkpoint.get("context_columns", []))
    forecast_horizon = int(
        detector_meta.get(
            "forecast_horizon",
            source_checkpoint.get("forecast_horizon", 1),
        )
    )
    window_size = int(source_checkpoint.get("window_size", detector_meta.get("window_size", 0)))
    forecast_input_window_size = int(
        detector_meta.get(
            "forecast_input_window_size",
            source_checkpoint.get("forecast_input_window_size", max(1, window_size - forecast_horizon)),
        )
    )

    return {
        "model_dir": resolved_model_dir,
        "device": resolved_device,
        "mode": requested_mode,
        "available_modes": mode_list,
        "detector_meta": detector_meta,
        "reconstruction_model": reconstruction_model,
        "forecasting_model": forecasting_model,
        "x_scaler": joblib.load(resolved_model_dir / "x_scaler.joblib"),
        "c_scaler": joblib.load(resolved_model_dir / "c_scaler.joblib"),
        "feature_columns": feature_columns,
        "context_columns": context_columns,
        "window_size": window_size,
        "forecast_horizon": forecast_horizon,
        "forecast_input_window_size": forecast_input_window_size,
        "alpha": alpha,
        "beta": beta,
    }


class StreamingHybridAnomalyDetector:
    """
    Maintains per-entity sequential windows for reconstruction, forecasting, or hybrid scoring.
    """

    def __init__(
        self,
        reconstruction_model: torch.nn.Module | None,
        forecasting_model: torch.nn.Module | None,
        x_scaler: Any,
        c_scaler: Any,
        detector_meta: dict[str, Any],
        feature_names: list[str],
        top_k_features: int = 5,
        device: str | torch.device = "cpu",
        model_mode: str | None = None,
    ) -> None:
        self.reconstruction_model = reconstruction_model
        self.forecasting_model = forecasting_model
        self.x_scaler = x_scaler
        self.c_scaler = c_scaler
        self.detector_meta = detector_meta
        self.feature_names = list(feature_names)
        self.top_k_features = int(top_k_features)
        self.device = torch.device(choose_device(str(device)))
        self.mode = normalize_model_mode(model_mode or detector_meta.get("default_mode", "reconstruction"))
        self.threshold = threshold_for_mode(detector_meta, self.mode)
        self.alpha, self.beta = normalize_hybrid_weights(
            detector_meta.get("alpha", 0.6),
            detector_meta.get("beta", 0.4),
        )
        self.window_size = int(detector_meta.get("window_size", 0))
        self.forecast_horizon = int(detector_meta.get("forecast_horizon", 1))
        self.buffers: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def set_mode(self, model_mode: str) -> None:
        self.mode = normalize_model_mode(model_mode)
        self.threshold = threshold_for_mode(self.detector_meta, self.mode)

    def update(
        self,
        entity_id: str,
        feature_row: np.ndarray,
        context_vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> StreamingInferenceResult:
        buffer = self.buffers[str(entity_id)]
        buffer.append(np.asarray(feature_row, dtype=np.float32))
        if len(buffer) < self.window_size:
            return StreamingInferenceResult(
                entity_id=str(entity_id),
                ready=False,
                mode=self.mode,
                metadata=metadata,
            )

        raw_window = np.asarray(buffer, dtype=np.float32)
        window = np.log1p(np.clip(raw_window, a_min=0.0, a_max=None))
        window_scaled = apply_3d_scaler(self.x_scaler, window[None, ...])
        context_scaled = self.c_scaler.transform(
            np.asarray(context_vector, dtype=np.float32).reshape(1, -1)
        ).astype(np.float32)

        recon_feature_errors = None
        recon_scores = None
        forecast_feature_errors = None
        forecast_scores = None
        if self.mode in {"reconstruction", "hybrid"}:
            if self.reconstruction_model is None:
                raise ValueError("Reconstruction checkpoint is required for reconstruction/hybrid mode.")
            recon_feature_errors, recon_scores = compute_reconstruction_outputs(
                model=self.reconstruction_model,
                x_scaled=window_scaled,
                c_scaled=context_scaled,
                batch_size=1,
                device=self.device,
                show_progress=False,
            )
        if self.mode in {"forecasting", "hybrid"}:
            if self.forecasting_model is None:
                raise ValueError("Forecasting checkpoint is required for forecasting/hybrid mode.")
            forecast_feature_errors, forecast_scores = compute_forecasting_outputs(
                model=self.forecasting_model,
                x_scaled=window_scaled,
                c_scaled=context_scaled,
                forecast_horizon=self.forecast_horizon,
                batch_size=1,
                device=self.device,
                show_progress=False,
            )

        combined = combine_mode_scores(
            mode=self.mode,
            recon_feature_errors=recon_feature_errors,
            recon_scores=recon_scores,
            forecast_feature_errors=forecast_feature_errors,
            forecast_scores=forecast_scores,
            alpha=self.alpha,
            beta=self.beta,
        )
        final_feature_errors = combined.final_feature_errors.astype(np.float32, copy=False)
        top_names, top_scores, _ = top_feature_summary(
            feature_errors=final_feature_errors,
            feature_names=self.feature_names,
            top_k=self.top_k_features,
        )
        final_score = float(combined.final_scores[0])
        return StreamingInferenceResult(
            entity_id=str(entity_id),
            ready=True,
            mode=self.mode,
            anomaly_score=final_score,
            final_score=final_score,
            recon_score=float(combined.recon_scores[0]) if combined.recon_scores.size else None,
            forecast_score=float(combined.forecast_scores[0]) if combined.forecast_scores.size else None,
            threshold=float(self.threshold),
            predicted_label=int(final_score > float(self.threshold)),
            top_k_features=top_names[0],
            top_k_feature_errors=top_scores[0],
            feature_error_vector=[float(value) for value in final_feature_errors[0]],
            recon_feature_error_vector=[
                float(value) for value in combined.recon_feature_errors[0]
            ] if combined.recon_feature_errors.size else None,
            forecast_feature_error_vector=[
                float(value) for value in combined.forecast_feature_errors[0]
            ] if combined.forecast_feature_errors.size else None,
            metadata=metadata or {},
        )
