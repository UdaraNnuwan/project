from __future__ import annotations

from typing import Any, Mapping

import torch

try:
    from model_forecasting import ContextGRUForecaster, build_forecasting_model_from_checkpoint
    from model_reconstruction import FiLM, FiLMBlock, FiLMAutoencoder, build_reconstruction_model_from_checkpoint
except ImportError:
    from .model_forecasting import ContextGRUForecaster, build_forecasting_model_from_checkpoint
    from .model_reconstruction import FiLM, FiLMBlock, FiLMAutoencoder, build_reconstruction_model_from_checkpoint


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device | str | None = None,
) -> FiLMAutoencoder:
    return build_reconstruction_model_from_checkpoint(checkpoint, device=device)
