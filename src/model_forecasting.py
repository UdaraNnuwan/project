from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ContextGRUForecaster(nn.Module):
    """
    Simple multivariate forecaster for anomaly scoring.

    Input shape:
        x: (B, T, F)
        context: (B, C)

    Output shape:
        (B, H, F)
    """

    def __init__(
        self,
        input_window_size: int,
        n_features: int,
        context_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        forecast_horizon: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_window_size = int(input_window_size)
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.forecast_horizon = int(forecast_horizon)

        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.n_features + self.context_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.context_projection = nn.Linear(self.context_dim, self.hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.forecast_horizon * self.n_features),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_steps = context.unsqueeze(1).expand(-1, x.size(1), -1)
        sequence = torch.cat([x, context_steps], dim=-1)
        _, hidden = self.encoder(sequence)
        encoded = hidden[-1]
        context_encoded = functional.relu(self.context_projection(context))
        fused = torch.cat([encoded, context_encoded], dim=-1)
        forecast = self.output_head(fused)
        return forecast.view(-1, self.forecast_horizon, self.n_features)


def build_forecasting_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device | str | None = None,
) -> ContextGRUForecaster:
    model = ContextGRUForecaster(
        input_window_size=int(
            checkpoint.get(
                "forecast_input_window_size",
                int(checkpoint["window_size"]) - int(checkpoint.get("forecast_horizon", 1)),
            )
        ),
        n_features=int(checkpoint["n_features"]),
        context_dim=int(checkpoint["context_dim"]),
        hidden_size=int(checkpoint.get("hidden_size", checkpoint.get("units", 64))),
        num_layers=int(checkpoint.get("num_layers", 1)),
        forecast_horizon=int(checkpoint.get("forecast_horizon", 1)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return model
