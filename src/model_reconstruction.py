from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as functional


class FiLM(nn.Module):
    """Feature-wise linear modulation driven by a static context vector."""

    def __init__(self, feature_units: int, context_dim: int) -> None:
        super().__init__()
        self.feature_units = int(feature_units)
        self.context_dim = int(context_dim)
        self.modulation = nn.Linear(self.context_dim, 2 * self.feature_units)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.modulation(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class FiLMBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(1, out_channels)
        self.film = FiLM(out_channels, context_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(x)
        hidden = self.norm(hidden)
        hidden = self.film(hidden, context)
        hidden = self.activation(hidden)
        return self.dropout(hidden)


class FiLMAutoencoder(nn.Module):
    """
    Multivariate FiLM-conditioned convolutional autoencoder.

    Input shape:
        x: (B, T, F)
        context: (B, C)
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        context_dim: int,
        units: int = 64,
        latent: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        self.units = int(units)
        self.latent = int(latent)

        # Encoder
        self.encoder_block_1 = FiLMBlock(n_features, units, context_dim, dropout=dropout)
        self.encoder_block_2 = FiLMBlock(units, units, context_dim, dropout=dropout)
        self.encoder_block_3 = FiLMBlock(units, units, context_dim, dropout=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.latent_dropout = nn.Dropout(dropout)
        self.latent_projection = nn.Linear(units, latent)

        # Decoder
        self.decoder_dense = nn.Linear(latent, window_size * units)
        self.decoder_block_1 = FiLMBlock(units, units, context_dim, dropout=dropout)
        self.decoder_block_2 = FiLMBlock(units, units, context_dim, dropout=dropout)
        
        self.decoder_conv = nn.Conv1d(units, units, kernel_size=3, padding=1)
        self.decoder_activation = nn.ReLU()
        self.output_projection = nn.Conv1d(units, n_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = x.transpose(1, 2)
        hidden = self.encoder_block_1(hidden, context)
        hidden = self.encoder_block_2(hidden, context)
        hidden = self.encoder_block_3(hidden, context)
        
        hidden = self.global_pool(hidden).squeeze(-1)
        hidden = self.latent_dropout(hidden)
        latent = functional.relu(self.latent_projection(hidden))

        decoded = functional.relu(self.decoder_dense(latent))
        decoded = decoded.view(-1, self.units, self.window_size)
        
        decoded = self.decoder_block_1(decoded, context)
        decoded = self.decoder_block_2(decoded, context)
        
        decoded = self.decoder_activation(self.decoder_conv(decoded))
        output = self.output_projection(decoded)
        return output.transpose(1, 2)


def build_reconstruction_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device | str | None = None,
) -> FiLMAutoencoder:
    model = FiLMAutoencoder(
        window_size=int(checkpoint["window_size"]),
        n_features=int(checkpoint["n_features"]),
        context_dim=int(checkpoint["context_dim"]),
        units=int(checkpoint.get("units", 64)),
        latent=int(checkpoint.get("latent", 64)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if device is not None:
        model.to(device)
    model.eval()
    return model
