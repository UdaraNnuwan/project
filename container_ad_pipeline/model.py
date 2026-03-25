from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, feature_units: int, context_dim: int):
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
    def __init__(self, in_channels: int, out_channels: int, context_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.film = FiLM(out_channels, context_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.film(x, context)
        return self.act(x)


class FiLMAutoencoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, context_dim: int, units: int = 64, latent: int = 64):
        super().__init__()
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        self.units = int(units)
        self.latent = int(latent)

        self.enc1 = FiLMBlock(n_features, units, context_dim)
        self.enc2 = FiLMBlock(units, units, context_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.z = nn.Linear(units, latent)

        self.dec_dense = nn.Linear(latent, window_size * units)
        self.dec1 = FiLMBlock(units, units, context_dim)
        self.dec_conv = nn.Conv1d(units, units, kernel_size=3, padding=1)
        self.dec_act = nn.ReLU()
        self.out = nn.Conv1d(units, n_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        h = self.enc1(x, context)
        h = self.enc2(h, context)
        h = self.gap(h).squeeze(-1)
        z = F.relu(self.z(h))

        d = F.relu(self.dec_dense(z))
        d = d.view(-1, self.units, self.window_size)
        d = self.dec1(d, context)
        d = self.dec_act(self.dec_conv(d))
        out = self.out(d)
        return out.transpose(1, 2)


def build_model_from_checkpoint(checkpoint: dict) -> FiLMAutoencoder:
    model = FiLMAutoencoder(
        window_size=int(checkpoint["window_size"]),
        n_features=int(checkpoint["n_features"]),
        context_dim=int(checkpoint["context_dim"]),
        units=int(checkpoint["units"]),
        latent=int(checkpoint["latent"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
