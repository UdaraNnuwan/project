import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.units = int(units)
        self.modulation = nn.Linear(self.units if False else 0, 0)  # placeholder removed below


class FiLM(nn.Module):
    def __init__(self, feature_units: int, context_dim: int):
        super().__init__()
        self.feature_units = int(feature_units)
        self.context_dim = int(context_dim)
        self.modulation = nn.Linear(self.context_dim, 2 * self.feature_units)

    def forward(self, x, context):
        # x: (B, C, T)
        # context: (B, context_dim)
        gamma_beta = self.modulation(context)              # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # (B, C), (B, C)
        gamma = gamma.unsqueeze(-1)                        # (B, C, 1)
        beta = beta.unsqueeze(-1)                          # (B, C, 1)
        return x * (1.0 + gamma) + beta


class FiLMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.film = FiLM(out_channels, context_dim)
        self.act = nn.ReLU()

    def forward(self, x, context):
        x = self.conv(x)
        x = self.film(x, context)
        x = self.act(x)
        return x


class FiLMAutoencoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, context_dim: int, units: int = 64, latent: int = 64):
        super().__init__()
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        self.units = int(units)
        self.latent = int(latent)

        # Encoder
        self.enc1 = FiLMBlock(n_features, units, context_dim)
        self.enc2 = FiLMBlock(units, units, context_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.z = nn.Linear(units, latent)

        # Decoder
        self.dec_dense = nn.Linear(latent, window_size * units)
        self.dec1 = FiLMBlock(units, units, context_dim)
        self.dec_conv = nn.Conv1d(units, units, kernel_size=3, padding=1)
        self.dec_act = nn.ReLU()
        self.out = nn.Conv1d(units, n_features, kernel_size=3, padding=1)

    def forward(self, x, context):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)

        h = self.enc1(x, context)
        h = self.enc2(h, context)
        h = self.gap(h).squeeze(-1)      # (B, units)
        z = F.relu(self.z(h))            # (B, latent)

        d = F.relu(self.dec_dense(z))    # (B, T*units)
        d = d.view(-1, self.units, self.window_size)  # (B, units, T)
        d = self.dec1(d, context)
        d = self.dec_act(self.dec_conv(d))
        out = self.out(d)                # (B, F, T)

        return out.transpose(1, 2)       # back to (B, T, F)