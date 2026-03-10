import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainAutoencoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, units: int = 64, latent: int = 64):
        super().__init__()
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.units = int(units)
        self.latent = int(latent)

        # Encoder
        self.enc1 = nn.Conv1d(n_features, units, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(units, units, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.z = nn.Linear(units, latent)

        # Decoder
        self.dec_dense = nn.Linear(latent, window_size * units)
        self.dec_conv = nn.Conv1d(units, units, kernel_size=3, padding=1)
        self.dec_act = nn.ReLU()
        self.out = nn.Conv1d(units, n_features, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)

        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = self.gap(h).squeeze(-1)
        z = F.relu(self.z(h))

        d = F.relu(self.dec_dense(z))
        d = d.view(-1, self.units, self.window_size)
        d = self.dec_act(self.dec_conv(d))
        out = self.out(d)

        return out.transpose(1, 2)  # back to (B, T, F)