"""
model_architecture.py  (PyTorch — Dual-Head Edition)
=====================================================
BiLSTM Autoencoder with FiLM (Feature-wise Linear Modulation) for
Containerised Anomaly Detection — Alibaba Cloud Trace 2018.

Architecture Overview
---------------------

  ┌──────────────────────────────────────────────────────────────────────┐
  │                    SHARED ENCODER                                     │
  │                                                                       │
  │   Time Series     BiLSTM(H1)  ──── BN ──── Dropout                  │
  │   (B, W, F)  ──►  BiLSTM(H2)  ──── Dropout                          │
  │                   last-step → Linear(latent_dim) → encoding           │
  │                                     FiLM(meta)                      │
  │   Metadata     ─►  γ, β ──► encoding * γ + β   (B, latent)          │
  │   (B, M)                                                              │
  └──────────────────────────────────────────────────────────────────────┘
                conditioned latent (B, latent_dim) 

  ┌─────────────────────────────┐   ┌───────────────────────────────────┐
  │    HEAD 1 — Reconstruction  │   │    HEAD 2 — Forecasting (t+1)     │
  │                             │   │                                   │
  │  seed_proj (latent→2*H2)    │   │  fc1: latent→fc_hidden  ReLU      │
  │  RepeatVector(W)            │   │  Dropout                          │
  │  BiLSTM(H2) ─ BN ─ Dropout  │   │  fc2: fc_hidden→n_ts_features     │
  │  BiLSTM(H1) ─ Dropout       │   │  Sigmoid → output [0,1]           │
  │  Linear(F) → Sigmoid        │   │                                   │
  │  output: (B, W, F)          │   │  output: (B, F)                   │
  └─────────────────────────────┘   └───────────────────────────────────┘
          MSE_recon                           MSE_forecast
               ╲                              ╱
         Total_Loss = α·MSE_recon + (1-α)·MSE_forecast

Symbols
-------
  B       = batch_size
  W       = window_size
  F       = n_ts_features
  M       = n_meta_features
  latent  = latent_dim
  H1, H2  = lstm_units[0], lstm_units[1]

Module Contents
---------------
  FiLMLayer                 – Feature-wise Linear Modulation conditioning
  BiLSTMEncoder             – Shared bidirectional LSTM encoder
  BiLSTMDecoder             – Reconstruction head (Head 1)
  ForecastingHead           – Next-step forecasting head (Head 2) [NEW]
  BiLSTMFiLMAutoencoder     – Original single-head model (backward compat)
  DualHeadBiLSTMFiLM        – Dual-head model (Reconstruction + Forecasting) [NEW]
  HybridAnomalyLoss         – Weighted dual-head loss criterion [NEW]
  build_model               – Factory for BiLSTMFiLMAutoencoder
  load_model                – Loader for BiLSTMFiLMAutoencoder
  build_dual_head_model     – Factory for DualHeadBiLSTMFiLM [NEW]
  load_dual_head_model      – Loader for DualHeadBiLSTMFiLM [NEW]

References
----------
  Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
  AAAI 2018.  https://arxiv.org/abs/1709.07871

  Malhotra et al., "LSTM-based Encoder-Decoder for Multi-Sensor Anomaly
  Detection", ICML 2016.

  Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
  Nonparametric Dynamic Thresholding", KDD 2018.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("model_architecture")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
)


# ===========================================================================
# ─── COMPONENT 1: FiLM Layer ────────────────────────────────────────────────
# ===========================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Given a conditioning vector ``c`` (derived from container metadata), the
    FiLM layer produces per-feature scale (γ) and shift (β) parameters that
    modulate the latent encoding ``x`` element-wise:

        FiLM(x, c) = γ(c) ⊙ x + β(c)

    This allows the model to adjust its internal representation based on the
    container type (API, database, cache, etc.), drastically reducing false
    positives across heterogeneous workloads — the core contribution of the
    BiLSTM-FiLM architecture.

    A single hidden projection forces the metadata to learn a compact non-linear
    summary before splitting into γ and β.  Both branches share this hidden
    layer to keep the parameter budget small.

    References
    ----------
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.  https://arxiv.org/abs/1709.07871

    Parameters
    ----------
    latent_dim      : int  – dimensionality of the feature vector to modulate
    n_meta_features : int  – dimensionality of the conditioning (metadata) input
    hidden_dim      : int  – intermediate FiLM projection size (default: latent_dim)
    """

    def __init__(
        self,
        latent_dim:      int,
        n_meta_features: int,
        hidden_dim:      Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim

        # Non-linear projection of the conditioning (metadata) vector.
        # We use ReLU so the network learns a useful non-linear summary of the
        # metadata before generating γ and β.
        self.cond_proj = nn.Sequential(
            nn.Linear(n_meta_features, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Joint γ/β output head — split in forward().
        # Output size is 2 × latent_dim: first half = γ, second = β.
        self.gamma_beta = nn.Linear(hidden_dim, 2 * latent_dim)

        # Initialise γ rows ≈ 1 and β rows ≈ 0 so FiLM starts as an approximate
        # identity transform.  This prevents instability at the start of training.
        nn.init.ones_(self.gamma_beta.weight[:latent_dim])   # γ rows → 1
        nn.init.zeros_(self.gamma_beta.weight[latent_dim:])  # β rows → 0
        nn.init.zeros_(self.gamma_beta.bias)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Tensor flow
        -----------
        cond  : (B, M)       — raw metadata vector
        h     : (B, hidden)  — non-linear metadata projection
        γ, β  : (B, latent)  — element-wise scale and shift
        output: (B, latent)  — modulated encoding

        Parameters
        ----------
        x    : Tensor (B, latent_dim) – the encoding to modulate
        cond : Tensor (B, M)          – the metadata conditioning vector

        Returns
        -------
        Tensor (B, latent_dim) – FiLM-modulated latent encoding
        """
        h          = self.cond_proj(cond)           # (B, hidden)
        gamma_beta = self.gamma_beta(h)             # (B, 2 * latent_dim)

        latent_dim = x.shape[-1]
        gamma = gamma_beta[:, :latent_dim]          # (B, latent) – scale
        beta  = gamma_beta[:, latent_dim:]          # (B, latent) – shift

        return gamma * x + beta                     # (B, latent_dim)


# ===========================================================================
# ─── COMPONENT 2: Shared BiLSTM Encoder ─────────────────────────────────────
# ===========================================================================

class BiLSTMEncoder(nn.Module):
    """
    Shared bidirectional LSTM encoder.

    Takes a min-max scaled time-series window (B, W, F) and produces a
    fixed-size latent encoding (B, latent_dim).  This encoding is then
    conditioned by the FiLM layer before being consumed by either (or both)
    output heads.

    Tensor flow
    -----------
    Input  : (B, W, F)
    BiLSTM1: (B, W, 2*H1)  — full sequence retained for BiLSTM2
    BN1    : (B, W, 2*H1)  — per-channel normalisation across time
    BiLSTM2: (B, W, 2*H2)  — second sequence of richer representations
    last   : (B, 2*H2)     — only the final timestep; collapses sequences
    Linear : (B, latent)   — bottleneck projection

    Parameters
    ----------
    n_ts_features : int
    latent_dim    : int
    lstm_units    : (int, int) — hidden units for the two stacked BiLSTMs
    dropout_rate  : float
    """

    def __init__(
        self,
        n_ts_features: int,
        latent_dim:    int,
        lstm_units:    Tuple[int, int] = (128, 64),
        dropout_rate:  float = 0.2,
    ) -> None:
        super().__init__()

        H1, H2 = lstm_units

        # ── Layer 1: first BiLSTM.  Output is 2*H1 because bidirectional.
        # We return the full sequence so Layer 2 can attend across all timesteps.
        self.bilstm1 = nn.LSTM(
            input_size=n_ts_features,
            hidden_size=H1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # BatchNorm1d operates on (B, C, L); we permute, norm, permute back.
        self.bn1   = nn.BatchNorm1d(H1 * 2)
        self.drop1 = nn.Dropout(dropout_rate)

        # ── Layer 2: second BiLSTM.  Output is 2*H2.
        self.bilstm2 = nn.LSTM(
            input_size=H1 * 2,
            hidden_size=H2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout_rate)

        # ── Bottleneck: collapse (B, 2*H2) → (B, latent_dim)
        # ReLU ensures non-negative activations, which is consistent with
        # the Sigmoid output of the decoder.
        self.bottleneck = nn.Sequential(
            nn.Linear(H2 * 2, latent_dim),
            nn.ReLU(inplace=True),
        )

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, W, F)

        Returns
        -------
        encoding : Tensor (B, latent_dim)
        """
        # ── BiLSTM 1: retain full sequence output ────────────────────────
        out1, _ = self.bilstm1(x)                   # (B, W, 2*H1)
        # BatchNorm1d expects (B, C, L); permute → norm → permute back.
        out1 = self.bn1(out1.permute(0, 2, 1)).permute(0, 2, 1)
        out1 = self.drop1(out1)                     # (B, W, 2*H1)

        # ── BiLSTM 2: collapse to last timestep ───────────────────────────
        out2, _ = self.bilstm2(out1)                # (B, W, 2*H2)
        # Take only the final timestep — this is the sequence summary.
        # BiLSTM concatenates fwd[:,-1,:] and bwd[:,0,:] at position W-1.
        last = out2[:, -1, :]                       # (B, 2*H2)
        last = self.drop2(last)

        # ── Bottleneck projection to fixed-size latent space ──────────────
        encoding = self.bottleneck(last)            # (B, latent_dim)
        return encoding


# ===========================================================================
# ─── COMPONENT 3a: BiLSTM Decoder — Head 1 (Reconstruction) ────────────────
# ===========================================================================

class BiLSTMDecoder(nn.Module):
    """
    Bidirectional LSTM decoder — Reconstruction Head.

    Takes the FiLM-conditioned latent vector (B, latent_dim) and reconstructs
    the original time-series window (B, W, F).

    Strategy: project latent → seed vector → repeat W times → two BiLSTMs
    (mirroring the encoder) → per-timestep linear projection → Sigmoid.

    The Sigmoid output constrains reconstructions to [0, 1], consistent with
    the Min-Max scaled input and the autoencoder training objective.

    Tensor flow (Head 1)
    --------------------
    z       : (B, latent)           — FiLM-conditioned encoding
    seed    : (B, 2*H2)            — projected seed
    seq     : (B, W, 2*H2)         — repeated across W timesteps
    BiLSTM1 : (B, W, 2*H2)
    BiLSTM2 : (B, W, 2*H1)
    output  : (B, W, F)            — reconstructed window ∈ [0, 1]
    """

    def __init__(
        self,
        n_ts_features: int,
        latent_dim:    int,
        window_size:   int,
        lstm_units:    Tuple[int, int] = (128, 64),
        dropout_rate:  float = 0.2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        H1, H2 = lstm_units

        # ── Project conditioned encoding to initial sequence seed
        self.seed_proj = nn.Sequential(
            nn.Linear(latent_dim, H2 * 2),
            nn.ReLU(inplace=True),
        )

        # ── Decoder BiLSTM 1 (mirrors encoder layer 2, reversed role)
        self.bilstm1 = nn.LSTM(
            input_size=H2 * 2,
            hidden_size=H2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bn1   = nn.BatchNorm1d(H2 * 2)
        self.drop1 = nn.Dropout(dropout_rate)

        # ── Decoder BiLSTM 2 (mirrors encoder layer 1, reversed role)
        self.bilstm2 = nn.LSTM(
            input_size=H2 * 2,
            hidden_size=H1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout_rate)

        # ── Per-timestep output projection: (2*H1) → F, constrained to [0,1]
        self.output_proj = nn.Sequential(
            nn.Linear(H1 * 2, n_ts_features),
            nn.Sigmoid(),
        )

    # -----------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, latent_dim)

        Returns
        -------
        reconstruction : Tensor (B, W, F)  ∈ [0, 1]
        """
        # Project and broadcast to create the initial sequence
        seed = self.seed_proj(z)                        # (B, 2*H2)
        seq  = seed.unsqueeze(1).expand(-1, self.window_size, -1)  # (B, W, 2*H2)

        # ── Decoder BiLSTM 1
        out1, _ = self.bilstm1(seq)                     # (B, W, 2*H2)
        out1 = self.bn1(out1.permute(0, 2, 1)).permute(0, 2, 1)
        out1 = self.drop1(out1)

        # ── Decoder BiLSTM 2
        out2, _ = self.bilstm2(out1)                    # (B, W, 2*H1)
        out2 = self.drop2(out2)

        # ── Per-timestep projection + Sigmoid
        reconstruction = self.output_proj(out2)         # (B, W, F)
        return reconstruction


# ===========================================================================
# ─── COMPONENT 3b: Forecasting Head — Head 2 (t+1 Prediction) ──────────────
# ===========================================================================

class ForecastingHead(nn.Module):
    """
    Dense forecasting head — predicts the NEXT timestep's metrics (t+1).

    Given the FiLM-conditioned latent representation of the current window
    [t-W+1 … t], this head predicts the expected metric vector at time t+1.

    This dual-head design adds a *predictive* loss signal alongside the
    *reconstructive* signal.  During normal operation the encoder learns to
    encode both the historical pattern AND its temporal continuation.
    Anomalies disrupt both signals simultaneously: the model cannot reconstruct
    the abnormal window AND cannot forecast the next step from it.

    Architecture
    ------------
    Two fully-connected layers with dropout and ReLU.  A Sigmoid output
    constrains predictions to [0, 1] to match Min-Max scaled targets.

    Tensor flow (Head 2)
    --------------------
    z     : (B, latent_dim)     — FiLM-conditioned shared encoding
    h1    : (B, fc_hidden_dim)  — dense projection + ReLU
    drop  : (B, fc_hidden_dim)  — regularisation dropout
    h2    : (B, F)              — output projection
    out   : (B, F)              — predicted t+1 metrics ∈ [0, 1]

    Parameters
    ----------
    n_ts_features : int   — number of features to forecast (= F)
    latent_dim    : int   — input latent dimensionality
    fc_hidden_dim : int   — intermediate dense layer size (default: latent_dim * 2)
    dropout_rate  : float — dropout probability
    """

    def __init__(
        self,
        n_ts_features: int,
        latent_dim:    int,
        fc_hidden_dim: Optional[int] = None,
        dropout_rate:  float = 0.2,
    ) -> None:
        super().__init__()

        if fc_hidden_dim is None:
            # Default: 2× latent_dim gives sufficient capacity without
            # over-parameterising relative to the encoder.
            fc_hidden_dim = latent_dim * 2

        self.head = nn.Sequential(
            # ── Dense layer 1: latent → fc_hidden_dim
            nn.Linear(latent_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # ── Dense layer 2: fc_hidden_dim → n_ts_features
            # Sigmoid maps output to [0, 1] — consistent with Min-Max scaling.
            nn.Linear(fc_hidden_dim, n_ts_features),
            nn.Sigmoid(),
        )

    # -----------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, latent_dim)

        Returns
        -------
        forecast : Tensor (B, F)  ∈ [0, 1]
            Predicted metric values for the next timestep t+1.
        """
        return self.head(z)                            # (B, F)


# ===========================================================================
# ─── COMPONENT 4a: Original Single-Head Autoencoder (Backward Compatible) ───
# ===========================================================================

class BiLSTMFiLMAutoencoder(nn.Module):
    """
    Original BiLSTM-FiLM Autoencoder (pure reconstruction).

    Kept for backward compatibility with notebooks 03, 04, and 05 which
    load this model via ``build_model()`` / ``load_model()``.

    For new training runs, prefer ``DualHeadBiLSTMFiLM``.

    Forward inputs
    --------------
    ts_input   : Tensor (B, W, F)  – time-series window
    meta_input : Tensor (B, M)     – metadata vector

    Forward outputs
    ---------------
    reconstruction : Tensor (B, W, F)
    """

    def __init__(
        self,
        window_size:     int,
        n_ts_features:   int,
        n_meta_features: int,
        latent_dim:      int = 64,
        lstm_units:      Tuple[int, int] = (128, 64),
        dropout_rate:    float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = BiLSTMEncoder(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )
        self.film = FiLMLayer(
            latent_dim=latent_dim,
            n_meta_features=n_meta_features,
        )
        self.decoder = BiLSTMDecoder(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            window_size=window_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )

    # -----------------------------------------------------------------------
    def encode(self, ts: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Return the FiLM-conditioned latent vector (B, latent_dim)."""
        z = self.encoder(ts)
        return self.film(z, meta)

    # -----------------------------------------------------------------------
    def forward(self, ts: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ts   : Tensor (B, W, F)
        meta : Tensor (B, M)

        Returns
        -------
        reconstruction : Tensor (B, W, F)
        """
        z = self.encoder(ts)        # (B, latent_dim)
        z = self.film(z, meta)      # (B, latent_dim)  — FiLM conditioned
        return self.decoder(z)      # (B, W, F)


# ===========================================================================
# ─── COMPONENT 4b: Dual-Head Model (Reconstruction + Forecasting) ───────────
# ===========================================================================

class DualHeadBiLSTMFiLM(nn.Module):
    """
    Hybrid Dual-Head BiLSTM-FiLM model.

    Extends the original autoencoder with a forecasting branch (Head 2) that
    predicts the next timestep's metrics alongside the reconstruction (Head 1).
    Both heads share the same BiLSTM encoder and FiLM conditioning layer,
    ensuring that the latent space must simultaneously encode:
      (a) enough information to reconstruct the current window, AND
      (b) enough temporal structure to predict the next step.

    This dual objective forces the encoder to learn richer, more generalisable
    representations of *normal* operation — improving anomaly detection
    by making both heads fail on anomalous inputs simultaneously.

    Forward inputs
    --------------
    ts_input   : Tensor (B, W, F)  – time-series window [t-W+1 … t]
    meta_input : Tensor (B, M)     – container metadata vector

    Forward outputs (a tuple of two tensors)
    ------------------------------------------
    reconstruction : Tensor (B, W, F)  — Head 1 output ∈ [0, 1]
    forecast       : Tensor (B, F)     — Head 2 output ∈ [0, 1]
                                          predicted metrics at t+1

    Parameters
    ----------
    window_size     : int
    n_ts_features   : int
    n_meta_features : int
    latent_dim      : int           – bottleneck size (default 64)
    lstm_units      : (int, int)    – BiLSTM hidden units (default (128, 64))
    dropout_rate    : float
    fc_hidden_dim   : int, optional – size of ForecastingHead hidden layer
    """

    def __init__(
        self,
        window_size:     int,
        n_ts_features:   int,
        n_meta_features: int,
        latent_dim:      int = 64,
        lstm_units:      Tuple[int, int] = (128, 64),
        dropout_rate:    float = 0.2,
        fc_hidden_dim:   Optional[int] = None,
    ) -> None:
        super().__init__()

        # ── Shared encoder + FiLM (identical to the original model) ───────
        self.encoder = BiLSTMEncoder(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )
        self.film = FiLMLayer(
            latent_dim=latent_dim,
            n_meta_features=n_meta_features,
        )

        # ── Head 1: Reconstruction decoder ────────────────────────────────
        self.reconstruction_head = BiLSTMDecoder(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            window_size=window_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )

        # ── Head 2: Forecasting head (t+1 predictor) ─────────────────────
        self.forecasting_head = ForecastingHead(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            fc_hidden_dim=fc_hidden_dim,
            dropout_rate=dropout_rate,
        )

    # -----------------------------------------------------------------------
    def encode(self, ts: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Return the shared FiLM-conditioned latent vector.
        Use at inference time for anomaly scoring without gradient tracking.

        Returns
        -------
        z : Tensor (B, latent_dim)
        """
        z = self.encoder(ts)        # (B, latent_dim)
        return self.film(z, meta)   # (B, latent_dim)  — FiLM conditioned

    # -----------------------------------------------------------------------
    def forward(
        self, ts: torch.Tensor, meta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full dual-head forward pass.

        Tensor flow summary
        -------------------
        ts   (B, W, F)  ──► BiLSTMEncoder ──► (B, latent)
                                    FiLM(meta)
                            (B, latent) ──┬──► BiLSTMDecoder ──► (B, W, F) [recon]
                                          └──► ForecastingHead ──► (B, F)  [t+1]

        Parameters
        ----------
        ts   : Tensor (B, W, F)  – scaled time-series window
        meta : Tensor (B, M)     – container metadata

        Returns
        -------
        reconstruction : Tensor (B, W, F)  — reconstructed input window
        forecast       : Tensor (B, F)     — predicted t+1 metrics
        """
        # ── Shared encoder: (B, W, F) → (B, latent_dim)
        z = self.encoder(ts)                   # (B, latent_dim)

        # ── FiLM conditioning: modulate latent by container metadata
        z = self.film(z, meta)                 # (B, latent_dim)

        # ── Head 1 — Reconstruction: (B, latent) → (B, W, F)
        reconstruction = self.reconstruction_head(z)   # (B, W, F)

        # ── Head 2 — Forecasting: (B, latent) → (B, F)
        forecast = self.forecasting_head(z)            # (B, F)

        return reconstruction, forecast


# ===========================================================================
# ─── COMPONENT 5: Hybrid Anomaly Loss ───────────────────────────────────────
# ===========================================================================

class HybridAnomalyLoss(nn.Module):
    """
    Weighted dual-head loss criterion.

    Combines the reconstruction MSE (Head 1) and forecasting MSE (Head 2)
    via a tunable alpha parameter:

        Total_Loss = alpha × MSE_reconstruction  +  (1 - alpha) × MSE_forecasting

    Alpha Guidance
    --------------
    alpha = 1.0   → pure reconstruction (degenerates to the original model)
    alpha = 0.0   → pure forecasting
    alpha = 0.5   → equal weight (recommended default; best empirical results
                    in multivariate time-series anomaly detection literature)
    alpha = 0.7   → reconstruction-dominant (better for stationary metrics)
    alpha = 0.3   → forecasting-dominant (better for trending metrics)

    Loss Surface
    ------------
    Each sub-loss is computed as mean MSE over all elements:
      MSE_recon  = mean( (ts_window - reconstruction)² )  over (B, W, F)
      MSE_fore   = mean( (next_step - forecast)² )         over (B, F)
    Both are already on a [0, 1] scale because inputs are Min-Max scaled.

    Parameters
    ----------
    alpha : float — reconstruction loss weight ∈ [0, 1]  (default 0.5)
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self.alpha = alpha
        self._mse  = nn.MSELoss(reduction="mean")

    # -----------------------------------------------------------------------
    def forward(
        self,
        reconstruction: torch.Tensor,  # (B, W, F)
        ts_window:      torch.Tensor,  # (B, W, F)  — target for Head 1
        forecast:       torch.Tensor,  # (B, F)
        next_step:      torch.Tensor,  # (B, F)     — target for Head 2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the hybrid loss.

        Parameters
        ----------
        reconstruction : Tensor (B, W, F)  — Head 1 output
        ts_window      : Tensor (B, W, F)  — original window (reconstruction target)
        forecast       : Tensor (B, F)     — Head 2 output
        next_step      : Tensor (B, F)     — true t+1 metrics (forecasting target)

        Returns
        -------
        total_loss  : scalar Tensor  — backpropagated loss
        loss_recon  : scalar Tensor  — Head 1 MSE (detached; for logging)
        loss_fore   : scalar Tensor  — Head 2 MSE (detached; for logging)
        """
        # Head 1: reconstruction error over the full (W, F) window
        loss_recon = self._mse(reconstruction, ts_window)   # scalar

        # Head 2: next-step forecasting error over single-timestep (F,)
        loss_fore  = self._mse(forecast, next_step)          # scalar

        # Weighted combination
        total_loss = self.alpha * loss_recon + (1.0 - self.alpha) * loss_fore

        return total_loss, loss_recon, loss_fore             # (scalar, scalar, scalar)

    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"HybridAnomalyLoss(alpha={self.alpha}  "
            f"[recon_weight={self.alpha:.2f}, "
            f"fore_weight={1-self.alpha:.2f}])"
        )


# ===========================================================================
# ─── FACTORY & LOADER: Original Model (Backward Compatible) ─────────────────
# ===========================================================================

def build_model(
    window_size:     int,
    n_ts_features:   int,
    n_meta_features: int,
    latent_dim:      int = 64,
    lstm_units:      Tuple[int, int] = (128, 64),
    dropout_rate:    float = 0.2,
    device:          Optional[torch.device] = None,
) -> BiLSTMFiLMAutoencoder:
    """
    Instantiate and return a ``BiLSTMFiLMAutoencoder`` on the given device.

    This factory is kept for backward compatibility with notebooks 03–05.
    For new experiments, use ``build_dual_head_model()``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMFiLMAutoencoder(
        window_size=window_size,
        n_ts_features=n_ts_features,
        n_meta_features=n_meta_features,
        latent_dim=latent_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built BiLSTMFiLMAutoencoder (single-head) | "
        "window=%d | ts_feat=%d | meta_feat=%d | latent=%d | params=%s",
        window_size, n_ts_features, n_meta_features, latent_dim, f"{n_p:,}",
    )
    return model


def load_model(
    checkpoint_path: str,
    window_size:     int,
    n_ts_features:   int,
    n_meta_features: int,
    latent_dim:      int = 64,
    lstm_units:      Tuple[int, int] = (128, 64),
    dropout_rate:    float = 0.2,
    device:          Optional[torch.device] = None,
) -> BiLSTMFiLMAutoencoder:
    """
    Rebuild the ``BiLSTMFiLMAutoencoder`` graph and load a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str — path to ``model.pt`` saved with ``torch.save(state_dict)``
    (remaining args same as ``build_model``)

    Returns
    -------
    BiLSTMFiLMAutoencoder in eval mode on ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        window_size=window_size, n_ts_features=n_ts_features,
        n_meta_features=n_meta_features, latent_dim=latent_dim,
        lstm_units=lstm_units, dropout_rate=dropout_rate, device=device,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded single-head weights from '%s' → device=%s", checkpoint_path, device)
    return model


# ===========================================================================
# ─── FACTORY & LOADER: Dual-Head Model ──────────────────────────────────────
# ===========================================================================

def build_dual_head_model(
    window_size:     int,
    n_ts_features:   int,
    n_meta_features: int,
    latent_dim:      int = 64,
    lstm_units:      Tuple[int, int] = (128, 64),
    dropout_rate:    float = 0.2,
    fc_hidden_dim:   Optional[int] = None,
    device:          Optional[torch.device] = None,
) -> DualHeadBiLSTMFiLM:
    """
    Instantiate and return a ``DualHeadBiLSTMFiLM`` model on the given device.

    Parameters
    ----------
    window_size     : int
    n_ts_features   : int
    n_meta_features : int
    latent_dim      : int           – bottleneck size (default 64)
    lstm_units      : (int, int)    – BiLSTM hidden units (default (128, 64))
    dropout_rate    : float
    fc_hidden_dim   : int, optional – ForecastingHead hidden size
                                      (default: latent_dim * 2)
    device          : torch.device or None (auto-selects CUDA if available)

    Returns
    -------
    DualHeadBiLSTMFiLM (on ``device``, in training mode)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualHeadBiLSTMFiLM(
        window_size=window_size,
        n_ts_features=n_ts_features,
        n_meta_features=n_meta_features,
        latent_dim=latent_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        fc_hidden_dim=fc_hidden_dim,
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Breakdown by sub-module for academic reporting
    n_enc  = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_film = sum(p.numel() for p in model.film.parameters() if p.requires_grad)
    n_rec  = sum(p.numel() for p in model.reconstruction_head.parameters() if p.requires_grad)
    n_fore = sum(p.numel() for p in model.forecasting_head.parameters() if p.requires_grad)

    logger.info(
        "Built DualHeadBiLSTMFiLM | window=%d | ts_feat=%d | meta_feat=%d | "
        "latent=%d | total_params=%s  (enc=%s, film=%s, recon=%s, fore=%s)",
        window_size, n_ts_features, n_meta_features, latent_dim,
        f"{n_p:,}", f"{n_enc:,}", f"{n_film:,}", f"{n_rec:,}", f"{n_fore:,}",
    )
    return model


def load_dual_head_model(
    checkpoint_path: str,
    window_size:     int,
    n_ts_features:   int,
    n_meta_features: int,
    latent_dim:      int = 64,
    lstm_units:      Tuple[int, int] = (128, 64),
    dropout_rate:    float = 0.2,
    fc_hidden_dim:   Optional[int] = None,
    device:          Optional[torch.device] = None,
) -> DualHeadBiLSTMFiLM:
    """
    Rebuild the ``DualHeadBiLSTMFiLM`` graph and load a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str — path to ``dual_head_model.pt``
    (remaining args same as ``build_dual_head_model``)

    Returns
    -------
    DualHeadBiLSTMFiLM in eval mode on ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_dual_head_model(
        window_size=window_size, n_ts_features=n_ts_features,
        n_meta_features=n_meta_features, latent_dim=latent_dim,
        lstm_units=lstm_units, dropout_rate=dropout_rate,
        fc_hidden_dim=fc_hidden_dim, device=device,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded dual-head weights from '%s' → device=%s", checkpoint_path, device)
    return model


# ===========================================================================
# ─── SELF-TEST ───────────────────────────────────────────────────────────────
# ===========================================================================
if __name__ == "__main__":
    WINDOW, TS_F, META_F, BATCH = 50, 7, 2, 4
    LATENT = 64
    device = torch.device("cpu")

    print("\n" + "="*60)
    print("  Self-Test: BiLSTMFiLMAutoencoder (single-head)")
    print("="*60)
    m1 = build_model(WINDOW, TS_F, META_F, LATENT, device=device)
    ts   = torch.rand(BATCH, WINDOW, TS_F)
    meta = torch.rand(BATCH, META_F)
    with torch.no_grad():
        recon  = m1(ts, meta)
        latent = m1.encode(ts, meta)
    print(f"  Input       : {tuple(ts.shape)}")
    print(f"  Latent      : {tuple(latent.shape)}")
    print(f"  Recon       : {tuple(recon.shape)}  range=[{recon.min():.3f},{recon.max():.3f}]")
    assert tuple(recon.shape)  == (BATCH, WINDOW, TS_F)
    assert tuple(latent.shape) == (BATCH, LATENT)
    assert 0.0 <= recon.min() and recon.max() <= 1.0
    print("  [OK] Single-head test PASSED.")

    print("\n" + "="*60)
    print("  Self-Test: DualHeadBiLSTMFiLM")
    print("="*60)
    m2 = build_dual_head_model(WINDOW, TS_F, META_F, LATENT, device=device)
    with torch.no_grad():
        recon2, forecast2 = m2(ts, meta)
        latent2 = m2.encode(ts, meta)
    print(f"  Input       : {tuple(ts.shape)}")
    print(f"  Latent      : {tuple(latent2.shape)}")
    print(f"  Recon (H1)  : {tuple(recon2.shape)}  range=[{recon2.min():.3f},{recon2.max():.3f}]")
    print(f"  Forecast(H2): {tuple(forecast2.shape)} range=[{forecast2.min():.3f},{forecast2.max():.3f}]")
    assert tuple(recon2.shape)    == (BATCH, WINDOW, TS_F)
    assert tuple(forecast2.shape) == (BATCH, TS_F)
    assert tuple(latent2.shape)   == (BATCH, LATENT)

    print("\n" + "="*60)
    print("  Self-Test: HybridAnomalyLoss")
    print("="*60)
    criterion = HybridAnomalyLoss(alpha=0.5)
    next_step = torch.rand(BATCH, TS_F)
    total, l_r, l_f = criterion(recon2, ts, forecast2, next_step)
    print(f"  {criterion}")
    print(f"  Total loss  : {total.item():.6f}")
    print(f"  Recon MSE   : {l_r.item():.6f}")
    print(f"  Fore  MSE   : {l_f.item():.6f}")
    assert abs(total.item() - (0.5 * l_r.item() + 0.5 * l_f.item())) < 1e-6
    print("  [OK] HybridAnomalyLoss test PASSED.")
    print("\n  [OK] All self-tests PASSED.\n")
