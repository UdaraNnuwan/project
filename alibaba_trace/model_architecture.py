from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
logger = logging.getLogger("model_architecture")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s - %(message)s",
)
class FiLMLayer(nn.Module):
    def __init__(
        self,
        latent_dim:      int,
        n_meta_features: int,
        hidden_dim:      Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(n_meta_features, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma_beta = nn.Linear(hidden_dim, 2 * latent_dim)
        nn.init.ones_(self.gamma_beta.weight[:latent_dim])   
        nn.init.zeros_(self.gamma_beta.weight[latent_dim:])  
        nn.init.zeros_(self.gamma_beta.bias)
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h          = self.cond_proj(cond)           
        gamma_beta = self.gamma_beta(h)             
        latent_dim = x.shape[-1]
        gamma = gamma_beta[:, :latent_dim]          
        beta  = gamma_beta[:, latent_dim:]          
        return gamma * x + beta                     
class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        n_ts_features: int,
        latent_dim:    int,
        lstm_units:    Tuple[int, int] = (128, 64),
        dropout_rate:  float = 0.2,
    ) -> None:
        super().__init__()
        H1, H2 = lstm_units
        self.bilstm1 = nn.LSTM(
            input_size=n_ts_features,
            hidden_size=H1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bn1   = nn.BatchNorm1d(H1 * 2)
        self.drop1 = nn.Dropout(dropout_rate)
        self.bilstm2 = nn.LSTM(
            input_size=H1 * 2,
            hidden_size=H2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout_rate)
        self.bottleneck = nn.Sequential(
            nn.Linear(H2 * 2, latent_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.bilstm1(x)                   
        out1 = self.bn1(out1.permute(0, 2, 1)).permute(0, 2, 1)
        out1 = self.drop1(out1)                     
        out2, _ = self.bilstm2(out1)                
        last = out2[:, -1, :]                       
        last = self.drop2(last)
        encoding = self.bottleneck(last)            
        return encoding
class BiLSTMDecoder(nn.Module):
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
        self.seed_proj = nn.Sequential(
            nn.Linear(latent_dim, H2 * 2),
            nn.ReLU(inplace=True),
        )
        self.bilstm1 = nn.LSTM(
            input_size=H2 * 2,
            hidden_size=H2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bn1   = nn.BatchNorm1d(H2 * 2)
        self.drop1 = nn.Dropout(dropout_rate)
        self.bilstm2 = nn.LSTM(
            input_size=H2 * 2,
            hidden_size=H1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout_rate)
        self.output_proj = nn.Sequential(
            nn.Linear(H1 * 2, n_ts_features),
            nn.Sigmoid(),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seed = self.seed_proj(z)                        
        seq  = seed.unsqueeze(1).expand(-1, self.window_size, -1)  
        out1, _ = self.bilstm1(seq)                     
        out1 = self.bn1(out1.permute(0, 2, 1)).permute(0, 2, 1)
        out1 = self.drop1(out1)
        out2, _ = self.bilstm2(out1)                    
        out2 = self.drop2(out2)
        reconstruction = self.output_proj(out2)         
        return reconstruction
class ForecastingHead(nn.Module):
    def __init__(
        self,
        n_ts_features: int,
        latent_dim:    int,
        fc_hidden_dim: Optional[int] = None,
        dropout_rate:  float = 0.2,
    ) -> None:
        super().__init__()
        if fc_hidden_dim is None:
            fc_hidden_dim = latent_dim * 2
        self.head = nn.Sequential(
            nn.Linear(latent_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, n_ts_features),
            nn.Sigmoid(),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)                            
class BiLSTMFiLMAutoencoder(nn.Module):
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
    def encode(self, ts: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ts)
        return self.film(z, meta)
    def forward(self, ts: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ts)        
        z = self.film(z, meta)      
        return self.decoder(z)      
class DualHeadBiLSTMFiLM(nn.Module):
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
        self.reconstruction_head = BiLSTMDecoder(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            window_size=window_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )
        self.forecasting_head = ForecastingHead(
            n_ts_features=n_ts_features,
            latent_dim=latent_dim,
            fc_hidden_dim=fc_hidden_dim,
            dropout_rate=dropout_rate,
        )
    def encode(self, ts: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ts)        
        return self.film(z, meta)   
    def forward(
        self, ts: torch.Tensor, meta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(ts)                   
        z = self.film(z, meta)                 
        reconstruction = self.reconstruction_head(z)   
        forecast = self.forecasting_head(z)            
        return reconstruction, forecast
class HybridAnomalyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self.alpha = alpha
        self._mse  = nn.MSELoss(reduction="mean")
    def forward(
        self,
        reconstruction: torch.Tensor,  
        ts_window:      torch.Tensor,  
        forecast:       torch.Tensor,  
        next_step:      torch.Tensor,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_recon = self._mse(reconstruction, ts_window)   
        loss_fore  = self._mse(forecast, next_step)          
        total_loss = self.alpha * loss_recon + (1.0 - self.alpha) * loss_fore
        return total_loss, loss_recon, loss_fore             
    def __repr__(self) -> str:
        return (
            f"HybridAnomalyLoss(alpha={self.alpha}  "
            f"[recon_weight={self.alpha:.2f}, "
            f"fore_weight={1-self.alpha:.2f}])"
        )
def build_model(
    window_size:     int,
    n_ts_features:   int,
    n_meta_features: int,
    latent_dim:      int = 64,
    lstm_units:      Tuple[int, int] = (128, 64),
    dropout_rate:    float = 0.2,
    device:          Optional[torch.device] = None,
) -> BiLSTMFiLMAutoencoder:
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
    logger.info("Loaded single-head weights from '%s' -> device=%s", checkpoint_path, device)
    return model
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
    logger.info("Loaded dual-head weights from '%s' -> device=%s", checkpoint_path, device)
    return model
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
