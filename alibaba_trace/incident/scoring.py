from __future__ import annotations
import json
import uuid
import logging
import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
logger = logging.getLogger("incident_response")
FEATURE_META: Dict[str, Dict] = {
    "cpu_util_percent": {
        "display_name": "CPU Utilisation (%)",
        "short_name":   "CPU%",
        "icon":         "",
        "severity_weight": 1.30,
    },
    "mem_util_percent": {
        "display_name": "Memory Utilisation (%)",
        "short_name":   "Mem%",
        "icon":         "",
        "severity_weight": 1.40,
    },
    "cpu_request": {
        "display_name": "CPU Request (cores)",
        "short_name":   "CPUReq",
        "icon":         "",
        "severity_weight": 1.00,
    },
    "mem_request": {
        "display_name": "Memory Request (GiB)",
        "short_name":   "MemReq",
        "icon":         "",
        "severity_weight": 1.00,
    },
    "net_in": {
        "display_name": "Network Inbound (MB/s)",
        "short_name":   "Net↓",
        "icon":         "",
        "severity_weight": 1.20,
    },
    "net_out": {
        "display_name": "Network Outbound (MB/s)",
        "short_name":   "Net↑",
        "icon":         "",
        "severity_weight": 1.20,
    },
    "disk_io_percent": {
        "display_name": "Disk I/O Utilisation (%)",
        "short_name":   "Disk%",
        "icon":         "",
        "severity_weight": 1.10,
    },
}
from .models import *
class SeverityScorer:
    SEVERITY_INT: Dict[str, int] = {
        "Normal": 0, "Warning": 1, "High": 2, "Critical": 3
    }
    def __init__(
        self,
        p95:  Optional[float] = None,
        p98:  Optional[float] = None,
        p995: Optional[float] = None,
    ) -> None:
        self.p95  = p95
        self.p98  = p98
        self.p995 = p995
        self._calibrated = all(v is not None for v in [p95, p98, p995])
    def calibrate(self, baseline_mse: np.ndarray) -> "SeverityScorer":
        arr = np.asarray(baseline_mse, dtype=np.float64).ravel()
        if len(arr) < 4:
            raise ValueError(f"Need ≥ 4 values, got {len(arr)}.")
        self.p95  = float(np.percentile(arr, 95.0))
        self.p98  = float(np.percentile(arr, 98.0))
        self.p995 = float(np.percentile(arr, 99.5))
        self._calibrated = True
        logger.info(
            "SeverityScorer calibrated — P95=%.6f | P98=%.6f | P99.5=%.6f",
            self.p95, self.p98, self.p995,
        )
        return self
    def score(self, mse: float) -> str:
        if not self._calibrated:
            raise RuntimeError("Call .calibrate() first.")
        if mse >= self.p995: return "Critical"
        if mse >= self.p98:  return "High"
        if mse >= self.p95:  return "Warning"
        return "Normal"
    def score_batch(self, mse_array: np.ndarray) -> List[str]:
        return [self.score(float(m)) for m in mse_array]
    def threshold_summary(self) -> Dict[str, float]:
        if not self._calibrated:
            raise RuntimeError("Not calibrated.")
        return {"p95": self.p95, "p98": self.p98, "p99.5": self.p995}
    def __repr__(self) -> str:
        if self._calibrated:
            return (
                f"SeverityScorer(p95={self.p95:.6f}, "
                f"p98={self.p98:.6f}, p99.5={self.p995:.6f})"
            )
        return "SeverityScorer(uncalibrated)"
class AdaptiveThresholdEngine:
    def __init__(
        self,
        baseline_mse: np.ndarray,
        window_size:  int   = 10_000,
        min_window:   int   = 200,
        update_every: int   = 50,
        quantiles:    Tuple[float, float, float] = (95.0, 98.0, 99.5),
    ) -> None:
        arr = np.asarray(baseline_mse, dtype=np.float64).ravel()
        if len(arr) < 4:
            raise ValueError(
                f"AdaptiveThresholdEngine requires ≥ 4 baseline samples, got {len(arr)}."
            )
        self._window_size  = int(window_size)
        self._min_window   = int(min_window)
        self._update_every = int(update_every)
        self._q            = tuple(quantiles)   
        seed = arr[-self._window_size:]  
        self._window: Deque[float] = deque(seed.tolist(), maxlen=self._window_size)
        self._initial_p95  = float(np.percentile(arr, self._q[0]))
        self._initial_p98  = float(np.percentile(arr, self._q[1]))
        self._initial_p995 = float(np.percentile(arr, self._q[2]))
        self.p95:  float = self._initial_p95
        self.p98:  float = self._initial_p98
        self.p995: float = self._initial_p995
        self.total_observed: int = 0   
        self.total_admitted: int = 0   
        self.total_rejected: int = 0   
        self._admissions_since_update: int = 0
        logger.info(
            "AdaptiveThresholdEngine initialised | window_size=%d | "
            "min_window=%d | update_every=%d | "
            "initial P95=%.6f P98=%.6f P99.5=%.6f",
            self._window_size, self._min_window, self._update_every,
            self.p95, self.p98, self.p995,
        )
    def score(self, mse: float) -> str:
        if mse >= self.p995: return "Critical"
        if mse >= self.p98:  return "High"
        if mse >= self.p95:  return "Warning"
        return "Normal"
    def observe(self, mse: float, severity: str) -> bool:
        self.total_observed += 1
        if severity != "Normal":
            self.total_rejected += 1
            return False
        self._window.append(float(mse))
        self.total_admitted += 1
        self._admissions_since_update += 1
        if self._admissions_since_update >= self._update_every:
            self.update_thresholds()
        return True
    def update_thresholds(self) -> None:
        if self.total_admitted < self._min_window:
            logger.debug(
                "AdaptiveThreshold cold-start: admitted=%d < min_window=%d "
                "— keeping static thresholds.",
                self.total_admitted, self._min_window,
            )
            return
        w_arr = np.array(self._window, dtype=np.float64)
        prev_p95 = self.p95  
        self.p95  = float(np.percentile(w_arr, self._q[0]))
        self.p98  = float(np.percentile(w_arr, self._q[1]))
        self.p995 = float(np.percentile(w_arr, self._q[2]))
        self._admissions_since_update = 0  
        drift = self.drift_magnitude
        logger.info(
            "AdaptiveThreshold updated | window=%d/%d | "
            "P95: %.6f → %.6f (drift=%.3f×) | "
            "P98=%.6f P99.5=%.6f | "
            "admitted=%d rejected=%d",
            len(self._window), self._window_size,
            prev_p95, self.p95, drift,
            self.p98, self.p995,
            self.total_admitted, self.total_rejected,
        )
    @property
    def drift_magnitude(self) -> float:
        if self._initial_p95 == 0.0:
            return 1.0
        return round(self.p95 / self._initial_p95, 4)
    @property
    def window_fill_pct(self) -> float:
        return round(len(self._window) / self._window_size * 100, 1)
    @property
    def anomaly_rate(self) -> float:
        if self.total_observed == 0:
            return 0.0
        return round(self.total_rejected / self.total_observed, 4)
    @property
    def is_warm(self) -> bool:
        return self.total_admitted >= self._min_window
    def threshold_summary(self) -> Dict[str, float]:
        return {
            "p95":                float(self.p95),
            "p98":                float(self.p98),
            "p99.5":              float(self.p995),
            "initial_p95":        float(self._initial_p95),
            "initial_p98":        float(self._initial_p98),
            "initial_p99.5":      float(self._initial_p995),
            "drift_magnitude":    self.drift_magnitude,
            "window_size_current": len(self._window),
            "window_size_max":    self._window_size,
            "window_fill_pct":    self.window_fill_pct,
            "total_observed":     self.total_observed,
            "total_admitted":     self.total_admitted,
            "total_rejected":     self.total_rejected,
            "anomaly_rate":       self.anomaly_rate,
            "adaptive_mode":      self.is_warm,
        }
    def drift_report(self) -> str:
        s = self.threshold_summary()
        mode = "ADAPTIVE" if s["adaptive_mode"] else "COLD-START (static)"
        lines = [
            "─" * 60,
            f"  AdaptiveThresholdEngine — Concept Drift Report",
            "─" * 60,
            f"  Mode             : {mode}",
            f"  Drift Magnitude  : {s['drift_magnitude']:.4f}×  "
            f"(initial P95={s['initial_p95']:.6f} → current={s['p95']:.6f})",
            f"  Adaptive Thresholds:",
            f"    P95  (Warning ) : {s['p95']:.6f}",
            f"    P98  (High    ) : {s['p98']:.6f}",
            f"    P99.5(Critical) : {s['p99.5']:.6f}",
            f"  Window           : {s['window_size_current']:,} / {s['window_size_max']:,}  "
            f"({s['window_fill_pct']:.1f}% full)",
            f"  Inferences Seen  : {s['total_observed']:,}",
            f"  Admitted (normal): {s['total_admitted']:,}",
            f"  Rejected (anom.) : {s['total_rejected']:,}",
            f"  Live Anomaly Rate : {s['anomaly_rate']*100:.2f}%",
            "─" * 60,
        ]
        return "\n".join(lines)
    def __repr__(self) -> str:
        return (
            f"AdaptiveThresholdEngine("
            f"window={len(self._window)}/{self._window_size}, "
            f"p95={self.p95:.6f}, drift={self.drift_magnitude:.3f}×, "
            f"admitted={self.total_admitted}, rejected={self.total_rejected})"
        )
