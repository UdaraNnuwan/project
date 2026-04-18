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

_UNKNOWN_META = {"display_name": "Unknown", "short_name": "?", "icon": "❓", "severity_weight": 1.0}

_GENERIC_FEATURE_FALLBACK = {
    "diagnosis": "Unexplained metric deviation from anticipated boundaries.",
    "action": "Inspect Kubernetes events and container logs for localized failures.",
    "escalation": "L1 Platform Support",
}


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

class ContextAwareRCA:

    def __init__(
        self,
        feature_cols:        List[str],
        secondary_threshold: float = 0.60,
    ) -> None:
        self.feature_cols        = list(feature_cols)
        self.secondary_threshold = float(secondary_threshold)
        self._weights = np.array(
            [FEATURE_META.get(f, _UNKNOWN_META)["severity_weight"]
             for f in feature_cols],
            dtype=np.float64,
        )

    def analyze(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        context:       ContainerContext,
    ) -> Dict:
        original      = np.asarray(original,      dtype=np.float64)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Shape mismatch: original={original.shape} vs "
                f"reconstructed={reconstructed.shape}"
            )

        raw_errors      = np.mean((original - reconstructed) ** 2, axis=0)  # (F,)
        weighted_errors = raw_errors * self._weights

        total_mse = float(raw_errors.sum()) or 1e-12

        feature_errors:  Dict[str, float] = {}
        feature_ranked:  List[Dict]       = []

        for idx, feat in enumerate(self.feature_cols):
            fm   = FEATURE_META.get(feat, _UNKNOWN_META)
            r_e  = float(raw_errors[idx])
            w_e  = float(weighted_errors[idx])
            frac = r_e / total_mse

            feature_errors[feat] = r_e
            feature_ranked.append({
                "feature":      feat,
                "display_name": fm["display_name"],
                "short_name":   fm["short_name"],
                "icon":         fm["icon"],
                "mse":          round(r_e,  8),
                "weighted_mse": round(w_e,  8),
                "error_pct":    round(frac * 100, 2),
            })

        feature_ranked.sort(key=lambda x: x["weighted_mse"], reverse=True)

        primary      = feature_ranked[0]
        primary_feat = primary["feature"]

        profile_features = context.profile()
        feat_profile     = profile_features.get(primary_feat, _GENERIC_FEATURE_FALLBACK)

        profile_meta = context.profile_meta()
        runbook_url  = (
            profile_meta.get("runbook_base", "https://runbooks.internal") +
            "/" + primary_feat.replace("_", "-")
        )

        max_w = primary["weighted_mse"]
        secondary_causes = [
            {**f, **profile_features.get(f["feature"], _GENERIC_FEATURE_FALLBACK)}
            for f in feature_ranked[1:]
            if f["weighted_mse"] >= self.secondary_threshold * max_w
        ]

        context_summary = {
            "container_type":         context.container_type,
            "container_type_display": context.display_name(),
            "tier":                   context.tier,
            "environment":            context.environment,
            "film_meta_vector":       context.film_vector_str(),
            "film_conditioning_note": (
                f"FiLM layer conditioned the BiLSTM reconstruction on "
                f"container_id='{context.container_id}' "
                f"(meta_vec={context.film_vector_str()}), enabling context-aware "
                f"anomaly detection for {context.display_name()} workloads."
            ),
        }

        return {
            "primary_metric":        primary_feat,
            "primary_display":       primary["display_name"],
            "primary_short":         primary["short_name"],
            "primary_error":         primary["mse"],
            "primary_error_percent": primary["error_pct"],
            "anomaly_name":          feat_profile.get("anomaly_name", "Resource Anomaly"),
            "diagnosis":             feat_profile.get("diagnosis",    _GENERIC_FEATURE_FALLBACK["diagnosis"]),
            "recommended_action":    feat_profile.get("action",       _GENERIC_FEATURE_FALLBACK["action"]),
            "escalation_path":       feat_profile.get("escalation",   _GENERIC_FEATURE_FALLBACK["escalation"]),
            "icon":                  primary["icon"],
            "runbook_url":           runbook_url,
            "secondary_causes":      secondary_causes,
            "feature_errors":        feature_errors,
            "feature_errors_ranked": feature_ranked,
            "context_summary":       context_summary,
        }

    def __repr__(self) -> str:
        return (
            f"ContextAwareRCA("
            f"features={self.feature_cols}, "
            f"secondary_threshold={self.secondary_threshold})"
        )
