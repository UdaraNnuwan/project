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

_GENERIC_PROFILE = {
    "_meta": {
        "display_name": "Generic Microservice",
        "profile_icon": "⚙️",
    },
    "expected_behavior": "Standard compute workloads.",
    "criticality": "Moderate",
}

CONTAINER_PROFILES: Dict[str, Dict] = {
    "api_gateway": {
        "_meta": {
            "display_name": "API Gateway",
            "profile_icon": "🔀",
        },
        "expected_behavior": "High network I/O, highly concurrent connection pools.",
        "criticality": "Critical",
    },
    "payment_svc": {
        "_meta": {
            "display_name": "Payment Service",
            "profile_icon": "💳",
        },
        "expected_behavior": "Low latency transactions.",
        "criticality": "Critical",
    },
    "worker_node": {
        "_meta": {
            "display_name": "Background Worker",
            "profile_icon": "🔨",
        },
        "expected_behavior": "High CPU utilization, batch processing.",
        "criticality": "Low",
    },
    "db_primary": {
        "_meta": {
            "display_name": "Primary Database",
            "profile_icon": "🗄️",
        },
        "expected_behavior": "High memory caching, high disk I/O.",
        "criticality": "Critical",
    }
}

@dataclass
class ContainerContext:
    container_id:     str
    machine_id:       str
    container_type:   str   # Key into CONTAINER_PROFILES
    tier:             str
    environment:      str
    namespace:        str
    pod_name:         str
    film_meta_vector: np.ndarray  # Shape (M,)

    def profile(self) -> Dict:
        return CONTAINER_PROFILES.get(self.container_type, _GENERIC_PROFILE)

    def profile_meta(self) -> Dict:
        return self.profile().get("_meta", _GENERIC_PROFILE["_meta"])

    def display_name(self) -> str:
        return self.profile_meta().get("display_name", "Unknown")

    def profile_icon(self) -> str:
        return self.profile_meta().get("profile_icon", "")

    def film_vector_str(self) -> str:
        vals = ", ".join(f"{v:.4f}" for v in self.film_meta_vector)
        return f"[{vals}]"

    def environment_tier_label(self) -> str:
        return f"{self.environment.capitalize()} / {self.tier}"

    def to_dict(self) -> Dict:
        return {
            "container_id":     self.container_id,
            "machine_id":       self.machine_id,
            "container_type":   self.container_type,
            "container_type_display": self.display_name(),
            "tier":             self.tier,
            "environment":      self.environment,
            "namespace":        self.namespace,
            "pod_name":         self.pod_name,
            "film_meta_vector": self.film_vector_str(),
            "film_meta_dim":    len(self.film_meta_vector),
            "profile_icon":     self.profile_icon(),
        }

    @classmethod
    def from_ids(
        cls,
        container_id:   str,
        machine_id:     str,
        container_type: str,
        tier:           str            = "backend",
        environment:    str            = "production",
        namespace:      str            = "default",
        pod_name:       Optional[str]  = None,
    ) -> "ContainerContext":
        LARGE_PRIME = 999_983
        cid_val = float(hash(container_id) % LARGE_PRIME) / LARGE_PRIME
        mid_val = float(hash(machine_id)   % LARGE_PRIME) / LARGE_PRIME
        film_vec = np.array([cid_val, mid_val], dtype=np.float32)

        return cls(
            container_id=container_id,
            machine_id=machine_id,
            container_type=container_type,
            tier=tier,
            environment=environment,
            namespace=namespace,
            pod_name=pod_name or f"{container_type}-pod-{container_id[:6]}",
            film_meta_vector=film_vec,
        )

    def __repr__(self) -> str:
        return (
            f"ContainerContext("
            f"container_id='{self.container_id}', "
            f"type='{self.container_type}', "
            f"tier='{self.tier}/{self.environment}', "
            f"film={self.film_vector_str()})"
        )

@dataclass(frozen=True)
class LLMAnalysis:
    verdict:        str
    root_cause:     str
    mitigation:     str
    confidence_pct: int
    model_used:     str
    raw_response:   str
    prompt_tokens:  int
    latency_ms:     float
    success:           bool
    verified_severity: str  = ""
    error_message:     str  = ""

    @property
    def is_true_anomaly(self) -> bool:
        return self.is_genuine()

    def is_genuine(self) -> bool:
        return "GENUINE" in self.verdict.upper()

    def effective_severity(self, model_severity: str) -> str:
        if "FALSE POSITIVE" in self.verdict.upper():
            return "Normal"
        if self.verified_severity in ("Critical", "High", "Warning", "Normal"):
            return self.verified_severity
        return model_severity

    def confidence_label(self) -> str:
        if self.confidence_pct >= 90: return "Very High"
        if self.confidence_pct >= 75: return "High"
        if self.confidence_pct >= 55: return "Moderate"
        return "Low"

    def to_dict(self) -> Dict:
        return {
            "verdict":        self.verdict,
            "root_cause":     self.root_cause,
            "mitigation":     self.mitigation,
            "confidence_pct": self.confidence_pct,
            "model_used":     self.model_used,
            "latency_ms":     round(self.latency_ms, 1),
            "success":        self.success,
            "error_message":  self.error_message,
        }

    def __str__(self) -> str:
        flag = "[AI]" if self.success else "[AI-DEMO]"
        return (
            f"{flag} {self.verdict}  (confidence={self.confidence_pct}%)  "
            f"via {self.model_used}  ({self.latency_ms:.0f} ms)\n"
            f"ROOT CAUSE: {self.root_cause}\n"
            f"MITIGATION: {self.mitigation}"
        )

@dataclass(frozen=True)
class IncidentEvent:
    event_id:           str
    timestamp_utc:      str
    context:            ContainerContext   # Full FiLM metadata
    severity:           str
    mse_score:          float
    primary_metric:     str               # FEATURE_COLS column name
    primary_display:    str               # Human-readable name
    anomaly_name:       str               # Profile-specific label
    diagnosis:          str               # Context-aware explanation
    recommended_action: str               # Step-by-step remediation
    escalation_path:    str               # Who to page
    icon:               str               # Feature icon emoji
    runbook_url:        str
    feature_errors:     Dict[str, float]  # {feature: per-feature MSE}
    alert_payload:      str               # Pre-rendered JSON string

    def to_alert_dict(self) -> Dict:
        return json.loads(self.alert_payload)

    def to_alert_json(self, indent: int = 2) -> str:
        return json.dumps(json.loads(self.alert_payload), indent=indent)

    def severity_colour(self) -> str:
        return {
            "Critical": "\033[91m",
            "High":     "\033[93m",
            "Warning":  "\033[94m",
            "Normal":   "\033[92m",
        }.get(self.severity, "")

    def __str__(self) -> str:
        reset = "\033[0m"
        c = self.severity_colour()
        return (
            f"{c}[{self.severity:>8}]{reset} "
            f"{self.context.profile_icon()} {self.context.container_type:>12} | "
            f"{self.context.container_id:20s} | "
            f"MSE={self.mse_score:.6f} | "
            f"{self.icon} {self.primary_display}"
        )
