"""
two_tier_verifier.py
====================
TwoTierVerifier — the single public entry point for the full Two-Tier
Verification pipeline.

Tier 1 (BiLSTM-FiLM Hybrid Scorer)
    Pre-computed hybrid scores (reconstruction + forecasting) are read from
    the incoming record dict. A configurable score-ratio gate decides whether
    the window graduates to Tier 2.

Tier 2 (GenAI Auditor)
    Only triggered for Tier-1 confirmed anomalies that exceed
    ``config.tier1_min_score_ratio``. Calls ``route_genai_call()`` with the
    enriched context package and returns the full structured decision.

Usage
-----
    from src.two_tier_verifier import TwoTierVerifier
    from src.config import GenAIConfig

    verifier = TwoTierVerifier(config=GenAIConfig())
    result = verifier.verify(record_dict)
    print(result.final_label, result.tier2_provider)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    from config import GenAIConfig
    from genai_auditor import build_tier2_context, route_genai_call
    from gpt_adjudicator import fallback_decision
except ImportError:
    from .config import GenAIConfig
    from .genai_auditor import build_tier2_context, route_genai_call
    from .gpt_adjudicator import fallback_decision


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TwoTierResult:
    """Full result returned by TwoTierVerifier.verify()."""

    # ── identity ────────────────────────────────────────────────────────────
    window_id: int
    container_id: str
    machine_id: str

    # ── Tier-1 info ──────────────────────────────────────────────────────────
    tier1_flagged: bool                     # Did Tier-1 flag this as anomaly?
    tier1_score: float
    tier1_threshold: float
    tier1_score_ratio: float
    tier1_recon_score: float | None
    tier1_forecast_score: float | None
    tier1_mode: str
    tier1_decision_reason: str

    # ── Tier-2 info ──────────────────────────────────────────────────────────
    tier2_triggered: bool                   # Was Tier-2 actually called?
    tier2_provider: str | None              # "gemini" | "openai" | None
    tier2_model: str | None
    tier2_latency_ms: float | None
    tier2_used_fallback: bool
    tier2_fallback_reason: str | None

    # ── Final decision (from Tier-2, or rule-based if Tier-2 not triggered) ─
    final_label: str
    final_severity: str
    final_root_cause: str
    final_impact_analysis: str
    final_step_by_step_recommendations: list[str] = field(default_factory=list)
    final_explanation: str = ""
    final_recommended_action: str = "ignore"

    # ── Raw payloads ─────────────────────────────────────────────────────────
    tier2_context: dict[str, Any] | None = None
    tier2_raw_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "container_id": self.container_id,
            "machine_id": self.machine_id,
            "tier1_flagged": self.tier1_flagged,
            "tier1_score": self.tier1_score,
            "tier1_threshold": self.tier1_threshold,
            "tier1_score_ratio": self.tier1_score_ratio,
            "tier1_recon_score": self.tier1_recon_score,
            "tier1_forecast_score": self.tier1_forecast_score,
            "tier1_mode": self.tier1_mode,
            "tier1_decision_reason": self.tier1_decision_reason,
            "tier2_triggered": self.tier2_triggered,
            "tier2_provider": self.tier2_provider,
            "tier2_model": self.tier2_model,
            "tier2_latency_ms": self.tier2_latency_ms,
            "tier2_used_fallback": self.tier2_used_fallback,
            "tier2_fallback_reason": self.tier2_fallback_reason,
            "final_label": self.final_label,
            "final_severity": self.final_severity,
            "final_root_cause": self.final_root_cause,
            "final_impact_analysis": self.final_impact_analysis,
            "final_step_by_step_recommendations": self.final_step_by_step_recommendations,
            "final_explanation": self.final_explanation,
            "final_recommended_action": self.final_recommended_action,
        }


# ---------------------------------------------------------------------------
# TwoTierVerifier
# ---------------------------------------------------------------------------

class TwoTierVerifier:
    """
    Orchestrates the Two-Tier Verification pipeline.

    Parameters
    ----------
    config : GenAIConfig
        Controls provider selection, thresholds, and API credentials.
    """

    def __init__(self, config: GenAIConfig | None = None) -> None:
        self.config = config or GenAIConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def verify(
        self,
        record: dict[str, Any],
        recent_logs: list[str] | None = None,
        recent_events: list[dict[str, Any]] | None = None,
    ) -> TwoTierResult:
        """
        Run Tier-1 gate → optionally Tier-2 GenAI audit.

        Parameters
        ----------
        record : dict
            A single prediction/stream row (must contain anomaly_score /
            final_score, threshold, and optional recon/forecast scores).
        recent_logs : list[str], optional
            Recent log lines associated with this container window.
        recent_events : list[dict], optional
            Recent Kubernetes / system events for enriched context.

        Returns
        -------
        TwoTierResult
        """
        # ── Extract Tier-1 fields ────────────────────────────────────────
        tier1_score = float(
            record.get("final_score", record.get("anomaly_score", 0.0))
        )
        tier1_threshold = float(
            record.get("threshold", record.get("final_threshold", 1e-6)) or 1e-6
        )
        tier1_score_ratio = tier1_score / max(tier1_threshold, 1e-9)
        tier1_flagged = bool(
            record.get("confirmed_anomaly")
            or record.get("decision") == "confirmed_anomaly"
            or record.get("status") == "confirmed_anomaly"
        )
        recon_score = record.get("recon_score")
        forecast_score = record.get("forecast_score")

        # ── Tier-1 rule-based label (before LLM) ────────────────────────
        tier1_rule_decision = fallback_decision(
            {
                "anomaly_score": tier1_score,
                "threshold": tier1_threshold,
                "top_k_features": record.get("top_k_features", []),
                "mode": record.get("mode", "hybrid"),
            }
        )

        base = dict(
            window_id=int(record.get("window_id", -1)),
            container_id=str(record.get("container_id", "unknown")),
            machine_id=str(record.get("machine_id", "unknown")),
            tier1_flagged=tier1_flagged,
            tier1_score=tier1_score,
            tier1_threshold=tier1_threshold,
            tier1_score_ratio=round(tier1_score_ratio, 4),
            tier1_recon_score=float(recon_score) if recon_score is not None else None,
            tier1_forecast_score=float(forecast_score) if forecast_score is not None else None,
            tier1_mode=str(record.get("mode", "hybrid")),
            tier1_decision_reason=str(record.get("decision_reason", "")),
        )

        # ── Decide whether to invoke Tier-2 ─────────────────────────────
        min_ratio = float(self.config.tier1_min_score_ratio)
        tier2_should_run = (
            tier1_flagged
            and self.config.tier2_enabled
            and tier1_score_ratio >= min_ratio
        )

        if not tier2_should_run:
            return TwoTierResult(
                **base,
                tier2_triggered=False,
                tier2_provider=None,
                tier2_model=None,
                tier2_latency_ms=None,
                tier2_used_fallback=False,
                tier2_fallback_reason=None,
                final_label=tier1_rule_decision["label"] if tier1_flagged else "normal",
                final_severity=tier1_rule_decision["severity"] if tier1_flagged else "low",
                final_root_cause=tier1_rule_decision.get("root_cause", ""),
                final_impact_analysis=tier1_rule_decision.get("impact_analysis", ""),
                final_step_by_step_recommendations=tier1_rule_decision.get(
                    "step_by_step_recommendations", []
                ),
                final_explanation=tier1_rule_decision.get("explanation", ""),
                final_recommended_action=tier1_rule_decision.get("recommended_action", "ignore"),
                tier2_context=None,
                tier2_raw_decision=None,
            )

        # ── Build enriched context for Tier-2 ───────────────────────────
        context = build_tier2_context(
            record=record,
            recent_logs=recent_logs,
            recent_events=recent_events,
        )

        # ── Call GenAI provider ──────────────────────────────────────────
        decision, meta = route_genai_call(summary=context, config=self.config)

        return TwoTierResult(
            **base,
            tier2_triggered=True,
            tier2_provider=meta.get("provider"),
            tier2_model=meta.get("model"),
            tier2_latency_ms=meta.get("latency_ms"),
            tier2_used_fallback=bool(meta.get("used_fallback", False)),
            tier2_fallback_reason=meta.get("reason"),
            final_label=decision["label"],
            final_severity=decision["severity"],
            final_root_cause=decision.get("root_cause", ""),
            final_impact_analysis=decision.get("impact_analysis", ""),
            final_step_by_step_recommendations=decision.get(
                "step_by_step_recommendations", []
            ),
            final_explanation=decision.get("explanation", ""),
            final_recommended_action=decision.get("recommended_action", "ignore"),
            tier2_context=context,
            tier2_raw_decision=decision,
        )

    def verify_batch(
        self,
        records: list[dict[str, Any]],
        recent_logs_map: dict[str, list[str]] | None = None,
        recent_events_map: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[TwoTierResult]:
        """
        Run verify() over a list of records (sequential; no concurrency).

        ``recent_logs_map`` and ``recent_events_map`` are optional dicts
        keyed by ``container_id`` for per-container log/event injection.
        """
        results: list[TwoTierResult] = []
        for record in records:
            cid = str(record.get("container_id", ""))
            logs = (recent_logs_map or {}).get(cid)
            events = (recent_events_map or {}).get(cid)
            results.append(self.verify(record, recent_logs=logs, recent_events=events))
        return results
