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

from .scoring import *
from .rca_statistical import *
from .rca_genai import *
from .formatters import *

class IncidentPipeline:

    def __init__(
        self,
        scorer:           SeverityScorer,
        rca:              ContextAwareRCA,
        alerter:          AlertGenerator,
        adaptive_engine:  Optional[AdaptiveThresholdEngine] = None,
    ) -> None:
        self.scorer          = scorer
        self.rca             = rca
        self.alerter         = alerter
        self.adaptive_engine = adaptive_engine

    def process(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        mse_score:     float,
        context:       ContainerContext,
        timestamp_utc: Optional[str] = None,
    ) -> IncidentEvent:
        if timestamp_utc is None:
            timestamp_utc = (
                datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "Z"
            )
        event_id = str(uuid.uuid4())

        if self.adaptive_engine is not None:
            severity = self.adaptive_engine.score(mse_score)
        else:
            severity = self.scorer.score(mse_score)

        if self.adaptive_engine is not None:
            admitted = self.adaptive_engine.observe(mse_score, severity)
            logger.debug(
                "AdaptiveEngine observation: mse=%.6f severity=%s admitted=%s "
                "window=%d/%d drift=%.3f×",
                mse_score, severity, admitted,
                len(self.adaptive_engine._window),
                self.adaptive_engine._window_size,
                self.adaptive_engine.drift_magnitude,
            )

        rca_result = self.rca.analyze(original, reconstructed, context)

        payload_dict = self.alerter.build_payload(
            context=context,
            severity=severity,
            mse_score=mse_score,
            rca_result=rca_result,
            timestamp_utc=timestamp_utc,
            event_id=event_id,
        )

        event = IncidentEvent(
            event_id=event_id,
            timestamp_utc=timestamp_utc,
            context=context,
            severity=severity,
            mse_score=float(mse_score),
            primary_metric=rca_result["primary_metric"],
            primary_display=rca_result["primary_display"],
            anomaly_name=rca_result["anomaly_name"],
            diagnosis=rca_result["diagnosis"],
            recommended_action=rca_result["recommended_action"],
            escalation_path=rca_result["escalation_path"],
            icon=rca_result["icon"],
            runbook_url=rca_result["runbook_url"],
            feature_errors=rca_result["feature_errors"],
            alert_payload=json.dumps(payload_dict, indent=2),
        )

        logger.info(
            "IncidentEvent | %s | [%s] %s container=%s mse=%.6f → %s",
            event_id[:8], severity, context.container_type,
            context.container_id, mse_score, rca_result["primary_metric"],
        )
        return event

    def process_batch(
        self,
        originals:      np.ndarray,
        reconstructions: np.ndarray,
        mse_scores:     np.ndarray,
        contexts:       List[ContainerContext],
        timestamps_utc: Optional[List[str]] = None,
    ) -> List[IncidentEvent]:
        n = len(mse_scores)
        if timestamps_utc is None:
            now = datetime.datetime.utcnow()
            timestamps_utc = [
                (now + datetime.timedelta(seconds=i)).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )[:-3] + "Z"
                for i in range(n)
            ]

        events = [
            self.process(
                original=originals[i],
                reconstructed=reconstructions[i],
                mse_score=float(mse_scores[i]),
                context=contexts[i],
                timestamp_utc=timestamps_utc[i],
            )
            for i in range(n)
        ]

        sev_counts = {s: sum(1 for e in events if e.severity == s)
                      for s in ("Critical", "High", "Warning", "Normal")}
        logger.info(
            "Batch complete: %d events | Critical=%d High=%d Warning=%d Normal=%d",
            len(events), sev_counts["Critical"], sev_counts["High"],
            sev_counts["Warning"], sev_counts["Normal"],
        )
        return events

    def __repr__(self) -> str:
        return (
            f"IncidentPipeline(\n"
            f"  scorer  = {self.scorer!r}\n"
            f"  rca     = {self.rca!r}\n"
            f"  alerter = {self.alerter!r}\n"
            f")"
        )

class SynchronousAlertDispatcher:

    def __init__(
        self,
        pipeline:       IncidentPipeline,
        genai_engine:   GenAIRCAEngine,
        threshold_p95:  float,
        threshold_p98:  float = 0.0,
        threshold_p995: float = 0.0,
        dry_run:        bool  = False,
    ) -> None:
        self.pipeline       = pipeline
        self.genai_engine   = genai_engine
        self.threshold_p95  = float(threshold_p95)
        self.threshold_p98  = float(threshold_p98)  or float(threshold_p95)
        self.threshold_p995 = float(threshold_p995) or float(threshold_p95)
        self.dry_run        = dry_run

    def run(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        mse_score:     float,
        context:       ContainerContext,
        bot_token:     str           = "",
        chat_id:       str           = "",
        recent_logs:   str           = "",
        timestamp_utc: Optional[str] = None,
    ) -> Dict:
        if timestamp_utc is None:
            timestamp_utc = (
                datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )

        result: Dict = {
            "triggered":       False,
            "incident":        None,
            "llm_analysis":    None,
            "final_severity":  "Normal",
            "is_true_anomaly": False,
            "used_fallback":   False,
            "telegram_result": {"success": False, "error": "not triggered"},
            "html_message":    "",
        }

        if mse_score <= self.threshold_p95:
            logger.debug(
                "SynchronousAlertDispatcher: mse=%.6f ≤ threshold=%.6f — no alert.",
                mse_score, self.threshold_p95,
            )
            return result

        result["triggered"] = True
        logger.info(
            "SynchronousAlertDispatcher: anomaly gate triggered — "
            "mse=%.6f (%.1f× threshold=%.6f)",
            mse_score, mse_score / self.threshold_p95, self.threshold_p95,
        )

        incident = self.pipeline.process(
            original=original, reconstructed=reconstructed,
            mse_score=mse_score, context=context, timestamp_utc=timestamp_utc,
        )
        result["incident"] = incident

        total_err   = sum(incident.feature_errors.values()) or 1e-12
        rca_for_llm: Dict = {
            "primary_metric":        incident.primary_metric,
            "primary_display":       incident.primary_display,
            "primary_error_percent": (
                incident.feature_errors.get(incident.primary_metric, 0)
                / total_err * 100
            ),
            "diagnosis":          incident.diagnosis,
            "recommended_action": incident.recommended_action,
            "feature_errors":     incident.feature_errors,
            "feature_errors_ranked": [
                {
                    "feature":      k,
                    "display_name": FEATURE_META.get(k, _UNKNOWN_META)["display_name"],
                    "short_name":   FEATURE_META.get(k, _UNKNOWN_META)["short_name"],
                    "icon":         FEATURE_META.get(k, _UNKNOWN_META)["icon"],
                    "mse":          round(v, 8),
                    "weighted_mse": round(
                        v * FEATURE_META.get(k, _UNKNOWN_META)["severity_weight"], 8
                    ),
                    "error_pct": round(v / total_err * 100, 2),
                }
                for k, v in sorted(
                    incident.feature_errors.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "context_summary": {},
        }

        llm_analysis:  Optional[LLMAnalysis] = None
        used_fallback: bool                  = False

        logger.info(
            "[STEP 2] Calling LLM synchronously — Telegram is BLOCKED "
            "until this returns (timeout=%gs).",
            self.genai_engine.timeout_seconds,
        )
        try:
            llm_analysis = self.genai_engine.analyze_with_llm(
                original_metrics      = original,
                reconstructed_metrics = reconstructed,
                mse                   = mse_score,
                context               = context,
                rca_result            = rca_for_llm,
                recent_logs           = recent_logs,
                threshold_p95         = self.threshold_p95,
                timestamp_utc         = timestamp_utc,
            )
            logger.info(
                "[STEP 2] LLM returned: verdict=%s verified_severity=%s "
                "confidence=%d%% latency=%.0f ms",
                llm_analysis.verdict,
                llm_analysis.verified_severity or "—",
                llm_analysis.confidence_pct,
                llm_analysis.latency_ms,
            )
        except Exception as llm_err:
            logger.error(
                "[STEP 2] LLM raised unexpected exception (%s) — "
                "building fallback LLMAnalysis.", llm_err,
            )
            used_fallback = True
            llm_analysis  = LLMAnalysis(
                verdict="UNCERTAIN",
                root_cause=(
                    f"LLM call failed ({type(llm_err).__name__}). "
                    f"Statistical diagnosis: {incident.diagnosis[:200]}"
                ),
                mitigation=incident.recommended_action.split("\n")[0],
                confidence_pct=0,
                model_used="FALLBACK (LLM unavailable)",
                raw_response="",
                prompt_tokens=0,
                latency_ms=0.0,
                success=False,
                verified_severity="",
                error_message=str(llm_err),
            )

        result["llm_analysis"]  = llm_analysis
        result["used_fallback"] = used_fallback

        model_severity  = incident.severity
        final_severity  = llm_analysis.effective_severity(model_severity)
        is_true_anomaly = llm_analysis.is_true_anomaly

        result["final_severity"]  = final_severity
        result["is_true_anomaly"] = is_true_anomaly

        logger.info(
            "[STEP 3] Severity resolved — model=%s → final=%s "
            "(is_true_anomaly=%s, llm_override=%s)",
            model_severity, final_severity,
            is_true_anomaly, llm_analysis.verified_severity or "—",
        )

        logger.info(
            "[STEP 4] Dispatching ONE Telegram alert "
            "(final_severity=%s, dry_run=%s)",
            final_severity, self.dry_run,
        )

        html_msg = format_telegram_alert(
            event             = incident,
            llm_analysis      = llm_analysis,
            threshold_p95     = self.threshold_p95,
            verified_severity = final_severity,
        )
        result["html_message"] = html_msg

        try:
            import sys
            import os
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            
            from alibaba_trace.utils.notifications import send_telegram_alert as centralized_send
            tg_result = centralized_send(
                message=html_msg,
                bot_token=bot_token,
                chat_id=chat_id,
                parse_mode="HTML",
                disable_notification=final_severity in ("Warning", "Normal"),
                dry_run=self.dry_run
            )
        except Exception as exc:
            logger.warning("Telegram send failed inside incident_response proxy (non-fatal): %s", exc)
            tg_result = {"success": False, "error": str(exc), "status_code": 0}
        result["telegram_result"] = tg_result

        logger.info(
            "[STEP 4] Telegram dispatch complete — success=%s msg_id=%s",
            tg_result.get("success"), tg_result.get("message_id"),
        )
        return result

    def __repr__(self) -> str:
        return (
            f"SynchronousAlertDispatcher("
            f"threshold_p95={self.threshold_p95:.6f}, "
            f"genai={self.genai_engine.openai_model}, "
            f"dry_run={self.dry_run})"
        )
