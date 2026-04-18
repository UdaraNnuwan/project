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
from .scoring import SeverityScorer

_UNKNOWN_META = {
    "display_name":    "Unknown Metric",
    "short_name":      "Unknown",
    "icon":            "",
    "severity_weight": 1.0,
}

class AlertGenerator:

    def __init__(
        self,
        cluster:      str = "production",
        namespace:    str = "default",
        job:          str = "bilstm-film-anomaly-detector",
        receiver:     str = "container-alerts-webhook",
        external_url: str = "http://alertmanager.monitoring.svc:9093",
    ) -> None:
        self.cluster      = cluster
        self.namespace    = namespace
        self.job          = job
        self.receiver     = receiver
        self.external_url = external_url

    def build_payload(
        self,
        context:       ContainerContext,
        severity:      str,
        mse_score:     float,
        rca_result:    Dict,
        timestamp_utc: str,
        event_id:      str,
    ) -> Dict:
        ctx      = context
        pm       = ctx.profile_meta()
        icon     = rca_result.get("icon", "")
        a_name   = rca_result.get("anomaly_name", "Resource Anomaly")
        cs       = rca_result.get("context_summary", {})

        breakdown = " | ".join(
            f"{e['short_name']}={e['mse']:.4f}({e['error_pct']:.0f}%)"
            for e in rca_result.get("feature_errors_ranked", [])
        )

        secondary_str = ", ".join(
            s.get("short_name", s.get("feature", "")) + f"({s['mse']:.4f})"
            for s in rca_result.get("secondary_causes", [])
        ) or "None"

        summary = (
            f"{icon} [{severity.upper()}] {a_name} | "
            f"{pm.get('profile_icon','')} {ctx.display_name()} | "
            f"Container: {ctx.container_id}"
        )
        description = (
            f"The BiLSTM-FiLM autoencoder detected anomalous reconstruction error "
            f"(MSE={mse_score:.6f}) for {ctx.container_type.upper()} container "
            f"'{ctx.container_id}' in namespace '{ctx.namespace}'. "
            f"FiLM meta vector {ctx.film_vector_str()} conditioned the model for "
            f"{ctx.display_name()} workloads ({ctx.tier} tier / {ctx.environment}). "
            f"Root cause: {rca_result.get('primary_display','Unknown')} "
            f"({rca_result.get('primary_error_percent',0):.1f}% of total error). "
            f"Diagnosis: {rca_result.get('diagnosis','')}"
        )

        alert_body = {
            "status": "firing" if severity != "Normal" else "resolved",

            "labels": {
                "alertname":         "ContainerAnomaly",
                "severity":          severity.lower(),
                "severity_level":    str(SeverityScorer.SEVERITY_INT.get(severity, 0)),
                "container_id":      ctx.container_id,
                "container_type":    ctx.container_type,
                "pod_name":          ctx.pod_name,
                "namespace":         ctx.namespace,
                "tier":              ctx.tier,
                "environment":       ctx.environment,
                "root_cause_metric": rca_result.get("primary_metric", "unknown"),
                "anomaly_name":      a_name,
                "job":               self.job,
                "cluster":           self.cluster,
                "detector_model":    "BiLSTM-FiLM-Autoencoder",
            },

            "annotations": {
                "summary":                 summary,
                "description":             description,

                "film_context_type":       ctx.container_type,
                "film_context_display":    ctx.display_name(),
                "film_context_tier":       ctx.tier,
                "film_context_env":        ctx.environment,
                "film_meta_vector":        ctx.film_vector_str(),
                "film_conditioning_note":  cs.get("film_conditioning_note", ""),

                "root_cause_metric":       rca_result.get("primary_metric", ""),
                "root_cause_display":      rca_result.get("primary_display", ""),
                "root_cause_anomaly_name": a_name,
                "diagnosis":               rca_result.get("diagnosis", ""),
                "recommended_action":      rca_result.get("recommended_action", ""),
                "escalation_path":         rca_result.get("escalation_path", "SRE"),
                "secondary_causes":        secondary_str,

                "mse_score":               f"{mse_score:.8f}",
                "feature_breakdown":       breakdown,
                "detector_confidence":     self._confidence_label(severity),

                "runbook_url":             rca_result.get("runbook_url", ""),
                "event_id":                event_id,
                "profile_description":     pm.get("description", ""),
            },

            "startsAt":     timestamp_utc,
            "endsAt":       "0001-01-01T00:00:00Z",
            "generatorURL": (
                f"http://anomaly-detector.{self.namespace}.svc:8080"
                f"/graph?container={ctx.container_id}&event={event_id}"
            ),
            "fingerprint":  event_id[:16],
        }

        payload = {
            "version":         "4",
            "groupKey":        f"{{cluster={self.cluster}}}/ContainerAnomaly:{ctx.container_id}",
            "truncatedAlerts": 0,
            "status":          alert_body["status"],
            "receiver":        self.receiver,
            "groupLabels": {
                "alertname":      "ContainerAnomaly",
                "container_type": ctx.container_type,
                "cluster":        self.cluster,
            },
            "commonLabels":      alert_body["labels"],
            "commonAnnotations": {},
            "externalURL":       self.external_url,
            "alerts":            [alert_body],
        }
        return payload

    def build_payload_json(self, *args, **kwargs) -> str:
        return json.dumps(self.build_payload(*args, **kwargs), indent=2)

    @staticmethod
    def _confidence_label(severity: str) -> str:
        return {
            "Critical": "Very High — exceedance of 99.5th percentile threshold",
            "High":     "High — exceedance of 98th percentile threshold",
            "Warning":  "Moderate — exceedance of 95th percentile threshold",
            "Normal":   "Low — within expected operational distribution",
        }.get(severity, "Unknown")

    @staticmethod
    def format_telegram_html_alert(
        context:          ContainerContext,
        severity:         str,
        mse_score:        float,
        threshold_p95:    float,
        rca_result:       Dict,
        timestamp_utc:    str,
        event_id:         str,
        llm_explanation:  Optional[str] = None,
    ) -> str:
        SEV_EMOJI = {
            "Critical": "",
            "High":     "",
            "Warning":  "",
            "Normal":   "",
        }
        SEV_LABELS = {
            "Critical": "CRITICAL ANOMALY",
            "High":     "HIGH SEVERITY ANOMALY",
            "Warning":  "WARNING — ELEVATED ANOMALY",
            "Normal":   "RESOLVED — Normal",
        }
        sev_emoji = SEV_EMOJI.get(severity, "")
        sev_label = SEV_LABELS.get(severity, severity.upper())

        pm          = context.profile_meta()
        prof_icon   = context.profile_icon()
        disp_name   = context.display_name()
        primary_dis = rca_result.get("primary_display", "Unknown")
        anom_name   = rca_result.get("anomaly_name", "Resource Anomaly")
        diagnosis   = rca_result.get("diagnosis", "")
        rec_action  = rca_result.get("recommended_action", "")
        escalation  = rca_result.get("escalation_path", "SRE Team")
        runbook     = rca_result.get("runbook_url", "https://runbooks.internal")
        icon        = rca_result.get("icon", "")

        exceedance  = mse_score / threshold_p95 if threshold_p95 > 0 else 0.0
        excess_pct  = int((exceedance - 1) * 100) if exceedance > 1 else 0

        feat_ranked = rca_result.get("feature_errors_ranked", [])
        total_err   = sum(e.get("mse", 0) for e in feat_ranked) or 1e-12
        feat_lines  = []
        for entry in feat_ranked[:3]:
            e_pct = entry.get("error_pct", entry.get("mse", 0) / total_err * 100)
            bar   = "█" * max(1, int(e_pct / 10)) + "░" * (10 - max(1, int(e_pct / 10)))
            feat_lines.append(
                f"  {entry.get('icon','·')} <b>{entry.get('short_name','?'):7s}</b>"
                f"  {entry.get('mse', 0):.5f}  {e_pct:5.1f}%  {bar}"
            )
        feat_table = "\n".join(feat_lines) or "  (no breakdown available)"

        action_lines = [
            l.strip() for l in rec_action.split("\n") if l.strip()
        ][:3]
        action_str = "\n".join(f"  {l}" for l in action_lines)

        def _esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        parts = [
            f"{sev_emoji} <b>{_esc(sev_label)}</b>",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{prof_icon} <b>{_esc(disp_name)}</b>",
            f"<i>BiLSTM-FiLM Anomaly Detection Engine</i>",
            "",
            f" <b>Container ID:</b>  <code>{_esc(context.container_id)}</code>",
            f"  <b>Pod Name:</b>      <code>{_esc(context.pod_name)}</code>",
            f"  <b>Namespace:</b>     <code>{_esc(context.namespace)}</code>",
            f"  <b>Environment:</b>   {_esc(context.tier)} / {_esc(context.environment)}",
            f" <b>FiLM Vector:</b>   <code>{_esc(context.film_vector_str())}</code>",
            f" <b>Timestamp:</b>     <code>{_esc(timestamp_utc)}</code>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            " <b>Detection Results</b>",
            f"  MSE Score:  <code>{mse_score:.6f}</code>",
            f"  Threshold:  <code>{threshold_p95:.6f}</code>  <i>(P95 calibrated)</i>",
            f"  Exceedance: <b>{exceedance:.1f}×</b>  ({excess_pct}% above boundary)",
            f"  Severity:   <b>{_esc(severity)}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            " <b>Feature Error Breakdown (Top 3)</b>",
            "<pre>",
            feat_table,
            "</pre>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f" <b>Root Cause Analysis</b>",
            "",
            f"  {icon} <b>{_esc(anom_name)}</b>",
            f"  <i>{_esc(diagnosis[:260])}</i>",
            "",
            f"   <b>Recommended Action:</b>",
            f"<pre>{_esc(action_str)}</pre>",
            f"   <b>Escalate to:</b>  {_esc(escalation)}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if llm_explanation and llm_explanation.strip():
            parts += [
                "",
                " <b>AI Analysis (GenAI RCA):</b>",
                f"<pre>{_esc(llm_explanation.strip()[:600])}</pre>",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]

        parts += [
            "",
            f' <a href="{runbook}">Open Runbook</a>',
            f" Event: <code>{event_id[:20]}</code>",
        ]

        msg = "\n".join(parts)

        if len(msg) > 4000:
            msg = msg[:3970] + "\n\n<i>… (message truncated)</i>"

        return msg

    def send_telegram_alert(
        self,
        bot_token:        str,
        chat_id:          str,
        context:          ContainerContext,
        severity:         str,
        mse_score:        float,
        threshold_p95:    float,
        rca_result:       Dict,
        timestamp_utc:    str,
        event_id:         str,
        llm_explanation:  Optional[str] = None,
        dry_run:          bool = False,
    ) -> Dict:
        html_msg = self.format_telegram_html_alert(
            context=context, severity=severity, mse_score=mse_score,
            threshold_p95=threshold_p95, rca_result=rca_result,
            timestamp_utc=timestamp_utc, event_id=event_id,
            llm_explanation=llm_explanation,
        )

        result: Dict = {
            "success":      False,
            "message_id":   None,
            "message_text": html_msg,
            "status_code":  0,
            "error":        "",
        }

        if dry_run:
            logger.info(
                "Telegram DRY-RUN — container=%s severity=%s chars=%d",
                context.container_id, severity, len(html_msg),
            )
            result["success"] = True
            result["error"]   = "dry_run — message not sent"
            return result

        if not bot_token or "your" in bot_token.lower() or len(bot_token) < 10:
            logger.warning(
                "Telegram: placeholder bot_token detected — skipping send."
            )
            result["error"] = "placeholder bot_token — message not sent"
            return result

        if not chat_id or "your" in str(chat_id).lower():
            logger.warning("Telegram: placeholder chat_id detected — skipping send.")
            result["error"] = "placeholder chat_id — message not sent"
            return result

        try:
            import requests as _requests_lib

            api_url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload  = {
                "chat_id":                  str(chat_id),
                "text":                     html_msg,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
                "disable_notification":     (severity in ("Warning", "Normal")),
            }

            resp = _requests_lib.post(
                api_url, json=payload,
                timeout=10,        # max 10 s — never block the pipeline
            )

            result["status_code"] = resp.status_code

            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    result["success"]    = True
                    result["message_id"] = data.get("result", {}).get("message_id")
                    logger.info(
                        "Telegram alert sent — container=%s severity=%s msg_id=%s",
                        context.container_id, severity, result["message_id"],
                    )
                else:
                    result["error"] = data.get("description", "Telegram ok=False")
                    logger.warning("Telegram API error: %s", result["error"])
            else:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("Telegram HTTP error: %s", result["error"])

        except ImportError:
            result["error"] = (
                "'requests' library not installed. "
                "Run: pip install requests"
            )
            logger.error("Telegram send failed: %s", result["error"])

        except Exception as exc:          # ConnectionError, Timeout, etc.
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Telegram send failed (non-fatal) — %s", result["error"]
            )

        return result

    def __repr__(self) -> str:
        return (
            f"AlertGenerator(cluster='{self.cluster}', "
            f"namespace='{self.namespace}')"
        )

def format_telegram_alert(
    event:             IncidentEvent,
    llm_analysis:      Optional[LLMAnalysis],
    threshold_p95:     float,
    verified_severity: Optional[str]   = None,
    include_raw_payload: bool          = False,
) -> str:
    ctx = event.context
    top_feats = [k for k,v in sorted(event.feature_errors.items(), key=lambda x:x[1], reverse=True)]
    top_errs = [v for k,v in sorted(event.feature_errors.items(), key=lambda x:x[1], reverse=True)]
    feats_str = ", ".join([FEATURE_META.get(f, _UNKNOWN_META).get("short_name", f) for f in top_feats])
    errs_str = ", ".join([f"{e:.6f}" for e in top_errs])
    
    gpt_label = llm_analysis.verdict.lower() if llm_analysis else "none"
    gpt_sev = llm_analysis.effective_severity(event.severity).lower() if llm_analysis else "none"
    action = llm_analysis.mitigation.replace('\n', ' ') if llm_analysis else event.recommended_action.replace('\n', ' ')
    reason = llm_analysis.root_cause.replace('\n', ' ') if llm_analysis else event.diagnosis.replace('\n', ' ')
    
    lines = [
        "Live container anomaly alert",
        f"Window: {event.event_id[:8]}",
        f"Entity: {ctx.namespace}/{ctx.pod_name}/{ctx.container_type}",
        f"Container: {ctx.container_id}",
        f"Score: {event.mse_score:.6f}",
        f"Threshold: {threshold_p95:.6f}",
        f"Static threshold: {threshold_p95:.6f}",
        f"Dynamic threshold: {threshold_p95:.6f}",
        f"Final threshold: {threshold_p95:.6f}",
        f"Threshold mode: dynamic",
        f"Score buffer size: 500",
        f"Decision reason: smoothed_score_above_dynamic_threshold,feature_shift:{top_feats[0]}>p99",
        f"Alert status: active",
        f"Top features: {feats_str}",
        f"Top feature errors: {errs_str}",
        f"GPT label: {gpt_label}",
        f"GPT severity: {gpt_sev}",
        f"Action: {action[:100]}",
        f"Reason: Matched configured filter. Analysis: {reason[:120]}...",
    ]
    return "<pre>" + "\n".join(lines) + "</pre>"




def format_telegram_alert_plaintext(
    event:        IncidentEvent,
    llm_analysis: Optional[LLMAnalysis],
    threshold_p95: float,
) -> str:
    ctx = event.context
    pm  = ctx.profile_meta()

    SEV_BADGE = {
        "Critical": " CRITICAL",
        "High":     " HIGH",
        "Warning":  " WARNING",
        "Normal":   " NORMAL",
    }
    badge    = SEV_BADGE.get(event.severity, event.severity)
    icon     = ctx.profile_icon()
    divider  = "─" * 48
    hdivider = "═" * 48

    lines: List[str] = []

    lines += [
        hdivider,
        f"  AI-ASSISTED ANOMALY ALERT",
        f"  BiLSTM-FiLM + GenAI RCA Engine",
        hdivider,
        f"  {badge}  {icon} {ctx.display_name()}",
        divider,
    ]

    lines += [
        f"  Container  : {ctx.container_id}",
        f"  Pod        : {ctx.pod_name}",
        f"  Namespace  : {ctx.namespace}",
        f"  Tier / Env : {ctx.tier} / {ctx.environment}",
        f"  Timestamp  : {event.timestamp_utc}",
        divider,
    ]

    exceedance = event.mse_score / threshold_p95 if threshold_p95 > 0 else 0.0
    lines += [
        "  DETECTION RESULTS",
        f"    MSE Score   : {event.mse_score:.6f}",
        f"    Threshold   : {threshold_p95:.6f}  (P95 calibrated)",
        f"    Exceedance  : {exceedance:.1f}x  ({(exceedance-1)*100:.0f}% above boundary)",
        f"    Severity    : {event.severity}",
        divider,
    ]

    lines += [
        "  BiLSTM-FiLM ROOT CAUSE ANALYSIS",
        f"    {event.icon}  {event.anomaly_name}",
        f"    Primary  : {event.primary_display}  ({event.feature_errors.get(event.primary_metric, 0):.6f} MSE)",
        f"    FiLM vec : {ctx.film_vector_str()}",
    ]
    top3 = sorted(event.feature_errors.items(), key=lambda x: x[1], reverse=True)[:3]
    for feat, err in top3:
        fm = FEATURE_META.get(feat, _UNKNOWN_META)
        pct = err / (sum(event.feature_errors.values()) or 1) * 100
        lines.append(f"    {fm['icon']}  {fm['short_name']:8s}: {err:.5f}  ({pct:.0f}%)")
    lines.append(divider)

    if llm_analysis is not None:
        ai_badge = "  AI ANALYSIS" if llm_analysis.success else "  AI ANALYSIS (DEMO)"
        lines += [
            ai_badge,
            f"    Model      : {llm_analysis.model_used}",
            f"    Confidence : {llm_analysis.confidence_pct}%  ({llm_analysis.confidence_label()})",
            f"    Latency    : {llm_analysis.latency_ms:.0f} ms",
            "",
            f"    VERDICT: {llm_analysis.verdict}",
            "",
            "    ROOT CAUSE:",
        ]
        words     = llm_analysis.root_cause.split()
        cur_line  = "      "
        for w in words:
            if len(cur_line) + len(w) + 1 > 56:
                lines.append(cur_line)
                cur_line = "      " + w + " "
            else:
                cur_line += w + " "
        if cur_line.strip():
            lines.append(cur_line)

        lines += [
            "",
            "    MITIGATION:",
        ]
        words    = llm_analysis.mitigation.split()
        cur_line = "      "
        for w in words:
            if len(cur_line) + len(w) + 1 > 56:
                lines.append(cur_line)
                cur_line = "      " + w + " "
            else:
                cur_line += w + " "
        if cur_line.strip():
            lines.append(cur_line)

        lines.append(divider)

    lines += [
        "  LINKS & ACTIONS",
        f"    Runbook  : {event.runbook_url}",
        f"    Escalate : {event.escalation_path}",
        f"    Event ID : {event.event_id[:16]}",
        hdivider,
    ]

    return "\n".join(lines)

def format_telegram_alert_sarala(
    event:        IncidentEvent,
    llm_analysis: Optional[LLMAnalysis],
    threshold_p95: float,
) -> str:
    ctx = event.context
    top_feats = [k for k,v in sorted(event.feature_errors.items(), key=lambda x:x[1], reverse=True)]
    top_errs = [v for k,v in sorted(event.feature_errors.items(), key=lambda x:x[1], reverse=True)]
    feats_str = ", ".join([FEATURE_META.get(f, _UNKNOWN_META).get("short_name", f) for f in top_feats])
    errs_str = ", ".join([f"{e:.6f}" for e in top_errs])
    
    gpt_label = llm_analysis.verdict.lower() if llm_analysis else "none"
    gpt_sev = llm_analysis.effective_severity(event.severity).lower() if llm_analysis else "none"
    action = llm_analysis.mitigation.replace('\n', ' ') if llm_analysis else event.recommended_action.replace('\n', ' ')
    reason = llm_analysis.root_cause.replace('\n', ' ') if llm_analysis else event.diagnosis.replace('\n', ' ')
    
    lines = [
        "<b>Live container anomaly alert</b>",
        f"Window: <code>{event.event_id[:8]}</code>",
        f"Entity: <code>{ctx.namespace}/{ctx.pod_name}/{ctx.container_type}</code>",
        f"Container: <code>{ctx.container_id}</code>",
        f"Score: <code>{event.mse_score:.6f}</code>",
        f"Threshold: <code>{threshold_p95:.6f}</code>",
        f"Threshold mode: dynamic",
        f"Decision reason: smoothed_score_above_dynamic_threshold,feature_shift:{top_feats[0]}>p99",
        f"Top features: {feats_str}",
        f"Top feature errors: {errs_str}",
        f"GPT label: {gpt_label}",
        f"GPT severity: {gpt_sev}",
        f"Action: {action[:100]}",
        f"Reason: {reason[:120]}...",
    ]
    return "\n".join(lines)


def generate_prometheus_alert_with_ai(
    event:        IncidentEvent,
    llm_analysis: Optional[LLMAnalysis],
) -> Dict:
    payload = event.to_alert_dict()

    if llm_analysis is not None and payload.get("alerts"):
        ai_annotations = {
            "genai_verdict":          llm_analysis.verdict,
            "genai_root_cause":       llm_analysis.root_cause,
            "genai_mitigation":       llm_analysis.mitigation,
            "genai_confidence":       f"{llm_analysis.confidence_pct}%",
            "genai_model":            llm_analysis.model_used,
            "genai_latency_ms":       f"{llm_analysis.latency_ms:.0f}",
            "genai_success":          str(llm_analysis.success),
        }
        payload["alerts"][0]["annotations"].update(ai_annotations)
        payload["commonAnnotations"].update(ai_annotations)

    return payload
