"""
genai_auditor.py
================
Unified GenAI provider abstraction for Tier-2 deep verification.

Supports:
  • Google Gemini  (primary  – set GENAI_PROVIDER=gemini)
  • OpenAI GPT     (fallback – set GENAI_PROVIDER=openai)

Public API
----------
build_tier2_context(record, recent_logs, recent_events) -> dict
    Enriches a Tier-1 anomaly record into a rich context payload for the LLM.

route_genai_call(summary, config) -> tuple[dict, dict]
    Selects the right provider, calls it, and cross-falls back on failure.
    Returns (decision_dict, call_metadata_dict).
"""

from __future__ import annotations

import json
import time
from typing import Any

try:
    from config import GenAIConfig
    from gpt_adjudicator import (
        DEFAULT_GPT_INSTRUCTIONS,
        RECOMMENDED_ACTIONS,
        RESPONSE_JSON_SCHEMA,
        VALID_LABELS,
        VALID_SEVERITIES,
        fallback_decision,
        load_prompt_template,
        normalize_decision,
    )
    from utils import safe_literal_list
except ImportError:
    from .config import GenAIConfig
    from .gpt_adjudicator import (
        DEFAULT_GPT_INSTRUCTIONS,
        RECOMMENDED_ACTIONS,
        RESPONSE_JSON_SCHEMA,
        VALID_LABELS,
        VALID_SEVERITIES,
        fallback_decision,
        load_prompt_template,
        normalize_decision,
    )
    from .utils import safe_literal_list


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_tier2_context(
    record: dict[str, Any],
    recent_logs: list[str] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build an enriched context payload for Tier-2 LLM verification.

    Adds Tier-1 decision metadata (score ratio, individual scores, feature
    deviations) on top of the standard window summary fields so the LLM has
    a complete picture.
    """
    top_k_features = record.get("top_k_features", [])
    if isinstance(top_k_features, str):
        top_k_features = safe_literal_list(top_k_features)

    top_k_feature_errors = record.get("top_k_feature_errors", [])
    if isinstance(top_k_feature_errors, str):
        top_k_feature_errors = safe_literal_list(top_k_feature_errors)

    feature_error_vector = record.get("feature_error_vector", [])
    if isinstance(feature_error_vector, str):
        feature_error_vector = safe_literal_list(feature_error_vector)

    recon_score = record.get("recon_score")
    forecast_score = record.get("forecast_score")
    final_score = float(record.get("final_score", record.get("anomaly_score", 0.0)))
    threshold = float(record.get("threshold", record.get("final_threshold", 1e-6)) or 1e-6)
    score_ratio = final_score / max(threshold, 1e-9)

    container_meta = {
        "container_app_du": record.get("container_app_du", record.get("app_du", "unknown")),
        "container_status": record.get("container_status", "unknown"),
        "machine_status": record.get("machine_status", "unknown"),
        "machine_failure_domain_1": record.get("machine_failure_domain_1", "unknown"),
        "machine_failure_domain_2": record.get("machine_failure_domain_2", "unknown"),
    }

    return {
        # ── identity ──────────────────────────────────────────────────────
        "window_id": int(record.get("window_id", -1)),
        "container_id": str(record.get("container_id", "unknown")),
        "machine_id": str(record.get("machine_id", "unknown")),
        "split": str(record.get("split", "")),
        "time_range": {
            "start_time": int(record.get("start_time", -1)),
            "end_time": int(record.get("end_time", -1)),
        },
        # ── Tier-1 scoring detail ─────────────────────────────────────────
        "tier1": {
            "mode": str(record.get("mode", "hybrid")),
            "recon_score": float(recon_score) if recon_score is not None else None,
            "forecast_score": float(forecast_score) if forecast_score is not None else None,
            "final_score": final_score,
            "threshold": threshold,
            "score_ratio": round(score_ratio, 4),
            "score_over_threshold": float(record.get("score_over_threshold", 0.0)),
            "dynamic_threshold": record.get("dynamic_threshold"),
            "z_score": record.get("z_score"),
            "smoothed_score": record.get("smoothed_score"),
            "decision": str(record.get("decision", record.get("status", "unknown"))),
            "decision_reason": str(record.get("decision_reason", "")),
            "consecutive_breach_count": record.get("current_consecutive_breach_count"),
        },
        # ── feature deviation ─────────────────────────────────────────────
        "top_k_features": top_k_features,
        "top_k_feature_errors": [float(v) for v in top_k_feature_errors],
        "feature_error_vector": [float(v) for v in feature_error_vector],
        # ── container / machine context ───────────────────────────────────
        "container_metadata": container_meta,
        # ── enriched observability ────────────────────────────────────────
        "recent_logs": (recent_logs or [])[:20],
        "recent_events": (recent_events or [])[:10],
    }


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

_GEMINI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": VALID_LABELS,
        },
        "severity": {
            "type": "string",
            "enum": VALID_SEVERITIES,
        },
        "root_cause": {"type": "string"},
        "impact_analysis": {"type": "string"},
        "step_by_step_recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "explanation": {"type": "string"},
        "recommended_action": {
            "type": "string",
            "enum": RECOMMENDED_ACTIONS,
        },
    },
    "required": [
        "label",
        "severity",
        "root_cause",
        "impact_analysis",
        "step_by_step_recommendations",
        "explanation",
        "recommended_action",
    ],
}


def call_gemini_api(
    summary: dict[str, Any],
    config: GenAIConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Call Google Gemini with structured JSON output enforcement.

    Returns (decision, call_metadata).
    """
    api_key = config.gemini_api_key
    if not api_key:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "missing_gemini_api_key",
            "provider": "gemini",
        }

    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "google_generativeai_not_installed",
            "provider": "gemini",
        }

    system_prompt = load_prompt_template(config.prompt_template_path)
    user_content = json.dumps(summary, ensure_ascii=True)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=config.gemini_model,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=_GEMINI_JSON_SCHEMA,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        t0 = time.perf_counter()
        response = model.generate_content(user_content)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "gemini_request_failed",
            "error": str(exc),
            "provider": "gemini",
        }

    response_text = getattr(response, "text", "") or ""
    if not response_text:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "gemini_empty_response",
            "provider": "gemini",
        }

    try:
        decision = json.loads(response_text)
    except json.JSONDecodeError:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "gemini_invalid_json",
            "raw_response": response_text[:500],
            "provider": "gemini",
        }

    return normalize_decision(decision, summary), {
        "used_fallback": False,
        "provider": "gemini",
        "model": config.gemini_model,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# OpenAI call (thin wrapper around gpt_adjudicator path)
# ---------------------------------------------------------------------------

def call_openai_api(
    summary: dict[str, Any],
    config: GenAIConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Call OpenAI via the existing gpt_adjudicator logic.

    Returns (decision, call_metadata).
    """
    try:
        from gpt_adjudicator import GPTConfig, call_openai_responses_api  # type: ignore[import]
    except ImportError:
        from .gpt_adjudicator import call_openai_responses_api  # type: ignore[import]
        from .config import GPTConfig  # type: ignore[import]

    # Bridge GenAIConfig → GPTConfig
    gpt_cfg = GPTConfig(
        prompt_template_path=config.prompt_template_path,
        openai_api_key_env=config.openai_api_key_env,
        model_env=config.openai_model_env,
        default_model=config.openai_default_model,
        request_timeout_seconds=config.request_timeout_seconds,
    )

    t0 = time.perf_counter()
    decision, meta = call_openai_responses_api(summary, config=gpt_cfg)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    meta["provider"] = "openai"
    meta["latency_ms"] = latency_ms
    return decision, meta


# ---------------------------------------------------------------------------
# Provider router
# ---------------------------------------------------------------------------

def route_genai_call(
    summary: dict[str, Any],
    config: GenAIConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Route the Tier-2 LLM call to the configured provider.

    If the primary provider fails AND a secondary is available, automatically
    retries with the secondary. Always returns a valid decision (never raises).

    Returns (decision, call_metadata).
    """
    provider = str(config.provider).lower().strip()

    if provider == "gemini":
        primary_fn, secondary_fn = call_gemini_api, call_openai_api
        primary_name, secondary_name = "gemini", "openai"
    else:
        primary_fn, secondary_fn = call_openai_api, call_gemini_api
        primary_name, secondary_name = "openai", "gemini"

    decision, meta = primary_fn(summary, config)

    # If primary used fallback, try secondary
    if meta.get("used_fallback") and not meta.get("reason", "").startswith("missing"):
        secondary_decision, secondary_meta = secondary_fn(summary, config)
        if not secondary_meta.get("used_fallback"):
            secondary_meta["primary_failed"] = True
            secondary_meta["primary_reason"] = meta.get("reason")
            return secondary_decision, secondary_meta

    meta.setdefault("provider", primary_name)
    return decision, meta
