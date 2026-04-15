from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

try:
    from config import GPTConfig
    from utils import ensure_directory, safe_literal_list, write_json
except ImportError:
    from .config import GPTConfig
    from .utils import ensure_directory, safe_literal_list, write_json


RECOMMENDED_ACTIONS = [
    "ignore",
    "monitor",
    "raise_alert",
    "inspect_container",
    "restart_container",
    "reschedule_pod",
    "scale_service",
    "investigate_node",
]
VALID_LABELS = ["normal", "warning", "critical", "fault_candidate"]
VALID_SEVERITIES = ["low", "medium", "high"]

DEFAULT_GPT_INSTRUCTIONS = """
You are assisting a multivariate anomaly detector for container telemetry.

The underlying autoencoder has already detected a threshold-crossing anomaly candidate. Your job is to act as the final semantic decision layer and filter noise.

Analyze the compact window summary to:
1. Provide a Root Cause Analysis matching the top deviating features.
2. Produce an Impact Analysis assessing the threat to the container or cluster.
3. Classify the severity explicitly (low/medium/high).
4. Provide Step-by-Step Recommendations for SREs.
5. Determine if it is a false positive ("normal" label).

Return JSON only matching the schema exactly.
Be extremely conservative with escalation:
- choose "normal" when the threshold crossing looks like a likely false positive or benign spike.
- choose "warning" for mild but credible issues.
- choose "fault_candidate" for meaningful anomalies that need deep investigation.
- choose "critical" for strong evidence of an active severe fault (e.g., node death, OOM, extreme network saturation).

Keep the explanations and step-by-step instructions concise, avoiding bloated rhetoric.
""".strip()

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
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
            "items": {"type": "string"}
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
        "recommended_action"
    ],
}


def load_prompt_template(path: str | Path | None) -> str:
    if path is None:
        return DEFAULT_GPT_INSTRUCTIONS
    prompt_path = Path(path)
    if not prompt_path.exists():
        return DEFAULT_GPT_INSTRUCTIONS
    return prompt_path.read_text(encoding="utf-8").strip()


def build_window_summary(
    record: dict[str, Any],
    recent_logs: list[str] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    top_k_features = record.get("top_k_features", [])
    if isinstance(top_k_features, str):
        top_k_features = safe_literal_list(top_k_features)

    top_k_feature_errors = record.get("top_k_feature_errors", [])
    if isinstance(top_k_feature_errors, str):
        top_k_feature_errors = safe_literal_list(top_k_feature_errors)

    feature_error_vector = record.get("feature_error_vector", [])
    if isinstance(feature_error_vector, str):
        feature_error_vector = safe_literal_list(feature_error_vector)

    context_summary = {
        "container_app_du": record.get("container_app_du", record.get("app_du", "unknown")),
        "container_status": record.get("container_status", "unknown"),
        "machine_status": record.get("machine_status", "unknown"),
        "machine_failure_domain_1": record.get("machine_failure_domain_1", "unknown"),
        "machine_failure_domain_2": record.get("machine_failure_domain_2", "unknown"),
    }

    return {
        "window_id": int(record.get("window_id", -1)),
        "container_id": str(record.get("container_id", "unknown")),
        "machine_id": str(record.get("machine_id", "unknown")),
        "mode": str(record.get("mode", "reconstruction")),
        "split": str(record.get("split", "")),
        "time_range": {
            "start_time": int(record.get("start_time", -1)),
            "end_time": int(record.get("end_time", -1)),
        },
        "recon_score": record.get("recon_score"),
        "forecast_score": record.get("forecast_score"),
        "final_score": float(record.get("final_score", record.get("anomaly_score", 0.0))),
        "anomaly_score": float(record.get("anomaly_score", 0.0)),
        "threshold": float(record.get("threshold", 0.0)),
        "dynamic_threshold": record.get("dynamic_threshold"),
        "final_threshold": record.get("final_threshold"),
        "score_over_threshold": float(record.get("score_over_threshold", 0.0)),
        "decision_reason": str(record.get("decision_reason", "")),
        "top_k_features": top_k_features,
        "top_k_feature_errors": [float(value) for value in top_k_feature_errors],
        "feature_error_vector": [float(value) for value in feature_error_vector],
        "context_summary": context_summary,
        "recent_logs": recent_logs or [],
        "recent_events": recent_events or [],
    }


def fallback_decision(summary: dict[str, Any]) -> dict[str, Any]:
    threshold = max(1e-6, float(summary["threshold"]))
    ratio = float(summary["anomaly_score"]) / threshold
    top_features = summary.get("top_k_features", [])
    top_text = ", ".join(top_features[:3]) if top_features else "no dominant features"
    mode = str(summary.get("mode", "reconstruction"))

    if ratio < 1.05:
        return {
            "label": "normal",
            "severity": "low",
            "root_cause": f"Minor threshold breach in {top_text}.",
            "impact_analysis": "No significant impact. Likely normal workload variance.",
            "step_by_step_recommendations": ["Ignore candidate."],
            "explanation": (
                f"The {mode} score is only marginally above threshold. "
                f"Top deviations: {top_text}. This looks more like a false positive than an active fault."
            ),
            "recommended_action": "ignore",
        }
    if ratio < 1.50:
        return {
            "label": "warning",
            "severity": "low",
            "root_cause": f"Sustained deviation in {top_text}.",
            "impact_analysis": "Potential early stage degradation.",
            "step_by_step_recommendations": ["Monitor telemetry.", "Check logs."],
            "explanation": (
                f"A weak but credible {mode} anomaly is concentrated in {top_text}. "
                f"Monitor the container and review recent events before escalation."
            ),
            "recommended_action": "monitor",
        }
    if ratio < 2.50:
        return {
            "label": "fault_candidate",
            "severity": "medium",
            "root_cause": f"Significant spike in {top_text}.",
            "impact_analysis": "Service performance is likely affected locally.",
            "step_by_step_recommendations": ["Inspect workload.", "Review pod states.", "Correlate with deployments."],
            "explanation": (
                f"The {mode} anomaly is materially above threshold and concentrated in {top_text}. "
                f"Treat this as a fault candidate pending operator review."
            ),
            "recommended_action": "inspect_container",
        }
    return {
        "label": "critical",
        "severity": "high",
        "root_cause": f"Severe fault isolated to {top_text}.",
        "impact_analysis": "High probability of cascading failure or node outage.",
        "step_by_step_recommendations": ["Isolate node.", "Alert on-call.", "Drain traffic."],
        "explanation": (
            f"The {mode} score is far above threshold and the largest deviations are in {top_text}. "
            f"This is consistent with an active high-severity runtime issue."
        ),
        "recommended_action": "raise_alert",
    }


def normalize_decision(decision: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    steps = decision.get("step_by_step_recommendations", [])
    if not isinstance(steps, list):
        steps = [str(steps)]
        
    normalized = {
        "label": str(decision.get("label", "")).strip().lower(),
        "severity": str(decision.get("severity", "")).strip().lower(),
        "root_cause": str(decision.get("root_cause", "")).strip(),
        "impact_analysis": str(decision.get("impact_analysis", "")).strip(),
        "step_by_step_recommendations": [str(s) for s in steps] if steps else [],
        "explanation": str(decision.get("explanation", "")).strip(),
        "recommended_action": str(decision.get("recommended_action", "")).strip(),
    }

    fallback = fallback_decision(summary)
    if normalized["label"] not in VALID_LABELS:
        normalized["label"] = fallback["label"]
    if normalized["severity"] not in VALID_SEVERITIES:
        normalized["severity"] = fallback["severity"]
    if normalized["recommended_action"] not in RECOMMENDED_ACTIONS:
        normalized["recommended_action"] = fallback["recommended_action"]
    if not normalized["explanation"]:
        normalized["explanation"] = fallback["explanation"]
    if not normalized["root_cause"]:
        normalized["root_cause"] = fallback["root_cause"]
    if not normalized["impact_analysis"]:
        normalized["impact_analysis"] = fallback["impact_analysis"]
    if not normalized["step_by_step_recommendations"]:
        normalized["step_by_step_recommendations"] = fallback["step_by_step_recommendations"]
        
    return normalized


def call_openai_responses_api(
    summary: dict[str, Any],
    config: GPTConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = load_prompt_template(config.prompt_template_path)
    api_key = config.api_key
    if not api_key:
        return fallback_decision(summary), {"used_fallback": True, "reason": "missing_api_key"}

    try:
        from openai import OpenAI
    except ImportError:
        return fallback_decision(summary), {"used_fallback": True, "reason": "openai_package_not_installed"}

    try:
        client = OpenAI(api_key=api_key, timeout=config.request_timeout_seconds)
        response = client.responses.create(
            model=config.model,
            instructions=prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(summary, ensure_ascii=True),
                        }
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "container_anomaly_decision",
                    "schema": RESPONSE_JSON_SCHEMA,
                    "strict": True,
                }
            },
        )
    except Exception as exc:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "openai_request_failed",
            "error": str(exc),
        }

    response_text = getattr(response, "output_text", "")
    if not response_text:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "empty_response_text",
            "response_id": getattr(response, "id", None),
            "model": config.model,
        }

    try:
        decision = json.loads(response_text)
    except json.JSONDecodeError:
        return fallback_decision(summary), {
            "used_fallback": True,
            "reason": "invalid_json_response",
            "response_id": getattr(response, "id", None),
            "model": config.model,
        }

    return normalize_decision(decision, summary), {
        "used_fallback": False,
        "model": config.model,
        "response_id": getattr(response, "id", None),
    }


def adjudicate_anomaly(
    record: dict[str, Any],
    config: GPTConfig,
    recent_logs: list[str] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = build_window_summary(record, recent_logs=recent_logs, recent_events=recent_events)
    decision, call_meta = call_openai_responses_api(summary, config=config)
    return {
        **record,
        "gpt_input_summary": summary,
        "structured_json": decision,
        "label": decision["label"],
        "severity": decision["severity"],
        "root_cause": decision["root_cause"],
        "impact_analysis": decision["impact_analysis"],
        "step_by_step_recommendations": decision.get("step_by_step_recommendations", []),
        "explanation": decision["explanation"],
        "recommended_action": decision["recommended_action"],
        "used_fallback": bool(call_meta.get("used_fallback", False)),
        "gpt_model": call_meta.get("model", config.model),
        "response_id": call_meta.get("response_id"),
        "call_reason": call_meta.get("reason"),
        "call_error": call_meta.get("error"),
    }


def adjudicate_record(
    record: dict[str, Any],
    config: GPTConfig,
    recent_logs: list[str] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return adjudicate_anomaly(
        record=record,
        config=config,
        recent_logs=recent_logs,
        recent_events=recent_events,
    )


def compare_ae_vs_gpt_decisions(adjudications: pd.DataFrame) -> pd.DataFrame:
    if adjudications.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "container_id",
                "machine_id",
                "mode",
                "anomaly_score",
                "ae_only_label",
                "gpt_label",
                "severity",
                "root_cause",
                "impact_analysis",
                "recommended_action",
                "explanation",
            ]
        )

    comparison = adjudications[
        [
            "window_id",
            "container_id",
            "machine_id",
            "mode",
            "anomaly_score",
            "label",
            "severity",
            "root_cause",
            "impact_analysis",
            "recommended_action",
            "explanation",
        ]
    ].copy()
    comparison = comparison.rename(columns={"label": "gpt_label"})
    comparison["ae_only_label"] = "fault_candidate" # Emulates what Autoencoder originally outputted.
    comparison = comparison[
        [
            "window_id",
            "container_id",
            "machine_id",
            "mode",
            "anomaly_score",
            "ae_only_label",
            "gpt_label",
            "severity",
            "root_cause",
            "impact_analysis",
            "recommended_action",
            "explanation",
        ]
    ]
    return comparison


def adjudicate_anomaly_records(
    prediction_csv_path: str | Path,
    config: GPTConfig | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    config = config or GPTConfig()
    ensure_directory(config.output_dir)

    predictions = pd.read_csv(prediction_csv_path)
    
    # Check if 'is_candidate' logic exists from eval loop to reduce API calls latency:
    if 'is_candidate' in predictions.columns:
        anomalous = predictions[predictions["is_candidate"] == True].copy()
    else:
        # Fallback to strictly looking at predicted_label if advanced candidates aren't flagged
        anomalous = predictions[predictions["predicted_label"] == 1].copy()
        
    anomalous = anomalous.sort_values("anomaly_score", ascending=False)
    keep_count = max_records or config.max_records
    anomalous = anomalous.head(keep_count).reset_index(drop=True)

    adjudications = [
        adjudicate_anomaly(record=row.to_dict(), config=config)
        for _, row in anomalous.iterrows()
    ]
    adjudication_df = pd.DataFrame(adjudications)
    comparison_df = compare_ae_vs_gpt_decisions(adjudication_df)

    adjudication_csv = config.output_dir / "gpt_adjudications.csv"
    adjudication_json = config.output_dir / "gpt_adjudications.json"
    comparison_csv = config.output_dir / "ae_vs_gpt_comparison.csv"
    comparison_json = config.output_dir / "ae_vs_gpt_comparison.json"

    adjudication_df.to_csv(adjudication_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    write_json(adjudication_json, {"records": adjudications})
    write_json(comparison_json, {"records": comparison_df.to_dict(orient="records")})

    return {
        "gpt_adjudications_csv": str(adjudication_csv.resolve()),
        "gpt_adjudications_json": str(adjudication_json.resolve()),
        "comparison_csv": str(comparison_csv.resolve()),
        "comparison_json": str(comparison_json.resolve()),
        "records": adjudications,
    }
