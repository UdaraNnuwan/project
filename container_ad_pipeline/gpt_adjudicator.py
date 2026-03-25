from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import GPTConfig
from .utils import extract_output_text, load_json, load_text_template, save_json


def heuristic_adjudication(payload: dict[str, Any]) -> dict[str, Any]:
    score = float(payload.get("anomaly_score", 0.0))
    threshold = float(payload.get("threshold", 1.0) or 1.0)
    normalized = score / max(threshold, 1e-6)
    top_features = payload.get("top_k_anomalous_features", [])

    if normalized < 1.10:
        label = "false_positive"
        severity = "none"
        explanation = "Reconstruction error only marginally exceeded threshold, so the alert is likely noise."
        recommended_action = "Keep monitoring and require repeated threshold crossings before escalation."
    elif normalized < 1.75:
        label = "needs_review"
        severity = "low"
        explanation = f"The anomaly is weak but persistent enough to review. Main contributors: {', '.join(top_features[:3])}."
        recommended_action = "Inspect the recent workload change, deployment diff, and container logs."
    elif normalized < 3.0:
        label = "anomaly"
        severity = "medium"
        explanation = f"The window shows a credible multivariate anomaly driven by {', '.join(top_features[:3])}."
        recommended_action = "Correlate the top metrics with service logs and check for resource contention."
    else:
        label = "anomaly"
        severity = "high"
        explanation = f"The anomaly score is far above threshold and the strongest deviations are {', '.join(top_features[:3])}."
        recommended_action = "Treat this as an active incident, triage the affected container, and consider mitigation."

    return {
        "label": label,
        "severity": severity,
        "explanation": explanation,
        "recommended_action": recommended_action,
    }


@dataclass
class GPTAdjudicator:
    config: GPTConfig

    def __post_init__(self) -> None:
        self.prompt_template: str | None = None
        self.schema_payload: dict[str, Any] | None = None

        prompt_path = Path(self.config.prompt_template_path)
        schema_path = Path(self.config.schema_path)

        if prompt_path.exists():
            self.prompt_template = load_text_template(prompt_path)
        if schema_path.exists():
            self.schema_payload = load_json(schema_path)

    def has_api_key(self) -> bool:
        return bool(os.getenv(self.config.api_key_env))

    def build_prompt(self, payload: dict[str, Any]) -> str:
        if not self.prompt_template:
            raise FileNotFoundError(f"Prompt template not found: {self.config.prompt_template_path}")
        logs = payload.get("recent_logs", [])
        events = payload.get("recent_events", [])
        return self.prompt_template.format(
            anomaly_score=payload.get("anomaly_score"),
            threshold=payload.get("threshold"),
            normalized_score=payload.get("normalized_score"),
            context_metadata=json.dumps(payload.get("context_metadata", {}), indent=2),
            top_k_features=json.dumps(payload.get("top_k_anomalous_features", []), indent=2),
            top_k_feature_errors=json.dumps(payload.get("top_k_feature_errors", []), indent=2),
            recent_logs=json.dumps(logs, indent=2),
            recent_events=json.dumps(events, indent=2),
        )

    def adjudicate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.enabled or not self.has_api_key():
            return heuristic_adjudication(payload)

        try:
            from openai import OpenAI
        except ImportError:
            return heuristic_adjudication(payload)

        if not self.schema_payload:
            return heuristic_adjudication(payload)

        client = OpenAI(api_key=os.getenv(self.config.api_key_env))
        prompt = self.build_prompt(payload)
        response = client.responses.create(
            model=self.config.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You adjudicate post-threshold multivariate container anomalies. Return JSON only.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": self.schema_payload["name"],
                    "strict": self.schema_payload.get("strict", True),
                    "schema": self.schema_payload["schema"],
                }
            },
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        output_text = extract_output_text(response)
        if not output_text:
            return heuristic_adjudication(payload)
        return json.loads(output_text)

    def adjudicate_dataframe(
        self,
        anomalous_windows: pd.DataFrame,
        threshold: float,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in anomalous_windows.itertuples(index=False):
            context_metadata = {
                "container_id": getattr(row, "container_id", ""),
                "machine_id": getattr(row, "machine_id", ""),
                "app_du": getattr(row, "app_du", ""),
                "container_status": getattr(row, "container_status", ""),
                "machine_status": getattr(row, "machine_status", ""),
                "start_time": getattr(row, "start_time", ""),
                "end_time": getattr(row, "end_time", ""),
            }
            payload = {
                "anomaly_score": float(row.anomaly_score),
                "threshold": float(threshold),
                "normalized_score": float(row.anomaly_score / max(threshold, 1e-6)),
                "top_k_anomalous_features": list(row.top_k_features),
                "top_k_feature_errors": list(row.top_k_feature_errors),
                "context_metadata": {key: value for key, value in context_metadata.items() if value != ""},
                "recent_logs": getattr(row, "recent_logs", []) if hasattr(row, "recent_logs") else [],
                "recent_events": getattr(row, "recent_events", []) if hasattr(row, "recent_events") else [],
            }
            decision = self.adjudicate(payload)
            rows.append(
                {
                    **context_metadata,
                    "anomaly_score": payload["anomaly_score"],
                    "normalized_score": payload["normalized_score"],
                    "top_k_features": payload["top_k_anomalous_features"],
                    "top_k_feature_errors": payload["top_k_feature_errors"],
                    **decision,
                }
            )

        result = pd.DataFrame(rows)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "gpt_adjudications.csv"
            json_path = output_dir / "gpt_adjudications.json"
            result.to_csv(csv_path, index=False)
            save_json(json_path, {"records": result.to_dict(orient="records")})
        return result
