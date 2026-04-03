from __future__ import annotations

from typing import Any

import requests


def _listify(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _first_nonempty(record: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def format_telegram_alert(record: dict[str, Any]) -> str:
    top_features = [str(value) for value in _listify(record.get("top_k_features"))[:5]]
    top_feature_errors = [float(value) for value in _listify(record.get("top_k_feature_errors"))[:5]]
    top_text = ", ".join(top_features) if top_features else "n/a"
    top_error_text = ", ".join(f"{value:.6f}" for value in top_feature_errors) if top_feature_errors else "n/a"

    title = "Live container anomaly alert" if str(record.get("entity_id", "")).strip() else "Critical container anomaly alert"
    lines = [
        title,
    ]

    if "window_id" in record:
        lines.append(f"Window: {int(record.get('window_id', -1))}")
    if "entity_id" in record:
        lines.append(f"Entity: {record.get('entity_id', 'unknown')}")
    lines.extend(
        [
            f"Container: {record.get('container_id', 'unknown')}",
            f"Machine: {record.get('machine_id', 'unknown')}",
        ]
    )
    split = str(record.get("split", "") or "").strip()
    if split:
        lines.append(f"Split: {split}")
    lines.extend(
        [
            f"Score: {float(record.get('anomaly_score', 0.0)):.6f}",
            f"Threshold: {float(record.get('threshold', 0.0)):.6f}",
        ]
    )
    static_threshold = record.get(
        "static_threshold",
        record.get("fallback_threshold", record.get("fixed_threshold")),
    )
    if static_threshold is not None:
        lines.append(f"Static threshold: {float(static_threshold):.6f}")
    if record.get("dynamic_threshold") is not None:
        lines.append(f"Dynamic threshold: {float(record.get('dynamic_threshold', 0.0)):.6f}")
    if record.get("final_threshold") is not None:
        lines.append(f"Final threshold: {float(record.get('final_threshold', 0.0)):.6f}")
    threshold_mode = _first_nonempty(record, ["threshold_mode"])
    if threshold_mode:
        lines.append(f"Threshold mode: {threshold_mode}")
    if record.get("score_buffer_size") is not None:
        lines.append(f"Score buffer size: {int(record.get('score_buffer_size', 0))}")
    z_score = record.get("z_score")
    z_reason = _first_nonempty(record, ["z_score_reason"])
    if z_score is not None:
        lines.append(f"Z-score: {float(z_score):.3f}")
    elif z_reason:
        lines.append(f"Z-score: n/a ({z_reason})")

    decision_reason = str(record.get("decision_reason", "") or "").strip()
    if decision_reason:
        lines.append(f"Decision reason: {decision_reason}")
    alert_suppressed_reason = _first_nonempty(record, ["alert_suppressed_reason"])
    if alert_suppressed_reason:
        lines.append(f"Alert status: suppressed ({alert_suppressed_reason})")
    lines.extend(
        [
            f"Top features: {top_text}",
            f"Top feature errors: {top_error_text}",
        ]
    )

    gpt_failed = bool(record.get("gpt_failed", False))
    if gpt_failed:
        lines.extend(
            [
                "GPT status: failed",
                f"GPT failure reason: {_first_nonempty(record, ['gpt_failure_reason', 'call_reason'], 'unknown')}",
            ]
        )
        model_explanation = _first_nonempty(record, ["model_explanation", "decision_reason"])
        if model_explanation:
            lines.append(f"Model candidate reason: {model_explanation}")
        recommended_action = _first_nonempty(record, ["recommended_action", "gpt_recommended_action"])
        if recommended_action:
            lines.append(f"Recommended action: {recommended_action}")
        return "\n".join(lines)

    gpt_label = _first_nonempty(record, ["gpt_label", "label"])
    gpt_severity = _first_nonempty(record, ["gpt_severity", "severity"])
    gpt_action = _first_nonempty(record, ["gpt_recommended_action", "recommended_action"])
    explanation = _first_nonempty(record, ["gpt_explanation", "explanation"])

    if gpt_label:
        lines.append(f"GPT label: {gpt_label}")
    if gpt_severity:
        lines.append(f"GPT severity: {gpt_severity}")
    if gpt_action:
        lines.append(f"Action: {gpt_action}")
    if explanation:
        lines.append(f"Reason: {explanation}")
    return "\n".join(lines)


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    message_text: str,
    timeout_seconds: int,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    endpoint = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        response = requests.post(
            endpoint,
            json={
                "chat_id": chat_id,
                "text": message_text,
                "disable_web_page_preview": True,
            },
            timeout=max(1, int(timeout_seconds)),
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok", False):
            return False, body, f"telegram_send_failed:{body.get('description', 'unknown_error')}"
        return True, body, None
    except Exception as exc:
        return False, None, f"telegram_send_failed:{str(exc)}"
