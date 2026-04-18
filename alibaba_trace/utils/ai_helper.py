from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger("AIHelper")

_GPT_TIMEOUT  = float(os.getenv("GPT_TIMEOUT_SECONDS", "18"))
_GPT_MAX_TOK  = 180
_GPT_TEMP     = 0.05
_FALLBACK_MSG = "GPT RCA unavailable – review top features manually."


def get_gpt_explanation(
    top_features: List[str],
    reason: str,
    scores: Dict[str, float],
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or len(api_key) < 12:
        logger.warning("No OpenAI API key configured.")
        return _FALLBACK_MSG

    model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(top_features, reason, scores)

    t0 = time.perf_counter()
    try:
        text    = _call_openai(api_key, model, prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("GPT RCA in %.0fms.", elapsed)
        return text or _FALLBACK_MSG
    except Exception as exc:
        logger.error("GPT call failed: %s", exc)
        return _FALLBACK_MSG


def get_configured_engine(temperature: float = 0.10):
    from alibaba_trace.incident.rca_genai import GenAIRCAEngine
    return GenAIRCAEngine(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=temperature,
    )


def _build_prompt(
    top_features: List[str],
    reason: str,
    scores: Dict[str, float],
) -> str:
    return (
        "You are an expert SRE. A Kubernetes container anomaly was detected.\n\n"
        f"Top anomalous features : {', '.join(top_features) or 'unknown'}\n"
        f"Trigger reason         : {reason}\n"
        f"All-feature MSE score  : {scores.get('all_score', 0.0):.6f}\n"
        f"Top-feature MSE score  : {scores.get('top_score', 0.0):.6f}\n\n"
        "In 2–3 concise sentences explain the likely root cause and one immediate "
        "remediation action. Plain text only, no markdown."
    )


def _call_openai(api_key: str, model: str, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai not installed. Run: pip install openai") from exc

    kwargs: dict = {"api_key": api_key, "timeout": _GPT_TIMEOUT}
    if api_key.startswith("github_pat_"):
        kwargs["base_url"] = "https://models.inference.ai.azure.com"

    client     = OpenAI(**kwargs)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert SRE. Plain text only."},
            {"role": "user",   "content": prompt},
        ],
        temperature=_GPT_TEMP,
        max_tokens=_GPT_MAX_TOK,
    )
    return completion.choices[0].message.content.strip()
