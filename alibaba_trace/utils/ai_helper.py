from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger("AIHelper")

_GPT_TIMEOUT = float(os.getenv("GPT_TIMEOUT_SECONDS", "18"))
_GPT_MAX_TOK = 180
_GPT_TEMP    = 0.05


def get_gpt_explanation(
    top_features: List[str],
    reason: str,
    scores: Dict[str, float],
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or len(api_key) < 12:
        logger.warning("GPT skipped: OPENAI_API_KEY not set or too short.")
        return "GPT RCA skipped: no API key configured."

    model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(top_features, reason, scores)

    t0 = time.perf_counter()
    try:
        text    = _call_openai(api_key, model, prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("GPT RCA completed in %.0fms.", elapsed)
        return text or "GPT returned empty response."
    except Exception as exc:
        return _openai_error_summary(exc)


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


def _openai_error_summary(exc: Exception) -> str:
    """Return a structured error string with code + reason for any OpenAI failure."""
    try:
        from openai import (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            PermissionDeniedError,
            NotFoundError,
            APIStatusError,
        )
        if isinstance(exc, AuthenticationError):
            msg = f"GPT ERROR [401 AuthenticationError]: Invalid API key or org. Detail: {exc.message}"
        elif isinstance(exc, PermissionDeniedError):
            msg = f"GPT ERROR [403 PermissionDenied]: Access denied. Detail: {exc.message}"
        elif isinstance(exc, NotFoundError):
            msg = f"GPT ERROR [404 NotFound]: Model or endpoint not found. Detail: {exc.message}"
        elif isinstance(exc, RateLimitError):
            msg = f"GPT ERROR [429 RateLimitError]: Quota exceeded or too many requests. Detail: {exc.message}"
        elif isinstance(exc, BadRequestError):
            msg = f"GPT ERROR [400 BadRequest]: Invalid request payload. Detail: {exc.message}"
        elif isinstance(exc, APITimeoutError):
            msg = f"GPT ERROR [Timeout]: Request exceeded {_GPT_TIMEOUT}s timeout."
        elif isinstance(exc, APIConnectionError):
            msg = f"GPT ERROR [ConnectionError]: Could not reach OpenAI API. Detail: {exc}"
        elif isinstance(exc, APIStatusError):
            msg = f"GPT ERROR [{exc.status_code} APIStatusError]: {exc.message}"
        else:
            msg = f"GPT ERROR [Unknown]: {type(exc).__name__}: {exc}"
    except ImportError:
        msg = f"GPT ERROR [ImportError]: openai package unavailable – {exc}"

    logger.error(msg)
    return msg
