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
from .prompts import PROMPT_TEMPLATE, DEMO_NARRATIVES

class GenAIRCAEngine:

    def __init__(
        self,
        openai_api_key:  Optional[str] = None,
        openai_model:    str           = "gpt-4o-mini",
        timeout_seconds: float         = 30.0,
        max_retries:     int           = 2,
        temperature:     float         = 0.10,
    ) -> None:
        self.openai_api_key  = openai_api_key  or ""
        self.openai_model    = openai_model
        self.timeout_seconds = timeout_seconds
        self.max_retries     = max_retries
        self.temperature     = temperature

    def analyze_with_llm(
        self,
        original_metrics:      np.ndarray,   # Original window: (W, F)
        reconstructed_metrics: np.ndarray,   # Model output: (W, F)
        mse:                   float,
        context:               ContainerContext,
        rca_result:            Dict,
        recent_logs:           str,
        threshold_p95:         float,
        timestamp_utc:         Optional[str] = None,
    ) -> LLMAnalysis:
        if timestamp_utc is None:
            timestamp_utc = (
                datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            )

        prompt = self._build_prompt(
            original_metrics, reconstructed_metrics, mse,
            context, rca_result, recent_logs, threshold_p95, timestamp_utc,
        )

        # Gemini support was removed, so only OpenAI or demo mode runs here.

        if self.openai_api_key and not self._is_placeholder(self.openai_api_key):
            try:
                raw, latency = self._call_openai(prompt)
                
                # Save the live LLM response for review.
                with open("gpt_logs.txt", "a", encoding="utf-8") as lf:
                    lf.write("================= GPT RESPONSE =================\n")
                    lf.write(raw + "\n\n")
                    
                return self._parse_response(
                    raw, self.openai_model, latency, True, len(prompt) // 4
                )
            except Exception as exc:
                logger.warning("OpenAI call failed (%s). Falling back to demo.", exc)

        logger.info("GenAIRCAEngine: using DEMO_MODE (no live API available).")
        raw     = self._demo_response(mse, context, rca_result, threshold_p95)
        
        # Save the demo response in the same log file.
        with open("gpt_logs.txt", "a", encoding="utf-8") as lf:
            lf.write("================= DUMMY (MOCK) RESPONSE =================\n")
            lf.write("(API Keys were not found, this is standard offline fallback generation)\n")
            lf.write(raw + "\n\n")
            
        return self._parse_response(
            raw, "DEMO_MODE (GPT-4o Simulated)", 0.0, False,
            len(prompt) // 4,
        )

    @staticmethod
    def _is_placeholder(key: str) -> bool:
        lower = key.lower()
        return (
            not key
            or "your-api-key" in lower
            or "placeholder" in lower
            or "xxxx" in lower
            or len(key) < 12
        )

    def _build_prompt(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        mse:           float,
        context:       ContainerContext,
        rca_result:    Dict,
        recent_logs:   str,
        threshold_p95: float,
        timestamp_utc: str,
    ) -> str:
        original      = np.asarray(original)
        reconstructed = np.asarray(reconstructed)

        feat_table_lines = []
        for entry in rca_result.get("feature_errors_ranked", []):
            marker = "  <<< PRIMARY" if entry["feature"] == rca_result.get("primary_metric") else ""
            feat_table_lines.append(
                f"  {entry['icon']} {entry['display_name']:30s}  "
                f"MSE={entry['mse']:.6f}  ({entry['error_pct']:.1f}%){marker}"
            )
        feature_table = "\n".join(feat_table_lines) or "  (no features ranked)"

        feature_table = "\n".join(feat_table_lines) or "  (no features ranked)"

        film_note = (
            f"The FiLM layer conditioned the BiLSTM reconstruction on "
            f"'{context.container_type}' workload context, adjusting the "
            f"expected reconstruction baseline for {context.display_name()}."
        )

        exceedance = mse / threshold_p95 if threshold_p95 > 0 else 0.0

        return PROMPT_TEMPLATE.format(
            timestamp_utc=timestamp_utc,
            severity=rca_result.get("severity", "Unknown") if "severity" in rca_result else "Critical",
            mse=mse,
            threshold_p95=threshold_p95,
            exceedance=exceedance,
            container_id=context.container_id,
            container_type=context.container_type,
            container_display=context.display_name(),
            tier=context.tier,
            environment=context.environment,
            pod_name=context.pod_name,
            namespace=context.namespace,
            film_vector=context.film_vector_str(),
            film_note=film_note,
            feature_table=feature_table,
            primary_metric=rca_result.get("primary_display", "Unknown"),
            primary_pct=rca_result.get("primary_error_percent", 0.0),
            stat_diagnosis=rca_result.get("diagnosis", ""),
            stat_action=rca_result.get("recommended_action", "").split("\n")[0],
            recent_logs=recent_logs.strip() or "(no logs available)",
        )
        
        # Save the prompt for later review.
        with open("gpt_logs.txt", "a", encoding="utf-8") as lf:
            lf.write("================= NEW GPT REQUEST =================\n")
            lf.write(f"TIMESTAMP: {timestamp_utc}\n")
            lf.write("PROMPT:\n")
            lf.write(prompt + "\n\n")
            
        return prompt

    def _call_openai(self, prompt: str) -> Tuple[str, float]:
        import time
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai not installed. Run: pip install openai"
            )
        kwargs = {"api_key": self.openai_api_key, "timeout": self.timeout_seconds}
        if self.openai_api_key.startswith("github_pat_"):
            kwargs["base_url"] = "https://models.inference.ai.azure.com"
            
        client = OpenAI(**kwargs)
        t0 = time.perf_counter()
        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Expert SRE and DevOps Architect. "
                        "Respond ONLY in the exact structured format requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return completion.choices[0].message.content.strip(), latency

    def _demo_response(
        self,
        mse:           float,
        context:       ContainerContext,
        rca_result:    Dict,
        threshold_p95: float,
    ) -> str:
        ctype   = context.container_type
        primary = rca_result.get("primary_metric", "")
        diag    = rca_result.get("diagnosis", "")
        action  = rca_result.get("recommended_action", "").split("\n")[0]
        pct     = rca_result.get("primary_error_percent", 50.0)
        excess  = mse / threshold_p95 if threshold_p95 > 0 else 1.0

        verdict = "GENUINE ANOMALY" if excess >= 3.0 else "GENUINE ANOMALY"

        

        type_narratives = DEMO_NARRATIVES.get(ctype, {})
        if primary in type_narratives:
            cause_text, mit_text = type_narratives[primary]
            cause_text = cause_text.format(pct=min(99, int(50 + pct * 0.45)))
        else:
            cause_text = (
                f"The {context.display_name()} container shows anomalous reconstruction error "
                f"primarily driven by {rca_result.get('primary_display', primary)} "
                f"({pct:.0f}% of total MSE={mse:.6f}), which is {excess:.1f}× above the P95 threshold. "
                f"{diag}"
            )
            mit_text = action or (
                "Execute kubectl describe pod and kubectl logs to gather additional context "
                "before escalating to the responsible engineering team."
            )

        confidence = min(99, max(72, int(70 + min(excess, 10) * 2.5 - (0 if excess > 5 else 5))))

        return (
            f"VERDICT: {verdict}\n"
            f"ROOT CAUSE: {cause_text}\n"
            f"MITIGATION: {mit_text}\n"
            f"CONFIDENCE: {confidence}%"
        )

    @staticmethod
    def _parse_response(
        raw_text:     str,
        model_used:   str,
        latency_ms:   float,
        success:      bool,
        prompt_tokens: int,
    ) -> LLMAnalysis:
        import re
        verdict           = "UNCERTAIN"
        root_cause        = raw_text   # Use the full response if parsing fails.
        mitigation        = "Investigate with kubectl describe pod and review application logs."
        confidence        = 70
        verified_severity = ""         # Filled from the SEVERITY field when present.

        try:
            m = re.search(r"VERDICT\s*:\s*(.+?)(?:\n|ROOT)", raw_text, re.IGNORECASE)
            if m:
                verdict = m.group(1).strip().rstrip(".")

            m = re.search(r"ROOT CAUSE\s*:\s*(.+?)(?:\nMITIGATION|\nCONFIDENCE|\nSEVERITY|$)",
                          raw_text, re.IGNORECASE | re.DOTALL)
            if m:
                root_cause = m.group(1).strip()

            m = re.search(r"MITIGATION\s*:\s*(.+?)(?:\nCONFIDENCE|\nSEVERITY|$)",
                          raw_text, re.IGNORECASE | re.DOTALL)
            if m:
                mitigation = m.group(1).strip()

            m = re.search(r"CONFIDENCE\s*:\s*(\d+)", raw_text, re.IGNORECASE)
            if m:
                confidence = min(100, max(0, int(m.group(1))))

            m = re.search(r"SEVERITY\s*:\s*(Critical|High|Warning|Normal)",
                          raw_text, re.IGNORECASE)
            if m:
                raw_sev = m.group(1).strip()
                verified_severity = raw_sev[0].upper() + raw_sev[1:].lower()

        except Exception as parse_err:
            logger.warning("LLM response parsing error: %s", parse_err)

        return LLMAnalysis(
            verdict=verdict,
            root_cause=root_cause,
            mitigation=mitigation,
            confidence_pct=confidence,
            model_used=model_used,
            raw_response=raw_text,
            prompt_tokens=prompt_tokens,
            latency_ms=latency_ms,
            success=success,
            verified_severity=verified_severity,
        )

    def __repr__(self) -> str:
        openai_ok = bool(self.openai_api_key and not self._is_placeholder(self.openai_api_key))
        return (
            f"GenAIRCAEngine("
            f"openai={self.openai_model}({'OK' if openai_ok else 'no key'}), "
            f"temp={self.temperature})"
        )
