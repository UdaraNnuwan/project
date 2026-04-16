from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import torch

try:
    from config import EvalConfig, GPTConfig, GenAIConfig
    from gpt_adjudicator import adjudicate_anomaly, build_window_summary, compare_ae_vs_gpt_decisions
    from hybrid_scoring import (
        available_modes,
        combine_mode_scores,
        compute_forecasting_outputs,
        compute_reconstruction_outputs,
        normalize_model_mode,
        threshold_for_mode,
    )
    from live_infer import StreamingHybridAnomalyDetector, load_model_artifacts
    from telegram_utils import format_telegram_alert, send_telegram_message
    from two_tier_verifier import TwoTierVerifier
    from utils import apply_3d_scaler, build_prediction_frame, ensure_directory, safe_literal_list, write_json
except ImportError:
    from .config import EvalConfig, GPTConfig, GenAIConfig
    from .gpt_adjudicator import adjudicate_anomaly, build_window_summary, compare_ae_vs_gpt_decisions
    from .hybrid_scoring import (
        available_modes,
        combine_mode_scores,
        compute_forecasting_outputs,
        compute_reconstruction_outputs,
        normalize_model_mode,
        threshold_for_mode,
    )
    from .live_infer import StreamingHybridAnomalyDetector, load_model_artifacts
    from .telegram_utils import format_telegram_alert, send_telegram_message
    from .two_tier_verifier import TwoTierVerifier
    from .utils import apply_3d_scaler, build_prediction_frame, ensure_directory, safe_literal_list, write_json


StreamingFiLMAnomalyDetector = StreamingHybridAnomalyDetector


def _eval_log(message: str, enabled: bool) -> None:
    if enabled:
        print(f"[evaluate] {message}")


def _fallback_split_metadata(split: str, rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": np.arange(int(rows), dtype=np.int64),
            "container_id": np.full(int(rows), "unknown_container", dtype=object),
            "machine_id": np.full(int(rows), "unknown_machine", dtype=object),
            "end_time": np.arange(int(rows), dtype=np.int64),
            "split": np.full(int(rows), split, dtype=object),
        }
    )


def _load_split_metadata(dataset_path: Path, split: str, rows: int) -> pd.DataFrame:
    metadata_path = dataset_path / f"window_metadata_{split}.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if len(metadata) == int(rows):
            return metadata.reset_index(drop=True)
    return _fallback_split_metadata(split=split, rows=rows)


def load_artifacts(
    dataset_dir: str | Path,
    model_dir: str | Path,
    model_mode: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    bundle = load_model_artifacts(model_dir=model_dir, device=device, model_mode=model_mode)
    split_paths = {
        "X_train": dataset_path / "X_train.npy",
        "X_test": dataset_path / "X_test.npy",
        "C_train": dataset_path / "C_train.npy",
        "C_test": dataset_path / "C_test.npy",
    }
    if all(path.exists() for path in split_paths.values()):
        x_train = np.load(split_paths["X_train"], mmap_mode="r")
        x_test = np.load(split_paths["X_test"], mmap_mode="r")
        c_train = np.load(split_paths["C_train"], mmap_mode="r")
        c_test = np.load(split_paths["C_test"], mmap_mode="r")
        bundle.update(
            {
                "X_train": x_train,
                "X_test": x_test,
                "C_train": c_train,
                "C_test": c_test,
                "metadata_train": _load_split_metadata(dataset_path, split="train", rows=int(x_train.shape[0])),
                "metadata_test": _load_split_metadata(dataset_path, split="test", rows=int(x_test.shape[0])),
            }
        )
        return bundle

    bundle.update(
        {
            "X": np.load(dataset_path / "X_all.npy", allow_pickle=False),
            "C": np.load(dataset_path / "C_all.npy", allow_pickle=False),
            "metadata": pd.read_csv(dataset_path / "window_metadata.csv"),
        }
    )
    return bundle


def select_split(
    x_all: np.ndarray,
    c_all: np.ndarray,
    metadata: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mask = metadata["split"].astype(str) == split
    indices = np.flatnonzero(mask.to_numpy())
    return (
        x_all[indices],
        c_all[indices],
        metadata.iloc[indices].reset_index(drop=True),
    )


def inject_synthetic_anomalies(
    x_scaled: np.ndarray,
    metadata: pd.DataFrame,
    anomaly_ratio: float,
    event_span: int,
    feature_count: int,
    spike_magnitude: float,
    noise_std: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    injected = x_scaled.copy()
    labels = np.zeros(len(injected), dtype=np.int32)
    event_ids = np.full(len(injected), -1, dtype=np.int32)
    event_rows: list[dict[str, Any]] = []

    target_events = max(1, int(len(injected) * anomaly_ratio / max(1, event_span)))
    event_id = 0
    eligible_groups = []
    for container_id, group in metadata.groupby("container_id", sort=False):
        group_indices = group.index.to_list()
        if len(group_indices) >= event_span:
            eligible_groups.append((container_id, group_indices))

    rng.shuffle(eligible_groups)
    for container_id, group_indices in eligible_groups:
        if event_id >= target_events:
            break
        max_start = len(group_indices) - event_span
        start_pos = int(rng.integers(0, max_start + 1))
        event_indices = group_indices[start_pos : start_pos + event_span]

        feature_total = injected.shape[2]
        chosen_features = rng.choice(feature_total, size=min(feature_count, feature_total), replace=False)
        segment_start = int(rng.integers(0, max(1, injected.shape[1] // 2)))
        segment_end = min(injected.shape[1], segment_start + max(2, injected.shape[1] // 3))
        mode = rng.choice(["spike", "drop", "noise"])

        for index in event_indices:
            window_slice = np.ix_(np.arange(segment_start, segment_end), chosen_features)
            if mode == "spike":
                injected[index][window_slice] += spike_magnitude
            elif mode == "drop":
                injected[index][window_slice] -= spike_magnitude
            else:
                injected[index][window_slice] += rng.normal(
                    0.0,
                    noise_std,
                    size=(segment_end - segment_start, len(chosen_features)),
                )
            labels[index] = 1
            event_ids[index] = event_id

        event_rows.append(
            {
                "event_id": event_id,
                "container_id": str(container_id),
                "window_indices": [int(value) for value in event_indices],
                "feature_indices": chosen_features.astype(int).tolist(),
                "feature_count": int(len(chosen_features)),
                "mode": str(mode),
                "segment_start": int(segment_start),
                "segment_end": int(segment_end),
            }
        )
        event_id += 1

    return injected, labels, event_ids, pd.DataFrame(event_rows)


def compute_binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    pr_auc = float(average_precision_score(labels, scores)) if labels.sum() > 0 else 0.0
    roc_auc = float(roc_auc_score(labels, scores)) if labels.sum() > 0 and len(np.unique(labels)) > 1 else 0.0
    fpr = float(fp / max(1, fp + tn))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "false_positive_rate": fpr,
    }


def compute_event_detection_metrics(
    event_ids: np.ndarray,
    predictions: np.ndarray,
    relaxed_tolerance: int,
) -> dict[str, Any]:
    valid_event_ids = [event_id for event_id in sorted(set(event_ids.tolist())) if event_id >= 0]
    if not valid_event_ids:
        return {
            "num_events": 0,
            "strict_detection_rate": 0.0,
            "relaxed_detection_rate": 0.0,
            "mean_detection_delay": None,
        }

    strict_hits = 0
    relaxed_hits = 0
    detection_delays: list[int] = []
    for event_id in valid_event_ids:
        indices = np.flatnonzero(event_ids == event_id)
        event_predictions = predictions[indices]
        first_positive = np.flatnonzero(event_predictions == 1)
        if len(first_positive) > 0:
            delay = int(first_positive[0])
            detection_delays.append(delay)
            if delay == 0:
                strict_hits += 1
            if delay <= relaxed_tolerance:
                relaxed_hits += 1

    return {
        "num_events": len(valid_event_ids),
        "strict_detection_rate": float(strict_hits / len(valid_event_ids)),
        "relaxed_detection_rate": float(relaxed_hits / len(valid_event_ids)),
        "mean_detection_delay": float(np.mean(detection_delays)) if detection_delays else None,
    }


def score_windows_for_mode(
    bundle: dict[str, Any],
    x_scaled: np.ndarray,
    c_scaled: np.ndarray,
    mode: str,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    normalized_mode = normalize_model_mode(mode)
    recon_errors = None
    recon_scores = None
    forecast_errors = None
    forecast_scores = None

    if normalized_mode in {"reconstruction", "hybrid"}:
        recon_errors, recon_scores = compute_reconstruction_outputs(
            model=bundle["reconstruction_model"],
            x_scaled=x_scaled,
            c_scaled=c_scaled,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )
    if normalized_mode in {"forecasting", "hybrid"}:
        forecast_errors, forecast_scores = compute_forecasting_outputs(
            model=bundle["forecasting_model"],
            x_scaled=x_scaled,
            c_scaled=c_scaled,
            forecast_horizon=int(bundle["forecast_horizon"]),
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )

    combined = combine_mode_scores(
        mode=normalized_mode,
        recon_feature_errors=recon_errors,
        recon_scores=recon_scores,
        forecast_feature_errors=forecast_errors,
        forecast_scores=forecast_scores,
        alpha=float(bundle["alpha"]),
        beta=float(bundle["beta"]),
    )
    return {
        "recon_scores": combined.recon_scores,
        "forecast_scores": combined.forecast_scores,
        "final_scores": combined.final_scores,
        "recon_feature_errors": combined.recon_feature_errors,
        "forecast_feature_errors": combined.forecast_feature_errors,
        "final_feature_errors": combined.final_feature_errors,
    }


def apply_streaming_decision_logic(
    prediction_frame: pd.DataFrame,
    static_threshold: float,
    config: EvalConfig,
) -> pd.DataFrame:
    sort_columns = [column for column in ("end_time", "window_id") if column in prediction_frame.columns]
    frame = prediction_frame.sort_values(sort_columns or ["window_id"]).reset_index(drop=True).copy()
    history: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=max(10, int(config.dynamic_threshold_history_limit)))
    )
    smoothing_history: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=max(1, int(config.smoothing_window)))
    )
    consecutive_counts: dict[str, int] = defaultdict(int)
    cooldown_remaining: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        record = row.to_dict()
        entity_id = str(record.get("container_id", record.get("entity_id", "unknown")))
        score = float(record.get("final_score", record.get("anomaly_score", 0.0)))
        score_history = history[entity_id]
        smoothing_window = smoothing_history[entity_id]
        smoothing_window.append(score)
        smoothed_score = float(np.mean(np.asarray(smoothing_window, dtype=np.float32)))

        dynamic_threshold = None
        z_score = None
        z_score_reason = None
        history_values = np.asarray(score_history, dtype=np.float32)
        if len(history_values) >= int(config.dynamic_threshold_min_history):
            dynamic_threshold = float(np.percentile(history_values, float(config.dynamic_threshold_percentile)))
            std = float(np.std(history_values))
            if std > 1e-8:
                z_score = float((score - float(np.mean(history_values))) / std)
            else:
                z_score_reason = "std_too_small"
        else:
            z_score_reason = f"warmup<{int(config.dynamic_threshold_min_history)}"

        final_threshold = float(dynamic_threshold) if dynamic_threshold is not None else float(static_threshold)
        threshold_mode = "dynamic" if dynamic_threshold is not None else "warmup"
        threshold_hit = smoothed_score > final_threshold
        z_score_hit = bool(
            config.z_score_enabled
            and z_score is not None
            and z_score > float(config.z_score_threshold)
        )

        if threshold_hit:
            consecutive_counts[entity_id] += 1
        else:
            consecutive_counts[entity_id] = 0
        current_consecutive = int(consecutive_counts[entity_id])

        decision_reason_parts: list[str] = []
        if threshold_hit:
            decision_reason_parts.append("smoothed_score_above_threshold")
            decision_reason_parts.append(
                f"consecutive_breach={current_consecutive}/{int(config.consecutive_breach_windows)}"
            )
        if z_score_hit:
            decision_reason_parts.append(f"z_score>{float(config.z_score_threshold):.1f}")

        if cooldown_remaining[entity_id] > 0 and (threshold_hit or z_score_hit):
            status = "suppressed"
            cooldown_remaining[entity_id] = max(0, cooldown_remaining[entity_id] - 1)
            decision_reason_parts.append("cooldown_active")
            consecutive_counts[entity_id] = 0
            confirmed = False
        elif threshold_hit and current_consecutive >= int(config.consecutive_breach_windows):
            status = "confirmed_anomaly"
            cooldown_remaining[entity_id] = int(config.cooldown_windows)
            consecutive_counts[entity_id] = 0
            decision_reason_parts.append("confirmed_after_consecutive_breaches")
            confirmed = True
        elif threshold_hit or z_score_hit:
            status = "candidate"
            confirmed = False
        else:
            status = "normal"
            confirmed = False
            if cooldown_remaining[entity_id] > 0:
                cooldown_remaining[entity_id] = max(0, cooldown_remaining[entity_id] - 1)

        score_history.append(score)
        record.update(
            {
                "dynamic_threshold": dynamic_threshold,
                "final_threshold": final_threshold,
                "threshold": final_threshold,
                "threshold_mode": threshold_mode,
                "smoothed_score": smoothed_score,
                "score_over_threshold": smoothed_score - final_threshold,
                "z_score": z_score,
                "z_score_reason": z_score_reason,
                "current_consecutive_breach_count": current_consecutive,
                "score_buffer_size": int(len(score_history)),
                "decision": status,
                "status": status,
                "anomaly_candidate": bool(status in {"candidate", "confirmed_anomaly", "suppressed"}),
                "confirmed_anomaly": bool(confirmed),
                "predicted_label": int(confirmed),
                "decision_reason": ",".join(decision_reason_parts) if decision_reason_parts else "below_all_candidate_rules",
            }
        )
        rows.append(record)

    return pd.DataFrame(rows)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


def extract_compact_alert_payload(record: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, pd.Series):
        record = record.to_dict()
    return {
        "window_id": int(record.get("window_id", -1)),
        "container_id": str(record.get("container_id", "unknown")),
        "machine_id": str(record.get("machine_id", "unknown")),
        "mode": str(record.get("mode", "reconstruction")),
        "recon_score": record.get("recon_score"),
        "forecast_score": record.get("forecast_score"),
        "final_score": float(record.get("final_score", record.get("anomaly_score", 0.0))),
        "anomaly_score": float(record.get("anomaly_score", record.get("final_score", 0.0))),
        "threshold": float(record.get("threshold", 0.0)),
        "dynamic_threshold": record.get("dynamic_threshold"),
        "final_threshold": record.get("final_threshold"),
        "top_k_features": safe_literal_list(record.get("top_k_features", [])),
        "top_k_feature_errors": safe_literal_list(record.get("top_k_feature_errors", [])),
        "feature_error_vector": safe_literal_list(record.get("feature_error_vector", [])),
        "score_over_threshold": float(record.get("score_over_threshold", 0.0)),
        "decision_reason": str(record.get("decision_reason", "")),
        "start_time": int(record.get("start_time", -1)),
        "end_time": int(record.get("end_time", -1)),
        "split": str(record.get("split", "")),
    }


def run_streaming_inference_flow(
    prediction_frame: pd.DataFrame,
    config: EvalConfig,
    gpt_config: GPTConfig | None = None,
    genai_config: GenAIConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply streaming decision logic and optionally invoke Tier-2 GenAI Auditor.

    Priority:
      1. If ``genai_config`` is provided → Two-Tier Verification path
         (Gemini-first, OpenAI fallback, with full provenance columns).
      2. Elif ``gpt_config`` is provided  → legacy OpenAI-only path.
      3. Else                             → Tier-1 only (no LLM calls).
    """
    stream_df = apply_streaming_decision_logic(
        prediction_frame=prediction_frame,
        static_threshold=float(prediction_frame["static_threshold"].iloc[0]),
        config=config,
    )
    gpt_rows: list[dict[str, Any]] = []
    stream_rows: list[dict[str, Any]] = []

    # Build TwoTierVerifier once if genai_config is supplied
    verifier: TwoTierVerifier | None = (
        TwoTierVerifier(config=genai_config) if genai_config is not None else None
    )

    for _, row in stream_df.iterrows():
        record = row.to_dict()
        payload = extract_compact_alert_payload(record)
        summary = build_window_summary(payload)
        record["compact_anomaly_summary"] = json_dumps(summary)
        is_confirmed = record.get("decision") == "confirmed_anomaly"

        record["tier1_flagged"] = is_confirmed
        record["tier2_triggered"] = False
        record["tier2_provider"] = None
        record["tier2_latency_ms"] = None
        record["final_label"] = None
        record["final_severity"] = None
        record["gpt_triggered"] = False

        if is_confirmed and verifier is not None:
            # ── Two-Tier Verification path ────────────────────────────────
            result = verifier.verify(
                record=record,
                recent_logs=None,
                recent_events=None,
            )
            record.update({
                "tier2_triggered": result.tier2_triggered,
                "tier2_provider": result.tier2_provider,
                "tier2_model": result.tier2_model,
                "tier2_latency_ms": result.tier2_latency_ms,
                "tier2_used_fallback": result.tier2_used_fallback,
                "tier2_fallback_reason": result.tier2_fallback_reason,
                "final_label": result.final_label,
                "final_severity": result.final_severity,
                "final_root_cause": result.final_root_cause,
                "final_impact_analysis": result.final_impact_analysis,
                "final_explanation": result.final_explanation,
                "final_recommended_action": result.final_recommended_action,
                # backward-compat aliases
                "gpt_triggered": result.tier2_triggered,
                "gpt_label": result.final_label,
                "gpt_severity": result.final_severity,
                "gpt_recommended_action": result.final_recommended_action,
                "gpt_explanation": result.final_explanation,
            })
            if result.tier2_triggered:
                gpt_rows.append({**record, "label": result.final_label,
                                  "severity": result.final_severity})

        elif is_confirmed and gpt_config is not None:
            # ── Legacy OpenAI-only path ───────────────────────────────────
            adjudicated = adjudicate_anomaly(payload, config=gpt_config)
            gpt_rows.append(adjudicated)
            record.update({
                "gpt_triggered": True,
                "gpt_label": adjudicated["label"],
                "gpt_severity": adjudicated["severity"],
                "gpt_recommended_action": adjudicated["recommended_action"],
                "gpt_explanation": adjudicated["explanation"],
                "final_label": adjudicated["label"],
                "final_severity": adjudicated["severity"],
                "tier2_triggered": True,
                "tier2_provider": adjudicated.get("genai_provider", "openai"),
            })

        stream_rows.append(record)

    stream_output = pd.DataFrame(stream_rows)
    gpt_df = pd.DataFrame(gpt_rows)
    comparison_df = compare_ae_vs_gpt_decisions(gpt_df)
    return stream_output, gpt_df, comparison_df


def send_telegram_notifications(
    alerts: pd.DataFrame,
    config: EvalConfig,
) -> dict[str, Any]:
    report = {"enabled": bool(config.telegram_enabled), "attempted": 0, "sent": 0, "failed": 0, "records": []}
    if not config.telegram_enabled:
        report["reason"] = "disabled"
        return report
    if not config.telegram_bot_token or not config.telegram_chat_id:
        report["reason"] = "missing_credentials"
        return report

    selected = alerts[alerts["decision"] == "confirmed_anomaly"].copy()
    if bool(config.telegram_require_gpt_reason):
        selected = selected[
            selected.get("gpt_explanation", pd.Series("", index=selected.index)).fillna("").astype(str).str.strip().ne("")
        ]
    selected = selected.head(max(0, int(config.telegram_max_alerts))).reset_index(drop=True)
    if selected.empty:
        report["reason"] = "no_matching_alerts"
        return report

    for _, row in selected.iterrows():
        report["attempted"] += 1
        sent, body, error = send_telegram_message(
            bot_token=str(config.telegram_bot_token),
            chat_id=str(config.telegram_chat_id),
            message_text=format_telegram_alert(row.to_dict()),
            timeout_seconds=int(config.telegram_timeout_seconds),
        )
        if sent:
            report["sent"] += 1
            report["records"].append({"window_id": int(row.get("window_id", -1)), "status": "sent", "message_id": body.get("result", {}).get("message_id")})
        else:
            report["failed"] += 1
            report["records"].append({"window_id": int(row.get("window_id", -1)), "status": "failed", "error": error})
    return report


def _save_binary_plots(mode_dir: Path, labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, str]:
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(labels, predictions, labels=[0, 1]), display_labels=["normal", "anomaly"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    confusion_path = mode_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=150)
    plt.close(fig)
    paths["confusion_matrix"] = str(confusion_path.resolve())

    if labels.sum() > 0 and len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        fig.tight_layout()
        roc_path = mode_dir / "roc_curve.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        paths["roc_curve"] = str(roc_path.resolve())

        precision, recall, _ = precision_recall_curve(labels, scores)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label="PR")
        ax.set_title("PR Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.tight_layout()
        pr_path = mode_dir / "pr_curve.png"
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        paths["pr_curve"] = str(pr_path.resolve())

    return paths


def _save_score_plots(mode_dir: Path, stream_df: pd.DataFrame) -> dict[str, str]:
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stream_df.index, stream_df["final_score"], label="score")
    ax.plot(stream_df.index, stream_df["final_threshold"], label="dynamic_threshold")
    ax.set_title("Score vs Dynamic Threshold")
    ax.set_xlabel("Window Order")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    score_path = mode_dir / "score_vs_threshold.png"
    fig.savefig(score_path, dpi=150)
    plt.close(fig)
    paths["score_vs_threshold"] = str(score_path.resolve())

    if {"recon_score", "forecast_score"}.issubset(stream_df.columns):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(stream_df["recon_score"], stream_df["forecast_score"], s=8, alpha=0.5)
        ax.set_title("Reconstruction vs Forecast Score")
        ax.set_xlabel("Recon score")
        ax.set_ylabel("Forecast score")
        fig.tight_layout()
        scatter_path = mode_dir / "recon_vs_forecast.png"
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        paths["recon_vs_forecast"] = str(scatter_path.resolve())

    return paths


def evaluate_model(
    config: EvalConfig | None = None,
    gpt_config: GPTConfig | None = None,
    genai_config: GenAIConfig | None = None,
) -> dict[str, Any]:
    config = config or EvalConfig()
    started_at = perf_counter()
    if config.include_gpt_in_stream and gpt_config is None:
        gpt_config = GPTConfig(output_dir=config.output_dir, evaluation_dir=config.output_dir)

    output_dir = ensure_directory(config.output_dir)
    bundle = load_artifacts(config.dataset_dir, config.model_dir, model_mode=config.model_mode, device=config.device)
    available = [
        normalize_model_mode(mode)
        for mode in config.eval_modes
        if mode in available_modes(bundle["detector_meta"]) or (mode == "hybrid" and {"reconstruction", "forecasting"}.issubset(set(available_modes(bundle["detector_meta"]))))
    ]
    if not available:
        available = [normalize_model_mode(config.model_mode)]

    if "X_test" in bundle:
        if config.split == "train":
            x_split = bundle["X_train"]
            c_split = bundle["C_train"]
            metadata = bundle["metadata_train"]
        else:
            x_split = bundle["X_test"]
            c_split = bundle["C_test"]
            metadata = bundle["metadata_test"]
    else:
        x_split, c_split, metadata = select_split(bundle["X"], bundle["C"], bundle["metadata"], config.split)

    x_scaled = apply_3d_scaler(bundle["x_scaler"], x_split)
    c_scaled = bundle["c_scaler"].transform(c_split).astype(np.float32)
    if config.use_synthetic_injection:
        x_eval, labels, event_ids, event_table = inject_synthetic_anomalies(
            x_scaled=x_scaled,
            metadata=metadata,
            anomaly_ratio=config.synthetic_anomaly_ratio,
            event_span=config.synthetic_event_span,
            feature_count=config.synthetic_feature_count,
            spike_magnitude=config.synthetic_spike_magnitude,
            noise_std=config.synthetic_noise_std,
            random_seed=config.random_seed,
        )
    else:
        x_eval = x_scaled
        labels = np.zeros(len(x_scaled), dtype=np.int32)
        event_ids = np.full(len(x_scaled), -1, dtype=np.int32)
        event_table = pd.DataFrame(columns=["event_id"])

    comparison_rows: list[dict[str, Any]] = []
    mode_outputs: dict[str, dict[str, Any]] = {}
    selected_mode_outputs: dict[str, Any] | None = None
    device = torch.device(bundle["device"])
    for mode in available:
        _eval_log(f"Evaluating mode={mode}", enabled=config.verbose)
        mode_dir = ensure_directory(output_dir / mode)
        scored = score_windows_for_mode(
            bundle=bundle,
            x_scaled=x_eval,
            c_scaled=c_scaled,
            mode=mode,
            batch_size=config.batch_size,
            device=device,
            show_progress=config.show_progress,
        )
        static_threshold = threshold_for_mode(bundle["detector_meta"], mode)
        prediction_frame = build_prediction_frame(
            metadata=metadata,
            scores=scored["final_scores"],
            threshold=static_threshold,
            feature_errors=scored["final_feature_errors"],
            feature_names=bundle["feature_columns"],
            top_k=config.top_k_features,
            labels=labels,
            predicted_labels=(scored["final_scores"] > static_threshold).astype(np.int32),
        )
        prediction_frame["mode"] = mode
        prediction_frame["recon_score"] = scored["recon_scores"].astype(float)
        prediction_frame["forecast_score"] = scored["forecast_scores"].astype(float)
        prediction_frame["final_score"] = scored["final_scores"].astype(float)
        prediction_frame["anomaly_score"] = scored["final_scores"].astype(float)
        prediction_frame["static_threshold"] = float(static_threshold)
        prediction_frame["event_id"] = event_ids.astype(int)

        # Resolve genai_config for Two-Tier Verification path
        _genai_config: GenAIConfig | None = None
        if config.include_tier2_in_stream:
            _genai_config = genai_config if genai_config is not None else GenAIConfig(
                output_dir=config.output_dir
            )

        stream_df, gpt_df, comparison_df = run_streaming_inference_flow(
            prediction_frame=prediction_frame,
            config=config,
            gpt_config=gpt_config if config.include_gpt_in_stream else None,
            genai_config=_genai_config,
        )
        confirmed_alerts = stream_df[stream_df["decision"] == "confirmed_anomaly"].copy()
        predicted_labels = stream_df["predicted_label"].to_numpy(dtype=np.int32)
        metrics = compute_binary_metrics(labels, predicted_labels, stream_df["final_score"].to_numpy(dtype=np.float32))
        event_metrics = compute_event_detection_metrics(
            event_ids=event_ids,
            predictions=predicted_labels,
            relaxed_tolerance=config.relaxed_detection_tolerance,
        )
        plot_paths = {}
        plot_paths.update(_save_binary_plots(mode_dir, labels, predicted_labels, stream_df["final_score"].to_numpy(dtype=np.float32)))
        plot_paths.update(_save_score_plots(mode_dir, stream_df))

        prediction_frame.to_csv(mode_dir / "window_level_predictions.csv", index=False)
        stream_df.to_csv(mode_dir / "realtime_stream_predictions.csv", index=False)
        confirmed_alerts.to_csv(mode_dir / "realtime_alert_candidates.csv", index=False)
        event_table.to_csv(mode_dir / "synthetic_events.csv", index=False)
        write_json(mode_dir / "event_metrics.json", event_metrics)
        if not gpt_df.empty:
            gpt_df.to_csv(mode_dir / "streaming_gpt_decisions.csv", index=False)
            comparison_df.to_csv(mode_dir / "ae_vs_gpt_comparison.csv", index=False)

        summary = {
            "mode": mode,
            "static_threshold": float(static_threshold),
            "num_windows": int(len(prediction_frame)),
            "num_positive_windows": int(labels.sum()),
            "num_confirmed_alerts": int(len(confirmed_alerts)),
            "num_gpt_decisions": int(len(gpt_df)),
            **metrics,
            "event_metrics": event_metrics,
            "plots": plot_paths,
        }
        write_json(mode_dir / "evaluation_summary.json", summary)
        comparison_rows.append(summary)
        mode_outputs[mode] = {
            "prediction_frame": prediction_frame,
            "stream_df": stream_df,
            "gpt_df": gpt_df,
            "comparison_df": comparison_df,
            "summary": summary,
            "mode_dir": mode_dir,
        }

        if mode == normalize_model_mode(config.model_mode):
            selected_mode_outputs = mode_outputs[mode]

    if selected_mode_outputs is None:
        selected_mode_outputs = mode_outputs.get(available[0]) if available else None
    if selected_mode_outputs is None:
        raise RuntimeError("No evaluation outputs were produced.")

    telegram_report = send_telegram_notifications(selected_mode_outputs["stream_df"], config)
    write_json(output_dir / "telegram_notifications.json", telegram_report)
    comparison_table = pd.DataFrame(comparison_rows).sort_values("f1", ascending=False)
    comparison_table.to_csv(output_dir / "mode_comparison.csv", index=False)
    write_json(output_dir / "evaluation_summary.json", {"selected_mode": config.model_mode, "modes": comparison_rows, "telegram_report": telegram_report})
    write_json(output_dir / "event_metrics.json", selected_mode_outputs["summary"]["event_metrics"])

    selected_mode_outputs["prediction_frame"].to_csv(output_dir / "window_level_predictions.csv", index=False)
    selected_mode_outputs["stream_df"].to_csv(output_dir / "realtime_stream_predictions.csv", index=False)
    selected_mode_outputs["stream_df"][selected_mode_outputs["stream_df"]["decision"] == "confirmed_anomaly"].to_csv(output_dir / "realtime_alert_candidates.csv", index=False)
    if not selected_mode_outputs["gpt_df"].empty:
        selected_mode_outputs["gpt_df"].to_csv(output_dir / "streaming_gpt_decisions.csv", index=False)
        selected_mode_outputs["comparison_df"].to_csv(output_dir / "ae_vs_gpt_comparison.csv", index=False)

    _eval_log(f"Completed evaluation in {perf_counter() - started_at:.1f}s", enabled=config.verbose)
    return {
        "evaluation_summary": selected_mode_outputs["summary"],
        "mode_comparison_csv": str((output_dir / "mode_comparison.csv").resolve()),
        "summary_json": str((output_dir / "evaluation_summary.json").resolve()),
        "event_metrics_json": str((output_dir / "event_metrics.json").resolve()),
        "telegram_report_json": str((output_dir / "telegram_notifications.json").resolve()),
        "streaming_frame": selected_mode_outputs["stream_df"],
        "gpt_frame": selected_mode_outputs["gpt_df"],
        "comparison_table": comparison_table,
    }


evaluate_research_pipeline = evaluate_model
