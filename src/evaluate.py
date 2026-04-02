from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
import torch
from tqdm.auto import tqdm

try:
    from config import EvalConfig, GPTConfig
    from gpt_adjudicator import adjudicate_anomaly, build_window_summary, compare_ae_vs_gpt_decisions
    from model import build_model_from_checkpoint
    from telegram_utils import format_telegram_alert, send_telegram_message
    from utils import (
        apply_3d_scaler,
        build_prediction_frame,
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        safe_literal_list,
        write_json,
    )
except ImportError:
    from .config import EvalConfig, GPTConfig
    from .gpt_adjudicator import adjudicate_anomaly, build_window_summary, compare_ae_vs_gpt_decisions
    from .model import build_model_from_checkpoint
    from .telegram_utils import format_telegram_alert, send_telegram_message
    from .utils import (
        apply_3d_scaler,
        build_prediction_frame,
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        safe_literal_list,
        write_json,
    )


def _eval_log(message: str, enabled: bool) -> None:
    if enabled:
        tqdm.write(f"[evaluate] {message}")


def load_artifacts(
    dataset_dir: str | Path,
    model_dir: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    model_path = Path(model_dir)
    checkpoint = torch.load(model_path / "film_ae.pt", map_location="cpu")
    detector_meta = joblib.load(model_path / "detector_meta.joblib")
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
        return {
            "X_train": x_train,
            "X_test": x_test,
            "C_train": c_train,
            "C_test": c_test,
            "metadata_train": _load_split_metadata(dataset_path, split="train", rows=int(x_train.shape[0])),
            "metadata_test": _load_split_metadata(dataset_path, split="test", rows=int(x_test.shape[0])),
            "feature_meta": joblib.load(dataset_path / "feature_meta.joblib"),
            "checkpoint": checkpoint,
            "x_scaler": joblib.load(model_path / "x_scaler.joblib"),
            "c_scaler": joblib.load(model_path / "c_scaler.joblib"),
            "detector_meta": detector_meta,
        }
    return {
        "X": np.load(dataset_path / "X_all.npy", allow_pickle=False),
        "C": np.load(dataset_path / "C_all.npy", allow_pickle=False),
        "metadata": pd.read_csv(dataset_path / "window_metadata.csv"),
        "feature_meta": joblib.load(dataset_path / "feature_meta.joblib"),
        "checkpoint": checkpoint,
        "x_scaler": joblib.load(model_path / "x_scaler.joblib"),
        "c_scaler": joblib.load(model_path / "c_scaler.joblib"),
        "detector_meta": detector_meta,
    }


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


def reconstruct_windows(
    model: torch.nn.Module,
    x_array: np.ndarray,
    c_array: np.ndarray,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    total_batches = max(1, (len(x_array) + batch_size - 1) // batch_size)
    progress = tqdm(
        range(0, len(x_array), batch_size),
        total=total_batches,
        desc="Reconstructing windows",
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    model.eval()
    with torch.no_grad():
        for start in progress:
            end = start + batch_size
            x_batch = torch.as_tensor(x_array[start:end], dtype=torch.float32, device=device)
            c_batch = torch.as_tensor(c_array[start:end], dtype=torch.float32, device=device)
            predictions.append(model(x_batch, c_batch).cpu().numpy())
            progress.set_postfix(windows=min(end, len(x_array)))
    progress.close()
    return np.concatenate(predictions, axis=0)


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
    pr_auc = float(average_precision_score(labels, scores)) if labels.sum() > 0 else 0.0
    if labels.sum() > 0 and len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
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


def build_realtime_alert_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    candidate_time_columns = [
        "end_time",
        "window_end_time",
        "ts_end",
        "timestamp",
        "window_id",
    ]
    sort_columns = [column for column in candidate_time_columns if column in predictions.columns]
    if not sort_columns:
        sort_columns = ["window_id"]

    realtime_stream = predictions.sort_values(sort_columns).reset_index(drop=True).copy()
    realtime_stream["realtime_step"] = np.arange(len(realtime_stream))
    realtime_stream["alert_ready"] = realtime_stream["predicted_label"] == 1
    return realtime_stream


def extract_compact_alert_payload(record: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, pd.Series):
        record = record.to_dict()
    return {
        "window_id": int(record.get("window_id", -1)),
        "container_id": str(record.get("container_id", "unknown")),
        "machine_id": str(record.get("machine_id", "unknown")),
        "anomaly_score": float(record.get("anomaly_score", 0.0)),
        "threshold": float(record.get("threshold", 0.0)),
        "top_k_features": safe_literal_list(record.get("top_k_features", [])),
        "top_k_feature_errors": safe_literal_list(record.get("top_k_feature_errors", [])),
        "feature_error_vector": safe_literal_list(record.get("feature_error_vector", [])),
        "score_over_threshold": float(record.get("score_over_threshold", 0.0)),
        "container_app_du": str(record.get("container_app_du", "unknown")),
        "container_status": str(record.get("container_status", "unknown")),
        "machine_status": str(record.get("machine_status", "unknown")),
        "machine_failure_domain_1": str(record.get("machine_failure_domain_1", "unknown")),
        "machine_failure_domain_2": str(record.get("machine_failure_domain_2", "unknown")),
        "start_time": int(record.get("start_time", -1)),
        "end_time": int(record.get("end_time", -1)),
        "split": str(record.get("split", "")),
    }


def run_streaming_inference_flow(
    prediction_frame: pd.DataFrame,
    gpt_config: GPTConfig | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    realtime_stream = build_realtime_alert_candidates(prediction_frame)
    stream_rows: list[dict[str, Any]] = []
    gpt_rows: list[dict[str, Any]] = []
    progress = tqdm(
        realtime_stream.iterrows(),
        total=len(realtime_stream),
        desc="Streaming inference",
        unit="window",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for _, row in progress:
        record = row.to_dict()
        payload = extract_compact_alert_payload(record)
        summary = build_window_summary(payload)

        stream_row = {
            **record,
            "gpt_triggered": False,
            "gpt_label": None,
            "gpt_severity": None,
            "gpt_recommended_action": None,
            "gpt_explanation": None,
            "compact_anomaly_summary": json_dumps(summary),
        }

        if int(record.get("predicted_label", 0)) == 1 and gpt_config is not None:
            adjudicated = adjudicate_anomaly(payload, config=gpt_config)
            gpt_rows.append(adjudicated)
            stream_row.update(
                {
                    "gpt_triggered": True,
                    "gpt_label": adjudicated["label"],
                    "gpt_severity": adjudicated["severity"],
                    "gpt_recommended_action": adjudicated["recommended_action"],
                    "gpt_explanation": adjudicated["explanation"],
                }
            )

        stream_rows.append(stream_row)
        progress.set_postfix(gpt=len(gpt_rows))

    progress.close()
    stream_df = pd.DataFrame(stream_rows)
    gpt_df = pd.DataFrame(gpt_rows)
    comparison_df = compare_ae_vs_gpt_decisions(gpt_df)
    return stream_df, gpt_df, comparison_df


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


def send_telegram_notifications(
    alerts: pd.DataFrame,
    config: EvalConfig,
    show_progress: bool = True,
) -> dict[str, Any]:
    report = {
        "enabled": bool(config.telegram_enabled),
        "attempted": 0,
        "sent": 0,
        "failed": 0,
        "reason": None,
        "records": [],
    }
    if not config.telegram_enabled:
        report["reason"] = "disabled"
        return report

    bot_token = config.telegram_bot_token
    chat_id = config.telegram_chat_id
    if not bot_token or not chat_id:
        report["reason"] = "missing_credentials"
        return report

    selected_alerts = alerts.copy()
    if bool(config.telegram_critical_only):
        gpt_label = selected_alerts.get("gpt_label")
        gpt_severity = selected_alerts.get("gpt_severity")
        if gpt_label is not None or gpt_severity is not None:
            selected_alerts = selected_alerts[
                (
                    selected_alerts.get("gpt_label", pd.Series("", index=selected_alerts.index)).fillna("").astype(str).str.lower()
                    == "critical"
                )
                | (
                    selected_alerts.get("gpt_severity", pd.Series("", index=selected_alerts.index)).fillna("").astype(str).str.lower()
                    == "high"
                )
            ]
        else:
            selected_alerts = selected_alerts.iloc[0:0]

    if bool(config.telegram_require_gpt_reason):
        selected_alerts = selected_alerts[
            selected_alerts.get("gpt_explanation", pd.Series("", index=selected_alerts.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]

    selected_alerts = selected_alerts.head(max(0, int(config.telegram_max_alerts))).reset_index(drop=True)
    if selected_alerts.empty:
        report["reason"] = "no_matching_alerts"
        return report

    progress = tqdm(
        selected_alerts.iterrows(),
        total=len(selected_alerts),
        desc="Sending Telegram alerts",
        unit="alert",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for _, row in progress:
        record = row.to_dict()
        report["attempted"] += 1
        try:
            sent, body, error = send_telegram_message(
                bot_token=bot_token,
                chat_id=chat_id,
                message_text=format_telegram_alert(record),
                timeout_seconds=int(config.telegram_timeout_seconds),
            )
            if not sent:
                raise RuntimeError(error or "telegram_send_failed")
            report["sent"] += 1
            report["records"].append(
                {
                    "window_id": int(record.get("window_id", -1)),
                    "status": "sent",
                    "message_id": body.get("result", {}).get("message_id"),
                }
            )
        except Exception as exc:
            report["failed"] += 1
            report["records"].append(
                {
                    "window_id": int(record.get("window_id", -1)),
                    "status": "failed",
                    "error": str(exc),
                }
            )
        progress.set_postfix(sent=report["sent"], failed=report["failed"])
    progress.close()
    return report


def evaluate_model(
    config: EvalConfig | None = None,
    gpt_config: GPTConfig | None = None,
) -> dict[str, Any]:
    config = config or EvalConfig()
    started_at = perf_counter()
    if config.include_gpt_in_stream and gpt_config is None:
        gpt_config = GPTConfig(output_dir=config.output_dir, evaluation_dir=config.output_dir)

    output_dir = ensure_directory(config.output_dir)
    resolved_device = choose_device(config.device)
    _eval_log(
        (
            f"Starting evaluation: split={config.split}, device={resolved_device}, "
            f"synthetic_injection={config.use_synthetic_injection}, "
            f"gpt_stream={config.include_gpt_in_stream}, telegram={config.telegram_enabled}"
        ),
        enabled=config.verbose,
    )

    bundle = load_artifacts(config.dataset_dir, config.model_dir)
    if "X_test" in bundle:
        if config.split == "train":
            x_split = bundle["X_train"]
            c_split = bundle["C_train"]
            metadata = bundle["metadata_train"]
        elif config.split == "test":
            x_split = bundle["X_test"]
            c_split = bundle["C_test"]
            metadata = bundle["metadata_test"]
        else:
            raise ValueError(
                f"Split '{config.split}' is not available for split memmap datasets. "
                "Use 'train' or 'test', or rebuild a legacy dataset with window_metadata.csv."
            )
    else:
        x_split, c_split, metadata = select_split(bundle["X"], bundle["C"], bundle["metadata"], config.split)
    feature_names = bundle["feature_meta"]["feature_columns"]
    _eval_log(
        f"Loaded split with {len(metadata)} windows and {len(feature_names)} features.",
        enabled=config.verbose,
    )

    _eval_log("Scaling input features and context vectors.", enabled=config.verbose)
    x_scaled = apply_3d_scaler(bundle["x_scaler"], x_split)
    c_scaled = bundle["c_scaler"].transform(c_split).astype(np.float32)

    if config.use_synthetic_injection:
        _eval_log("Injecting synthetic anomalies into evaluation windows.", enabled=config.verbose)
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

    device = torch.device(resolved_device)
    _eval_log("Running model reconstruction.", enabled=config.verbose)
    model = build_model_from_checkpoint(bundle["checkpoint"], device=device)
    predictions = reconstruct_windows(
        model=model,
        x_array=x_eval,
        c_array=c_scaled,
        batch_size=config.batch_size,
        device=device,
        show_progress=config.show_progress,
    )

    _eval_log("Computing anomaly scores and threshold decisions.", enabled=config.verbose)
    feature_errors = compute_feature_error_matrix(x_eval, predictions)
    scores = compute_window_scores(feature_errors)
    threshold = float(bundle["detector_meta"]["threshold"])
    predicted_labels = (scores > threshold).astype(np.int32)

    prediction_frame = build_prediction_frame(
        metadata=metadata,
        scores=scores,
        threshold=threshold,
        feature_errors=feature_errors,
        feature_names=feature_names,
        top_k=config.top_k_features,
        labels=labels,
        predicted_labels=predicted_labels,
    )
    prediction_frame["event_id"] = event_ids.astype(int)

    _eval_log("Simulating realtime stream and optional GPT adjudication.", enabled=config.verbose)
    stream_df, gpt_df, comparison_df = run_streaming_inference_flow(
        prediction_frame=prediction_frame,
        gpt_config=gpt_config if config.include_gpt_in_stream else None,
        show_progress=config.show_progress,
    )
    realtime_alerts = stream_df[stream_df["alert_ready"]].copy()

    metrics = compute_binary_metrics(labels, predicted_labels, scores)
    event_metrics = compute_event_detection_metrics(
        event_ids=event_ids,
        predictions=predicted_labels,
        relaxed_tolerance=config.relaxed_detection_tolerance,
    )

    evaluation_summary = {
        "threshold": threshold,
        **metrics,
        "num_windows": int(len(prediction_frame)),
        "num_positive_windows": int(labels.sum()),
        "num_realtime_alerts": int(len(realtime_alerts)),
        "num_gpt_decisions": int(len(gpt_df)),
        "split": config.split,
    }
    _eval_log(
        (
            f"Detected {int(predicted_labels.sum())} anomalous windows at threshold {threshold:.6f}. "
            f"Realtime alerts={len(realtime_alerts)}, GPT decisions={len(gpt_df)}."
        ),
        enabled=config.verbose,
    )

    prediction_csv_path = output_dir / "window_level_predictions.csv"
    top_windows_path = output_dir / "top_anomalous_windows.csv"
    realtime_stream_path = output_dir / "realtime_stream_predictions.csv"
    realtime_alerts_path = output_dir / "realtime_alert_candidates.csv"
    streaming_gpt_path = output_dir / "streaming_gpt_decisions.csv"
    comparison_csv_path = output_dir / "ae_vs_gpt_comparison.csv"
    comparison_json_path = output_dir / "ae_vs_gpt_comparison.json"
    event_metrics_path = output_dir / "event_metrics.json"
    summary_path = output_dir / "evaluation_summary.json"
    telegram_report_path = output_dir / "telegram_notifications.json"

    prediction_frame.to_csv(prediction_csv_path, index=False)
    prediction_frame.sort_values("anomaly_score", ascending=False).head(100).to_csv(top_windows_path, index=False)
    stream_df.to_csv(realtime_stream_path, index=False)
    realtime_alerts.to_csv(realtime_alerts_path, index=False)
    event_table.to_csv(output_dir / "synthetic_events.csv", index=False)
    write_json(event_metrics_path, event_metrics)

    if not gpt_df.empty:
        gpt_df.to_csv(streaming_gpt_path, index=False)
        comparison_df.to_csv(comparison_csv_path, index=False)
        write_json(comparison_json_path, {"records": comparison_df.to_dict(orient="records")})

    if config.telegram_enabled:
        _eval_log("Sending Telegram notifications for selected alerts.", enabled=config.verbose)
    telegram_report = send_telegram_notifications(
        realtime_alerts,
        config=config,
        show_progress=config.show_progress,
    )
    write_json(telegram_report_path, telegram_report)
    evaluation_summary["telegram_notifications_sent"] = int(telegram_report["sent"])
    evaluation_summary["telegram_notifications_failed"] = int(telegram_report["failed"])
    write_json(summary_path, evaluation_summary)
    elapsed_seconds = perf_counter() - started_at
    _eval_log(
        (
            f"Completed in {elapsed_seconds:.1f}s. "
            f"F1={evaluation_summary['f1']:.4f}, PR-AUC={evaluation_summary['pr_auc']:.4f}, "
            f"ROC-AUC={evaluation_summary['roc_auc']:.4f}."
        ),
        enabled=config.verbose,
    )

    return {
        "prediction_csv": str(prediction_csv_path.resolve()),
        "top_windows_csv": str(top_windows_path.resolve()),
        "realtime_stream_csv": str(realtime_stream_path.resolve()),
        "realtime_alerts_csv": str(realtime_alerts_path.resolve()),
        "streaming_gpt_csv": str(streaming_gpt_path.resolve()) if streaming_gpt_path.exists() else None,
        "comparison_csv": str(comparison_csv_path.resolve()) if comparison_csv_path.exists() else None,
        "summary_json": str(summary_path.resolve()),
        "event_metrics_json": str(event_metrics_path.resolve()),
        "telegram_report_json": str(telegram_report_path.resolve()),
        "evaluation_summary": evaluation_summary,
        "event_metrics": event_metrics,
        "telegram_report": telegram_report,
        "streaming_frame": stream_df,
        "gpt_frame": gpt_df,
    }


evaluate_research_pipeline = evaluate_model


@dataclass
class StreamingInferenceResult:
    entity_id: str
    ready: bool
    anomaly_score: float | None = None
    threshold: float | None = None
    predicted_label: int | None = None
    top_k_features: list[str] | None = None
    top_k_feature_errors: list[float] | None = None
    feature_error_vector: list[float] | None = None
    metadata: dict[str, Any] | None = None


class StreamingFiLMAnomalyDetector:
    """
    Maintains sequential per-entity windows for near-real-time inference.

    GPT is intentionally not called here. This class only produces the compact
    anomaly summary that can be sent to the post-threshold adjudicator.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x_scaler: Any,
        c_scaler: Any,
        detector_meta: dict[str, Any],
        feature_names: list[str],
        top_k_features: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.x_scaler = x_scaler
        self.c_scaler = c_scaler
        self.threshold = float(detector_meta["threshold"])
        self.window_size = int(detector_meta["window_size"])
        self.feature_names = feature_names
        self.top_k_features = int(top_k_features)
        self.device = torch.device(device)
        self.buffers: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(
        self,
        entity_id: str,
        feature_row: np.ndarray,
        context_vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> StreamingInferenceResult:
        buffer = self.buffers[str(entity_id)]
        buffer.append(np.asarray(feature_row, dtype=np.float32))
        if len(buffer) < self.window_size:
            return StreamingInferenceResult(entity_id=str(entity_id), ready=False, metadata=metadata)

        window = np.stack(list(buffer), axis=0)
        window = np.log1p(np.clip(window, a_min=0.0, a_max=None)).astype(np.float32)
        window_scaled = apply_3d_scaler(self.x_scaler, window[None, ...])
        context_scaled = self.c_scaler.transform(
            np.asarray(context_vector, dtype=np.float32).reshape(1, -1)
        ).astype(np.float32)

        with torch.no_grad():
            reconstructed = self.model(
                torch.as_tensor(window_scaled, dtype=torch.float32, device=self.device),
                torch.as_tensor(context_scaled, dtype=torch.float32, device=self.device),
            ).cpu().numpy()

        feature_errors = compute_feature_error_matrix(window_scaled, reconstructed)[0]
        score = float(compute_window_scores(feature_errors[None, ...])[0])
        top_indices = np.argsort(feature_errors)[::-1][: self.top_k_features]
        predicted_label = int(score > self.threshold)
        return StreamingInferenceResult(
            entity_id=str(entity_id),
            ready=True,
            anomaly_score=score,
            threshold=self.threshold,
            predicted_label=predicted_label,
            top_k_features=[self.feature_names[index] for index in top_indices],
            top_k_feature_errors=[float(feature_errors[index]) for index in top_indices],
            feature_error_vector=[float(value) for value in feature_errors],
            metadata=metadata or {},
        )
