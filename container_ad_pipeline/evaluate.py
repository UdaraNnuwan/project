from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .config import EvalConfig
from .dataset import DatasetBundle
from .train import (
    load_trained_model,
    reconstruction_feature_errors,
    reconstruction_scores,
    select_split,
    transform_inputs,
)
from .utils import safe_pr_auc, safe_roc_auc, save_json


@dataclass
class EvaluationArtifacts:
    predictions: pd.DataFrame
    summary: dict[str, Any]
    event_metrics: dict[str, Any]
    top_windows: pd.DataFrame


def inject_synthetic_anomalies(
    X: np.ndarray,
    feature_columns: list[str],
    anomaly_ratio: float = 0.1,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    n_samples, _, n_features = X_noisy.shape
    n_events = max(1, int(n_samples * anomaly_ratio))

    labels = np.zeros(n_samples, dtype=np.int32)
    events: list[dict[str, Any]] = []
    event_id = 0
    indices = rng.choice(np.arange(n_samples), size=n_events, replace=False)
    indices.sort()

    for index in indices:
        affected_count = int(rng.integers(1, min(3, n_features) + 1))
        affected_features = rng.choice(np.arange(n_features), size=affected_count, replace=False)
        anomaly_kind = str(rng.choice(["spike", "drift", "drop", "noise"]))

        for feature_idx in affected_features:
            baseline = np.std(X_noisy[:, :, feature_idx]) + 1e-6
            if anomaly_kind == "spike":
                X_noisy[index, -6:, feature_idx] += 4.0 * baseline
            elif anomaly_kind == "drift":
                X_noisy[index, :, feature_idx] += np.linspace(0.0, 3.0 * baseline, X_noisy.shape[1])
            elif anomaly_kind == "drop":
                X_noisy[index, :, feature_idx] -= 2.0 * baseline
            else:
                X_noisy[index, :, feature_idx] += rng.normal(0.0, 1.5 * baseline, size=X_noisy.shape[1])

        labels[index] = 1
        events.append(
            {
                "event_id": event_id,
                "start_window": int(index),
                "end_window": int(index),
                "anomaly_type": anomaly_kind,
                "affected_features": [feature_columns[i] for i in affected_features],
            }
        )
        event_id += 1

    return X_noisy.astype(np.float32), labels, pd.DataFrame(events)


def predict_with_loaded_model(model, X_scaled: np.ndarray, C_scaled: np.ndarray, batch_size: int = 256) -> np.ndarray:
    from .train import predict_reconstructions

    return predict_reconstructions(model, X_scaled, C_scaled, batch_size=batch_size)


def run_model_inference(
    bundle: DatasetBundle,
    model_path: str | Path,
    x_scaler,
    c_scaler,
    split_name: str = "test",
    injected_X: np.ndarray | None = None,
) -> pd.DataFrame:
    X_split, C_split, meta_split = select_split(bundle, split_name)
    X_eval = injected_X if injected_X is not None else X_split
    X_scaled, C_scaled = transform_inputs(X_eval, C_split, x_scaler, c_scaler)

    model = load_trained_model(model_path)
    X_pred = predict_with_loaded_model(model, X_scaled, C_scaled)
    feature_errors = reconstruction_feature_errors(X_scaled, X_pred)
    scores = reconstruction_scores(X_scaled, X_pred)

    predictions = meta_split.copy()
    predictions["anomaly_score"] = scores
    predictions["feature_error_vector"] = feature_errors.tolist()
    predictions["top_feature_rank"] = feature_errors.argsort(axis=1)[:, ::-1].tolist()
    return predictions


def add_top_k_feature_columns(predictions: pd.DataFrame, feature_columns: list[str], top_k: int) -> pd.DataFrame:
    result = predictions.copy()
    top_names: list[list[str]] = []
    top_scores: list[list[float]] = []
    for feature_error_vector in result["feature_error_vector"]:
        vector = np.asarray(feature_error_vector, dtype=np.float32)
        indices = np.argsort(vector)[::-1][:top_k]
        top_names.append([feature_columns[i] for i in indices])
        top_scores.append([float(vector[i]) for i in indices])
    result["top_k_features"] = top_names
    result["top_k_feature_errors"] = top_scores
    return result


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: np.ndarray,
    threshold: float,
    eval_config: EvalConfig,
    events: pd.DataFrame | None = None,
) -> EvaluationArtifacts:
    scores = predictions["anomaly_score"].to_numpy(dtype=np.float32)
    y_hat = (scores >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, y_hat, average="binary", zero_division=0)

    predictions = predictions.copy()
    if "window_position" not in predictions.columns:
        predictions["window_position"] = np.arange(len(predictions), dtype=int)
    predictions["label"] = labels.astype(int)
    predictions["predicted_label"] = y_hat.astype(int)
    predictions["threshold"] = float(threshold)
    predictions["score_over_threshold"] = predictions["anomaly_score"] - float(threshold)

    summary = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": safe_pr_auc(labels, scores),
        "roc_auc": safe_roc_auc(labels, scores),
        "num_windows": int(len(predictions)),
        "num_positive_windows": int(labels.sum()),
    }

    event_metrics = evaluate_early_detection(y_hat, events, eval_config.relaxed_detection_tolerance)
    top_windows = predictions.sort_values("anomaly_score", ascending=False).head(eval_config.top_n_windows).reset_index(drop=True)
    return EvaluationArtifacts(predictions=predictions, summary=summary, event_metrics=event_metrics, top_windows=top_windows)


def evaluate_early_detection(predictions: np.ndarray, events: pd.DataFrame | None, tolerance: int) -> dict[str, Any]:
    if events is None or events.empty:
        return {
            "strict_event_recall": None,
            "relaxed_event_recall": None,
            "mean_detection_delay": None,
        }

    strict_hits = 0
    relaxed_hits = 0
    delays: list[int] = []
    for row in events.itertuples(index=False):
        start_idx = int(row.start_window)
        end_idx = int(row.end_window)
        strict_hits += int(predictions[start_idx] == 1)

        relaxed_end = min(len(predictions) - 1, start_idx + tolerance, end_idx + tolerance)
        detection_indices = np.flatnonzero(predictions[start_idx : relaxed_end + 1]) + start_idx
        if len(detection_indices) > 0:
            relaxed_hits += 1
            delays.append(int(detection_indices[0] - start_idx))

    total_events = int(len(events))
    return {
        "strict_event_recall": float(strict_hits / total_events) if total_events else None,
        "relaxed_event_recall": float(relaxed_hits / total_events) if total_events else None,
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "num_events": total_events,
    }


def save_evaluation_outputs(artifacts: EvaluationArtifacts, output_dir: str | Path, eval_config: EvalConfig) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / eval_config.save_predictions_filename
    top_windows_path = output_dir / "top_anomalous_windows.csv"
    summary_path = output_dir / eval_config.save_summary_filename
    events_path = output_dir / "event_metrics.json"

    artifacts.predictions.to_csv(predictions_path, index=False)
    artifacts.top_windows.to_csv(top_windows_path, index=False)
    save_json(summary_path, artifacts.summary)
    save_json(events_path, artifacts.event_metrics)

    return {
        "predictions": str(predictions_path),
        "top_windows": str(top_windows_path),
        "summary": str(summary_path),
        "event_metrics": str(events_path),
    }
