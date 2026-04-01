from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from evaluate import StreamingFiLMAnomalyDetector
from model import build_model_from_checkpoint


@dataclass
class RuntimeConfig:
    model_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cpu"
    top_k_features: int = 5


class CategoryEncoder:
    """
    Encodes categorical context using the saved category metadata from training.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.columns = list(metadata.get("columns", []))
        categories = metadata.get("categories", {})
        self.mappings: dict[str, dict[str, int]] = {
            column: {value: index for index, value in enumerate(categories.get(column, []))}
            for column in self.columns
        }

    def encode(self, values: dict[str, Any]) -> np.ndarray:
        encoded: list[float] = []
        for column in self.columns:
            raw = str(values.get(column, "unknown") or "unknown")
            encoded.append(float(self.mappings.get(column, {}).get(raw, -1)))
        return np.asarray(encoded, dtype=np.float32)


class EntityStatePreprocessor:
    """
    Mirrors the row-level preprocessing logic used during streaming training:
    - fill missing features from the previous row of the same entity
    - clip features to >= 0
    - keep raw cleaned features for the detector
    - build the numeric+categorical context vector in the exact training order
    """

    def __init__(
        self,
        feature_columns: list[str],
        numeric_context_columns: list[str],
        categorical_context_columns: list[str],
        category_encoder: CategoryEncoder,
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_context_columns = numeric_context_columns
        self.categorical_context_columns = categorical_context_columns
        self.category_encoder = category_encoder
        self.last_feature_values: dict[str, np.ndarray] = {}

    def process(
        self,
        entity_id: str,
        features: dict[str, Any],
        context_numeric: dict[str, Any],
        context_categorical: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        previous = self.last_feature_values.get(entity_id)
        if previous is None:
            previous = np.zeros(len(self.feature_columns), dtype=np.float32)

        raw_features = np.asarray(
            [pd_to_float(features.get(column)) for column in self.feature_columns],
            dtype=np.float32,
        )
        cleaned_features = np.where(np.isfinite(raw_features), raw_features, previous)
        cleaned_features = np.clip(cleaned_features, a_min=0.0, a_max=None).astype(np.float32)
        self.last_feature_values[entity_id] = cleaned_features

        numeric_vector = np.asarray(
            [pd_to_float(context_numeric.get(column), default=0.0) for column in self.numeric_context_columns],
            dtype=np.float32,
        )
        categorical_payload = {
            column: str(context_categorical.get(column, "unknown") or "unknown")
            for column in self.categorical_context_columns
        }
        categorical_vector = self.category_encoder.encode(categorical_payload)
        context_vector = np.concatenate([numeric_vector, categorical_vector], axis=0).astype(np.float32)

        return cleaned_features, context_vector


def pd_to_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


class MetricPayload(BaseModel):
    entity_id: str = Field(..., description="Container or pod identifier used as the streaming entity key.")
    machine_id: str = Field(default="unknown")
    time_stamp: int | None = Field(default=None)

    features: dict[str, float | None]
    context_numeric: dict[str, float | int | None] = Field(default_factory=dict)
    context_categorical: dict[str, str | None] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveAnomalyService:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.bundle = self._load_bundle(config.model_dir, device=config.device)
        self.preprocessor = EntityStatePreprocessor(
            feature_columns=self.bundle["feature_columns"],
            numeric_context_columns=self.bundle["numeric_context_columns"],
            categorical_context_columns=self.bundle["categorical_context_columns"],
            category_encoder=self.bundle["category_encoder"],
        )
        self.detector = StreamingFiLMAnomalyDetector(
            model=self.bundle["model"],
            x_scaler=self.bundle["x_scaler"],
            c_scaler=self.bundle["c_scaler"],
            detector_meta=self.bundle["detector_meta"],
            feature_names=self.bundle["feature_columns"],
            top_k_features=config.top_k_features,
            device=config.device,
        )

    def _load_bundle(self, model_dir: Path, device: str) -> dict[str, Any]:
        checkpoint = torch.load(model_dir / "film_ae.pt", map_location=device)
        model = build_model_from_checkpoint(checkpoint, device=device)

        x_scaler = joblib.load(model_dir / "x_scaler.joblib")
        c_scaler = joblib.load(model_dir / "c_scaler.joblib")
        detector_meta = joblib.load(model_dir / "detector_meta.joblib")
        category_metadata = joblib.load(model_dir / "context_encoder.joblib")

        feature_columns = list(checkpoint["feature_columns"])
        context_columns = list(checkpoint["context_columns"])

        categorical_columns = list(category_metadata.get("columns", []))
        numeric_columns = [column for column in context_columns if column not in categorical_columns]

        return {
            "model": model,
            "x_scaler": x_scaler,
            "c_scaler": c_scaler,
            "detector_meta": detector_meta,
            "feature_columns": feature_columns,
            "context_columns": context_columns,
            "numeric_context_columns": numeric_columns,
            "categorical_context_columns": categorical_columns,
            "category_encoder": CategoryEncoder(category_metadata),
        }

    def ingest(self, payload: MetricPayload) -> dict[str, Any]:
        cleaned_features, context_vector = self.preprocessor.process(
            entity_id=payload.entity_id,
            features=payload.features,
            context_numeric=payload.context_numeric,
            context_categorical=payload.context_categorical,
        )

        metadata = {
            "machine_id": payload.machine_id,
            "time_stamp": payload.time_stamp,
            **payload.metadata,
        }

        result = self.detector.update(
            entity_id=payload.entity_id,
            feature_row=cleaned_features,
            context_vector=context_vector,
            metadata=metadata,
        )

        response = {
            "entity_id": payload.entity_id,
            "machine_id": payload.machine_id,
            "time_stamp": payload.time_stamp,
            "window_ready": bool(result.ready),
            "threshold": float(self.detector.threshold),
            "window_size": int(self.detector.window_size),
        }

        if not result.ready:
            response["status"] = "buffering"
            return response

        response.update(
            {
                "status": "anomaly" if int(result.predicted_label or 0) == 1 else "normal",
                "predicted_label": int(result.predicted_label or 0),
                "anomaly_score": float(result.anomaly_score or 0.0),
                "top_k_features": result.top_k_features or [],
                "top_k_feature_errors": result.top_k_feature_errors or [],
                "feature_error_vector": result.feature_error_vector or [],
                "metadata": result.metadata or {},
            }
        )
        return response


def create_app(service: LiveAnomalyService) -> FastAPI:
    app = FastAPI(title="FiLM AE Live Anomaly Inference Service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "threshold": float(service.detector.threshold),
            "window_size": int(service.detector.window_size),
            "num_entities_in_memory": int(len(service.detector.buffers)),
            "feature_columns": service.bundle["feature_columns"],
            "context_columns": service.bundle["context_columns"],
        }

    @app.post("/ingest")
    def ingest(payload: MetricPayload) -> dict[str, Any]:
        return service.ingest(payload)

    return app


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Run live streaming anomaly inference service.")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model artifacts.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k-features", default=5, type=int)
    args = parser.parse_args()

    return RuntimeConfig(
        model_dir=Path(args.model_dir).expanduser().resolve(),
        host=args.host,
        port=args.port,
        device=args.device,
        top_k_features=args.top_k_features,
    )


if __name__ == "__main__":
    runtime = parse_args()
    service = LiveAnomalyService(runtime)
    app = create_app(service)

    import uvicorn

    uvicorn.run(app, host=runtime.host, port=runtime.port)
