from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

try:
    from config import DatasetConfig, TrainConfig
    from dataset import (
        CONTAINER_META_COLUMNS,
        CONTAINER_USAGE_COLUMNS,
        MACHINE_META_COLUMNS,
        MACHINE_USAGE_COLUMNS,
        clean_container_meta,
        clean_container_usage,
        clean_machine_meta,
        clean_machine_usage,
        iter_tar_csv_chunks,
    )
    from model import FiLMAutoencoder
    from utils import (
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        set_random_seed,
        write_json,
    )
except ImportError:
    from .config import DatasetConfig, TrainConfig
    from .dataset import (
        CONTAINER_META_COLUMNS,
        CONTAINER_USAGE_COLUMNS,
        MACHINE_META_COLUMNS,
        MACHINE_USAGE_COLUMNS,
        clean_container_meta,
        clean_container_usage,
        clean_machine_meta,
        clean_machine_usage,
        iter_tar_csv_chunks,
    )
    from .model import FiLMAutoencoder
    from .utils import (
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        set_random_seed,
        write_json,
    )


DEFAULT_FEATURE_COLUMNS = DatasetConfig().feature_columns
DEFAULT_CONTAINER_CONTEXT_NUMERIC = DatasetConfig().container_context_numeric
DEFAULT_CONTAINER_CONTEXT_CATEGORICAL = DatasetConfig().container_context_categorical
DEFAULT_MACHINE_CONTEXT_NUMERIC = DatasetConfig().machine_context_numeric
DEFAULT_MACHINE_CONTEXT_CATEGORICAL = DatasetConfig().machine_context_categorical


@dataclass
class StreamingTrainConfig:
    raw_data_dir: Path = field(default_factory=lambda: DatasetConfig().raw_data_dir)
    model_dir: Path = field(default_factory=lambda: TrainConfig().model_dir / "streaming")
    container_meta_name: str = "container_meta.tar.gz"
    container_usage_name: str = "container_usage.tar.gz"
    machine_meta_name: str = "machine_meta.tar.gz"
    machine_usage_name: str = "machine_usage.tar.gz"
    chunksize: int = 200_000
    max_usage_rows: int | None = DatasetConfig().max_usage_rows
    max_container_meta_rows: int | None = None
    max_machine_usage_rows: int | None = DatasetConfig().max_machine_usage_rows
    max_machine_meta_rows: int | None = None
    container_limit: int | None = None
    window_size: int = DatasetConfig().window_size
    stride: int = DatasetConfig().stride
    train_ratio: float = DatasetConfig().train_ratio
    val_ratio: float = DatasetConfig().val_ratio
    min_window_observed_ratio: float = DatasetConfig().min_window_observed_ratio
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    container_context_numeric: tuple[str, ...] = DEFAULT_CONTAINER_CONTEXT_NUMERIC
    container_context_categorical: tuple[str, ...] = DEFAULT_CONTAINER_CONTEXT_CATEGORICAL
    machine_context_numeric: tuple[str, ...] = DEFAULT_MACHINE_CONTEXT_NUMERIC
    machine_context_categorical: tuple[str, ...] = DEFAULT_MACHINE_CONTEXT_CATEGORICAL
    batch_size: int = TrainConfig().batch_size
    epochs: int = 20
    learning_rate: float = TrainConfig().learning_rate
    weight_decay: float = TrainConfig().weight_decay
    patience: int = 8
    lr_plateau_patience: int = TrainConfig().lr_plateau_patience
    lr_plateau_factor: float = TrainConfig().lr_plateau_factor
    threshold_quantile: float = TrainConfig().threshold_quantile
    random_seed: int = TrainConfig().random_seed
    units: int = TrainConfig().units
    latent: int = TrainConfig().latent
    score_mode: str = TrainConfig().score_mode
    device: str = "cuda"

    @property
    def archive_paths(self) -> dict[str, Path]:
        return {
            "container_meta": self.raw_data_dir / self.container_meta_name,
            "container_usage": self.raw_data_dir / self.container_usage_name,
            "machine_meta": self.raw_data_dir / self.machine_meta_name,
            "machine_usage": self.raw_data_dir / self.machine_usage_name,
        }

    @property
    def context_columns(self) -> tuple[str, ...]:
        return (
            self.container_context_numeric
            + self.machine_context_numeric
            + self.container_context_categorical
            + self.machine_context_categorical
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_data_dir"] = str(self.raw_data_dir)
        payload["model_dir"] = str(self.model_dir)
        payload["archive_paths"] = {key: str(value) for key, value in self.archive_paths.items()}
        payload["context_columns"] = list(self.context_columns)
        return payload


@dataclass
class ArchiveSpec:
    name: str
    path: Path
    columns: list[str]
    cleaner: Callable[[pd.DataFrame], pd.DataFrame]
    max_rows: int | None = None


@dataclass
class WindowSample:
    entity_id: str
    machine_id: str
    time_stamp: int
    window: np.ndarray
    context: np.ndarray
    split: str


class PeekableRecordStream:
    def __init__(self, iterator: Iterator[dict[str, Any]]) -> None:
        self._iterator = iterator
        self._next: dict[str, Any] | None = None
        self._load_next()

    def _load_next(self) -> None:
        try:
            self._next = next(self._iterator)
        except StopIteration:
            self._next = None

    def peek(self) -> dict[str, Any] | None:
        return self._next

    def pop(self) -> dict[str, Any] | None:
        record = self._next
        self._load_next()
        return record


class OnlineStandardScaler:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.count = 0
        self.mean_: np.ndarray | None = None
        self.m2_: np.ndarray | None = None

    def partial_fit(self, array: np.ndarray) -> "OnlineStandardScaler":
        values = np.asarray(array, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.size == 0:
            return self

        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)

        if self.mean_ is None or self.m2_ is None:
            self.count = batch_count
            self.mean_ = batch_mean
            self.m2_ = batch_m2
            return self

        delta = batch_mean - self.mean_
        total = self.count + batch_count
        self.mean_ = self.mean_ + delta * (batch_count / total)
        self.m2_ = self.m2_ + batch_m2 + (delta * delta) * (self.count * batch_count / total)
        self.count = total
        return self

    @property
    def var_(self) -> np.ndarray:
        if self.mean_ is None or self.m2_ is None or self.count == 0:
            raise RuntimeError("Scaler has not been fitted.")
        return np.maximum(self.m2_ / max(1, self.count), self.eps)

    @property
    def scale_(self) -> np.ndarray:
        return np.sqrt(self.var_)

    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        values = np.asarray(array, dtype=np.float32)
        return ((values - self.mean_.astype(np.float32)) / self.scale_.astype(np.float32)).astype(np.float32)


class CategoryVocabulary:
    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)
        self.values: dict[str, set[str]] = {column: set() for column in self.columns}
        self.mappings: dict[str, dict[str, int]] = {}

    def observe(self, record: dict[str, Any]) -> None:
        for column in self.columns:
            value = str(record.get(column, "unknown") or "unknown")
            self.values[column].add(value)

    def freeze(self) -> None:
        self.mappings = {
            column: {value: index for index, value in enumerate(sorted(values))}
            for column, values in self.values.items()
        }

    def encode(self, record: dict[str, Any], columns: list[str]) -> np.ndarray:
        encoded: list[float] = []
        for column in columns:
            value = str(record.get(column, "unknown") or "unknown")
            encoded.append(float(self.mappings.get(column, {}).get(value, -1)))
        return np.asarray(encoded, dtype=np.float32)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "categories": {
                column: sorted(values)
                for column, values in self.values.items()
            },
        }


@dataclass
class EntitySplitBoundaries:
    train_windows: int
    val_windows: int
    test_windows: int

    @property
    def total_windows(self) -> int:
        return self.train_windows + self.val_windows + self.test_windows

    def split_for(self, window_index: int) -> str:
        if window_index < self.train_windows:
            return "train"
        if window_index < self.train_windows + self.val_windows:
            return "val"
        return "test"


@dataclass
class EntityWindowState:
    features: deque[np.ndarray]
    observed: deque[np.ndarray]
    rows_seen: int = 0
    windows_emitted: int = 0


class ChunkedAlibabaRawLoader:
    """
    Streams raw Alibaba archives directly from tar.gz members.

    Some archives are not globally ordered by `time_stamp`, so each stream is
    normalized into timestamp order before the streaming as-of join runs.
    """

    CONTAINER_META_RENAME = {
        "app_du": "container_app_du",
        "status": "container_status",
        "cpu_request": "container_cpu_request",
        "cpu_limit": "container_cpu_limit",
        "mem_size": "container_mem_size",
    }
    MACHINE_USAGE_RENAME = {
        "cpu_util": "machine_cpu_util",
        "mem_util": "machine_mem_util",
        "mem_gps": "machine_mem_gps",
        "mpki": "machine_mpki",
        "net_in": "machine_net_in",
        "net_out": "machine_net_out",
        "disk_io": "machine_disk_io",
    }
    MACHINE_META_RENAME = {
        "failure_domain_1": "machine_failure_domain_1",
        "failure_domain_2": "machine_failure_domain_2",
        "cpu_num": "machine_cpu_num",
        "mem_size": "machine_mem_size",
        "status": "machine_status",
    }

    def __init__(self, config: StreamingTrainConfig, allowed_entities: set[str] | None = None) -> None:
        self.config = config
        self.allowed_entities = allowed_entities

    def _archive_specs(self) -> dict[str, ArchiveSpec]:
        return {
            "container_usage": ArchiveSpec(
                name="container_usage",
                path=self.config.archive_paths["container_usage"],
                columns=list(CONTAINER_USAGE_COLUMNS),
                cleaner=clean_container_usage,
                max_rows=self.config.max_usage_rows,
            ),
            "container_meta": ArchiveSpec(
                name="container_meta",
                path=self.config.archive_paths["container_meta"],
                columns=list(CONTAINER_META_COLUMNS),
                cleaner=clean_container_meta,
                max_rows=self.config.max_container_meta_rows,
            ),
            "machine_usage": ArchiveSpec(
                name="machine_usage",
                path=self.config.archive_paths["machine_usage"],
                columns=list(MACHINE_USAGE_COLUMNS),
                cleaner=clean_machine_usage,
                max_rows=self.config.max_machine_usage_rows,
            ),
            "machine_meta": ArchiveSpec(
                name="machine_meta",
                path=self.config.archive_paths["machine_meta"],
                columns=list(MACHINE_META_COLUMNS),
                cleaner=clean_machine_meta,
                max_rows=self.config.max_machine_meta_rows,
            ),
        }

    def _iter_records(self, spec: ArchiveSpec) -> Iterator[dict[str, Any]]:
        frames: list[pd.DataFrame] = []
        for chunk in iter_tar_csv_chunks(
            tar_path=spec.path,
            column_names=spec.columns,
            chunksize=self.config.chunksize,
            max_rows=spec.max_rows,
        ):
            frame = spec.cleaner(chunk)
            if frame.empty:
                continue
            frames.append(frame)
        if not frames:
            return

        ordered = pd.concat(frames, ignore_index=True)
        ordered = ordered.sort_values("time_stamp", kind="stable").reset_index(drop=True)
        for record in ordered.to_dict(orient="records"):
            yield record

    @staticmethod
    def _advance_store(
        stream: PeekableRecordStream,
        store: dict[str, dict[str, Any]],
        key_field: str,
        rename_map: dict[str, str],
        timestamp: int,
    ) -> None:
        while stream.peek() is not None and int(stream.peek()["time_stamp"]) <= timestamp:
            record = stream.pop()
            if record is None:
                break
            key = str(record[key_field])
            store[key] = {
                rename_map.get(column, column): value
                for column, value in record.items()
                if column != key_field
            }

    def iter_aligned_rows(self) -> Iterator[dict[str, Any]]:
        specs = self._archive_specs()
        usage_reader = self._iter_records(specs["container_usage"])
        container_meta_stream = PeekableRecordStream(self._iter_records(specs["container_meta"]))
        machine_usage_stream = PeekableRecordStream(self._iter_records(specs["machine_usage"]))
        machine_meta_stream = PeekableRecordStream(self._iter_records(specs["machine_meta"]))

        container_meta_store: dict[str, dict[str, Any]] = {}
        machine_usage_store: dict[str, dict[str, Any]] = {}
        machine_meta_store: dict[str, dict[str, Any]] = {}
        selected_entities: set[str] = set()

        for usage_record in usage_reader:
            entity_id = str(usage_record["container_id"])
            if self.allowed_entities is not None and entity_id not in self.allowed_entities:
                continue

            if self.config.container_limit is not None and self.allowed_entities is None:
                if entity_id not in selected_entities and len(selected_entities) >= self.config.container_limit:
                    continue
                selected_entities.add(entity_id)

            timestamp = int(usage_record["time_stamp"])
            machine_id = str(usage_record["machine_id"])

            self._advance_store(
                container_meta_stream,
                container_meta_store,
                key_field="container_id",
                rename_map=self.CONTAINER_META_RENAME,
                timestamp=timestamp,
            )
            self._advance_store(
                machine_usage_stream,
                machine_usage_store,
                key_field="machine_id",
                rename_map=self.MACHINE_USAGE_RENAME,
                timestamp=timestamp,
            )
            self._advance_store(
                machine_meta_stream,
                machine_meta_store,
                key_field="machine_id",
                rename_map=self.MACHINE_META_RENAME,
                timestamp=timestamp,
            )

            aligned = {
                **usage_record,
                **container_meta_store.get(entity_id, {}),
                **machine_usage_store.get(machine_id, {}),
                **machine_meta_store.get(machine_id, {}),
            }
            aligned["entity_id"] = entity_id
            aligned["machine_id"] = machine_id
            aligned["time_stamp"] = timestamp
            yield aligned


class InMemoryStreamingPreprocessor:
    def __init__(self, config: StreamingTrainConfig, vocabulary: CategoryVocabulary | None = None) -> None:
        self.config = config
        self.vocabulary = vocabulary
        self.feature_columns = list(config.feature_columns)
        self.numeric_context_columns = list(
            config.container_context_numeric + config.machine_context_numeric
        )
        self.categorical_context_columns = list(
            config.container_context_categorical + config.machine_context_categorical
        )
        self.last_feature_values: dict[str, np.ndarray] = {}

    def process(self, row: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(row["entity_id"])
        raw_features = np.asarray(
            [pd.to_numeric(row.get(column), errors="coerce") for column in self.feature_columns],
            dtype=np.float32,
        )
        observed_mask = np.isfinite(raw_features).astype(np.float32)
        previous = self.last_feature_values.get(entity_id)
        if previous is None:
            previous = np.zeros(len(self.feature_columns), dtype=np.float32)

        cleaned_features = np.where(np.isfinite(raw_features), raw_features, previous)
        cleaned_features = np.clip(cleaned_features, a_min=0.0, a_max=None).astype(np.float32)
        transformed_features = np.log1p(cleaned_features).astype(np.float32)
        self.last_feature_values[entity_id] = cleaned_features

        numeric_context = np.asarray(
            [
                float(pd.to_numeric(row.get(column), errors="coerce"))
                if pd.notna(pd.to_numeric(row.get(column), errors="coerce"))
                else 0.0
                for column in self.numeric_context_columns
            ],
            dtype=np.float32,
        )
        categorical_context = {
            column: str(row.get(column, "unknown") or "unknown")
            for column in self.categorical_context_columns
        }
        if self.vocabulary is not None:
            encoded = self.vocabulary.encode(categorical_context, self.categorical_context_columns)
        else:
            encoded = np.zeros(len(self.categorical_context_columns), dtype=np.float32)

        context_vector = np.concatenate([numeric_context, encoded], axis=0).astype(np.float32)
        return {
            "entity_id": entity_id,
            "machine_id": str(row["machine_id"]),
            "time_stamp": int(row["time_stamp"]),
            "features": transformed_features,
            "observed_mask": observed_mask,
            "categorical_context": categorical_context,
            "context_vector": context_vector,
        }


class SlidingWindowStream:
    def __init__(
        self,
        config: StreamingTrainConfig,
        split_plan: dict[str, EntitySplitBoundaries],
    ) -> None:
        self.config = config
        self.split_plan = split_plan
        self.states: dict[str, EntityWindowState] = {}
        self.total_windows_seen = 0

    def _state_for(self, entity_id: str) -> EntityWindowState:
        state = self.states.get(entity_id)
        if state is None:
            state = EntityWindowState(
                features=deque(maxlen=self.config.window_size),
                observed=deque(maxlen=self.config.window_size),
            )
            self.states[entity_id] = state
        return state

    def update(self, row: dict[str, Any]) -> WindowSample | None:
        entity_id = str(row["entity_id"])
        boundaries = self.split_plan.get(entity_id)
        if boundaries is None or boundaries.total_windows == 0:
            return None

        state = self._state_for(entity_id)
        state.rows_seen += 1
        state.features.append(row["features"])
        state.observed.append(row["observed_mask"])

        if state.rows_seen < self.config.window_size:
            return None
        if (state.rows_seen - self.config.window_size) % self.config.stride != 0:
            return None

        window_index = state.windows_emitted
        state.windows_emitted += 1
        self.total_windows_seen += 1

        observed_ratio = float(np.mean(np.stack(state.observed, axis=0)))
        if observed_ratio < self.config.min_window_observed_ratio:
            return None

        return WindowSample(
            entity_id=entity_id,
            machine_id=str(row["machine_id"]),
            time_stamp=int(row["time_stamp"]),
            window=np.stack(state.features, axis=0).astype(np.float32),
            context=row["context_vector"].astype(np.float32),
            split=boundaries.split_for(window_index),
        )


class StreamingWindowDataset(IterableDataset):
    def __init__(
        self,
        config: StreamingTrainConfig,
        split: str,
        split_plan: dict[str, EntitySplitBoundaries],
        vocabulary: CategoryVocabulary,
        x_scaler: OnlineStandardScaler,
        c_scaler: OnlineStandardScaler,
        allowed_entities: set[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.split_plan = split_plan
        self.vocabulary = vocabulary
        self.x_scaler = x_scaler
        self.c_scaler = c_scaler
        self.allowed_entities = allowed_entities
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        loader = ChunkedAlibabaRawLoader(self.config, allowed_entities=self.allowed_entities)
        preprocessor = InMemoryStreamingPreprocessor(self.config, vocabulary=self.vocabulary)
        window_stream = SlidingWindowStream(self.config, split_plan=self.split_plan)

        for aligned_row in loader.iter_aligned_rows():
            processed = preprocessor.process(aligned_row)
            window_sample = window_stream.update(processed)
            if window_sample is None or window_sample.split != self.split:
                continue

            scaled_window = self.x_scaler.transform(window_sample.window)
            scaled_context = self.c_scaler.transform(window_sample.context.reshape(1, -1))[0]
            yield (
                torch.as_tensor(scaled_window, dtype=torch.float32, device=self.device),
                torch.as_tensor(scaled_context, dtype=torch.float32, device=self.device),
            )


def _split_window_counts(group_size: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if group_size <= 0:
        return 0, 0, 0
    if group_size == 1:
        return 1, 0, 0
    if group_size == 2:
        return 1, 0, 1

    train_count = max(1, int(round(group_size * train_ratio)))
    val_count = max(1, int(round(group_size * val_ratio)))
    if train_count + val_count >= group_size:
        val_count = 1
        train_count = max(1, group_size - 2)

    test_count = group_size - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1
    return train_count, val_count, test_count


def build_streaming_statistics(
    config: StreamingTrainConfig,
) -> tuple[set[str] | None, dict[str, int], CategoryVocabulary]:
    loader = ChunkedAlibabaRawLoader(config)
    preprocessor = InMemoryStreamingPreprocessor(config)
    vocabulary = CategoryVocabulary(list(config.container_context_categorical + config.machine_context_categorical))
    row_counts: dict[str, int] = defaultdict(int)
    allowed_entities_order: list[str] = []
    allowed_entities_seen: set[str] = set()

    for aligned_row in loader.iter_aligned_rows():
        processed = preprocessor.process(aligned_row)
        entity_id = processed["entity_id"]

        if config.container_limit is not None and entity_id not in allowed_entities_seen:
            if len(allowed_entities_seen) >= config.container_limit:
                continue
            allowed_entities_seen.add(entity_id)
            allowed_entities_order.append(entity_id)

        if config.container_limit is None and entity_id not in allowed_entities_seen:
            allowed_entities_seen.add(entity_id)
            allowed_entities_order.append(entity_id)

        row_counts[entity_id] += 1
        vocabulary.observe(processed["categorical_context"])

    vocabulary.freeze()
    allowed_entities = set(allowed_entities_order) if config.container_limit is not None else None
    if allowed_entities is not None:
        row_counts = {key: value for key, value in row_counts.items() if key in allowed_entities}
    return allowed_entities, row_counts, vocabulary


def build_split_plan(
    config: StreamingTrainConfig,
    row_counts: dict[str, int],
) -> dict[str, EntitySplitBoundaries]:
    split_plan: dict[str, EntitySplitBoundaries] = {}
    for entity_id, row_count in row_counts.items():
        num_windows = 0
        if row_count >= config.window_size:
            num_windows = 1 + (row_count - config.window_size) // config.stride
        train_count, val_count, test_count = _split_window_counts(
            num_windows,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
        )
        split_plan[entity_id] = EntitySplitBoundaries(
            train_windows=train_count,
            val_windows=val_count,
            test_windows=test_count,
        )
    return split_plan


def fit_streaming_scalers(
    config: StreamingTrainConfig,
    split_plan: dict[str, EntitySplitBoundaries],
    vocabulary: CategoryVocabulary,
    allowed_entities: set[str] | None,
) -> tuple[OnlineStandardScaler, OnlineStandardScaler]:
    x_scaler = OnlineStandardScaler()
    c_scaler = OnlineStandardScaler()
    loader = ChunkedAlibabaRawLoader(config, allowed_entities=allowed_entities)
    preprocessor = InMemoryStreamingPreprocessor(config, vocabulary=vocabulary)
    window_stream = SlidingWindowStream(config, split_plan=split_plan)

    for aligned_row in loader.iter_aligned_rows():
        processed = preprocessor.process(aligned_row)
        window_sample = window_stream.update(processed)
        if window_sample is None or window_sample.split != "train":
            continue
        x_scaler.partial_fit(window_sample.window)
        c_scaler.partial_fit(window_sample.context.reshape(1, -1))

    return x_scaler, c_scaler


def make_streaming_dataloader(
    config: StreamingTrainConfig,
    split: str,
    split_plan: dict[str, EntitySplitBoundaries],
    vocabulary: CategoryVocabulary,
    x_scaler: OnlineStandardScaler,
    c_scaler: OnlineStandardScaler,
    allowed_entities: set[str] | None,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = StreamingWindowDataset(
        config=config,
        split=split,
        split_plan=split_plan,
        vocabulary=vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        allowed_entities=allowed_entities,
        device=device if device is not None and device.type == "cuda" else None,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def run_epoch(
    model: FiLMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_rows = 0

    for x_batch, c_batch in loader:
        x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
        c_batch = c_batch.to(device, non_blocking=device.type == "cuda")

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            reconstructed = model(x_batch, c_batch)
            loss = criterion(reconstructed, x_batch)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = int(x_batch.size(0))
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size

    return total_loss / max(1, total_rows)


def fit_detection_threshold(
    model: FiLMAutoencoder,
    loader: DataLoader,
    quantile: float,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            c_batch = c_batch.to(device, non_blocking=device.type == "cuda")
            reconstructed = model(x_batch, c_batch)
            feature_errors = compute_feature_error_matrix(
                x_batch.cpu().numpy(),
                reconstructed.cpu().numpy(),
            )
            scores.append(compute_window_scores(feature_errors))

    if not scores:
        return 0.0, np.zeros(0, dtype=np.float32)

    all_scores = np.concatenate(scores, axis=0).astype(np.float32)
    threshold = float(np.quantile(all_scores, quantile))
    return threshold, all_scores


def train_streaming_model(config: StreamingTrainConfig | None = None) -> dict[str, Any]:
    config = config or StreamingTrainConfig()
    set_random_seed(config.random_seed)
    model_dir = ensure_directory(config.model_dir)

    allowed_entities, row_counts, vocabulary = build_streaming_statistics(config)
    split_plan = build_split_plan(config, row_counts)
    x_scaler, c_scaler = fit_streaming_scalers(
        config=config,
        split_plan=split_plan,
        vocabulary=vocabulary,
        allowed_entities=allowed_entities,
    )
    device = torch.device(choose_device(config.device))

    train_loader = make_streaming_dataloader(
        config=config,
        split="train",
        split_plan=split_plan,
        vocabulary=vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        allowed_entities=allowed_entities,
        device=device,
    )
    val_loader = make_streaming_dataloader(
        config=config,
        split="val",
        split_plan=split_plan,
        vocabulary=vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        allowed_entities=allowed_entities,
        device=device,
    )

    context_dim = len(config.container_context_numeric + config.machine_context_numeric)
    context_dim += len(config.container_context_categorical + config.machine_context_categorical)
    model = FiLMAutoencoder(
        window_size=config.window_size,
        n_features=len(config.feature_columns),
        context_dim=context_dim,
        units=config.units,
        latent=config.latent,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_plateau_patience,
        factor=config.lr_plateau_factor,
    )

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        started = perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": perf_counter() - started,
                "improved": improved,
            }
        )

        if improved:
            best_val_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold, val_scores = fit_detection_threshold(
        model=model,
        loader=val_loader,
        quantile=config.threshold_quantile,
        device=device,
    )

    split_totals = {
        "train": int(sum(boundaries.train_windows for boundaries in split_plan.values())),
        "val": int(sum(boundaries.val_windows for boundaries in split_plan.values())),
        "test": int(sum(boundaries.test_windows for boundaries in split_plan.values())),
    }
    history_df = pd.DataFrame(history_rows)
    history_path = model_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "window_size": int(config.window_size),
        "n_features": int(len(config.feature_columns)),
        "context_dim": int(context_dim),
        "units": int(config.units),
        "latent": int(config.latent),
        "feature_columns": list(config.feature_columns),
        "context_columns": list(config.context_columns),
        "streaming": True,
    }
    detector_meta = {
        "threshold": threshold,
        "threshold_quantile": float(config.threshold_quantile),
        "window_size": int(config.window_size),
        "n_features": int(len(config.feature_columns)),
        "context_dim": int(context_dim),
        "score_mode": config.score_mode,
        "num_entities": int(len(split_plan)),
        "num_train_windows": split_totals["train"],
        "num_val_windows": split_totals["val"],
        "num_test_windows": split_totals["test"],
        "best_val_loss": float(best_val_loss),
        "val_score_mean": float(np.mean(val_scores)) if len(val_scores) > 0 else 0.0,
        "val_score_std": float(np.std(val_scores)) if len(val_scores) > 0 else 0.0,
        "streaming_source": "raw_chunked_archives",
    }

    torch.save(checkpoint, model_dir / "film_ae.pt")
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(c_scaler, model_dir / "c_scaler.joblib")
    joblib.dump(detector_meta, model_dir / "detector_meta.joblib")
    joblib.dump(vocabulary.to_metadata(), model_dir / "context_encoder.joblib")
    write_json(model_dir / "detector_meta.json", detector_meta)

    summary = {
        "model_dir": str(model_dir.resolve()),
        "history_path": str(history_path.resolve()),
        "model_path": str((model_dir / "film_ae.pt").resolve()),
        "train_config": config.to_dict(),
        "detector_meta": detector_meta,
        "split_totals": split_totals,
        "num_entities": int(len(split_plan)),
        "allowed_entities": int(len(allowed_entities)) if allowed_entities is not None else None,
        "category_metadata": vocabulary.to_metadata(),
        "streaming_notes": {
            "intermediate_files": "none",
            "chunk_boundary_windows": (
                "Each entity keeps only the last window_size-1 rows in a deque. "
                "When the next chunk arrives, those tail rows are still in memory, "
                "so the first windows in the new chunk continue seamlessly."
            ),
            "scaling_strategy": (
                "A second raw-data pass fits online feature/context scalers on train windows only. "
                "Training then consumes a third pass with frozen scalers."
            ),
        },
    }
    write_json(model_dir / "training_summary.json", summary)
    return summary


train_streaming_research_pipeline = train_streaming_model
