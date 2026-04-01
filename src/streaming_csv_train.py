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
    from .model import FiLMAutoencoder
    from .utils import (
        choose_device,
        compute_feature_error_matrix,
        compute_window_scores,
        ensure_directory,
        set_random_seed,
        write_json,
    )


@dataclass
class StreamingCSVTrainConfig:
    csv_path: Path
    model_dir: Path
    id_column: str
    time_column: str
    feature_columns: tuple[str, ...]
    context_numeric_columns: tuple[str, ...] = ()
    context_categorical_columns: tuple[str, ...] = ()
    chunksize: int = 200_000
    window_size: int = 24
    stride: int = 6
    min_window_observed_ratio: float = 0.60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    lr_plateau_patience: int = 4
    lr_plateau_factor: float = 0.5
    threshold_quantile: float = 0.995
    random_seed: int = 42
    units: int = 64
    latent: int = 64
    score_mode: str = "mean_feature_mse"
    device: str = "cuda"

    @property
    def use_columns(self) -> list[str]:
        return list(
            dict.fromkeys(
                [self.id_column, self.time_column]
                + list(self.feature_columns)
                + list(self.context_numeric_columns)
                + list(self.context_categorical_columns)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["csv_path"] = str(self.csv_path)
        payload["model_dir"] = str(self.model_dir)
        payload["use_columns"] = self.use_columns
        return payload


@dataclass
class WindowSample:
    entity_id: str
    time_value: int
    window: np.ndarray
    context: np.ndarray
    split: str


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
            self.values[column].add(str(record.get(column, "unknown") or "unknown"))

    def freeze(self) -> None:
        self.mappings = {
            column: {value: index for index, value in enumerate(sorted(values))}
            for column, values in self.values.items()
        }

    def encode(self, record: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                float(self.mappings.get(column, {}).get(str(record.get(column, "unknown") or "unknown"), -1))
                for column in self.columns
            ],
            dtype=np.float32,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "categories": {column: sorted(values) for column, values in self.values.items()},
        }


class ChunkedCSVReader:
    def __init__(self, config: StreamingCSVTrainConfig) -> None:
        self.config = config

    def __iter__(self) -> Iterator[pd.DataFrame]:
        reader = pd.read_csv(
            self.config.csv_path,
            usecols=self.config.use_columns,
            chunksize=self.config.chunksize,
        )
        for chunk in reader:
            if chunk.empty:
                continue
            chunk = chunk.loc[:, self.config.use_columns].copy()
            chunk = chunk.sort_values([self.config.id_column, self.config.time_column], kind="stable")
            yield chunk.reset_index(drop=True)


class InMemoryCSVPreprocessor:
    def __init__(
        self,
        config: StreamingCSVTrainConfig,
        category_vocabulary: CategoryVocabulary | None = None,
        filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        feature_engineering_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        self.config = config
        self.category_vocabulary = category_vocabulary
        self.filter_fn = filter_fn
        self.feature_engineering_fn = feature_engineering_fn or self._default_feature_engineering
        self.last_feature_values: dict[str, np.ndarray] = {}
        self.last_time_values: dict[str, int] = {}

    def _default_feature_engineering(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk[self.config.time_column] = pd.to_numeric(chunk[self.config.time_column], errors="coerce")
        return chunk

    def iter_rows(self, chunk: pd.DataFrame) -> Iterator[dict[str, Any]]:
        chunk = chunk.loc[:, self.config.use_columns].copy()
        chunk = chunk.dropna(subset=[self.config.id_column, self.config.time_column])
        chunk = self.feature_engineering_fn(chunk)
        chunk = chunk.dropna(subset=[self.config.id_column, self.config.time_column])
        if self.filter_fn is not None:
            chunk = self.filter_fn(chunk)
        if chunk.empty:
            return

        for record in chunk.to_dict(orient="records"):
            entity_id = str(record[self.config.id_column])
            time_value = int(pd.to_numeric(record[self.config.time_column], errors="coerce"))
            previous_time = self.last_time_values.get(entity_id)
            if previous_time is not None and time_value < previous_time:
                raise ValueError(
                    f"Input CSV must be ordered by {self.config.id_column}, {self.config.time_column}. "
                    f"Observed {time_value} after {previous_time} for entity {entity_id}."
                )

            raw_features = np.asarray(
                [pd.to_numeric(record.get(column), errors="coerce") for column in self.config.feature_columns],
                dtype=np.float32,
            )
            observed_mask = np.isfinite(raw_features).astype(np.float32)
            previous = self.last_feature_values.get(entity_id)
            if previous is None:
                previous = np.zeros(len(self.config.feature_columns), dtype=np.float32)
            cleaned_features = np.where(np.isfinite(raw_features), raw_features, previous)
            cleaned_features = np.clip(cleaned_features, a_min=0.0, a_max=None).astype(np.float32)
            transformed_features = np.log1p(cleaned_features).astype(np.float32)
            self.last_feature_values[entity_id] = cleaned_features

            context_numeric = np.asarray(
                [
                    float(pd.to_numeric(record.get(column), errors="coerce"))
                    if pd.notna(pd.to_numeric(record.get(column), errors="coerce"))
                    else 0.0
                    for column in self.config.context_numeric_columns
                ],
                dtype=np.float32,
            )

            previous_time = previous_time if previous_time is not None else time_value
            delta_seconds = max(0, time_value - previous_time)
            self.last_time_values[entity_id] = time_value
            context_numeric = np.concatenate(
                [context_numeric, np.asarray([float(delta_seconds)], dtype=np.float32)],
                axis=0,
            )

            categorical_context = {
                column: str(record.get(column, "unknown") or "unknown")
                for column in self.config.context_categorical_columns
            }
            if self.category_vocabulary is not None:
                encoded_context = self.category_vocabulary.encode(categorical_context)
            else:
                encoded_context = np.zeros(len(self.config.context_categorical_columns), dtype=np.float32)

            context_vector = np.concatenate([context_numeric, encoded_context], axis=0).astype(np.float32)
            yield {
                "entity_id": entity_id,
                "time_value": time_value,
                "features": transformed_features,
                "observed_mask": observed_mask,
                "context_vector": context_vector,
                "categorical_context": categorical_context,
            }


class SlidingWindowBuilder:
    def __init__(self, config: StreamingCSVTrainConfig, split_plan: dict[str, EntitySplitBoundaries]) -> None:
        self.config = config
        self.split_plan = split_plan
        self.states: dict[str, EntityWindowState] = {}

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
        entity_id = row["entity_id"]
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

        observed_ratio = float(np.mean(np.stack(state.observed, axis=0)))
        window_index = state.windows_emitted
        state.windows_emitted += 1
        if observed_ratio < self.config.min_window_observed_ratio:
            return None

        return WindowSample(
            entity_id=entity_id,
            time_value=int(row["time_value"]),
            window=np.stack(state.features, axis=0).astype(np.float32),
            context=row["context_vector"].astype(np.float32),
            split=boundaries.split_for(window_index),
        )


class StreamingCSVWindowDataset(IterableDataset):
    def __init__(
        self,
        config: StreamingCSVTrainConfig,
        split: str,
        split_plan: dict[str, EntitySplitBoundaries],
        category_vocabulary: CategoryVocabulary,
        x_scaler: OnlineStandardScaler,
        c_scaler: OnlineStandardScaler,
        filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        feature_engineering_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.split_plan = split_plan
        self.category_vocabulary = category_vocabulary
        self.x_scaler = x_scaler
        self.c_scaler = c_scaler
        self.filter_fn = filter_fn
        self.feature_engineering_fn = feature_engineering_fn
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        reader = ChunkedCSVReader(self.config)
        preprocessor = InMemoryCSVPreprocessor(
            self.config,
            category_vocabulary=self.category_vocabulary,
            filter_fn=self.filter_fn,
            feature_engineering_fn=self.feature_engineering_fn,
        )
        windows = SlidingWindowBuilder(self.config, self.split_plan)

        for chunk in reader:
            for row in preprocessor.iter_rows(chunk):
                window_sample = windows.update(row)
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
    config: StreamingCSVTrainConfig,
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    feature_engineering_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[dict[str, int], CategoryVocabulary]:
    row_counts: dict[str, int] = defaultdict(int)
    category_vocabulary = CategoryVocabulary(list(config.context_categorical_columns))
    reader = ChunkedCSVReader(config)
    preprocessor = InMemoryCSVPreprocessor(
        config,
        category_vocabulary=None,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
    )

    for chunk in reader:
        for row in preprocessor.iter_rows(chunk):
            row_counts[row["entity_id"]] += 1
            category_vocabulary.observe(row["categorical_context"])

    category_vocabulary.freeze()
    return row_counts, category_vocabulary


def build_split_plan(
    config: StreamingCSVTrainConfig,
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
    config: StreamingCSVTrainConfig,
    split_plan: dict[str, EntitySplitBoundaries],
    category_vocabulary: CategoryVocabulary,
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    feature_engineering_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[OnlineStandardScaler, OnlineStandardScaler]:
    x_scaler = OnlineStandardScaler()
    c_scaler = OnlineStandardScaler()
    reader = ChunkedCSVReader(config)
    preprocessor = InMemoryCSVPreprocessor(
        config,
        category_vocabulary=category_vocabulary,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
    )
    windows = SlidingWindowBuilder(config, split_plan)

    for chunk in reader:
        for row in preprocessor.iter_rows(chunk):
            window_sample = windows.update(row)
            if window_sample is None or window_sample.split != "train":
                continue
            x_scaler.partial_fit(window_sample.window)
            c_scaler.partial_fit(window_sample.context.reshape(1, -1))

    return x_scaler, c_scaler


def make_streaming_csv_dataloader(
    config: StreamingCSVTrainConfig,
    split: str,
    split_plan: dict[str, EntitySplitBoundaries],
    category_vocabulary: CategoryVocabulary,
    x_scaler: OnlineStandardScaler,
    c_scaler: OnlineStandardScaler,
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    feature_engineering_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = StreamingCSVWindowDataset(
        config=config,
        split=split,
        split_plan=split_plan,
        category_vocabulary=category_vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
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


def train_streaming_csv_model(
    config: StreamingCSVTrainConfig,
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    feature_engineering_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    set_random_seed(config.random_seed)
    model_dir = ensure_directory(config.model_dir)

    row_counts, category_vocabulary = build_streaming_statistics(
        config,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
    )
    split_plan = build_split_plan(config, row_counts)
    x_scaler, c_scaler = fit_streaming_scalers(
        config,
        split_plan=split_plan,
        category_vocabulary=category_vocabulary,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
    )
    device = torch.device(choose_device(config.device))

    train_loader = make_streaming_csv_dataloader(
        config,
        split="train",
        split_plan=split_plan,
        category_vocabulary=category_vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
        device=device,
    )
    val_loader = make_streaming_csv_dataloader(
        config,
        split="val",
        split_plan=split_plan,
        category_vocabulary=category_vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        filter_fn=filter_fn,
        feature_engineering_fn=feature_engineering_fn,
        device=device,
    )

    context_dim = len(config.context_numeric_columns) + 1 + len(config.context_categorical_columns)
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
    history_path = model_dir / "training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "window_size": int(config.window_size),
        "n_features": int(len(config.feature_columns)),
        "context_dim": int(context_dim),
        "units": int(config.units),
        "latent": int(config.latent),
        "feature_columns": list(config.feature_columns),
        "context_columns": list(config.context_numeric_columns) + ["delta_seconds"] + list(config.context_categorical_columns),
        "streaming_csv": True,
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
        "streaming_source": "chunked_csv",
    }

    torch.save(checkpoint, model_dir / "film_ae.pt")
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(c_scaler, model_dir / "c_scaler.joblib")
    joblib.dump(detector_meta, model_dir / "detector_meta.joblib")
    write_json(model_dir / "detector_meta.json", detector_meta)
    write_json(model_dir / "training_summary.json", {
        "model_dir": str(model_dir.resolve()),
        "history_path": str(history_path.resolve()),
        "model_path": str((model_dir / "film_ae.pt").resolve()),
        "train_config": config.to_dict(),
        "detector_meta": detector_meta,
        "split_totals": split_totals,
        "category_metadata": category_vocabulary.to_metadata(),
        "pipeline": {
            "flow": "raw csv -> chunk read -> preprocess in memory -> sliding windows -> dataloader -> training",
            "intermediate_dataset_files": "none",
            "chunk_boundary_windows": (
                "Each entity keeps only the last window_size - 1 rows in a small deque. "
                "That overlap buffer is reused when the next chunk arrives."
            ),
            "scaling_strategy": (
                "A separate in-memory pass fits online scalers on train windows only, "
                "then later passes reuse those frozen scalers."
            ),
        },
    })
    return {
        "model_dir": str(model_dir.resolve()),
        "history_path": str(history_path.resolve()),
        "detector_meta": detector_meta,
        "split_totals": split_totals,
    }
