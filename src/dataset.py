from __future__ import annotations

import csv
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
import tarfile
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    from config import DatasetConfig
    from utils import choose_device, ensure_directory, write_json
except ImportError:
    from .config import DatasetConfig
    from .utils import choose_device, ensure_directory, write_json


CONTAINER_META_COLUMNS = [
    "container_id",
    "machine_id",
    "time_stamp",
    "app_du",
    "status",
    "cpu_request",
    "cpu_limit",
    "mem_size",
]
CONTAINER_USAGE_COLUMNS = [
    "container_id",
    "machine_id",
    "time_stamp",
    "cpu_util",
    "mem_util",
    "cpi",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io",
]
MACHINE_META_COLUMNS = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
]
MACHINE_USAGE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util",
    "mem_util",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io",
]


def _csv_reader_kwargs(column_names: list[str]) -> dict[str, Any]:
    return {
        "header": None,
        "names": column_names,
        "sep": ",",
        "dtype": "string",
        "engine": "python",
        "na_filter": False,
        "on_bad_lines": "skip",
        "quoting": csv.QUOTE_NONE,
    }


def inspect_archive_schema(
    tar_path: str | Path,
    column_names: list[str],
    sample_rows: int = 5,
) -> dict[str, Any]:
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.next()
        while member is not None and not member.isfile():
            member = tar.next()
        if member is None:
            raise FileNotFoundError(f"No file members found in archive: {tar_path}")
        file_obj = tar.extractfile(member)
        wrapper = TextIOWrapper(file_obj, encoding="utf-8")
        sample = pd.read_csv(
            wrapper,
            nrows=sample_rows,
            **_csv_reader_kwargs(column_names),
        )

    return {
        "archive": str(tar_path),
        "member": member.name,
        "columns": column_names,
        "sample_rows": sample.to_dict(orient="records"),
        "null_fraction": sample.isna().mean().to_dict(),
    }


def inspect_raw_archives(config: DatasetConfig) -> dict[str, Any]:
    return {
        name: inspect_archive_schema(path, columns)
        for (name, path), columns in [
            (("container_meta", config.archive_paths["container_meta"]), CONTAINER_META_COLUMNS),
            (("container_usage", config.archive_paths["container_usage"]), CONTAINER_USAGE_COLUMNS),
            (("machine_meta", config.archive_paths["machine_meta"]), MACHINE_META_COLUMNS),
            (("machine_usage", config.archive_paths["machine_usage"]), MACHINE_USAGE_COLUMNS),
        ]
    }


def iter_tar_csv_chunks(
    tar_path: str | Path,
    column_names: list[str],
    chunksize: int,
    max_rows: int | None = None,
) -> Iterable[pd.DataFrame]:
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.next()
        while member is not None and not member.isfile():
            member = tar.next()
        if member is None:
            raise FileNotFoundError(f"No file members found in archive: {tar_path}")

        file_obj = tar.extractfile(member)
        wrapper = TextIOWrapper(file_obj, encoding="utf-8")
        reader = pd.read_csv(
            wrapper,
            chunksize=chunksize,
            **_csv_reader_kwargs(column_names),
        )

        rows_returned = 0
        for chunk in reader:
            if max_rows is not None and rows_returned >= max_rows:
                break

            if max_rows is not None:
                remaining = max_rows - rows_returned
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            rows_returned += len(chunk)
            yield chunk


def _clean_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def clean_container_meta(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.copy()
    frame["container_id"] = _clean_string(frame["container_id"])
    frame["machine_id"] = _clean_string(frame["machine_id"])
    frame["app_du"] = _clean_string(frame["app_du"]).fillna("unknown")
    frame["status"] = _clean_string(frame["status"]).fillna("unknown")
    frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    for column in ["cpu_request", "cpu_limit", "mem_size"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["container_id", "time_stamp"])
    return frame


def clean_container_usage(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.copy()
    frame["container_id"] = _clean_string(frame["container_id"])
    frame["machine_id"] = _clean_string(frame["machine_id"])
    frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    frame["disk_io"] = pd.to_numeric(frame["disk_io"], errors="coerce").replace([-1, 101], np.nan)
    for column in ["cpu_util", "mem_util", "cpi", "mem_gps", "mpki", "net_in", "net_out"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["container_id", "machine_id", "time_stamp"])
    return frame


def clean_machine_meta(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.copy()
    frame["machine_id"] = _clean_string(frame["machine_id"])
    frame["failure_domain_1"] = _clean_string(frame["failure_domain_1"]).fillna("unknown")
    frame["failure_domain_2"] = _clean_string(frame["failure_domain_2"]).fillna("unknown")
    frame["status"] = _clean_string(frame["status"]).fillna("unknown")
    frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    frame["cpu_num"] = pd.to_numeric(frame["cpu_num"], errors="coerce")
    frame["mem_size"] = pd.to_numeric(frame["mem_size"], errors="coerce")
    frame = frame.dropna(subset=["machine_id", "time_stamp"])
    return frame


def clean_machine_usage(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.copy()
    frame["machine_id"] = _clean_string(frame["machine_id"])
    frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    for column in ["cpu_util", "mem_util", "mem_gps", "mpki", "net_in", "net_out", "disk_io"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["machine_id", "time_stamp"])
    return frame


def load_container_usage_subset(config: DatasetConfig) -> tuple[pd.DataFrame, set[str], set[str]]:
    selected_container_ids: set[str] = set()
    selected_machine_ids: set[str] = set()
    chunks: list[pd.DataFrame] = []

    for chunk in iter_tar_csv_chunks(
        config.archive_paths["container_usage"],
        column_names=CONTAINER_USAGE_COLUMNS,
        chunksize=config.chunksize,
        max_rows=config.max_usage_rows,
    ):
        frame = clean_container_usage(chunk)
        if config.container_limit is not None:
            new_ids = [cid for cid in frame["container_id"].dropna().unique().tolist() if cid not in selected_container_ids]
            capacity = max(0, config.container_limit - len(selected_container_ids))
            if capacity > 0:
                selected_container_ids.update(new_ids[:capacity])
            frame = frame[frame["container_id"].isin(selected_container_ids)]

        if frame.empty:
            continue

        selected_container_ids.update(frame["container_id"].dropna().astype(str).unique().tolist())
        selected_machine_ids.update(frame["machine_id"].dropna().astype(str).unique().tolist())
        chunks.append(frame)

    if not chunks:
        raise RuntimeError("No container usage rows were loaded from the raw archive.")

    usage = pd.concat(chunks, ignore_index=True)
    usage = usage.sort_values(["container_id", "time_stamp"]).reset_index(drop=True)
    return usage, selected_container_ids, selected_machine_ids


def load_filtered_archive(
    tar_path: Path,
    column_names: list[str],
    chunksize: int,
    cleaner: Any,
    filter_column: str,
    allowed_ids: set[str],
    max_rows: int | None = None,
) -> pd.DataFrame:
    if not tar_path.exists() or not allowed_ids:
        return pd.DataFrame(columns=column_names)

    chunks: list[pd.DataFrame] = []
    for chunk in iter_tar_csv_chunks(
        tar_path,
        column_names=column_names,
        chunksize=chunksize,
        max_rows=max_rows,
    ):
        frame = cleaner(chunk)
        frame = frame[frame[filter_column].isin(allowed_ids)]
        if not frame.empty:
            chunks.append(frame)

    if not chunks:
        return pd.DataFrame(columns=column_names)

    return pd.concat(chunks, ignore_index=True)


@dataclass
class ContextEncoder:
    numeric_columns: list[str]
    categorical_columns: list[str]
    categories: dict[str, list[str]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> "ContextEncoder":
        categories = {}
        for column in categorical_columns:
            values = (
                frame[column]
                .fillna("unknown")
                .astype(str)
                .sort_values()
                .drop_duplicates()
                .tolist()
            )
            categories[column] = values
        return cls(
            numeric_columns=list(numeric_columns),
            categorical_columns=list(categorical_columns),
            categories=categories,
        )

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            parts.append(values.reshape(-1, 1))

        for column in self.categorical_columns:
            mapping = {value: index for index, value in enumerate(self.categories.get(column, []))}
            values = (
                frame[column]
                .fillna("unknown")
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .to_numpy(dtype=np.float32)
            )
            parts.append(values.reshape(-1, 1))

        if not parts:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "categories": self.categories,
        }


def _merge_asof_by(left: pd.DataFrame, right: pd.DataFrame, by: str) -> pd.DataFrame:
    if right.empty:
        return left.copy()

    extra_columns = [column for column in right.columns if column not in {by, "time_stamp"}]
    right_groups = {
        group_key: group.sort_values("time_stamp").reset_index(drop=True)
        for group_key, group in right.groupby(by, sort=False)
    }

    merged_parts: list[pd.DataFrame] = []
    for group_key, left_group in left.groupby(by, sort=False):
        left_group = left_group.sort_values("time_stamp").reset_index().rename(columns={"index": "__row_id"})
        right_group = right_groups.get(group_key)

        if right_group is None or right_group.empty:
            for column in extra_columns:
                left_group[column] = np.nan
            merged_parts.append(left_group)
            continue

        merged_group = pd.merge_asof(
            left_group,
            right_group.drop(columns=[by]),
            on="time_stamp",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_parts.append(merged_group)

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values("__row_id").drop(columns="__row_id").reset_index(drop=True)
    return merged


def align_usage_with_context(
    usage: pd.DataFrame,
    container_meta: pd.DataFrame,
    machine_usage: pd.DataFrame,
    machine_meta: pd.DataFrame,
) -> pd.DataFrame:
    container_meta_prepared = container_meta.rename(
        columns={
            "app_du": "container_app_du",
            "status": "container_status",
            "cpu_request": "container_cpu_request",
            "cpu_limit": "container_cpu_limit",
            "mem_size": "container_mem_size",
        }
    )[
        [
            "container_id",
            "time_stamp",
            "container_app_du",
            "container_status",
            "container_cpu_request",
            "container_cpu_limit",
            "container_mem_size",
        ]
    ]

    machine_usage_prepared = machine_usage.rename(
        columns={
            "cpu_util": "machine_cpu_util",
            "mem_util": "machine_mem_util",
            "mem_gps": "machine_mem_gps",
            "mpki": "machine_mpki",
            "net_in": "machine_net_in",
            "net_out": "machine_net_out",
            "disk_io": "machine_disk_io",
        }
    )

    machine_meta_prepared = machine_meta.rename(
        columns={
            "failure_domain_1": "machine_failure_domain_1",
            "failure_domain_2": "machine_failure_domain_2",
            "cpu_num": "machine_cpu_num",
            "mem_size": "machine_mem_size",
            "status": "machine_status",
        }
    )

    merged = _merge_asof_by(usage, container_meta_prepared, by="container_id")
    merged = _merge_asof_by(merged, machine_usage_prepared, by="machine_id")
    merged = _merge_asof_by(merged, machine_meta_prepared, by="machine_id")
    merged = merged.sort_values(["container_id", "time_stamp"]).reset_index(drop=True)

    numeric_columns = [
        "cpu_util",
        "mem_util",
        "cpi",
        "mem_gps",
        "mpki",
        "net_in",
        "net_out",
        "disk_io",
        "container_cpu_request",
        "container_cpu_limit",
        "container_mem_size",
        "machine_cpu_num",
        "machine_mem_size",
        "machine_cpu_util",
        "machine_mem_util",
        "machine_mem_gps",
        "machine_mpki",
        "machine_net_in",
        "machine_net_out",
        "machine_disk_io",
    ]
    categorical_columns = [
        "container_app_du",
        "container_status",
        "machine_failure_domain_1",
        "machine_failure_domain_2",
        "machine_status",
    ]

    for column in numeric_columns:
        if column not in merged.columns:
            merged[column] = np.nan
    for column in categorical_columns:
        if column not in merged.columns:
            merged[column] = "unknown"

    merged[numeric_columns] = (
        merged.groupby("container_id", group_keys=False)[numeric_columns]
        .apply(lambda frame: frame.ffill().bfill())
        .fillna(0.0)
    )

    for column in categorical_columns:
        merged[column] = (
            merged.groupby("container_id")[column]
            .transform(lambda series: series.ffill().bfill())
            .fillna("unknown")
            .astype(str)
        )

    return merged


def _assign_group_splits(group_size: int, train_ratio: float, val_ratio: float) -> list[str]:
    if group_size == 1:
        return ["train"]
    if group_size == 2:
        return ["train", "test"]

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

    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def assign_temporal_splits(metadata: pd.DataFrame, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    split_values = pd.Series(index=metadata.index, dtype="string")
    for _, group in metadata.groupby("container_id", sort=False):
        group_sorted = group.sort_values("end_time")
        labels = _assign_group_splits(len(group_sorted), train_ratio=train_ratio, val_ratio=val_ratio)
        split_values.loc[group_sorted.index] = labels
    metadata = metadata.copy()
    metadata["split"] = split_values.astype(str)
    return metadata


def generate_sliding_windows(
    aligned_df: pd.DataFrame,
    config: DatasetConfig,
    context_encoder: ContextEncoder,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows_x: list[np.ndarray] = []
    windows_c: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    context_matrix = context_encoder.transform_frame(aligned_df[list(config.context_columns)])
    device = torch.device(choose_device("cuda"))
    row_counts = aligned_df.groupby("container_id", sort=False).size()
    total_window_candidates = int(
        sum(
            1 + (int(row_count) - config.window_size) // config.stride
            for row_count in row_counts
            if int(row_count) >= config.window_size
        )
    )

    window_id = 0
    progress = tqdm(
        total=total_window_candidates,
        desc="Sliding windows",
        unit="window",
        dynamic_ncols=True,
    )
    for container_id, frame in aligned_df.groupby("container_id", sort=False):
        frame = frame.sort_values("time_stamp").reset_index(drop=True)
        features = frame[list(config.feature_columns)].copy()
        features = features.ffill().bfill()
        feature_values = features.to_numpy(dtype=np.float32)

        context_values = context_matrix[aligned_df.index[aligned_df["container_id"] == container_id]]
        times = frame["time_stamp"].to_numpy(dtype=np.int64)
        machine_ids = frame["machine_id"].fillna("unknown").astype(str).to_numpy()

        if len(frame) < config.window_size:
            continue

        feature_tensor = torch.as_tensor(feature_values, dtype=torch.float32, device=device)
        feature_tensor = torch.log1p(torch.clamp(feature_tensor, min=0.0))
        window_tensor = feature_tensor.unfold(0, config.window_size, config.stride).permute(0, 2, 1)
        observed_ratio = torch.isfinite(window_tensor).float().mean(dim=(1, 2))
        keep_mask = observed_ratio >= config.min_window_observed_ratio
        progress.update(int(window_tensor.shape[0]))

        if not bool(torch.any(keep_mask).item()):
            continue

        kept_windows = torch.nan_to_num(
            window_tensor[keep_mask],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).contiguous().cpu().numpy()
        keep_indices = keep_mask.cpu().numpy().astype(bool)
        end_indices = np.arange(
            config.window_size - 1,
            len(frame),
            config.stride,
            dtype=np.int64,
        )[keep_indices]

        for window, end_index in zip(kept_windows, end_indices):
            windows_x.append(window.astype(np.float32, copy=False))
            windows_c.append(context_values[end_index].astype(np.float32))

            start_index = int(end_index - config.window_size + 1)
            row = frame.iloc[int(end_index)]
            records.append(
                {
                    "window_id": window_id,
                    "entity_id": str(container_id),
                    "container_id": str(container_id),
                    "machine_id": str(machine_ids[int(end_index)]),
                    "start_index": start_index,
                    "end_index": int(end_index),
                    "start_time": int(times[start_index]),
                    "end_time": int(times[int(end_index)]),
                    "container_app_du": str(row.get("container_app_du", "unknown")),
                    "container_status": str(row.get("container_status", "unknown")),
                    "machine_status": str(row.get("machine_status", "unknown")),
                    "machine_failure_domain_1": str(row.get("machine_failure_domain_1", "unknown")),
                    "machine_failure_domain_2": str(row.get("machine_failure_domain_2", "unknown")),
                }
            )
            window_id += 1
    progress.close()

    if not windows_x:
        raise RuntimeError("No sliding windows were produced from the aligned dataset.")

    metadata = pd.DataFrame(records)
    metadata = assign_temporal_splits(
        metadata,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    return (
        np.stack(windows_x).astype(np.float32),
        np.stack(windows_c).astype(np.float32),
        metadata,
    )


def _train_test_window_counts(num_windows: int, train_split: float) -> tuple[int, int]:
    if num_windows <= 0:
        return 0, 0
    if num_windows == 1:
        return 1, 0

    train_count = int(np.floor(num_windows * train_split))
    train_count = max(1, min(train_count, num_windows - 1))
    test_count = num_windows - train_count
    return train_count, test_count


def build_train_test_split_plan(
    config: DatasetConfig,
    row_counts: dict[str, int],
) -> dict[str, EntitySplitBoundaries]:
    try:
        from streaming_train import EntitySplitBoundaries
    except ImportError:
        from .streaming_train import EntitySplitBoundaries

    split_plan: dict[str, EntitySplitBoundaries] = {}
    for entity_id, row_count in row_counts.items():
        num_windows = 0
        if row_count >= config.window_size:
            num_windows = 1 + (row_count - config.window_size) // config.stride
        train_count, test_count = _train_test_window_counts(num_windows, config.train_ratio)
        split_plan[entity_id] = EntitySplitBoundaries(
            train_windows=train_count,
            val_windows=0,
            test_windows=test_count,
        )
    return split_plan


def _streaming_config_from_dataset_config(config: DatasetConfig) -> StreamingTrainConfig:
    try:
        from streaming_train import StreamingTrainConfig
    except ImportError:
        from .streaming_train import StreamingTrainConfig

    return StreamingTrainConfig(
        raw_data_dir=config.raw_data_dir,
        model_dir=config.output_dir,
        container_meta_name=config.container_meta_name,
        container_usage_name=config.container_usage_name,
        machine_meta_name=config.machine_meta_name,
        machine_usage_name=config.machine_usage_name,
        chunksize=config.chunksize,
        max_usage_rows=config.max_usage_rows,
        max_container_meta_rows=config.max_container_meta_rows,
        max_machine_usage_rows=config.max_machine_usage_rows,
        max_machine_meta_rows=config.max_machine_meta_rows,
        container_limit=config.container_limit,
        window_size=config.window_size,
        stride=config.stride,
        train_ratio=config.train_ratio,
        val_ratio=0.0,
        min_window_observed_ratio=config.min_window_observed_ratio,
        feature_columns=config.feature_columns,
        container_context_numeric=config.container_context_numeric,
        container_context_categorical=config.container_context_categorical,
        machine_context_numeric=config.machine_context_numeric,
        machine_context_categorical=config.machine_context_categorical,
    )


def _create_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    ensure_directory(path.parent)
    if path.exists():
        try:
            path.unlink()
        except PermissionError as exc:
            raise OSError(
                f"Cannot overwrite existing dataset artifact at '{path}'. "
                "On Windows this usually means the .npy file is still open in the current "
                "Jupyter kernel or another Python process. Restart the kernel or write to a "
                "new output directory before rebuilding the dataset."
            ) from exc

    try:
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    except OSError as exc:
        raise OSError(
            f"Failed to create memmap '{path}' with shape={shape}. "
            "If this output directory was used earlier in the notebook, restart the kernel "
            "or switch to a fresh dataset output directory before rebuilding."
        ) from exc


def _write_split_memmaps(
    config: DatasetConfig,
    split_plan: dict[str, Any],
    vocabulary: Any,
    allowed_entities: set[str] | None,
) -> dict[str, Any]:
    try:
        from streaming_train import ChunkedAlibabaRawLoader, InMemoryStreamingPreprocessor, SlidingWindowStream
    except ImportError:
        from .streaming_train import ChunkedAlibabaRawLoader, InMemoryStreamingPreprocessor, SlidingWindowStream

    streaming_config = _streaming_config_from_dataset_config(config)
    loader = ChunkedAlibabaRawLoader(streaming_config, allowed_entities=allowed_entities)
    preprocessor = InMemoryStreamingPreprocessor(streaming_config, vocabulary=vocabulary)
    window_stream = SlidingWindowStream(streaming_config, split_plan=split_plan)

    num_features = len(config.feature_columns)
    context_dim = len(config.container_context_numeric + config.machine_context_numeric)
    context_dim += len(config.container_context_categorical + config.machine_context_categorical)
    train_windows = int(sum(boundaries.train_windows for boundaries in split_plan.values()))
    test_windows = int(sum(boundaries.test_windows for boundaries in split_plan.values()))

    x_train_path = config.output_dir / "X_train.npy"
    x_test_path = config.output_dir / "X_test.npy"
    c_train_path = config.output_dir / "C_train.npy"
    c_test_path = config.output_dir / "C_test.npy"
    train_metadata_path = config.output_dir / "window_metadata_train.csv"
    test_metadata_path = config.output_dir / "window_metadata_test.csv"

    x_train = _create_memmap(x_train_path, (train_windows, config.window_size, num_features))
    x_test = _create_memmap(x_test_path, (test_windows, config.window_size, num_features))
    c_train = _create_memmap(c_train_path, (train_windows, context_dim))
    c_test = _create_memmap(c_test_path, (test_windows, context_dim))

    train_index = 0
    test_index = 0
    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    total_windows = int(sum(boundaries.total_windows for boundaries in split_plan.values()))
    progress = tqdm(
        total=total_windows,
        desc="Sliding windows",
        unit="window",
        dynamic_ncols=True,
    )
    for aligned_row in loader.iter_aligned_rows():
        processed = preprocessor.process(aligned_row)
        windows_seen_before = window_stream.total_windows_seen
        window_sample = window_stream.update(processed)
        progress.update(window_stream.total_windows_seen - windows_seen_before)
        if window_sample is None:
            continue

        if window_sample.split == "train":
            x_train[train_index] = window_sample.window
            c_train[train_index] = window_sample.context
            train_records.append(
                {
                    "window_id": train_index,
                    "entity_id": str(window_sample.entity_id),
                    "container_id": str(window_sample.entity_id),
                    "machine_id": str(window_sample.machine_id),
                    "end_time": int(window_sample.time_stamp),
                    "split": "train",
                }
            )
            train_index += 1
        elif window_sample.split == "test":
            x_test[test_index] = window_sample.window
            c_test[test_index] = window_sample.context
            test_records.append(
                {
                    "window_id": test_index,
                    "entity_id": str(window_sample.entity_id),
                    "container_id": str(window_sample.entity_id),
                    "machine_id": str(window_sample.machine_id),
                    "end_time": int(window_sample.time_stamp),
                    "split": "test",
                }
            )
            test_index += 1
    progress.close()

    x_train.flush()
    x_test.flush()
    c_train.flush()
    c_test.flush()
    pd.DataFrame(train_records).to_csv(train_metadata_path, index=False)
    pd.DataFrame(test_records).to_csv(test_metadata_path, index=False)

    return {
        "x_train_path": x_train_path,
        "x_test_path": x_test_path,
        "c_train_path": c_train_path,
        "c_test_path": c_test_path,
        "train_metadata_path": train_metadata_path,
        "test_metadata_path": test_metadata_path,
        "x_train_shape": tuple(x_train.shape),
        "x_test_shape": tuple(x_test.shape),
        "c_train_shape": tuple(c_train.shape),
        "c_test_shape": tuple(c_test.shape),
    }


def _write_forecasting_split_artifacts(
    output_dir: Path,
    split_name: str,
    full_windows: np.ndarray,
    forecast_horizon: int,
) -> dict[str, Any]:
    if forecast_horizon <= 0 or forecast_horizon >= int(full_windows.shape[1]):
        raise ValueError(
            "forecast_horizon must be >= 1 and smaller than the saved window_size "
            f"(got forecast_horizon={forecast_horizon}, window_size={int(full_windows.shape[1])})."
        )

    x_forecast_path = output_dir / f"X_forecast_{split_name}.npy"
    y_forecast_path = output_dir / f"y_forecast_{split_name}.npy"
    input_shape = (
        int(full_windows.shape[0]),
        int(full_windows.shape[1]) - int(forecast_horizon),
        int(full_windows.shape[2]),
    )
    target_shape = (
        int(full_windows.shape[0]),
        int(forecast_horizon),
        int(full_windows.shape[2]),
    )

    x_forecast = _create_memmap(x_forecast_path, input_shape)
    y_forecast = _create_memmap(y_forecast_path, target_shape)
    x_forecast[:] = np.asarray(full_windows[:, :-forecast_horizon, :], dtype=np.float32)
    y_forecast[:] = np.asarray(full_windows[:, -forecast_horizon:, :], dtype=np.float32)
    x_forecast.flush()
    y_forecast.flush()

    return {
        "x_forecast_path": x_forecast_path,
        "y_forecast_path": y_forecast_path,
        "x_forecast_shape": tuple(x_forecast.shape),
        "y_forecast_shape": tuple(y_forecast.shape),
    }


def build_research_dataset(config: DatasetConfig | None = None) -> dict[str, Any]:
    config = config or DatasetConfig()
    ensure_directory(config.output_dir)
    try:
        from streaming_train import build_streaming_statistics
    except ImportError:
        from .streaming_train import build_streaming_statistics

    raw_schema = inspect_raw_archives(config)
    streaming_config = _streaming_config_from_dataset_config(config)
    allowed_entities, row_counts, vocabulary = build_streaming_statistics(streaming_config)
    split_plan = build_train_test_split_plan(config, row_counts)
    split_artifacts = _write_split_memmaps(
        config=config,
        split_plan=split_plan,
        vocabulary=vocabulary,
        allowed_entities=allowed_entities,
    )
    forecasting_train = _write_forecasting_split_artifacts(
        output_dir=config.output_dir,
        split_name="train",
        full_windows=np.load(split_artifacts["x_train_path"], mmap_mode="r"),
        forecast_horizon=config.forecast_horizon,
    )
    forecasting_test = _write_forecasting_split_artifacts(
        output_dir=config.output_dir,
        split_name="test",
        full_windows=np.load(split_artifacts["x_test_path"], mmap_mode="r"),
        forecast_horizon=config.forecast_horizon,
    )

    feature_meta_path = config.output_dir / "feature_meta.joblib"
    dataset_meta_path = config.output_dir / "dataset_meta.json"
    summary_path = config.output_dir / "dataset_build_summary.json"
    schema_path = config.output_dir / "raw_schema.json"
    encoder_path = config.output_dir / "context_encoder.joblib"

    joblib.dump(vocabulary.to_metadata(), encoder_path)

    feature_meta = {
        "feature_columns": list(config.feature_columns),
        "context_columns": list(config.context_columns),
        "entity_columns": ["container_id"],
        "timestamp_column": "time_stamp",
        "preprocessing": "raw Alibaba archives -> chunked read -> streaming asof joins -> fill -> clip(min=0) -> log1p",
        "window_size": int(config.window_size),
        "stride": int(config.stride),
        "forecast_horizon": int(config.forecast_horizon),
        "forecast_input_window_size": int(config.window_size - config.forecast_horizon),
        "context_encoder": vocabulary.to_metadata(),
        "train_split": float(config.train_ratio),
    }
    joblib.dump(feature_meta, feature_meta_path)

    dataset_meta = {
        "source": "alibaba_raw_archives",
        "train_windows": int(split_artifacts["x_train_shape"][0]),
        "test_windows": int(split_artifacts["x_test_shape"][0]),
        "window_size": int(config.window_size),
        "forecast_horizon": int(config.forecast_horizon),
        "forecast_input_window_size": int(config.window_size - config.forecast_horizon),
        "num_features": int(len(config.feature_columns)),
        "context_dim": int(len(config.context_columns)),
        "feature_columns": list(config.feature_columns),
        "context_columns": list(config.context_columns),
        "split_counts": {
            "train": int(split_artifacts["x_train_shape"][0]),
            "test": int(split_artifacts["x_test_shape"][0]),
        },
        "forecast_shapes": {
            "X_forecast_train": forecasting_train["x_forecast_shape"],
            "y_forecast_train": forecasting_train["y_forecast_shape"],
            "X_forecast_test": forecasting_test["x_forecast_shape"],
            "y_forecast_test": forecasting_test["y_forecast_shape"],
        },
    }
    joblib.dump(dataset_meta, config.output_dir / "dataset_meta.joblib")
    write_json(dataset_meta_path, dataset_meta)
    write_json(schema_path, raw_schema)

    summary = {
        "X_train": str(split_artifacts["x_train_path"].resolve()),
        "X_test": str(split_artifacts["x_test_path"].resolve()),
        "C_train": str(split_artifacts["c_train_path"].resolve()),
        "C_test": str(split_artifacts["c_test_path"].resolve()),
        "X_train_shape": split_artifacts["x_train_shape"],
        "X_test_shape": split_artifacts["x_test_shape"],
        "X_forecast_train": str(forecasting_train["x_forecast_path"].resolve()),
        "y_forecast_train": str(forecasting_train["y_forecast_path"].resolve()),
        "X_forecast_test": str(forecasting_test["x_forecast_path"].resolve()),
        "y_forecast_test": str(forecasting_test["y_forecast_path"].resolve()),
        "X_forecast_train_shape": forecasting_train["x_forecast_shape"],
        "y_forecast_train_shape": forecasting_train["y_forecast_shape"],
        "X_forecast_test_shape": forecasting_test["x_forecast_shape"],
        "y_forecast_test_shape": forecasting_test["y_forecast_shape"],
        "train_metadata": str(split_artifacts["train_metadata_path"].resolve()),
        "test_metadata": str(split_artifacts["test_metadata_path"].resolve()),
        "feature_meta": str(feature_meta_path.resolve()),
        "dataset_meta": str(dataset_meta_path.resolve()),
        "context_encoder": str(encoder_path.resolve()),
        "config": config.to_dict(),
    }
    write_json(summary_path, summary)

    print(f"Final dataset shape | X_train: {split_artifacts['x_train_shape']}")
    print(f"Final dataset shape | X_test: {split_artifacts['x_test_shape']}")
    return summary


def load_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    x_train_path = dataset_path / "X_train.npy"
    x_test_path = dataset_path / "X_test.npy"
    c_train_path = dataset_path / "C_train.npy"
    c_test_path = dataset_path / "C_test.npy"
    x_forecast_train_path = dataset_path / "X_forecast_train.npy"
    x_forecast_test_path = dataset_path / "X_forecast_test.npy"
    y_forecast_train_path = dataset_path / "y_forecast_train.npy"
    y_forecast_test_path = dataset_path / "y_forecast_test.npy"
    if x_train_path.exists() and x_test_path.exists():
        bundle = {
            "X_train": np.load(x_train_path, mmap_mode="r"),
            "X_test": np.load(x_test_path, mmap_mode="r"),
            "C_train": np.load(c_train_path, mmap_mode="r"),
            "C_test": np.load(c_test_path, mmap_mode="r"),
            "feature_meta": joblib.load(dataset_path / "feature_meta.joblib"),
            "dataset_meta": joblib.load(dataset_path / "dataset_meta.joblib"),
        }
        if (
            x_forecast_train_path.exists()
            and x_forecast_test_path.exists()
            and y_forecast_train_path.exists()
            and y_forecast_test_path.exists()
        ):
            bundle.update(
                {
                    "X_forecast_train": np.load(x_forecast_train_path, mmap_mode="r"),
                    "X_forecast_test": np.load(x_forecast_test_path, mmap_mode="r"),
                    "y_forecast_train": np.load(y_forecast_train_path, mmap_mode="r"),
                    "y_forecast_test": np.load(y_forecast_test_path, mmap_mode="r"),
                }
            )
        return bundle
    return {
        "X": np.load(dataset_path / "X_all.npy", allow_pickle=False),
        "C": np.load(dataset_path / "C_all.npy", allow_pickle=False),
        "metadata": pd.read_csv(dataset_path / "window_metadata.csv"),
        "feature_meta": joblib.load(dataset_path / "feature_meta.joblib"),
        "dataset_meta": joblib.load(dataset_path / "dataset_meta.joblib"),
    }


build_dataset = build_research_dataset
build_windows = generate_sliding_windows
