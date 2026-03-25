from __future__ import annotations

import tarfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .config import DatasetConfig, PathConfig
from .utils import dataframe_to_records, load_json, save_json


CONTAINER_META_COLS = ["container_id", "machine_id", "time_stamp", "app_du", "status", "cpu_request", "cpu_limit", "mem_size"]
CONTAINER_USAGE_COLS = [
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
MACHINE_META_COLS = ["machine_id", "time_stamp", "failure_domain_1", "failure_domain_2", "cpu_num", "mem_size", "status"]
MACHINE_USAGE_COLS = [
    "machine_id",
    "time_stamp",
    "machine_cpu_util",
    "machine_mem_util",
    "machine_mem_gps",
    "machine_mpki",
    "machine_net_in",
    "machine_net_out",
    "machine_disk_io",
]


@dataclass
class DatasetBundle:
    X: np.ndarray
    C: np.ndarray
    metadata: pd.DataFrame
    feature_meta: dict[str, Any]
    dataset_meta: dict[str, Any]
    context_encoder: Any


def iter_csv_from_tar(
    tar_path: str | Path,
    usecols: list[int] | None = None,
    dtype: dict[int, str] | None = None,
    chunksize: int | None = None,
    max_files: int | None = None,
) -> Iterator[pd.DataFrame]:
    tar_path = Path(tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing archive: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        file_count = 0
        for member in tar:
            if not member.isfile():
                continue
            if max_files is not None and file_count >= max_files:
                break
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            file_count += 1
            with TextIOWrapper(extracted, encoding="utf-8") as wrapper:
                reader = pd.read_csv(
                    wrapper,
                    header=None,
                    usecols=usecols,
                    dtype=dtype,
                    chunksize=chunksize,
                    engine="c",
                    low_memory=False,
                )
                if chunksize is None:
                    yield reader
                else:
                    for chunk in reader:
                        yield chunk
        if file_count == 0:
            raise RuntimeError(f"No CSV members found inside {tar_path}")


def _load_archive_frame(
    tar_path: str | Path,
    columns: list[str],
    chunksize: int | None,
    max_rows: int | None,
    max_files: int | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    rows_loaded = 0

    dtype = {idx: "string" if idx in (0, 1, 3, 4) else "float64" for idx in range(len(columns))}
    if columns == MACHINE_META_COLS:
        dtype = {0: "string", 1: "float64", 2: "string", 3: "string", 4: "float64", 5: "float64", 6: "string"}
    elif columns == MACHINE_USAGE_COLS:
        dtype = {0: "string", 1: "float64", 2: "float64", 3: "float64", 4: "float64", 5: "float64", 6: "float64", 7: "float64", 8: "float64"}
    elif columns == CONTAINER_META_COLS:
        dtype = {0: "string", 1: "string", 2: "float64", 3: "string", 4: "string", 5: "float64", 6: "float64", 7: "float64"}
    elif columns == CONTAINER_USAGE_COLS:
        dtype = {0: "string", 1: "string", 2: "float64", 3: "float64", 4: "float64", 5: "float64", 6: "float64", 7: "float64", 8: "float64", 9: "float64", 10: "float64"}

    for chunk in iter_csv_from_tar(
        tar_path,
        usecols=list(range(len(columns))),
        dtype=dtype,
        chunksize=chunksize,
        max_files=max_files,
    ):
        chunk = chunk.iloc[:, : len(columns)].copy()
        chunk.columns = columns
        if max_rows is not None and rows_loaded + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_loaded].copy()
        parts.append(chunk)
        rows_loaded += len(chunk)
        if max_rows is not None and rows_loaded >= max_rows:
            break

    if not parts:
        raise RuntimeError(f"No rows loaded from archive: {tar_path}")
    return pd.concat(parts, ignore_index=True)


def _clean_frame(df: pd.DataFrame, string_cols: list[str], numeric_cols: list[str], timestamp_column: str) -> pd.DataFrame:
    result = df.copy()
    for column in string_cols:
        result[column] = result[column].astype("string")
    for column in numeric_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result[timestamp_column] = pd.to_numeric(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column, *[col for col in string_cols if col.endswith("_id") or col == "container_id" or col == "machine_id"]])
    result = result.sort_values(timestamp_column).reset_index(drop=True)
    return result


def load_alibaba_raw_frames(paths: PathConfig, dataset_config: DatasetConfig) -> dict[str, pd.DataFrame]:
    container_meta = _load_archive_frame(
        paths.raw_container_meta_tar,
        CONTAINER_META_COLS,
        chunksize=dataset_config.chunksize,
        max_rows=dataset_config.max_meta_rows,
        max_files=dataset_config.max_container_meta_files,
    )
    container_usage = _load_archive_frame(
        paths.raw_container_usage_tar,
        CONTAINER_USAGE_COLS,
        chunksize=dataset_config.chunksize,
        max_rows=dataset_config.max_usage_rows,
        max_files=dataset_config.max_container_usage_files,
    )
    machine_meta = _load_archive_frame(
        paths.raw_machine_meta_tar,
        MACHINE_META_COLS,
        chunksize=dataset_config.chunksize,
        max_rows=dataset_config.max_machine_meta_rows,
        max_files=dataset_config.max_machine_meta_files,
    )
    machine_usage = _load_archive_frame(
        paths.raw_machine_usage_tar,
        MACHINE_USAGE_COLS,
        chunksize=dataset_config.chunksize,
        max_rows=dataset_config.max_machine_usage_rows,
        max_files=dataset_config.max_machine_usage_files,
    )

    container_meta = _clean_frame(
        container_meta,
        string_cols=["container_id", "machine_id", "app_du", "status"],
        numeric_cols=["cpu_request", "cpu_limit", "mem_size"],
        timestamp_column="time_stamp",
    )
    container_usage = _clean_frame(
        container_usage,
        string_cols=["container_id", "machine_id"],
        numeric_cols=["cpu_util", "mem_util", "cpi", "mem_gps", "mpki", "net_in", "net_out", "disk_io"],
        timestamp_column="time_stamp",
    )
    machine_meta = _clean_frame(
        machine_meta,
        string_cols=["machine_id", "failure_domain_1", "failure_domain_2", "status"],
        numeric_cols=["cpu_num", "mem_size"],
        timestamp_column="time_stamp",
    )
    machine_usage = _clean_frame(
        machine_usage,
        string_cols=["machine_id"],
        numeric_cols=["machine_cpu_util", "machine_mem_util", "machine_mem_gps", "machine_mpki", "machine_net_in", "machine_net_out", "machine_disk_io"],
        timestamp_column="time_stamp",
    )

    if dataset_config.max_containers:
        keep_ids = container_usage["container_id"].dropna().astype("string").unique()[: dataset_config.max_containers]
        container_usage = container_usage[container_usage["container_id"].isin(keep_ids)].copy()
        container_meta = container_meta[container_meta["container_id"].isin(keep_ids)].copy()
        machine_ids = container_usage["machine_id"].dropna().astype("string").unique()
        machine_meta = machine_meta[machine_meta["machine_id"].isin(machine_ids)].copy()
        machine_usage = machine_usage[machine_usage["machine_id"].isin(machine_ids)].copy()

    return {
        "container_meta": container_meta.reset_index(drop=True),
        "container_usage": container_usage.reset_index(drop=True),
        "machine_meta": machine_meta.reset_index(drop=True),
        "machine_usage": machine_usage.reset_index(drop=True),
    }


def _merge_asof_by(left: pd.DataFrame, right: pd.DataFrame, by: str, left_ts: str, right_ts: str) -> pd.DataFrame:
    left_sorted = left.sort_values([by, left_ts]).reset_index(drop=True)
    right_sorted = right.sort_values([by, right_ts]).reset_index(drop=True)
    merged_parts: list[pd.DataFrame] = []

    right_groups = {key: group for key, group in right_sorted.groupby(by, sort=False)}
    for key, left_group in left_sorted.groupby(by, sort=False):
        right_group = right_groups.get(key)
        if right_group is None or right_group.empty:
            merged_parts.append(left_group.copy())
            continue
        right_group = right_group.drop(columns=[by], errors="ignore")
        merged_parts.append(
            pd.merge_asof(
                left_group.sort_values(left_ts),
                right_group.sort_values(right_ts),
                left_on=left_ts,
                right_on=right_ts,
                direction="backward",
            )
        )

    return pd.concat(merged_parts, ignore_index=True) if merged_parts else left_sorted


def join_alibaba_frames(frames: dict[str, pd.DataFrame], dataset_config: DatasetConfig) -> pd.DataFrame:
    usage = frames["container_usage"].copy()

    container_meta = frames["container_meta"].rename(
        columns={
            "machine_id": "meta_machine_id",
            "time_stamp": "container_meta_time_stamp",
            "app_du": "container_app_du",
            "status": "container_status",
            "cpu_request": "container_cpu_request",
            "cpu_limit": "container_cpu_limit",
            "mem_size": "container_mem_size",
        }
    )
    usage = _merge_asof_by(usage, container_meta, by="container_id", left_ts="time_stamp", right_ts="container_meta_time_stamp")

    if dataset_config.use_machine_meta_context:
        machine_meta = frames["machine_meta"].rename(
            columns={
                "time_stamp": "machine_meta_time_stamp",
                "failure_domain_1": "machine_failure_domain_1",
                "failure_domain_2": "machine_failure_domain_2",
                "cpu_num": "machine_cpu_num",
                "mem_size": "machine_mem_size",
                "status": "machine_status",
            }
        )
        usage = _merge_asof_by(usage, machine_meta, by="machine_id", left_ts="time_stamp", right_ts="machine_meta_time_stamp")

    if dataset_config.use_machine_usage_context:
        machine_usage = frames["machine_usage"].rename(columns={"time_stamp": "machine_usage_time_stamp"})
        usage = _merge_asof_by(usage, machine_usage, by="machine_id", left_ts="time_stamp", right_ts="machine_usage_time_stamp")

    return usage.reset_index(drop=True)


def preprocess_joined_alibaba_frame(df: pd.DataFrame, dataset_config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()

    result[dataset_config.feature_columns] = result.groupby(dataset_config.entity_columns)[dataset_config.feature_columns].transform(
        lambda group: group.ffill().bfill()
    )
    result[dataset_config.feature_columns] = result[dataset_config.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    result[dataset_config.feature_columns] = result[dataset_config.feature_columns].clip(lower=dataset_config.clip_min_value)

    if dataset_config.log1p_transform:
        result[dataset_config.feature_columns] = np.log1p(result[dataset_config.feature_columns].to_numpy(dtype=np.float32))

    numeric_context_columns = list(dataset_config.context_numeric_columns)
    categorical_context_columns = list(dataset_config.context_categorical_columns)

    for column in numeric_context_columns:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).astype(np.float32)

    for column in categorical_context_columns:
        if column not in result.columns:
            result[column] = "unknown"
        result[column] = result[column].astype("string").fillna("unknown")

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    encoded_categorical = encoder.fit_transform(result[categorical_context_columns].astype(str))
    encoded_df = pd.DataFrame(encoded_categorical, columns=categorical_context_columns, index=result.index)

    result[categorical_context_columns] = encoded_df
    result[dataset_config.context_columns] = result[numeric_context_columns + categorical_context_columns].astype(np.float32)

    context_encoder = {
        "categorical_columns": categorical_context_columns,
        "numeric_columns": numeric_context_columns,
        "categories": {
            column: [str(value) for value in categories]
            for column, categories in zip(categorical_context_columns, encoder.categories_)
        },
    }
    return result, context_encoder


def generate_alibaba_windows(df: pd.DataFrame, dataset_config: DatasetConfig, context_encoder: dict[str, Any]) -> DatasetBundle:
    window_size = int(dataset_config.window_size)
    stride = int(dataset_config.stride)

    windows: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    window_id = 0

    for container_id, group in df.groupby("container_id", sort=False):
        group = group.sort_values(dataset_config.timestamp_column).reset_index(drop=True)
        if len(group) < max(dataset_config.min_points_per_entity, window_size):
            continue

        feature_matrix = group[dataset_config.feature_columns].to_numpy(dtype=np.float32, copy=True)
        context_matrix = group[dataset_config.context_columns].to_numpy(dtype=np.float32, copy=True)
        timestamps = group[dataset_config.timestamp_column].to_numpy()

        for start_idx in range(0, len(group) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_df = group.iloc[start_idx:end_idx]
            window = feature_matrix[start_idx:end_idx]
            if np.isnan(window).mean() > 0.2:
                continue
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

            windows.append(window)
            contexts.append(context_matrix[end_idx - 1])
            rows.append(
                {
                    "window_id": window_id,
                    "entity_id": str(container_id),
                    "container_id": str(container_id),
                    "machine_id": str(window_df.iloc[-1].get("machine_id", "")),
                    "start_index": int(start_idx),
                    "end_index": int(end_idx - 1),
                    "start_time": int(timestamps[start_idx]),
                    "end_time": int(timestamps[end_idx - 1]),
                    "app_du": str(window_df.iloc[-1].get("container_app_du", "")),
                    "container_status": str(window_df.iloc[-1].get("container_status", "")),
                    "machine_status": str(window_df.iloc[-1].get("machine_status", "")),
                }
            )
            window_id += 1

    if not windows:
        raise RuntimeError("No sliding windows were created from the Alibaba raw archives.")

    metadata = pd.DataFrame(rows).sort_values(["end_time", "entity_id", "window_id"]).reset_index(drop=True)
    X = np.asarray(windows, dtype=np.float32)
    C = np.asarray(contexts, dtype=np.float32)
    feature_meta = {
        "feature_columns": list(dataset_config.feature_columns),
        "context_columns": list(dataset_config.context_columns),
        "entity_columns": list(dataset_config.entity_columns),
        "timestamp_column": dataset_config.timestamp_column,
        "preprocessing": "raw Alibaba archives -> asof joins -> groupwise fill -> clip(min=0) -> log1p",
        "window_size": dataset_config.window_size,
        "stride": dataset_config.stride,
        "context_encoder": context_encoder,
        "sample_windows": dataframe_to_records(metadata, limit=5),
    }
    dataset_meta = {
        "source": dataset_config.source,
        "num_windows": int(len(X)),
        "window_size": int(dataset_config.window_size),
        "num_features": int(X.shape[-1]),
        "context_dim": int(C.shape[-1]),
        "feature_columns": list(dataset_config.feature_columns),
        "context_columns": list(dataset_config.context_columns),
    }
    return DatasetBundle(X=X, C=C, metadata=metadata, feature_meta=feature_meta, dataset_meta=dataset_meta, context_encoder=context_encoder)


def assign_splits(metadata: pd.DataFrame, dataset_config: DatasetConfig) -> pd.DataFrame:
    if not np.isclose(dataset_config.train_fraction + dataset_config.val_fraction + dataset_config.test_fraction, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    result = metadata.copy()
    total = len(result)
    train_end = int(total * dataset_config.train_fraction)
    val_end = train_end + int(total * dataset_config.val_fraction)

    result["split"] = "test"
    result.loc[: train_end - 1, "split"] = "train"
    result.loc[train_end: val_end - 1, "split"] = "val"
    return result


def save_dataset_bundle(bundle: DatasetBundle, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / "X_all.npy"
    c_path = output_dir / "C_all.npy"
    metadata_path = output_dir / "window_metadata.csv"
    feature_meta_path = output_dir / "feature_meta.joblib"
    dataset_meta_path = output_dir / "dataset_meta.json"
    context_encoder_path = output_dir / "context_encoder.joblib"
    build_summary_path = output_dir / "dataset_build_summary.json"

    np.save(x_path, bundle.X)
    np.save(c_path, bundle.C)
    bundle.metadata.to_csv(metadata_path, index=False)
    joblib.dump(bundle.feature_meta, feature_meta_path)
    joblib.dump(bundle.context_encoder, context_encoder_path)
    save_json(dataset_meta_path, bundle.dataset_meta)
    save_json(
        build_summary_path,
        {
            "X": str(x_path.resolve()),
            "C": str(c_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "feature_meta": str(feature_meta_path.resolve()),
            "dataset_meta": str(dataset_meta_path.resolve()),
            "context_encoder": str(context_encoder_path.resolve()),
        },
    )

    return {
        "X": str(x_path),
        "C": str(c_path),
        "metadata": str(metadata_path),
        "feature_meta": str(feature_meta_path),
        "dataset_meta": str(dataset_meta_path),
        "context_encoder": str(context_encoder_path),
        "build_summary": str(build_summary_path),
    }


def build_dataset_from_raw_archives(
    paths: PathConfig,
    dataset_config: DatasetConfig,
    output_dir: str | Path,
) -> tuple[DatasetBundle, dict[str, str]]:
    frames = load_alibaba_raw_frames(paths, dataset_config)
    joined = join_alibaba_frames(frames, dataset_config)
    preprocessed, context_encoder = preprocess_joined_alibaba_frame(joined, dataset_config)
    bundle = generate_alibaba_windows(preprocessed, dataset_config, context_encoder)
    bundle.metadata = assign_splits(bundle.metadata, dataset_config)
    paths_out = save_dataset_bundle(bundle, output_dir)
    return bundle, paths_out


def load_metrics_csv(csv_path: str | Path, dataset_config: DatasetConfig) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    required_columns = [dataset_config.timestamp_column, *dataset_config.feature_columns, *dataset_config.context_columns]
    df = pd.read_csv(csv_path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df[required_columns].copy()
    df[dataset_config.timestamp_column] = pd.to_datetime(df[dataset_config.timestamp_column], utc=True, errors="coerce")
    df = df.dropna(subset=[dataset_config.timestamp_column]).sort_values(dataset_config.context_columns + [dataset_config.timestamp_column])
    return df.reset_index(drop=True)


def preprocess_metrics_frame(df: pd.DataFrame, dataset_config: DatasetConfig) -> pd.DataFrame:
    result = df.copy()
    feature_columns = dataset_config.feature_columns

    result[feature_columns] = result[feature_columns].apply(pd.to_numeric, errors="coerce")
    result[feature_columns] = result.groupby(dataset_config.entity_columns)[feature_columns].transform(
        lambda group: group.ffill().bfill().fillna(0.0)
    )
    result[feature_columns] = result[feature_columns].clip(lower=dataset_config.clip_min_value)

    if dataset_config.log1p_transform:
        result[feature_columns] = np.log1p(result[feature_columns].to_numpy(dtype=np.float32))

    return result


def fit_context_encoder(df: pd.DataFrame, dataset_config: DatasetConfig) -> OrdinalEncoder:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    encoder.fit(df[dataset_config.context_columns].astype(str))
    return encoder


def generate_sliding_windows(df: pd.DataFrame, dataset_config: DatasetConfig, context_encoder: OrdinalEncoder) -> DatasetBundle:
    window_size = int(dataset_config.window_size)
    stride = int(dataset_config.stride)
    feature_columns = dataset_config.feature_columns
    context_columns = dataset_config.context_columns
    timestamp_column = dataset_config.timestamp_column
    entity_columns = dataset_config.entity_columns

    windows: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    window_id = 0

    for entity_key, entity_df in df.groupby(entity_columns, sort=False):
        entity_df = entity_df.sort_values(timestamp_column).reset_index(drop=True)
        if len(entity_df) < max(dataset_config.min_points_per_entity, window_size):
            continue

        entity_id = "::".join(map(str, entity_key if isinstance(entity_key, tuple) else [entity_key]))
        feature_matrix = entity_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        encoded_context = context_encoder.transform(entity_df[context_columns].astype(str))

        for start_idx in range(0, len(entity_df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_df = entity_df.iloc[start_idx:end_idx]
            windows.append(feature_matrix[start_idx:end_idx])
            contexts.append(encoded_context[end_idx - 1].astype(np.float32))
            rows.append(
                {
                    "window_id": window_id,
                    "entity_id": entity_id,
                    "start_index": int(start_idx),
                    "end_index": int(end_idx - 1),
                    "start_time": str(window_df.iloc[0][timestamp_column]),
                    "end_time": str(window_df.iloc[-1][timestamp_column]),
                }
            )
            window_id += 1

    if not windows:
        raise RuntimeError("No sliding windows were created.")

    metadata = pd.DataFrame(rows).sort_values(["end_time", "entity_id", "window_id"]).reset_index(drop=True)
    X = np.asarray(windows, dtype=np.float32)
    C = np.asarray(contexts, dtype=np.float32)

    feature_meta = {
        "feature_columns": feature_columns,
        "context_columns": context_columns,
        "entity_columns": entity_columns,
        "timestamp_column": timestamp_column,
        "preprocessing": "baseline csv -> groupwise fill -> clip(min=0) -> log1p",
        "window_size": window_size,
        "stride": stride,
        "sample_windows": dataframe_to_records(metadata, limit=5),
    }
    dataset_meta = {
        "source": "optional_baseline_csv",
        "num_windows": int(len(X)),
        "window_size": int(window_size),
        "num_features": int(X.shape[-1]),
        "context_dim": int(C.shape[-1]),
        "feature_columns": feature_columns,
        "context_columns": context_columns,
    }

    return DatasetBundle(X=X, C=C, metadata=metadata, feature_meta=feature_meta, dataset_meta=dataset_meta, context_encoder=context_encoder)


def build_dataset_from_csv(csv_path: str | Path, dataset_config: DatasetConfig, output_dir: str | Path) -> tuple[DatasetBundle, dict[str, str]]:
    raw_df = load_metrics_csv(csv_path, dataset_config)
    clean_df = preprocess_metrics_frame(raw_df, dataset_config)
    context_encoder = fit_context_encoder(clean_df, dataset_config)
    bundle = generate_sliding_windows(clean_df, dataset_config, context_encoder)
    bundle.metadata = assign_splits(bundle.metadata, dataset_config)
    paths = save_dataset_bundle(bundle, output_dir)
    return bundle, paths


def load_dataset_bundle(processed_dir: str | Path) -> DatasetBundle:
    processed_dir = Path(processed_dir)
    X = np.load(processed_dir / "X_all.npy", allow_pickle=False)
    C = np.load(processed_dir / "C_all.npy", allow_pickle=False)
    metadata = pd.read_csv(processed_dir / "window_metadata.csv")
    feature_meta = joblib.load(processed_dir / "feature_meta.joblib")
    dataset_meta = load_json(processed_dir / "dataset_meta.json")
    context_encoder = joblib.load(processed_dir / "context_encoder.joblib")
    return DatasetBundle(
        X=X,
        C=C,
        metadata=metadata,
        feature_meta=feature_meta,
        dataset_meta=dataset_meta,
        context_encoder=context_encoder,
    )
