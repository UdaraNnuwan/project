import io
import time
import tarfile
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
)
logger = logging.getLogger("data_pipeline")

FEATURE_COLS: List[str] = [
    "cpu_util_percent",
    "mem_util_percent",
    "cpu_request",
    "mem_request",
    "net_in",
    "net_out",
    "disk_io_percent",
]

META_COLS: List[str] = [
    "container_id",
    "machine_id",
]

TIME_COL: str = "time_stamp"

ALIBABA_V2018_COLS = [
    "container_id", "machine_id", "time_stamp",
    "cpu_util_percent", "mem_util_percent",
    "cpu_request", "mem_request", "unknown_metrics",
    "net_in", "net_out", "disk_io_percent"
]

WINDOW_SIZE: int = 50

STRIDE: int = 10

CHUNK_SIZE: int = 5000

from .scaling import StreamingMinMaxScaler
from .dataset import AlibabaTraceDataset
from .utils import count_csv_rows_in_tar_gz

def build_dataloader(
    tar_path: str,
    csv_member_name: str,
    scaler: StreamingMinMaxScaler,
    feature_cols: List[str] = FEATURE_COLS,
    meta_cols: List[str] = META_COLS,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    chunk_size: int = CHUNK_SIZE,
    batch_size: int = 32,
    split: str = "train",
    train_ratio: float = 0.70,
    total_rows: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    include_next_step: bool = False,
    max_chunks: Optional[int] = None,
) -> DataLoader:
    dataset = AlibabaTraceDataset(
        tar_path=tar_path,
        csv_member_name=csv_member_name,
        scaler=scaler,
        feature_cols=feature_cols,
        meta_cols=meta_cols,
        window_size=window_size,
        stride=stride,
        chunk_size=chunk_size,
        split=split,
        train_ratio=train_ratio,
        total_rows=total_rows,
        include_next_step=include_next_step,
        max_chunks=max_chunks,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.info(
        "Built DataLoader | split=%s | window=%d | stride=%d | batch=%d | "
        "workers=%d | include_next_step=%s",
        split, window_size, stride, batch_size, num_workers, include_next_step,
    )
    return loader

def build_split_dataloaders(
    tar_path: str,
    csv_member_name: str,
    scaler: StreamingMinMaxScaler,
    feature_cols: List[str] = FEATURE_COLS,
    meta_cols: List[str] = META_COLS,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    chunk_size: int = CHUNK_SIZE,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    num_workers: int = 4,
    pin_memory: bool = True,
    include_next_step: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    logger.info(
        "[build_split_dataloaders] Phase 1/2 — counting rows in the archive …"
    )
    total_rows: int = count_csv_rows_in_tar_gz(tar_path, csv_member_name)
    training_row_limit = int(total_rows * train_ratio)

    logger.info(
        "[build_split_dataloaders] Split summary:\n"
        "  Total rows         : %d\n"
        "  Train ratio        : %.1f %%  → rows 0 … %d\n"
        "  Test  ratio        : %.1f %%  → rows %d … %d",
        total_rows,
        train_ratio * 100, training_row_limit - 1,
        (1 - train_ratio) * 100, training_row_limit, total_rows - 1,
    )

    logger.info(
        "[build_split_dataloaders] Phase 2/2 — constructing DataLoaders …"
    )
    _kwargs = dict(
        tar_path=tar_path,
        csv_member_name=csv_member_name,
        scaler=scaler,
        feature_cols=feature_cols,
        meta_cols=meta_cols,
        window_size=window_size,
        stride=stride,
        chunk_size=chunk_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
        total_rows=total_rows,
        num_workers=num_workers,
        pin_memory=pin_memory,
        include_next_step=include_next_step,
    )

    train_loader = build_dataloader(split="train", **_kwargs)
    test_loader  = build_dataloader(split="test",  **_kwargs)

    return train_loader, test_loader, total_rows
