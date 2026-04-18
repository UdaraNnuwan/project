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

def _encode_metadata(df_chunk: pd.DataFrame, meta_cols: List[str]) -> np.ndarray:
    encoded = np.zeros((len(df_chunk), len(meta_cols)), dtype=np.float32)
    LARGE_PRIME = 999_983

    for col_idx, col in enumerate(meta_cols):
        if col in df_chunk.columns:
            col_series = df_chunk[col].astype(str).fillna("__MISSING__")
            encoded[:, col_idx] = (
                col_series.map(lambda s: hash(s) % LARGE_PRIME) / LARGE_PRIME
            ).astype(np.float32)
        else:
            logger.warning("Metadata column '%s' not found; filling with zeros.", col)

    return encoded

class AlibabaTraceDataset(IterableDataset):

    def __init__(
        self,
        tar_path: str,
        csv_member_name: str,
        scaler: StreamingMinMaxScaler,
        feature_cols: List[str] = FEATURE_COLS,
        meta_cols: List[str] = META_COLS,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        chunk_size: int = CHUNK_SIZE,
        split: str = "train",
        train_ratio: float = 0.70,
        total_rows: Optional[int] = None,
        include_next_step: bool = False,
        max_chunks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tar_path          = tar_path
        self.csv_member        = csv_member_name
        self.scaler            = scaler
        self.feature_cols      = feature_cols
        self.meta_cols         = meta_cols
        self.window_size       = window_size
        self.stride            = stride
        self.chunk_size        = chunk_size
        self.split             = split
        self.train_ratio       = train_ratio
        self.include_next_step = include_next_step
        self.max_chunks        = max_chunks

        if total_rows is not None and total_rows > 0:
            self._training_row_limit: int = int(total_rows * train_ratio)
            self._use_exact_split: bool   = True
            logger.info(
                "AlibabaTraceDataset [split=%s]: exact split — "
                "total=%d | train_limit=%d (%.1f %%) | test_start=%d (%.1f %%)",
                split,
                total_rows,
                self._training_row_limit,
                train_ratio * 100,
                self._training_row_limit,
                (1 - train_ratio) * 100,
            )
        else:
            self._training_row_limit = 0          # unused in this branch
            self._use_exact_split    = False
            logger.warning(
                "AlibabaTraceDataset [split=%s]: total_rows not supplied — "
                "falling back to SENTINEL heuristic (less precise). "
                "Pass total_rows=count_csv_rows_in_tar_gz(...) for exact splits.",
                split,
            )

    def __iter__(self) -> Iterator[Tuple]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id          if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        n_ts_feat   = len(self.feature_cols)
        n_meta_feat = len(self.meta_cols)
        all_cols    = self.feature_cols + self.meta_cols + [TIME_COL]

        ts_buffer:   List[np.ndarray] = []
        meta_buffer: List[np.ndarray] = []

        chunk_idx  = 0  # Global chunk counter (for worker sharding + split logic)
        processed_chunks = 0
        row_cursor = 0  # Running total of rows consumed so far (all workers combined)

        _SENTINEL = 1_000_000_000

        with tarfile.open(self.tar_path, mode="r:gz") as tar:
            member = tar.getmember(self.csv_member)
            fobj   = tar.extractfile(member)
            if fobj is None:
                raise RuntimeError(
                    f"Cannot open '{self.csv_member}' inside '{self.tar_path}'."
                )

            for chunk in pd.read_csv(
                fobj,
                chunksize=self.chunk_size,
                names=ALIBABA_V2018_COLS,
                header=None,
                low_memory=True,
                na_values=["", "NA", "N/A", "nan", "NaN", "null", "NULL", "-1"],
            ):
                if self.max_chunks is not None and processed_chunks >= self.max_chunks:
                    logger.info(f"Reached configured limit of {self.max_chunks} chunks. Stopping iteration early.")
                    break
                    
                n_rows = len(chunk)

                if (chunk_idx % num_workers) != worker_id:
                    chunk_idx  += 1
                    row_cursor += n_rows
                    continue

                if self._use_exact_split:
                    in_train_region = row_cursor < self._training_row_limit
                else:
                    in_train_region = (row_cursor / _SENTINEL) < self.train_ratio

                if self.split == "train" and not in_train_region:
                    chunk_idx  += 1
                    row_cursor += n_rows
                    continue
                if self.split == "test" and in_train_region:
                    chunk_idx  += 1
                    row_cursor += n_rows
                    continue

                processed_chunks += 1

                chunk.ffill(inplace=True)
                chunk.bfill(inplace=True)
                
                fill_dict_ds = {c: 0.0 for c in self.feature_cols if c in chunk.columns}
                for mc in self.meta_cols:
                    if mc in chunk.columns:
                        fill_dict_ds[mc] = "missing"
                chunk.fillna(value=fill_dict_ds, inplace=True)
                
                print(f"[Worker {worker_id}] Processing Chunk Round {chunk_idx + 1}: rows {row_cursor} to {row_cursor + n_rows} (split={self.split})")
                import sys; sys.stdout.flush()

                ts_arr = chunk.reindex(
                    columns=self.feature_cols, fill_value=0.0
                ).values.astype(np.float32)
                ts_arr = self.scaler.transform(ts_arr)  # Scale to [0, 1]

                meta_arr = _encode_metadata(chunk, self.meta_cols)

                ts_buffer.append(ts_arr)
                meta_buffer.append(meta_arr)

                ts_full   = np.concatenate(ts_buffer,   axis=0)
                meta_full = np.concatenate(meta_buffer, axis=0)

                buf_len = len(ts_full)
                i = 0

                min_end = (self.window_size + 1
                           if self.include_next_step
                           else self.window_size)

                while i + min_end <= buf_len:
                    ts_win   = ts_full[i : i + self.window_size]        # (W, F)
                    meta_vec = meta_full[i + self.window_size - 1]      # (M,)

                    if self.include_next_step:
                        next_row = ts_full[i + self.window_size]        # (F,)
                        yield (
                            torch.from_numpy(ts_win.copy()),    # ts_window (W,F)
                            torch.from_numpy(meta_vec.copy()),  # meta_vec   (M,)
                            torch.from_numpy(ts_win.copy()),    # recon tgt  (W,F)
                            torch.from_numpy(next_row.copy()),  # fore tgt   (F,)
                        )
                    else:
                        yield (
                            torch.from_numpy(ts_win.copy()),    # ts_window  (W,F)
                            torch.from_numpy(meta_vec.copy()),  # meta_vec   (M,)
                            torch.from_numpy(ts_win.copy()),    # target     (W,F)
                        )

                    i += self.stride

                tail_start = max(0, i - (self.window_size - 1))
                ts_buffer   = [ts_full[tail_start:]]
                meta_buffer = [meta_full[tail_start:]]

                chunk_idx  += 1
                row_cursor += n_rows

        logger.info(
            "Dataset iterator exhausted [split=%s, worker=%d/%d]",
            self.split, worker_id, num_workers,
        )
