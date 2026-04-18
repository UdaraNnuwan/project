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

SCALE_PERCENTILE_LO: float = 1.0
SCALE_PERCENTILE_HI: float = 99.0

class StreamingMinMaxScaler:

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        tar_path: str,
        csv_member_name: str,
        max_chunks: int = 20,
    ) -> "StreamingMinMaxScaler":
        logger.info("Fitting StreamingMinMaxScaler (up to %d chunks)…", max_chunks)

        running_lo = np.full(self.n_features, np.inf,  dtype=np.float64)
        running_hi = np.full(self.n_features, -np.inf, dtype=np.float64)

        chunks_seen = 0
        with tarfile.open(tar_path, mode="r:gz") as tar:
            member = tar.getmember(csv_member_name)
            fobj = tar.extractfile(member)
            if fobj is None:
                raise RuntimeError(
                    f"Could not open '{csv_member_name}' inside {tar_path}"
                )

            for chunk in pd.read_csv(
                fobj,
                chunksize=CHUNK_SIZE,
                names=ALIBABA_V2018_COLS,
                header=None,
                low_memory=True,
            ):
                if chunks_seen >= max_chunks:
                    break

                chunk.dropna(how="all", inplace=True)
                chunk.ffill(inplace=True)
                fill_dict = {c: 0.0 for c in self.feature_cols if c in chunk.columns}
                chunk.fillna(value=fill_dict, inplace=True)

                arr = chunk.reindex(columns=self.feature_cols, fill_value=0.0).values.astype(np.float64)

                lo = np.percentile(arr, SCALE_PERCENTILE_LO, axis=0)
                hi = np.percentile(arr, SCALE_PERCENTILE_HI, axis=0)

                running_lo = np.minimum(running_lo, lo)
                running_hi = np.maximum(running_hi, hi)

                chunks_seen += 1

        self.min_ = running_lo.astype(np.float32)
        self.max_ = running_hi.astype(np.float32)
        self._fitted = True
        logger.info(
            "Scaler fitted: min=%s  max=%s", self.min_.round(4), self.max_.round(4)
        )
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        eps = 1e-8
        range_ = (self.max_ - self.min_) + eps
        scaled = (arr.astype(np.float32) - self.min_) / range_
        return np.clip(scaled, 0.0, 1.0)

    def set_params(self, min_: np.ndarray, max_: np.ndarray) -> None:
        self.min_ = np.asarray(min_, dtype=np.float32)
        self.max_ = np.asarray(max_, dtype=np.float32)
        self._fitted = True
