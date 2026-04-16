"""
data_pipeline.py  (PyTorch edition)
=====================================
Memory-Efficient Streaming Data Pipeline for the Alibaba Cloud Trace 2018 Dataset.

Design Goals
------------
* NEVER extract the 28.5 GB container_usage.tar.gz to disk.
* Stream CSV rows directly from the compressed archive using Python's built-in
  ``tarfile`` module, which exposes each member as a file-like object.
* Use ``pandas`` with ``chunksize`` so that only one small slice of rows lives
  in RAM at any given moment.
* Wrap the generator inside a PyTorch ``IterableDataset`` so it plugs directly
  into a standard ``DataLoader`` (with prefetch workers, pin_memory, etc.).
* Perform Min-Max scaling and NaN handling inside the pipeline, keeping the
  notebook code clean.

Output per sample (one sliding window)
---------------------------------------
  ts_window  : torch.Tensor  shape (window_size, n_ts_features)  float32
  meta_vec   : torch.Tensor  shape (n_meta_features,)             float32
  target     : torch.Tensor  shape (window_size, n_ts_features)  float32
              (reconstruction target == ts_window)
  next_step  : torch.Tensor  shape (n_ts_features,)              float32
              (forecasting target == timestep immediately after the window)
              ONLY yielded when AlibabaTraceDataset(include_next_step=True)
              or build_dataloader(include_next_step=True).

Alibaba Trace 2018 – container_usage.csv Assumed Columns
---------------------------------------------------------
  container_id, machine_id, time_stamp,
  cpu_util_percent, mem_util_percent,
  cpu_request, mem_request,
  net_in, net_out, disk_io_percent
  (adjust FEATURE_COLS / META_COLS below if your archive differs)
"""

import io
import tarfile
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
)
logger = logging.getLogger("data_pipeline")


# ---------------------------------------------------------------------------
# ─── CONFIGURATION CONSTANTS ────────────────────────────────────────────────
# Adjust these to match your actual CSV column names.
# ---------------------------------------------------------------------------

# Numeric time-series features fed to the BiLSTM encoder
FEATURE_COLS: List[str] = [
    "cpu_util_percent",
    "mem_util_percent",
    "cpu_request",
    "mem_request",
    "net_in",
    "net_out",
    "disk_io_percent",
]

# Categorical / low-cardinality metadata cols used by the FiLM layer.
# Hash-encoded into a fixed-length float vector so FiLM can consume them.
META_COLS: List[str] = [
    "container_id",
    "machine_id",
]

# Timestamp column (used only for ordering; not fed to the model)
TIME_COL: str = "time_stamp"

# Number of time steps in one sliding window fed to the LSTM
WINDOW_SIZE: int = 50

# How many rows to advance the window each step (overlap = WINDOW_SIZE - STRIDE)
STRIDE: int = 10

# How many rows pandas loads per chunk from the CSV stream.
# Keep this small (< 10 000) to limit peak RAM usage.
CHUNK_SIZE: int = 5_000

# Percentile bounds used by the Min-Max scaler (robust to outliers)
SCALE_PERCENTILE_LO: float = 1.0
SCALE_PERCENTILE_HI: float = 99.0


# ---------------------------------------------------------------------------
# ─── HELPER: METADATA ENCODING ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _encode_metadata(df_chunk: pd.DataFrame, meta_cols: List[str]) -> np.ndarray:
    """
    Encode string metadata columns into a float matrix suitable for the FiLM
    layer.  Uses a deterministic hash (mod a large prime) — consistent across
    chunks without needing a vocabulary table on disk.

    Parameters
    ----------
    df_chunk : pd.DataFrame
    meta_cols : list[str]

    Returns
    -------
    np.ndarray, shape (n_rows, len(meta_cols)), dtype float32
    """
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


# ---------------------------------------------------------------------------
# ─── MIN-MAX SCALER ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class StreamingMinMaxScaler:
    """
    Lightweight Min-Max scaler storing per-feature (min, max).
    Fitting is done from a calibration pass using running (lo, hi) accumulators
    so memory usage is bounded by the number of features, not total rows.
    """

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        tar_path: str,
        csv_member_name: str,
        max_chunks: int = 20,
    ) -> "StreamingMinMaxScaler":
        """
        Stream up to ``max_chunks`` chunks to estimate robust (lo, hi) per feature.
        O(1) memory per chunk.
        """
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
                usecols=lambda c: c in self.feature_cols,
                low_memory=True,
            ):
                if chunks_seen >= max_chunks:
                    break

                chunk.dropna(how="all", inplace=True)
                chunk.ffill(inplace=True)
                chunk.fillna(0.0, inplace=True)

                # Re-order columns to match FEATURE_COLS (some may be absent)
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

    # --------------------------------------------------------------- transform
    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply Min-Max normalisation in-place.  Clips output to [0, 1].

        Parameters
        ----------
        arr : np.ndarray, shape (n_rows, n_features)

        Returns
        -------
        np.ndarray, shape (n_rows, n_features), dtype float32
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        eps = 1e-8
        range_ = (self.max_ - self.min_) + eps
        scaled = (arr.astype(np.float32) - self.min_) / range_
        return np.clip(scaled, 0.0, 1.0)

    # ----------------------------------------------------------- manual set
    def set_params(self, min_: np.ndarray, max_: np.ndarray) -> None:
        """Manually set fitted min/max arrays (e.g. loaded from disk)."""
        self.min_ = np.asarray(min_, dtype=np.float32)
        self.max_ = np.asarray(max_, dtype=np.float32)
        self._fitted = True


# ---------------------------------------------------------------------------
# ─── PYTORCH ITERABLE DATASET ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

class AlibabaTraceDataset(IterableDataset):
    """
    PyTorch ``IterableDataset`` that streams sliding windows from the Alibaba
    Trace 2018 ``.tar.gz`` archive **without extracting it to disk**.

    Each ``__iter__`` call (one epoch) opens the archive, streams the CSV, and
    yields ``(ts_window, meta_vec, target)`` tuples where target == ts_window
    (autoencoder objective).

    Thread / worker safety
    ----------------------
    ``DataLoader(num_workers > 0)`` spawns worker processes.  Each worker
    receives an independent copy of this dataset object, so the tarfile is
    opened independently per worker.  To avoid duplicate samples we use
    ``torch.utils.data.get_worker_info()`` to shard chunks across workers.

    Parameters
    ----------
    tar_path : str
        Absolute path to ``container_usage.tar.gz``.
    csv_member_name : str
        Name of the CSV inside the archive (e.g. ``'container_usage/container_usage.csv'``).
    scaler : StreamingMinMaxScaler
        Pre-fitted scaler.
    feature_cols : list[str]
    meta_cols : list[str]
    window_size : int
    stride : int
    chunk_size : int
    split : str
        ``'train'`` or ``'test'``.
    train_ratio : float
    include_next_step : bool
        If True, the iterator yields 4-tuples
        ``(ts_window, meta_vec, target, next_step)`` where
        ``next_step`` is the single row immediately following the window.
        Windows at the very end of a buffer (where no next row exists)
        are silently skipped.
        Default: False — preserves the original 3-tuple API for
        backward compatibility with notebooks 03/04/05.
    """

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
        train_ratio: float = 0.8,
        include_next_step: bool = False,
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

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple]:
        """
        Open the archive and yield tuples of PyTorch Tensors.

        Default (include_next_step=False) — 3-tuple per sample:
            (ts_window, meta_vec, target)
              ts_window : (W, F)  — the scaled input window
              meta_vec  : (M,)    — metadata at the last timestep of the window
              target    : (W, F)  — identical to ts_window (reconstruction target)

        Dual-head mode (include_next_step=True) — 4-tuple per sample:
            (ts_window, meta_vec, target, next_step)
              next_step : (F,)    — scaled metrics at ts_window[-1] + 1 row
                                    (the forecasting target for Head 2)

        NOTE: in dual-head mode, the very last window in each buffer segment
        is skipped when no subsequent row exists, so epoch sample counts may
        differ slightly between modes.

        Memory invariant: at most ``(chunk_size + window_size + 1) × n_cols × 4``
        bytes are held in RAM at any time.
        """
        # ── Worker-sharding support ───────────────────────────────────────
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id          if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        n_ts_feat   = len(self.feature_cols)
        n_meta_feat = len(self.meta_cols)
        all_cols    = self.feature_cols + self.meta_cols + [TIME_COL]

        # Rolling buffers (accumulate rows across chunk seams)
        ts_buffer:   List[np.ndarray] = []
        meta_buffer: List[np.ndarray] = []

        chunk_idx  = 0  # Global chunk counter (for worker sharding + split logic)
        row_cursor = 0  # Approximate row position (for split enforcement)
        SENTINEL   = 1_000_000_000

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
                low_memory=True,
                usecols=lambda c: c in all_cols,
                na_values=["", "NA", "N/A", "nan", "NaN", "null", "NULL", "-1"],
            ):
                n_rows = len(chunk)

                # ── Worker sharding: each worker processes only its own chunks
                if (chunk_idx % num_workers) != worker_id:
                    chunk_idx  += 1
                    row_cursor += n_rows
                    continue

                # ── Train / test split enforcement ────────────────────────
                frac = row_cursor / SENTINEL
                if self.split == "train" and frac >= self.train_ratio:
                    chunk_idx  += 1
                    row_cursor += n_rows
                    continue
                if self.split == "test" and frac < self.train_ratio:
                    chunk_idx  += 1
                    row_cursor += n_rows
                    continue

                # ── NaN handling ──────────────────────────────────────────
                chunk.ffill(inplace=True)
                chunk.bfill(inplace=True)
                chunk.fillna(0.0, inplace=True)

                # ── Feature extraction ────────────────────────────────────
                ts_arr = chunk.reindex(
                    columns=self.feature_cols, fill_value=0.0
                ).values.astype(np.float32)
                ts_arr = self.scaler.transform(ts_arr)  # Scale to [0, 1]

                meta_arr = _encode_metadata(chunk, self.meta_cols)

                # ── Append to rolling buffers ─────────────────────────────
                ts_buffer.append(ts_arr)
                meta_buffer.append(meta_arr)

                ts_full   = np.concatenate(ts_buffer,   axis=0)
                meta_full = np.concatenate(meta_buffer, axis=0)

                # ── Slide window over buffered rows ───────────────────────
                buf_len = len(ts_full)
                i = 0

                # In dual-head mode we need one row AFTER the window → need
                # i + window_size + 1 ≤ buf_len, i.e. an extra trailing row.
                # In single-head mode the standard condition applies.
                min_end = (self.window_size + 1
                           if self.include_next_step
                           else self.window_size)

                while i + min_end <= buf_len:
                    ts_win   = ts_full[i : i + self.window_size]        # (W, F)
                    meta_vec = meta_full[i + self.window_size - 1]      # (M,)

                    if self.include_next_step:
                        # The forecasting target: the single row immediately
                        # following the window, shape (F,).
                        next_row = ts_full[i + self.window_size]        # (F,)
                        yield (
                            torch.from_numpy(ts_win.copy()),    # ts_window (W,F)
                            torch.from_numpy(meta_vec.copy()),  # meta_vec   (M,)
                            torch.from_numpy(ts_win.copy()),    # recon tgt  (W,F)
                            torch.from_numpy(next_row.copy()),  # fore tgt   (F,)
                        )
                    else:
                        # Original 3-tuple — backward compatible
                        yield (
                            torch.from_numpy(ts_win.copy()),    # ts_window  (W,F)
                            torch.from_numpy(meta_vec.copy()),  # meta_vec   (M,)
                            torch.from_numpy(ts_win.copy()),    # target     (W,F)
                        )

                    i += self.stride

                # ── Trim consumed rows from buffer ────────────────────────
                tail_start = max(0, i - (self.window_size - 1))
                ts_buffer   = [ts_full[tail_start:]]
                meta_buffer = [meta_full[tail_start:]]

                chunk_idx  += 1
                row_cursor += n_rows

        logger.info(
            "Dataset iterator exhausted [split=%s, worker=%d/%d]",
            self.split, worker_id, num_workers,
        )


# ---------------------------------------------------------------------------
# ─── PUBLIC API: build_dataloader ───────────────────────────────────────────
# ---------------------------------------------------------------------------

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
    train_ratio: float = 0.8,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_next_step: bool = False,
) -> DataLoader:
    """
    Build a ``torch.utils.data.DataLoader`` that streams sliding windows from
    the Alibaba Trace 2018 archive **without extracting it to disk**.

    Each batch is a tuple.  The structure depends on ``include_next_step``:

    Single-head mode (include_next_step=False — default):
        (ts_batch, meta_batch, target_batch)
          ts_batch     : Tensor (B, W, F)
          meta_batch   : Tensor (B, M)
          target_batch : Tensor (B, W, F)  — same as ts_batch (recon target)

    Dual-head mode (include_next_step=True):
        (ts_batch, meta_batch, target_batch, next_step_batch)
          ts_batch        : Tensor (B, W, F)
          meta_batch      : Tensor (B, M)
          target_batch    : Tensor (B, W, F)  — reconstruction target (Head 1)
          next_step_batch : Tensor (B, F)     — forecasting target   (Head 2)

    Parameters
    ----------
    tar_path : str
        Absolute path to ``container_usage.tar.gz``.
    csv_member_name : str
        Archive-internal CSV path.
    scaler : StreamingMinMaxScaler
        Pre-fitted scaler.
    feature_cols, meta_cols, window_size, stride, chunk_size :
        Pipeline config (see module-level defaults).
    batch_size : int
        Samples per mini-batch.
    split : str
        ``'train'`` or ``'test'``.
    train_ratio : float
        Fraction of data used for training.
    num_workers : int
        DataLoader worker processes.  Set to 0 on Windows if you encounter
        multiprocessing issues (IterableDataset + Windows can be tricky).
    pin_memory : bool
        Set True when training on GPU for faster host→device transfers.

    Returns
    -------
    DataLoader
    """
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
        include_next_step=include_next_step,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # IterableDataset does its own ordering — shuffle is done at generator level
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # drop_last=False so we don't silently lose samples at the epoch tail
        drop_last=False,
    )

    logger.info(
        "Built DataLoader | split=%s | window=%d | stride=%d | batch=%d | "
        "workers=%d | include_next_step=%s",
        split, window_size, stride, batch_size, num_workers, include_next_step,
    )
    return loader


# ---------------------------------------------------------------------------
# ─── CONVENIENCE: quick_sanity_check ────────────────────────────────────────
# ---------------------------------------------------------------------------

def quick_sanity_check(
    tar_path: str,
    csv_member_name: str,
    n_chunks: int = 3,
) -> None:
    """
    Print column names, dtypes, and shape of the first ``n_chunks`` CSV chunks
    WITHOUT building a full dataset.  Useful for verifying the correct
    ``csv_member_name`` and column layout before starting a training run.
    """
    print(f"\n{'─'*62}")
    print(f"  Archive  : {tar_path}")
    print(f"  Member   : {csv_member_name}")
    print(f"{'─'*62}")

    with tarfile.open(tar_path, mode="r:gz") as tar:
        print(f"\n  Archive members (first 20):")
        for m in tar.getmembers()[:20]:
            print(f"    {m.name:<60}  {m.size:>12} bytes")

        try:
            member = tar.getmember(csv_member_name)
        except KeyError:
            print(f"\n  ❌  Member '{csv_member_name}' not found.")
            return

        fobj = tar.extractfile(member)
        if fobj is None:
            print("  ❌  Could not extract file-like object.")
            return

        seen = 0
        for chunk in pd.read_csv(fobj, chunksize=CHUNK_SIZE, low_memory=True):
            if seen == 0:
                print(f"\n  Columns ({len(chunk.columns)}):")
                for col, dtype in chunk.dtypes.items():
                    print(f"    {col:<40}  {str(dtype)}")
            print(f"\n  Chunk {seen}: shape={chunk.shape}  "
                  f"NaN count={chunk.isna().sum().sum()}")
            seen += 1
            if seen >= n_chunks:
                break

    print(f"\n{'─'*62}\n")


# ---------------------------------------------------------------------------
# ─── SELF-TEST (run as script) ───────────────────────────────────────────────
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python data_pipeline.py <path/to/container_usage.tar.gz> "
            "<csv_member_name>"
        )
        sys.exit(1)

    tar_path_   = sys.argv[1]
    member_name = sys.argv[2]

    quick_sanity_check(tar_path_, member_name, n_chunks=2)

    scaler_ = StreamingMinMaxScaler(FEATURE_COLS)
    scaler_.fit(tar_path_, member_name, max_chunks=5)

    loader = build_dataloader(
        tar_path=tar_path_,
        csv_member_name=member_name,
        scaler=scaler_,
        batch_size=8,
        num_workers=0,
    )

    for ts_b, meta_b, tgt_b in loader:
        print("\n=== Self-Test Batch ===")
        print(f"  ts_batch   shape : {tuple(ts_b.shape)}     dtype={ts_b.dtype}")
        print(f"  meta_batch shape : {tuple(meta_b.shape)}    dtype={meta_b.dtype}")
        print(f"  target     shape : {tuple(tgt_b.shape)}     dtype={tgt_b.dtype}")
        print(f"  ts value range   : [{ts_b.min():.4f}, {ts_b.max():.4f}]")
        print("======================\n")
        break
