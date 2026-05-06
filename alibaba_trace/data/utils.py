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

from .dataset import *
from .loaders import *
def count_csv_rows_in_tar_gz(
    tar_path: str,
    csv_member_name: str,
    *,
    read_buffer_bytes: int = 1 << 20,  # 1 MB works well for streaming reads.
) -> int:
    logger.info(
        "[count_csv_rows_in_tar_gz] Streaming '%s' to count rows …",
        csv_member_name,
    )
    t0 = time.perf_counter()

    with tarfile.open(tar_path, mode="r:gz") as tar:
        try:
            member = tar.getmember(csv_member_name)
        except KeyError as exc:
            raise RuntimeError(
                f"Member '{csv_member_name}' not found inside '{tar_path}'. "
                f"Available members: {[m.name for m in tar.getmembers()[:10]]} …"
            ) from exc

        fobj = tar.extractfile(member)
        if fobj is None:
            raise RuntimeError(
                f"Could not obtain a file-like object for '{csv_member_name}' "
                f"inside '{tar_path}'."
            )

        newline_count = 0
        last_byte     = b""  # Used to check if the file ends with a newline.

        while True:
            raw = fobj.read(read_buffer_bytes)
            if not raw:
                break
            newline_count += raw.count(b"\n")
            last_byte      = raw[-1:]  # Keep the last byte from this block.

        if last_byte and last_byte != b"\n":
            newline_count += 1

    data_rows = max(0, newline_count - 1)

    elapsed_s   = time.perf_counter() - t0
    size_mb     = member.size / (1 << 20)   # Uncompressed size in MB.
    throughput  = size_mb / elapsed_s if elapsed_s > 0 else float("inf")

    logger.info(
        "[count_csv_rows_in_tar_gz] Done: %d data rows | %.1fs | %.0f MB/s",
        data_rows, elapsed_s, throughput,
    )
    return data_rows

def quick_sanity_check(
    tar_path: str,
    csv_member_name: str,
    n_chunks: int = 3,
) -> None:
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
            print(f"\n  [ERROR]  Member '{csv_member_name}' not found.")
            return

        fobj = tar.extractfile(member)
        if fobj is None:
            print("  [ERROR]  Could not extract file-like object.")
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
