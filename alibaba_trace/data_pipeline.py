"""
data_pipeline.py — PROXY MODULE
=============================================
This file has been refactored into the `alibaba_trace.data` package.
This script is preserved purely for backward compatibility so that existing imports
continue to work seamlessly.
"""

import sys
import os

# Keep imports working when Jupyter starts inside alibaba_trace.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from alibaba_trace.data.scaling import StreamingMinMaxScaler
from alibaba_trace.data.dataset import AlibabaTraceDataset, FEATURE_COLS, META_COLS, WINDOW_SIZE
from alibaba_trace.data.loaders import build_dataloader, build_split_dataloaders
from alibaba_trace.data.utils import count_csv_rows_in_tar_gz, quick_sanity_check

# Keep this helper available for older scripts that import it directly.
from alibaba_trace.data.dataset import _encode_metadata
