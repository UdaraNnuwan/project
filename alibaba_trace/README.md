# Container Anomaly Detection — Alibaba Cloud Trace 2018
### BiLSTM + FiLM Autoencoder · Memory-Efficient Streaming Pipeline

> **Thesis project** — Detect anomalies in containerised environments using the
> Alibaba Cloud Trace 2018 dataset, with zero local extraction of the 28.5 GB
> `container_usage.tar.gz` archive.

---

## Project Structure

```
alibaba_trace/
├── data_pipeline.py               ← Streaming pipeline (tarfile + pandas + tf.data)
├── model_architecture.py          ← BiLSTM Autoencoder + FiLM layer
├── 01_Model_Training.ipynb        ← Train the model, save weights + history
├── 02_Evaluation_and_Thesis_Graphs.ipynb  ← Evaluate + generate thesis figures
├── requirements.txt
└── README.md
outputs/                           ← Created automatically at runtime
├── model.weights.h5               ← Best checkpoint
├── training_history.json          ← Loss / MAE per epoch + run metadata
├── scaler_params.npz              ← Fitted min/max arrays
└── thesis_figures/
    ├── fig1_loss_curve.png        ← 300 DPI
    ├── fig2_error_dist.png        ← 300 DPI
    ├── fig3_confusion_matrix.png  ← 300 DPI
    └── fig4_qualitative_reconstruction.png
```

---

## Quick Start

### 1 · Install dependencies
```bash
pip install -r alibaba_trace/requirements.txt
```

### 2 · Discover archive member names
Open `01_Model_Training.ipynb` and run **Section 1 (Discover Archive Layout)**.
This prints all member names inside the `.tar.gz` so you can set `CSV_MEMBER` correctly.

### 3 · Edit the configuration block
In both notebooks, edit the `# CONFIGURATION` cell:
```python
TAR_PATH   = r'C:\path\to\container_usage.tar.gz'   # ← your actual path
CSV_MEMBER = 'container_usage/container_usage.csv'   # ← from discover step
```

### 4 · Run the notebooks in order
1. `01_Model_Training.ipynb` — trains the model (streams data, never extracts)
2. `02_Evaluation_and_Thesis_Graphs.ipynb` — evaluates and saves thesis figures

---

## Architecture

### `data_pipeline.py`
| Component | Description |
|---|---|
| `StreamingMinMaxScaler` | Fits robust (1st–99th percentile) min/max from a calibration sample of chunks; transform is O(1) |
| `_sliding_window_generator` | Python generator; opens archive with `tarfile`, streams CSV rows with `pd.read_csv(chunksize=...)`, maintains a rolling buffer across chunk boundaries |
| `build_tf_dataset` | Wraps the generator in `tf.data.Dataset.from_generator`; applies shuffle → batch → map (autoencoder format) → prefetch |
| `quick_sanity_check` | Lists archive members, prints column names and NaN counts — zero data loaded to RAM |

**Memory guarantee**: At any moment, at most `CHUNK_SIZE × columns × 4 bytes + shuffle_buffer × window_bytes` is held in RAM.

### `model_architecture.py`
```
Time Series Input (B, W, F)
    ↓
BiLSTM(128) → BN → BiLSTM(64) → Dropout
    ↓
Dense(latent_dim=64)  ← Encoding
    ↓
FiLMLayer(γ, β ← Dense(metadata))  ← Conditioned Encoding
    ↓
Dense → RepeatVector(W) → BiLSTM(64) → BN → BiLSTM(128)
    ↓
TimeDistributed(Dense(F, sigmoid))  ← Reconstruction (B, W, F)
```

**FiLM**: `γ * encoding + β` — lightweight, interpretable metadata conditioning without a separate model branch.

---

## Anomaly Detection Strategy

| Step | Detail |
|---|---|
| Train | MSE reconstruction loss on normal (unlabelled) windows |
| Score | Per-sample MSE between original and reconstructed window |
| Threshold | **Dynamic 95th percentile** of test-set reconstruction errors |
| Flag | Samples with MSE > threshold → Anomaly |

---

## Thesis Figures

| Figure | File | Description |
|---|---|---|
| 1 | `fig1_loss_curve.png` | Train vs. Val MSE with generalisation gap shading |
| 2 | `fig2_error_dist.png` | Error distribution histogram + KDE + threshold line |
| 3 | `fig3_confusion_matrix.png` | Raw counts + normalised %, annotated with P/R/F1 |
| 4 | `fig4_qualitative_reconstruction.png` | Original vs. reconstructed for normal & anomalous windows |

All figures are **300 DPI PNG**, suitable for direct inclusion in LaTeX/Word.

---

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `WINDOW_SIZE` | 50 | Timesteps per sample |
| `STRIDE` | 10 | Window overlap = 80 % |
| `CHUNK_SIZE` | 5 000 | Rows per pandas chunk |
| `LATENT_DIM` | 64 | Bottleneck dimension |
| `LSTM_UNITS` | (128, 64) | BiLSTM hidden units |
| `DROPOUT_RATE` | 0.2 | Applied after each BiLSTM |
| `THRESHOLD_PERCENTILE` | 95 | Dynamic anomaly threshold |
| `ANOMALY_INJECTION_RATIO` | 5 % | Synthetic labels for evaluation |
