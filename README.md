# Container Anomaly Detection Research Pipeline

This project uses a FiLM-conditioned autoencoder as the core multivariate anomaly detector for Alibaba container telemetry. GPT is only used after anomaly score threshold crossing for interpretation, false-positive filtering, severity classification, and recommended action generation.

## Notebook Entry Points

- `notebooks/00_quickstart_smoke.ipynb`: fastest runnable notebook using the tiny synthetic smoke dataset in `data/smoke_processed/`
- `notebooks/01_build_dataset.ipynb` to `notebooks/04_gpt.ipynb`: research notebook flow using Alibaba raw archives and reduced research-smoke outputs

## Project Structure

```text
project/
|-- notebooks/
|   |-- 00_quickstart_smoke.ipynb
|   |-- 01_build_dataset.ipynb
|   |-- 02_train.ipynb
|   |-- 03_eval.ipynb
|   `-- 04_gpt.ipynb
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- gpt_adjudicator.py
|   |-- config.py
|   |-- streaming_csv_train.py
|   |-- streaming_train.py
|   `-- utils.py
|-- data/
|-- models/
|-- results/
|-- generate_research_notebooks.py
|-- README.md
`-- .env
```

Quickstart artifacts live under `data/smoke_processed/`, `models/smoke/`, and `results/smoke/`.

Research smoke artifacts live under `data/research_processed_smoke_auto/`, `models/research_smoke_auto/`, and `results/research_smoke_auto/`.

## Pipeline

1. `notebooks/00_quickstart_smoke.ipynb`
   Uses the bundled tiny smoke dataset.
   Retrains the notebook-friendly baseline on CPU.
   Runs evaluation and writes outputs to `results/smoke/`.

2. `notebooks/01_build_dataset.ipynb`
   Loads `container_meta.tar.gz`, `container_usage.tar.gz`, `machine_meta.tar.gz`, and `machine_usage.tar.gz` directly from the Alibaba archives.
   Cleans and aligns container and machine context.
   Builds a larger chunked research dataset with a 10,000,000-row raw usage target and saves artifacts to `data/research_processed_smoke_auto/`.

3. `notebooks/02_train.ipynb`
   Loads `data/research_processed_smoke_auto/`.
   Initializes the FiLM autoencoder from [`src/model.py`](/c:/Users/kaspe/Desktop/Project/project/src/model.py).
   Runs training and validation, plots losses, fits the threshold, and saves checkpoints to `models/research_smoke_auto/`.

4. `notebooks/03_eval.ipynb`
   Loads the trained model.
   Runs sequential inference in time order.
   Computes anomaly scores, feature-wise reconstruction errors, top-k anomalous features, standard metrics, and strict/relaxed early-detection metrics.
   Saves outputs to `results/research_smoke_auto/`.

5. `notebooks/04_gpt.ipynb`
   Loads threshold-crossing anomaly candidates from `results/research_smoke_auto/`.
   Builds compact GPT inputs from anomaly score, top-k features, feature errors, and context.
   Calls the OpenAI Responses API for structured JSON output.
   Compares AE-only vs AE+GPT decisions and saves outputs to `results/research_smoke_auto/`.

## Streaming Evaluation Flow

`src/evaluate.py` includes a streaming-style time-ordered evaluation path:

1. Process windows in timestamp order.
2. Compute the FiLM autoencoder reconstruction score for each window.
3. If `score > threshold`, extract top-k feature errors.
4. Build a compact anomaly summary.
5. Optionally send that summary to GPT.
6. Store the GPT decision alongside the AE output.

GPT never replaces anomaly detection. It only adjudicates threshold-crossing windows.

## Environment

The default raw archive location is `data/raw/`. Override it with:

```powershell
$env:CONTAINER_AD_RAW_DATA_DIR="C:\path\to\raw\archives"
```

Optional OpenAI settings:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-5-mini-2025-08-07"
```

## Run

Regenerate the notebooks if needed:

```powershell
.\venv\Scripts\python.exe .\generate_research_notebooks.py
```

Run the notebooks in order:

1. `notebooks/00_quickstart_smoke.ipynb` for the fastest initial run
2. `notebooks/01_build_dataset.ipynb`
3. `notebooks/02_train.ipynb`
4. `notebooks/03_eval.ipynb`
5. `notebooks/04_gpt.ipynb`

For raw-data training without writing intermediate processed datasets, use the streaming trainer in [`src/streaming_train.py`](/c:/Users/kaspe/Desktop/Project/project/src/streaming_train.py). It reads the raw archives in chunks, preprocesses each chunk in memory, carries only tail rows across chunk boundaries, fits online scalers in a separate pass, and trains directly from streamed sliding windows.

For a direct single-CSV path with `pandas.read_csv(..., chunksize=...)`, use [`src/streaming_csv_train.py`](/c:/Users/kaspe/Desktop/Project/project/src/streaming_csv_train.py). It streams raw CSV chunks, preprocesses them entirely in memory, creates sliding windows on the fly, and feeds batches directly into training without intermediate CSV/parquet/npy outputs.

## Core Modules

- [`src/dataset.py`](/c:/Users/kaspe/Desktop/Project/project/src/dataset.py): raw archive loading, preprocessing, context creation, and sliding windows
- [`src/model.py`](/c:/Users/kaspe/Desktop/Project/project/src/model.py): FiLM autoencoder
- [`src/train.py`](/c:/Users/kaspe/Desktop/Project/project/src/train.py): training, validation, threshold fitting, checkpoint saving
- [`src/evaluate.py`](/c:/Users/kaspe/Desktop/Project/project/src/evaluate.py): scoring, metrics, early detection, streaming-style post-threshold flow
- [`src/gpt_adjudicator.py`](/c:/Users/kaspe/Desktop/Project/project/src/gpt_adjudicator.py): Responses API integration and structured JSON handling
- [`src/config.py`](/c:/Users/kaspe/Desktop/Project/project/src/config.py): central configuration
- [`src/streaming_csv_train.py`](/c:/Users/kaspe/Desktop/Project/project/src/streaming_csv_train.py): direct chunked CSV preprocessing, overlap-buffer windowing, iterable dataset, and streaming training
- [`src/streaming_train.py`](/c:/Users/kaspe/Desktop/Project/project/src/streaming_train.py): chunked raw-data loader, in-memory preprocessing, iterable sliding-window dataset, and direct streaming training
- [`src/utils.py`](/c:/Users/kaspe/Desktop/Project/project/src/utils.py): helper utilities
