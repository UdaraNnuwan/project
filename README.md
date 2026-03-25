# Multivariate Container Anomaly Detection with GPT Adjudication

This project keeps the FiLM-conditioned multivariate autoencoder as the core anomaly detector and adds a post-threshold GPT decision layer for interpretation, false-positive filtering, severity assignment, and action generation.

## Research Pipeline

1. Data preprocessing and sliding-window generation
2. FiLM autoencoder training
3. Reconstruction-error anomaly scoring
4. Top-k feature error extraction
5. GPT adjudication after threshold crossing
6. Structured JSON incident output

## Target File Tree

```text
container_ad_pipeline/
  __init__.py
  config.py
  dataset.py
  evaluate.py
  gpt_adjudicator.py
  model.py
  prompt_template.txt
  schemas/
    gpt_adjudication_schema.json
  train.py
  utils.py
notebooks/
  01_build_dataset.ipynb
  02_train_film_ae.ipynb
  03_eval_results.ipynb
  04_gpt_adjudication.ipynb
artifacts/research_pipeline/
data/research_processed/
.env.example
README.md
```

## What Stayed the Same

- The autoencoder remains the primary detector.
- Input windows remain multivariate with shape `(B, T, F)`.
- FiLM conditioning still uses the context vector path.
- GPT is not used for sequence reconstruction or anomaly scoring.

## What Changed

- The previous notebook-heavy workflow has been refactored into reusable Python modules.
- The notebooks now act as clean research drivers rather than holding duplicated core logic.
- Evaluation now includes synthetic anomaly injection, early-detection metrics, and top-k feature inspection.
- A GPT adjudication layer is available after threshold crossing and returns schema-constrained JSON.

## Notebooks

### `01_build_dataset.ipynb`

- Loads the Prometheus container metrics CSV
- Cleans and `log1p` transforms the multivariate features
- Encodes context metadata
- Builds sliding windows
- Assigns train/val/test splits
- Saves reusable dataset artifacts

### `02_train_film_ae.ipynb`

- Imports the FiLM autoencoder from `container_ad_pipeline.model`
- Trains the model
- Plots train/validation loss
- Fits a validation threshold from reconstruction scores
- Saves checkpoint, scalers, and detector metadata

### `03_eval_results.ipynb`

- Loads the trained model and scalers
- Injects synthetic anomalies for controlled testing
- Computes reconstruction scores and top-k feature errors
- Evaluates precision, recall, F1, PR-AUC, ROC-AUC
- Reports strict and relaxed early-detection metrics
- Saves scored windows and top anomalous windows

### `04_gpt_adjudication.ipynb`

- Loads threshold-crossing anomalous windows
- Adds top-k feature evidence and optional recent logs
- Calls GPT through the OpenAI Responses API when an API key is present
- Falls back to a deterministic heuristic when no API key is available
- Saves adjudication CSV/JSON outputs
- Compares AE-only vs AE+GPT decisions

## Main Modules

### `container_ad_pipeline/model.py`

FiLM blocks and the PyTorch FiLM autoencoder. The legacy `src/model.py` now re-exports this implementation for compatibility.

### `container_ad_pipeline/dataset.py`

Dataset construction, context encoding, sliding windows, split assignment, and artifact saving/loading.

### `container_ad_pipeline/train.py`

Scaler fitting, PyTorch training loop, validation threshold fitting, checkpoint export, and inference helpers.

### `container_ad_pipeline/evaluate.py`

Synthetic anomaly injection, reconstruction-score inference, top-k feature extraction, classical detection metrics, and early-detection metrics.

### `container_ad_pipeline/gpt_adjudicator.py`

Post-threshold GPT adjudication using:

- anomaly score
- normalized score
- top-k anomalous features
- context metadata
- optional logs/events

The returned JSON includes:

- `label`
- `severity`
- `explanation`
- `recommended_action`

## Data and Artifacts

### Dataset outputs

Saved under `data/research_processed/`:

- `X_all.npy`
- `C_all.npy`
- `window_metadata.csv`
- `feature_meta.joblib`
- `context_encoder.joblib`
- `dataset_meta.json`
- `dataset_build_summary.json`

### Training outputs

Saved under `artifacts/research_pipeline/`:

- `film_ae.pt`
- `x_scaler.joblib`
- `c_scaler.joblib`
- `detector_meta.json`
- `detector_meta.joblib`

### Evaluation outputs

Saved under `artifacts/research_pipeline/evaluation/`:

- `window_level_predictions.csv`
- `top_anomalous_windows.csv`
- `evaluation_summary.json`
- `event_metrics.json`

### GPT outputs

Saved under `artifacts/research_pipeline/gpt/`:

- `gpt_adjudications.csv`
- `gpt_adjudications.json`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional environment setup:

```bash
copy .env.example .env
```

Populate `OPENAI_API_KEY` in `.env` if you want live GPT adjudication. If the key is missing, notebook 04 uses heuristic fallback mode automatically.

## Notes

- The current implementation uses the Prometheus-derived container metrics CSV already present in the repository.
- The notebooks were executed successfully in this workspace. On Windows, Jupyter notebook execution may require `JUPYTER_ALLOW_INSECURE_WRITES=true` unless your environment already supports secure runtime-file writes.
- The GPT layer is explicitly downstream of the autoencoder threshold and does not replace the anomaly detector.
