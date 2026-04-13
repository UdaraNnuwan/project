# Container Anomaly Detection Pipeline

This branch adds a selectable multivariate anomaly stack with three modes:

- `reconstruction`: FiLM autoencoder reconstruction error
- `forecasting`: GRU forecasting error on the next timestep / short horizon
- `hybrid`: `ALPHA * recon_score + BETA * forecast_score`

GPT adjudication and Telegram alerts remain downstream of confirmed anomalies only.

## Main Config

Use `MODEL_MODE` to select the scoring path:

```powershell
$env:MODEL_MODE="reconstruction"
$env:MODEL_MODE="forecasting"
$env:MODEL_MODE="hybrid"
```

Hybrid weights are configurable:

```powershell
$env:ALPHA="0.6"
$env:BETA="0.4"
```

Forecasting horizon is configurable:

```powershell
$env:FORECAST_HORIZON="1"
```

## Notebooks

Run in this order:

1. `notebooks/01_build_dataset.ipynb`
2. `notebooks/02_train.ipynb`
3. `notebooks/03_eval.ipynb`
4. `notebooks/04_gpt.ipynb`

`notebooks/01_build_dataset.ipynb` now writes both reconstruction windows and forecasting inputs/targets.

`notebooks/02_train.ipynb` trains reconstruction, forecasting, and hybrid-ready artifacts.

`notebooks/03_eval.ipynb` compares all three modes and writes:

- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- false positive rate
- score-vs-threshold plots
- reconstruction-vs-forecast score plots
- confusion matrix / ROC / PR curves

`notebooks/04_gpt.ipynb` operates on confirmed anomalies only.

## Code Layout

- `src/model_reconstruction.py`: FiLM autoencoder
- `src/model_forecasting.py`: GRU forecaster
- `src/hybrid_scoring.py`: reconstruction / forecasting / hybrid score computation
- `src/live_infer.py`: shared artifact loader and live streaming detector
- `src/dataset.py`: dataset build plus forecasting targets
- `src/train.py`: mode-aware training and threshold fitting
- `src/evaluate.py`: three-mode evaluation, candidate/confirmed logic, GPT/Telegram hooks
- `src/gpt_adjudicator.py`: GPT payloads now include mode and component scores
- `src/telegram_utils.py`: Telegram alerts now include mode and hybrid score context
- `prometheus/direct_prometheus_infer.py`: Prometheus live path with adaptive thresholds, consecutive confirmation, GPT, and Telegram
- `product/inference_service.py`: live service path using the shared hybrid detector

## Artifacts

Training writes:

- `reconstruction_model.pt`
- `film_ae.pt` (legacy reconstruction-compatible name)
- `forecasting_model.pt`
- `x_scaler.joblib`
- `c_scaler.joblib`
- `detector_meta.json` / `detector_meta.joblib`
- `threshold_config.json`

Dataset build writes:

- `X_train.npy`, `X_test.npy`
- `C_train.npy`, `C_test.npy`
- `X_forecast_train.npy`, `X_forecast_test.npy`
- `y_forecast_train.npy`, `y_forecast_test.npy`

## Live Behavior

The live Prometheus path keeps:

- per-entity rolling dynamic thresholds
- candidate vs confirmed anomaly states
- consecutive breach confirmation
- smoothing
- cooldown suppression
- GPT calls only for confirmed anomalies
- Telegram alerts only for confirmed anomalies

Each logged decision includes mode, component scores, final score, thresholds, z-score context, volatility context, consecutive breach count, GPT status, and Telegram status.

## GPT / Telegram

Required env vars:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-4.1-mini"
$env:TELEGRAM_BOT_TOKEN="..."
$env:TELEGRAM_CHAT_ID="..."
```

GPT receives:

- entity
- mode
- score / threshold context
- top features
- decision reason

Telegram includes:

- entity
- mode
- reconstruction / forecasting / final scores when available
- threshold context
- decision reason
- GPT summary when available

## Regenerate Notebooks

```powershell
.\venv\Scripts\python.exe .\generate_research_notebooks.py
```
