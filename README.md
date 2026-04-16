# BiLSTM-FiLM Container Anomaly Detection
### Dual-Head Hybrid Reconstruction + Forecasting Architecture

A production-grade, context-aware anomaly detection system for containerised
environments, built on the Alibaba Cloud Trace 2018 dataset.

---

## Architecture

```
Input (B, W, F) + Metadata (B, M)
        │
┌───────▼──────────────────────────────┐
│         SHARED ENCODER               │
│  BiLSTM(128) → BN → BiLSTM(64)      │
│  → Linear(64) → FiLM(meta) → z      │
└───────┬──────────────┬───────────────┘
        │              │
┌───────▼──────┐ ┌─────▼────────────────┐
│  HEAD 1      │ │  HEAD 2              │
│ Reconstruction│ │  Forecasting (t+1)   │
│  (B, W, F)   │ │  (B, F)             │
└──────────────┘ └──────────────────────┘

Loss = α·MSE_recon + (1-α)·MSE_forecast   (α = 0.5)
```

## Project Structure

```
project/
├── alibaba_trace/
│   ├── model_architecture.py      # BiLSTM-FiLM dual-head model (PyTorch)
│   ├── data_pipeline.py           # Streaming dataset from .tar.gz archive
│   ├── incident_response.py       # Alerting engine (Prometheus + Telegram + GenAI RCA)
│   ├── 01_Model_Training.ipynb    # Train the dual-head model
│   ├── 02_Evaluation_and_Thesis_Graphs.ipynb  # 300 DPI thesis figures
│   ├── 03_Alerting_and_RCA_Prototype.ipynb    # Prometheus + Telegram alerts
│   ├── 04_Data_Injection_and_Accuracy_Testing.ipynb  # FiLM context validation
│   ├── 05_GenAI_RCA_Integration.ipynb         # Gemini/GPT-4 root cause analysis
│   ├── requirements.txt
│   └── outputs/                   # Saved model checkpoints + figures (git-ignored)
├── .env                           # API keys (git-ignored)
└── .gitignore
```

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r alibaba_trace/requirements.txt
```

### 2. Launch JupyterLab
```bash
python -m jupyter lab --notebook-dir=alibaba_trace
```

### 3. Run notebooks in order
| Notebook | Purpose |
|----------|---------|
| `01_Model_Training` | Train dual-head BiLSTM-FiLM → saves `outputs/dual_head_model.pt` |
| `02_Evaluation_and_Thesis_Graphs` | Score distributions, 300 DPI thesis figures |
| `03_Alerting_and_RCA_Prototype` | Prometheus webhooks + Telegram alerts |
| `04_Data_Injection_and_Accuracy_Testing` | Controlled FiLM context validation |
| `05_GenAI_RCA_Integration` | Gemini/OpenAI root cause analysis |

> **DEMO mode:** All notebooks work without the Alibaba archive.
> Synthetic normal data is generated automatically.

## Features

| Feature | Value |
|---------|-------|
| Time-series features | 7 (CPU, Mem, Net In/Out, Disk I/O, CPU/Mem Request) |
| Metadata features (FiLM) | 2 (container_id, machine_id) |
| Window size | 50 timesteps |
| Stride | 10 rows |
| Latent dimension | 64 |
| Total parameters | ~705,000 |

## Environment Variables (`.env`)
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
```
