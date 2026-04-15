from __future__ import annotations

import json
from pathlib import Path
import textwrap


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


COMMON_IMPORTS = """
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(".."))
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(os.path.abspath("."))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATASET_DIR = (PROJECT_ROOT / "data" / "research_processed_smoke_auto").resolve()
MODEL_DIR = (PROJECT_ROOT / "models" / "research_smoke_auto").resolve()
RESULTS_DIR = (PROJECT_ROOT / "results" / "research_smoke_auto").resolve()
"""

nb01 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 01 Build Dataset

        This notebook prepares one shared multivariate dataset for reconstruction and forecasting. 
        It loads raw node/container features and creates normalized rolling windows.
        """
    ),
    code_cell(
        """
        import pandas as pd
        from src.config import DatasetConfig
        from src.dataset import build_dataset, load_dataset
        from src.utils import read_json
        """
    ),
    markdown_cell(
        """
        ### Configure Dataset
        We initialize the dataset config. Using defaults ensures it reads standard metrics from your `.env` seamlessly!
        """
    ),
    code_cell(
        """
        # Removed hardcoded parameters to get static metrics values directly from .env config!
        cfg = DatasetConfig(output_dir=DATASET_DIR)
        cfg_dict = cfg.to_dict()
        for k, v in cfg_dict.items():
            if not str(k).endswith('_columns'):
                print(f"{k}: {v}")
        """
    ),
    markdown_cell("### Build and Load Dataset"),
    code_cell(
        """
        build_summary = build_dataset(cfg)
        print("Build Summary:", build_summary)
        """
    ),
    code_cell(
        """
        dataset_bundle = load_dataset(DATASET_DIR)
        print("Dataset Shapes:")
        print("X_train (Features):", dataset_bundle["X_train"].shape)
        print("X_test (Features):", dataset_bundle["X_test"].shape)
        if dataset_bundle.get("X_forecast_train") is not None:
             print("X_forecast_train:", dataset_bundle["X_forecast_train"].shape)
        """
    ),
    markdown_cell("### Inspection Summary"),
    code_cell(
        """
        summary_df = pd.Series(read_json(DATASET_DIR / "dataset_build_summary.json")).to_frame("Value")
        display(summary_df)
        """
    ),
]


nb02 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 02 Train

        Trains the core anomaly detection models:
        - `reconstruction`: A FiLM autoencoder mapping context variables to detect unexpected signals.
        - `forecasting`: Predicting next frames.
        - `hybrid`: Combines both approaches via a weighted score.
        """
    ),
    code_cell(
        """
        import pandas as pd
        from src.config import TrainConfig
        from src.train import train_model
        from src.utils import read_json
        """
    ),
    markdown_cell("### Train Reconstruction Model (FiLM Autoencoder)"),
    code_cell(
        """
        recon_cfg = TrainConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            model_mode="reconstruction",
            device="cuda" # Changed from cpu to setup for GPU accel if accessible
        )
        recon_summary = train_model(recon_cfg)
        print("Model saved to:", recon_summary['model_dir'])
        print("Artifact maps:", recon_summary['artifact_paths'])
        display(recon_summary["detector_meta"]["modes"]["reconstruction"])
        """
    ),
    markdown_cell("### Train Forecasting Model"),
    code_cell(
        """
        forecast_cfg = TrainConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            model_mode="forecasting",
            device="cuda"
        )
        forecast_summary = train_model(forecast_cfg)
        print("Model saved to:", forecast_summary['model_dir'])
        """
    ),
    markdown_cell("### Train Hybrid Threshold"),
    code_cell(
        """
        hybrid_cfg = TrainConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            model_mode="hybrid",
            device="cuda"
        )
        hybrid_summary = train_model(hybrid_cfg)
        display(hybrid_summary["detector_meta"])
        """
    ),
    markdown_cell("### Training History Visualization"),
    code_cell(
        """
        import matplotlib.pyplot as plt
        history = pd.read_csv(MODEL_DIR / "training_history_all.csv")
        display(history.tail())
        
        plt.figure(figsize=(10,4))
        for mode in history['model_name'].unique():
            subset = history[history['model_name'] == mode]
            plt.plot(subset['epoch'], subset['val_loss'], label=mode + " val_loss")
            plt.plot(subset['epoch'], subset['train_loss'], label=mode + " train_loss", linestyle='--')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.show()
        """
    ),
    code_cell(
        """
        print("Threshold Configuration:")
        display(read_json(MODEL_DIR / "threshold_config.json"))
        """
    ),
]


nb03 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 03 Evaluate

        Evaluates the trained models, runs the mathematical dynamic threshold validation, and creates visual inspection plots for anomalies.
        """
    ),
    code_cell(
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from src.config import EvalConfig
        from src.evaluate import evaluate_model
        from src.utils import read_json
        """
    ),
    markdown_cell("### Run Base Evaluation"),
    code_cell(
        """
        eval_cfg = EvalConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            output_dir=RESULTS_DIR,
            model_mode="hybrid",
            eval_modes=("reconstruction", "forecasting", "hybrid"),
            split="test",
            use_synthetic_injection=True,
            include_gpt_in_stream=False,
            device="cuda",
        )
        eval_summary = evaluate_model(eval_cfg)
        display(eval_summary["evaluation_summary"])
        """
    ),
    markdown_cell("### Dynamic Threshold Analysis\nUsing `.env` populated configurations, compute standard metrics."),
    code_cell(
        """
        hybrid_stream = pd.read_csv(RESULTS_DIR / "hybrid" / "realtime_stream_predictions.csv")
        
        # Make sure dynamic threshold uses the exact anomaly_score variable (final_score)
        anomaly_score = hybrid_stream['final_score'].values
        
        dyn_percentile = eval_cfg.dynamic_threshold_percentile
        z_threshold = eval_cfg.z_score_threshold
        history_limit = eval_cfg.dynamic_threshold_history_limit
        
        dynamic_thresholds = np.zeros_like(anomaly_score)
        z_scores = np.zeros_like(anomaly_score)
        candidate_flags = np.zeros_like(anomaly_score, dtype=bool)

        for i in range(len(anomaly_score)):
            start_idx = max(0, i - history_limit)
            history = anomaly_score[start_idx:i]
            
            if len(history) > 5:
                dyn_thresh = np.percentile(history, dyn_percentile)
                mean_hist = np.mean(history)
                std_hist = np.std(history) + 1e-8
                z = (anomaly_score[i] - mean_hist) / std_hist
            else:
                dyn_thresh = anomaly_score[i]
                z = 0.0
                
            dynamic_thresholds[i] = dyn_thresh
            z_scores[i] = z
            candidate_flags[i] = (anomaly_score[i] > dyn_thresh) or (z > z_threshold)

        hybrid_stream['calculated_dynamic_threshold'] = dynamic_thresholds
        hybrid_stream['calculated_z_score'] = z_scores
        hybrid_stream['is_candidate'] = candidate_flags
        
        # Save evaluation outputs
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "hybrid").mkdir(parents=True, exist_ok=True)
        hybrid_stream.to_csv(RESULTS_DIR / "hybrid" / "realtime_stream_predictions_enhanced.csv", index=False)
        display(hybrid_stream[['window_id', 'final_score', 'calculated_dynamic_threshold', 'calculated_z_score', 'is_candidate']].head())
        """
    ),
    markdown_cell("### Visualization: Score Distributions"),
    code_cell(
        """
        plt.figure(figsize=(12, 5))
        sns.histplot(hybrid_stream['final_score'], bins=50, kde=True, color='blue', alpha=0.6)
        plt.axvline(hybrid_stream['final_threshold'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Final Threshold')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.savefig(RESULTS_DIR / 'hybrid' / 'anomaly_score_distribution.png')
        plt.show()
        """
    ),
    markdown_cell("### Visualization: Top Anomalous Windows"),
    code_cell(
        """
        top_anomalies = hybrid_stream.sort_values(by='final_score', ascending=False).head(10)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_anomalies['window_id'].astype(str), y=top_anomalies['final_score'], palette='Reds_r')
        plt.title('Top 10 Anomalous Windows by Final Score')
        plt.xticks(rotation=45)
        plt.savefig(RESULTS_DIR / 'hybrid' / 'top_anomalous_windows.png')
        plt.show()
        """
    ),
    markdown_cell("### Visualization: Paper-Style Time Series & Thresholds"),
    code_cell(
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Subplot 1: Components
        axes[0].plot(hybrid_stream['window_id'], hybrid_stream['recon_score'], label='Reconstruction Score', alpha=0.7)
        axes[0].plot(hybrid_stream['window_id'], hybrid_stream['forecast_score'], label='Forecast Score', alpha=0.7)
        axes[0].set_title('Component Anomaly Scores Over Time')
        axes[0].legend()
        axes[0].grid(True)
        
        # Subplot 2: Dynamic vs Final
        axes[1].plot(hybrid_stream['window_id'], hybrid_stream['final_score'], label='Final Anomaly Score', color='purple')
        axes[1].plot(hybrid_stream['window_id'], hybrid_stream['calculated_dynamic_threshold'], label='Dynamic Threshold', color='orange', linestyle='--')
        axes[1].plot(hybrid_stream['window_id'], hybrid_stream['final_threshold'], label='Static Threshold', color='red', linestyle=':')
        
        # Highlight candidates
        candidate_indices = hybrid_stream[hybrid_stream['is_candidate']]['window_id']
        axes[1].scatter(candidate_indices, hybrid_stream[hybrid_stream['is_candidate']]['final_score'], color='red', marker='x', label='Candidate Flags')
        
        axes[1].set_title('Final Score vs Dynamic/Static Thresholds')
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'hybrid' / 'paper_style_timeseries.png')
        plt.show()
        """
    )
]


nb04 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 04 GPT Adjudication

        This workflow delegates the **final semantic decision layer** to GPT.
        
        * GPT *only* processes anomaly candidates (not all windows) to reduce cost and noise.
        * If the GPT label == `normal`, it marks the incident as `skipped_for_alerting`.
        """
    ),
    code_cell(
        """
        import pandas as pd
        from src.config import GPTConfig
        from src.gpt_adjudicator import adjudicate_anomaly_records, build_window_summary, call_openai_responses_api
        from src.utils import read_json
        """
    ),
    markdown_cell("### Extract Candidates"),
    code_cell(
        """
        cfg = GPTConfig(
            evaluation_dir=RESULTS_DIR,
            output_dir=RESULTS_DIR,
            max_records=20, # Evaluates max top 20 verified candidates
        )

        try:
            confirmed_alerts = pd.read_csv(RESULTS_DIR / "realtime_alert_candidates.csv")
        except FileNotFoundError:
            # Fallback if realtime alerts missing, get from streaming enriched.
            stream = pd.read_csv(RESULTS_DIR / "hybrid" / "realtime_stream_predictions_enhanced.csv")
            confirmed_alerts = stream[stream['is_candidate'] == True].copy()
            
        print(f"Loaded {len(confirmed_alerts)} anomaly candidates for GPT processing.")
        """
    ),
    markdown_cell("### Adjudication Flow"),
    code_cell(
        """
        adjudication_summary = adjudicate_anomaly_records(
            prediction_csv_path=RESULTS_DIR / "hybrid" / "realtime_stream_predictions.csv", # Update logic path
            config=cfg,
            max_records=cfg.max_records,
        )
        """
    ),
    markdown_cell("### Results Preview Table\nSkip standard routing if GPT believes the state is 'normal'."),
    code_cell(
        """
        try:
            gpt_results = pd.read_csv(RESULTS_DIR / "gpt_adjudicated_alerts.csv")
            # Apply logic: if GPT label == normal, mark as skipped
            is_normal = gpt_results['gpt_label'].str.lower().str.contains('normal', na=False)
            gpt_results['skipped_for_alerting'] = is_normal
            
            # Save the updated frame
            gpt_results.to_csv(RESULTS_DIR / "gpt_adjudicated_alerts.csv", index=False)
            
            display_cols = ['window_id', 'gpt_label', 'skipped_for_alerting', 'gpt_reason']
            display_cols = [c for c in display_cols if c in gpt_results.columns]
            
            print("GPT Decisions Preview:")
            display(gpt_results[display_cols].head(10))
        except FileNotFoundError:
            print("No GPT records generated. Check API Key or candidate limits.")
        """
    )
]


nb00 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 00 Quickstart Notes

        Notebook order:
        1. `01_build_dataset.ipynb`
        2. `02_train.ipynb`
        3. `03_eval.ipynb`
        4. `04_gpt.ipynb`
        """
    ),
]


def main() -> None:
    write_notebook(NOTEBOOK_DIR / "00_quickstart_smoke.ipynb", nb00)
    write_notebook(NOTEBOOK_DIR / "01_build_dataset.ipynb", nb01)
    write_notebook(NOTEBOOK_DIR / "02_train.ipynb", nb02)
    write_notebook(NOTEBOOK_DIR / "03_eval.ipynb", nb03)
    write_notebook(NOTEBOOK_DIR / "04_gpt.ipynb", nb04)


if __name__ == "__main__":
    main()
