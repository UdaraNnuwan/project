from __future__ import annotations

import json
from pathlib import Path
import textwrap


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
QUICKSTART_DATASET_DIR = PROJECT_ROOT / "data" / "smoke_processed"
QUICKSTART_MODEL_DIR = PROJECT_ROOT / "models" / "smoke"
QUICKSTART_RESULTS_DIR = PROJECT_ROOT / "results" / "smoke"
RESEARCH_DATASET_DIR = PROJECT_ROOT / "data" / "research_processed_smoke_auto"
RESEARCH_MODEL_DIR = PROJECT_ROOT / "models" / "research_smoke_auto"
RESEARCH_RESULTS_DIR = PROJECT_ROOT / "results" / "research_smoke_auto"


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


COMMON_IMPORT_CELL = """
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(".."))
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(os.path.abspath("."))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
"""


QUICKSTART_PATHS_CELL = f"""
SMOKE_DATASET_DIR = (PROJECT_ROOT / "data" / "smoke_processed").resolve()
SMOKE_MODEL_DIR = (PROJECT_ROOT / "models" / "smoke").resolve()
SMOKE_RESULTS_DIR = (PROJECT_ROOT / "results" / "smoke").resolve()
"""


RESEARCH_PATHS_CELL = f"""
RESEARCH_DATASET_DIR = (PROJECT_ROOT / "data" / "research_processed_smoke_auto").resolve()
RESEARCH_MODEL_DIR = (PROJECT_ROOT / "models" / "research_smoke_auto").resolve()
RESEARCH_RESULTS_DIR = (PROJECT_ROOT / "results" / "research_smoke_auto").resolve()
"""


nb00 = [
    code_cell(COMMON_IMPORT_CELL),
    code_cell(QUICKSTART_PATHS_CELL),
    markdown_cell(
        """
        # 00 Quickstart Smoke Run

        This is the fastest notebook entry point in the repository.
        It uses the tiny synthetic smoke dataset under `data/smoke_processed/`,
        retrains the FiLM autoencoder, and writes evaluation outputs under `results/smoke/`.
        """
    ),
    code_cell(
        """
        import pandas as pd

        from src.dataset import load_dataset
        from src.config import EvalConfig, TrainConfig
        from src.evaluate import evaluate_model
        from src.train import train_model
        from src.utils import read_json
        """
    ),
    markdown_cell("## Confirm The Small Smoke Dataset"),
    code_cell(
        """
        smoke_bundle = load_dataset(SMOKE_DATASET_DIR)
        {
            "X_shape": smoke_bundle["X"].shape,
            "C_shape": smoke_bundle["C"].shape,
            "split_counts": smoke_bundle["metadata"]["split"].value_counts().sort_index().to_dict(),
            "feature_columns": smoke_bundle["feature_meta"]["feature_columns"],
        }
        """
    ),
    markdown_cell("## Train A Notebook-Friendly Baseline"),
    code_cell(
        """
        train_cfg = TrainConfig(
            dataset_dir=SMOKE_DATASET_DIR,
            model_dir=SMOKE_MODEL_DIR,
            epochs=8,
            batch_size=4,
            patience=4,
            units=8,
            latent=4,
            device="cpu",
        )
        train_summary = train_model(train_cfg)
        train_summary["detector_meta"]
        """
    ),
    code_cell(
        """
        history = pd.read_csv(SMOKE_MODEL_DIR / "training_history.csv")
        history
        """
    ),
    markdown_cell("## Evaluate The Trained Smoke Model"),
    code_cell(
        """
        eval_cfg = EvalConfig(
            dataset_dir=SMOKE_DATASET_DIR,
            model_dir=SMOKE_MODEL_DIR,
            output_dir=SMOKE_RESULTS_DIR,
            split="test",
            top_k_features=3,
            use_synthetic_injection=False,
            include_gpt_in_stream=False,
            device="cpu",
        )
        eval_summary = evaluate_model(eval_cfg)
        eval_summary["evaluation_summary"]
        """
    ),
    code_cell(
        """
        predictions = pd.read_csv(SMOKE_RESULTS_DIR / "window_level_predictions.csv")
        predictions[
            [
                "window_id",
                "container_id",
                "anomaly_score",
                "top_k_features",
                "predicted_label",
            ]
        ].head()
        """
    ),
    code_cell(
        """
        read_json(SMOKE_RESULTS_DIR / "evaluation_summary.json")
        """
    ),
]


nb01 = [
    code_cell(COMMON_IMPORT_CELL),
    code_cell(RESEARCH_PATHS_CELL),
    markdown_cell(
        """
        # 01 Build Dataset

        This notebook builds the multivariate FiLM autoencoder dataset directly from the raw Alibaba archives:
        - `container_meta.tar.gz`
        - `container_usage.tar.gz`
        - `machine_meta.tar.gz`
        - `machine_usage.tar.gz`

        `container_usage` is the main multivariate time-series source. Container metadata and machine-level data are used as context.
        """
    ),
    code_cell(
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        from src.config import DatasetConfig
        from src.dataset import (
            CONTAINER_META_COLUMNS,
            MACHINE_META_COLUMNS,
            MACHINE_USAGE_COLUMNS,
            ContextEncoder,
            align_usage_with_context,
            build_dataset,
            clean_container_meta,
            clean_machine_meta,
            clean_machine_usage,
            generate_sliding_windows,
            inspect_raw_archives,
            load_filtered_archive,
            load_container_usage_subset,
        )
        from src.utils import read_json
        """
    ),
    markdown_cell("## Configuration"),
    code_cell(
        """
        cfg = DatasetConfig(
            raw_data_dir=DatasetConfig().raw_data_dir,
            output_dir=RESEARCH_DATASET_DIR,
            container_limit=None,
            max_usage_rows=10_000_000,
            max_container_meta_rows=None,
            max_machine_usage_rows=10_000_000,
            max_machine_meta_rows=None,
            window_size=30,
            stride=5,
        )
        cfg.to_dict()
        """
    ),
    markdown_cell("## Raw Data Loading From tar.gz"),
    code_cell(
        """
        archive_status = {
            name: path.exists()
            for name, path in cfg.archive_paths.items()
        }
        archive_status
        """
    ),
    code_cell(
        """
        raw_schema = inspect_raw_archives(cfg)
        raw_schema["container_usage"]
        """
    ),
    markdown_cell("## Schema Inspection"),
    code_cell(
        """
        pd.DataFrame(
            {
                "archive": list(raw_schema.keys()),
                "columns": [schema["columns"] for schema in raw_schema.values()],
                "member": [schema["member"] for schema in raw_schema.values()],
            }
        )
        """
    ),
    markdown_cell("## Preprocessing"),
    code_cell(
        """
        usage_df, container_ids, machine_ids = load_container_usage_subset(cfg)

        container_meta_df = load_filtered_archive(
            cfg.archive_paths["container_meta"],
            column_names=CONTAINER_META_COLUMNS,
            chunksize=cfg.chunksize,
            cleaner=clean_container_meta,
            filter_column="container_id",
            allowed_ids=container_ids,
            max_rows=cfg.max_container_meta_rows,
        )
        machine_usage_df = load_filtered_archive(
            cfg.archive_paths["machine_usage"],
            column_names=MACHINE_USAGE_COLUMNS,
            chunksize=cfg.chunksize,
            cleaner=clean_machine_usage,
            filter_column="machine_id",
            allowed_ids=machine_ids,
            max_rows=cfg.max_machine_usage_rows,
        )
        machine_meta_df = load_filtered_archive(
            cfg.archive_paths["machine_meta"],
            column_names=MACHINE_META_COLUMNS,
            chunksize=cfg.chunksize,
            cleaner=clean_machine_meta,
            filter_column="machine_id",
            allowed_ids=machine_ids,
            max_rows=cfg.max_machine_meta_rows,
        )

        aligned_df = align_usage_with_context(
            usage=usage_df,
            container_meta=container_meta_df,
            machine_usage=machine_usage_df,
            machine_meta=machine_meta_df,
        )

        print("container_usage rows:", len(usage_df))
        print("container_meta rows:", len(container_meta_df))
        print("machine_usage rows:", len(machine_usage_df))
        print("machine_meta rows:", len(machine_meta_df))
        aligned_df.head()
        """
    ),
    markdown_cell("## Feature Selection"),
    code_cell(
        """
        print("Feature columns:")
        print(list(cfg.feature_columns))

        print("\\nContext columns:")
        print(list(cfg.context_columns))
        """
    ),
    markdown_cell("## Context Creation"),
    code_cell(
        """
        encoder = ContextEncoder.fit(
            aligned_df,
            numeric_columns=list(cfg.container_context_numeric + cfg.machine_context_numeric),
            categorical_columns=list(cfg.container_context_categorical + cfg.machine_context_categorical),
        )
        encoder.to_metadata()
        """
    ),
    markdown_cell("## Sliding Window Generation"),
    code_cell(
        """
        X_preview, C_preview, window_metadata_preview = generate_sliding_windows(
            aligned_df=aligned_df,
            config=cfg,
            context_encoder=encoder,
        )

        print("X shape:", X_preview.shape)
        print("C shape:", C_preview.shape)
        window_metadata_preview.head()
        """
    ),
    markdown_cell("## Train/Val/Test Split"),
    code_cell(
        """
        split_counts = window_metadata_preview["split"].value_counts().sort_index()
        split_counts
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(6, 3))
        split_counts.plot(kind="bar")
        plt.title("Window Split Distribution")
        plt.ylabel("Count")
        plt.show()
        """
    ),
    markdown_cell("## Save Artifacts To data/"),
    code_cell(
        """
        build_summary = build_dataset(cfg)
        read_json(cfg.output_dir / "dataset_build_summary.json")
        """
    ),
]


nb02 = [
    code_cell(COMMON_IMPORT_CELL),
    code_cell(RESEARCH_PATHS_CELL),
    markdown_cell(
        """
        # 02 Train

        This notebook trains the FiLM-conditioned autoencoder on the research smoke dataset
        and saves checkpoints under `models/research_smoke_auto/`.
        """
    ),
    code_cell(
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import torch

        from src.config import TrainConfig
        from src.dataset import load_dataset
        from src.model import FiLMAutoencoder
        from src.train import load_processed_dataset, split_by_metadata, train_model
        from src.utils import read_json
        """
    ),
    markdown_cell("## Load Processed Dataset"),
    code_cell(
        """
        cfg = TrainConfig(
            dataset_dir=RESEARCH_DATASET_DIR,
            model_dir=RESEARCH_MODEL_DIR,
            epochs=20,
            batch_size=128,
            patience=8,
        )
        dataset_bundle = load_processed_dataset(cfg.dataset_dir)
        dataset_bundle["X"].shape, dataset_bundle["C"].shape
        """
    ),
    code_cell(
        """
        split_bundle = split_by_metadata(
            dataset_bundle["X"],
            dataset_bundle["C"],
            dataset_bundle["metadata"],
        )
        {
            split_name: {
                "X": split_data[0].shape,
                "C": split_data[1].shape,
                "rows": len(split_data[2]),
            }
            for split_name, split_data in split_bundle.items()
        }
        """
    ),
    markdown_cell("## Initialize FiLM Autoencoder"),
    code_cell(
        """
        x_train, c_train, _ = split_bundle["train"]
        model = FiLMAutoencoder(
            window_size=x_train.shape[1],
            n_features=x_train.shape[2],
            context_dim=c_train.shape[1],
            units=cfg.units,
            latent=cfg.latent,
        )
        model
        """
    ),
    markdown_cell("## Training Loop, Validation Loop, And Checkpoint Save"),
    code_cell(
        """
        train_summary = train_model(cfg)
        train_summary
        """
    ),
    markdown_cell("## Plot Losses"),
    code_cell(
        """
        history = pd.read_csv(cfg.model_dir / "training_history.csv")
        history.tail()
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(7, 4))
        plt.plot(history["epoch"], history["train_loss"], label="train")
        plt.plot(history["epoch"], history["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("FiLM Autoencoder Training History")
        plt.legend()
        plt.show()
        """
    ),
    markdown_cell("## Threshold Fitting"),
    code_cell(
        """
        detector_meta = read_json(cfg.model_dir / "detector_meta.json")
        detector_meta
        """
    ),
]


nb03 = [
    code_cell(COMMON_IMPORT_CELL),
    code_cell(RESEARCH_PATHS_CELL),
    markdown_cell(
        """
        # 03 Evaluate

        This notebook loads the trained model, runs sequential inference, computes anomaly scores and reconstruction errors, evaluates early detection, and saves outputs under `results/research_smoke_auto/`.
        """
    ),
    code_cell(
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        from src.config import EvalConfig
        from src.evaluate import evaluate_model
        from src.utils import read_json
        """
    ),
    markdown_cell("## Sequential Inference And Anomaly Score Calculation"),
    code_cell(
        """
        cfg = EvalConfig(
            dataset_dir=RESEARCH_DATASET_DIR,
            model_dir=RESEARCH_MODEL_DIR,
            output_dir=RESEARCH_RESULTS_DIR,
            split="test",
            top_k_features=5,
            use_synthetic_injection=True,
            include_gpt_in_stream=False,
        )
        eval_summary = evaluate_model(cfg)
        eval_summary["evaluation_summary"]
        """
    ),
    code_cell(
        """
        predictions = pd.read_csv(cfg.output_dir / "window_level_predictions.csv")
        predictions.head()
        """
    ),
    markdown_cell("## Feature-Wise Reconstruction Error And Top-k Anomalous Features"),
    code_cell(
        """
        predictions[
            [
                "window_id",
                "container_id",
                "anomaly_score",
                "top_k_features",
                "top_k_feature_errors",
            ]
        ].sort_values("anomaly_score", ascending=False).head(10)
        """
    ),
    markdown_cell("## Precision, Recall, F1, PR-AUC, And ROC-AUC"),
    code_cell(
        """
        evaluation_summary = read_json(cfg.output_dir / "evaluation_summary.json")
        evaluation_summary
        """
    ),
    markdown_cell("## Strict And Relaxed Early Detection Evaluation"),
    code_cell(
        """
        event_metrics = read_json(cfg.output_dir / "event_metrics.json")
        event_metrics
        """
    ),
    markdown_cell("## Streaming-Style Inference Flow"),
    code_cell(
        """
        realtime_stream = pd.read_csv(cfg.output_dir / "realtime_stream_predictions.csv")
        realtime_stream.head()
        """
    ),
    markdown_cell("## Visualizations"),
    code_cell(
        """
        plt.figure(figsize=(7, 4))
        predictions["anomaly_score"].plot(kind="hist", bins=40)
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Score")
        plt.show()
        """
    ),
    code_cell(
        """
        top_windows = pd.read_csv(cfg.output_dir / "top_anomalous_windows.csv")
        plt.figure(figsize=(8, 4))
        plt.plot(top_windows["anomaly_score"].head(20).to_numpy())
        plt.title("Top Anomalous Window Scores")
        plt.xlabel("Rank")
        plt.ylabel("Anomaly score")
        plt.show()
        """
    ),
]


nb04 = [
    code_cell(COMMON_IMPORT_CELL),
    code_cell(RESEARCH_PATHS_CELL),
    markdown_cell(
        """
        # 04 GPT

        This notebook loads threshold-crossing anomaly candidates, builds compact GPT inputs, calls the OpenAI Responses API, parses structured JSON output, and compares AE-only vs AE+GPT decisions.

        GPT is only used after anomaly score threshold crossing. The FiLM autoencoder remains the anomaly detector.
        """
    ),
    code_cell(
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        from src.config import GPTConfig
        from src.gpt_adjudicator import (
            adjudicate_anomaly,
            adjudicate_anomaly_records,
            build_window_summary,
            call_openai_responses_api,
            compare_ae_vs_gpt_decisions,
        )
        from src.utils import read_json
        """
    ),
    markdown_cell("## Load Anomalous Windows Or Evaluation Results"),
    code_cell(
        """
        cfg = GPTConfig(
            evaluation_dir=RESEARCH_RESULTS_DIR,
            output_dir=RESEARCH_RESULTS_DIR,
            max_records=100,
        )

        alerts = pd.read_csv(cfg.evaluation_dir / "realtime_alert_candidates.csv")
        alerts = alerts.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        alerts.head()
        """
    ),
    markdown_cell("## Build Compact GPT Input"),
    code_cell(
        """
        sample_payload = alerts.iloc[0].to_dict() if len(alerts) > 0 else {}
        sample_summary = build_window_summary(sample_payload) if sample_payload else {}
        sample_summary
        """
    ),
    markdown_cell("## Call OpenAI Responses API"),
    code_cell(
        """
        if sample_summary:
            sample_decision, sample_meta = call_openai_responses_api(sample_summary, cfg)
        else:
            sample_decision, sample_meta = {}, {}

        sample_meta, sample_decision
        """
    ),
    markdown_cell("## Parse Structured JSON Output"),
    code_cell(
        """
        sample_decision
        """
    ),
    markdown_cell("## Compare AE-only Vs AE+GPT Decisions"),
    code_cell(
        """
        adjudication_summary = adjudicate_anomaly_records(
            prediction_csv_path=cfg.evaluation_dir / "window_level_predictions.csv",
            config=cfg,
            max_records=cfg.max_records,
        )
        adjudication_summary
        """
    ),
    code_cell(
        """
        adjudications = pd.read_csv(cfg.output_dir / "gpt_adjudications.csv")
        comparison = pd.read_csv(cfg.output_dir / "ae_vs_gpt_comparison.csv")

        adjudications.head(), comparison.head()
        """
    ),
    markdown_cell("## Visualize GPT Decisions"),
    code_cell(
        """
        if len(adjudications) > 0:
            plt.figure(figsize=(6, 3))
            adjudications["severity"].value_counts().plot(kind="bar")
            plt.title("GPT Severity Distribution")
            plt.ylabel("Count")
            plt.show()

        read_json(cfg.output_dir / "ae_vs_gpt_comparison.json")
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
