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

        This notebook prepares one shared multivariate dataset for:
        - reconstruction windows
        - forecasting inputs/targets
        - hybrid experiments that combine both scores
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
    code_cell(
        """
        cfg = DatasetConfig(
            output_dir=DATASET_DIR,
            window_size=30,
            stride=5,
            forecast_horizon=1,
        )
        cfg.to_dict()
        """
    ),
    code_cell(
        """
        build_summary = build_dataset(cfg)
        build_summary
        """
    ),
    code_cell(
        """
        dataset_bundle = load_dataset(DATASET_DIR)
        {
            "X_train": dataset_bundle["X_train"].shape,
            "X_test": dataset_bundle["X_test"].shape,
            "X_forecast_train": dataset_bundle.get("X_forecast_train").shape if dataset_bundle.get("X_forecast_train") is not None else None,
            "y_forecast_train": dataset_bundle.get("y_forecast_train").shape if dataset_bundle.get("y_forecast_train") is not None else None,
            "feature_meta": dataset_bundle["feature_meta"],
        }
        """
    ),
    code_cell(
        """
        read_json(DATASET_DIR / "dataset_build_summary.json")
        """
    ),
]


nb02 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 02 Train

        Train the selectable anomaly models:
        - `reconstruction`
        - `forecasting`
        - `hybrid` (trains both and saves hybrid threshold metadata)
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
    code_cell(
        """
        recon_cfg = TrainConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            model_mode="reconstruction",
            epochs=10,
            batch_size=128,
            device="cpu",
        )
        recon_summary = train_model(recon_cfg)
        recon_summary["detector_meta"]["modes"]["reconstruction"]
        """
    ),
    code_cell(
        """
        forecast_cfg = TrainConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            model_mode="forecasting",
            epochs=10,
            batch_size=128,
            forecast_horizon=1,
            device="cpu",
        )
        forecast_summary = train_model(forecast_cfg)
        forecast_summary["detector_meta"]["modes"]["forecasting"]
        """
    ),
    code_cell(
        """
        hybrid_cfg = TrainConfig(
            dataset_dir=DATASET_DIR,
            model_dir=MODEL_DIR,
            model_mode="hybrid",
            epochs=10,
            batch_size=128,
            alpha=0.6,
            beta=0.4,
            forecast_horizon=1,
            device="cpu",
        )
        hybrid_summary = train_model(hybrid_cfg)
        hybrid_summary["detector_meta"]
        """
    ),
    code_cell(
        """
        pd.read_csv(MODEL_DIR / "training_history_all.csv").tail()
        """
    ),
    code_cell(
        """
        read_json(MODEL_DIR / "threshold_config.json")
        """
    ),
]


nb03 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 03 Evaluate

        Evaluate reconstruction, forecasting, and hybrid modes together and compare:
        - precision
        - recall
        - F1
        - ROC-AUC
        - PR-AUC
        - false positive rate
        """
    ),
    code_cell(
        """
        import pandas as pd

        from src.config import EvalConfig
        from src.evaluate import evaluate_model
        from src.utils import read_json
        """
    ),
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
            device="cpu",
        )
        eval_summary = evaluate_model(eval_cfg)
        eval_summary["evaluation_summary"]
        """
    ),
    code_cell(
        """
        comparison = pd.read_csv(RESULTS_DIR / "mode_comparison.csv")
        comparison
        """
    ),
    code_cell(
        """
        hybrid_stream = pd.read_csv(RESULTS_DIR / "hybrid" / "realtime_stream_predictions.csv")
        hybrid_stream[
            [
                "window_id",
                "mode",
                "recon_score",
                "forecast_score",
                "final_score",
                "final_threshold",
                "decision",
            ]
        ].head()
        """
    ),
    code_cell(
        """
        read_json(RESULTS_DIR / "hybrid" / "evaluation_summary.json")
        """
    ),
    code_cell(
        """
        read_json(RESULTS_DIR / "hybrid" / "event_metrics.json")
        """
    ),
]


nb04 = [
    code_cell(COMMON_IMPORTS),
    markdown_cell(
        """
        # 04 GPT

        GPT adjudication only runs on confirmed anomalies from the selected live/evaluation mode.
        The compact payload includes the mode, component scores, threshold context, and dominant features.
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
    code_cell(
        """
        cfg = GPTConfig(
            evaluation_dir=RESULTS_DIR,
            output_dir=RESULTS_DIR,
            max_records=20,
        )

        confirmed_alerts = pd.read_csv(RESULTS_DIR / "realtime_alert_candidates.csv")
        confirmed_alerts.head()
        """
    ),
    code_cell(
        """
        sample_payload = confirmed_alerts.iloc[0].to_dict() if len(confirmed_alerts) > 0 else {}
        sample_summary = build_window_summary(sample_payload) if sample_payload else {}
        sample_summary
        """
    ),
    code_cell(
        """
        if sample_summary:
            sample_decision, sample_meta = call_openai_responses_api(sample_summary, cfg)
        else:
            sample_decision, sample_meta = {}, {}

        sample_meta, sample_decision
        """
    ),
    code_cell(
        """
        adjudication_summary = adjudicate_anomaly_records(
            prediction_csv_path=RESULTS_DIR / "window_level_predictions.csv",
            config=cfg,
            max_records=cfg.max_records,
        )
        adjudication_summary
        """
    ),
    code_cell(
        """
        read_json(RESULTS_DIR / "ae_vs_gpt_comparison.json")
        """
    ),
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
