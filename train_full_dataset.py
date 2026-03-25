from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from container_ad_pipeline.config import PipelineConfig
from container_ad_pipeline.dataset import build_dataset_from_raw_archives, load_dataset_bundle
from container_ad_pipeline.train import train_film_autoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the processed dataset from the full raw Alibaba archives and train the FiLM autoencoder."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "research_processed_full",
        help="Directory for processed full-dataset artifacts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts") / "research_pipeline_full",
        help="Directory for trained model artifacts.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the processed dataset even if cached artifacts already exist.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. 'auto' selects CUDA when available.",
    )
    return parser.parse_args()


def configure_full_dataset(config: PipelineConfig, processed_dir: Path, artifacts_dir: Path, args: argparse.Namespace) -> PipelineConfig:
    config.paths.processed_dir = processed_dir
    config.paths.artifacts_dir = artifacts_dir
    config.paths.checkpoint_path = artifacts_dir / "film_ae.pt"
    config.paths.x_scaler_path = artifacts_dir / "x_scaler.joblib"
    config.paths.c_scaler_path = artifacts_dir / "c_scaler.joblib"
    config.paths.detector_meta_json = artifacts_dir / "detector_meta.json"
    config.paths.detector_meta_joblib = artifacts_dir / "detector_meta.joblib"
    config.paths.evaluation_dir = artifacts_dir / "evaluation"
    config.paths.adjudication_dir = artifacts_dir / "gpt"

    config.dataset.max_container_meta_files = None
    config.dataset.max_container_usage_files = None
    config.dataset.max_machine_meta_files = None
    config.dataset.max_machine_usage_files = None
    config.dataset.max_usage_rows = None
    config.dataset.max_machine_usage_rows = None
    config.dataset.max_meta_rows = None
    config.dataset.max_machine_meta_rows = None
    config.dataset.max_containers = None

    config.train.epochs = int(args.epochs)
    config.train.batch_size = int(args.batch_size)
    if args.device == "auto":
        config.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config.train.device = args.device

    return config


def processed_bundle_exists(processed_dir: Path) -> bool:
    required = [
        processed_dir / "X_all.npy",
        processed_dir / "C_all.npy",
        processed_dir / "window_metadata.csv",
        processed_dir / "feature_meta.joblib",
        processed_dir / "context_encoder.joblib",
        processed_dir / "dataset_meta.json",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    processed_dir = (project_root / args.processed_dir).resolve()
    artifacts_dir = (project_root / args.artifacts_dir).resolve()

    config = configure_full_dataset(PipelineConfig(), processed_dir, artifacts_dir, args)

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    config.paths.adjudication_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raw dataset root: {config.paths.raw_container_usage_tar.parent}")
    print(f"Processed output: {config.paths.processed_dir}")
    print(f"Artifacts output: {config.paths.artifacts_dir}")
    print(f"Training device: {config.train.device}")

    if args.rebuild_dataset or not processed_bundle_exists(config.paths.processed_dir):
        print("Building processed dataset from full raw archives...")
        bundle, saved_paths = build_dataset_from_raw_archives(config.paths, config.dataset, config.paths.processed_dir)
        print("Processed dataset saved:")
        print(json.dumps(saved_paths, indent=2))
    else:
        print("Using cached processed dataset.")
        bundle = load_dataset_bundle(config.paths.processed_dir)

    print(
        "Dataset summary: "
        f"windows={bundle.dataset_meta['num_windows']}, "
        f"window_size={bundle.dataset_meta['window_size']}, "
        f"features={bundle.dataset_meta['num_features']}, "
        f"context_dim={bundle.dataset_meta['context_dim']}"
    )

    print("Training FiLM autoencoder...")
    artifacts = train_film_autoencoder(
        bundle=bundle,
        train_config=config.train,
        checkpoint_path=config.paths.checkpoint_path,
        x_scaler_path=config.paths.x_scaler_path,
        c_scaler_path=config.paths.c_scaler_path,
        detector_meta_json=config.paths.detector_meta_json,
        detector_meta_joblib=config.paths.detector_meta_joblib,
    )

    print("Training complete.")
    print(
        json.dumps(
            {
                "checkpoint": str(config.paths.checkpoint_path),
                "x_scaler": str(config.paths.x_scaler_path),
                "c_scaler": str(config.paths.c_scaler_path),
                "detector_meta_json": str(config.paths.detector_meta_json),
                "detector_meta_joblib": str(config.paths.detector_meta_joblib),
                "threshold": artifacts.detector_meta["threshold"],
                "num_train_windows": artifacts.detector_meta["num_train_windows"],
                "num_val_windows": artifacts.detector_meta["num_val_windows"],
                "best_val_loss": artifacts.detector_meta["best_val_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
