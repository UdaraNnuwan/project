from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "data"

DEFAULT_FEATURE_COLUMNS = [
    "cpu_util",
    "mem_util",
    "cpi",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io",
]

DEFAULT_PROM_FEATURE_COLUMNS = [
    "cpu_util",
    "mem_util",
    "net_in",
    "net_out",
    "disk_read",
    "disk_write",
    "mem_rss",
    "mem_cache",
]

DEFAULT_CONTEXT_COLUMNS = ["namespace", "pod", "container"]
DEFAULT_ALIBABA_CONTEXT_NUMERIC_COLUMNS = [
    "container_cpu_request",
    "container_cpu_limit",
    "container_mem_size",
    "machine_cpu_num",
    "machine_mem_size",
    "machine_cpu_util",
    "machine_mem_util",
    "machine_mem_gps",
    "machine_mpki",
    "machine_net_in",
    "machine_net_out",
    "machine_disk_io",
]
DEFAULT_ALIBABA_CONTEXT_CATEGORICAL_COLUMNS = [
    "container_app_du",
    "container_status",
    "machine_failure_domain_1",
    "machine_failure_domain_2",
    "machine_status",
]


@dataclass
class PathConfig:
    raw_container_meta_tar: Path = DATASET_ROOT / "container_meta.tar.gz"
    raw_container_usage_tar: Path = DATASET_ROOT / "container_usage.tar.gz"
    raw_machine_meta_tar: Path = DATASET_ROOT / "machine_meta.tar.gz"
    raw_machine_usage_tar: Path = DATASET_ROOT / "machine_usage.tar.gz"
    raw_metrics_csv: Path = PROJECT_ROOT / "prometheus" / "prometheus_timeseries_metrics.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "research_processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "research_pipeline"
    checkpoint_path: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "film_ae.pt"
    x_scaler_path: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "x_scaler.joblib"
    c_scaler_path: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "c_scaler.joblib"
    detector_meta_json: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "detector_meta.json"
    detector_meta_joblib: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "detector_meta.joblib"
    evaluation_dir: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "evaluation"
    adjudication_dir: Path = PROJECT_ROOT / "artifacts" / "research_pipeline" / "gpt"


@dataclass
class DatasetConfig:
    source: str = "alibaba_raw_archives"
    timestamp_column: str = "time_stamp"
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    context_numeric_columns: list[str] = field(default_factory=lambda: list(DEFAULT_ALIBABA_CONTEXT_NUMERIC_COLUMNS))
    context_categorical_columns: list[str] = field(default_factory=lambda: list(DEFAULT_ALIBABA_CONTEXT_CATEGORICAL_COLUMNS))
    context_columns: list[str] = field(
        default_factory=lambda: list(DEFAULT_ALIBABA_CONTEXT_NUMERIC_COLUMNS) + list(DEFAULT_ALIBABA_CONTEXT_CATEGORICAL_COLUMNS)
    )
    entity_columns: list[str] = field(default_factory=lambda: ["container_id"])
    window_size: int = 24
    stride: int = 6
    min_points_per_entity: int = 24
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    log1p_transform: bool = True
    clip_min_value: float = 0.0
    impute_strategy: str = "ffill_bfill_zero"
    chunksize: int = 100_000
    max_container_meta_files: int | None = 10
    max_container_usage_files: int | None = 10
    max_machine_meta_files: int | None = None
    max_machine_usage_files: int | None = 10
    max_usage_rows: int = 300_000_000
    max_machine_usage_rows: int = 600_000_000
    max_meta_rows: int | None = None
    max_machine_meta_rows: int | None = None
    max_containers: int = 2_000
    use_machine_usage_context: bool = True
    use_machine_meta_context: bool = True


@dataclass
class TrainConfig:
    random_seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 80
    patience: int = 12
    units: int = 64
    latent: int = 16
    threshold_quantile: float = 0.995
    num_workers: int = 0
    device: str = "cuda"


@dataclass
class EvalConfig:
    anomaly_ratio: float = 0.10
    anomaly_seed: int = 7
    top_k_features: int = 5
    top_n_windows: int = 20
    relaxed_detection_tolerance: int = 3
    save_predictions_filename: str = "window_level_predictions.csv"
    save_summary_filename: str = "evaluation_summary.json"


@dataclass
class GPTConfig:
    enabled: bool = True
    api_key_env: str = "OPENAI_API_KEY"
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    max_output_tokens: int = 500
    temperature: float = 0.1
    top_k_features: int = 5
    prompt_template_path: Path = PROJECT_ROOT / "container_ad_pipeline" / "prompt_template.txt"
    schema_path: Path = PROJECT_ROOT / "container_ad_pipeline" / "schemas" / "gpt_adjudication_schema.json"


@dataclass
class RealtimeConfig:
    prom_url: str = "http://35.206.92.147:9090/api/v1/query"
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_PROM_FEATURE_COLUMNS))
    poll_interval_seconds: int = 30
    query_timeout_seconds: int = 20
    warmup_windows: int = 20
    top_k_features: int = 3
    target_namespace: str | None = None
    target_pod: str | None = None
    target_container: str | None = None
    raw_snapshot_csv: Path = PROJECT_ROOT / "product" / "raw_snapshots.csv"
    log_file: Path = PROJECT_ROOT / "product" / "result_dual_status.log"
    legacy_model_dir: Path = PROJECT_ROOT / "saved_model"
    legacy_model_path: Path = PROJECT_ROOT / "saved_model" / "ae_model.pt"
    legacy_x_scaler_path: Path = PROJECT_ROOT / "saved_model" / "scaler.joblib"
    legacy_c_scaler_path: Path = PROJECT_ROOT / "saved_model" / "ctx_scaler.joblib"
    legacy_detector_meta_path: Path = PROJECT_ROOT / "saved_model" / "detector_meta.joblib"
    legacy_ae_meta_path: Path = PROJECT_ROOT / "saved_model" / "ae_model_meta.joblib"
    all_anomaly_consecutive_hits: int = 3
    all_clear_consecutive_normals: int = 3
    all_score_history_size: int = 100
    all_dynamic_threshold_std_multiplier: float = 4.0
    all_min_threshold_factor: float = 1.0
    top_anomaly_consecutive_hits: int = 2
    top_clear_consecutive_normals: int = 2
    top_score_history_size: int = 100
    top_dynamic_threshold_std_multiplier: float = 4.0
    top_min_threshold_factor: float = 1.0
    enable_gpt_adjudication: bool = False


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)


def ensure_pipeline_directories(config: PipelineConfig) -> None:
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    config.paths.adjudication_dir.mkdir(parents=True, exist_ok=True)
