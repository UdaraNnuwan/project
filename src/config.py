from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHUNK_SIZE = 10_000
WINDOW_SIZE = 30
STRIDE = 5
TRAIN_SPLIT = 0.80
MIN_USAGE_ROWS = 500_000

DEFAULT_RAW_ROOT_CANDIDATES = (
    RAW_DATA_DIR,
    PROJECT_ROOT.parent / "dataset" / "data",
    DATA_DIR,
)


def load_project_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env(PROJECT_ROOT / ".env")


def first_existing_path(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


@dataclass
class DatasetConfig:
    raw_data_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_RAW_DATA_DIR",
            first_existing_path(DEFAULT_RAW_ROOT_CANDIDATES),
        )
    )
    output_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_DATASET_OUTPUT_DIR",
            PROCESSED_DATA_DIR,
        )
    )
    container_meta_name: str = "container_meta.tar.gz"
    container_usage_name: str = "container_usage.tar.gz"
    machine_meta_name: str = "machine_meta.tar.gz"
    machine_usage_name: str = "machine_usage.tar.gz"
    chunksize: int = CHUNK_SIZE
    max_usage_rows: int | None = MIN_USAGE_ROWS
    max_container_meta_rows: int | None = None
    max_machine_usage_rows: int | None = MIN_USAGE_ROWS
    max_machine_meta_rows: int | None = None
    container_limit: int | None = None
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    train_ratio: float = TRAIN_SPLIT
    val_ratio: float = 0.0
    min_window_observed_ratio: float = 0.60
    feature_columns: tuple[str, ...] = (
        "cpu_util",
        "mem_util",
        "cpi",
        "mem_gps",
        "mpki",
        "net_in",
        "net_out",
        "disk_io",
    )
    container_context_numeric: tuple[str, ...] = (
        "container_cpu_request",
        "container_cpu_limit",
        "container_mem_size",
    )
    container_context_categorical: tuple[str, ...] = (
        "container_app_du",
        "container_status",
    )
    machine_context_numeric: tuple[str, ...] = (
        "machine_cpu_num",
        "machine_mem_size",
        "machine_cpu_util",
        "machine_mem_util",
        "machine_mem_gps",
        "machine_mpki",
        "machine_net_in",
        "machine_net_out",
        "machine_disk_io",
    )
    machine_context_categorical: tuple[str, ...] = (
        "machine_failure_domain_1",
        "machine_failure_domain_2",
        "machine_status",
    )

    @property
    def context_columns(self) -> tuple[str, ...]:
        return (
            self.container_context_numeric
            + self.machine_context_numeric
            + self.container_context_categorical
            + self.machine_context_categorical
        )

    @property
    def archive_paths(self) -> dict[str, Path]:
        return {
            "container_meta": self.raw_data_dir / self.container_meta_name,
            "container_usage": self.raw_data_dir / self.container_usage_name,
            "machine_meta": self.raw_data_dir / self.machine_meta_name,
            "machine_usage": self.raw_data_dir / self.machine_usage_name,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_data_dir"] = str(self.raw_data_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["archive_paths"] = {key: str(value) for key, value in self.archive_paths.items()}
        payload["context_columns"] = list(self.context_columns)
        return payload


@dataclass
class TrainConfig:
    dataset_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_DATASET_OUTPUT_DIR",
            PROCESSED_DATA_DIR,
        )
    )
    model_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_MODEL_DIR",
            MODELS_DIR,
        )
    )
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    lr_plateau_patience: int = 4
    lr_plateau_factor: float = 0.5
    threshold_quantile: float = 0.995
    random_seed: int = 42
    units: int = 64
    latent: int = 64
    score_mode: str = "mean_feature_mse"
    device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["model_dir"] = str(self.model_dir)
        return payload


@dataclass
class EvalConfig:
    dataset_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_DATASET_OUTPUT_DIR",
            PROCESSED_DATA_DIR,
        )
    )
    model_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_MODEL_DIR",
            MODELS_DIR,
        )
    )
    output_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_RESULTS_DIR",
            RESULTS_DIR,
        )
    )
    split: str = "test"
    batch_size: int = 256
    top_k_features: int = 5
    synthetic_anomaly_ratio: float = 0.08
    synthetic_event_span: int = 4
    synthetic_feature_count: int = 2
    synthetic_spike_magnitude: float = 4.0
    synthetic_noise_std: float = 0.6
    relaxed_detection_tolerance: int = 2
    use_synthetic_injection: bool = True
    include_gpt_in_stream: bool = False
    telegram_enabled: bool = False
    telegram_critical_only: bool = True
    telegram_require_gpt_reason: bool = True
    telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    telegram_chat_id_env: str = "TELEGRAM_CHAT_ID"
    telegram_max_alerts: int = 20
    telegram_timeout_seconds: int = 15
    random_seed: int = 42
    device: str = "cuda"
    verbose: bool = True
    show_progress: bool = True

    @property
    def telegram_bot_token(self) -> str | None:
        return os.getenv(self.telegram_bot_token_env)

    @property
    def telegram_chat_id(self) -> str | None:
        return os.getenv(self.telegram_chat_id_env)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["model_dir"] = str(self.model_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["telegram_bot_token_present"] = bool(self.telegram_bot_token)
        payload["telegram_chat_id_present"] = bool(self.telegram_chat_id)
        return payload


@dataclass
class GPTConfig:
    evaluation_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_RESULTS_DIR",
            RESULTS_DIR,
        )
    )
    output_dir: Path = field(
        default_factory=lambda: env_path(
            "CONTAINER_AD_GPT_RESULTS_DIR",
            RESULTS_DIR,
        )
    )
    prompt_template_path: Path | None = field(
        default_factory=lambda: (
            env_path("CONTAINER_AD_GPT_PROMPT_PATH", PROJECT_ROOT / "gpt_prompt.txt")
            if os.getenv("CONTAINER_AD_GPT_PROMPT_PATH")
            else None
        )
    )
    openai_api_key_env: str = "OPENAI_API_KEY"
    model_env: str = "OPENAI_MODEL"
    default_model: str = "gpt-5-mini-2025-08-07"
    max_records: int = 100
    top_k_features: int = 5
    request_timeout_seconds: int = 60

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.openai_api_key_env)

    @property
    def model(self) -> str:
        return os.getenv(self.model_env, self.default_model)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation_dir"] = str(self.evaluation_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["prompt_template_path"] = (
            str(self.prompt_template_path) if self.prompt_template_path else None
        )
        payload["model"] = self.model
        payload["api_key_present"] = bool(self.api_key)
        return payload


CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "raw_data_dir": str(RAW_DATA_DIR),
    "processed_data_dir": str(PROCESSED_DATA_DIR),
    "models_dir": str(MODELS_DIR),
    "results_dir": str(RESULTS_DIR),
    "dataset": DatasetConfig().to_dict(),
    "train": TrainConfig().to_dict(),
    "evaluate": EvalConfig().to_dict(),
    "gpt": GPTConfig().to_dict(),
}
