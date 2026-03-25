from __future__ import annotations

import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from container_ad_pipeline.config import PipelineConfig
from container_ad_pipeline.realtime import RealtimeMonitor


def _apply_environment_overrides(config: PipelineConfig) -> PipelineConfig:
    config.realtime.prom_url = os.getenv("PROM_URL", config.realtime.prom_url)
    config.realtime.target_namespace = os.getenv("TARGET_NAMESPACE", config.realtime.target_namespace)
    config.realtime.target_pod = os.getenv("TARGET_POD", config.realtime.target_pod)
    config.realtime.target_container = os.getenv("TARGET_CONTAINER", config.realtime.target_container)
    config.realtime.poll_interval_seconds = int(os.getenv("POLL_INTERVAL_SECONDS", config.realtime.poll_interval_seconds))
    config.realtime.enable_gpt_adjudication = os.getenv("ENABLE_GPT_ADJUDICATION", "false").lower() in ("1", "true", "yes")
    return config


def main() -> None:
    config = _apply_environment_overrides(PipelineConfig())
    monitor = RealtimeMonitor(config)
    monitor.run_forever()


if __name__ == "__main__":
    main()
