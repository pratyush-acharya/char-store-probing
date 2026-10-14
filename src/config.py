"""Minimal configuration: which model to load, in which dtype, and whether to trust remote code."""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    # Some models ship remote modeling code; a few pin a newer transformers than installed
    # (e.g. Phi-4-mini). Scripts pass --no-remote-code for those to force the native class.
    trust_remote_code: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)


def apply_env_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """TRUST_REMOTE_CODE=0/1 in the environment overrides the config default."""
    v = os.environ.get("TRUST_REMOTE_CODE")
    if v is not None:
        config.model.trust_remote_code = v.strip().lower() in ("1", "true", "yes", "on")
    return config
