"""One YAML file per run, validated into a typed config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    """Run settings shared by every stage.

    ``model`` and ``data`` are free-form sections owned by the stage code (T1 and T2),
    which validates them with its own schema.
    Unknown top-level keys are rejected so typos fail loudly.
    """

    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(min_length=1)
    output_dir: Path
    seed: int = 0
    max_steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    grad_accum_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_steps: int = Field(default=0, ge=0)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1)
    grad_clip_norm: float | None = Field(default=1.0, gt=0)
    precision: Literal["bf16", "fp32"] = "bf16"
    device: str = "auto"
    checkpoint_every_minutes: float = Field(default=30.0, gt=0)
    checkpoint_every_steps: int | None = Field(default=None, ge=1)
    keep_last_checkpoints: int = Field(default=2, ge=1)
    log_every_steps: int = Field(default=10, ge=1)
    resume: Literal["auto", "never"] | Path = "auto"
    model: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _warmup_fits_in_run(self) -> RunConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        return self


def load_run_config(path: Path) -> RunConfig:
    """Load and validate a run config from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return RunConfig.model_validate(raw)
