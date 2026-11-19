"""Tiny deterministic training setup shared by the trainer tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mlis.train.core.config import RunConfig
from mlis.train.core.data_position import IndexedBatchStream
from mlis.train.core.optim import build_optimizer, build_scheduler
from mlis.train.core.trainer import LossOutput, TrainComponents, Trainer


class TinyNet(nn.Module):
    """Has dropout so that resume equality also proves the RNG state is restored."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(123)
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.frozen(x))


def make_dataset(size: int = 24) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    return [
        (torch.randn(4, generator=generator), torch.randn(1, generator=generator))
        for _ in range(size)
    ]


def collate(items: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"x": torch.stack([i[0] for i in items]), "y": torch.stack([i[1] for i in items])}


def make_config(tmp_path: Path, **overrides) -> RunConfig:
    max_steps = overrides.get("max_steps", 10)
    values = {
        "run_name": "tiny",
        "output_dir": tmp_path,
        "seed": 5,
        "max_steps": 10,
        "batch_size": 4,
        "learning_rate": 0.05,
        "warmup_steps": min(2, max_steps - 1),
        "precision": "fp32",
        "device": "cpu",
        "checkpoint_every_minutes": 1000.0,
        "log_every_steps": 1,
    }
    return RunConfig(**{**values, **overrides})


def make_trainer(config: RunConfig, clock=None) -> tuple[Trainer, TinyNet]:
    model = TinyNet()

    def compute_loss(batch: dict[str, torch.Tensor]) -> LossOutput:
        loss = torch.nn.functional.mse_loss(model(batch["x"]), batch["y"])
        return LossOutput(loss=loss, metrics={"mse": float(loss.detach())})

    optimizer = build_optimizer(model, config)
    components = TrainComponents(
        model=model,
        compute_loss=compute_loss,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, config),
        data=IndexedBatchStream(make_dataset(), config.batch_size, config.seed, collate),
    )
    kwargs = {} if clock is None else {"clock": clock}
    return Trainer(config, components, **kwargs), model
