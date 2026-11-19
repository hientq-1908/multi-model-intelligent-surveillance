"""Optimizer and learning-rate schedule builders shared by the stages."""

from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from mlis.train.core.config import RunConfig


def build_optimizer(model: nn.Module, config: RunConfig) -> Optimizer:
    """AdamW over the trainable parameters only."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    return AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(optimizer: Optimizer, config: RunConfig) -> LambdaLR:
    """Linear warmup, then cosine decay to ``min_lr_ratio`` of the peak at ``max_steps``."""
    warmup, total, floor = config.warmup_steps, config.max_steps, config.min_lr_ratio

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)
