"""Shared trainer: bf16 autocast, periodic atomic checkpoints, exact resume, run log.

The trainer knows nothing about stages. Stage code supplies a ``compute_loss`` function
and the model, optimizer, scheduler and data stream; the trainer owns the loop.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mlis.train.core.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from mlis.train.core.config import RunConfig
from mlis.train.core.data_position import ResumableData
from mlis.train.core.run_log import RunLog
from mlis.train.core.seed import capture_rng_state, restore_rng_state, seed_everything

SECONDS_PER_MINUTE = 60.0
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@dataclass(frozen=True)
class LossOutput:
    """What ``compute_loss`` returns: a scalar loss and float metrics to log."""

    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainComponents:
    """Everything stage code hands to the trainer."""

    model: nn.Module
    compute_loss: Callable[[Any], LossOutput]
    optimizer: Optimizer
    scheduler: LRScheduler
    data: ResumableData


@dataclass(frozen=True)
class TrainResult:
    final_step: int
    stopped_early: bool
    checkpoint: Path | None
    losses: list[float]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    """Run ``max_steps`` optimizer steps with checkpointing and exact resume.

    Checkpoints hold only parameters with ``requires_grad`` (so LoRA and adapter runs do not
    write the frozen 4B base model), the optimizer, scheduler, step, every RNG and the data
    position.
    A checkpoint is written every ``checkpoint_every_minutes``, at the end of the run, and
    when SIGTERM or SIGINT arrives (the step in flight finishes first).
    A crash in the middle of a step writes nothing: resume restarts from the last checkpoint.
    """

    def __init__(
        self,
        config: RunConfig,
        components: TrainComponents,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._parts = components
        self._clock = clock
        self._device = resolve_device(config.device)
        self._checkpoint_dir = config.output_dir / "checkpoints"
        self._log = RunLog(config.output_dir, num_gpus=RunLog.count_gpus(self._device), clock=clock)
        self._step = 0
        self._stop_requested = False
        self._last_checkpoint_time = clock()
        self._last_checkpoint: Path | None = None
        self._losses: list[float] = []

    @property
    def step(self) -> int:
        return self._step

    def request_stop(self) -> None:
        """Ask the loop to checkpoint and return after the current step."""
        self._stop_requested = True

    def train(self) -> TrainResult:
        seed_everything(self._config.seed)
        self._parts.model.to(self._device)
        resumed_from = self._maybe_resume()
        self._log.event(
            "start",
            step=self._step,
            resumed_from=str(resumed_from) if resumed_from else None,
            run_name=self._config.run_name,
            device=str(self._device),
        )
        with _stop_signal_handlers(self.request_stop):
            while self._step < self._config.max_steps and not self._stop_requested:
                self._run_optimizer_step()
                self._maybe_checkpoint()
        stopped_early = self._step < self._config.max_steps
        self._save("stop" if stopped_early else "final")
        self._log.event("end", step=self._step, stopped_early=stopped_early)
        self._log.write_summary(final_step=self._step, stopped_early=stopped_early)
        return TrainResult(self._step, stopped_early, self._last_checkpoint, list(self._losses))

    def _run_optimizer_step(self) -> None:
        cfg, parts = self._config, self._parts
        parts.model.train()
        parts.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        metrics: dict[str, float] = {}
        for _ in range(cfg.grad_accum_steps):
            batch = _to_device(next(parts.data), self._device)
            with self._autocast():
                output = parts.compute_loss(batch)
            (output.loss / cfg.grad_accum_steps).backward()
            total_loss += float(output.loss.detach()) / cfg.grad_accum_steps
            metrics = output.metrics
        if cfg.grad_clip_norm is not None:
            trainable = [p for p in parts.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
        parts.optimizer.step()
        parts.scheduler.step()
        self._step += 1
        self._losses.append(total_loss)
        if self._step % cfg.log_every_steps == 0 or self._step == cfg.max_steps:
            self._log.event(
                "step",
                step=self._step,
                loss=total_loss,
                lr=parts.scheduler.get_last_lr()[0],
                **metrics,
            )

    def _autocast(self) -> Any:
        if self._config.precision == "bf16":
            return torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _maybe_checkpoint(self) -> None:
        cfg = self._config
        due_by_time = (
            self._clock() - self._last_checkpoint_time
            >= cfg.checkpoint_every_minutes * SECONDS_PER_MINUTE
        )
        due_by_steps = (
            cfg.checkpoint_every_steps is not None and self._step % cfg.checkpoint_every_steps == 0
        )
        if due_by_time or due_by_steps:
            self._save("periodic")

    def _save(self, reason: str) -> None:
        payload = {
            "step": self._step,
            "model": _trainable_state(self._parts.model),
            "optimizer": self._parts.optimizer.state_dict(),
            "scheduler": self._parts.scheduler.state_dict(),
            "rng": capture_rng_state(),
            "data": self._parts.data.state_dict(),
            "run_log": self._log.state_dict(),
            "losses": list(self._losses),
            "config": self._config.model_dump(mode="json"),
        }
        self._last_checkpoint = save_checkpoint(
            self._checkpoint_dir, self._step, payload, keep_last=self._config.keep_last_checkpoints
        )
        self._last_checkpoint_time = self._clock()
        self._log.event(
            "checkpoint", step=self._step, reason=reason, path=str(self._last_checkpoint)
        )

    def _maybe_resume(self) -> Path | None:
        policy = self._config.resume
        if policy == "never":
            return None
        path = find_latest_checkpoint(self._checkpoint_dir) if policy == "auto" else Path(policy)
        if path is None:
            return None
        self._restore(load_checkpoint(path))
        return path

    def _restore(self, payload: dict[str, Any]) -> None:
        parts = self._parts
        result = parts.model.load_state_dict(payload["model"], strict=False)
        trainable_names = {n for n, p in parts.model.named_parameters() if p.requires_grad}
        missing_trainable = trainable_names & set(result.missing_keys)
        if missing_trainable or result.unexpected_keys:
            raise RuntimeError(
                f"checkpoint does not match the model: missing {sorted(missing_trainable)}, "
                f"unexpected {sorted(result.unexpected_keys)}"
            )
        parts.optimizer.load_state_dict(payload["optimizer"])
        parts.scheduler.load_state_dict(payload["scheduler"])
        parts.data.load_state_dict(payload["data"])
        self._log.load_state_dict(payload["run_log"])
        self._step = int(payload["step"])
        self._losses = list(payload["losses"])
        restore_rng_state(payload["rng"])


def _trainable_state(model: nn.Module) -> dict[str, Tensor]:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list | tuple):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch


class _stop_signal_handlers:
    """Route SIGTERM and SIGINT to ``on_stop`` for the duration of the loop.

    Handlers can only be installed from the main thread; elsewhere this does nothing.
    """

    def __init__(self, on_stop: Callable[[], None]) -> None:
        self._on_stop = on_stop
        self._previous: dict[signal.Signals, Any] = {}

    def __enter__(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in STOP_SIGNALS:
            self._previous[sig] = signal.signal(sig, self._handle)

    def __exit__(self, *_: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def _handle(self, _signum: int, _frame: FrameType | None) -> None:
        self._on_stop()
