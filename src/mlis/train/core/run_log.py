"""Append-only run log with GPU-hour accounting."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

RUN_LOG_FILE = "run_log.jsonl"
SUMMARY_FILE = "summary.json"
SECONDS_PER_HOUR = 3600.0


class RunLog:
    """JSONL event log plus running totals of wall time and GPU time.

    GPU-hours are wall-clock hours times the number of GPUs in use, which is what a rented
    pod bills. The totals are saved in checkpoints, so a resumed run keeps counting from
    where the checkpoint left off.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        num_gpus: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        self._path = output_dir / RUN_LOG_FILE
        self._num_gpus = num_gpus
        self._clock = clock
        self._segment_start = clock()
        self._carried_wall_seconds = 0.0

    @staticmethod
    def count_gpus(device: torch.device) -> int:
        return torch.cuda.device_count() if device.type == "cuda" else 0

    @property
    def wall_seconds(self) -> float:
        return self._carried_wall_seconds + (self._clock() - self._segment_start)

    @property
    def gpu_hours(self) -> float:
        return self.wall_seconds * self._num_gpus / SECONDS_PER_HOUR

    def state_dict(self) -> dict[str, float]:
        return {"wall_seconds": self.wall_seconds}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._carried_wall_seconds = float(state["wall_seconds"])
        self._segment_start = self._clock()

    def event(self, name: str, **fields: Any) -> None:
        record = {
            "event": name,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "wall_hours": round(self.wall_seconds / SECONDS_PER_HOUR, 6),
            "gpu_hours": round(self.gpu_hours, 6),
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def write_summary(self, **fields: Any) -> None:
        summary = {
            "wall_hours": self.wall_seconds / SECONDS_PER_HOUR,
            "gpu_hours": self.gpu_hours,
            "num_gpus": self._num_gpus,
            **fields,
        }
        final = self._output_dir / SUMMARY_FILE
        tmp = final.with_name(final.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, final)
