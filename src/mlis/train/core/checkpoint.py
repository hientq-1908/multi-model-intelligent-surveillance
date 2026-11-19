"""Atomic checkpoint files with a ``latest`` pointer and retention."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_PATTERN = re.compile(r"^step-(\d{8})\.pt$")
LATEST_FILE = "latest"


def checkpoint_name(step: int) -> str:
    return f"step-{step:08d}.pt"


def save_checkpoint(directory: Path, step: int, payload: dict[str, Any], *, keep_last: int) -> Path:
    """Write ``step-<n>.pt`` atomically, point ``latest`` at it and prune older files.

    The payload goes to a temporary file in the same directory, is fsynced, and is renamed
    over the final name, so a crash or a lost pod never leaves a half-written checkpoint
    under a real name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / checkpoint_name(step)
    _atomic_write(final, lambda handle: torch.save(payload, handle))
    _atomic_write(directory / LATEST_FILE, lambda handle: handle.write(final.name.encode()))
    _prune(directory, keep_last)
    return final


def find_latest_checkpoint(directory: Path) -> Path | None:
    """Return the newest checkpoint, trusting ``latest`` and falling back to a directory scan."""
    pointer = directory / LATEST_FILE
    if pointer.is_file():
        candidate = directory / pointer.read_text().strip()
        if candidate.is_file():
            return candidate
    steps = _list_checkpoints(directory)
    return steps[-1][1] if steps else None


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint written by ``save_checkpoint``.

    The payload holds NumPy and Python RNG state, so it is not a plain tensor file;
    only load checkpoints this project wrote.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_write(final: Path, write: Any) -> None:
    tmp = final.with_name(final.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


def _list_checkpoints(directory: Path) -> list[tuple[int, Path]]:
    found = []
    for path in directory.iterdir() if directory.is_dir() else []:
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _prune(directory: Path, keep_last: int) -> None:
    for _, path in _list_checkpoints(directory)[:-keep_last]:
        path.unlink(missing_ok=True)
