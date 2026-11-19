"""Data streams whose position can be saved and restored exactly."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, TypeVar

BatchT = TypeVar("BatchT")


class ResumableData(Protocol):
    """Infinite batch stream with a restorable position.

    Streaming sources (WebDataset shards) implement the same protocol, using the ``skip``
    argument of ``mlis.data.shards.iter_shards`` to return to a saved position.
    """

    def __iter__(self) -> Iterator[Any]: ...

    def __next__(self) -> Any: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class IndexedBatchStream:
    """Deterministic shuffled batches over a map-style dataset, across epochs, without workers.

    The order of each epoch depends only on ``seed`` and the epoch number, never on global
    RNG state, so the position ``(epoch, batch_in_epoch)`` is enough to resume exactly.
    The last partial batch of an epoch is dropped.
    """

    def __init__(
        self,
        dataset: Sequence[Any],
        batch_size: int,
        seed: int,
        collate: Callable[[list[Any]], Any],
    ) -> None:
        if len(dataset) < batch_size:
            raise ValueError(f"dataset of {len(dataset)} is smaller than batch size {batch_size}")
        self._dataset = dataset
        self._batch_size = batch_size
        self._seed = seed
        self._collate = collate
        self._batches_per_epoch = len(dataset) // batch_size
        self._epoch = 0
        self._batch_in_epoch = 0
        self._order_epoch: int | None = None
        self._order: list[int] = []

    def __iter__(self) -> IndexedBatchStream:
        return self

    def __next__(self) -> Any:
        if self._batch_in_epoch >= self._batches_per_epoch:
            self._epoch += 1
            self._batch_in_epoch = 0
        start = self._batch_in_epoch * self._batch_size
        indices = self._epoch_order()[start : start + self._batch_size]
        self._batch_in_epoch += 1
        return self._collate([self._dataset[i] for i in indices])

    def _epoch_order(self) -> list[int]:
        if self._order_epoch != self._epoch:
            order = list(range(len(self._dataset)))
            random.Random(f"{self._seed}:{self._epoch}").shuffle(order)
            self._order, self._order_epoch = order, self._epoch
        return self._order

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "batch_in_epoch": self._batch_in_epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = int(state["epoch"])
        self._batch_in_epoch = int(state["batch_in_epoch"])
