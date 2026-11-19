"""WebDataset-compatible tar shards.

Each sample is two adjacent tar members that share a key: ``<sample_id>.json`` holds the
SampleRecord and ``<sample_id>.<ext>`` holds the original image bytes.
Plain ``tarfile`` is used so the format needs no extra dependency, and any WebDataset
reader can still open the shards.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from mlis.data.schema import ImageRef, SampleRecord

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
DEFAULT_MAX_SAMPLES_PER_SHARD = 1000
DEFAULT_MAX_BYTES_PER_SHARD = 512 * 1024 * 1024


class ShardError(ValueError):
    """A shard is malformed or a write request is invalid."""


@dataclass(frozen=True)
class ShardSample:
    record: SampleRecord
    image_bytes: bytes
    shard: str


class ShardWriter:
    """Write samples into numbered shards ``<prefix>-00000.tar`` under ``out_dir``.

    A shard is written to a ``.tmp`` file and renamed when it is full or the writer closes,
    so a crash never leaves a truncated shard under its final name.
    """

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        *,
        max_samples: int = DEFAULT_MAX_SAMPLES_PER_SHARD,
        max_bytes: int = DEFAULT_MAX_BYTES_PER_SHARD,
    ) -> None:
        if max_samples < 1 or max_bytes < 1:
            raise ShardError("max_samples and max_bytes must be positive")
        self._out_dir = Path(out_dir)
        self._prefix = prefix
        self._max_samples = max_samples
        self._max_bytes = max_bytes
        self._shard_index = 0
        self._tar: tarfile.TarFile | None = None
        self._tmp_path: Path | None = None
        self._final_path: Path | None = None
        self._samples_in_shard = 0
        self._bytes_in_shard = 0
        self._seen_ids: set[str] = set()
        self.shard_names: list[str] = []
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._discard_open_shard()

    def write(self, record: SampleRecord, image_bytes: bytes, *, ext: str) -> SampleRecord:
        """Add one sample and return the record with its image pointing into the shard."""
        ext = ext.lower().lstrip(".")
        if ext not in IMAGE_EXTENSIONS:
            raise ShardError(f"unsupported image extension {ext!r}; use one of {IMAGE_EXTENSIONS}")
        if record.sample_id in self._seen_ids:
            raise ShardError(f"duplicate sample_id {record.sample_id!r}")
        if self._tar is None:
            self._open_next_shard()
        assert self._tar is not None
        shard_name = self.shard_names[-1]
        member = f"{record.sample_id}.{ext}"
        packed = record.model_copy(
            update={"image": ImageRef(path=member, shard=shard_name, box=record.image.box)}
        )
        payload = packed.model_dump_json(exclude_none=True).encode("utf-8")
        _add_member(self._tar, f"{record.sample_id}.json", payload)
        _add_member(self._tar, member, image_bytes)
        self._seen_ids.add(record.sample_id)
        self._samples_in_shard += 1
        self._bytes_in_shard += len(payload) + len(image_bytes)
        if self._samples_in_shard >= self._max_samples or self._bytes_in_shard >= self._max_bytes:
            self._finish_open_shard()
        return packed

    def close(self) -> None:
        if self._tar is not None:
            self._finish_open_shard()

    def _open_next_shard(self) -> None:
        name = f"{self._prefix}-{self._shard_index:05d}.tar"
        self._shard_index += 1
        self._final_path = self._out_dir / name
        self._tmp_path = self._out_dir / (name + ".tmp")
        self._tar = tarfile.open(self._tmp_path, "w")  # noqa: SIM115 - closed in _finish/_discard
        self._samples_in_shard = 0
        self._bytes_in_shard = 0
        self.shard_names.append(name)

    def _finish_open_shard(self) -> None:
        assert self._tar is not None and self._tmp_path is not None and self._final_path is not None
        self._tar.close()
        with self._tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self._tmp_path, self._final_path)
        self._tar = None

    def _discard_open_shard(self) -> None:
        if self._tar is not None and self._tmp_path is not None:
            self._tar.close()
            self._tmp_path.unlink(missing_ok=True)
            self.shard_names.pop()
            self._tar = None


def _add_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


def iter_shard(path: Path) -> Iterator[ShardSample]:
    """Yield the samples of one shard in file order.

    Members of one sample must be adjacent. A key without exactly one json and one image
    member raises ShardError.
    """
    path = Path(path)
    current_key: str | None = None
    parts: dict[str, bytes] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, ext = member.name.partition(".")
            if key != current_key:
                if current_key is not None:
                    yield _assemble(path, current_key, parts)
                current_key, parts = key, {}
            handle = tar.extractfile(member)
            assert handle is not None
            if ext in parts:
                raise ShardError(f"{path}: duplicate member {member.name!r}")
            parts[ext] = handle.read()
    if current_key is not None:
        yield _assemble(path, current_key, parts)


def iter_shards(paths: Sequence[Path], *, skip: int = 0) -> Iterator[ShardSample]:
    """Yield samples across shards in order, dropping the first ``skip`` samples.

    ``skip`` is how a resumed run returns to its data position.
    """
    remaining = skip
    for path in paths:
        for sample in iter_shard(Path(path)):
            if remaining > 0:
                remaining -= 1
                continue
            yield sample


def _assemble(path: Path, key: str, parts: dict[str, bytes]) -> ShardSample:
    images = [ext for ext in parts if ext in IMAGE_EXTENSIONS]
    if "json" not in parts or len(images) != 1 or len(parts) != 2:
        raise ShardError(
            f"{path}: sample {key!r} must hold one json and one image member, found {sorted(parts)}"
        )
    record = SampleRecord.model_validate(json.loads(parts["json"]))
    if record.sample_id != key:
        raise ShardError(f"{path}: member key {key!r} differs from sample_id {record.sample_id!r}")
    return ShardSample(record=record, image_bytes=parts[images[0]], shard=path.name)


def pack_samples(
    items: Iterable[tuple[SampleRecord, bytes, str]],
    out_dir: Path,
    prefix: str,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES_PER_SHARD,
    max_bytes: int = DEFAULT_MAX_BYTES_PER_SHARD,
) -> tuple[list[SampleRecord], list[str]]:
    """Pack ``(record, image_bytes, ext)`` items into shards.

    Returns the packed records (image refs rewritten) and the shard file names.
    """
    packed: list[SampleRecord] = []
    with ShardWriter(out_dir, prefix, max_samples=max_samples, max_bytes=max_bytes) as writer:
        for record, image_bytes, ext in items:
            packed.append(writer.write(record, image_bytes, ext=ext))
    return packed, writer.shard_names
