"""Open a packed dataset from its manifest, whether images are in shards or loose files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mlis.data.images import decode_image
from mlis.data.manifest import iter_manifest, read_manifest_header
from mlis.data.schema import SampleRecord
from mlis.data.shards import iter_shards


@dataclass(frozen=True)
class LoadedSample:
    record: SampleRecord
    image: Image.Image


def iter_image_bytes(manifest_path: Path, *, skip: int = 0) -> Iterator[tuple[SampleRecord, bytes]]:
    """Yield ``(record, encoded image bytes)`` in manifest order, dropping the first ``skip``.

    Shard-backed manifests stream the shards in the order the header lists them.
    File-backed manifests read ``<root>/<image.path>``.
    """
    manifest_path = Path(manifest_path)
    header = read_manifest_header(manifest_path)
    base = manifest_path.parent
    if header.shards:
        shard_paths = [base / name for name in header.shards]
        for sample in iter_shards(shard_paths, skip=skip):
            yield sample.record, sample.image_bytes
        return
    root = base / header.root if header.root else base
    for index, record in enumerate(iter_manifest(manifest_path)):
        if index >= skip:
            yield record, (root / record.image.path).read_bytes()


def iter_samples(manifest_path: Path, *, skip: int = 0) -> Iterator[LoadedSample]:
    """Yield decoded samples with the crop box already applied."""
    for record, data in iter_image_bytes(manifest_path, skip=skip):
        yield LoadedSample(record=record, image=decode_image(data, record.image))
