"""Deterministic synthetic dataset used as the test fixture by every task.

The images are flat colour blocks, so they are tiny and carry no real data.
Regenerate the committed copy with ``uv run python -m mlis.data.synthetic tests/fixtures``.
"""

from __future__ import annotations

import argparse
import io
import shutil
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from mlis.data.manifest import write_manifest
from mlis.data.schema import ImageRef, LicenceTag, SampleRecord, Split
from mlis.data.shards import pack_samples

FIXTURE_SHARD_DIR = "tiny_shards"
FIXTURE_FILE_DIR = "tiny_files"
FIXTURE_DATASET = "synthetic-tiny"
PERSON_IMAGE_SIZE = (32, 64)
GENERAL_IMAGE_SIZE = (64, 48)

_COLOURS = {
    "red": (200, 30, 30),
    "blue": (30, 60, 200),
    "green": (30, 160, 60),
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "yellow": (230, 200, 40),
}
_COLOUR_NAMES = list(_COLOURS)
_NUM_GENERAL = 6
_NUM_PERSON = 6


def make_records() -> list[tuple[SampleRecord, bytes]]:
    """Return 12 samples: 6 general (no person data) and 6 person samples over 3 identities."""
    items: list[tuple[SampleRecord, bytes]] = []
    for index in range(_NUM_GENERAL):
        colour = _COLOUR_NAMES[index % len(_COLOUR_NAMES)]
        record = SampleRecord(
            sample_id=f"gen{index:03d}",
            image=ImageRef(path=f"images/gen{index:03d}.png"),
            texts=[f"a {colour} rectangle", f"an image filled with {colour}"],
            source="synthetic-general",
            split=Split.TEST if index == _NUM_GENERAL - 1 else Split.TRAIN,
            licence=LicenceTag.SYNTHETIC,
        )
        items.append((record, _png(GENERAL_IMAGE_SIZE, [colour])))
    for index in range(_NUM_PERSON):
        upper = _COLOUR_NAMES[index % len(_COLOUR_NAMES)]
        lower = _COLOUR_NAMES[(index + 3) % len(_COLOUR_NAMES)]
        swapped = _COLOUR_NAMES[(index + 1) % len(_COLOUR_NAMES)]
        record = SampleRecord(
            sample_id=f"per{index:03d}",
            image=ImageRef(path=f"images/per{index:03d}.png"),
            texts=[f"a person with a {upper} top and {lower} trousers"],
            source="synthetic-person",
            split=Split.VAL if index == _NUM_PERSON - 1 else Split.TRAIN,
            licence=LicenceTag.SYNTHETIC,
            person_id=f"synthetic-person:{index // 2}",
            attributes={"upper_colour": upper, "lower_colour": lower},
            negatives=[f"a person with a {swapped} top and {lower} trousers"],
        )
        items.append((record, _png(PERSON_IMAGE_SIZE, [upper, lower])))
    return items


def build_fixture(out_dir: Path) -> None:
    """Write both fixture layouts under ``out_dir``, replacing any previous copy."""
    out_dir = Path(out_dir)
    items = make_records()
    _write_shard_layout(out_dir / FIXTURE_SHARD_DIR, items)
    _write_file_layout(out_dir / FIXTURE_FILE_DIR, items)


def _write_shard_layout(directory: Path, items: Sequence[tuple[SampleRecord, bytes]]) -> None:
    _reset(directory)
    packed, shard_names = pack_samples(
        ((record, data, "png") for record, data in items),
        directory,
        "tiny",
        max_samples=8,
    )
    write_manifest(
        directory / "manifest.jsonl",
        packed,
        dataset=FIXTURE_DATASET,
        shards=shard_names,
        created_by="mlis.data.synthetic",
    )


def _write_file_layout(directory: Path, items: Sequence[tuple[SampleRecord, bytes]]) -> None:
    _reset(directory)
    (directory / "images").mkdir()
    for record, data in items:
        (directory / record.image.path).write_bytes(data)
    write_manifest(
        directory / "manifest.jsonl",
        (record for record, _ in items),
        dataset=FIXTURE_DATASET,
        root=".",
        created_by="mlis.data.synthetic",
    )


def _reset(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


def _png(size: tuple[int, int], colours: list[str]) -> bytes:
    """Stack one flat block per colour from top to bottom."""
    width, height = size
    image = Image.new("RGB", size)
    band = height // len(colours)
    for position, name in enumerate(colours):
        bottom = height if position == len(colours) - 1 else (position + 1) * band
        image.paste(_COLOURS[name], (0, position * band, width, bottom))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the synthetic fixture dataset.")
    parser.add_argument("out_dir", type=Path)
    build_fixture(parser.parse_args(argv).out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
