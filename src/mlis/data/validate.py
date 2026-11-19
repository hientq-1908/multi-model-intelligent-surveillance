"""Validate a packed dataset: header, records, ids, shards and images."""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from mlis.data.images import decode_image
from mlis.data.manifest import ManifestError, iter_manifest, read_manifest_header
from mlis.data.reader import iter_image_bytes
from mlis.data.schema import LicenceTag, SampleRecord
from mlis.data.shards import ShardError

RESTRICTED_LICENCES = frozenset(
    {LicenceTag.NON_COMMERCIAL, LicenceTag.RESEARCH_ONLY, LicenceTag.UNSTATED}
)


def validate_dataset(manifest_path: Path, *, decode_images: bool = True) -> list[str]:
    """Return a list of problems. An empty list means the dataset is valid."""
    manifest_path = Path(manifest_path)
    problems: list[str] = []
    try:
        header = read_manifest_header(manifest_path)
        records = list(iter_manifest(manifest_path))
    except (ManifestError, OSError) as err:
        return [str(err)]

    if header.num_samples != len(records):
        problems.append(f"header says {header.num_samples} samples but file has {len(records)}")
    problems.extend(_duplicate_id_problems(records))
    for name in header.shards:
        if not (manifest_path.parent / name).is_file():
            problems.append(f"missing shard {name}")
    if header.shards:
        problems.extend(_shard_reference_problems(records, header.shards))
    elif any(record.image.shard for record in records):
        problems.append("records reference shards but the header lists none")
    if decode_images and not problems:
        problems.extend(_image_problems(manifest_path, records))
    return problems


def _duplicate_id_problems(records: Sequence[SampleRecord]) -> list[str]:
    counts = Counter(record.sample_id for record in records)
    return [f"duplicate sample_id {sid!r} x{n}" for sid, n in counts.items() if n > 1]


def _shard_reference_problems(records: Sequence[SampleRecord], shards: Sequence[str]) -> list[str]:
    problems = []
    listed = set(shards)
    for record in records:
        if record.image.shard is None:
            problems.append(f"{record.sample_id}: shard-backed manifest but image has no shard")
        elif record.image.shard not in listed:
            problems.append(f"{record.sample_id}: shard {record.image.shard!r} not in header")
    return problems


def _image_problems(manifest_path: Path, records: Sequence[SampleRecord]) -> list[str]:
    problems = []
    try:
        for record, data in iter_image_bytes(manifest_path):
            try:
                decode_image(data, record.image)
            except (OSError, ValueError) as err:
                problems.append(f"{record.sample_id}: cannot decode image: {err}")
    except (ShardError, OSError, tarfile.TarError) as err:
        problems.append(f"cannot read images: {err}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a packed dataset manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--no-decode", action="store_true", help="Skip decoding every image.")
    args = parser.parse_args(argv)
    problems = validate_dataset(args.manifest, decode_images=not args.no_decode)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        header = read_manifest_header(args.manifest)
        print(f"ok: {header.dataset}, {header.num_samples} samples")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
