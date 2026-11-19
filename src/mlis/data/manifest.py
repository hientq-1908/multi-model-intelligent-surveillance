"""JSONL manifest: one header line, then one SampleRecord per line."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

from mlis.data.schema import ManifestHeader, SampleRecord

HEADER_KEY = "manifest_header"


class ManifestError(ValueError):
    """The manifest file is malformed."""


def write_manifest(
    path: Path,
    records: Iterable[SampleRecord],
    *,
    dataset: str,
    root: str | None = None,
    shards: list[str] | None = None,
    created_by: str | None = None,
) -> ManifestHeader:
    """Write a manifest atomically and return its header.

    Records stream through a temporary body file because the header holds the sample count.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body_path = path.with_name(path.name + ".body.tmp")
    final_tmp = path.with_name(path.name + ".tmp")
    count = 0
    try:
        with body_path.open("w", encoding="utf-8") as body:
            for record in records:
                body.write(record.model_dump_json(exclude_none=True) + "\n")
                count += 1
        header = ManifestHeader(
            dataset=dataset,
            num_samples=count,
            root=root,
            shards=shards or [],
            created_by=created_by,
        )
        with (
            final_tmp.open("w", encoding="utf-8") as out,
            body_path.open("r", encoding="utf-8") as body,
        ):
            out.write(json.dumps({HEADER_KEY: header.model_dump(exclude_none=True)}) + "\n")
            shutil.copyfileobj(body, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(final_tmp, path)
    finally:
        body_path.unlink(missing_ok=True)
        final_tmp.unlink(missing_ok=True)
    return header


def read_manifest_header(path: Path) -> ManifestHeader:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _parse_header(handle.readline(), path)


def iter_manifest(path: Path) -> Iterator[SampleRecord]:
    """Yield records lazily, skipping the header. Raises ManifestError on a bad line."""
    with Path(path).open("r", encoding="utf-8") as handle:
        _parse_header(handle.readline(), path)
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            try:
                yield SampleRecord.model_validate_json(line)
            except ValueError as err:
                raise ManifestError(f"{path}:{line_number}: invalid record: {err}") from err


def read_manifest(path: Path) -> tuple[ManifestHeader, list[SampleRecord]]:
    return read_manifest_header(path), list(iter_manifest(path))


def _parse_header(line: str, path: Path) -> ManifestHeader:
    try:
        payload = json.loads(line)
        return ManifestHeader.model_validate(payload[HEADER_KEY])
    except (ValueError, KeyError, TypeError) as err:
        raise ManifestError(f"{path}:1: first line must be a '{HEADER_KEY}' object: {err}") from err
