import json

import pytest

from mlis.data.manifest import (
    ManifestError,
    iter_manifest,
    read_manifest,
    read_manifest_header,
    write_manifest,
)
from mlis.data.schema import ImageRef, LicenceTag, SampleRecord, Split


def records(count):
    return [
        SampleRecord(
            sample_id=f"s{i:03d}",
            image=ImageRef(path=f"images/s{i:03d}.png"),
            texts=[f"text {i}"],
            source="demo",
            split=Split.TRAIN,
            licence=LicenceTag.PERMISSIVE,
        )
        for i in range(count)
    ]


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "m.jsonl"
    header = write_manifest(path, records(5), dataset="demo", root="img", shards=["a.tar"])
    read_header, read_records = read_manifest(path)
    assert read_header == header
    assert header.num_samples == 5
    assert read_records == records(5)


def test_file_is_header_line_then_one_record_per_line(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest(path, records(2), dataset="demo")
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert "manifest_header" in json.loads(lines[0])
    assert json.loads(lines[1])["sample_id"] == "s000"


def test_write_leaves_no_temporary_files(tmp_path):
    write_manifest(tmp_path / "m.jsonl", records(1), dataset="demo")
    assert [p.name for p in tmp_path.iterdir()] == ["m.jsonl"]


def test_failed_write_keeps_previous_manifest(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest(path, records(2), dataset="demo")

    def exploding():
        yield from records(1)
        raise RuntimeError("source broke")

    with pytest.raises(RuntimeError):
        write_manifest(path, exploding(), dataset="demo")
    assert read_manifest_header(path).num_samples == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.jsonl"]


def test_empty_manifest_is_valid(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest(path, [], dataset="demo")
    assert read_manifest(path)[1] == []


def test_missing_header_raises(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(records(1)[0].model_dump_json() + "\n")
    with pytest.raises(ManifestError, match="first line"):
        read_manifest_header(path)


def test_bad_record_line_reports_line_number(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest(path, records(1), dataset="demo")
    with path.open("a") as handle:
        handle.write('{"sample_id": "x"}\n')
    with pytest.raises(ManifestError, match=r":3:"):
        list(iter_manifest(path))
