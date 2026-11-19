import tarfile

import pytest

from mlis.data.schema import ImageRef, LicenceTag, SampleRecord, Split
from mlis.data.shards import ShardError, ShardWriter, iter_shard, iter_shards, pack_samples


def record(sample_id):
    return SampleRecord(
        sample_id=sample_id,
        image=ImageRef(path="ignored.png"),
        texts=["t"],
        source="demo",
        split=Split.TRAIN,
        licence=LicenceTag.PERMISSIVE,
    )


def items(count):
    return [(record(f"s{i:03d}"), f"bytes-{i}".encode(), "png") for i in range(count)]


def test_pack_rolls_over_to_numbered_shards(tmp_path):
    packed, names = pack_samples(items(5), tmp_path, "demo", max_samples=2)
    assert names == ["demo-00000.tar", "demo-00001.tar", "demo-00002.tar"]
    assert [p.image.shard for p in packed] == [names[0]] * 2 + [names[1]] * 2 + [names[2]]
    assert packed[0].image.path == "s000.png"


def test_roundtrip_preserves_records_and_bytes(tmp_path):
    packed, names = pack_samples(items(5), tmp_path, "demo", max_samples=2)
    samples = list(iter_shards([tmp_path / n for n in names]))
    assert [s.record for s in samples] == packed
    assert [s.image_bytes for s in samples] == [f"bytes-{i}".encode() for i in range(5)]


def test_shards_are_webdataset_layout(tmp_path):
    _, names = pack_samples(items(1), tmp_path, "demo")
    with tarfile.open(tmp_path / names[0]) as tar:
        assert tar.getnames() == ["s000.json", "s000.png"]


def test_skip_resumes_at_sample_position_across_shards(tmp_path):
    _, names = pack_samples(items(5), tmp_path, "demo", max_samples=2)
    paths = [tmp_path / n for n in names]
    resumed = [s.record.sample_id for s in iter_shards(paths, skip=3)]
    assert resumed == ["s003", "s004"]


def test_no_temporary_files_after_close(tmp_path):
    pack_samples(items(3), tmp_path, "demo", max_samples=2)
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_writer_discards_open_shard(tmp_path):
    with pytest.raises(RuntimeError), ShardWriter(tmp_path, "demo") as writer:
        writer.write(record("s000"), b"x", ext="png")
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_duplicate_sample_id_and_bad_extension_rejected(tmp_path):
    with ShardWriter(tmp_path, "demo") as writer:
        writer.write(record("s000"), b"x", ext="png")
        with pytest.raises(ShardError, match="duplicate"):
            writer.write(record("s000"), b"x", ext="png")
        with pytest.raises(ShardError, match="extension"):
            writer.write(record("s001"), b"x", ext="exe")


def test_reader_rejects_sample_missing_its_image(tmp_path):
    path = tmp_path / "bad.tar"
    packed, _ = pack_samples(items(1), tmp_path / "ok", "demo")
    with tarfile.open(path, "w") as tar:
        import io

        payload = packed[0].model_dump_json().encode()
        info = tarfile.TarInfo("s000.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ShardError, match="one json and one image"):
        list(iter_shard(path))
