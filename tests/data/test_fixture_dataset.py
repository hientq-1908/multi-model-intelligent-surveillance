from mlis.data.manifest import read_manifest
from mlis.data.reader import iter_samples
from mlis.data.schema import LicenceTag
from mlis.data.synthetic import build_fixture, make_records
from mlis.data.validate import main, validate_dataset


def test_committed_shard_fixture_is_valid(shard_manifest):
    assert validate_dataset(shard_manifest) == []


def test_committed_file_fixture_is_valid(file_manifest):
    assert validate_dataset(file_manifest) == []


def test_both_layouts_hold_the_same_samples(shard_manifest, file_manifest):
    shard_records = [(r.sample_id, r.texts) for r in read_manifest(shard_manifest)[1]]
    file_records = [(r.sample_id, r.texts) for r in read_manifest(file_manifest)[1]]
    assert shard_records == file_records
    assert len(shard_records) == 12


def test_fixture_covers_every_optional_field(shard_manifest):
    records = read_manifest(shard_manifest)[1]
    person = [r for r in records if r.person_id]
    assert len(person) == 6
    assert all(r.attributes and r.negatives for r in person)
    assert {r.split.value for r in records} == {"train", "val", "test"}
    assert {r.licence for r in records} == {LicenceTag.SYNTHETIC}


def test_samples_decode_to_rgb_images(shard_manifest, file_manifest):
    for manifest in (shard_manifest, file_manifest):
        samples = list(iter_samples(manifest))
        assert len(samples) == 12
        assert all(s.image.mode == "RGB" for s in samples)
        assert {s.image.size for s in samples} == {(32, 64), (64, 48)}


def test_committed_fixture_matches_the_generator(tmp_path, shard_manifest, file_manifest):
    build_fixture(tmp_path)
    for generated, committed in (
        (tmp_path / "tiny_shards" / "manifest.jsonl", shard_manifest),
        (tmp_path / "tiny_files" / "manifest.jsonl", file_manifest),
    ):
        assert read_manifest(generated)[1] == read_manifest(committed)[1]
    assert len(make_records()) == 12


def test_validator_flags_corrupted_dataset(tmp_path):
    build_fixture(tmp_path)
    manifest = tmp_path / "tiny_shards" / "manifest.jsonl"
    next((tmp_path / "tiny_shards").glob("*.tar")).unlink()
    problems = validate_dataset(manifest)
    assert any("missing shard" in p for p in problems)
    assert main([str(manifest)]) == 1


def test_validator_flags_count_mismatch_and_duplicates(tmp_path):
    build_fixture(tmp_path)
    manifest = tmp_path / "tiny_files" / "manifest.jsonl"
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join([*lines, lines[1]]) + "\n")
    problems = validate_dataset(manifest)
    assert any("header says 12" in p for p in problems)
    assert any("duplicate sample_id" in p for p in problems)


def test_validator_flags_undecodable_image(tmp_path):
    build_fixture(tmp_path)
    (tmp_path / "tiny_files" / "images" / "gen000.png").write_bytes(b"not a png")
    problems = validate_dataset(tmp_path / "tiny_files" / "manifest.jsonl")
    assert any("gen000" in p and "decode" in p for p in problems)


def test_validator_cli_reports_ok(shard_manifest, capsys):
    assert main([str(shard_manifest)]) == 0
    assert "12 samples" in capsys.readouterr().out
