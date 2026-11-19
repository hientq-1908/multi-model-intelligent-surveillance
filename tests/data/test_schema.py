import pytest
from pydantic import ValidationError

from mlis.data.schema import ImageRef, LicenceTag, SampleRecord, Split


def make(**overrides):
    base = {
        "sample_id": "a001",
        "image": ImageRef(path="images/a001.png"),
        "texts": ["a caption"],
        "source": "demo-set",
        "split": Split.TRAIN,
        "licence": LicenceTag.SYNTHETIC,
    }
    return SampleRecord(**{**base, **overrides})


def test_minimal_record_has_optional_fields_unset():
    record = make()
    assert record.person_id is None
    assert record.attributes is None
    assert record.negatives is None


def test_full_record_roundtrips_through_json():
    record = make(
        person_id="demo-set:7",
        attributes={"upper_colour": "red"},
        negatives=["a blue caption"],
        extra={"camera": 3},
    )
    assert SampleRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("bad_id", ["", "has space", "dot.in.id", "../x", "-leading"])
def test_rejects_unsafe_sample_id(bad_id):
    with pytest.raises(ValidationError):
        make(sample_id=bad_id)


def test_rejects_empty_or_blank_texts():
    with pytest.raises(ValidationError):
        make(texts=[])
    with pytest.raises(ValidationError):
        make(texts=["ok", "  "])
    with pytest.raises(ValidationError):
        make(negatives=[""])


def test_rejects_person_id_without_source_prefix():
    with pytest.raises(ValidationError):
        make(person_id="7")


@pytest.mark.parametrize("path", ["/abs/path.png", "../up.png", "a/../../b.png"])
def test_rejects_image_path_outside_root(path):
    with pytest.raises(ValidationError):
        ImageRef(path=path)


def test_rejects_degenerate_box():
    with pytest.raises(ValidationError):
        ImageRef(path="a.png", box=(10, 10, 5, 20))
    assert ImageRef(path="a.png", box=(0, 0, 5, 20)).box == (0, 0, 5, 20)


def test_rejects_unknown_fields_and_values():
    with pytest.raises(ValidationError):
        make(unexpected=1)
    with pytest.raises(ValidationError):
        make(split="holdout")
    with pytest.raises(ValidationError):
        make(source="Upper Case")
