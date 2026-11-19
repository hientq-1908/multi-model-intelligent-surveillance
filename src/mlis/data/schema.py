"""Typed sample record shared by every data source and every training stage."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

# No dots: WebDataset splits a member name into key and extension at the first dot.
_SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SOURCE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class LicenceTag(StrEnum):
    """Coarse licence class of the data a sample came from.

    The tag decides where a sample may be stored or shown.
    Everything except PERMISSIVE and CC_BY is research-only and must never be uploaded
    to a public hub or shown in the README or demo (see docs/intent.md, Cost rules).
    """

    PERMISSIVE = "permissive"
    CC_BY = "cc-by"
    NON_COMMERCIAL = "non-commercial"
    RESEARCH_ONLY = "research-only"
    UNSTATED = "unstated"
    SYNTHETIC = "synthetic"


class ImageRef(BaseModel):
    """Pointer to the image of a sample.

    ``path`` is relative to the dataset root named in the manifest header, or the member
    stem inside a shard when ``shard`` is set.
    ``box`` is an optional crop ``(x0, y0, x1, y1)`` in pixels of the referenced image.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    shard: str | None = None
    box: tuple[float, float, float, float] | None = None

    @field_validator("path")
    @classmethod
    def _path_must_be_relative(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError(f"image path must be relative and stay inside the root: {value!r}")
        return value

    @field_validator("box")
    @classmethod
    def _box_must_be_ordered(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is not None:
            x0, y0, x1, y1 = value
            if not (x0 < x1 and y0 < y1) or min(value) < 0:
                raise ValueError(f"box must satisfy 0 <= x0 < x1 and 0 <= y0 < y1: {value}")
        return value


class SampleRecord(BaseModel):
    """One training or evaluation sample.

    Every dataset task converts its source into a stream of these records.
    Stage code reads only this type, never a dataset-specific format.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Unique within one manifest.")
    image: ImageRef
    texts: list[str] = Field(
        min_length=1,
        description="One or more texts for the image: captions, queries or answers.",
    )
    source: str = Field(description="Dataset name in lowercase, for example 'coco-karpathy'.")
    split: Split
    licence: LicenceTag
    person_id: str | None = Field(
        default=None,
        description="Identity label, prefixed by source ('rstpreid:1234'), so ids never collide.",
    )
    attributes: dict[str, Any] | None = Field(
        default=None, description="Attribute name to value, for example {'upper_colour': 'red'}."
    )
    negatives: list[str] | None = Field(
        default=None, description="Hard-negative texts that must not match the image."
    )
    extra: dict[str, Any] | None = Field(
        default=None, description="Source-specific fields that stage code never reads."
    )

    @field_validator("sample_id")
    @classmethod
    def _sample_id_is_safe(cls, value: str) -> str:
        if not _SAMPLE_ID_PATTERN.match(value):
            raise ValueError(f"sample_id must match {_SAMPLE_ID_PATTERN.pattern}: {value!r}")
        return value

    @field_validator("source")
    @classmethod
    def _source_is_lowercase(cls, value: str) -> str:
        if not _SOURCE_PATTERN.match(value):
            raise ValueError(f"source must match {_SOURCE_PATTERN.pattern}: {value!r}")
        return value

    @field_validator("texts", "negatives")
    @classmethod
    def _texts_are_not_blank(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not text.strip() for text in value):
            raise ValueError("texts and negatives must not contain blank strings")
        return value

    @model_validator(mode="after")
    def _person_id_is_namespaced(self) -> SampleRecord:
        if self.person_id is not None and ":" not in self.person_id:
            raise ValueError(f"person_id must look like '<source>:<id>': {self.person_id!r}")
        return self


class ManifestHeader(BaseModel):
    """First line of a manifest file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    dataset: str = Field(
        description="Name of the packed dataset, for example 'coco-karpathy-train'."
    )
    num_samples: int = Field(ge=0)
    root: str | None = Field(
        default=None,
        description="Directory that ImageRef.path is relative to, relative to the manifest file.",
    )
    shards: list[str] = Field(
        default_factory=list, description="Shard files, relative to the manifest."
    )
    created_by: str | None = None
