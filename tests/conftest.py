from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def shard_manifest() -> Path:
    """Committed tiny dataset, images packed in tar shards."""
    return FIXTURES / "tiny_shards" / "manifest.jsonl"


@pytest.fixture
def file_manifest() -> Path:
    """Committed tiny dataset, images as loose PNG files."""
    return FIXTURES / "tiny_files" / "manifest.jsonl"
