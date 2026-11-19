import pytest
import yaml
from pydantic import ValidationError

from mlis.train.core.config import load_run_config

VALID = {
    "run_name": "demo",
    "output_dir": "runs/demo",
    "max_steps": 100,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "warmup_steps": 10,
    "model": {"num_queries": 64},
    "data": {"manifests": ["a.jsonl"]},
}


def write(tmp_path, content):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(content))
    return path


def test_loads_valid_yaml_with_defaults(tmp_path):
    config = load_run_config(write(tmp_path, VALID))
    assert config.precision == "bf16"
    assert config.checkpoint_every_minutes == 30.0
    assert config.resume == "auto"
    assert config.model == {"num_queries": 64}
    assert config.data["manifests"] == ["a.jsonl"]


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_run_config(write(tmp_path, {**VALID, "learning_rte": 1}))


@pytest.mark.parametrize(
    "bad",
    [{"max_steps": 0}, {"learning_rate": 0}, {"precision": "fp16"}, {"warmup_steps": 100}],
)
def test_invalid_values_are_rejected(tmp_path, bad):
    with pytest.raises(ValidationError):
        load_run_config(write(tmp_path, {**VALID, **bad}))


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_run_config(path)


def test_example_configs_in_repo_are_valid(tmp_path):
    from pathlib import Path

    configs = sorted((Path(__file__).parents[2] / "configs").glob("*.yaml"))
    assert configs
    for path in configs:
        load_run_config(path)
