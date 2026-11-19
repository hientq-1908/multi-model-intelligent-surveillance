import os

import pytest
import torch

import mlis.train.core.checkpoint as checkpoint
from mlis.train.core.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_save_and_load_roundtrip(tmp_path):
    payload = {"step": 3, "w": torch.arange(4.0), "nested": {"a": [1, 2]}}
    path = save_checkpoint(tmp_path, 3, payload, keep_last=2)
    assert path.name == "step-00000003.pt"
    loaded = load_checkpoint(path)
    assert loaded["step"] == 3 and torch.equal(loaded["w"], payload["w"])


def test_latest_pointer_and_retention(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, step, {"step": step}, keep_last=2)
    assert find_latest_checkpoint(tmp_path).name == "step-00000004.pt"
    assert sorted(p.name for p in tmp_path.glob("step-*.pt")) == [
        "step-00000003.pt",
        "step-00000004.pt",
    ]


def test_find_latest_falls_back_to_directory_scan(tmp_path):
    save_checkpoint(tmp_path, 7, {"step": 7}, keep_last=2)
    (tmp_path / "latest").unlink()
    assert find_latest_checkpoint(tmp_path).name == "step-00000007.pt"


def test_find_latest_on_empty_or_missing_directory(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "nope") is None


def test_crash_during_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, 1, {"step": 1}, keep_last=3)

    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint.torch, "save", crash)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path, 2, {"step": 2}, keep_last=3)
    monkeypatch.undo()
    assert find_latest_checkpoint(tmp_path).name == "step-00000001.pt"
    assert load_checkpoint(find_latest_checkpoint(tmp_path))["step"] == 1
    assert sorted(os.listdir(tmp_path)) == ["latest", "step-00000001.pt"]
