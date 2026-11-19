import json
import os
import signal

import pytest
import torch

from mlis.train.core.checkpoint import find_latest_checkpoint, load_checkpoint
from tests.train.helpers import make_config, make_trainer


def params(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_resume_after_stop_matches_uninterrupted_run(tmp_path):
    straight_trainer, straight_model = make_trainer(make_config(tmp_path / "straight"))
    straight = straight_trainer.train()

    interrupted_config = make_config(tmp_path / "resumed")
    first, _ = make_trainer(interrupted_config)
    original_step = first._run_optimizer_step

    def stop_after_four() -> None:
        original_step()
        if first.step == 4:
            first.request_stop()

    first._run_optimizer_step = stop_after_four
    partial = first.train()
    assert partial.stopped_early and partial.final_step == 4

    second, resumed_model = make_trainer(interrupted_config)
    resumed = second.train()

    assert resumed.final_step == 10 and not resumed.stopped_early
    assert resumed.losses == straight.losses
    for name, value in params(straight_model).items():
        assert torch.equal(value, params(resumed_model)[name]), name


def test_resume_restores_optimizer_scheduler_and_data_position(tmp_path):
    config = make_config(tmp_path, max_steps=6, checkpoint_every_steps=3)
    trainer, _ = make_trainer(config)
    trainer.train()
    payload = load_checkpoint(find_latest_checkpoint(tmp_path / "checkpoints"))
    assert payload["step"] == 6
    assert payload["scheduler"]["last_epoch"] == 6
    assert payload["optimizer"]["state"]
    assert payload["data"] == {
        "epoch": 0,
        "batch_in_epoch": 6,
    }  # 6 batches of 4 = the 24-sample epoch


def test_checkpoint_holds_only_trainable_parameters(tmp_path):
    trainer, _ = make_trainer(make_config(tmp_path, max_steps=2))
    trainer.train()
    payload = load_checkpoint(find_latest_checkpoint(tmp_path / "checkpoints"))
    assert set(payload["model"]) == {"head.1.weight", "head.1.bias"}


def test_resume_never_starts_from_scratch(tmp_path):
    make_trainer(make_config(tmp_path, max_steps=3))[0].train()
    trainer, _ = make_trainer(make_config(tmp_path, max_steps=3, resume="never"))
    assert trainer.train().losses and trainer.step == 3


def test_resume_from_explicit_path(tmp_path):
    make_trainer(make_config(tmp_path / "a", max_steps=4))[0].train()
    path = find_latest_checkpoint(tmp_path / "a" / "checkpoints")
    trainer, _ = make_trainer(make_config(tmp_path / "b", max_steps=6, resume=path))
    result = trainer.train()
    assert result.final_step == 6 and len(result.losses) == 6


def test_mismatched_checkpoint_is_rejected(tmp_path):
    make_trainer(make_config(tmp_path, max_steps=2))[0].train()
    path = find_latest_checkpoint(tmp_path / "checkpoints")
    payload = load_checkpoint(path)
    del payload["model"]["head.1.bias"]
    torch.save(payload, path)
    trainer, _ = make_trainer(make_config(tmp_path, max_steps=4))
    with pytest.raises(RuntimeError, match="does not match"):
        trainer.train()


def test_grad_accumulation_counts_one_optimizer_step_per_cycle(tmp_path):
    config = make_config(tmp_path, max_steps=3, grad_accum_steps=2)
    trainer, _ = make_trainer(config)
    trainer.train()
    payload = load_checkpoint(find_latest_checkpoint(tmp_path / "checkpoints"))
    assert payload["step"] == 3
    assert payload["data"] == {"epoch": 0, "batch_in_epoch": 6}  # 3 steps x 2 batches


def test_training_reduces_loss(tmp_path):
    result = make_trainer(make_config(tmp_path, max_steps=60, learning_rate=0.05))[0].train()
    assert sum(result.losses[-5:]) < sum(result.losses[:5])


def test_bf16_autocast_runs_on_cpu(tmp_path):
    result = make_trainer(make_config(tmp_path, precision="bf16", max_steps=3))[0].train()
    assert all(loss == loss for loss in result.losses)  # no NaN


def test_sigterm_checkpoints_and_stops_after_current_step(tmp_path):
    trainer, _ = make_trainer(make_config(tmp_path, max_steps=50))
    original_step = trainer._run_optimizer_step

    def send_sigterm_at_three() -> None:
        original_step()
        if trainer.step == 3:
            os.kill(os.getpid(), signal.SIGTERM)

    trainer._run_optimizer_step = send_sigterm_at_three
    result = trainer.train()
    assert result.stopped_early and result.final_step == 3
    assert load_checkpoint(find_latest_checkpoint(tmp_path / "checkpoints"))["step"] == 3


def test_signal_handlers_are_restored_after_training(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    make_trainer(make_config(tmp_path, max_steps=1))[0].train()
    assert signal.getsignal(signal.SIGTERM) is before


def test_periodic_checkpoint_follows_the_clock(tmp_path):
    now = [0.0]

    def clock() -> float:
        return now[0]

    config = make_config(
        tmp_path, max_steps=6, checkpoint_every_minutes=30.0, keep_last_checkpoints=10
    )
    trainer, _ = make_trainer(config, clock=clock)
    original_step = trainer._run_optimizer_step

    def tick() -> None:
        original_step()
        now[0] += 20 * 60  # twenty minutes per step

    trainer._run_optimizer_step = tick
    trainer.train()
    events = [json.loads(line) for line in (tmp_path / "run_log.jsonl").read_text().splitlines()]
    periodic = [
        e["step"] for e in events if e["event"] == "checkpoint" and e["reason"] == "periodic"
    ]
    assert periodic == [2, 4, 6]
