import json
import random

import numpy as np
import torch

from mlis.train.core.data_position import IndexedBatchStream
from mlis.train.core.run_log import RunLog
from mlis.train.core.seed import capture_rng_state, restore_rng_state, seed_everything


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_is_reproducible():
    seed_everything(11)
    first = draw()
    seed_everything(11)
    assert draw() == first
    seed_everything(12)
    assert draw() != first


def test_restore_rng_state_replays_the_same_stream():
    seed_everything(1)
    state = capture_rng_state()
    expected = draw()
    restore_rng_state(state)
    assert draw() == expected


def test_run_log_counts_gpu_hours_and_carries_over_resume(tmp_path):
    now = [0.0]
    log = RunLog(tmp_path, num_gpus=2, clock=lambda: now[0])
    now[0] = 1800.0
    assert log.wall_seconds == 1800.0
    assert log.gpu_hours == 1.0  # half an hour on two GPUs

    state = log.state_dict()
    resumed = RunLog(tmp_path, num_gpus=2, clock=lambda: now[0])
    resumed.load_state_dict(state)
    now[0] += 1800.0
    assert resumed.gpu_hours == 2.0


def test_run_log_writes_events_and_summary(tmp_path):
    log = RunLog(tmp_path, num_gpus=0)
    log.event("step", step=1, loss=0.5)
    log.event("end", step=1)
    log.write_summary(final_step=1)
    events = [json.loads(line) for line in (tmp_path / "run_log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["step", "end"]
    assert events[0]["loss"] == 0.5 and events[0]["gpu_hours"] == 0.0
    assert json.loads((tmp_path / "summary.json").read_text())["final_step"] == 1


def collate(items):
    return list(items)


def test_batch_stream_covers_each_sample_once_per_epoch_and_reshuffles():
    stream = IndexedBatchStream(list(range(12)), 4, seed=0, collate=collate)
    epoch_one = [next(stream) for _ in range(3)]
    epoch_two = [next(stream) for _ in range(3)]
    assert sorted(x for b in epoch_one for x in b) == list(range(12))
    assert sorted(x for b in epoch_two for x in b) == list(range(12))
    assert epoch_one != epoch_two


def test_batch_stream_position_restores_exactly():
    stream = IndexedBatchStream(list(range(12)), 4, seed=3, collate=collate)
    for _ in range(4):
        next(stream)
    state = stream.state_dict()
    expected = [next(stream) for _ in range(5)]
    fresh = IndexedBatchStream(list(range(12)), 4, seed=3, collate=collate)
    fresh.load_state_dict(state)
    assert [next(fresh) for _ in range(5)] == expected
