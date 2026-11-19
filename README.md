# mlis - person search and description for surveillance video

A multi-model vision-language pipeline: search recorded video for a person by text description, then describe that person.
DINOv3 (frozen) feeds a Q-Former, a projector maps its 64 query tokens into Qwen3, and Qwen3 is trained with LoRA only.
It is a personal self-learning project and a CV piece, not a deployed surveillance product.

The plan is in [docs/intent.md](docs/intent.md).
This repository currently holds the shared foundation: the data contract, the model interfaces and the trainer core.
There is no training code per stage, no dataset packing and no trained weights yet.

## Research-only data notice

This project is for self-learning and research only.
No commercial use of the code, data or any trained weights.

The author owns none of the person-search data.
CUHK-PEDES is used from an unofficial Hugging Face copy of data its authors release on request only.
ICFG-PEDES and RSTPReid state no dataset licence and are built from MSMT17, whose agreement allows results to be published but not images.
MALS, UCA and other sources are research-only or non-commercial.
Treat all of them as research-only.

The repository never contains dataset images, captions, processed annotations, cached features or model weights, apart from a tiny synthetic test fixture.
No image from MSMT17, CUHK-PEDES or RAP is shown in this README, the write-up or the demo.
Datasets with unverified or restrictive terms are never uploaded to the Hugging Face Hub.
The licence of every dataset is listed in [docs/intent.md](docs/intent.md#data).

## Layout

```
src/mlis/
  data/              data contract: schema, manifest, shards, reader, validation, synthetic fixture
    general/         D1  general image-text sources (COCO Karpathy, BLIP3o CC12M shards)
    person/          D2  person sources (CUHK-PEDES, ICFG-PEDES, RSTPReid, MALS)
    built_text/      D3  code-built text sets (attribute captions, hard negatives, queries, questions)
    instruction/     D4  instruction data (mix665k, Visual Genome person crops, UCA frames)
    leakage/         D5  leakage filter against the evaluation sets
  models/            image encoder, DINOv3 wrapper, Q-Former, projector, LLM wrapper (Qwen3)
  train/
    core/            shared trainer: config, seed, checkpoints, run log, loop
    stage1/          T1  frozen DINOv3 + Q-Former, ITC / ITM / ITG
    stage23/         T2  projector + Qwen3 LoRA
  eval/              T3  evaluation harness
  preflight/         T3  cloud preflight
configs/             one YAML file per run
docs/                intent and the data contract
tests/               CPU tests, mirroring src/, plus tests/fixtures
```

Each later task adds files only inside its own area.
The subpackages marked with a task id are empty on purpose and only importable.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
uv sync
```

`uv.lock` is committed, so every machine resolves the same versions.
Add a dependency with `uv add <package>`.

## Run the tests

```bash
uv run pytest                 # CPU tests only, no downloads, a few seconds
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
```

CI runs the same three commands on every pull request and on `main`.

Tests that load real weights from the Hugging Face Hub are marked `network` and skipped by default.
Run them on a machine that has access to the gated DINOv3 repository:

```bash
uv run pytest -m network
```

## Data contract

All data flows through one typed record, a JSONL manifest and WebDataset tar shards.
The full description is in [docs/data-contract.md](docs/data-contract.md).

```python
from mlis.data.reader import iter_samples

for sample in iter_samples("tests/fixtures/tiny_shards/manifest.jsonl"):
    print(sample.record.sample_id, sample.record.texts, sample.image.size)
```

Validate any packed dataset with:

```bash
uv run python -m mlis.data.validate path/to/manifest.jsonl
```

## Model interfaces

| Part | Interface | Implementations |
|---|---|---|
| Image encoder | `ImageEncoder`: pixels `[B, 3, H, W]` to patch tokens `[B, N, D]` | `Dinov3Encoder` (frozen DINOv3 ViT-L/16, registers dropped), `RandomImageEncoder` (CPU test stand-in) |
| Q-Former | `QFormerBase`: image tokens to 64 x 768 query tokens plus a pooled retrieval embedding | `QFormer`, with a BERT-base initialisation path (`QFormer.from_bert_base`) |
| Projector | `Projector`: query tokens to the LLM hidden size | `MLPProjector` |
| LLM | `LLMWrapper`: input embeddings in, logits and loss out | `Qwen3Wrapper` (`Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-0.6B` for smoke tests) |

These are interfaces with minimal working implementations, sized for CPU tests.
They have no losses and no training logic.
ITC, ITM and ITG belong to stage 1, and LoRA belongs to stages 2 and 3.

## Trainer core

`mlis.train.core` holds what every stage shares:

- one YAML file per run, validated into `RunConfig` (see `configs/example_tiny.yaml`);
- bf16 autocast, gradient accumulation and gradient clipping;
- a checkpoint every N minutes, at the end of the run and on SIGTERM or SIGINT, written atomically;
- exact resume of the model, optimizer, scheduler, step, every RNG and the data position;
- an append-only run log (`run_log.jsonl`) and a `summary.json` with wall-clock and GPU hours;
- `seed_everything` and a resumable batch stream.

Checkpoints hold only the parameters that require gradients, so LoRA and adapter runs do not write the frozen base model.
A crash in the middle of a step writes nothing, and the next run resumes from the last checkpoint.

## Licence

Code: Apache-2.0, see [LICENSE](LICENSE).
Datasets and model weights keep their own terms.
