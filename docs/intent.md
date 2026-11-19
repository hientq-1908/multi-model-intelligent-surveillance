# Intent: person search and description for surveillance video

Status: draft, updated 2026-11-16 with the author's decisions on data, publishing, the main model, cloud training, the person-search data sources and reference-only scores.
Updated 2026-11-19 with the corrections from a feasibility review dated 2026-11-18 and four further decisions; see [Decisions](#decisions).
Items marked **(proposed)** were suggested by Claude and not yet confirmed.
See [Open questions](#open-questions).

Architecture reference: [idea.md](idea.md).
Requirements and design: [spec.md](spec.md).

- Repository: `git@github.com:hientq-1908/multi-model-intelligent-surveillance.git`
- Work tracking: Jira project MLIS, <https://hientq.atlassian.net/jira/software/projects/MLIS/boards/1>

## Purpose

Build a multi-model vision-language pipeline end to end, as a personal self-learning project and a CV piece.
There is one author, no team and no commercial use.

The priority is to show what the pipeline can do and how it was built.
Benchmark scores are a sanity check for the author's reference, not the goal, and are not compared with published results.

## Use case

Search recorded surveillance video for a person by text description, then describe what that person looks like, where they are, and what they do.

Example: the query "person with a red T-shirt and hat" returns matching people.
For each match the system writes a description of the person and their surroundings, and for a tracked person a timestamped timeline of their actions.

## Users

- **Primary reader:** someone reviewing the author's CV (recruiter, interviewer, engineer).
  The project is a personal showcase.
- **Demo user:** a person who types a description and inspects the results on a fixed set of public clips.

There is no production operator.
This is not a deployed surveillance product.

## System

```
Indexing:  video -> frames -> person detector + tracker -> crops
           -> DINOv3 + Q-Former (stage 1 model) -> vectors in an index

Query:     text -> vector search -> ITM re-rank -> top matches
           -> crop with margin -> full model (Qwen3) -> description
           -> track segments -> timeline and highlights
```

| Part | Role | Trained here |
|---|---|---|
| DINOv3 ViT-L/16 | Image features | No, frozen |
| Q-Former, 64 queries | Compress and align to language; search vectors | Yes |
| Projector | Map to the LLM input space | Yes |
| Qwen3-4B | Describe, confirm matches, summarise | LoRA only |
| Person detector and tracker | Find and follow people | No, off the shelf |

"Fully finetune" in the original request means training the adapter parts and LoRA across all three stages.
The backbones stay frozen.

## Phases

Each phase ends with a working result.
The phases are capabilities of one model: there is a single stage 1 Q-Former, used for the index and inside the full model, and stages 2 and 3 each run once.
Phase 1 is therefore checked on the final model, after the person data is in.

| # | Result | Status |
|---|---|---|
| 1 | Three-stage pipeline on general data: image search, captioning, question answering on single images | Must-have, first milestone |
| 2 | Person search: detector, crop index, text query, ITM re-ranking | Must-have, first milestone |
| 3 | Description of each match and its surroundings, from one enlarged crop | Must-have, first milestone |
| 4a | Track timeline: change-based frame picking, per-segment descriptions, text summary; highlights as timestamps | Experiment option, after the first milestone |
| 4b | True multi-frame input, trained with multi-frame data | Experiment option, after the first milestone |

Phase 4 is an experiment: build both options and compare them on the same clips.
It is outside the first milestone (see [Decisions](#decisions)).

## Experiments

Each compares two choices on the same data and evaluation, and is reported in the write-up.

| # | Question | Option A | Option B | Compared on |
|---|---|---|---|---|
| E1 | Should the Q-Former train after stage 1? | Frozen in stages 2 and 3. Outputs can be cached. Run first | Trained in stage 2, and optionally 3. No caching | Caption and answer quality on a fixed set, GPU-hours |
| E2 | How should a long track be described? | 4a: describe segments, then summarise as text | 4b: several frames in one prompt, with multi-frame training | MEVA reference timelines, same clips |

`idea.md` does not say which option the original author used for E1.
E1 is part of the first milestone; E2 comes after it with phase 4.
Option A's cost advantage is small: the DINOv3 and Q-Former forward pass for 300k images takes 10 to 15 minutes, under 2% of an LLM stage, so option B's extra cost is mainly the Q-Former backward pass, which is small next to the 4B LLM (review estimate).
E1 is therefore a quality comparison more than a cost one.

## Deliverables

- Public GitHub repository with one training script per stage.
  Code licence: Apache-2.0.
- Results: metrics, examples and failure cases from the trained model, in the repository and the write-up.
- No trained weights are released (see [Decisions](#decisions)).
  DINOv3 and Qwen3 are not redistributed either.
- Demo on a fixed set of clips, run locally from the repository.
  It is not hosted.
- Write-up: each stage, design reasons (why a Q-Former with a text-free encoder), what broke, what was learned.

## Success criteria **(proposed)**

- Each stage passes a small check: retrieval after stage 1, captions after stage 2, answers after stage 3.
- Person search is reported as Recall@1/5/10, for reference only, with no target number.
- Descriptions are judged by hand on a fixed example set, with failures shown.

## Non-goals

- Matching or beating InA-Bridge's reported scores, or any leaderboard.
- Reading text in images, counting, or small-object detail.
  The 64-token design is weak at these.
- Live camera feeds or real-time operation.
- Identifying people by name or face.
- Unfreezing DINOv3 or the Qwen3 base weights.
- Generating video.
  Highlights are timestamps into the original clip.
- Releasing trained weights.

## Constraints

- **Use:** self-learning and research only.
  No commercial use of the code, data or trained weights.
- **Compute:** all training runs on a rented RTX A6000 48 GB on RunPod community cloud, in bf16, including smoke tests.
  The local machine (one 8 GB GPU) is for writing code, CPU tests and demo inference, where Qwen3-4B runs in NF4.
  See [Compute plan](#compute-plan).
- **Data:** public datasets only, for research use, with no commercial use.
  The project collects no footage of its own, and the demo shows MEVA clips only.
  The person-search sets are for research use, and their licences are unstated or non-commercial.
- **Known technical limits:**
  - DINOv3 needs bf16 or fp32; a maintainer states it does not support fp16 (<https://github.com/facebookresearch/dinov3/issues/181>).
  - The Hugging Face DINOv3 processor resizes every image to 224x224 by default, which squashes tall person crops; its RoPE accepts non-square input with each side a multiple of 16 (<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>).
  - DINOv3 jitters RoPE coordinates in train mode, so the frozen encoder stays in `eval()` mode.
  - Qwen3 thinking mode must be off.
    With `enable_thinking=False` the Qwen3-4B chat template still emits an empty `<think>\n\n</think>\n\n` block on assistant turns (<https://huggingface.co/Qwen/Qwen3-4B>).
  - Drop DINOv3 register tokens and feed patch tokens to the Q-Former **(proposed)**.
    ViT-L/16 has 4 registers after CLS, so the Hugging Face output layout is `[CLS, 4 registers, patches]` and the patch tokens are `last_hidden_state[:, 5:, :]`.

## Data

Rule: public datasets only, with no manual labelling.
Datasets may be combined and converted by code or by an AI model.

Status key: **V** = size or licence checked on the web on 2026-11-16, or by the feasibility review on 2026-11-18. **U** = not yet confirmed on an official page.
Subset sizes are proposals and depend on the compute budget.

### Stage 1: Q-Former (image-text pairs)

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| COCO Captions | ~118k images, 5 captions each | Annotations CC BY 4.0, images under Flickr terms ([COCO terms]) | General base, run first | V |
| CC3M | 3,318,333 web image-caption training pairs, by URL. About 61% of a 400-URL sample still resolved on 2026-11-18. The repository was archived on 2026-04-19 | "May be freely used for any purpose" ([CC3M]) | Optional 300k to 500k subset | V |
| CUHK-PEDES | Official training split: 34,054 images, 11,003 identities, 68,126 descriptions (about 2 each) | Authors release on request, academic only ([CUHK-PEDES]). Used here from an unofficial Hugging Face copy, which has no licence tag and no statement of permission ([CUHK-PEDES HF copy]) | Person search training and a test split made here | V |
| ICFG-PEDES | 54,522 images, 4,102 identities, 1 description each; train 34,674, test 19,848. "All images are gathered from the MSMT17 database" ([SSAN paper]) | Public download: since 2026-06-06 the SSAN repository links a Baidu Pan share ([SSAN]). No terms of its own; MSMT17 terms apply in spirit | Official training split for training; official test split as a reference test set | V |
| RSTPReid | 20,505 images, 4,101 identities, 2 descriptions each; train 18,505, val 1,000, test 1,000. Built from MSMT17, with files renamed | Public Google Drive download, linked from the official repository. The repository's MIT licence covers "the Software"; there are no dataset terms ([RSTPReid]) | Official training split for training; official test split as a reference test set | V |
| MSMT17 (source of ICFG-PEDES and RSTPReid, not downloaded) | Person re-identification images | Official page offline since July 2019. Its agreement: academic research only, no further distribution, "The performance computed on MSMT17 can be reported and released. No images in MSMT17 could be published" | No MSMT17-derived image is shown in the README or demo | V |
| MALS | 1,510,330 synthetic person images with captions and 27 attributes. Generated with Stable Diffusion prompted by CUHK-PEDES and ICFG-PEDES captions, then re-captioned by BLIP ([APTM paper]) | "Can only be used for research; any commercial usage is forbidden" ([APTM]) | 200k to 300k subset. Images are generated, so no real people | V |
| PA-100K | 100k pedestrian images, 26 attributes | CC BY 4.0 ([PA-100K]) | Attribute source for generated captions | V |
| UPAR | Unified attributes with colours, over PA-100K, PETA, RAPv2, Market-1501 | Annotations CC BY-NC-SA 3.0 DE, images follow their sources ([UPAR]). RAPv2 needs an agreement signed by someone with legal authority for an organisation ([RAP]) | Colour attributes for generated captions | V |

CUHK-PEDES has 40,206 images and 80,412 descriptions in all, but its authors release it only on request.
This project uses an unofficial Hugging Face copy of the training split (`MaulikMadhavi/CUHK-PEDES-processed`, 34,052 rows, two fewer than the official training split), joins it to the authors' public `reid_raw.json` for identities and original captions, and makes its own train, validation and test split by identity.
Joining by file basename works for 34,050 images; two VIPeR names (`000_45.bmp`, `339_90.bmp`) occur in both `cam_a` and `cam_b` and need the camera folder to resolve.
Decided by the author on 2026-11-16 and confirmed on 2026-11-19 (see [Decisions](#decisions)).
The author owns none of this data.
Below, "PEDES" means CUHK-PEDES, ICFG-PEDES and RSTPReid.

### Stage 2: projector alignment (captioning through the LLM)

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| BLIP3o-Pretrain-Long-Caption | ~27.2M image and long-caption pairs (about 120 tokens, written by Qwen2.5-VL-7B), WebDataset shards: 2,891 shards, 1,374 GB in all | Card says Apache-2.0, but the shards hold images under their source terms: `sa_*` shards (565 GB) are SA-1B, "research purposes only"; `webdataset_JDB_*` shards (172 GB) are JourneyDB, non-commercial and not to be redistributed; `webdataset_shard_*` shards (637 GB) are CC12M-sourced, and CC12M "may be freely used for any purpose" ([BLIP3o], [CC12M]) | 300k to 500k subset, taken as whole shards | V |
| CUHK-PEDES, ICFG-PEDES, RSTPReid | Same as stage 1 | Same | Captions reused as "describe this person" targets | as above |

### Stage 3: instruction tuning

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| `llava_v1_5_mix665k` | 665,298 image conversations: COCO 364,100, Visual Genome 86,417, OCR-VQA 80,000, GQA 72,140, TextVQA 21,953, text-only 40,688. 563,345 remain after dropping OCR-VQA and TextVQA | JSON CC BY 4.0, and use "should abide by the policy of OpenAI"; images per source ([mix665k]). `visualgenome.org` no longer serves the dataset; use the University of Washington mirror | General instruction following. Drop the OCR-VQA and TextVQA parts, since text reading is a non-goal | V |
| UCF-Crime Annotation (UCA) | 1,854 surveillance videos, 23,542 timestamped sentences; train 1,165, val 379, test 310 videos. The README says 110.7 hours; the listed durations sum to 121.9 hours. Videos come from UCF-Crime | Repository is Apache-2.0, but the CVPR 2024 supplement says the data is "intended solely for academic and research purposes" ([UCA]) | Frames from each event span paired with its sentence: surveillance-view captions | V |
| UCF-Crime (source of UCA videos) | 1,900 videos, 128 hours, collected from YouTube and LiveLeak; one 102.96 GB zip | No licence or terms stated on the page ([UCF-Crime]) | Video source for UCA only | V |
| SurveillanceVQA-589K | ~589k question-answer pairs on surveillance video, 12 question types. No videos included; sources are MSAD (201 videos), MEVA (720), NWPU (255) and UCA (1,854), split at random at clip level | Hugging Face tag MIT, but the paper only plans to "release ... under a research-friendly license" ([SurveillanceVQA paper]) | Surveillance questions for 4b, only if its videos are shown not to overlap the UCA test split (spec 7.4). They do overlap: its training split holds clips from 308 of the 310 UCA test videos, so it is not used unless filtered by video name | V |

### Phase 4b: multi-frame training

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| UCA | As above | As above | Several frames per event with its sentence | as above |
| M4-Instruct | ~1.18M multi-image and video instruction samples. The Hugging Face repository holds 615,814 multi-image and 255,000 video samples; the 1.18M total includes 307k single-image samples not in it. 36 zips, 233.8 GB, plus about 139 GB of video frames for the video part | CC BY 4.0, plus the OpenAI policy clause ([M4-Instruct]) | Small subset of the multi-image components, to teach several images in one prompt | V |
| LLaVA-Video-178K | 178k video captions, ~961k open QA, 1,281 GB | Card says academic research and education only | Not used; M4-Instruct is chosen (spec 7.3) | V |

### Demo and evaluation video (not used for training)

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| MEVA | Multi-camera surveillance-style video with consenting actors, 1080p at 30 fps; 5-minute clips of 70 to 250 MB. 37 activity types with tracks, but boxes exist only for actors and props involved in annotated activities, not for every person ([MEVA repository], [MEVA paper]) | CC BY 4.0, free download from AWS ([MEVA licence]) | Demo clips; activity annotations as reference timelines for phase 4, scored on annotated actors only | V |

[COCO terms]: https://cocodataset.org/dataset/termsofuse.htm
[CC3M]: https://github.com/google-research-datasets/conceptual-captions
[CUHK-PEDES]: https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description
[CUHK-PEDES HF copy]: https://huggingface.co/datasets/MaulikMadhavi/CUHK-PEDES-processed
[SSAN]: https://github.com/zifyloo/SSAN
[SSAN paper]: https://arxiv.org/abs/2107.12666
[RSTPReid]: https://github.com/NjtechCVLab/RSTPReid-Dataset
[APTM]: https://github.com/Shuyu-XJTU/APTM
[APTM paper]: https://arxiv.org/abs/2306.02898
[PA-100K]: https://github.com/xh-liu/HydraPlus-Net
[UPAR]: https://github.com/speckean/upar_dataset
[RAP]: https://github.com/dangweili/RAP
[BLIP3o]: https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption
[CC12M]: https://github.com/google-research-datasets/conceptual-12m
[mix665k]: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json
[UCA]: https://github.com/Xuange923/Surveillance-Video-Understanding
[UCF-Crime]: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
[SurveillanceVQA paper]: https://arxiv.org/abs/2505.12589
[M4-Instruct]: https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data
[MEVA licence]: https://mevadata.org/resources/MEVA-data-license.txt
[MEVA repository]: https://gitlab.kitware.com/meva/meva-data-repo
[MEVA paper]: https://arxiv.org/abs/2012.00914

### Data built by code or model

| Built set | From | Method | Feeds |
|---|---|---|---|
| Attribute captions | PA-100K, UPAR, MALS attributes | Code template turns attributes into a sentence; optional rewording by a text LLM | Stage 1 |
| Hard negative captions | Any attribute or PEDES caption | Code swaps one attribute (red to blue, hat to no hat) to make a near-miss | Stage 1 ITM. Targets the "red shirt and hat on the same person" weakness |
| Attribute yes/no questions | PA-100K, UPAR | Code writes "Is this person wearing a hat?" with balanced yes and no | Stage 3, for the confirm step |
| Match checks | PEDES pairs plus hard negatives | Code writes "Does this person match: <text>?" | Stage 3, for the confirm step |
| Person question-answer pairs | PEDES captions | Text LLM turns each caption into questions and answers, from the caption only | Stage 3 |
| Surveillance frame captions | UCA | Code samples frames inside each timestamped event | Stage 3, phase 4b |
| Reference timelines | MEVA activity annotations | Code turns "actor, activity, start, end" into timeline text | Phase 4 evaluation only |
| Person region descriptions | Visual Genome regions and relationships on person boxes (images already in `llava_v1_5_mix665k`) | Code crops each person box with the inference margin and pairs it with its region text | Stage 3, for describing action and place from a crop with margin |
| Short queries | Attribute templates, PEDES captions | Code takes one to three attribute phrases or a single clause; optional short rewording by a text LLM | Stage 1, so search works on the 5 to 10 words a user types. CUHK-PEDES captions average 23.6 words (median 21) |
| Multi-frame timeline targets | UCA | Code writes every sentence inside a window of frames as `[start-end] sentence` lines | Phase 4b |

### Evaluation sets

- General search: COCO Karpathy 5k test split.
  These images, and the POPE images, are excluded from all training data.
  4,407 of the 5,000 Karpathy test images are in COCO train2017, so stage 1 trains on Karpathy train plus restval (113,287 images), not on train2017.
  `llava_v1_5_mix665k` touches 2,038 Karpathy test images (through its COCO, Visual Genome and GQA parts) and 194 POPE images, so it is filtered by COCO id.
- Person search: RSTPReid and ICFG-PEDES official test splits, and a CUHK-PEDES test split of 1,000 identities made here from the official training split, Recall@1/5/10.
  Training images that are near-duplicates of these test images, and images of the same MSMT17 people in the other MSMT17-based set, are dropped, because the person datasets share sources.
  RSTPReid uses all 4,101 MSMT17 identities and ICFG-PEDES 4,102, so every RSTPReid test identity also appears in ICFG-PEDES.
  RSTPReid renamed its files, so this rule needs image near-duplicate search, not file names.
  CUHK-PEDES source images include Market-1501 and VIPeR, which UPAR (through Market-1501 and PETA) also covers.
- Person search on surveillance-style images: attribute retrieval on the PA-100K test split, which needs no access request; hand-judged precision at 5 on fixed MEVA queries.
- Descriptions: UCA test sentences; fixed hand-checked examples from MEVA.
- Phase 4: MEVA reference timelines, option 4a against 4b.
  MEVA demo and evaluation clips are picked outside the 720 MEVA videos that SurveillanceVQA-589K uses.

### Data risks

- ICFG-PEDES and RSTPReid state no dataset licence, and both are built from MSMT17, whose agreement allows only results, not images, to be published.
  CUHK-PEDES comes from an unofficial copy of data its authors release on request only.
  All three are treated as research-only, and the README states that the author owns none of them.
  Adding person data after the person run means rerunning the person run and stages 2 and 3.
- Several licences are non-commercial or research-only.
  This matches the project's research-only use.
  The repository lists every dataset with its licence, and the demo carries a research-only notice.
  No weights are released (see [Decisions](#decisions)), which removes the question of what the person-search sets and the DINOv3 licence allow for released weights.
- BLIP3o's shards mix CC12M-sourced images with SA-1B (research only) and JourneyDB (non-commercial) images, so the licence of a subset depends on which shards it takes.
- MALS images were generated from CUHK-PEDES and ICFG-PEDES captions.
  Whether test-split captions were among the prompts is not known.
- UCF-Crime is mostly crime and accident footage, so it skews what the model describes and contains violent scenes.
- Generated captions and questions carry the errors of the model or template that made them.
- A single frame cannot show an event that unfolds over time.
  UCA sentences are paired with single frames only for short spans, balanced between normal and anomaly videos and capped at a small share of stage 3; full events are used for phase 4b.
- Access can stall the start: DINOv3 on Hugging Face is approved by hand, and a maintainer says affiliation and country are common reasons for rejection (<https://github.com/facebookresearch/dinov3/issues/376>).
- A design review on 2026-11-16 found the gaps these rules close.
  See [spec.md](spec.md), section 2.9.

## Compute plan

The money budget is not a constraint: the planned GPU-hours cost well under $100 at review-time prices (see [Cost estimate](#cost-estimate)).
This plan is stated in compute: GPU memory and GPU-hours.
All figures are estimates, not measurements.
Rule: time the first 200 steps of every run, record peak memory, and extrapolate before letting it continue.

### Hardware (decided)

- Training: one RTX A6000 48 GB on RunPod community cloud, Qwen3-4B in bf16 for stages 2, 3 and 4b.
  No 4-bit training.
- Storage: a persistent RunPod network volume holds the datasets and checkpoints, so they survive a lost pod.
- Checkpoints every 30 to 60 minutes, because community hosts can disappear.
- Local demo: the 8 GB GPU, with Qwen3-4B in NF4.
  The NF4 demo model is not the bf16 model that was trained and evaluated.

### GPU memory per stage (estimated)

| Stage | What sits in memory | Qwen3-4B, bf16 |
|---|---|---|
| 1. Q-Former | DINOv3 frozen (0.6 GB in bf16, no gradients), Q-Former with three loss passes, batch ~128 | 15 to 20 GB without activation checkpointing; 12 to 16 GB needs checkpointing or a batch near 64. No LLM |
| 2. Projector | LLM frozen (8.05 GB of weights), gradients pass through it; long captions, micro-batch 8 to 16 with gradient accumulation | 14 to 20 GB with activation checkpointing; the fp32 logits over the 151,936-token vocabulary are the largest item unless the LM head runs only at target positions |
| 3. LoRA | LLM with LoRA, conversations, micro-batch 8 to 16 | 14 to 20 GB with activation checkpointing |
| 4b. Multi-frame | 8 to 16 frames (512 to 1,024 visual tokens), batch ~4 | About the same token count as stage 2, so similar at micro-batch 1 or 2 |
| Inference and demo | One image or track at a time, Qwen3-4B in NF4 (2.65 GB), DINOv3, Q-Former, projector, detector | About 4.5 to 5.5 GB in all, so every model fits the 8 GB GPU at once |

All stages fit a 48 GB card in bf16 with room to spare; they would also fit 24 GB in bf16 with micro-batching.
The server also needs 32 GB of system RAM and 4 to 8 CPU cores for data loading (stage 1 needs about 100 to 150 decoded images per second).

Disk: the planned datasets are about 330 to 375 GB as downloaded and 105 to 160 GB once repacked, with model weights (about 25 GB) and resumable checkpoints (10 to 15 GB) included.
The persistent volume is sized from these figures.

### Where each job runs

| Job | Machine |
|---|---|
| All code, CPU tests with tiny stand-in models, template-built sets | Local machine |
| Packing large datasets, model-built sets | Cloud server, writing to the persistent volume (a CPU-only server is enough for packing) |
| Smoke tests: overfit 100 samples per stage with Qwen3-0.6B | Cloud server |
| Training runs | Cloud server, one RTX A6000 48 GB, bf16 |
| Detector, tracker, indexing, demo inference | Local machine, or the cloud server if 8 GB is not enough |

No full training run starts until the cloud preflight has passed on that server: CPU tests, real-weight checks, the smoke test, a checkpoint save-and-resume round trip, then the 200-step timing projection.
See [spec.md](spec.md), section 7.11.

### LLM size ladder

| Model | Hidden size | Role |
|---|---|---|
| Qwen3-0.6B | 1024 | Cloud smoke tests |
| Qwen3-1.7B | 2048 | Fallback only; not expected to be needed, since money is not a constraint (see [Cost estimate](#cost-estimate)) |
| Qwen3-4B | 2560 | Main model for all phases |

Stage 1 does not involve the LLM, so one trained Q-Former serves all three sizes.
Only the projector and LoRA are retrained per LLM.

### Small recipe

| Run | Data | Estimated RTX 4090 GPU-hours |
|---|---|---|
| Stage 1, general | COCO Captions, ~590k pairs, about 5 epochs, plus the BLIP3o subset with shortened captions | 9 to 16 |
| Text generation | Model-built sets, batched | 3 to 5 |
| Stage 1, person | CUHK-PEDES, ICFG-PEDES and RSTPReid, MALS subset, generated captions, ~400k pairs | 3 to 5 |
| Stage 2, Qwen3-4B | BLIP3o subset 300k, plus person descriptions, 1 epoch | 11 to 21 |
| Stage 3, Qwen3-4B | `llava_v1_5_mix665k` subset 300k, plus person questions and UCA frames | 29 to 39 |
| Phase 4b, Qwen3-4B | UCA events, M4-Instruct subset ~50k | 13 to 27 |
| **Planned total** | | **70 to 115** |
| Reruns and failed runs | Roughly the same again | 60 to 100 |
| Preflight, packing, feature caching on the server | | 5 to 15 |

These hours were first estimated for one RTX 4090 with Qwen3-4B in 4-bit, for a realistic total with reruns of 135 to 230 GPU-hours.
The review's FLOP arithmetic and published training costs (LLaVA-1.5, <https://github.com/haotian-liu/LLaVA>; MiniGPT-4, <https://arxiv.org/abs/2304.10592>) find them plausible and probably 1.2 to 2 times conservative.
On the chosen RTX A6000 in bf16, which is about 0.88 times the 4090 4-bit hours (slightly slower on stage 1, faster on the LLM stages because bf16 LoRA runs about 1.34 times faster than NF4, <https://arxiv.org/abs/2311.03687>), the realistic total is about **118 to 202 GPU-hours**.
None of these figures are measured.

### Cost estimate

Prices are review-time estimates from 2026-11-18 and will change.

| Item | Price | Estimate |
|---|---|---|
| RTX A6000 48 GB, RunPod community, 118 to 202 GPU-hours | $0.33 per GPU-hour (<https://www.runpod.io/pricing>) | about $39 to $67 |
| Persistent network volume | $0.07 per GB-month for the first TB, no ingress or egress fees (<https://docs.runpod.io/pods/pricing>) | about $28 per month for 400 GB |

A CPU-only packing machine for a few days is small next to the GPU cost.

### Cost rules

- Pack each dataset into shards and keep them on the persistent RunPod volume, so each pod mounts them instead of downloading again.
  Sets with unverified or restrictive terms (MSMT17-derived sets, CUHK-PEDES, MALS, UCA, JourneyDB shards) are never uploaded to the Hugging Face Hub.
- Cache what is frozen.
  Q-Former outputs for stages 2 and 3 are about 96 KiB per image (64 x 768 in bf16), if the Q-Former is kept frozen there.
- Save checkpoints to the persistent volume every 30 to 60 minutes and make every script resumable, because community-cloud hosts can disappear.
- Stop the machine when a run ends.
  Log GPU-hours per run against the estimate.
- Change one thing per rerun.

### Order of work **(proposed)**

Code is written by a coding agent, so coding is not the limiting step.
The schedule is set by GPU run time, data download and packing, and the author's review of each result.

The first milestone is phases 1 to 3 plus experiment E1, in 4 to 6 weeks (see [Decisions](#decisions)).
Phase 4 and E2 come after it.

| Step | Work | What it waits on | Estimated elapsed time |
|---|---|---|---|
| 0 | Start downloads; request DINOv3 access; check dataset terms | Nothing. Do first | 1 day to start |
| 1 | Pipeline code with CPU tests, data packing, first cloud session with preflight for all three stages | Downloads, DINOv3 approval and upload speed | 2 to 3 days |
| 2 | Baseline, stage 1 general run and retrieval check; detector, tracker and index code | 9 to 16 GPU-hours | 2 to 3 days |
| 3 | Stage 1 person run with the three PEDES sets, which fixes the single Q-Former; person search: phase 2 complete | 3 to 5 GPU-hours | 2 to 3 days |
| 4 | All instruction data; stage 2 once, stage 3 once; confirm and describe: phases 1 and 3 complete | 40 to 60 GPU-hours, review between stages | 5 to 8 days |
| 5 | Experiment E1: option B reruns stage 2, and optionally stage 3, with the Q-Former trained | E1 reruns | 3 to 5 days |
| 6 | Local demo, publish code and results, write-up: first milestone complete | Author's review | 2 to 3 days |
| 7 | After the first milestone: phase 4a, then 4b, and experiment E2 | 13 to 27 GPU-hours | 3 to 6 days |

GPU-hours in this table are the RTX 4090 estimates from the small recipe.
Steps 0 to 6 sum to about 17 to 26 working days with no reruns.
Reruns add GPU-hours and days in whichever step they occur, and multi-stage models usually need two to four iterations of stage 1 and of the first instruction-tuning run, which gives the 4 to 6 weeks for the first milestone.
The full scope including step 7 is more like 7 to 10 weeks for one person (review estimate).

## Risks

- Vector search is weak at attribute combinations ("red shirt" and "hat" on the same person).
  ITM re-ranking and confirmation by the full model are the mitigations.
- Person-search quality ceiling: a frozen text-free encoder with a Q-Former whose text side learns from about 1M pairs is likely to land near pre-CLIP methods, roughly 55 to 68 Recall@1 on CUHK-PEDES against 73 to 79 for current CLIP-based methods (review estimate).
  Pre-CLIP methods with text encoders trained from scratch reach 49 to 61, and with a pretrained BERT 60 to 66 (for example <https://arxiv.org/abs/2107.12666>, <https://arxiv.org/abs/2306.02898>).
- Surveillance views differ from the general training data (high angle, low quality).
- Single-frame encoding misses fast or subtle motion.
- Scope: four phases plus three training stages is large for one person.
  The stages run strictly in sequence, and any change to the Q-Former invalidates stages 2 and 3.
- Evaluation leakage: most planned test sets overlap some training source (see [Evaluation sets](#evaluation-sets)); the filters must exist before data is packed.
- Detector and tracker licences: Ultralytics YOLO is AGPL-3.0, and Ultralytics says all trained YOLO models fall under AGPL-3.0 (<https://www.ultralytics.com/license>); BoxMOT is also AGPL-3.0.
  Either conflicts with the Apache-2.0 code licence.

## Decisions

Recorded on 2026-11-19, after the feasibility review.

| # | Decision | Rationale |
|---|---|---|
| 1 | Release no weights. The project publishes code and results only. | The DINOv3 licence (clause 1.b.i) lets derivative works be distributed only under the DINOv3 Agreement (<https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md>), and the MSMT17 and CUHK-PEDES terms do not clearly allow it, so releasing nothing avoids the conflict. |
| 2 | First milestone: phases 1 to 3 plus E1, in 4 to 6 weeks. Phase 4 and E2 follow. | The serial critical path and rerun allowance make the full scope 7 to 10 weeks; this cut ships a complete result first. |
| 3 | CUHK-PEDES: keep the unofficial Hugging Face copy for training and evaluation. | The official copy is by email only, with uncertain reply; with no weights released, the copy is used only for research results. |
| 4 | Training hardware: RTX A6000 48 GB on RunPod community cloud, training in bf16, not 4-bit; NF4 only for the 8 GB local demo. A persistent RunPod volume holds datasets, and checkpoints are saved every 30 to 60 minutes because community hosts can disappear. | It is the cheapest option at review time (about $39 to $67) and removes the slower, more fragile 4-bit training path. |

These close the former open questions on the money budget, the order of work, weight release and the CUHK-PEDES source, and the cloud provider and GPU size.

## Review recommendations under consideration

The feasibility review of 2026-11-18 made further recommendations that are not adopted as plan changes yet.
Numbers refer to its Recommendations list.
Recommendations 7, 9 and 15 are covered by decisions 2 and 4, and recommendation 12 (data table corrections) is applied above.

- 1: request DINOv3 and CUHK-PEDES access on day 0; if Meta refuses DINOv3, switch to DINOv2 ViT-L (Apache-2.0, ungated) rather than a mirror.
- 2: run zero-shot baselines first (SigLIP2 and DINOv3 dino.txt retrieval, Qwen3-VL-4B descriptions) to set the bar and give a fallback demo.
- 3: a week-1 kill test of the core design on COCO Karpathy and RSTPReid before any long run.
- 4: build one leakage filter (COCO ids, PEDES near-duplicates, UPAR restricted to PA-100K images, SurveillanceVQA and MEVA video names) before packing.
- 5: initialise the Q-Former from BERT-base, as BLIP-2 did.
- 6: an identity-aware contrastive target for person data, and a momentum queue or larger effective batch for ITC.
- 7 (remaining part): evaluate the NF4 demo configuration separately from the bf16 model.
- 8: switch to Qwen3-4B-Instruct-2507 (same hidden size, no thinking switch).
- 10: take BLIP3o shards from the CC12M-sourced `webdataset_shard_*` set only.
- 11: use RF-DETR (N to L) or D-FINE with ByteTrack; avoid Ultralytics and BoxMOT.
- 13: replace part of the template captions with person captions written by Qwen3-VL on PA-100K images.
- 14: do not build feature caching for DINOv3 or the Q-Former.
- 16: drop CC3M scraping.
- 17: drop RAPv2, and with it most of UPAR, unless an institution signs.
- 18 and 19: publishing rules; 18 (weights as DINOv3 derivatives) is moot with no weights released, and 19 (never publish images, captions, processed annotations or cached features; show no MSMT17, CUHK-PEDES or RAP images) still applies to the README and write-up.

## Open questions

1. Confirm the memory and GPU-hour estimates with a first timed run per stage on the RTX A6000.
2. Is the stage 1 demo for general images kept as its own showcase, or only person search?
3. Remaining unverified data facts: whether MALS used PEDES test captions as prompts, and the exact MSMT17 terms, since its official page is offline.
4. Size of the persistent RunPod volume, from the disk figures in [Compute plan](#compute-plan).
