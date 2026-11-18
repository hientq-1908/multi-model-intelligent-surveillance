# Intent: person search and description for surveillance video

Status: draft, updated 2026-11-16 with the author's decisions on data, publishing, the main model, cloud training, the person-search data sources and reference-only scores. Items marked **(proposed)** were suggested by Claude and not yet confirmed. See [Open questions](#open-questions).

Architecture reference: [idea.md](idea.md). Requirements and design: [spec.md](spec.md).

- Repository: `git@github.com:hientq-1908/multi-model-intelligent-surveillance.git`
- Work tracking: Jira project MLIS, <https://hientq.atlassian.net/jira/software/projects/MLIS/boards/1>

## Purpose

Build a multi-model vision-language pipeline end to end, as a personal self-learning project and a CV piece. There is one author, no team and no commercial use.

The priority is to show what the pipeline can do and how it was built. Benchmark scores are a sanity check for the author's reference, not the goal, and are not compared with published results.

## Use case

Search recorded surveillance video for a person by text description, then describe what that person looks like, where they are, and what they do.

Example: the query "person with a red T-shirt and hat" returns matching people. For each match the system writes a description of the person and their surroundings, and for a tracked person a timestamped timeline of their actions.

## Users

- **Primary reader:** someone reviewing the author's CV (recruiter, interviewer, engineer). The project is a personal showcase.
- **Demo user:** a person who types a description and inspects the results on a fixed set of public clips.

There is no production operator. This is not a deployed surveillance product.

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

"Fully finetune" in the original request means training the adapter parts and LoRA across all three stages. The backbones stay frozen.

## Phases

Each phase ends with a working result. The phases are capabilities of one model: there is a single stage 1 Q-Former, used for the index and inside the full model, and stages 2 and 3 each run once. Phase 1 is therefore checked on the final model, after the person data is in.

| # | Result | Status |
|---|---|---|
| 1 | Three-stage pipeline on general data: image search, captioning, question answering on single images | Must-have |
| 2 | Person search: detector, crop index, text query, ITM re-ranking | Must-have |
| 3 | Description of each match and its surroundings, from one enlarged crop | Must-have |
| 4a | Track timeline: change-based frame picking, per-segment descriptions, text summary; highlights as timestamps | Experiment option |
| 4b | True multi-frame input, trained with multi-frame data | Experiment option |

Phase 4 is an experiment: build both options and compare them on the same clips.

## Experiments

Each compares two choices on the same data and evaluation, and is reported in the write-up.

| # | Question | Option A | Option B | Compared on |
|---|---|---|---|---|
| E1 | Should the Q-Former train after stage 1? | Frozen in stages 2 and 3. Outputs can be cached, so it is cheaper. Run first | Trained in stage 2, and optionally 3. No caching | Caption and answer quality on a fixed set, GPU-hours |
| E2 | How should a long track be described? | 4a: describe segments, then summarise as text | 4b: several frames in one prompt, with multi-frame training | MEVA reference timelines, same clips |

`idea.md` does not say which option the original author used for E1.

## Deliverables

- Public GitHub repository with one training script per stage. Code licence: Apache-2.0.
- Trained weights (Q-Former, projector, LoRA adapter), published under CC BY-NC-SA 4.0 with a card that states research use and lists every training dataset. DINOv3 and Qwen3 are not redistributed.
- Demo on a fixed set of clips, run locally from the repository. It is not hosted.
- Write-up: each stage, design reasons (why a Q-Former with a text-free encoder), what broke, what was learned.

## Success criteria **(proposed)**

- Each stage passes a small check: retrieval after stage 1, captions after stage 2, answers after stage 3.
- Person search is reported as Recall@1/5/10, for reference only, with no target number.
- Descriptions are judged by hand on a fixed example set, with failures shown.

## Non-goals

- Matching or beating InA-Bridge's reported scores, or any leaderboard.
- Reading text in images, counting, or small-object detail. The 64-token design is weak at these.
- Live camera feeds or real-time operation.
- Identifying people by name or face.
- Unfreezing DINOv3 or the Qwen3 base weights.
- Generating video. Highlights are timestamps into the original clip.

## Constraints

- **Use:** self-learning and research only. No commercial use of the code, data or trained weights.
- **Compute:** all training runs on a rented cloud GPU server, including smoke tests. The local machine (one 8 GB GPU) is for writing code, CPU tests and demo inference. See [Compute plan](#compute-plan).
- **Data:** public datasets only, for research use, with no commercial use. The project collects no footage of its own, and the demo shows MEVA clips only. The person-search sets are for research use, and their licences are unstated or non-commercial.
- **Known technical limits:** DINOv3 needs bf16 or fp32; Qwen3 thinking mode must be off; drop DINOv3 register tokens and feed patch tokens to the Q-Former **(proposed)**.

## Data

Rule: public datasets only, with no manual labelling. Datasets may be combined and converted by code or by an AI model.

Status key: **V** = size or licence checked on the web on 2026-11-16. **U** = from memory or secondary sources, check the official page before use. Subset sizes are proposals and depend on the compute budget.

### Stage 1: Q-Former (image-text pairs)

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| COCO Captions | ~118k images, 5 captions each | Annotations CC BY 4.0, images under Flickr terms | General base, run first | U |
| CC3M | ~3M web image-caption pairs, by URL (some links dead) | Free to use, Google terms | Optional 300k to 500k subset | U |
| CUHK-PEDES | Official training split: 34,052 images, 11,003 identities, about 2 descriptions each | Authors release on request, academic only. Used here from an unofficial Hugging Face copy | Person search training and a test split made here | V (checked 2026-11-16) |
| ICFG-PEDES | 54,522 images, 4,102 identities, 1 description each. Images from MSMT17 **(U)** | Public download, linked from the SSAN repository. No licence stated; MSMT17 terms may apply **(U)** | Official training split for training; official test split as a reference test set | V size, V download, U licence |
| RSTPReid | 20,505 images, 4,101 identities, 2 descriptions each | Public download, linked from the official repository. No licence stated | Official training split for training; official test split as a reference test set | V size, V download, U licence |
| MALS | 1.51M synthetic person images with captions and 27 attributes | Released with the APTM code, terms to check | 200k to 300k subset. Images are generated, so no real people | V size, U licence |
| PA-100K | 100k pedestrian images, 26 attributes | CC BY 4.0 | Attribute source for generated captions | V |
| UPAR | Unified attributes with colours, over PA-100K, PETA, RAPv2, Market-1501 | Annotations CC BY-NC-SA 3.0, images follow their sources | Colour attributes for generated captions | V |

CUHK-PEDES has 40,206 images and 80,412 descriptions in all, but its authors release it only on request. This project uses an unofficial Hugging Face copy of the training split (`MaulikMadhavi/CUHK-PEDES-processed`), joins it to the authors' public `reid_raw.json` for identities and original captions, and makes its own train, validation and test split by identity. Decided by the author on 2026-11-16. The author owns none of this data. Below, "PEDES" means CUHK-PEDES, ICFG-PEDES and RSTPReid.

### Stage 2: projector alignment (captioning through the LLM)

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| BLIP3o-Pretrain-Long-Caption | ~27.2M image and long-caption pairs, WebDataset shards | Apache-2.0 | 300k to 500k subset, taken as whole shards | V |
| CUHK-PEDES, ICFG-PEDES, RSTPReid | Same as stage 1 | Same | Captions reused as "describe this person" targets | as above |

### Stage 3: instruction tuning

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| `llava_v1_5_mix665k` | ~665k image conversations over COCO, GQA, Visual Genome, OCR-VQA, TextVQA | Mixed, per source | General instruction following. Drop the OCR-VQA and TextVQA parts, since text reading is a non-goal | U |
| UCF-Crime Annotation (UCA) | 1,854 surveillance videos, ~23.5k timestamped sentences, ~110 hours. Videos come from UCF-Crime | Licence in the CVPR 2024 supplement, to check | Frames from each event span paired with its sentence: surveillance-view captions | V size, U licence |
| SurveillanceVQA-589K | ~589k question-answer pairs on surveillance video, 12 question types | MIT on Hugging Face; source videos not documented on the card | Surveillance questions for 4b, only if its videos are shown not to overlap the UCA test split (spec 7.4) | V licence, U content |

### Phase 4b: multi-frame training

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| UCA | As above | As above | Several frames per event with its sentence | as above |
| M4-Instruct | ~1.18M multi-image and video instruction samples | CC BY 4.0 | Small subset, to teach several images in one prompt | V |
| LLaVA-Video-178K | 178k video captions, ~961k open QA | Card says academic research and education only | Not used; M4-Instruct is chosen (spec 7.3) | V |

### Demo and evaluation video (not used for training)

| Dataset | Content | Licence / access | Use | Status |
|---|---|---|---|---|
| MEVA | Multi-camera surveillance-style video with consenting actors, activity annotations with tracks | CC BY 4.0, free download from AWS | Demo clips; activity annotations as reference timelines for phase 4 | V licence, U annotation detail |

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
| Short queries | Attribute templates, PEDES captions | Code takes one to three attribute phrases or a single clause; optional short rewording by a text LLM | Stage 1, so search works on the 5 to 10 words a user types |
| Multi-frame timeline targets | UCA | Code writes every sentence inside a window of frames as `[start-end] sentence` lines | Phase 4b |

### Evaluation sets

- General search: COCO Karpathy 5k test split. These images, and the POPE images, are excluded from all training data.
- Person search: RSTPReid and ICFG-PEDES official test splits, and a CUHK-PEDES test split of 1,000 identities made here from the official training split, Recall@1/5/10. Training images that are near-duplicates of these test images, and images of the same MSMT17 people in the other MSMT17-based set, are dropped, because the person datasets share sources.
- Person search on surveillance-style images: attribute retrieval on the PA-100K test split, which needs no access request; hand-judged precision at 5 on fixed MEVA queries.
- Descriptions: UCA test sentences; fixed hand-checked examples from MEVA.
- Phase 4: MEVA reference timelines, option 4a against 4b.

### Data risks

- ICFG-PEDES and RSTPReid state no licence, and ICFG-PEDES images come from MSMT17, which has its own terms **(U)**. CUHK-PEDES comes from an unofficial copy of data its authors release on request only. All three are treated as research-only, and the README states that the author owns none of them. Adding person data after the person run means rerunning the person run and stages 2 and 3.
- Several licences are non-commercial or research-only. This matches the project's research-only use. The repository lists every dataset with its licence. Released weights are CC BY-NC-SA 4.0 and the demo carries a research-only notice. Check what the person-search sets allow for trained models before uploading weights.
- UCF-Crime is mostly crime and accident footage, so it skews what the model describes and contains violent scenes.
- Generated captions and questions carry the errors of the model or template that made them.
- A single frame cannot show an event that unfolds over time. UCA sentences are paired with single frames only for short spans, balanced between normal and anomaly videos and capped at a small share of stage 3; full events are used for phase 4b.
- A design review on 2026-11-16 found the gaps these rules close. See [spec.md](spec.md), section 2.9.

## Compute plan **(proposed)**

The money budget is not set yet. This plan is stated in compute: GPU memory and GPU-hours. All figures are estimates, not measurements. Rule: time the first 200 steps of every run, record peak memory, and extrapolate before letting it continue.

### GPU memory per stage (estimated)

| Stage | What sits in memory | Qwen3-1.7B | Qwen3-4B |
|---|---|---|---|
| 1. Q-Former | DINOv3 frozen, Q-Former with three loss passes, batch ~128 | 12 to 16 GB (no LLM) | same |
| 2. Projector | LLM frozen, gradients pass through it; long captions, batch ~16 | 12 to 16 GB | 18 to 24 GB, or 12 to 16 GB in 4-bit |
| 3. LoRA | LLM with LoRA, conversations, batch ~16 | 14 to 20 GB | 22 to 30 GB, or 14 to 18 GB in 4-bit |
| 4b. Multi-frame | 8 to 16 frames (512 to 1,024 visual tokens), batch ~4 | 16 to 20 GB | over 24 GB, or about 20 GB in 4-bit |
| Inference and demo | One image or track at a time | about 5 GB | about 5 GB in 4-bit |

Recommendation: one 24 GB GPU (RTX 4090 or 3090 class), with the main model Qwen3-4B loaded in 4-bit for stages 2, 3 and 4b. A 48 GB GPU is the alternative, for Qwen3-4B without 4-bit. The server also needs about 150 GB of disk and 32 GB of system RAM.

### Where each job runs

| Job | Machine |
|---|---|
| All code, CPU tests with tiny stand-in models, template-built sets | Local machine |
| Packing large datasets, model-built sets | Cloud server (a CPU-only server is enough for packing) |
| Smoke tests: overfit 100 samples per stage with Qwen3-0.6B | Cloud server |
| Training runs | Cloud server, one 24 GB GPU (RTX 4090 or 3090 class) |
| Detector, tracker, indexing, demo inference | Local machine, or the cloud server if 8 GB is not enough |

No full training run starts until the cloud preflight has passed on that server: CPU tests, real-weight checks, the smoke test, a checkpoint save-and-resume round trip, then the 200-step timing projection. See [spec.md](spec.md), section 7.11.

### LLM size ladder

| Model | Hidden size | Role |
|---|---|---|
| Qwen3-0.6B | 1024 | Cloud smoke tests |
| Qwen3-1.7B | 2048 | Fallback if the budget cannot cover Qwen3-4B |
| Qwen3-4B | 2560 | Main model for all phases |

Stage 1 does not involve the LLM, so one trained Q-Former serves all three sizes. Only the projector and LoRA are retrained per LLM.

### Small recipe

| Run | Data | Estimated GPU-hours |
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

Hours are for one RTX 4090 class GPU. A 3090 has the same memory and is roughly 1.5 to 2 times slower. The Qwen3-4B figures are scaled from the earlier estimates (40 to 60 hours for stages 2 and 3 on 4B, against 14 to 23 on 1.7B); none are measured. Realistic total with reruns: 135 to 230 GPU-hours.

### Cost rules

- Pack each dataset into shards. Sets whose terms allow it are stored in a private Hugging Face dataset repository, and each server pulls what it needs, so no idle storage volume is paid for. Sets with unverified or restrictive terms stay off the Hub by default: they are downloaded or copied to the server directly and deleted when it is released.
- Cache what is frozen. Q-Former outputs for stages 2 and 3 are about 100 KB per image, if the Q-Former is kept frozen there.
- Save checkpoints to remote storage every 30 to 60 minutes and make every script resumable, so cheaper interruptible machines can be used.
- Stop the machine when a run ends. Log GPU-hours per run against the estimate.
- Change one thing per rerun.

### Order of work **(proposed)**

Code is written by a coding agent, so coding is not the limiting step. The schedule is set by GPU run time, data download and packing, and the author's review of each result.

| Step | Work | What it waits on | Estimated elapsed time |
|---|---|---|---|
| 0 | Start downloads; check dataset terms | Nothing. Do first | 1 day to start |
| 1 | Pipeline code with CPU tests, data packing, first cloud session with preflight for all three stages | Downloads and upload speed | 2 to 3 days |
| 2 | Baseline, stage 1 general run and retrieval check; detector, tracker and index code | 9 to 16 GPU-hours | 2 to 3 days |
| 3 | Stage 1 person run with the three PEDES sets, which fixes the single Q-Former; person search: phase 2 complete | 3 to 5 GPU-hours | 2 to 3 days |
| 4 | All instruction data; stage 2 once, stage 3 once; confirm and describe: phases 1 and 3 complete | 40 to 60 GPU-hours, review between stages | 5 to 8 days |
| 5 | Phase 4a, then 4b, and experiments E1 and E2 | 13 to 27 GPU-hours plus E1 reruns | 3 to 6 days |
| 6 | Local demo, publish code and weights, write-up | Author's review | 2 to 3 days |

Total: about 3 to 5 weeks elapsed, most of it waiting on training runs and review, not on coding. Reruns add GPU-hours and days in whichever step they occur.

## Risks

- Vector search is weak at attribute combinations ("red shirt" and "hat" on the same person). ITM re-ranking and confirmation by the full model are the mitigations.
- Surveillance views differ from the general training data (high angle, low quality).
- Single-frame encoding misses fast or subtle motion.
- Scope: four phases plus three training stages is large for one person.

## Open questions

1. Money budget for rented GPUs: to be decided later by the author. The compute estimate is 135 to 230 GPU-hours on a 24 GB GPU with Qwen3-4B, including reruns.
2. Order of work: confirmed in outline by the author's decision that the full model is trained once (about 3 to 5 weeks elapsed).
3. Confirm the memory and GPU-hour estimates with a first timed run per stage.
4. Is the stage 1 demo for general images kept as its own showcase, or only person search?
5. Dataset terms: ICFG-PEDES and RSTPReid state no licence, and ICFG-PEDES images come from MSMT17. Check what they allow, including releasing trained weights. CUHK-PEDES is taken from an unofficial copy: decide before release whether weights trained on it are published.
6. Check every row marked **U** in [Data](#data) against its official page.
7. Cloud provider, and 24 GB with 4-bit loading against 48 GB without.
