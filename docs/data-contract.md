# Data contract

Every dataset task converts its source into the same three things: a stream of typed sample records, a JSONL manifest, and optionally WebDataset tar shards.
Stage code reads only these, never a dataset-specific format.
The code lives in `src/mlis/data/`: `schema.py`, `manifest.py`, `shards.py`, `reader.py`, `validate.py`, `synthetic.py`.

## Sample record

`mlis.data.schema.SampleRecord` is one image with one or more texts.

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | str | Unique within a manifest. Letters, digits, `_` and `-` only, so it is a safe shard key |
| `image` | `ImageRef` | `path` (relative, never absolute or containing `..`), optional `shard`, optional `box` crop `(x0, y0, x1, y1)` in pixels |
| `texts` | list[str] | One or more captions, queries or answers. At least one, none blank |
| `source` | str | Lowercase dataset name, for example `coco-karpathy` |
| `split` | `train` / `val` / `test` | Split the sample belongs to |
| `licence` | `LicenceTag` | `permissive`, `cc-by`, `non-commercial`, `research-only`, `unstated` or `synthetic` |
| `person_id` | str, optional | Identity label prefixed with the source, such as `rstpreid:1234`, so ids never collide across sets |
| `attributes` | dict, optional | Attribute name to value, such as `{"upper_colour": "red"}` |
| `negatives` | list[str], optional | Hard-negative texts that must not match the image |
| `extra` | dict, optional | Source-specific fields that stage code never reads |

Unknown fields are rejected.
Records with the licence tags `non-commercial`, `research-only` and `unstated` are never uploaded to a public hub and never shown in the README or demo.

A cropped region (a person box in a larger image) is expressed with `image.box`.
Readers apply the box when they decode (`mlis.data.images.decode_image`).

## Manifest (JSONL)

One file per packed dataset.
The first line is a header object, every following line is one record.

```jsonl
{"manifest_header": {"schema_version": 1, "dataset": "coco-karpathy-train", "num_samples": 2, "root": "images", "shards": []}}
{"sample_id": "gen000", "image": {"path": "gen000.png"}, "texts": ["a red rectangle"], "source": "synthetic-general", "split": "train", "licence": "synthetic"}
{"sample_id": "gen001", "image": {"path": "gen001.png"}, "texts": ["a blue rectangle"], "source": "synthetic-general", "split": "train", "licence": "synthetic"}
```

Header fields:

| Field | Meaning |
|---|---|
| `schema_version` | Currently 1 |
| `dataset` | Name of the packed dataset |
| `num_samples` | Number of record lines, checked by the validator |
| `root` | Directory that `image.path` is relative to, relative to the manifest file. Used for loose image files |
| `shards` | Shard files in reading order, relative to the manifest. Empty for loose files |
| `created_by` | Optional tool or task name |

`write_manifest` writes atomically (temporary file, fsync, rename), so a reader never sees a half-written manifest.
`iter_manifest` reads lazily and reports the line number of any invalid record.

## Shards (WebDataset tar)

A manifest with a non-empty `shards` list stores its images in tar files next to it, named `<prefix>-00000.tar`, `<prefix>-00001.tar`, and so on.
Each sample is two adjacent members that share a key, the `sample_id`:

```
gen000.json   the SampleRecord, with image.shard and image.path pointing at the image member
gen000.png    the original image bytes, untouched
gen001.json
gen001.png
```

Any WebDataset reader can open these shards.
`ShardWriter` starts a new shard after `max_samples` samples or `max_bytes` bytes, writes each shard under a `.tmp` name and renames it when it is complete, and discards the open shard if the writer exits with an error.
`pack_samples` is the one-call helper: give it `(record, image_bytes, ext)` items and it returns the packed records and the shard names, ready for `write_manifest`.

Reading:

- `iter_shards(paths, skip=n)` streams samples in order and drops the first `n`, which is how a resumed run returns to its saved data position.
- `iter_samples(manifest)` yields decoded RGB images with the crop box applied, for both shard and loose-file manifests.
- `iter_image_bytes(manifest)` yields the raw bytes if the caller decodes them itself.

## Validation

```bash
uv run python -m mlis.data.validate path/to/manifest.jsonl
```

It checks the header, every record, the sample count, duplicate ids, that every listed shard exists and that every image decodes.
It exits non-zero and prints each problem.
Every dataset task runs it on its output and calls `validate_dataset` from its tests.

## Test fixture

`tests/fixtures/` holds a tiny synthetic dataset in both layouts:

- `tiny_shards/`: manifest plus tar shards (8 samples per shard).
- `tiny_files/`: manifest plus loose PNG files under `images/`.

Both hold the same 12 samples: 6 general image-caption pairs and 6 person samples over 3 identities with attributes and hard negatives, across `train`, `val` and `test`.
The images are flat colour blocks and contain no real data.
The pytest fixtures `shard_manifest`, `file_manifest` and `fixtures_dir` (in `tests/conftest.py`) return the paths.

Regenerate the committed files with:

```bash
uv run python -m mlis.data.synthetic tests/fixtures
```

A test checks that the committed copy still matches the generator.

## Rules for dataset tasks

- Add files only inside your own subpackage of `mlis.data` (`general`, `person`, `built_text`, `instruction`, `leakage`).
  Do not change the contract files above. If the contract is missing something, raise it instead of working around it.
- Use `extra` for source-specific fields.
- Prefix `person_id` with the source name.
- Set the licence tag honestly. When a source states no licence, use `unstated`.
- Never commit dataset files, shards or model weights. Only the synthetic fixture is committed.
