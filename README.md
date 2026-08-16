# AraPhD-Benchmark

Data and pipeline for **ArabPhD**, a cause-diagnostic Arabic Visual
Hallucination Evaluation benchmark for multimodal LLMs (see the project
proposal for the full research design: 5 evaluation modes — base, sec, icc,
ccs, nota — built on a shared per-image pool via an Arabic Cultural Visual
Vocabulary taxonomy).

This repo is stage 1 of that pipeline: building and captioning the
**candidate image pool** that every mode's VQA triplets are derived from.

## Status

281 candidate images across 5 categories. Caption pass **complete** (281/281,
0 validation errors).

```
cd pipeline && python3 fill_captions.py status
```

| category | images |
|---|---:|
| architecture | 80 |
| attire | 21 |
| cuisine | 88 |
| objects | 2 |
| script | 90 |
| **total** | **281** |

## Repo layout

```
data/
  candidate_pool/
    image_pool.json        # the pool: one record per image (see image_schema.json)
    image_schema.json       # field-by-field schema/definitions for image_pool.json
    manifest_unified.csv    # raw source manifest (pre-pool)
    backups/                # auto-created pre-merge snapshots of image_pool.json (gitignored)
    images/
      architecture/ attire/ cuisine/ objects/ script/   # the images themselves

pipeline/
  build_image_pool.py     # manifest_unified.csv -> image_pool.json (stage 1)
  captioning_prompt.md    # canonical prompt for the captioning pass (stage 1b)
  fill_captions.py        # control script: status / next / validate / merge
  batches/                # per-batch caption results, merged in via fill_captions.py
```

## Pipeline

**Stage 1 — build the pool.** `pipeline/build_image_pool.py` merges
`data/candidate_pool/manifest_unified.csv` into `image_pool.json`, one
record per image, with 4 fields left `null`: `inferred_identity_ar`,
`inferred_identity_en`, `caption_ar`, `h_item_ar`.

**Stage 1b — caption the pool.** Each image is looked at directly (vision
pass) and captioned per the fixed prompt in `pipeline/captioning_prompt.md`,
which defines exactly what each field means:

- `inferred_identity_en` / `inferred_identity_ar` — the correct, specific
  identity, vision-verified against the image (not trusted blindly from
  source metadata, which is sometimes wrong — e.g. raw Wikimedia filenames).
- `caption_ar` — a neutral, detailed Arabic visual description that does
  **not** name the identity (kept usable for base-mode VQA generation
  without leaking the answer).
- `h_item_ar` — a plausible-but-**wrong** Arabic foil from the same domain
  (e.g. a different real tower, a different real dish) — not a category
  label. Feeds the sec/icc misleading-context modes and the nota distractor
  pool.

`pipeline/fill_captions.py` is the control layer around that pass (run from
inside `pipeline/`):

```bash
cd pipeline
python3 fill_captions.py status                          # progress by category
python3 fill_captions.py next --n 10 --category cuisine   # pick next unfilled, print image + hints (no model call)
python3 fill_captions.py validate --file batches/batch_002.json
python3 fill_captions.py merge --file batches/batch_002.json
```

`validate` enforces: all 4 fields present and non-empty, Arabic script
present, `caption_ar` within a sane length range, `caption_ar` doesn't leak
`inferred_identity_ar` verbatim, and `h_item_ar` != `inferred_identity_ar`.
`merge` re-validates, snapshots `image_pool.json` to
`data/candidate_pool/backups/` first, then writes. A batch result file is
`{"image_id": {the 4 fields}, ...}`.

**Stage 2 (not yet built).** Per-mode triplet generation (base/sec/icc/ccs/
nota) consumes the completed `image_pool.json`.

## Licensing

Code: MIT (see `LICENSE`). Images: each record in `image_pool.json` carries
its own source `license` field — a mix of MIT, Apache-2.0, CC0, CC-BY, and
CC-BY-SA from Kaggle, Hugging Face, Wikimedia Commons, and The Met Open
Access (`source` field has the exact provenance URL per image). Check that
field before redistributing any individual image outside this repo.
