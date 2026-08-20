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

**Stage 2 (partial).** Per-mode triplet generation (base/sec/icc/ccs/nota)
consumes the completed `image_pool.json` / `arabphd_full_candidate_pool.json`.

**Stage 2b — nota (Mode 5), complete: 350/350, human-reviewed.**
`pipeline/nota_question_prompt.md` is the system prompt an LLM follows to
turn one candidate-pool record into its MCDR/OEDR/UDR triplet (+ matched
control for 50/100 items) — same per-image vision-verified fields the pool
already carries (`inferred_identity_ar`, `implicit_rejection_set`), no new
judgment calls. `pipeline/fill_nota_questions.py` is the control layer only
(mirrors `fill_captions.py`): `select`/`status` track the 100-item quota,
`next` prints the next unfilled items + hints (no model call), `validate`/
`merge` check a batch against the prompt's rules and fold it into
`arabphd_nota_questions.json`. Generated batch-by-batch in
`pipeline/nota_batches/`, one LLM pass per batch reading
`nota_question_prompt.md` as its system prompt.

```bash
cd pipeline && python3 fill_nota_questions.py status
```

The 350 records match the proposal's own Mode 5 spec — 100 triplets + a
matched 50-triplet control subset, evaluated under MCDR/OEDR/UDR
(`info/arabphd_proposal.pdf`, §4, "Mode 5, ArabPhD-nota"). `validate` checks
schema conformance (no answer leakage, distractors match the pool, option
counts correct); project lead (Hassan Barmandah) has additionally done a
human review pass over the 350 records.

*Cross-item repetition of `question_ar` is intentional, not a defect —
checked against the literature, not assumed.* Many of the 100 items share
identical `question_ar` text on purpose (`nota_question_prompt.md` ties the
question to a fixed per-item frame so MCDR/OEDR/UDR + control all ask the
*same* question). Two papers were checked for precedent:
- **MMBench** (Liu, Duan, Zhang, Li, Zhang, Zhao, Yuan, Wang, He, Liu, et al.,
  *"MMBench: Is Your Multi-modal Model an All-around Player?"*, ECCV 2024) —
  the actual origin of this pattern: MMBench groups questions into ~20 fixed
  "ability" categories and reuses one templated question across every image
  in a category.
- **MM-UPD** (Miyai, Yang, Zhang, Ming, Yu, Irie, Li, Li, Liu, Aizawa,
  *"Unsolvable Problem Detection: Robust Understanding Evaluation for Large
  Multimodal Models"*, arXiv:2403.20331) — the paper this repo's nota design
  cites directly. It's built by adapting MMBench (Appendix B.1) and inherits
  the per-category templating unchanged.

Verified directly, not taken on faith: pulling ~800 of the 820 real rows of
MM-UPD's own `mmaad_aad` file (2026-08-20, via HF LFS range request — see
`reference_datasets/mm_upd_miyai2025/`) found **218 unique question strings
out of 802 rows (≈27%)**, with the single most-repeated string reused across
**52 different images**. Our 22 unique strings across 100 items (≈22%) is
the same order of magnitude, for the same underlying reason in both cases:
the image + option set carry the per-item signal, not question-text novelty.
Both papers do the templated-question thing as a matter of course, not as an
exception — this isn't a corner either benchmark treats as a flaw to avoid.

## Licensing

Code: MIT (see `LICENSE`). Images: each record in `image_pool.json` carries
its own source `license` field — a mix of MIT, Apache-2.0, CC0, CC-BY, and
CC-BY-SA from Kaggle, Hugging Face, Wikimedia Commons, and The Met Open
Access (`source` field has the exact provenance URL per image). Check that
field before redistributing any individual image outside this repo.
