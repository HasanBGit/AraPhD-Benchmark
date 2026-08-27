# AraPhD-Benchmark

Data and pipeline for **ArabPhD**, a cause-diagnostic Arabic Visual
Hallucination Evaluation benchmark for multimodal LLMs. The project proposal
has the full research design: 5 evaluation modes (base, sec, icc, ccs, nota),
all built on a shared per-image pool via an Arabic Cultural Visual Vocabulary
taxonomy.

This repo covers two things: **stage 1** builds and captions the
**candidate image pool** that every mode's VQA triplets are derived from,
and **stage 2** is per-mode question generation plus MLLM evaluation. Mode 1
is complete end-to-end; Modes 2-5 are candidate-data-only or in progress
(see [Status](#status)).

## Getting started

```bash
git clone https://github.com/HasanBGit/AraPhD-Benchmark.git
cd AraPhD-Benchmark
pip install requests          # the only third-party dependency, used by modes/mode1_base/run_mode1_openrouter.py
```

Requires Python 3.9+. No other setup for browsing/editing the data or
pipeline scripts (`fill_captions.py`, `fill_nota_questions.py`, etc. are
stdlib-only).

To **run Mode 1 evaluation** (call an MLLM via OpenRouter against the 270
Mode 1 questions), you also need an OpenRouter API key:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # gitignored, repo root
cd modes/mode1_base
python3 run_mode1_openrouter.py run --dry-run   # sanity check: builds the task list, no API calls, no cost
```

See [Stage 2a](#pipeline) below for the full command reference (sample vs.
full runs, `--concurrency`, `metrics`, `table`).

**Quick orientation, if you're new to this repo:**
1. `pipeline/arabphd_full_candidate_pool.json` is the one file to read first. Every question-ready record across every mode lives here (currently just Mode 1's 270).
2. `data/candidate_pool/` is the raw material (images plus per-image annotations) that pool is built from.
3. `modes/<mode>/` holds each mode's own generation scripts and, once you run it, its results.
4. `info/arabphd_proposal.tex` / `.pdf` is the actual research proposal this repo implements. Read it if something in the data structure doesn't make sense on its own.

## Status

**Stage 1 (candidate pool):** 281 candidate images across 5 categories.
Caption pass **complete** (281/281, 0 validation errors).

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

**Stage 2a (Mode 1 evaluation):** 270/270 images have generated questions
(540 VQA triplets), merged into the pool and evaluated end-to-end against 4
OpenRouter models on the same 50-image / 100-question subset:

| model | accuracy | Yes-recall | No-recall | **PhD Index** | yes-bias |
|---|---:|---:|---:|---:|---:|
| google/gemini-2.5-flash | 82.0% | 90.0% | 74.0% | **0.812** | 58.0% |
| google/gemini-2.5-flash-lite | 72.0% | 80.0% | 64.0% | **0.711** | 58.0% |
| qwen/qwen2.5-vl-72b-instruct | 64.0% | 62.0% | 66.0% | **0.639** | 48.0% |
| openai/gpt-4o-mini | 71.0% | 96.0% | 46.0% | **0.622** | 75.0% |

Full breakdowns in `modes/mode1_base/mode1_results/comparison_table.md`
(regenerate anytime with `python3 run_mode1_openrouter.py table`). Modes 2-4
(sec/icc/ccs) have raw candidate annotations only (`data/candidate_pool/image_pool.json`,
281 records) but no generated questions yet. Mode 5 (nota) has 350 generated
records, but it isn't wired into the current pool; see
[Stage 2b](#pipeline) below.

## Repo layout

```
data/
  candidate_pool/
    image_pool.json        # raw candidate pool: one record per image (see image_schema.json)
    image_schema.json       # field-by-field schema/definitions for image_pool.json
    manifest_unified.csv    # raw source manifest (pre-pool)
    image_pool_mode1.json   # Mode 1 source: 317 image records (311 usable), pre-merge
    manifest_mode1.csv      # Mode 1 source manifest
    hitem_mode1.csv         # Mode 1 source hitem annotations
    lexicon_ar.json         # Arabic lexicon reference used while building Mode 1
    backups/                # auto-created pre-merge snapshots of image_pool.json (gitignored)
    images/
      architecture/ attire/ cuisine/ objects/ script/   # the images themselves (gitignored)

pipeline/
  build_image_pool.py             # manifest_unified.csv -> image_pool.json (stage 1)
  captioning_prompt.md            # canonical prompt for the captioning pass (stage 1b)
  fill_captions.py                # control script: status / next / validate / merge
  batches/                        # per-batch caption results, merged in via fill_captions.py
  arabphd_full_candidate_pool.json  # THE merged pool: promoted, question-ready entries per mode
  review_ui/                      # local review tool for the merged pool
  backups/                        # pre-merge snapshots of arabphd_full_candidate_pool.json

modes/                             # one subfolder per evaluation mode
  mode1_base/
    run_mode1_openrouter.py       # OpenRouter inference script: run / metrics / table subcommands
    questions_base.json           # source: 270 images x 2 questions, merged into arabphd_full_candidate_pool.json
    BaseMode.ipynb                 # original Colab pilot (Gemini SDK), prompt_version=1 origin
    mode1_results/                # results_mode1_<label>_<model>.csv per run, + comparison_table.md/.csv
  mode2_sec/                      # not started
  mode3_icc/                      # not started
  mode4_ccs/                      # not started
  mode5_nota/                     # in progress, not yet wired to arabphd_full_candidate_pool.json
    fill_nota_questions.py
    nota_question_prompt.md
    arabphd_nota_questions.json
    nota_batches/
    backups/

```

A record is only promoted into `pipeline/arabphd_full_candidate_pool.json` once it has an
actual generated question set for its mode, not just candidate-pool
annotations (caption/h_item/etc.); that's the bar `mode1` cleared by
merging in Mahyoub's questions. `mode` on each entry (`mode1`, `sec`, `icc`,
`ccs`) says which evaluation protocol it belongs to; currently only `mode1`
(270 entries) is populated. The 200 sec/icc candidate annotations that were
here previously still exist as raw candidates in
`data/candidate_pool/image_pool.json` (281 records total); they're not
promoted into the full pool until sec/icc have real question sets of their
own.

## Pipeline

**Stage 1: build the pool.** `pipeline/build_image_pool.py` merges
`data/candidate_pool/manifest_unified.csv` into `image_pool.json`, one
record per image, with 4 fields left `null`: `inferred_identity_ar`,
`inferred_identity_en`, `caption_ar`, `h_item_ar`.

**Stage 1b: caption the pool.** Each image is looked at directly (vision
pass) and captioned per the fixed prompt in `pipeline/captioning_prompt.md`,
which defines exactly what each field means:

- `inferred_identity_en` / `inferred_identity_ar`: the correct, specific
  identity, vision-verified against the image (not trusted blindly from
  source metadata, which is sometimes wrong, e.g. raw Wikimedia filenames).
- `caption_ar`: a neutral, detailed Arabic visual description that does
  **not** name the identity (kept usable for base-mode VQA generation
  without leaking the answer).
- `h_item_ar`: a plausible-but-**wrong** Arabic foil from the same domain
  (e.g. a different real tower, a different real dish), not a category
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
consumes the completed `image_pool.json` and promotes finished records into
`pipeline/arabphd_full_candidate_pool.json`, the single merged pool. Each
mode's generation work lives under `modes/<mode>/`.

**Stage 2a: Mode 1 (base), complete, 270/270.** Built from
`modes/mode1_base/questions_base.json` (270 images, 2 yes/no questions each:
one asking about the true identity, one about a plausible-but-wrong
`h_item_ar`; sourced from `data/candidate_pool/image_pool_mode1.json` /
`manifest_mode1.csv` / `hitem_mode1.csv`), merged into
`pipeline/arabphd_full_candidate_pool.json` as `mode: "mode1"` entries.
`modes/mode1_base/run_mode1_openrouter.py` runs inference against any
OpenRouter vision model (default `google/gemini-2.5-flash`), reusing the
frozen `prompt_version=1` protocol from the original
`modes/mode1_base/BaseMode.ipynb` Colab pilot (same system instruction,
temperature=0, yes/no-only output contract) so results are comparable
across both:

```bash
cd modes/mode1_base
export OPENROUTER_API_KEY=sk-or-...        # or a .env file at the repo root

python3 run_mode1_openrouter.py run --run-label sample                    # balanced 10 images / 20 tasks (seed=42)
python3 run_mode1_openrouter.py run --run-label full                      # all 270 images / 540 tasks
python3 run_mode1_openrouter.py run --run-label full --limit 100          # first 50 images / 100 tasks only
python3 run_mode1_openrouter.py run --run-label full --limit 100 \
    --model openai/gpt-4o-mini --concurrency 30                           # any OpenRouter vision model; fire N requests in parallel
python3 run_mode1_openrouter.py metrics --file mode1_results/results_mode1_full_google-gemini-2.5-flash.csv
python3 run_mode1_openrouter.py table --out mode1_results/comparison_table.md --csv-out mode1_results/comparison_table.csv
```

Runs are resumable (re-run the same command to pick up where an
interrupted/rate-limited run left off) and cost-checkable: OpenRouter's
`GET /api/v1/key` endpoint (with your key as the bearer token) returns real
spend, not an estimate. `metrics` reports accuracy, Yes-recall/No-recall,
and the Arabic PhD Index (harmonic mean of the two, per the proposal's
metric definition) for one results file. `table` scans every results CSV in
`mode1_results/` and writes the cross-model comparison shown in
[Status](#status) above. Regenerate it any time results change instead of
recomputing those numbers by hand.

**Stage 2b: nota (Mode 5), in progress.** `modes/mode5_nota/nota_question_prompt.md`
is the system prompt an LLM follows to turn one candidate-pool record into
its MCDR/OEDR/UDR triplet (plus a matched control for 50/100 items), using
the same per-image vision-verified fields the pool carries
(`inferred_identity_ar`, `implicit_rejection_set`). No new judgment calls.
`modes/mode5_nota/fill_nota_questions.py` is the control layer only (mirrors
`fill_captions.py`): `select`/`status` track the 100-item quota, `next`
prints the next unfilled items + hints (no model call), `validate`/`merge`
check a batch against the prompt's rules and fold it into
`arabphd_nota_questions.json`. Generated batch-by-batch in
`modes/mode5_nota/nota_batches/`, one LLM pass per batch reading
`nota_question_prompt.md` as its system prompt. `arabphd_nota_questions.json`
already holds 350 records (100 triplets + a 50-item matched control) built
against the pool's earlier sec/icc candidate entries, but those entries
aren't in `arabphd_full_candidate_pool.json` right now (only `mode1` is), so
`fill_nota_questions.py status`'s quota count won't be meaningful again
until sec/icc candidate data is back in the pool (or the script is repointed
at `data/candidate_pool/image_pool.json`). Not wired up for evaluation yet.

```bash
cd modes/mode5_nota && python3 fill_nota_questions.py status
```

The 350 records match the proposal's own Mode 5 spec: 100 triplets plus a
matched 50-triplet control subset, evaluated under MCDR/OEDR/UDR
(`info/arabphd_proposal.pdf`, §4, "Mode 5, ArabPhD-nota"). `validate` checks
schema conformance (no answer leakage, distractors match the pool, option
counts correct); project lead (Hassan Barmandah) has additionally done a
human review pass over the 350 records.

*Cross-item repetition of `question_ar` is intentional, not a defect.
This was checked against the literature, not assumed.* Many of the 100 items share
identical `question_ar` text on purpose (`nota_question_prompt.md` ties the
question to a fixed per-item frame so MCDR/OEDR/UDR + control all ask the
*same* question). Two papers were checked for precedent:
- **MMBench** (Liu, Duan, Zhang, Li, Zhang, Zhao, Yuan, Wang, He, Liu, et al.,
  *"MMBench: Is Your Multi-modal Model an All-around Player?"*, ECCV 2024).
  This is the actual origin of the pattern: MMBench groups questions into
  ~20 fixed "ability" categories and reuses one templated question across
  every image in a category.
- **MM-UPD** (Miyai, Yang, Zhang, Ming, Yu, Irie, Li, Li, Liu, Aizawa,
  *"Unsolvable Problem Detection: Robust Understanding Evaluation for Large
  Multimodal Models"*, arXiv:2403.20331), the paper this repo's nota design
  cites directly. It's built by adapting MMBench (Appendix B.1) and inherits
  the per-category templating unchanged.

Verified directly, not taken on faith: pulling ~800 of the 820 real rows of
MM-UPD's own `mmaad_aad` file (2026-08-20, via HF LFS range request, see
`reference_datasets/mm_upd_miyai2025/`) found **218 unique question strings
out of 802 rows (≈27%)**, with the single most-repeated string reused across
**52 different images**. Our 22 unique strings across 100 items (≈22%) is
the same order of magnitude, for the same underlying reason in both cases:
the image and option set carry the per-item signal, not question-text
novelty. Both papers do the templated-question thing as a matter of course,
not as an exception; it isn't a corner either benchmark treats as a flaw to
avoid.

## Licensing

Code: MIT (see `LICENSE`). Images: each record in `image_pool.json` carries
its own source `license` field, a mix of MIT, Apache-2.0, CC0, CC-BY, and
CC-BY-SA from Kaggle, Hugging Face, Wikimedia Commons, and The Met Open
Access (`source` field has the exact provenance URL per image). Check that
field before redistributing any individual image outside this repo.
