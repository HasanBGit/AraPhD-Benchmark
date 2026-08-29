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
(regenerate anytime with `python3 run_mode1_openrouter.py table`).

**Modes 2-3 (sec/icc), context generation in progress.** Misleading-caption
content (the actual sec/icc adversarial context, matching PhD's
`context.sec`/`context.icc` design) is built for 200 candidates
(`modes/mode2_sec/candidate_pool_sec.json`, 98; `modes/mode3_icc/candidate_pool_icc.json`,
102), 150 of which are exported into test sets (75 each,
`arabphd_test_set_sec.json` / `arabphd_test_set_icc.json`). None overlap with
the 96 candidates flagged `rejected` in the review pass. Still missing
before promotion into the main pool: explicit question text
(`question_yes_ar`/`question_no_ar`-equivalent) per the hard promotion rule
in `image_schema.json`, and an evaluation script (no `run_mode2_openrouter.py`
yet).

**Mode 4 (ccs), 15/15 images organized and evaluated end-to-end, same
Yes/No protocol as Liu et al. 2025's PhD-ccs.**
`modes/mode4_ccs/arabphd_ccs_questions.json` holds 15 AI-generated Arab/
Islamic cultural-norm-violation images (`modes/mode4_ccs/images/`) across 6
categories (Sacred Space Violations, Religious & Attire Contradictions,
Seasonal & Holiday Context Mix, Sacred Text & OCR Hijacking, Subtle
Historical Anachronisms, Tashkeel Diacritics Contrast). Each image carries
the originally-delivered open-ended trap question/ground-truth/
failure_mechanism (kept as documentation) plus a proper Yes/No pair
(`question_yes_ar`/`answer_yes`, `question_no_ar`/`answer_no`), matching
the paper's CCS/CS pairing (Table 3): `question_yes_ar` is the actual norm
violation depicted (GT=نعم), `question_no_ar` is an authored, plausible
"normal" counterpart that is *not* depicted (GT=لا) -- mechanically derived
from each record's existing ground truth, the same way Mode 1's h_item-based
no-questions were derived, not new invented facts. `run_mode4_openrouter.py`
scores it exactly like Mode 1 (base): forced single-word yes/no output,
same system instruction, no context text (PhD-ccs is plain image+question
per the paper), same accuracy / Yes-recall / No-recall / PhD Index metrics.
An earlier free-text + LLM-judge version of this script and its results are
kept in `modes/mode4_ccs/backups/pre_yesno_protocol_20260829/` for
reference. Only 15 images / 30 questions exist so far, short of the
proposal's 50-triplet target for this mode.

```bash
cd modes/mode4_ccs
python3 run_mode4_openrouter.py run --concurrency 5
python3 run_mode4_openrouter.py metrics --file mode4_results/results_mode4_ccs_google-gemini-2.5-flash.csv
python3 run_mode4_openrouter.py table --out mode4_results/comparison_table.md
```

| model | n | accuracy | Yes-recall | No-recall | PhD Index | clarity |
|---|---:|---:|---:|---:|---:|---:|
| google/gemini-2.5-flash | 30 | 93.3% | 93.3% | 93.3% | 0.933 | 100.0% |
| qwen/qwen2.5-vl-72b-instruct | 30 | 90.0% | 86.7% | 93.3% | 0.899 | 100.0% |
| openai/gpt-4o-mini | 30 | 83.3% | 93.3% | 73.3% | 0.821 | 100.0% |
| google/gemini-2.5-flash-lite | 30 | 73.3% | 73.3% | 73.3% | 0.733 | 100.0% |

Mode 5 (nota) was reset and rebuilt
2026-08-28 from the 50 sec/icc candidates not used in the Mode 2/3 test sets
(the old 350 records were built off a stale, pre-review pool and are
backed up, not deleted); it now has 120 records (30 images x mcdr/oedr/udr +
control), fully mechanically generated and disjoint from Mode 1/sec/icc/
rejected images; see [Stage 2b](#pipeline) below.

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
  mode2_sec/
    candidate_pool_sec.json        # 98 candidates with misleading_caption_ar/en + trap_reasoning; no questions yet
    arabphd_test_set_sec.json      # 75 exported for evaluation once questions exist
  mode3_icc/
    candidate_pool_icc.json        # 102 candidates with misleading_caption_ar/en + trap_reasoning; no questions yet
    arabphd_test_set_icc.json      # 75 exported for evaluation once questions exist
  mode4_ccs/
    arabphd_ccs_questions.json    # 15 CCS images x Yes/No pair (+ original trap question/ground truth as docs)
    images/                        # the 15 AI-generated CCS images (renamed from their original messy filenames)
    run_mode4_openrouter.py       # OpenRouter inference script: run / metrics / table subcommands
    mode4_results/                 # results_mode4_ccs_<model>.csv per run, + comparison_table.md
    backups/pre_yesno_protocol_20260829/  # earlier free-text + LLM-judge version, kept for reference
  mode5_nota/                     # rebuilt 2026-08-28, not yet wired to arabphd_full_candidate_pool.json
    fill_nota_questions.py        # status / select / next / generate / validate / merge
    nota_question_prompt.md
    arabphd_nota_questions.json   # 120 records: 30 images x mcdr/oedr/udr + control
    nota_batches/
    backups/                      # includes the pre-reset 350-record file and old batches

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

**Stage 2b: nota (Mode 5), reset and rebuilt 2026-08-28.** `modes/mode5_nota/nota_question_prompt.md`
defines the MCDR/OEDR/UDR triplet (plus a matched control) each source item
turns into, using `inferred_identity_ar` and `implicit_rejection_set`. Every
field that produces is fully mechanical given those two, so
`modes/mode5_nota/fill_nota_questions.py generate` builds it directly (no
separate LLM authoring pass): `status`/`select` track the quota, `next`
prints hints for manual inspection, `generate` mechanically assembles a
batch, `validate`/`merge` gate it into `arabphd_nota_questions.json` the
same as before.

Source is now the 50 sec/icc candidates *not* exported into the Mode 2/3
test sets (75 went to each of `modes/mode2_sec/`/`modes/mode3_icc/`, out of
the 200-record corrected, peer-reviewed sec/icc pool), so no image is reused
across sec, icc, and nota. That dedups by unique identity down to 30 items,
all control-eligible at this smaller scale: **120 records total** (30 x
mcdr/oedr/udr + control), verified to have zero image overlap with Mode 1,
the sec/icc test sets, or the 96 candidates flagged `rejected` in the review
pass. The prior 350-record set was built off a stale, pre-review pool;
backed up (not deleted) to `modes/mode5_nota/backups/`.

```bash
cd modes/mode5_nota
python3 fill_nota_questions.py status
python3 fill_nota_questions.py generate --apply   # rebuild + merge a fresh batch if the quota ever grows
```

`validate` checks schema conformance (no answer leakage, distractors match
the pool, option counts correct, control records reinstate the real answer).
Still needs native-speaker verification per the Annotation Pipeline, and
isn't wired into `arabphd_full_candidate_pool.json` or an evaluation script
yet.

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
