# MM-UPD (Miyai et al., ACL 2025) — reference for ArabPhD-nota (Mode 5)

Source: **Unsolvable Problem Detection: Robust Understanding Evaluation for
Large Multimodal Models**
- Paper: https://arxiv.org/abs/2403.20331
- Code: https://github.com/AtsuMiyai/UPD
- Dataset: https://huggingface.co/datasets/MM-UPD/MM-UPD
- Images: sourced from MMBench (embedded as base64 JPEG directly in the TSVs)

## Important distinction — read this first

The design note (`ArabPhD_Modes_and_Recommendation.md`) describes nota's
"3 things" as **MCDR / OEDR / UDR** — three ways of *asking* about the same
underlying absent-answer items (with NOTA option shown / not shown but
permitted / not mentioned at all). That specific 3-condition protocol comes
from **Wang et al. 2026** (arXiv:2606.08239, Duke, video QA) — a very recent
paper (submitted June 2026) that does not appear to have a public dataset
release yet; it's a diagnostic *protocol* applied to existing video-QA
benchmarks, not a new item bank. Nothing was found to download from it.

**MM-UPD's "3 things" are different**: three *types of unsolvability*, not
three prompting conditions —
1. **AAD** (Absent Answer Detection) — correct option is deleted from the
   choice set. This is the closest direct precedent to ArabPhD-nota.
2. **IASD** (Incompatible Answer Set Detection) — the whole option set is
   swapped for one from an unrelated question (random shuffle).
3. **IVQD** (Incompatible Visual Question Detection) — the question doesn't
   match the image at all (image and question pairs are shuffled).

Both papers share the same *underlying idea* (does the model fabricate an
answer instead of recognizing nothing fits?) and MM-UPD is cited directly in
the ArabPhD-nota design doc's source table. Treat MM-UPD as the concrete,
downloadable methodology reference; treat Wang et al. as the source of the
specific MCDR/OEDR/UDR condition-naming/protocol to reuse.

## What's here

Each config below also has a matched **"standard"** file — the same
question, but with the correct answer left in (the *solvable* control). This
is exactly the "matched control subset" ArabPhD-nota calls for to compute
the false-abstention rate — MM-UPD already does this, per config, at scale.

| File | Config (HF: `load_dataset("MM-UPD/MM-UPD", "<config>")`) | What it is |
|---|---|---|
| `aad_sample.json` | `mmaad_base` (aad subset) | correct option deleted |
| `iasd_sample.json` | `mmiasd_base` (iasd subset) | answer set swapped for a random one |
| `ivqd_sample.json` | `mmivqd_base` (ivqd subset) | image/question mismatched |
| `standard_matched_control_sample.json` | `mmaad_base` (standard subset) | matched control, answer present |

3 rows each (full sizes: AAD 820, IASD 919, IVQD 356 questions — TSVs are
74–224MB each because images are embedded as base64, so only samples were
pulled here via HTTP range requests, not full files).

`example_images/` has one decoded JPEG per config (the image for row 0 of
each sample file).

## How to read a record (AAD example)

```json
{
  "index": "1",
  "question": "What is correct Python code to generate the content of the image?",
  "A": "...", "B": "...", "C": "...", "D": "", "E": "",
  "answer": "F",
  "masked_answer": "fruits = [\"apple\", \"banana\", \"cherry\"]\nfor x in fruits:\n  print(x)",
  "category": "structuralized_imagetext_understanding",
  "image": "<base64 JPEG>",
  "source": "code",
  "l2-category": "logic_reasoning",
  "split": "test",
  "type": "upd"
}
```

- `answer: "F"` — "F" isn't an option letter here, it's their fixed code for
  **"none of the above / can't be determined"** (mirrors ArabPhD-nota's
  planned *lā tūjad ijāba ṣaḥīḥa* option). A model scores correctly only if
  it picks F.
- `masked_answer` — the correct answer that was **removed** from A–E, kept
  for provenance/auditing. Directly analogous to ArabPhD-nota's
  "remove the correct answer, keep 2–3 grounded distractors" design.
- The **standard** file for the same `index` has the real answer back in
  A–E and `answer` pointing at a real letter — that's the matched control.

## How to download the full dataset yourself

```python
from datasets import load_dataset
ds = load_dataset("MM-UPD/MM-UPD", "mmaad_base")   # or mmiasd_base, mmivqd_base, +"_option" variant
```

or via `huggingface_hub.list_repo_files("MM-UPD/MM-UPD", repo_type="dataset")`
to browse/pull individual TSVs directly.
