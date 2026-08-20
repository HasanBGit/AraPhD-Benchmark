# PhD (Liu et al., CVPR 2025) — original dataset, samples for all 4 modes

Source: **PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset**
- Paper: https://arxiv.org/abs/2403.11116
- Code/data.json: https://github.com/jiazhen-code/PhD
- Full dataset also on HuggingFace: https://huggingface.co/datasets/AIMClab-RUC/PhD
- Images: COCO 2014 (train2014/val2014) for base/iac/icc; AI-generated for ccs
  (Google Drive, linked from the GitHub README — not mirrored here, see caveat below)

This is the benchmark ArabPhD's Modes 1–4 (base/sec/icc/ccs) directly mirror.
ArabPhD's "sec" = PhD's "iac" (inaccurate context) — renamed but same role.

## What's here

- `phd_samples_base_iac_icc.json` — 10 records (2 per task × 5 tasks: object,
  attribute, sentiment, positional, counting), pulled from the full
  `data.json` (17,597 records covering PhD-base/iac/icc — these three modes
  share one record each, since `context.iac` and `context.icc` are just two
  extra fields hung off the same base yes/no question pair).
- `phd_samples_ccs.json` — 6 records from the separate 753-record CCS pool
  (AI-generated counter-common-sense images; no `hitem`/`context`, has
  `ccs_description` instead).
- `example_images/` — the actual COCO images for the 10 base/iac/icc samples
  above, fetched from `images.cocodataset.org` by `image_id` (filename tells
  you train2014 vs val2014).

## How to read one record (base/iac/icc)

```json
{
  "task": "attribute",
  "yes_question": "Is the table made of wood in the image?",
  "no_question": "Is the table made of glass in the image?",
  "context": {
    "icc": "<a whole misleading paragraph, factually wrong about the image>",
    "iac": "<a whole paragraph of true-but-irrelevant noise>"
  },
  "image_id": "000000044432",
  "hitem": "glass",
  "subject": "table",
  "gt": "wood"
}
```

- **PhD-base** = image + `yes_question` (answer: yes) and image + `no_question`
  (answer: no). Two triplets per record, one per polarity — that's how they
  hit 102k triplets from far fewer underlying images/scenes.
- **PhD-iac** = same two questions, but the prompt is prefixed/paired with
  `context.iac` (true but irrelevant noise text) before asking.
- **PhD-icc** = same two questions, paired with `context.icc` (a fluent
  paragraph that's flatly wrong about the image — e.g. claims a glass table
  when it's wood). Tests whether the model trusts the text over the image.
- `hitem` = the specific hallucination-inducing term (here "glass" — plausible
  but wrong material for that subject). `gt` = ground truth.

## How to read a CCS record

```json
{
  "yes_question": "Are the ostrich feathers green?",
  "no_question": "Are the ostrich feathers black?",
  "ccs_description": "The image shows ostrich feathers as green, which is
    unusual since they are typically black or brown...",
  "image_id": "232",
  "task": "attribute"
}
```

Same yes/no question-pair structure as base, but the *image itself* (not the
text) is the counter-common-sense element — `image_id` here indexes into
`CCS_images/{id}.png`, AI-generated, not COCO.

## Caveat / what's not included

- Only 10+6 sample records here, not the full 17,597+753. `data.json`
  (19MB) was downloaded in full during this session but not committed —
  re-fetch from the GitHub link above if you need more than the samples.
- CCS images themselves are on Google Drive (not a direct-download API-friendly
  link), so `example_images/` only has the base/iac/icc COCO photos, not a
  CCS image. The `ccs_description` field is enough to see the QA structure
  even without the image.
