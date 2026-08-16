"""
Merge manifest_unified.csv into the unified per-image annotation schema
(stage 1 of the pipeline: per-image pool -> stage 2: per-mode triplets).

Usage:
    python3 build_image_pool.py
Writes image_pool.json into ../data/candidate_pool/.
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "candidate_pool"
MANIFEST = DATA_DIR / "manifest_unified.csv"
OUT = DATA_DIR / "image_pool.json"


def slugify(text: str) -> str:
    text = re.sub(r"\.[A-Za-z]+$", "", text)  # drop extension
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.lower()


def none_if_blank(value: str):
    value = (value or "").strip()
    return value if value else None


def build_entry(row: dict) -> dict:
    category = row["category"]
    image_id = f"{category}_{slugify(row['filename'])}"

    return {
        "image_id": image_id,
        "filename": row["filename"],
        # computed from the current repo layout, not trusted from the manifest
        # (manifest_unified.csv's own local_path column predates this repo's move/reorg)
        "local_path": f"data/candidate_pool/images/{category}/{row['filename']}",
        "category": category,
        # existing provenance / manifest metadata (carried through, not regenerated)
        "subject": none_if_blank(row["subject"]),
        "ground_truth_subject": none_if_blank(row["ground_truth_subject"]),
        "country": none_if_blank(row["country"]),
        "region": none_if_blank(row["region"]),
        "medium": none_if_blank(row["medium"]),
        "license": none_if_blank(row["license"]),
        "artist": none_if_blank(row["artist"]),
        "source": none_if_blank(row["source"]),
        # to be filled by the vision/verification + translation pass (stage 1b)
        "inferred_identity_ar": None,
        "inferred_identity_en": None,
        "caption_ar": None,
        "h_item_ar": None,
    }


def main():
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pool = [build_entry(row) for row in rows]

    by_category = {}
    for entry in pool:
        by_category[entry["category"]] = by_category.get(entry["category"], 0) + 1

    OUT.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(pool)} entries to {OUT}")
    print("By category:", by_category)


if __name__ == "__main__":
    main()
