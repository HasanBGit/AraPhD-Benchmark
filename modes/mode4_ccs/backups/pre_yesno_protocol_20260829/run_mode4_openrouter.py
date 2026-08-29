"""
Mode 4 (ArabPhD-ccs, Cultural Counter-Common-Sense) evaluation via OpenRouter.

Protocol departs from Mode 1/2/3 on purpose -- read this before assuming it's
a bug that there's no Yes/No parsing here.

Liu et al. 2025 (CVPR) build PhD-ccs as strict binary VQA: a CCS description
("a car with square wheels") is mechanically turned into a Yes question and
a paired No question, scored the same way as PhD-base (see info/Liu_PhD_A_
ChatGPT-Prompted_Visual_Hallucination_Evaluation_Dataset_CVPR_2025_paper (1)
.pdf, secs 3.3/3.5). The 15 ArabPhD-ccs images and questions actually
delivered for this mode (modes/mode4_ccs/arabphd_ccs_questions.json,
originally gemini-code-1787877704755.json at the repo root) are NOT built
that way: each image carries one open-ended "trap" question
(vqa_trap_ar/en) plus a descriptive Arabic ground-truth explanation of the
norm violation (ground_truth_ar/en) and a failure_mechanism note, not a
Yes/No pair. Forcing that into a Yes/No template would mean inventing
question text that doesn't exist in the source data, so this script scores
the data as delivered instead.

That puts Mode 4 in the "subjective evaluation" quadrant of Liu et al.'s own
Table 1 (human/LLM judge against free-text output) rather than the
"objective evaluation" (Yes/No) quadrant every other ArabPhD mode uses.
Protocol here, two calls per image:
  1. Vision call: target MLLM sees the image + vqa_trap_ar, answers freely
     in Arabic (temperature 0, no forced single-word output).
  2. Judge call: a text-only judge model compares that answer against
     ground_truth_ar/failure_mechanism and outputs one word: "صحيح"
     (correct -- the violation was caught) or "هلوسة" (hallucinated --
     it was missed/normalized, matching failure_mechanism).

Reported metric is CCS Detection Rate (== judge-correct / total), the
Cause-III analogue of accuracy; there is no PhD Index here since there is no
Yes/No pair to take a harmonic mean of.

Usage (run from modes/mode4_ccs/, or anywhere -- paths resolve from repo root):
    export OPENROUTER_API_KEY=sk-or-...      # or a .env file at repo root
    python3 run_mode4_openrouter.py run --dry-run
    python3 run_mode4_openrouter.py run --concurrency 5
    python3 run_mode4_openrouter.py run --model openai/gpt-4o-mini --judge-model google/gemini-2.5-flash
    python3 run_mode4_openrouter.py metrics --file mode4_results/results_mode4_ccs_google-gemini-2.5-flash.csv
    python3 run_mode4_openrouter.py table
"""
import argparse
import base64
import csv
import json
import mimetypes
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
QUESTION_SET = HERE / "arabphd_ccs_questions.json"
RESULTS_DIR = HERE / "mode4_results"
MODE_LABEL = "mode4_ccs"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_AR = (
    "انظر إلى الصورة بعناية وأجب عن السؤال التالي باللغة العربية بالاعتماد "
    "حصراً على ما تراه فعلياً في الصورة. صف أي عنصر يبدو غير معتاد أو "
    "متناقضاً مع سياق المشهد إن وُجد. لا تفترض وجود عناصر غير ظاهرة، ولا "
    "تتجاهل أي تفصيل ظاهر مهما بدا غير متوقع. اكتب إجابة موجزة (2-3 جمل)."
)
JUDGE_SYSTEM_INSTRUCTION_AR = (
    "أنت مقيّم يقارن إجابة نموذج بصري بالإجابة المرجعية الصحيحة لسؤال عن "
    "صورة صُممت عمداً لتحتوي على مخالفة ثقافية أو دينية أو تناقض بصري "
    "متعمد. حدد هل ذكرت إجابة النموذج المخالفة/التناقض الموصوف في الإجابة "
    "المرجعية بوضوح (حتى لو بصياغة مختلفة)، أم فشلت في ذلك (تجاهلته، "
    "طبّعته، أو لم تلاحظه إطلاقاً).\n\n"
    "أجب بكلمة واحدة فقط: \"صحيح\" إذا رصد النموذج المخالفة، أو \"هلوسة\" "
    "إذا فشل في رصدها. لا تضف أي شرح أو علامات ترقيم إضافية."
)
TEMPERATURE = 0
PROMPT_VERSION = 1
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
VISION_MAX_TOKENS = 300
JUDGE_MAX_TOKENS = 16
FALLBACK_MAX_TOKENS = 512

FIELDNAMES = [
    "task_id", "image_id", "filename", "category",
    "vqa_trap_ar", "ground_truth_ar", "failure_mechanism", "raw_response",
    "judge_verdict", "is_correct", "error_type", "detail", "attempts",
    "latency_s", "model_name", "judge_model", "provider",
    "prompt_version", "temperature", "timestamp",
]
PERMANENT_ERRORS = {"image_load_error", "safety_block", "api_error_400"}
# api_error_403 is deliberately NOT permanent: OpenRouter returns 403 for both
# real auth failures and "key limit exceeded" (a prepaid-credit cap), and the
# latter is exactly the kind of thing a re-run after topping up should retry.

_CORRECT_WORDS = {"صحيح", "correct"}
_HALLUCINATED_WORDS = {"هلوسة", "hallucinated", "خطأ"}


# ---------------------------------------------------------------------------
def load_question_set():
    return json.loads(QUESTION_SET.read_text(encoding="utf-8"))


def build_tasks(records, limit=None):
    tasks = []
    for r in records:
        image_path = str((REPO_ROOT / r["local_path"]).resolve())
        tasks.append({
            "task_id": r["image_id"], "image_id": r["image_id"], "filename": r["filename"],
            "image_path": image_path, "category": r["category"],
            "vqa_trap_ar": r["vqa_trap_ar"], "ground_truth_ar": r["ground_truth_ar"],
            "failure_mechanism": r["failure_mechanism"],
        })
    if limit:
        tasks = tasks[:limit]
    return tasks


def preflight(tasks):
    """Skip tasks whose image is missing rather than hard-fail the whole run."""
    missing = [t for t in tasks if not os.path.isfile(t["image_path"]) or os.path.getsize(t["image_path"]) == 0]
    if missing:
        missing_ids = {t["task_id"] for t in missing}
        print(f"⚠️  skipping {len(missing)} task(s) with a missing/empty image file:")
        for t in missing[:20]:
            print("   ", t["task_id"], "->", t["image_path"])
        tasks[:] = [t for t in tasks if t["task_id"] not in missing_ids]
    return tasks


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OPENROUTER_API_KEY":
                return v.strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found. export it or add it to a .env file at the repo root.")


def parse_verdict(raw):
    if not raw:
        return "unclear"
    text = raw.strip().strip(" \n\t\"'.,!?؟،؛:-")
    if text in _CORRECT_WORDS or text.lower() in _CORRECT_WORDS:
        return "correct"
    if text in _HALLUCINATED_WORDS or text.lower() in _HALLUCINATED_WORDS:
        return "hallucinated"
    tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    has_correct = any(t in _CORRECT_WORDS or t.lower() in _CORRECT_WORDS for t in tokens)
    has_halluc = any(t in _HALLUCINATED_WORDS or t.lower() in _HALLUCINATED_WORDS for t in tokens)
    if has_correct and not has_halluc:
        return "correct"
    if has_halluc and not has_correct:
        return "hallucinated"
    return "unclear"


def guess_mime(path):
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/jpeg"


def call_vision_model(session, api_key, model, image_path, prompt_text, max_tokens, reasoning_off, timeout):
    with open(image_path, "rb") as fh:
        img_bytes = fh.read()
    data_uri = f"data:{guess_mime(image_path)};base64,{base64.b64encode(img_bytes).decode('ascii')}"
    payload = {
        "model": model, "temperature": TEMPERATURE, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION_AR},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
    }
    if reasoning_off:
        payload["reasoning"] = {"enabled": False}
    return _post(session, api_key, payload, timeout)


def call_judge_model(session, api_key, judge_model, task, raw_response, max_tokens, timeout):
    user_text = (
        f"السؤال: {task['vqa_trap_ar']}\n"
        f"الإجابة المرجعية (الحقيقة الأرضية): {task['ground_truth_ar']}\n"
        f"نمط الفشل المتوقع: {task['failure_mechanism']}\n\n"
        f"إجابة النموذج التي يجب تقييمها: {raw_response}\n\n"
        "هل رصد النموذج المخالفة؟ أجب بكلمة واحدة: صحيح أو هلوسة"
    )
    payload = {
        "model": judge_model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTION_AR},
            {"role": "user", "content": user_text},
        ],
    }
    return _post(session, api_key, payload, timeout)


def _post(session, api_key, payload, timeout):
    headers = {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HasanBGit/AraPhD-Benchmark",
        "X-Title": f"ArabPhD {MODE_LABEL} evaluation",
    }
    return session.post(API_URL, headers=headers, json=payload, timeout=timeout)


def _extract_text(resp):
    """Returns (text, error_type, detail) -- error_type empty on success."""
    if resp.status_code == 400:
        try:
            msg = resp.json().get("error", {}).get("message", resp.text)
        except ValueError:
            msg = resp.text
        return "", "api_error_400", msg[:200]
    if resp.status_code == 403:
        return "", "api_error_403", resp.text[:200]
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        et = "rate_limited" if resp.status_code == 429 else f"server_error_{resp.status_code}"
        return "", et, resp.text[:200]
    if resp.status_code != 200:
        return "", f"api_error_{resp.status_code}", resp.text[:200]
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    raw = ((choice.get("message") or {}).get("content") or "").strip()
    finish_reason = (choice.get("finish_reason") or "").lower()
    if not raw:
        blocked = any(k in finish_reason for k in ("content_filter", "safety"))
        return "", "safety_block" if blocked else "empty_response", finish_reason
    return raw, "", ""


def mk_result(task, model, judge_model, raw, judge_verdict, error_type, detail, attempts, latency, reasoning_off):
    is_correct = int(judge_verdict == "correct")
    return {
        "task_id": task["task_id"], "image_id": task["image_id"], "filename": task["filename"],
        "category": task["category"], "vqa_trap_ar": task["vqa_trap_ar"],
        "ground_truth_ar": task["ground_truth_ar"], "failure_mechanism": task["failure_mechanism"],
        "raw_response": raw, "judge_verdict": judge_verdict, "is_correct": is_correct,
        "error_type": error_type, "detail": detail, "attempts": attempts, "latency_s": latency,
        "model_name": model, "judge_model": judge_model, "provider": "openrouter",
        "prompt_version": PROMPT_VERSION, "temperature": TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_one_task(session, api_key, model, judge_model, task, min_gap, max_retries, state, last_call):
    if not os.path.isfile(task["image_path"]):
        return mk_result(task, model, judge_model, "", "unclear", "image_load_error", "file not found", 0, 0.0, True)

    reasoning_off = state.get(model, True)
    raw, error_type, detail, attempts, latency = "", "", "", 0, 0.0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        gap = min_gap - (time.time() - last_call[0])
        if gap > 0:
            time.sleep(gap)
        max_tokens = VISION_MAX_TOKENS if reasoning_off else FALLBACK_MAX_TOKENS
        t0 = time.time()
        last_call[0] = t0
        try:
            resp = call_vision_model(session, api_key, model, task["image_path"], task["vqa_trap_ar"],
                                      max_tokens, reasoning_off, timeout=60)
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            return mk_result(task, model, judge_model, "", "unclear", "exception", str(e)[:200], attempt, round(time.time() - t0, 2), reasoning_off)

        latency = round(time.time() - t0, 2)
        raw, error_type, detail = _extract_text(resp)

        if error_type == "api_error_400" and "reasoning" in detail.lower() and reasoning_off:
            state[model] = False
            reasoning_off = False
            continue
        if error_type == "empty_response" and reasoning_off and attempt < max_retries:
            reasoning_off = False
            continue
        if error_type in ("rate_limited",) or error_type.startswith("server_error_"):
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
        break

    if error_type:
        return mk_result(task, model, judge_model, raw, "unclear", error_type, detail, attempts, latency, reasoning_off)

    # -- judge call --
    for jattempt in range(1, max_retries + 1):
        try:
            jresp = call_judge_model(session, api_key, judge_model, task, raw, JUDGE_MAX_TOKENS, timeout=30)
        except requests.RequestException as e:
            if jattempt < max_retries:
                time.sleep(min(2 ** jattempt, 30) + random.uniform(0, 1))
                continue
            return mk_result(task, model, judge_model, raw, "unclear", "judge_exception", str(e)[:200], attempts, latency, reasoning_off)
        jtext, jerror, jdetail = _extract_text(jresp)
        if jerror in ("rate_limited",) or jerror.startswith("server_error_"):
            if jattempt < max_retries:
                time.sleep(min(2 ** jattempt, 30) + random.uniform(0, 1))
                continue
        if jerror:
            return mk_result(task, model, judge_model, raw, "unclear", f"judge_{jerror}", jdetail, attempts, latency, reasoning_off)
        return mk_result(task, model, judge_model, raw, parse_verdict(jtext), "", "", attempts, latency, reasoning_off)

    return mk_result(task, model, judge_model, raw, "unclear", "judge_exhausted_retries", "", attempts, latency, reasoning_off)


def is_final(row):
    return row.get("judge_verdict") in ("correct", "hallucinated") or row.get("error_type") in PERMANENT_ERRORS


def cmd_run(args):
    records = load_question_set()
    tasks = build_tasks(records, args.limit)
    preflight(tasks)

    model_slug = re.sub(r"[^a-z0-9.]+", "-", args.model.lower())
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"results_{MODE_LABEL}_{model_slug}.csv"

    print(f"model={args.model} judge={args.judge_model} tasks={len(tasks)} out={out_path}")
    if args.dry_run:
        print("dry run — no API calls made.")
        return

    api_key = load_api_key()
    done = set()
    if out_path.is_file():
        with open(out_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if is_final(row):
                    done.add(row["task_id"])
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"already complete: {len(done)} | remaining: {len(todo)}")
    if not todo:
        print("nothing to do.")
        cmd_metrics_from_path(out_path)
        return

    state = {}
    new_file = not out_path.is_file()
    session = requests.Session()
    write_lock = threading.Lock()

    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()

        if args.concurrency <= 1:
            last_call = [0.0]
            consec_rate_limited = 0
            for i, task in enumerate(todo, 1):
                res = run_one_task(session, api_key, args.model, args.judge_model, task, args.min_gap, args.max_retries, state, last_call)
                writer.writerow(res)
                fh.flush()
                status = res["judge_verdict"] if not res["error_type"] else f"⚠️{res['error_type']}"
                print(f"[{i}/{len(todo)}] {task['task_id']:<20} -> {status}")
                if res["error_type"] == "rate_limited":
                    consec_rate_limited += 1
                    if consec_rate_limited >= 5:
                        print("stopping: 5 consecutive rate-limit errors. Re-run later to resume.")
                        break
                else:
                    consec_rate_limited = 0
        else:
            print(f"running {len(todo)} tasks with concurrency={args.concurrency} (no pacing between requests)")
            done_count = 0
            rate_limited_count = 0

            def _worker(task):
                return run_one_task(session, api_key, args.model, args.judge_model, task, 0.0, args.max_retries, state, [0.0])

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_worker, t): t for t in todo}
                for fut in as_completed(futures):
                    task = futures[fut]
                    res = fut.result()
                    with write_lock:
                        writer.writerow(res)
                        fh.flush()
                        done_count += 1
                        status = res["judge_verdict"] if not res["error_type"] else f"⚠️{res['error_type']}"
                        print(f"[{done_count}/{len(todo)}] {task['task_id']:<20} -> {status}")
                        if res["error_type"] == "rate_limited":
                            rate_limited_count += 1
            if rate_limited_count:
                print(f"note: {rate_limited_count}/{len(todo)} tasks hit rate limits -- re-run to resume the rest.")

    print(f"\nsaved results to {out_path}")
    cmd_metrics_from_path(out_path)


# ---------------------------------------------------------------------------
def compute_metrics(path):
    rows_by_task = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows_by_task[row["task_id"]] = row
    rows = list(rows_by_task.values())

    def b(x):
        return str(x) == "1"

    total = len(rows)
    if total == 0:
        return None
    parsed = [r for r in rows if r["judge_verdict"] in ("correct", "hallucinated")]
    detection_rate = sum(b(r["is_correct"]) for r in rows) / total

    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    category_rates = {
        cat: sum(b(r["is_correct"]) for r in cat_rows) / len(cat_rows)
        for cat, cat_rows in by_category.items()
    }

    model_names = {r.get("model_name") for r in rows if r.get("model_name")}
    return {
        "path": path, "model_name": next(iter(model_names), "?") if len(model_names) == 1 else "/".join(sorted(model_names)),
        "n_tasks": total, "n_parsed": len(parsed), "clarity": len(parsed) / total,
        "detection_rate": detection_rate, "category_rates": category_rates,
        "error_types": dict(Counter(r["error_type"] for r in rows if r["error_type"])),
    }


def cmd_metrics_from_path(path):
    m = compute_metrics(path)
    if m is None:
        print("no rows.")
        return
    print(f"\n========== {Path(path).name} ==========")
    print(f"tasks: {m['n_tasks']} | judged: {m['n_parsed']} | clarity: {m['clarity']:.1%}")
    print(f"CCS Detection Rate: {m['detection_rate']:.1%}")
    print("by category:")
    for cat, rate in sorted(m["category_rates"].items()):
        print(f"  {cat}: {rate:.1%}")
    if m["error_types"]:
        print("error types:", m["error_types"])


def cmd_metrics(args):
    cmd_metrics_from_path(Path(args.file))


def cmd_table(args):
    files = sorted(RESULTS_DIR.glob(f"results_{MODE_LABEL}_*.csv"))
    rows = [m for m in (compute_metrics(f) for f in files) if m]
    if not rows:
        print(f"no results CSVs found in {RESULTS_DIR}")
        return
    rows.sort(key=lambda m: m["detection_rate"], reverse=True)
    header = ["model", "n", "CCS Detection Rate", "clarity"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in rows:
        lines.append("| {model} | {n} | {dr:.1%} | {cl:.1%} |".format(
            model=m["model_name"], n=m["n_tasks"], dr=m["detection_rate"], cl=m["clarity"]))
    table_md = "\n".join(lines)
    print(table_md)
    if args.out:
        Path(args.out).write_text(table_md + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    r.add_argument("--limit", type=int, default=None, help="cap number of tasks (1 per image)")
    r.add_argument("--out", default=None)
    r.add_argument("--min-gap", type=float, default=1.0)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--max-retries", type=int, default=5)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("metrics")
    m.add_argument("--file", required=True)
    m.set_defaults(func=cmd_metrics)

    t = sub.add_parser("table")
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_table)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
