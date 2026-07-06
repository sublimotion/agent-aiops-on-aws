"""Stratified train + eval data builder for Phase 1a (and optional 1b).

Fetches from Hugging Face datasets, normalizes to a single schema, and writes
JSONL files. Idempotent: re-running with same --seed produces byte-identical output.

Output schema (each line is one record):
    {
      "id": "math500-train-0000",
      "dataset": "math500",            # used as `dataset` arg to extract/grade
      "question": "...",
      "answer": "...",                 # gold; for code: test code; for triviaqa: alias list
      "entry_point": "...",            # code only
      "license": "MIT",
      "source": "hendrycks_test",      # canonical HF dataset name
      "metadata": { ... }              # dataset-specific extras
    }

Usage:
    # Phase 1a only (default):
    python -m scripts.build_data --out-dir data/ --seed 17

    # Phase 1a + 1b:
    python -m scripts.build_data --out-dir data/ --seed 17 --include-phase1b

    # Dry run (counts only, no HF download):
    python -m scripts.build_data --dry-run

License gates:
    - QuALITY is non-commercial → script REFUSES to include unless
      --accept-quality-license is passed; otherwise substitutes LongBench-v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset spec: how to fetch, normalize, and split
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """One row in the dataset matrix."""
    name: str                                   # canonical name we use ('math500', 'gsm8k', ...)
    hf_path: str                                # HF dataset ID
    hf_config: Optional[str]                    # optional HF config name
    split: str                                  # 'train' | 'test' | 'validation' etc.
    license: str
    train_count: int                            # how many to take for our train set
    eval_count: int                             # how many to take for our eval set (held-out)
    normalize: Callable[[dict], Optional[dict]] # row → our schema or None to skip
    phase: str = "1a"                           # '1a' or '1b'
    contamination_filter: Optional[Callable[[dict], bool]] = None  # row → True to keep


# ---------------------------------------------------------------------------
# Per-dataset normalizers
# ---------------------------------------------------------------------------

def _norm_math500(row):
    return {
        "question": row["problem"],
        "answer": row["answer"],
        "metadata": {"level": row.get("level"), "type": row.get("type")},
    }


def _norm_gsm8k(row):
    # GSM8K answer is "reasoning... #### N"
    return {
        "question": row["question"],
        "answer": row["answer"],
        "metadata": {},
    }


def _norm_mmlu(row):
    # HF cais/mmlu schema: question, choices, answer (int 0-3)
    if not isinstance(row.get("answer"), int):
        return None
    return {
        "question": (row["question"] + "\n\n"
                     + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"]))),
        "answer": chr(65 + row["answer"]),  # → "A".."D"
        "metadata": {"subject": row.get("subject")},
    }


def _norm_mmlu_pro(row):
    # TIGER-Lab/MMLU-Pro: options is a list, answer is letter A-J
    if not row.get("answer"):
        return None
    options_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row["options"]))
    return {
        "question": row["question"] + "\n\n" + options_str,
        "answer": row["answer"],
        "metadata": {"category": row.get("category"), "src": row.get("src")},
    }


def _norm_triviaqa(row):
    # mandarjoshi/trivia_qa rc.nocontext: question + answer.aliases (list)
    aliases = row["answer"]["aliases"]
    if not aliases:
        return None
    return {
        "question": row["question"],
        "answer": aliases,
        "metadata": {"source": row.get("source"), "value": row["answer"]["value"]},
    }


def _norm_humaneval(row):
    return {
        "question": row["prompt"],
        "answer": row["test"],
        "entry_point": row["entry_point"],
        "metadata": {"task_id": row["task_id"]},
    }


def _norm_mbpp(row):
    # google-research-datasets/mbpp: text, code, test_list, test_setup_code
    test_block = row.get("test_setup_code", "") + "\n" + "\n".join(row["test_list"])
    return {
        "question": row["text"],
        "answer": test_block,
        "entry_point": "",  # MBPP tests are standalone asserts, no check(fn) call needed
        "metadata": {"task_id": row["task_id"]},
    }


def _norm_livecodebench(row):
    # livecodebench: question_content, public_test_cases, private_test_cases, contest_date
    # We use private_test_cases (real eval signal). 'date' is YYYY-MM-DD.
    private = row.get("private_test_cases", [])
    if not private:
        return None
    # Build a small test runner that asserts each (input, output) pair
    test_lines = []
    for tc in private[:5]:  # first 5 to keep grader fast
        test_lines.append(f"assert solution({tc['input']!r}) == {tc['output']!r}")
    return {
        "question": row["question_content"],
        "answer": "\n".join(test_lines),
        "entry_point": "solution",
        "metadata": {"contest_date": row.get("contest_date"), "difficulty": row.get("difficulty")},
    }


def _lcb_cutoff_filter(row) -> bool:
    """Drop LiveCodeBench items after 2025-01-01 to avoid contamination."""
    d = row.get("contest_date") or row.get("date") or ""
    return d < "2025-02"   # keep through 2025-01


def _norm_gpqa_diamond(row):
    # Idavidrein/gpqa, gpqa_diamond: Question, Correct Answer, Incorrect Answer 1..3
    correct = row["Correct Answer"]
    incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    # Stable shuffle by question hash so position is deterministic per record
    h = int(hashlib.md5(row["Question"].encode()).hexdigest(), 16)
    rng = random.Random(h)
    options = incorrect + [correct]
    rng.shuffle(options)
    correct_idx = options.index(correct)
    options_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    return {
        "question": row["Question"] + "\n\n" + options_str,
        "answer": chr(65 + correct_idx),
        "metadata": {"subdomain": row.get("Subdomain"), "high_level_domain": row.get("High-level domain")},
    }


def _norm_aime25(row):
    # opencompass/AIME2025 schema: {"question": "...", "answer": "70"}
    q = row.get("question") or row.get("problem")
    a = row.get("answer")
    if not q or a is None:
        return None
    return {
        "question": q,
        "answer": str(a),
        "metadata": {},
    }


def _norm_bbh(row):
    # lukaemon/bbh: input, target. target is short answer; some tasks are MCQ "(A)" form.
    return {
        "question": row["input"],
        "answer": row["target"].strip("()") if row["target"].startswith("(") else row["target"],
        "metadata": {"task": row.get("task")},
    }


def _norm_quality(row):
    # emozilla/quality: article, question, options[A..D], answer (0-3)
    if not isinstance(row.get("answer"), int):
        return None
    article = row["article"][:50000]   # cap context for cost
    options_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row["options"]))
    return {
        "question": f"Article:\n{article}\n\nQuestion: {row['question']}\n\n{options_str}",
        "answer": chr(65 + row["answer"]),
        "metadata": {"length": len(row["article"])},
    }


def _norm_longbench_v2(row):
    # THUDM/LongBench-v2: context, question, choice_A..D, answer ("A".."D")
    options_str = "\n".join(f"{c}. {row['choice_' + c]}" for c in "ABCD")
    return {
        "question": f"Context:\n{row['context'][:50000]}\n\nQuestion: {row['question']}\n\n{options_str}",
        "answer": row["answer"].strip().upper()[:1],
        "metadata": {"domain": row.get("domain"), "length": row.get("length")},
    }


def _norm_bfcl(row):
    # gorilla-llm/Berkeley-Function-Calling-Leaderboard: question, function (gold), id, category
    # We keep the function-spec inside the question (so the worker knows the schema).
    fn_spec = row.get("function") or row.get("functions") or {}
    if not fn_spec:
        return None
    expected = row.get("ground_truth") or row.get("answer")
    if not expected:
        return None
    return {
        "question": (
            f"You have access to the following function:\n{json.dumps(fn_spec, indent=2)}\n\n"
            f"User request: {row['question']}\n\n"
            "Respond with a JSON object inside a ```json``` block: "
            '{"name": "<function>", "arguments": {<args>}}'
        ),
        "answer": json.dumps(expected) if not isinstance(expected, str) else expected,
        "metadata": {"category": row.get("category"), "id": row.get("id")},
    }


def _norm_mgsm(row):
    # juletxara/mgsm test: question, answer (string with #### N), answer_number (int)
    q = row.get("question")
    if not q:
        return None
    if "answer_number" in row and row["answer_number"] is not None:
        a = str(row["answer_number"])
    else:
        a = str(row.get("answer", "")).strip()
    if not a:
        return None
    return {
        "question": q,
        "answer": a,
        "metadata": {"language": row.get("language", "en")},
    }


def _norm_mtbench(row):
    # philschmid/mt-bench: {question_id, category, turns: list[str]}. Take 1st turn (single-turn).
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        q = turns[0]
    elif isinstance(turns, str):
        # Sometimes serialized as "['turn1', 'turn2']"
        try:
            import ast
            parsed = ast.literal_eval(turns)
            q = parsed[0] if isinstance(parsed, list) and parsed else ""
        except (SyntaxError, ValueError):
            q = ""
    else:
        q = row.get("question_1") or row.get("question", "")
    if not q:
        return None
    return {
        "question": q,
        "answer": "",   # judge-only; grader returns False, reward layer must call judge_fn
        "metadata": {"category": row.get("category"), "id": row.get("question_id")},
    }


# ---------------------------------------------------------------------------
# Dataset matrix
# ---------------------------------------------------------------------------

PHASE_1A_SPECS = [
    DatasetSpec("math500", "HuggingFaceH4/MATH-500", None, "test", "MIT", 300, 100, _norm_math500),
    DatasetSpec("gsm8k", "openai/gsm8k", "main", "train", "MIT", 200, 0, _norm_gsm8k),
    DatasetSpec("gsm8k", "openai/gsm8k", "main", "test", "MIT", 0, 100, _norm_gsm8k),
    DatasetSpec("mmlu", "cais/mmlu", "all", "auxiliary_train", "MIT", 200, 0, _norm_mmlu),
    DatasetSpec("mmlu", "cais/mmlu", "all", "test", "MIT", 0, 100, _norm_mmlu),
    DatasetSpec("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", "train", "CC-BY-SA", 200, 0, _norm_triviaqa),
    DatasetSpec("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", "validation", "CC-BY-SA", 0, 100, _norm_triviaqa),
    DatasetSpec("humaneval", "openai/openai_humaneval", None, "test", "MIT", 0, 164, _norm_humaneval),
    DatasetSpec("mbpp", "google-research-datasets/mbpp", "full", "train", "CC-BY-4.0", 200, 0, _norm_mbpp),
    DatasetSpec("mbpp", "google-research-datasets/mbpp", "full", "test", "CC-BY-4.0", 0, 100, _norm_mbpp),
    # NOTE: livecodebench/code_generation_lite uses a deprecated dataset script.
    # We use livecodebench/release_v6 (parquet-based) instead — same content, modern format.
    # If that's unavailable too, just skip LCB; HumanEval + MBPP cover the code axis.
    DatasetSpec("livecodebench", "livecodebench/release_v6", None, "test",
                "MIT", 100, 0, _norm_livecodebench, contamination_filter=_lcb_cutoff_filter),
    # BBH has 27 sub-task configs; we sample from a representative 5 (matches BBH-mini convention).
    # See _expand_bbh below — this DatasetSpec is a sentinel; build() handles BBH specially.
    DatasetSpec("bbh-sample", "lukaemon/bbh", "object_counting", "test", "Apache-2.0", 50, 0, _norm_bbh),
    DatasetSpec("bbh-sample", "lukaemon/bbh", "logical_deduction_three_objects", "test", "Apache-2.0", 50, 0, _norm_bbh),
    DatasetSpec("bbh-sample", "lukaemon/bbh", "navigate", "test", "Apache-2.0", 50, 0, _norm_bbh),
    DatasetSpec("bbh-sample", "lukaemon/bbh", "date_understanding", "test", "Apache-2.0", 25, 0, _norm_bbh),
    DatasetSpec("bbh-sample", "lukaemon/bbh", "reasoning_about_colored_objects", "test", "Apache-2.0", 25, 0, _norm_bbh),
    # GPQA-Diamond is gated (auto-acceptance); document as a manual step.
    # Run once with HF_TOKEN env var set + after accepting terms at https://huggingface.co/datasets/Idavidrein/gpqa
    DatasetSpec("gpqa-diamond", "Idavidrein/gpqa", "gpqa_diamond", "train", "CC-BY-4.0", 0, 100, _norm_gpqa_diamond),
    DatasetSpec("aime25", "opencompass/AIME2025", "AIME2025-I", "test", "MIT", 0, 15, _norm_aime25),
    DatasetSpec("aime25", "opencompass/AIME2025", "AIME2025-II", "test", "MIT", 0, 15, _norm_aime25),
    # Long-context — gated by license flag
    DatasetSpec("longbench-v2", "THUDM/LongBench-v2", None, "train", "Apache-2.0", 100, 50, _norm_longbench_v2),
    # QuALITY only included if --accept-quality-license is passed; substitutes LongBench-v2
    DatasetSpec("quality", "emozilla/quality", None, "validation", "non-commercial", 100, 50, _norm_quality),
]

PHASE_1B_SPECS = [
    DatasetSpec("bfcl", "gorilla-llm/Berkeley-Function-Calling-Leaderboard", None, "train",
                "Apache-2.0", 200, 80, _norm_bfcl, phase="1b"),
    # MGSM en/train only has 8 items (few-shot pool); use test split instead.
    # Multilingual diversity comes from sampling other language configs at eval time.
    DatasetSpec("mgsm", "juletxara/mgsm", "en", "test", "MIT", 100, 60, _norm_mgsm, phase="1b"),
    DatasetSpec("mmlu-pro", "TIGER-Lab/MMLU-Pro", None, "test", "MIT", 100, 100, _norm_mmlu_pro, phase="1b"),
    # MTBench: single NDJSON file at philschmid/mt-bench/question.jsonl. Custom loader path.
    DatasetSpec("mtbench", "philschmid/mt-bench", None, "train", "Apache-2.0", 0, 80, _norm_mtbench, phase="1b"),
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _fetch_bfcl_ndjson(rng: random.Random) -> list[dict]:
    """BFCL is shipped as individual NDJSON files. Pull a few simple-call subsets."""
    import urllib.request
    files = [
        "BFCL_v3_simple.json",
        "BFCL_v3_exec_simple.json",
        "BFCL_v3_live_simple.json",
    ]
    base = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
    rows = []
    for f in files:
        try:
            with urllib.request.urlopen(f"{base}/{f}", timeout=60) as resp:
                content = resp.read().decode("utf-8")
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            log.warning("  bfcl: failed %s: %s", f, e)
    return rows


def fetch_and_take(spec: DatasetSpec, n_train: int, n_eval: int,
                   rng: random.Random) -> tuple[list, list]:
    """Returns (train_records, eval_records). Empty lists if dry-run / unavailable."""
    from datasets import load_dataset
    log.info("loading %s (%s, %s) — split=%s", spec.name, spec.hf_path,
             spec.hf_config or "(no-config)", spec.split)

    # Special case: BFCL ships individual NDJSON files, not a HF dataset proper
    if spec.hf_path == "gorilla-llm/Berkeley-Function-Calling-Leaderboard":
        rows = _fetch_bfcl_ndjson(rng)
    elif spec.hf_path == "philschmid/mt-bench":
        # Single NDJSON file
        import urllib.request
        with urllib.request.urlopen(
            "https://huggingface.co/datasets/philschmid/mt-bench/resolve/main/question.jsonl",
            timeout=60,
        ) as resp:
            content = resp.read().decode("utf-8")
        rows = [json.loads(ln) for ln in content.splitlines() if ln.strip()]
    else:
        ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split, trust_remote_code=False)
        rows = list(ds)
    if spec.contamination_filter:
        before = len(rows)
        rows = [r for r in rows if spec.contamination_filter(r)]
        log.info("  contamination filter dropped %d/%d rows", before - len(rows), before)

    # Normalize, drop None
    normalized = []
    for i, r in enumerate(rows):
        try:
            n = spec.normalize(r)
        except Exception as e:
            log.warning("  normalize failure on row %d (%s): %s", i, spec.name, e)
            continue
        if n is None:
            continue
        rec = {
            "id": f"{spec.name}-{spec.split}-{i:05d}",
            "dataset": spec.name,
            "license": spec.license,
            "source": spec.hf_path,
            **n,
        }
        normalized.append(rec)

    if not normalized:
        log.warning("  no normalized rows for %s", spec.name)
        return [], []

    rng.shuffle(normalized)

    train = normalized[:n_train]
    # Don't reuse train rows for eval — take from after the train slice
    eval_ = normalized[n_train:n_train + n_eval]

    log.info("  yielded %d train + %d eval (from %d normalized)",
             len(train), len(eval_), len(normalized))
    return train, eval_


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        n = 0
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def build(out_dir: Path, seed: int, include_phase1b: bool,
          accept_quality_license: bool, dry_run: bool) -> dict:
    specs = list(PHASE_1A_SPECS)
    # License gating: drop QuALITY unless explicitly accepted; substitute LongBench-v2 by default.
    if not accept_quality_license:
        specs = [s for s in specs if s.name != "quality"]
        log.info("QuALITY license not accepted; using LongBench-v2 for long-context (default).")
    else:
        # If user accepted QuALITY, drop LongBench-v2 to avoid double-counting long-context items.
        specs = [s for s in specs if s.name != "longbench-v2"]
        log.info("QuALITY license accepted; using QuALITY (skipping LongBench-v2).")

    if include_phase1b:
        specs.extend(PHASE_1B_SPECS)

    summary = {"phase_1a": {}, "phase_1b": {}, "totals": {"train": 0, "eval": 0}}
    rng = random.Random(seed)

    train_all: list = []
    eval_by_dataset: dict[str, list] = {}

    failed_specs: list[str] = []
    for spec in specs:
        try:
            train, eval_ = ([], []) if dry_run else fetch_and_take(spec, spec.train_count, spec.eval_count, rng)
        except Exception as e:
            log.warning("[skip] %s/%s failed: %s", spec.hf_path, spec.split, e)
            failed_specs.append(f"{spec.name} ({spec.hf_path}/{spec.split}): {type(e).__name__}")
            train, eval_ = [], []

        bucket = "phase_1b" if spec.phase == "1b" else "phase_1a"
        summary[bucket].setdefault(spec.name, {"train": 0, "eval": 0})
        summary[bucket][spec.name]["train"] += len(train)
        summary[bucket][spec.name]["eval"] += len(eval_)
        summary["totals"]["train"] += len(train)
        summary["totals"]["eval"] += len(eval_)

        train_all.extend(train)
        eval_by_dataset.setdefault(spec.name, []).extend(eval_)
    summary["failed"] = failed_specs

    # Final shuffle of the combined train set so the trainer's `random.sample`
    # doesn't pull a contiguous block from one dataset.
    rng.shuffle(train_all)

    if not dry_run:
        n_train = write_jsonl(train_all, out_dir / "train.jsonl")
        log.info("wrote %d train items → %s", n_train, out_dir / "train.jsonl")
        for ds_name, recs in eval_by_dataset.items():
            n = write_jsonl(recs, out_dir / "eval" / f"{ds_name}.jsonl")
            log.info("  wrote %d eval items → %s", n, out_dir / "eval" / f"{ds_name}.jsonl")

        # License-summary file — for Phase 0 license gate audit
        license_summary = {}
        for spec in specs:
            license_summary.setdefault(spec.license, []).append(spec.name)
        (out_dir / "licenses.json").write_text(json.dumps(license_summary, indent=2))

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--include-phase1b", action="store_true")
    p.add_argument("--accept-quality-license", action="store_true",
                   help="Include QuALITY (non-commercial). Default substitutes LongBench-v2.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the dataset matrix without downloading anything.")
    args = p.parse_args()

    summary = build(
        out_dir=Path(args.out_dir), seed=args.seed,
        include_phase1b=args.include_phase1b,
        accept_quality_license=args.accept_quality_license,
        dry_run=args.dry_run,
    )

    print("\n=== Build summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
