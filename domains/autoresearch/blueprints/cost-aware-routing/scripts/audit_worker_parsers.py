"""Gate 0.2b — Per-worker output-format audit.

For each (worker, dataset) cell, run N=20 questions through the worker,
attempt extraction, and report extract_rate + judge_agreement. Fail any cell
< 90% extract or < 95% judge_agreement.

Usage:
    python -m scripts.audit_worker_parsers \\
        --pool configs/pool.yaml \\
        --eval-data data/eval_subsets.json \\
        --out results/parser_audit.json \\
        --n 20

Eval data format (data/eval_subsets.json):
    { "math500": [{"question": ..., "gold": ...}, ...],
      "mmlu":    [{"question": ..., "gold": "A"}, ...],
      "humaneval": [{"question": ..., "gold": "...", "entry_point": "func"}, ...] }

Outputs JSON like:
    { "ord_0": { "math500": {"extract_rate": 0.95, "judge_agreement": 0.92,
                             "fallback_methods": {...}, "samples": [...] },
                  "mmlu":    { ... } },
      ... }

The 'samples' key contains 3 raw responses per cell so we can diff actual
output formats across the new workers (Kimi K2 Thinking, Opus extended,
GLM 5, MiniMax M2.5, etc.).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .worker_proxy import WorkerPool
from .extractors import extract
from .graders import grade

DATASETS = ("math500", "mmlu", "humaneval")
PROMPT_TEMPLATES = {
    "math500": "Solve the following problem. Write your reasoning, then the final numeric or symbolic answer in \\boxed{{}}.\n\nProblem: {question}",
    "mmlu":    "Choose the correct option (A, B, C, or D). Reply with the letter only on the final line.\n\n{question}",
    "humaneval": "Implement the function below. Return Python code in a ```python code block. Do not call the function.\n\n{question}",
}


async def audit_one_cell(pool: WorkerPool, ord_: int, dataset: str, samples: list[dict], n: int):
    """Audit a single (worker, dataset) cell."""
    extracted_ok = 0
    judge_ok = 0
    fallback_methods: dict[str, int] = {}
    raw_samples = []

    template = PROMPT_TEMPLATES[dataset]
    for i, item in enumerate(samples[:n]):
        prompt = template.format(question=item["question"])
        result = await pool.call(ord_, prompt, temperature=0.2)
        if result.error:
            raw_samples.append({"q_idx": i, "error": result.error})
            continue
        ans, method = extract(result.text, dataset)
        fallback_methods[method] = fallback_methods.get(method, 0) + 1
        if ans:
            extracted_ok += 1
            kw = {"entry_point": item["entry_point"]} if "entry_point" in item else {}
            if grade(ans, item["gold"], dataset, **kw):
                judge_ok += 1
        if i < 3:
            raw_samples.append({
                "q_idx": i,
                "raw_response": result.text[:1500],
                "extracted": ans,
                "method": method,
                "input_tok": result.input_tok,
                "output_tok": result.output_tok,
                "latency_ms": round(result.latency_ms, 0),
            })

    return {
        "extract_rate": extracted_ok / n,
        "judge_agreement": judge_ok / n,
        "fallback_methods": fallback_methods,
        "samples": raw_samples,
    }


async def main_async(args):
    pool = WorkerPool(args.pool, seed=None)
    eval_data = json.loads(Path(args.eval_data).read_text())

    report = {}
    for ord_ in sorted(pool.workers):
        worker_name = pool.workers[ord_].name
        report[f"ord_{ord_}_{worker_name}"] = {}
        for ds in DATASETS:
            samples = eval_data.get(ds, [])
            if len(samples) < args.n:
                print(f"[skip] {worker_name} × {ds}: only {len(samples)} samples available", file=sys.stderr)
                continue
            print(f"[run] ord_{ord_} {worker_name} × {ds} (n={args.n})", file=sys.stderr)
            cell = await audit_one_cell(pool, ord_, ds, samples, args.n)
            report[f"ord_{ord_}_{worker_name}"][ds] = cell
            status = "PASS" if cell["extract_rate"] >= 0.90 and cell["judge_agreement"] >= 0.95 else "FAIL"
            print(f"    extract={cell['extract_rate']:.2f} judge={cell['judge_agreement']:.2f}  {status}",
                  file=sys.stderr)

    # Summary table
    print("\n=== Summary ===", file=sys.stderr)
    print(f"{'worker':<24} | {'math500':<14} | {'mmlu':<14} | {'humaneval':<14}", file=sys.stderr)
    for k, cells in report.items():
        row = [k]
        for ds in DATASETS:
            c = cells.get(ds, {})
            if c:
                row.append(f"e={c['extract_rate']:.2f} j={c['judge_agreement']:.2f}")
            else:
                row.append("(skipped)")
        print(f"{row[0]:<24} | {row[1]:<14} | {row[2]:<14} | {row[3]:<14}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.out}", file=sys.stderr)

    # Exit non-zero if any cell fails (Phase 0 gate semantics)
    failed = [
        f"{k} × {ds}"
        for k, cells in report.items()
        for ds, c in cells.items()
        if c["extract_rate"] < 0.90 or c["judge_agreement"] < 0.95
    ]
    if failed:
        print(f"\n[FAIL] {len(failed)} cells below threshold:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("[PASS] all cells ≥0.90 extract / ≥0.95 judge_agreement", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="configs/pool.yaml")
    p.add_argument("--eval-data", required=True)
    p.add_argument("--out", default="results/parser_audit.json")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
