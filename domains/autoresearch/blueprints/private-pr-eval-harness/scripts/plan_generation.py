#!/usr/bin/env python3
"""Phase 2 exit — plan the generation cells (does NOT spend; validates topology).

The fan-out reality (discovered via `fe agent launch --dry-run`): the runtime
launches ONE agent against ONE commit — it is built to run THIS repo's specs, not
to fan out over N sealed target-repo workspaces at different base commits. So the
private-PR eval's generation stage cannot be "one fe agent launch per task" against
this repo. Two correct options:

  (A) IN-IMAGE LOOP (recommended): one agent-runtime Job per {harness} cell whose
      container entrypoint iterates the sealed task set — for each task it seals a
      workspace at base_commit (seal_workspace.py), runs the harness headless, and
      writes predictions/<cell>/<pr>.diff to S3. 2 harnesses = 2 Jobs, not 2xN.
      Matches the runtime's "long detached Job + S3 artifacts" model exactly.

  (B) PER-TASK LAUNCH: NxM `fe agent launch --commit <base>` calls. Rejected —
      the launcher checks out THIS repo's commit, not the target repo's base; would
      need a per-task synthetic spec + repo override the runtime doesn't expose.

This script emits the PLAN for option (A): the 2 cells and the per-cell task list,
plus the harness command each cell would run per task. It calls `fe agent launch
--dry-run` once per cell to prove the Job renders. No pods are created.

Pilot cells (2): claude-code (Bedrock) and opencode (self-hosted vLLM) — the two
best single harnesses from prior blueprints (VP-SWE-bench 58% / agent-harness 22%).

Usage:
  python3 plan_generation.py --task-file results/tasks.jsonl --n 10 \
      --cluster qn-sglang-eks-cluster --out results/generation-plan.json
"""
import argparse, json, subprocess, pathlib, sys

CELLS = [
    {"cell": "claude-code", "harness": "claude-code",
     "model_route": "bedrock:us.anthropic.claude-opus-4-8",
     "note": "closed-model cell; run role already scoped for Bedrock"},
    {"cell": "opencode", "harness": "opencode",
     "model_route": "vllm:glm5.2-endpoint",
     "note": "open-model cell; harness points at self-hosted SGLang/vLLM"},
]

HARNESS_CMD = {
    # per-task, inside the sealed workspace; headless, tools enabled
    "claude-code": ("claude -p \"{prompt}\" --output-format json --max-turns 30 "
                    "--allowedTools Bash,Read,Write,Edit,Glob,Grep"),
    "opencode": "opencode run --model vllm/glm5.2 --prompt-file task.txt (headless)",
}


def dry_run_cell(spec, cluster, harness):
    r = subprocess.run(
        ["fe", "agent", "launch", spec, "--cluster", cluster,
         "--harness", harness, "--dry-run"],
        capture_output=True, text=True)
    # `fe agent launch --dry-run` prints the banner to stderr, the Job YAML to stdout
    combined = r.stdout + r.stderr
    ok = "DRY RUN" in combined and "kind: Job" in r.stdout
    rid = ""
    for line in combined.splitlines():
        if "run-id:" in line:
            rid = line.split("run-id:")[1].strip()
            break
    return ok, rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="results/tasks.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cluster", default="qn-sglang-eks-cluster")
    ap.add_argument("--spec",
                    default="domains/autoresearch/specs/private-pr-eval-harness.md")
    ap.add_argument("--out", default="results/generation-plan.json")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.task_file)][:args.n]
    plan = {"topology": "A: in-image loop, one Job per harness cell",
            "cluster": args.cluster, "n_tasks": len(tasks), "cells": []}

    for c in CELLS:
        ok, rid = dry_run_cell(args.spec, args.cluster, c["harness"])
        plan["cells"].append({
            **c,
            "dry_run_job_renders": ok,
            "example_run_id": rid,
            "per_task_cmd": HARNESS_CMD.get(c["harness"], "?"),
            "tasks": [{"pr": t["pr_number"], "base_commit": t["base_commit"],
                       "tier": t.get("complexity_tier")} for t in tasks],
        })
        print(f"[plan] cell={c['cell']:<12} harness={c['harness']:<12} "
              f"job_renders={ok} tasks={len(tasks)}")

    pathlib.Path(args.out).write_text(json.dumps(plan, indent=2))
    all_ok = all(c["dry_run_job_renders"] for c in plan["cells"])
    print(f"[plan] wrote {args.out}")
    print(f"[plan] all cells render a valid Job: {all_ok} "
          f"(NO pods created — this is a dry-run plan)")
    print("[plan] NEXT (separate budgeted launch, NOT the pilot): build the cell "
          "container entrypoint that loops the task set, then launch 2 Jobs.")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
