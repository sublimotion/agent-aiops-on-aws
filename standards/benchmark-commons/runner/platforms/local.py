#!/usr/bin/env python3
"""Local platform — executes a COMPILED plan against a localhost/bare-metal
endpoint. Used for bare metal, spot instances, or localhost testing.

This platform decides WHERE the benchmark runs. It does NOT decide WHAT runs —
that is `compiler.compile_card`, the single deterministic source of truth. The
old version built argv inline and silently dropped unmapped dataset types; that
logic now lives in registry.py + compiler.py and fails closed.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path

# runner/ is the parent of platforms/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compiler import compile_card  # noqa: E402
from registry import UnsupportedWorkload  # noqa: E402


# vendor benchmark entrypoints
TOOL_ENTRYPOINT = {
    "vllm": ["python3", "-m", "vllm.entrypoints.openai.bench_serving"],
    "sglang": ["python3", "-m", "sglang.bench_serving", "--backend", "sglang"],
}


def build_step_command(tool: str, endpoint: str, model: str, step_argv: list[str]) -> list[str]:
    """Prepend the vendor entrypoint + connection flags to a compiled step's argv."""
    base = list(TOOL_ENTRYPOINT[tool])
    base += ["--base-url", endpoint]
    if tool == "vllm":
        base += ["--model", model, "--save-result"]
    elif tool == "sglang":
        base += ["--model", model]
    return base + step_argv


def main():
    parser = argparse.ArgumentParser(description="Local platform — execute a compiled benchmark plan")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workload", required=True, type=Path, help="Workload card YAML")
    parser.add_argument("--sidecar", type=Path, help="benchmark.yaml sidecar (for overrides)")
    parser.add_argument("--tool", required=True, choices=list(TOOL_ENTRYPOINT))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    card = yaml.safe_load(open(args.workload))
    sidecar = yaml.safe_load(open(args.sidecar)) if args.sidecar else {}
    model = sidecar.get("model", {}).get("id", "default")

    # Compile — fail closed and loud on anything unmapped.
    try:
        plan = compile_card(card, sidecar, args.tool)
    except UnsupportedWorkload as e:
        print(f"ERROR: workload did not compile: {e}", file=sys.stderr)
        print("This is a card/sidecar problem. Fix the declaration or add a "
              "registry handler — do not hand-write a one-off driver.", file=sys.stderr)
        sys.exit(3)

    if plan.kind == "orchestrated":
        msg = (f"Card '{plan.catalog_id}' requires the '{plan.orchestrator}' "
               f"orchestrated executor.\nReason: {plan.reason}")
        if args.dry_run:
            print(f"[DRY RUN] {msg}")
            return
        # Orchestrated executors live in orchestrators.py; dispatch when present.
        try:
            import orchestrators  # noqa: E402
            runner = getattr(orchestrators, plan.orchestrator, None)
        except ImportError:
            runner = None
        if runner is None:
            print(f"ERROR: {msg}\n\nThis executor is registered but not yet "
                  f"implemented. It is NOT a vendor-tool workload — running it "
                  f"requires the bespoke '{plan.orchestrator}'.", file=sys.stderr)
            sys.exit(4)
        runner(plan, args.endpoint, model, args.output, sidecar)
        return

    # Vendor plan: one vendor command per step.
    print(f"Compiled '{plan.catalog_id}' -> {len(plan.steps)} vendor step(s) [{args.tool}]")
    for i, step in enumerate(plan.steps):
        cmd = build_step_command(args.tool, args.endpoint, model, step.argv)
        if args.dry_run:
            print(f"  [step {i+1}/{len(plan.steps)}] {step.label}:")
            print(f"    {' '.join(cmd)}")
            continue
        print(f"  [step {i+1}/{len(plan.steps)}] {step.label}: executing...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  step {step.label} failed (exit {result.returncode}):", file=sys.stderr)
            print(result.stderr[-2000:] if result.stderr else "no stderr", file=sys.stderr)
            sys.exit(1)
        # Per-step raw output: <output-stem>__<label>.json
        step_out = args.output.with_name(f"{args.output.stem}__{step.label}.json")
        import glob
        produced = sorted(glob.glob("*.json"), key=lambda f: Path(f).stat().st_mtime, reverse=True)
        if produced:
            Path(produced[0]).rename(step_out)
            print(f"    raw -> {step_out}")

    if args.dry_run:
        print("Dry run complete. No benchmark executed.")


if __name__ == "__main__":
    main()
