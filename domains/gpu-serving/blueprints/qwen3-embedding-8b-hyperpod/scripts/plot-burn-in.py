#!/usr/bin/env python3
"""
Render a simple ASCII drift curve from burn-in results.
Works on both burn-in-progress.json (partial) and burn-in-final.json.
No matplotlib dependency; outputs a plain-text sparkline + table.
"""
import argparse, json, sys
from pathlib import Path


def bar(val, lo, hi, width=30):
    if hi == lo:
        return "|" + "=" * (width // 2)
    frac = (val - lo) / (hi - lo)
    filled = int(frac * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="results/burn-in/burn-in-progress.json",
                        help="Path to burn-in JSON")
    parser.add_argument("--output",
                        default="results/burn-in/drift-plot.txt")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr)
        sys.exit(1)

    d = json.load(open(p))
    slices = d.get("slices", [])
    if not slices:
        print("No slices yet", file=sys.stderr)
        sys.exit(1)

    throughputs = [s["output_throughput"] for s in slices]
    lo, hi = min(throughputs), max(throughputs)
    span = hi - lo if hi > lo else 1.0

    # Find the drift baseline (post-warmup average)
    warmup = (d.get("warmup_s", 600)) // (d.get("slice_duration_s", 300))
    post = slices[warmup:]
    baseline_tp = (sum(s["output_throughput"] for s in post[:4]) /
                    min(4, len(post))) if post else throughputs[0]

    out = [f"=== Burn-in drift curve ===",
           f"Slices: {len(slices)}/{d.get('duration_s', 3600) // d.get('slice_duration_s', 300)}",
           f"Concurrency: {d.get('concurrency')}",
           f"Post-warmup baseline: {baseline_tp:.2f} req/s",
           f"",
           f"{'#':>3} {'tp (req/s)':>10} {'p50':>5} {'p99':>5} {'err':>4} {'Δ vs base':>10}  visual"]

    for s in slices:
        tp = s["output_throughput"]
        delta_pct = ((tp - baseline_tp) / baseline_tp * 100) if baseline_tp else 0
        delta_str = f"{delta_pct:+.2f}%"
        phase = "warm" if s["slice_idx"] < warmup else "    "
        visual = bar(tp, lo, hi)
        out.append(f"{s['slice_idx']+1:>3} {tp:>10.1f} "
                   f"{s['latency_p50']:>5.0f} {s['latency_p99']:>5.0f} "
                   f"{s['failed']:>4} {delta_str:>10}  {visual} {phase}")

    # Stability verdict if final
    if "stability" in d:
        s = d["stability"]
        out.append("")
        out.append(f"=== Final stability ===")
        out.append(f"hour_1_throughput: {s['hour_1_throughput']:.2f} req/s")
        out.append(f"final_throughput:  {s['final_throughput']:.2f} req/s")
        out.append(f"drift_pct:         {s['throughput_drift_pct']:+.2f}%")
        out.append(f"unrecoverable:     {s['unrecoverable_errors']}")
        gate = "PASS ✅" if s["drift_gate_passed"] else "FAIL ❌"
        out.append(f"gate:              {gate}")

    text = "\n".join(out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
