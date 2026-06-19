"""Cross-spec stage-attribution comparison.

Aggregates profiler artifacts grouped by an arbitrary key (variant tag,
fixture blueprint, replica index, etc.) and prints a stacked-bar
visualization plus a CSV.

Usage:
    python stage_compare.py results/**/*.json --group-by variant.snapshotter
    python stage_compare.py results/**/*.json --group-by experiment --csv out.csv

When run without --csv, prints an ASCII stacked bar to stdout. Useful for
quick eyeball comparisons before doing serious plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

STAGE_ORDER = [
    "node_provision",
    "image_pull",
    "container_start",
    "model_load",
    "jit_compile",
    "first_token_warmup",
]


def lookup(d: dict, dotted: str) -> Any:
    """variant.snapshotter -> d['variant']['snapshotter']."""
    cur = d
    for k in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def aggregate(paths: list[Path], group_by: str) -> dict[str, dict[str, list[float]]]:
    """{group_value: {stage: [seconds, ...]}}."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        art = json.loads(path.read_text())
        key = str(lookup(art, group_by) or "_unset")
        for stage, body in art.get("stages", {}).items():
            v = body.get("elapsed_s")
            if v is not None and v >= 0:
                out[key][stage].append(v)
        gap_total = 0.0
        for v in art.get("gaps", {}).values():
            if v >= 0:
                gap_total += v
        out[key]["_gaps"].append(gap_total)
    return out


def medians(agg: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    return {
        k: {s: statistics.median(v) if v else 0.0 for s, v in stages.items()}
        for k, stages in agg.items()
    }


def render_ascii(meds: dict[str, dict[str, float]], width: int = 60) -> str:
    """Render a stacked bar per group."""
    if not meds:
        return "(no data)"
    totals = {k: sum(stages.get(s, 0.0) for s in STAGE_ORDER) + stages.get("_gaps", 0.0)
              for k, stages in meds.items()}
    max_total = max(totals.values()) if totals else 1.0
    glyphs = {
        "node_provision": "N", "image_pull": "I", "container_start": "C",
        "model_load": "M", "jit_compile": "J", "first_token_warmup": "F",
        "_gaps": ".",
    }

    label_width = max(len(k) for k in meds) + 2
    lines = []
    lines.append(f"Legend: {'  '.join(f'{g}={s}' for s, g in glyphs.items())}")
    lines.append("")
    for key in sorted(meds, key=lambda k: totals[k]):
        stages = meds[key]
        total = totals[key]
        bar = ""
        for stage in STAGE_ORDER + ["_gaps"]:
            seconds = stages.get(stage, 0.0)
            chars = int(round(seconds / max_total * width))
            bar += glyphs[stage] * chars
        lines.append(f"{key:<{label_width}} |{bar:<{width}}| {total:7.1f}s")
    return "\n".join(lines)


def write_csv(meds: dict[str, dict[str, float]], path: Path) -> None:
    fieldnames = ["group"] + STAGE_ORDER + ["_gaps", "total"]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, stages in meds.items():
            row = {"group": key}
            total = 0.0
            for s in STAGE_ORDER:
                v = stages.get(s, 0.0)
                row[s] = f"{v:.2f}"
                total += v
            row["_gaps"] = f"{stages.get('_gaps', 0.0):.2f}"
            row["total"] = f"{total + stages.get('_gaps', 0.0):.2f}"
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", nargs="+")
    p.add_argument("--group-by", default="experiment",
                   help="dotted path into artifact, e.g. variant.snapshotter")
    p.add_argument("--csv", help="write CSV summary to this path")
    args = p.parse_args()

    paths = [Path(a) for a in args.artifacts]
    agg = aggregate(paths, args.group_by)
    meds = medians(agg)

    if args.csv:
        write_csv(meds, Path(args.csv))
        print(f"wrote {args.csv}", file=sys.stderr)
    else:
        print(render_ascii(meds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
