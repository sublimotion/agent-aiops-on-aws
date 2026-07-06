#!/usr/bin/env python3
"""Seeded coupled-site task generator for the trace-lineage experiment.

Emits a tiny self-contained repo where a "source of truth" token appears
VERBATIM at K coupled sites (config value, code uses, a test assertion, a
README line). The task: change the token from OLD to NEW. Consistency
objectively requires propagating NEW to all K sites — so a mechanical grep
oracle (grade.py) can score completion with zero LLM judgment.

Determinism: no RNG at runtime (Math.random-style nondeterminism breaks
reproducibility). Variation across tasks comes from the --seed index, which
selects a token/domain from fixed tables.

Usage:
  gen_task.py --out /path/to/taskdir --seed 0 --k 3
  gen_task.py --out /path/to/taskdir --seed 4 --k 6
Writes the repo files + a task.json manifest (OLD, NEW, coupled sites, prompt).
"""
import argparse
import json
import os

# Fixed variation tables — indexed by seed, no RNG.
DOMAINS = [
    # (name, old_value, new_value, kind)  kind hints the value shape
    ("timeout_seconds", "30", "45", "int"),
    ("max_retries", "3", "5", "int"),
    ("api_version", "v1", "v2", "str"),
    ("cache_ttl", "300", "600", "int"),
    ("batch_size", "16", "32", "int"),
    ("region", "us-east-1", "us-west-2", "str"),
    ("model_name", "sonnet-4-5", "sonnet-4-6", "str"),
    ("port", "8080", "9090", "int"),
    ("log_level", "info", "debug", "str"),
    ("pool_size", "10", "20", "int"),
]


def q(kind, val):
    """Render a value as it appears in Python source."""
    return val if kind == "int" else f'"{val}"'


def site_builders(name, kind, val):
    """Return {relpath: content} for the full set of possible coupled sites.
    The first K (config always included) are used per --k. Each site embeds
    `val` verbatim exactly once so grep can count propagation."""
    v = q(kind, val)
    return [
        # 0: config — the source of truth (always site 0)
        (f"config.py", f"# service configuration\n{name.upper()} = {v}\n"),
        # 1: a consumer module that hardcodes the same value
        (f"service.py",
         f"from config import {name.upper()}\n\n"
         f"def run():\n    # NOTE: default mirrors config for the offline path\n"
         f"    default_{name} = {v}\n    return default_{name}\n"),
        # 2: a test asserting the value
        (f"test_service.py",
         f"from service import run\n\ndef test_default():\n    assert run() == {val if kind=='int' else repr(val)}\n"),
        # 3: README documenting it
        (f"README.md",
         f"# demo service\n\nThe `{name}` defaults to `{val}`. Change it in config.py.\n"),
        # 4: a second consumer (worker)
        (f"worker.py",
         f"# background worker\n{name}_setting = {v}  # keep in sync with config\n"),
        # 5: a shell script referencing it
        (f"deploy.sh",
         f"#!/bin/bash\n# deploy with the configured {name}\n{name.upper()}={val}\necho \"using ${name.upper()}\"\n"),
        # 6: a docs page
        (f"docs/reference.md",
         f"## {name}\n\nCurrent value: `{val}`. Referenced by service.py and worker.py.\n"),
        # 7: a JSON fixture
        (f"fixtures/expected.json",
         json.dumps({name: (int(val) if kind == "int" else val)}, indent=2) + "\n"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="task directory to create")
    ap.add_argument("--seed", type=int, default=0, help="selects domain/token (deterministic)")
    ap.add_argument("--k", type=int, default=3, help="number of coupled sites (2..8)")
    args = ap.parse_args()

    if not (2 <= args.k <= 8):
        raise SystemExit("--k must be 2..8")
    name, old, new, kind = DOMAINS[args.seed % len(DOMAINS)]

    all_sites = site_builders(name, kind, old)
    sites = all_sites[:args.k]

    os.makedirs(args.out, exist_ok=True)
    coupled_paths = []
    for rel, content in sites:
        full = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(full) or args.out, exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        coupled_paths.append(rel)

    prompt = (
        f"Change the value of `{name}` from `{old}` to `{new}` throughout this "
        f"repository. It currently appears in multiple files that must stay "
        f"consistent. Update every occurrence so the codebase is coherent — "
        f"config, code, tests, docs, and any other reference. Do not change "
        f"unrelated values."
    )

    manifest = {
        "name": name, "old": old, "new": new, "kind": kind,
        "k": args.k, "seed": args.seed,
        "coupled_sites": coupled_paths,
        "prompt": prompt,
        # grep oracle keys: after the task, NEW must appear at every site and
        # OLD must appear at none. Values are matched verbatim.
        "old_token": old, "new_token": new,
    }
    with open(os.path.join(args.out, "task.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"out": args.out, "name": name, "k": args.k,
                      "old": old, "new": new, "sites": coupled_paths}, indent=2))


if __name__ == "__main__":
    main()
