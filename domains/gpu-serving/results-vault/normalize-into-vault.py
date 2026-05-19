#!/usr/bin/env python3
"""normalize-into-vault.py — Consolidate blueprint artifacts into the vault.

Contract: one artifact = one operating point.

- Conformant artifacts (no rolled-up sweep) are symlinked from the blueprint.
- Rolled-up artifacts (extensions.sweep_levels / extensions.context_sweep) are
  materialized into per-point files written under the vault. Each materialized
  file records the source path under extensions.normalized_from.

Run after a benchmark session, then rebuild-index.py.

  python3 domains/gpu-serving/results-vault/normalize-into-vault.py
  python3 domains/gpu-serving/results-vault/rebuild-index.py
"""
from __future__ import annotations
import json
import os
import re
import shutil
import uuid
from pathlib import Path

VAULT = Path(__file__).resolve().parent
BLUEPRINTS = VAULT.parent / "blueprints"


def _find_pair_length_sweep(ext: dict) -> list | None:
    rr = ext.get("reranker") or {}
    pls = rr.get("pair_length_sweep") if isinstance(rr, dict) else None
    return pls if isinstance(pls, list) and pls else None


def _is_single_point(d: dict) -> bool:
    ext = d.get("extensions") or {}
    if any(k in ext for k in ("sweep_levels", "context_sweep")):
        return False
    if _find_pair_length_sweep(ext):
        return False
    return True


def _strip_sweep_filename(name: str, suffix: str) -> str:
    return re.sub(r"_\d{8}T\d{6}Z\.json$", suffix, name)


def _materialize_one(base: dict, level: dict, source_rel: str, axis: dict | None = None) -> dict:
    art = json.loads(json.dumps(base))
    art["artifact_id"] = str(uuid.uuid4())
    m = dict(art.get("metrics") or {})
    for k in ("duration_s", "completed", "failed", "output_toks_per_s",
              "request_throughput", "total_input_tokens", "total_output_tokens"):
        if k in level:
            m[k] = level[k]
    if "e2e_ms" in level:
        m["e2e_ms"] = level["e2e_ms"]
    if "ttft_ms" in level:
        m["ttft_ms"] = level["ttft_ms"]
    if "tpot_ms" in level:
        m["tpot_ms"] = level["tpot_ms"]
    if "itl_ms" in level:
        m["itl_ms"] = level["itl_ms"]
    completed = level.get("completed", m.get("completed", 0))
    failed = level.get("failed", m.get("failed", 0))
    total = (completed or 0) + (failed or 0)
    m["error_rate"] = (failed / total) if total else 0.0
    c = level.get("concurrency")
    if c is not None:
        m["max_concurrent_requests"] = c
    art["metrics"] = m

    load = dict((art.get("workload") or {}).get("load") or {})
    for k in ("levels", "current_level", "level"):
        load.pop(k, None)
    load["type"] = "concurrency"
    if c is not None:
        load["concurrency"] = c
    art.setdefault("workload", {})["load"] = load
    if axis:
        ds = dict((art.get("workload") or {}).get("dataset") or {})
        ds.update(axis)
        art["workload"]["dataset"] = ds

    ext = dict(art.get("extensions") or {})
    ext.pop("sweep_levels", None)
    ext.pop("context_sweep", None)
    ext["normalized_from"] = source_rel
    art["extensions"] = ext
    return art


def _is_stale_materialized(p: Path, source: Path) -> bool:
    try:
        d = json.load(open(p))
    except Exception:
        return False
    nf = (d.get("extensions") or {}).get("normalized_from")
    return nf is not None and source.name in nf


def normalize(source: Path, blueprint_rel: str) -> list[Path]:
    """Return list of vault paths produced (symlink or written file)."""
    try:
        d = json.load(open(source))
    except Exception as e:
        print(f"  skip (parse error): {source.name}: {e}")
        return []

    if _is_single_point(d):
        target = VAULT / source.name
        if target.is_symlink() or target.exists():
            target.unlink()
        rel = os.path.relpath(source, VAULT)
        target.symlink_to(rel)
        return [target]

    ext = d.get("extensions") or {}
    source_rel = os.path.relpath(source, VAULT.parent)
    written: list[Path] = []
    base = json.loads(json.dumps(d))
    base["extensions"] = {k: v for k, v in (base.get("extensions") or {}).items()
                         if k not in ("sweep_levels", "context_sweep")}

    if "sweep_levels" in ext:
        for lvl in ext["sweep_levels"]:
            c = lvl.get("concurrency")
            if c is None:
                continue
            art = _materialize_one(base, lvl, source_rel)
            stem = _strip_sweep_filename(source.name, f"_c{c}.json")
            out = VAULT / stem
            json.dump(art, open(out, "w"), indent=2)
            written.append(out)

    pls = _find_pair_length_sweep(ext)
    if pls:
        # strip the nested sweep from base.reranker
        rr = dict((base.get("extensions") or {}).get("reranker") or {})
        rr.pop("pair_length_sweep", None)
        base.setdefault("extensions", {})["reranker"] = rr
        for lvl in pls:
            pl = lvl.get("pair_length")
            if pl is None:
                continue
            art = _materialize_one(base, lvl, source_rel,
                                   axis={"pair_length": pl})
            stem = _strip_sweep_filename(source.name, f"_pl{pl}.json")
            out = VAULT / stem
            json.dump(art, open(out, "w"), indent=2)
            written.append(out)

    if "context_sweep" in ext:
        for ctx in ext["context_sweep"]:
            toks = ctx.get("approx_tokens") or ctx.get("context_tokens")
            for lvl in ctx.get("levels", []):
                c = lvl.get("concurrency")
                if c is None or toks is None:
                    continue
                art = _materialize_one(base, lvl, source_rel,
                                       axis={"context_tokens": toks})
                stem = _strip_sweep_filename(source.name, f"_ctx{toks}_c{c}.json")
                out = VAULT / stem
                json.dump(art, open(out, "w"), indent=2)
                written.append(out)

    return written


def main():
    sources: list[tuple[Path, str]] = []
    for bp in sorted(BLUEPRINTS.iterdir()):
        if not bp.is_dir():
            continue
        for sub in ("results/standard", "results/artifacts"):
            d = bp / sub
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                sources.append((p, bp.name))

    # Clean stale materialized files before regenerating.
    materialized_to_keep: set[str] = set()
    rolled_up_source_names: set[str] = set()

    by_kind = {"symlinked": 0, "materialized": 0, "skipped": 0, "blueprints": set()}
    for src, bp_name in sources:
        try:
            d = json.load(open(src))
        except Exception:
            by_kind["skipped"] += 1
            continue
        if not _is_single_point(d):
            rolled_up_source_names.add(src.name)
        produced = normalize(src, bp_name)
        if not produced:
            by_kind["skipped"] += 1
            continue
        by_kind["blueprints"].add(bp_name)
        for p in produced:
            materialized_to_keep.add(p.name)
            if p.is_symlink():
                by_kind["symlinked"] += 1
            else:
                by_kind["materialized"] += 1

    # Drop symlinks pointing at rolled-up sources — they're superseded by the
    # materialized per-point files.
    for p in VAULT.glob("*.json"):
        if p.is_symlink() and p.name in rolled_up_source_names:
            p.unlink()
            print(f"  pruned rolled-up symlink: {p.name}")

    # Drop orphaned materialized files (rolled-up source removed) — but keep
    # any vault-local files that aren't from a known source.
    for p in VAULT.glob("*.json"):
        if p.name in {"index.json"}:
            continue
        if p.name in materialized_to_keep:
            continue
        if p.is_symlink():
            tgt = (VAULT / os.readlink(p))
            if not tgt.exists():
                p.unlink()
                print(f"  pruned dangling symlink: {p.name}")
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if (d.get("extensions") or {}).get("normalized_from"):
            p.unlink()
            print(f"  pruned stale materialized: {p.name}")

    print(f"Normalized: {by_kind['symlinked']} symlinked, "
          f"{by_kind['materialized']} materialized, "
          f"{by_kind['skipped']} skipped, "
          f"across {len(by_kind['blueprints'])} blueprints.")
    print("Next: python3 domains/gpu-serving/results-vault/rebuild-index.py")


if __name__ == "__main__":
    main()
