#!/usr/bin/env python3
"""corpus.py — harvest structured knowledge from blueprint lessons.md field notes.

The gap between a spec ("deploy with TP8") and a blueprint ("TP8 failed, TP4
works") is the hard-won knowledge. That delta is recorded in each blueprint's
`lessons.md` YAML frontmatter (the Field Note Schema, see docs/card-format.md):
`outcome`, `failure_categories`, model/engine/hardware, and the `*_learn_commands`
that feed the card library.

This module reads those field notes and turns them into a queryable corpus so the
resolver can warn: "a prior deployment of this model on this engine recorded
failure_category=fp8_block_size_mismatch — see blueprint X." It is the I/O layer
(like a benchmark platform); the compiler stays pure and operates over the
already-loaded LessonsCorpus, never touching the filesystem itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# THE failure_category vocabulary. This dict is the single source of truth:
#   - its KEYS are the closed set of categories a lessons.md may declare
#     (docs/card-format.md documents the same set for humans; the conformance
#     test asserts the two stay in sync),
#   - its VALUE is the registry rule id the category corroborates, or None when
#     no deterministic check guards it yet (None = a candidate for a future rule).
# A recorded category with a non-None rule lets the resolver say "a codified
# check exists for this — confirm it fired"; None categories surface as a bare
# warning. Keep this in sync with what blueprints actually emit: a category
# emitted but absent here fails the conformance test (and is silently un-mapped).
CATEGORY_TO_RULE = {
    # --- codified: a deterministic registry rule catches this from config ---
    "fp8_block_size_mismatch": "fp8-moe-tp-divisibility",
    "max_position_embeddings_mismatch": "max-model-len-vs-position",
    "ami": "b200-requires-al2023",
    "kv_eviction": "hicache-size-vs-device-pool",
    # --- not yet codified (no config-checkable rule): surfaced as warnings ---
    # hardware / platform
    "nccl": None,
    "driver": None,
    "efa": None,
    "oom": None,
    "disk_pressure": None,
    "kubeconfig_context_switch": None,
    # container / image / deps
    "container": None,
    "image_compatibility": None,
    "dependency_conflict": None,
    "missing_shared_lib": None,
    "tls_incompatibility": None,
    "huggingface_cli_deprecation": None,
    # serving / model behavior
    "tool_call_parser_incompatibility": None,
    "tool_timeout": None,
    # catch-all
    "other": None,
}


@dataclass(frozen=True)
class FieldNote:
    model: str
    engine: str
    hardware: str
    gpu_arch: Optional[str]
    outcome: str
    failure_categories: tuple[str, ...]
    date: Optional[str]
    source: str                     # path to the lessons.md

    def matches(self, model_id: str, engine: str, instance_type: str) -> bool:
        """Loose match: same engine and the model/hardware token overlaps.

        Model ids vary (org/Model-FP8 vs short slug), so compare on a normalized
        token rather than exact equality. Hardware matches on instance-type prefix
        (p6-b300 == p6-b300.48xlarge).
        """
        if engine and self.engine and engine.lower() != self.engine.lower():
            return False
        a, b = _model_token(model_id), _model_token(self.model)
        if not a or not b:
            return False
        return a in b or b in a


def _model_token(name: str) -> str:
    """Reduce a model name/id to a comparable token (lowercase, no org/quant)."""
    if not name:
        return ""
    base = name.split("/")[-1].lower()
    # strip common quant/precision suffixes so Qwen3-235B-A22B-FP8 ~ qwen3-235b
    base = re.sub(r"[-_](fp8|fp4|int4|int8|bf16|awq|gptq|a\d+b)\b", "", base)
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return base


@dataclass(frozen=True)
class LessonsCorpus:
    notes: tuple[FieldNote, ...] = field(default_factory=tuple)

    def prior_failures(self, model_id: str, engine: str,
                       instance_type: str) -> list[FieldNote]:
        """Field notes for the same model/engine that recorded any failure category."""
        return [n for n in self.notes
                if n.failure_categories and n.matches(model_id, engine, instance_type)]

    def recorded_categories(self, model_id: str, engine: str,
                            instance_type: str) -> dict[str, str]:
        """Map each recorded failure_category -> the source blueprint that hit it."""
        out: dict[str, str] = {}
        for n in self.prior_failures(model_id, engine, instance_type):
            for cat in n.failure_categories:
                out.setdefault(cat, n.source)
        return out


def _parse_frontmatter(text: str) -> Optional[dict]:
    m = _FM_RE.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _note_from_frontmatter(fm: dict, source: str) -> Optional[FieldNote]:
    if not fm.get("model"):
        return None
    cats = fm.get("failure_categories") or []
    if not isinstance(cats, list):
        cats = []
    return FieldNote(
        model=str(fm.get("model", "")),
        engine=str(fm.get("engine", "")),
        hardware=str(fm.get("hardware", "")),
        gpu_arch=fm.get("gpu_arch"),
        outcome=str(fm.get("outcome", "unknown")),
        failure_categories=tuple(str(c) for c in cats),
        date=fm.get("deployment_date") or fm.get("date"),
        source=source,
    )


def load_corpus(repo_root: str | Path) -> LessonsCorpus:
    """Scan domains/*/blueprints/*/lessons.md and parse field-note frontmatter.

    This is the I/O boundary. Returns a frozen corpus the pure compiler consumes.
    Files without frontmatter are skipped silently (prose-only lessons are fine).
    """
    root = Path(repo_root)
    notes: list[FieldNote] = []
    for lessons in sorted(root.glob("domains/*/blueprints/*/lessons.md")):
        fm = _parse_frontmatter(lessons.read_text(encoding="utf-8", errors="ignore"))
        if not fm:
            continue
        rel = str(lessons.relative_to(root))
        note = _note_from_frontmatter(fm, rel)
        if note:
            notes.append(note)
    return LessonsCorpus(notes=tuple(notes))
