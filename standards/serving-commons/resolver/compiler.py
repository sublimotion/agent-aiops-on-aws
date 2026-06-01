#!/usr/bin/env python3
"""compiler.py — the pure serving-config resolver.

`compile_serving_config(cfg)` runs every rule in `registry.CHECKS` over a
ServingConfig and returns a ValidationReport. It is PURE: no I/O, no network, no
GPU. The same config always yields the same report.

Fail-CLOSED contract: if any check returns verdict="fail", the report's
`ok` is False. Callers that gate a deployment should refuse to proceed —
`raise_if_invalid()` does exactly that, raising `InvalidServingConfig` with all
failures and their fixes. This mirrors benchmark-commons' compile_card, which
raises UnsupportedWorkload rather than silently degrading.

Warnings and info findings never block; they are surfaced so an operator (or an
agent reading the report) sees the hard-won caveat without having to remember it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from model import ServingConfig, from_sidecar  # re-exported for convenience
from registry import CHECKS, Finding
from corpus import LessonsCorpus, CATEGORY_TO_RULE


class InvalidServingConfig(Exception):
    """Raised when a serving config violates one or more hard rules."""

    def __init__(self, report: "ValidationReport"):
        self.report = report
        lines = [f"Serving config has {len(report.failures)} hard-rule "
                 f"violation(s)" + (f" [{report.config.source}]" if report.config.source else "") + ":"]
        for f in report.failures:
            lines.append(f"\n  ✗ [{f.rule}] {f.reason}")
            lines.append(f"    source: {f.source}")
            if f.fix:
                lines.append(f"    fix:    {f.fix}")
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class ValidationReport:
    config: ServingConfig
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "fail"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "warn"]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "info"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_invalid(self) -> "ValidationReport":
        if not self.ok:
            raise InvalidServingConfig(self)
        return self


def _corpus_findings(cfg: ServingConfig, corpus: LessonsCorpus) -> list[Finding]:
    """Turn prior recorded failures for this model/engine into Findings.

    PURE — operates over the already-loaded corpus. This is where the spec↔blueprint
    gap surfaces: a config may pass every static rule yet a past deployment of the
    same model/engine recorded a failure_category. We replay that as a finding so the
    operator sees the hard-won caveat. A category that maps to a codified rule (via
    CATEGORY_TO_RULE) is INFO when that rule already fired in this report would be
    redundant, so we surface it as 'warn' to prompt a look; uncodified categories are
    'warn' too (no deterministic check guards them yet).
    """
    recorded = corpus.recorded_categories(cfg.model.id, cfg.engine.name,
                                           cfg.hardware.instance_type)
    out: list[Finding] = []
    for cat, src in sorted(recorded.items()):
        rule_id = CATEGORY_TO_RULE.get(cat)
        codified = f" A codified check exists for this ({rule_id}); confirm it fired." \
            if rule_id else " No deterministic check guards this category yet."
        out.append(Finding(
            rule=f"prior-failure:{cat}",
            verdict="warn",
            reason=(f"A prior deployment of {cfg.model.id} on {cfg.engine.name} "
                    f"recorded failure_category={cat}.{codified}"),
            source=src,
            fix=("Read the lessons.md field note before deploying; the recorded "
                 "failure may apply to this config."),
        ))
    return out


def compile_serving_config(cfg: ServingConfig,
                           corpus: Optional[LessonsCorpus] = None) -> ValidationReport:
    """Run all applicable rules. PURE — no side effects.

    If `corpus` is supplied (already loaded from disk via corpus.load_corpus), prior
    recorded failures for the same model/engine are replayed as findings, connecting
    the empirical blueprint lessons to the deterministic rule registry.
    """
    findings: list[Finding] = []
    for check in CHECKS:
        result = check(cfg)
        if result is not None:
            findings.append(result)
    if corpus is not None:
        findings.extend(_corpus_findings(cfg, corpus))
    return ValidationReport(config=cfg, findings=findings)


def validate_sidecar(sidecar: dict, *, card: dict | None = None,
                     source: str | None = None,
                     corpus: Optional[LessonsCorpus] = None) -> ValidationReport:
    """Parse a sidecar dict and validate it in one call."""
    return compile_serving_config(
        from_sidecar(sidecar, card=card, source=source), corpus=corpus)


# Re-exports so callers import from one module.
__all__ = [
    "compile_serving_config", "validate_sidecar", "ValidationReport",
    "InvalidServingConfig", "from_sidecar", "ServingConfig",
    "LessonsCorpus",
]
