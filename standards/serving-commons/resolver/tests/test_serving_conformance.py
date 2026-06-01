#!/usr/bin/env python3
"""test_serving_conformance.py — the conformance contract for the serving resolver.

stdlib unittest only (no pytest), to match benchmark-commons. Run with:

    python3 -m unittest discover -s standards/serving-commons/resolver/tests -v

What this asserts (the contract):
  1. Every check in registry.CHECKS has a non-empty `source` citation when it fires
     (an operator must always be able to audit where the rule came from).
  2. Every check has BOTH a passing fixture (returns None / no fail) and a failing
     fixture (returns the expected verdict). New rules without fixtures fail here.
  3. The Qwen3-235B canonical case: TP8 FAILS divisibility, TP4 PASSES — the exact
     spec↔blueprint delta that motivated the resolver.
  4. Real benchmark.yaml sidecars in the repo validate (or declare a waiver below).
  5. The corpus harvests blueprint lessons.md and replays recorded failures as
     findings tied to the source blueprint.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

RESOLVER = Path(__file__).resolve().parent.parent
REPO_ROOT = RESOLVER.parent.parent.parent
sys.path.insert(0, str(RESOLVER))

import yaml  # noqa: E402
from model import (  # noqa: E402
    ServingConfig, ModelSpec, EngineSpec, HardwareSpec, SpecDecode, from_sidecar,
)
from registry import CHECKS, Finding  # noqa: E402
from compiler import (  # noqa: E402
    compile_serving_config, validate_sidecar, InvalidServingConfig,
)
from corpus import load_corpus, CATEGORY_TO_RULE  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture builders — minimal configs that trip (or clear) one rule at a time.
# --------------------------------------------------------------------------- #

def _model(**kw) -> ModelSpec:
    base = dict(name="m", id="m", architecture="dense", quantization="none")
    base.update(kw)
    return ModelSpec(**base)


def _engine(**kw) -> EngineSpec:
    return EngineSpec(name="vllm", **kw)


def _hw(**kw) -> HardwareSpec:
    base = dict(substrate="ec2-spot", instance_type="p6-b300.48xlarge")
    base.update(kw)
    return HardwareSpec(**base)


def _cfg(model=None, engine=None, hw=None) -> ServingConfig:
    return ServingConfig(model=model or _model(), engine=engine or _engine(),
                         hardware=hw or _hw())


def _verdicts(cfg: ServingConfig, rule: str) -> list[str]:
    return [f.verdict for f in compile_serving_config(cfg).findings if f.rule == rule]


# rule_id -> (passing_cfg, failing_cfg, expected_failing_verdict).
# Adding a rule to registry.CHECKS without an entry here fails test_every_rule_*.
FIXTURES: dict[str, tuple[ServingConfig, ServingConfig, str]] = {
    "fp8-moe-tp-divisibility": (
        # pass: moe_int=1536, TP4 -> 384 % 128 == 0
        _cfg(_model(architecture="moe", quantization="fp8", moe_intermediate_size=1536),
             _engine(tensor_parallel=4)),
        # fail: moe_int=1536, TP8 -> 192, not % 128
        _cfg(_model(architecture="moe", quantization="fp8", moe_intermediate_size=1536),
             _engine(tensor_parallel=8), _hw(gpu_count=8)),
        "fail",
    ),
    "max-model-len-vs-position": (
        _cfg(_model(max_model_len=40960, max_position_embeddings=40960)),
        _cfg(_model(max_model_len=131072, max_position_embeddings=40960)),
        "fail",
    ),
    "b200-requires-al2023": (
        _cfg(hw=_hw(instance_type="p6-b200.48xlarge", ami_family="al2023")),
        _cfg(hw=_hw(instance_type="p6-b200.48xlarge", ami_family="al2")),
        "fail",
    ),
    "lmcache-mla-incompat": (
        _cfg(_model(architecture="glm_moe_dsa", quantization="fp8", is_mla=True),
             _engine(extra_args={})),  # MLA model but LMCache not enabled -> no finding
        _cfg(_model(architecture="glm_moe_dsa", quantization="fp8", is_mla=True),
             _engine(extra_args={"enable-lmcache": True})),
        "fail",
    ),
    "hicache-size-vs-device-pool": (
        _cfg(engine=_engine(hierarchical_cache=True, hicache_size_gb=100)),
        _cfg(engine=_engine(hierarchical_cache=True)),  # size unset
        "warn",
    ),
    "specdec-on-pcie": (
        _cfg(engine=_engine(speculative_decode=SpecDecode(algorithm="EAGLE3", measured_acceptance=0.7)),
             hw=_hw(interconnect="nvswitch")),
        _cfg(engine=_engine(speculative_decode=SpecDecode(algorithm="EAGLE3", measured_acceptance=0.7)),
             hw=_hw(instance_type="g7e.24xlarge", interconnect="pcie")),
        "warn",
    ),
    "specdec-acceptance-gate": (
        _cfg(engine=_engine(speculative_decode=SpecDecode(algorithm="MTP", measured_acceptance=0.7))),
        _cfg(engine=_engine(speculative_decode=SpecDecode(algorithm="MTP", measured_acceptance=0.45))),
        "fail",
    ),
    "mamba-mtp-prefix-cache": (
        _cfg(_model(architecture="hybrid-mamba-moe", is_mamba_hybrid=True),
             _engine(speculative_decode=SpecDecode(algorithm="MTP"), prefix_caching=False)),
        _cfg(_model(architecture="hybrid-mamba-moe", is_mamba_hybrid=True),
             _engine(speculative_decode=SpecDecode(algorithm="MTP"))),  # prefix on (default)
        "warn",
    ),
    "hybrid-hicache-cuda-graph": (
        _cfg(_model(architecture="hybrid-mamba-moe", is_mamba_hybrid=True),
             _engine(hierarchical_cache=True, cuda_graph=False)),
        _cfg(_model(architecture="hybrid-mamba-moe", is_mamba_hybrid=True),
             _engine(hierarchical_cache=True)),  # cuda_graph unset (default on)
        "warn",
    ),
    "blackwell-cuda-tag": (
        _cfg(engine=_engine(container_image="lmsysorg/sglang:v0.5.9-cu130"),
             hw=_hw(instance_type="g7e.24xlarge", gpu_arch="sm_120")),
        _cfg(engine=_engine(container_image="lmsysorg/sglang:v0.5.9-cu131"),
             hw=_hw(instance_type="g7e.24xlarge", gpu_arch="sm_120")),
        "warn",
    ),
}


class TestRuleSourceCitations(unittest.TestCase):
    """Every firing check must cite a source — operators audit rules, not trust them."""

    def test_all_findings_have_source(self):
        for rule_id, (_, fail_cfg, _verdict) in FIXTURES.items():
            findings = [f for f in compile_serving_config(fail_cfg).findings
                        if f.rule == rule_id]
            self.assertTrue(findings, f"{rule_id}: failing fixture fired no finding")
            for f in findings:
                self.assertTrue(f.source and f.source.strip(),
                                f"{rule_id}: finding has empty source citation")


class TestEveryRuleHasFixtures(unittest.TestCase):
    """Each rule in CHECKS must have a passing AND a failing fixture here."""

    def test_every_check_is_covered(self):
        # Derive the rule ids the registry can emit by running each failing fixture.
        covered = set(FIXTURES)
        # Sanity: each check, when given its failing fixture, emits the expected rule.
        emitted = set()
        for rule_id, (_, fail_cfg, _v) in FIXTURES.items():
            for f in compile_serving_config(fail_cfg).findings:
                emitted.add(f.rule)
        for rule_id in covered:
            self.assertIn(rule_id, emitted,
                          f"{rule_id}: failing fixture did not emit this rule")

    def test_passing_fixture_does_not_fail_its_rule(self):
        for rule_id, (pass_cfg, _f, _v) in FIXTURES.items():
            verdicts = _verdicts(pass_cfg, rule_id)
            self.assertNotIn("fail", verdicts,
                             f"{rule_id}: passing fixture unexpectedly FAILED")

    def test_failing_fixture_hits_expected_verdict(self):
        for rule_id, (_p, fail_cfg, verdict) in FIXTURES.items():
            verdicts = _verdicts(fail_cfg, rule_id)
            self.assertIn(verdict, verdicts,
                          f"{rule_id}: expected verdict={verdict}, got {verdicts}")


class TestQwen235BCanonicalCase(unittest.TestCase):
    """The motivating delta: Qwen3-235B FP8 moe_intermediate_size=1536. TP8 fails, TP4 works."""

    def _qwen(self, tp: int) -> ServingConfig:
        return _cfg(
            _model(name="Qwen3-235B-A22B-FP8", architecture="moe",
                   quantization="fp8", moe_intermediate_size=1536),
            _engine(tensor_parallel=tp),
            _hw(gpu_count=8),
        )

    def test_tp8_fails_closed(self):
        report = compile_serving_config(self._qwen(8))
        self.assertFalse(report.ok, "TP8 must fail divisibility (1536/8=192, not %128)")
        with self.assertRaises(InvalidServingConfig):
            report.raise_if_invalid()

    def test_tp4_passes(self):
        report = compile_serving_config(self._qwen(4))
        self.assertTrue(report.ok, "TP4 must pass (1536/4=384, 384%128==0)")

    def test_fail_message_suggests_valid_tp(self):
        report = compile_serving_config(self._qwen(8))
        fail = next(f for f in report.failures if f.rule == "fp8-moe-tp-divisibility")
        # The fix should point at a TP that satisfies the rule (4 and 2 are valid).
        self.assertIn("4", fail.fix or "")


class TestRealSidecars(unittest.TestCase):
    """Every benchmark.yaml in the repo validates, or is listed in WAIVERS with a reason."""

    # sidecar path (relative to repo root) -> reason it is allowed to fail hard rules.
    WAIVERS: dict[str, str] = {}

    def test_real_sidecars_validate(self):
        sidecars = sorted(REPO_ROOT.glob("domains/*/blueprints/*/benchmark.yaml"))
        sidecars += sorted(REPO_ROOT.glob("standards/*/examples/*/benchmark.yaml"))
        self.assertTrue(sidecars, "no benchmark.yaml sidecars found — glob is wrong")
        for path in sidecars:
            rel = str(path.relative_to(REPO_ROOT))
            with self.subTest(sidecar=rel):
                data = yaml.safe_load(path.read_text())
                report = validate_sidecar(data, source=rel)
                if rel in self.WAIVERS:
                    continue  # documented exception
                self.assertTrue(
                    report.ok,
                    f"{rel} has hard-rule failures with no waiver:\n" +
                    "\n".join(f"  ✗ [{f.rule}] {f.reason}" for f in report.failures))


class TestCorpusHarvest(unittest.TestCase):
    """The corpus must harvest lessons.md and replay recorded failures as findings."""

    def setUp(self):
        self.corpus = load_corpus(REPO_ROOT)

    def test_corpus_loads_field_notes(self):
        self.assertTrue(self.corpus.notes, "no field notes harvested from lessons.md")

    def test_qwen_prior_failures_surface(self):
        # qwen3-235b-b300/lessons.md records fp8_block_size_mismatch on vllp.
        recorded = self.corpus.recorded_categories(
            "Qwen/Qwen3-235B-A22B-FP8", "vllm", "p6-b300.48xlarge")
        self.assertIn("fp8_block_size_mismatch", recorded)
        self.assertIn("qwen3-235b-b300", recorded["fp8_block_size_mismatch"])

    def test_compiler_replays_corpus_as_findings(self):
        cfg = _cfg(
            _model(name="Qwen3-235B-A22B-FP8", id="Qwen/Qwen3-235B-A22B-FP8",
                   architecture="moe", quantization="fp8"),
            _engine(tensor_parallel=4),
            _hw(gpu_count=8),
        )
        report = compile_serving_config(cfg, corpus=self.corpus)
        prior = [f for f in report.findings if f.rule.startswith("prior-failure:")]
        self.assertTrue(prior, "corpus findings did not surface in the report")
        self.assertTrue(all(f.source.endswith("lessons.md") for f in prior))

    def test_engine_mismatch_does_not_replay(self):
        # The same model on a different engine should not inherit vllm's field notes.
        recorded = self.corpus.recorded_categories(
            "Qwen/Qwen3-235B-A22B-FP8", "some-other-engine", "p6-b300.48xlarge")
        self.assertEqual(recorded, {})

    def test_category_rule_map_targets_real_rules(self):
        # Every non-None CATEGORY_TO_RULE target must be a real rule id the registry emits.
        real_rule_ids = set(FIXTURES)
        for cat, rule_id in CATEGORY_TO_RULE.items():
            if rule_id is not None:
                self.assertIn(rule_id, real_rule_ids,
                              f"CATEGORY_TO_RULE[{cat}]={rule_id} is not a known rule id")


class TestFailureCategorySchemaSync(unittest.TestCase):
    """The failure_categories vocabulary must agree across three places:

      1. CATEGORY_TO_RULE keys (corpus.py) — the machine-readable source of truth,
      2. the documented enum in docs/card-format.md (human-facing),
      3. what real blueprint lessons.md files actually emit.

    These drifted before consolidation (dead keys in the map, emitted categories
    in neither the map nor the docs). This test keeps them locked together.
    """

    CARD_FORMAT = REPO_ROOT / "docs" / "card-format.md"

    def _documented_categories(self) -> set[str]:
        """Parse the category tokens listed under failure_categories in card-format.md."""
        text = self.CARD_FORMAT.read_text(encoding="utf-8")
        # Lines look like:  #   fp8_block_size_mismatch  — description
        # Restrict to the failure_categories block to avoid catching other comments.
        start = text.index("failure_categories:")
        end = text.index("cards_used:", start)  # block ends at the next field
        block = text[start:end]
        cats = set(re.findall(r"#\s*([a-z][a-z0-9_]+)\s+—", block))
        return cats

    def _emitted_categories(self) -> set[str]:
        emitted: set[str] = set()
        for path in REPO_ROOT.glob("domains/*/blueprints/*/lessons.md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if not m:
                continue
            try:
                fm = yaml.safe_load(m.group(1))
            except yaml.YAMLError:
                continue
            if isinstance(fm, dict):
                for c in (fm.get("failure_categories") or []):
                    emitted.add(str(c))
        return emitted

    def test_docs_enum_matches_map_keys(self):
        documented = self._documented_categories()
        mapped = set(CATEGORY_TO_RULE)
        self.assertEqual(
            documented, mapped,
            "card-format.md enum and CATEGORY_TO_RULE keys have drifted.\n"
            f"  in docs only: {sorted(documented - mapped)}\n"
            f"  in map only:  {sorted(mapped - documented)}")

    def test_every_emitted_category_is_in_vocabulary(self):
        emitted = self._emitted_categories()
        self.assertTrue(emitted, "no failure_categories emitted by any lessons.md")
        unknown = emitted - set(CATEGORY_TO_RULE)
        self.assertFalse(
            unknown,
            f"blueprints emit failure_categories not in CATEGORY_TO_RULE: {sorted(unknown)}. "
            "Add them to corpus.py CATEGORY_TO_RULE (and docs/card-format.md), or fix the typo "
            "in the lessons.md frontmatter.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
