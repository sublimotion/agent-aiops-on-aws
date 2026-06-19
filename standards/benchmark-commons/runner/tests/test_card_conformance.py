#!/usr/bin/env python3
"""Conformance test — the CI gate that keeps the framework deterministic.

Asserts, for EVERY card in workloads/*.yaml and BOTH tools (vllm, sglang):
  1. compile_card resolves to a plan that is either VENDOR (concrete argv) or
     ORCHESTRATED (named executor + reason) — never a silent default, never an
     unhandled exception.
  2. Every (dataset.type, load.type) pair in the catalog has a registry entry.
  3. Sweep cards expand to exactly the declared number of steps.
  4. The goodput SLO propagates into vendor argv when the card declares one.
  5. Shared-prefix cards actually emit prefix flags (the old silent-drop bug).

Pure stdlib unittest — no pytest dependency (the target hosts ship bare python).
Run:  python3 -m unittest discover -s runner/tests -v
   or  python3 runner/tests/test_card_conformance.py
"""
from __future__ import annotations

import glob
import os
import sys
import unittest

# Make runner/ importable whether invoked via discover or directly.
RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RUNNER_DIR)
ARTIFACT_DIR = os.path.dirname(RUNNER_DIR)
WORKLOAD_DIR = os.path.join(ARTIFACT_DIR, "workloads")

import yaml  # noqa: E402
from compiler import compile_card, ExecutionPlan  # noqa: E402
from registry import (  # noqa: E402
    DATASET_HANDLERS, LOAD_EXPANDERS, UnsupportedWorkload,
)

TOOLS = ("vllm", "sglang")

# Minimal but schema-plausible fixture sidecar. Includes the trace path the
# trace-replay/real cards need so they compile in vendor mode; includes a
# cohost ensemble so that card's orchestrator resolves with full context.
FIXTURE_SIDECAR = {
    "model": {"name": "fixture-model", "id": "org/fixture-model"},
    "engine": {"name": "vllm"},
    "infrastructure": {"substrate": "local", "instance_type": "g7e.48xlarge"},
    "workload_overrides": {
        "trace": {"source": "sharegpt", "sample_size": 200},
    },
    "slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
}


def load_cards() -> dict[str, dict]:
    cards = {}
    for f in sorted(glob.glob(os.path.join(WORKLOAD_DIR, "*.yaml"))):
        c = yaml.safe_load(open(f))
        cid = c.get("catalog_id", os.path.basename(f)[:-5])
        cards[cid] = c
    return cards


CARDS = load_cards()


class TestEveryCardResolves(unittest.TestCase):
    """Each card must compile to a deterministic plan for both tools."""

    def test_all_cards_resolve(self):
        self.assertEqual(len(CARDS), 23, "expected 23 workload cards")
        failures = []
        for cid, card in CARDS.items():
            for tool in TOOLS:
                try:
                    plan = compile_card(card, FIXTURE_SIDECAR, tool)
                except UnsupportedWorkload as e:
                    # An UnsupportedWorkload with a clear message is an
                    # acceptable *deterministic* outcome ONLY for a genuinely
                    # unmapped pair. None should exist today — flag it.
                    failures.append(f"{cid}/{tool}: UnsupportedWorkload: {e}")
                    continue
                except Exception as e:  # noqa: BLE001
                    failures.append(f"{cid}/{tool}: unexpected {type(e).__name__}: {e}")
                    continue
                self.assertIsInstance(plan, ExecutionPlan)
                self.assertIn(plan.kind, ("vendor", "orchestrated"))
                if plan.kind == "vendor":
                    self.assertTrue(plan.steps, f"{cid}/{tool}: vendor plan has no steps")
                    for s in plan.steps:
                        self.assertIn("--dataset-name", s.argv,
                                      f"{cid}/{tool}: step {s.label} missing dataset flag "
                                      f"(silent-default bug)")
                else:
                    self.assertTrue(plan.orchestrator,
                                    f"{cid}/{tool}: orchestrated plan has no executor")
                    self.assertTrue(plan.reason,
                                    f"{cid}/{tool}: orchestrated plan has no reason")
        self.assertEqual(failures, [], "cards failed to resolve:\n" + "\n".join(failures))


class TestRegistryCoverage(unittest.TestCase):
    """Every (dataset.type, load.type) pair in the catalog is registered."""

    def test_dataset_types_registered(self):
        missing = sorted({c["dataset"]["type"] for c in CARDS.values()}
                         - set(DATASET_HANDLERS))
        self.assertEqual(missing, [], f"dataset.type with no handler: {missing}")

    def test_load_types_registered(self):
        declared = set()
        for c in CARDS.values():
            ld = c.get("load", {})
            if "type" in ld:
                declared.add(ld["type"])
            # quantization-pareto uses load.modes, handled by the compiler.
        missing = sorted(declared - set(LOAD_EXPANDERS))
        self.assertEqual(missing, [], f"load.type with no expander: {missing}")


class TestSweepExpansion(unittest.TestCase):
    """Sweep cards must expand to the declared number of concrete steps."""

    def test_qps_sweep_step_count(self):
        plan = compile_card(CARDS["qps-sweep"], FIXTURE_SIDECAR, "vllm")
        self.assertEqual(len(plan.steps), len(CARDS["qps-sweep"]["load"]["rates"]))

    def test_concurrency_sweep_step_count(self):
        plan = compile_card(CARDS["concurrency-sweep"], FIXTURE_SIDECAR, "vllm")
        self.assertEqual(len(plan.steps), len(CARDS["concurrency-sweep"]["load"]["levels"]))

    def test_quant_pareto_modes(self):
        plan = compile_card(CARDS["quantization-pareto"], FIXTURE_SIDECAR, "vllm")
        self.assertEqual({s.mode for s in plan.steps}, {"offline", "server"})

    def test_rag_1m_context_tiers_distinct(self):
        """rag-1m-context sweeps 5 prefix-length tiers; each step must carry a
        DISTINCT prefix length, else the sweep is meaningless."""
        plan = compile_card(CARDS["rag-1m-context"], FIXTURE_SIDECAR, "vllm")
        tiers = CARDS["rag-1m-context"]["dataset"]["context_tiers"]
        self.assertEqual(len(plan.steps), len(tiers))
        prefix_lens = []
        for s in plan.steps:
            i = s.argv.index("--prefix-repetition-prefix-len")
            prefix_lens.append(s.argv[i + 1])
        self.assertEqual(len(set(prefix_lens)), len(tiers),
                         f"tier prefix lengths not distinct: {prefix_lens}")


class TestGoodputPropagation(unittest.TestCase):
    """The goodput SLO must reach vendor argv (borrowed from vllm-skills)."""

    def test_goodput_in_argv(self):
        plan = compile_card(CARDS["chatbot-short"], FIXTURE_SIDECAR, "vllm")
        argv = plan.steps[0].argv
        self.assertIn("--goodput", argv)
        joined = " ".join(argv)
        self.assertIn("ttft:300", joined)  # chatbot-short slo ttft_p99_ms: 300
        self.assertIn("tpot:50", joined)


class TestPrefixFidelity(unittest.TestCase):
    """Shared-prefix cards must emit real prefix flags — regression guard for
    the old fail-open behaviour where NO dataset flags were emitted and prefix
    caching was never exercised."""

    def test_vllm_prefix_repetition(self):
        for cid in ("rag-long-context", "shared-prefix-multitenant"):
            plan = compile_card(CARDS[cid], FIXTURE_SIDECAR, "vllm")
            argv = plan.steps[0].argv
            self.assertIn("prefix_repetition", argv, f"{cid}: not using prefix_repetition")
            self.assertIn("--prefix-repetition-prefix-len", argv)

    def test_sglang_gsp(self):
        plan = compile_card(CARDS["rag-long-context"], FIXTURE_SIDECAR, "sglang")
        self.assertIn("generated-shared-prefix", plan.steps[0].argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
