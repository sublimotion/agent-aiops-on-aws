# E_harness3 — Reward-Regime × Authoring-Locus Matrix

Tests **whether the external-author advantage grows as the verifiable reward weakens** —
the unifying law proposed by E_harness2's null. One 2×3 matrix crosses **authoring
locus** (self vs external) with **reward regime** (verifiable → withheld → consensus):

| | **self-author** | **external-author** |
|---|---|---|
| **Verifiable** (DBBench, SQL execution) | **A** = E_harness2 L2 (loaded) | **C** = E_harness2 L3 (loaded) |
| **Withheld** (DBBench, reward blinded to the *author*) | **B** (new) | **D** (new) |
| **Consensus** (FinanceBench, LLM-judge vs reference) | **E** (new) | **F** (new) |

**Core hypothesis (monotonic):** locus gap `(external − self)` is ≈0 when verifiable
(A≈C, confirmed by E_harness2), re-emerges when reward is withheld (D>B), and is
largest under consensus reward where the LLM-judge *is* the verification asymmetry
(F>E). If it holds: **external authoring pays in proportion to the verification
asymmetry it supplies** — unifying E_harness2 and E_fin1.

Spec: `domains/autoresearch/specs/e-harness3-reward-regime-x-locus.md`
Depends on: **E_harness2** (cells A/C + the whole DBBench/Bedrock toolchain, reused),
**E_fin1** (FinanceBench-is-messy + judge-calibration lessons → the Stage-0 judge gate),
verifier-reward T4/T5.

## Two HARD GATES (both PASS — see `results/`)

1. **Reward-withholding leak audit** (`leak_audit.py` → `results/leak_audit.json`) —
   B/D are a clean ablation only if the *author* cannot see DBBench pass/fail. We
   prove it two ways on 240 real trajectories: (a) a **structural** gate — the
   withheld digest exposes no reward FIELD (no `Gold answer:` line, no `(WRONG)`
   tag), validated against the visible (A/C) digest as a positive control; (b)
   **empirical separability** — the strongest single digest channel recovers reward
   at AUC 0.547 ≪ 0.90, i.e. informative-but-not-readable. Plus the key
   carryover-auditor fix: under withholding the author is invoked on **every** task
   (reward-independent schedule), so the *invocation itself* leaks nothing.

2. **FinanceBench judge gate** (`judge_gate.py` → `results/judge_gate.json`) — per
   E_fin1, a same-tier judge can ENGAGE without DISCRIMINATING, so stability alone
   is insufficient. The gate measures **discrimination (AUC on labeled pairs)** and
   a **near-miss numeric probe** (on-topic answers perturbed ×1.4 — the exact E_fin1
   failure mode), not just temperature stability. If the judge fails, E/F are
   reported "judge-confounded, inconclusive" and B/D still stand.

## Reuse discipline

- **A/C are LOADED, never re-run** — `results/A_*.jsonl`/`C_*.jsonl` are E_harness2's
  `L2_*`/`L3_*` verbatim. The DBBench eval set is regenerated with the same seed (42)
  so B/D pair on the **same 120 task_ids** as A/C (verified).
- DBBench toolchain (`dbbench_common.py`, `agent_loop.py`, vendored frozen
  Life-Harness) copied from E_harness2 — includes the multi-tool-turn Bedrock fix.
- `jit_authoring.py` generalizes E_harness2's authoring with a second axis
  (reward visibility); `finbench_common.py` is the new consensus-regime machinery.

## Layout

```
scripts/
  dbbench_common.py     # (reused) Bedrock converse, SQLite oracle, vendored scorer
  agent_loop.py         # (reused) DBBench episode, L2-style loop for A/B/C/D
  jit_authoring.py      # authoring with LOCUS × REWARD-VISIBILITY axes (A/B/C/D)
  finbench_common.py    # FinanceBench worker + LLM-judge (E/F)
  run_dbbench_cell.py   # run a DBBench cell (B/D; A/C parity available)
  run_finbench_cell.py  # run a FinanceBench cell (E/F)
  leak_audit.py         # HARD GATE: reward-withholding leak audit (B/D)
  judge_gate.py         # HARD GATE: judge discrimination + near-miss (E/F)
  run_all.sh            # 4 parallel lanes: B/D × {haiku,sonnet}, E/F × {haiku,sonnet}
  prepare_data.py       # (reused) rebuild verified DBBench n=120 (seed 42 → matches A/C ids)
  analyze.py            # matrix, locus gaps + CIs, monotonic-trend test, RQ4
vendor/                 # (reused) frozen Life-Harness DBBench harness + result_processor
data/                   # (gitignored) regenerated DBBench n=120 + FinanceBench n=150
results/                # A/B/C/D/E/F jsonl, leak_audit.json, judge_gate.json, matrix.json, report.md
```

## Reproduce

```bash
git clone --depth 1 https://github.com/Tianshi-Xu/Life-Harness /tmp/Life-Harness
git clone --depth 1 https://github.com/patronus-ai/financebench /tmp/financebench
cd scripts
cp /tmp/Life-Harness/AgentBench/data/dbbench/db_out_new.jsonl ../data/
cp /tmp/financebench/data/financebench_open_source.jsonl ../data/
python3 prepare_data.py --n 120 --seed 42          # rebuild DBBench eval (matches A/C ids)
# load A/C from E_harness2:
for m in haiku sonnet; do
  cp ../../e-harness2-jit-vs-offline-authoring/results/L2_$m.jsonl ../results/A_$m.jsonl
  cp ../../e-harness2-jit-vs-offline-authoring/results/L3_${m}_v-haiku.jsonl ../results/C_$m.jsonl
done
python3 leak_audit.py                               # HARD GATE (B/D)
python3 judge_gate.py --model haiku                 # HARD GATE (E/F)
bash run_all.sh                                     # B/D/E/F, both workers
python3 analyze.py
```

See `results/report.md` for findings.
