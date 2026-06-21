# E_fin1 — Cross-Domain Skill-Verifier Replication (FinQA)

Tests whether the adversarial skill-verifier result from the coding domain
(precision **0.40 → 0.92** from confirmatory→adversarial reframing, the
verifier-reward v009 result) **transfers off-coding** to financial numeric
reasoning on FinQA — a near-structural clone of the coding setup minus test
suites: objective numeric ground truth (`qa.exe_ans`), no human-in-the-loop,
~$0.03/eval target.

Spec: `domains/autoresearch/specs/e-fin1-finqa-skill-verifier.md`
Depends on: `verifier-reward` (v009 adversarial-ensemble pattern, 0.92/$0.03 coding baseline).

## What this runs

1. **Generate** ~100 agent answers (Haiku) over FinQA dev — numeric answer +
   reasoning from question + table + gold supporting text.
2. **Score** each against `qa.exe_ans` by exact-match with rounding/units
   tolerance (the Docker-execution analog: $0, objective).
3. **Verifier A (confirmatory)**: 1 call, "rate 1–5 for correctness".
4. **Verifier B (adversarial, v009 analog)**: 4-call temperature ensemble
   (1@t=0.0, 3@t=0.3); **4/4 unanimous `likely_correct` = confident**.
   v009-only, no confirmatory gate (verifier-reward T10b: v009-only is the ceiling).
5. **Analyze**: precision/recall/AUC on the confident subset, adversarial vs
   confirmatory, vs the coding 2.3× lift; cost/eval vs $0.03; calibration check.

## Platform

API-driven via Bedrock (`aws bedrock-runtime converse` CLI — **no boto3, no GPU**).
Region `us-east-2`. Models verified callable 2026-06-21: Haiku 4.5, Sonnet 4.6,
Nova Pro (the last for the RQ2 cross-verifier calibration check).

## Layout

```
scripts/
  finqa_common.py     # Bedrock CLI call, context serialization, exact_match scorer, cost
  prepare_data.py     # deterministic n=100 dev sample
  smoke_test.py       # Stage-0 gate: scorer directional checks + schema + live API
  generate_answers.py # Stage 1: agent answers + exact-match labels
  run_verifiers.py    # Stage 2-3: confirmatory + adversarial(4-call) verifiers
  analyze.py          # Stage 7: precision/recall/AUC/cost/calibration report
skills/finqa-verifier/versions/
  confirmatory.md     # Verifier A rubric (1-5 rating)
  adversarial.md      # Verifier B rubric (v009 port: assume-wrong, recompute, attack)
data/finqa_dev_n100.json     # sampled eval set (seed=42)
results/                     # answers, verifier outputs, aggregate, analysis
```

## Reproduce

```bash
# clone FinQA original (MIT) — uses qa.exe_ans as numeric ground truth
git clone --depth 1 https://github.com/czyssrs/FinQA /tmp/FinQA
cd scripts
python3 prepare_data.py --src /tmp/FinQA/dataset/dev.json --n 100 --seed 42 \
    --out ../data/finqa_dev_n100.json
python3 smoke_test.py --data ../data/finqa_dev_n100.json          # Stage-0 gate
python3 generate_answers.py --data ../data/finqa_dev_n100.json \
    --out ../results/agent_answers.jsonl --model haiku
python3 run_verifiers.py --answers ../results/agent_answers.jsonl \
    --data ../data/finqa_dev_n100.json --out ../results/verifier_results.jsonl
python3 analyze.py --results ../results/verifier_results.jsonl \
    --out ../results/aggregate.json
```

Findings: see `results/analysis.md` and `lessons.md`.
