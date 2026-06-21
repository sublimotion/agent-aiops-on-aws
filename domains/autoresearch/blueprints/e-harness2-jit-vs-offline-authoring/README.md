# E_harness2 — Layered Ablation of Harness Authoring (JIT vs offline, self vs external)

Layered ablation on **DBBench** (AgentBench SQL env from the Life-Harness repo),
driven by **Bedrock Claude**, isolating two axes of harness-*authoring*: **when**
a harness is authored (offline-evolve-then-freeze vs runtime/JIT) and **who**
authors it (the worker self-correcting vs an external verifier agent).

Spec: `domains/autoresearch/specs/e-harness2-jit-vs-offline-authoring.md`
Depends on: Life-Harness (`github.com/Tianshi-Xu/Life-Harness`, the DBBench
harness, used frozen for L1); verifier-reward T5/T4 priors; E_fin1/E_fin2 transfer law.

## The four layers (each = one isolated delta)

| Layer | Adds | Isolates |
|-------|------|----------|
| **L0 bare** | worker + 2 tools, no harness | floor |
| **L1 offline-frozen** | VENDORED Life-Harness harness (H2/H3/H4/H5), used frozen | does the harness help (replication GATE) |
| **L2 JIT self** | worker authors its own interventions in-loop from its own failures | authoring TIME (offline→runtime) |
| **L3 JIT external** | a separate verifier agent authors interventions from the worker's failures | authoring LOCUS (self→external) |

Three deltas: **L0→L1** (replication gate), **L1→L2** (is the offline freeze
necessary?), **L2→L3** (headline: does external authoring beat self? — the T5
self-critique-hurts law applied to harness construction).

## Platform & key design decisions

- **Bedrock via `aws bedrock-runtime converse` CLI with native tool-use.** No
  pip/boto3/SDK is installable here, exactly as E_fin1 found. The OpenAI Agents
  SDK path named in the spec is replaced by the native Bedrock function-calling
  equivalent; the Stage-0 smoke test + L0→L1 replication gate are the guards
  against misreading any SDK-path artifact (spec "Known Limitations"). Region
  `us-east-2`; workers Haiku 4.5 + Sonnet 4.6; external verifier (L3) = Haiku.
- **Oracle = SELECT-family DBBench tasks with a VERIFIED self-contained label.**
  DBBench INSERT/UPDATE/DELETE tasks score via MySQL `md5()`/`group_concat`
  table-hashing that the official Life-Harness code *explicitly leaves
  unimplemented for SQLite* (`task.py:607`). Replicating it would violate the
  verification-primitives lesson "never assume the eval harness works". We use
  the SELECT-family path (`label` + the vendored `DBResultProcessor`) and keep
  only tasks whose gold SQL, run in our SQLite env, reproduces the gold label.
  → 120 stratified tasks, oracle verified 120/120 in Stage 0.
- **L1 harness is the vendored Life-Harness module, used frozen** — not a
  reimplementation (`vendor/life_harness_dbbench.py`, 43 skills, H2/H3/H4/H5),
  driven through the same hooks `task.py` uses.

## Layout

```
scripts/
  dbbench_common.py   # Bedrock converse (tool-use + backoff), SQLite oracle, vendored scorer
  prepare_data.py     # build verified n=120 stratified eval set (oracle gate)
  smoke_test.py       # Stage-0 HARD GATE: offline oracle + live correct-answer + L1-layers-fire
  agent_loop.py       # one episode under any layer; L1 wires frozen Life-Harness hooks
  jit_authoring.py    # L2 self / L3 external intervention authoring + capped JitStore
  run_layer.py        # run a layer×model with incremental checkpoint + resume
  analyze.py          # Pass@1, paired-bootstrap deltas+CIs, per-type, RQ4 transfer
vendor/               # frozen Life-Harness DBBench harness + result_processor
data/                 # db_out_new.jsonl (full tables) + dbbench_eval.json (verified set)
results/              # per-layer×model jsonl, analysis.json, report.md
```

## Reproduce

```bash
git clone --depth 1 https://github.com/Tianshi-Xu/Life-Harness /tmp/Life-Harness
cd scripts
python3 prepare_data.py --n 120 --seed 42
python3 smoke_test.py --model haiku --n 5          # Stage-0 hard gate
for M in haiku sonnet; do
  python3 run_layer.py --layer L0 --model $M
  python3 run_layer.py --layer L1 --model $M       # check L0->L1 gate before L2/L3
  python3 run_layer.py --layer L2 --model $M
  python3 run_layer.py --layer L3 --model $M --verifier haiku   # external author = haiku
done
python3 analyze.py
```

See `results/report.md` for findings.
