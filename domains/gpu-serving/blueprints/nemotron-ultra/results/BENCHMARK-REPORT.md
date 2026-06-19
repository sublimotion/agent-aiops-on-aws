# Nemotron-3-Ultra-550B-A55B-NVFP4 — Benchmark Report

**Question:** Can a self-hosted deployment beat DeepInfra's published
**300 tok/s single-stream** and **$2.50/M output** for Nemotron-3-Ultra?

**Verdict (revised after vLLM re-test):** **Speed: essentially MATCHED — 297.7 tok/s
single-stream decode (0.99×), 6/12 prompts clear 300.** Retail cost still loses:
~$6.00/M output at node saturation vs DeepInfra's $2.50/M.

The first run (SGLang/B300) concluded "300 tok/s is unreachable, acceptance is an
inherent model property." **That conclusion was wrong.** It was SGLang bug
[#21138](https://github.com/sgl-project/sglang/issues/21138) — NemotronH MTP gets ~0%
real acceptance on SGLang (accept_len 2.4 = base token only). The **same weights on
vLLM v0.22.0 reach accept_len 3.54 and decode 297.7 tok/s** — a 1.7× single-stream
jump on **half the GPUs** (B200 TP4 vs B300 TP8).

---

## Two runs, side by side

| | Run 1 (first) | Run 2 (re-test) |
|---|---|---|
| Engine | SGLang v0.5.12.post1-cu130 | **vLLM v0.22.0** |
| Hardware | p6-b300.48xlarge, 8× B300, TP8 | **p6-b200.48xlarge, 4× B200, TP4** |
| Spec decode | EAGLE/MTP wide-tree (topk=4, draft=16) | **nemotron_h_mtp k=5 (native)** |
| accept_len (diverse prompts) | 2.43 (bugged, #21138) | **3.54** |
| Single-stream decode median | 176.8 tok/s | **297.7 tok/s** |
| Single-stream wall (e2e) median | 130.1 tok/s | **267.6 tok/s** |
| Peak aggregate (c=256) | ~1040 tok/s | **~1883 tok/s** |
| $/M output @ saturation | $7.21 (@ $27/hr) | **$6.00 (@ $40.57/hr)** |

---

## Setup (Run 2 — vLLM, the authoritative run)

| | |
|---|---|
| Model | `nvidia/Nemotron-3-Ultra-550B-A55B-NVFP4` (550B total, 55B active MoE) |
| Quant | NVFP4 native (MoE experts), FP8 (mamba) — MIXED_PRECISION, group_size=16 |
| Arch | Hybrid Mamba-2 + LatentMoE + Select-Attention (`nemotron_h`) |
| Hardware | p6-b200.48xlarge, 4× B200 (TP4) of 8, NVSwitch, us-east-2 az2 spot |
| Engine | vLLM v0.22.0, NVIDIA min-latency recipe, `nemotron_h_mtp` k=5, flashinfer MoE (latency), trtllm allreduce |
| Spend rate | ~$40.57/hr spot (full 8-GPU instance billed) |
| Methodology | Real diverse prompts, temp=1.0, top_p=0.95. Single-stream = streaming `/v1/completions`; accept_len from Prometheus `/metrics`; cost = node-aggregate at saturation. |
| Date | 2026-06-06 |

DeepInfra reference (Artificial Analysis measured): **300 tok/s** single-stream,
**$0.50/M in · $2.50/M out**, 262K context.

---

## P0 — Smoke gate (PASSED)

health 200 ✅ · model registered ✅ · completion ✅ · tool call (`qwen3_coder`) ✅ ·
live MTP acceptance (accept_len 3.81 on a single reasoning prompt) ✅. Reasoning parser
(`nemotron_v3`) is wired but the model inlines reasoning without `<think>` delimiters on
short prompts — benchmarks use `/v1/completions` so this is immaterial. Recipe fix vs
deployment card: vLLM v0.22.0 dropped `--disable-log-requests` (remove it).

## P2 — Single-stream speed (THE headline)

| Metric | median | mean | min | max | vs 300 |
|---|---|---|---|---|---|
| **decode tok/s** (post-TTFT) | **297.7** | 294.2 | 197.6 | 398.9 | **0.99×** |
| wall tok/s (end-to-end) | 267.6 | 276.8 | 193.8 | 360.0 | 0.89× |

accept_len **3.54** (accept_rate 0.508, k=5). **6/12 prompts clear 300 decode, 5/12
clear 300 wall.** TTFT median 0.057s. The decode median sits right on the 300 line —
DeepInfra's SLA is effectively matched on vLLM. NVIDIA publishes accept_len 4.9–5.5
(likely lower temp / specific workloads), so further headroom exists. **Acceptance is
still the gate — but it's an engine-implementation property, not a model ceiling.**

## P1 — Standard workload suite (vLLM TP4)

| Workload | conc | in→out | agg tok/s | e2e median | err | Run 1 (SGLang) |
|---|---|---|---|---|---|---|
| chatbot-short | 16 | 256→128 | 1336.6 | 1.53s | 0 | 451.9 (2.96×) |
| chatbot-long | 4 | 32768→512 | 459.6 | 4.45s | 0 | 166.7 (2.75×) |
| rag-long-context | 8 | 16384→256 | 1233.3 | 1.66s | 0 | 251.2 (4.91×) |
| coding-agent | 8 | 4096→2048 | 2008.4 | 8.16s | 0 | 637.7 (3.15×) |
| sharegpt-production-mix | 16 | 512→256 | 1574.5 | 2.60s | 0 | 662.6 (2.37×) |

vLLM TP4/B200 beats SGLang TP8/B300 on **every** workload, 2.4×–4.9×, on half the GPUs.
(accept_len omitted — synthetic input padding is repetitive and inflates it to the K+1
cap; realistic diverse-prompt accept_len is the single-stream 3.54.)

## P2 — Concurrency sweep (vLLM TP4)

| conc | agg tok/s | per-req median | accept_len | err |
|---|---|---|---|---|
| 1 | 269.2 | 263.6 | 3.56 | 0 |
| 4 | 449.2 | 153.7 | 3.56 | 0 |
| 16 | 1457.1 | 119.0 | 3.58 | 0 |
| 32 | 1760.6 | 64.9 | 3.55 | 0 |
| 64 | 1841.9 | 63.0 | 3.55 | 0 |
| 128 | 1870.1 | 62.6 | 3.56 | 0 |
| 256 | 1882.9 | 62.7 | 3.57 | 0 |

Saturates at **~1883 tok/s aggregate at c≥64** (1.8× the SGLang node). accept_len holds
flat at ~3.56 across all concurrency — MTP doesn't degrade under load.

## Pressure test — sustained saturation (vLLM TP4)

c=256 held 150s: **0 errors**, steady **~1847 tok/s**, no preemption collapse, no KV
exhaustion. VRAM ~171/183 GB on the 4 active GPUs, ~248W. Production-stable.

## P3 — Cost vs DeepInfra

| Scenario | agg tok/s | $/M output | vs $2.50 |
|---|---|---|---|
| sustained c=256 | 1846.6 | **$6.10** | 2.44× |
| peak sweep c=256 | 1882.9 | **$5.99** | 2.40× |
| single-stream c=1 | 297.7 | $37.86 | 15.1× |

At full batch, self-host is **~$6.00/M output, still 2.4× DeepInfra's $2.50/M retail**.
DeepInfra amortizes owned hardware across multi-tenant batching; a single spot instance
can't match $/M.

**2-replica headroom (PROJECTED, not measured).** The benchmark used only **4 of 8 GPUs**
(TP4) — the other 4 sat idle. A second independent TP4 replica fits on GPUs 4–7 (each
replica used ~171/183 GB/GPU; replicas don't share GPUs), ~doubling aggregate throughput
at the **same** instance cost:

| Scenario | agg tok/s | $/M output | vs $2.50 | measured? |
|---|---|---|---|---|
| 1 replica (as tested) | 1,883 | $5.99 | 2.40× | yes |
| 2 replicas, ideal 2.0× | ~3,766 | **~$2.99** | 1.20× | projected |
| 2 replicas, realistic 1.8× | ~3,389 | ~$3.33 | 1.33× | projected |

This would bring self-host to **~infra-level parity** with retail (~$3 vs ~$2.50, and our
number carries zero margin while retail includes theirs). It does **not** change the
single-stream verdict — a replica serves *different* requests, so per-request latency
stays 297.7 tok/s. Caveat: assumes near-linear replica scaling; not validated on hardware
(node was torn down). A single TP8 replica across all 8 GPUs is the alternative — likely
better single-stream latency but ~flat $/M.

## P4 — Long context (vLLM TP4)

| tier | prompt tokens | TTFT | e2e | decode tok/s | Run 1 (SGLang) |
|---|---|---|---|---|---|
| 64k | 48,082 | 2.45s | 2.99s | 468.1 | — |
| 128k | 97,052 | 2.76s | 3.29s | 483.6 | 131.0 (3.7×) |
| 256k | 194,090 | 4.05s | 4.55s | 514.3 | 119.1 (4.3×) |

**1M context is architecturally impossible** — `max_position_embeddings=262144` (256k).
Feasible tiers (64k/128k/256k) are all stable at **468–514 tok/s decode** with no OOM;
TTFT only 2.4–4.0s. vLLM's MTP path dominates at long context too (3.7×–4.3× vs SGLang).

---

## Bottom line

- **Speed: 300 tok/s essentially MATCHED on vLLM** — decode median 297.7 (0.99×), 6/12
  prompts clear it. The first-run miss (177 tok/s) was **SGLang bug #21138**, not a model
  ceiling. accept_len 2.4→3.54 by switching engines; NVIDIA documents 4.9–5.5 so more
  headroom is plausible (lower temp, draft-length tuning).
- **Cost: self-host ~$6.00/M output vs DeepInfra $2.50/M** — retail wins ~2.4× on a
  single TP4 replica. But the test used only 4 of 8 GPUs; a projected 2nd replica on the
  idle 4 ~doubles throughput at flat cost → ~$3.00/M (1.2×), essentially infra-level
  parity (untested — node torn down before validating replica scaling).
- **Throughput: vLLM/B200/TP4 beats SGLang/B300/TP8 everywhere** (1.8× peak aggregate,
  2.4–4.9× per-workload) on **half the GPUs** — vLLM's native `nemotron_h_mtp` is the
  decisive factor.
- **Engine choice matters more than hardware here.** Same weights, same prompts: vLLM's
  MTP implementation works, SGLang's is broken for NemotronH.

See `lessons.md` for config corrections and `results/standard/*-vllm.tsv` for raw data.
First-run SGLang data retained in `results/standard/*.tsv` (non-`-vllm`) for comparison.
