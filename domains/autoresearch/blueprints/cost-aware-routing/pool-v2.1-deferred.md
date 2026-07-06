# Pool v2.1 — deferred to Phase 1.5

Designed during Phase 1a smoke (2026-05-25). **Not active in Phase 1a or 1b** — those phases use the v1 all-Bedrock pool in `configs/pool.yaml`.

## Why deferred

Pool churn at the smoke→launch threshold would have:
- Cost ~2h of operator/instance time ($19) and ~30 min of Bedrock spend on a re-run of Gate 0.2b on the 4 new workers
- Destabilized the existing smoke (which already validated the v1 pool: 43% correct iter-0, $0.001/q, healthy histogram entropy)
- Required redoing extractor audits + reward landscape with fresh per-worker output budgets

Smoke #2 (2026-05-25 15:38) showed the v1 pool is healthy enough for Phase 1a. Pool diversification is most impactful at **Phase 1.5** when we introduce **multi-step orchestration** — routing between subagents in trajectories benefits more from provider diversity + low-latency local workers than single-pick does.

## Pool v2.1 design (for Phase 1.5)

**12 workers, FP8 quantized self-hosted + Bedrock managed:**

| Ord | Code | Model | Hosting | GPU | $/M in | $/M out | Specialty |
|---|---|---|---|---|---|---|---|
| 0 | alpha | **Phi-4-mini-instruct-FP8** (3.8B) | local vLLM | shared GPU 1 | ~$0 | ~$0 | SLM cost floor (Microsoft) |
| 1 | beta | Nova Micro | Bedrock | — | $0.035 | $0.14 | Amazon SLM |
| 2 | gamma | MiniMax M2.5 | Bedrock | — | $0.30 | $1.20 | 205K ctx, agentic |
| 3 | delta | Mistral Large 3 | Bedrock | — | $0.50 | $1.50 | Multilingual |
| 4 | epsilon | DeepSeek V3.2 | Bedrock | — | $0.62 | $1.85 | Reasoning value |
| 5 | zeta | **gemma-4-26B-A4B-it-FP8** (26B/4B-active MoE) | local vLLM | GPU 2 | ~$0.001/q amortized | ~$0.001/q | Hybrid reasoning (Google) |
| 6 | eta | **Magistral-Small-2509-FP8** (24B) | local vLLM | GPU 3 | ~$0.001/q | ~$0.001/q | Reasoning specialist (Mistral) |
| 7 | theta | **Nemotron-3-Nano-30B-A3B-FP8** (30B/3B-active MoE) | local vLLM | GPU 4 | ~$0.001/q | ~$0.001/q | Tool-use (NVIDIA) |
| 8 | iota | Kimi K2 Thinking | Bedrock | — | $0.60 | $2.50 | Extended-thinking |
| 9 | kappa | Haiku 4.5 | Bedrock | — | $1.00 | $5.00 | Anthropic cheap |
| 10 | lambda | Sonnet 4.6 | Bedrock | — | $3.00 | $15.00 | Anthropic mid |
| 11 | mu | Opus 4.7 | Bedrock | — | $15.00 | $75.00 | Anthropic frontier |

**Drops vs v1**: Llama 4 Scout, Kimi K2.5, GLM-5 (replaced by self-hosted equivalents in capability axes).

**Provider mix**: 9 distinct providers — Microsoft, Amazon, MiniMax, Mistral, DeepSeek, Google, NVIDIA, Moonshot, Anthropic. Much more diverse than v1's 7 providers; better for Gate 0.4 brand-bias measurement and for cross-pool generalization (RQ#3).

## FP8 model repos (verified 2026-05-25)

| Slot | HF repo | bf16 size | FP8 size | Maintainer | Downloads |
|---|---|---|---|---|---|
| 0 | `pytorch/Phi-4-mini-instruct-FP8` | 7.6 GB | ~4 GB | PyTorch | 1.5K |
| 5 | `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` | 52 GB | ~28 GB | Red Hat AI | 244K |
| 6 | `unsloth/Magistral-Small-2509-FP8-Dynamic` | 48 GB | ~26 GB | Unsloth | 83 |
| 7 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | 60 GB | ~32 GB | NVIDIA official | 912K |

All fit on single H100 80GB at FP8 with comfortable KV cache headroom.

## Setup checklist when Phase 1.5 starts

1. `huggingface-cli download` each FP8 repo to `/mnt/nvme/cost-aware-routing/hf-cache/` (~90 GB total)
2. Spin up 4 vLLM serve instances on GPUs 1-4 (Phi-4-mini shares GPU 1 with Gemma if memory allows; otherwise standalone on a slim allocation)
3. Add `local_endpoint` field to WorkerConfig (e.g., `http://127.0.0.1:8001/v1`) and `is_local: true` flag
4. In `worker_proxy.py`: route local workers to OpenAI-compatible vLLM endpoint, Bedrock workers via Converse API
5. Re-run **Gate 0.2b parser audit** on 4 new workers × 3 datasets = 12 cells
6. Update `cost.py` cost model for self-hosted: amortize as `instance_$/hr ÷ throughput_tok/hr`
7. Update neutral code list: 11 → 12 (add `worker_mu`)
8. Re-run reward landscape; verify min adjacent gap ≥ 0.005
9. Update spec § "Worker pool" to reflect v2.1 mix

## Open questions to revisit

- Does FP8 quantization measurably hurt the math/reasoning benchmarks for Magistral / Nemotron? Spot-check 20 questions before committing.
- Phi-4-mini and Gemma-4-E4B (4B) might both be worth slotting at the cost floor — currently picking Phi only. Run a small bake-off if cost-floor diversity matters.
- For Phase 2 multi-step delegation, do we add a router-side worker (e.g., a 2nd Qwen3-8B instance) as a "decompose" specialist? Probably yes; flag for Phase 2 design.
