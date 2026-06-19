---
name: manage-constraints
description: View, query, add, or demote constraints in the kernel optimization constraint database. The database is append-only — constraints are never deleted, only demoted in severity if contradicted by evidence. Use to inspect what the agent has learned, manually inject domain knowledge, or query relevant constraints before generating a candidate.
---

# Manage Constraints

## Overview

The constraint database is the compound learning mechanism — the kernel optimization equivalent of the learned-verifier's signal combination. It accumulates institutional knowledge from every candidate evaluation and injects it into future generations to prevent repeat failures and accelerate convergence.

## Database Format

Stored at `results/constraints.jsonl`. Each line:

```json
{
  "id": "C042",
  "timestamp": "2026-05-15T14:30:00Z",
  "type": "hard|soft|positive|meta",
  "severity": "permanent|high|medium|low|demoted",
  "region": "moe_dispatch|mla_decode|megakernel|pipeline|general",
  "rule": "Human-readable constraint text",
  "evidence": "What triggered this constraint (candidate IDs, measurements)",
  "injection_text": "Exact text injected into generation prompts when this constraint is active",
  "source": "l0|l1|l2|l3|l4|manual|upstream",
  "contradicted_by": null,
  "applied_count": 14,
  "last_applied": "2026-05-15T15:00:00Z"
}
```

## Constraint Types

| Type | Severity | Injection | Can Override? |
|------|----------|-----------|---------------|
| **hard** | permanent | Always injected for matching region | No — architecture/hardware facts |
| **soft** | high/medium | Injected with "prefer to avoid" preamble | Yes, if agent justifies in header comment |
| **positive** | medium | Injected as "this approach worked" suggestion | N/A — guidance, not restriction |
| **meta** | low | Injected into state vector decisions | Yes — strategy guidance |

## Commands

### Query constraints for a target region

```
/manage-constraints query --region moe_dispatch
```

Returns all active constraints for that region, sorted by severity.

### Add a manual constraint (domain knowledge injection)

```
/manage-constraints add --type hard --region moe_dispatch \
  --rule "K2.6 has 384 experts with n_group=1 — top-8 selected from full pool without group constraints" \
  --evidence "config.json: n_routed_experts=384, n_group=1, topk_group=1" \
  --source manual
```

### Demote a constraint (contradicted by new evidence)

```
/manage-constraints demote C042 --reason "Candidate 067 succeeded with tile_k=128 after adding explicit L2 flush"
```

Constraint is never deleted — `contradicted_by` field updated, severity set to "demoted."

### View learning rate

```
/manage-constraints stats
```

Returns:
- Total constraints: N (hard: X, soft: Y, positive: Z, meta: W)
- Constraints added per decile of candidates
- Average injection size (tokens)
- Most-applied constraints (top-5)
- Demoted constraints count

## Pre-Seeded Constraints (Phase 1)

Before any candidates are generated, seed the database with:

### Hardware (hard, permanent)
- SM103: 227KB shared memory per block
- SM103: 2.4 TB/s peak HBM bandwidth
- SM103: ~2400 TFLOPS FP8 peak
- SM103: Full TMA support, TCGEN5 available
- SM103: 128 registers per thread default
- B300: 8 GPUs via NVLink 5 (full bisection BW)

### Architecture (hard, permanent)
- K2.6: 384 routed experts, 8 active per token, 1 shared expert
- K2.6: n_group=1, topk_group=1 (no expert grouping)
- K2.6: moe_intermediate_size=2048 per expert
- K2.6: scoring_func=sigmoid, topk_method=noaux_tc
- K2.6: num_attention_heads=64, kv_lora_rank=512
- K2.6: qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
- K2.6: INT4 QAT quantization (not post-hoc)

### Known Issues (soft, from upstream PRs)
- FlashMLA accuracy regression on some models with cuDNN backend (vLLM #41490)
- FP8-MMA tile-content sensitivity on B200 (vLLM #41480) — may apply to B300
- DeepGEMM contiguity assertion with MTP (SGLang fix merged)

## Convergence Measurement

The constraint database enables measuring learning rate:
- **Early candidates** (1-20): Many L0/L1 failures → many hard constraints added
- **Mid candidates** (20-60): Fewer compile failures, more L3/L4 data → soft + positive constraints
- **Late candidates** (60+): Constraint injection narrows search space → faster promotion

Track `candidates_per_promotion` per decile. Target: ≥1.5x acceleration (last decile vs first decile).

## When NOT to Use

- To manually edit kernel code (use `/generate-candidate`)
- To run benchmarks (use `/verify-kernel`)
- To profile hardware (use `/profile-kernel`)
