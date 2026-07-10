# llm-d HyperPod LMCache — Progress

**Date**: 2026-07-10
**Spec**: `domains/gpu-serving/specs/llmd-hyperpod-lmcache.md`
**Region**: `us-west-2` — **reusing the existing dynamo-hyperpod-lmcache infra** (no new cluster)
**Hardware**: `ml.g6e.xlarge` (1 GPU) — the resolved SKU from the dynamo run
**Model**: `Qwen/Qwen3-0.6B`

## Reused Infrastructure (verified live 2026-07-10)

| Resource | Value |
|---|---|
| HyperPod cluster | `dynamo-hp-lmcache-west-20260709` (InService) |
| EKS cluster | `qn-sglang-eks-cluster` (K8s 1.32 — meets ≤1.32 requirement) |
| GPU node | `hyperpod-i-0ebd814e4a113cdce` @ `10.2.37.31`, instance group `g6e-workers` |
| ai-toolkit daemon | `aws-hyperpod/ai-toolkit-lqrj6` @ `:9200`, shm `ai_toolkit_cache` (1 GiB), running 21h |
| Tiered storage | Enabled (20%) |
| GPU freed | `dynamo-lmcache-worker` scaled to 0 (dynamo run PASS/archived); 1 GPU allocatable |
| Gateway stack | **NONE present** — no Gateway API / Istio / InferencePool / Envoy CRDs. Stage 2 must install from scratch. |
| kubeconfig | `aws eks update-kubeconfig --region us-west-2 --name qn-sglang-eks-cluster` |

**Single-GPU constraint**: only 1 GPU on this node → Stage 6 (two-replica same-node sharing) will be SKIPPED with reason. The store→restart→replay proof (Stage 5) is single-replica and unaffected.

## Status

| Stage | Result | Notes |
|---|---|---|
| Stage 0 — Carryover Audit | COMPLETE | Spec written and audited; prior `llmd-hyperpod` L2 config + `dynamo-hyperpod-lmcache` determinism knobs carried forward. |
| Stage 1 — HyperPod Tiered Storage Discovery | COMPLETE | Reused dynamo infra: tiered storage enabled, ai-toolkit daemon running on `hyperpod-i-0ebd814e4a113cdce` @ 10.2.37.31:9200, GPU freed. See Reused Infrastructure table above. |
| Stage 2 — llm-d Gateway Stack | PASS | GAIE v1.5.0 CRDs + llm-d-router-standalone v0.9.0 (tiered-prefix-cache). Chart provisions EPP SA+RBAC+InferencePool+Envoy. Router co-located on GPU node (cross-node networking gap workaround). |
| Stage 3 — llm-d vLLM Baseline (L0) | PASS | Qwen3-0.6B via llm-d modelservice; `/v1/completions` HTTP 200 through router→EPP→vLLM (300ms); full GPU+CPU prefix-cache scorer set active. |
| Stage 4 — LMCache Connector to HyperPod L2 | PASS | `LMCacheConnectorV1` opens `ai_toolkit_cache`, connects to `sagemaker-hyperpod://10.2.37.31:9200`, and stores: `Stored 743/743 tokens, 8.0 GB/s`. Required type:File shm mount + daemon-recreate sequence. |
| Stage 5 — Store→Restart→Replay L2 Hit Proof | PASS | **Core deliverable.** Fresh pod after restart: `external_prefix_cache_hits_total 0→742`, `Retrieved 743/743 tokens`, hit rate 99.9%, latency 3375ms→379ms (8.9x). Artifact: `results/e2e-telemetry-llmd-l2-proof-20260710.json`. |
| Stage 6 — Two-Replica Same-Node Sharing | SKIPPED | ml.g6e.xlarge has a single GPU; cannot place two GPU replicas on the node. |
| Stage 7 — Optional FSx L3 | NOT RUN | Optional; L2 is the deliverable. |

## Current Decisions

- (record region / SKU fallback reason here)
- (record llm-d chart versions + image tag + LMCache version vs ai-toolkit daemon version here)

## Outcome

**PASS** (2026-07-10) — llm-d on HyperPod using the ai-toolkit managed L2 daemon via LMCache, proven
by a store→restart→replay L2 hit (742 external hits on a fresh pod, 99.9% hit rate, 8.9x latency
speedup). This is the llm-d twin of the dynamo-hyperpod-lmcache result, on the same reused cluster.

Key adaptations vs the original spec (upstream llm-d restructured since the spec was written):
- Gateway stack is now GAIE v1.5.0 CRDs + `llm-d-router-standalone` v0.9.0 Helm chart (tiered-prefix-cache
  values), not the old helmfile/`ms-values` path. The chart auto-provisions EPP SA+RBAC+InferencePool+Envoy.
- Worker image is `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` (has the SageMaker HyperPod LMCache
  adapter), not stock `vllm/vllm-openai` or `llm-d-cuda`.
- Router co-located on the GPU node to bypass a cross-node pod-networking gap.
- `type:File` shm mount + daemon-recreate sequence required (see lessons.md).

## Follow-ups (not blocking)

1. Cross-node pod networking (HyperPod GPU node ↔ vanilla EKS nodes) — open the HyperPod node SG to the
   EKS pod CIDR so the router doesn't have to co-locate on the GPU node.
2. Stage 6 two-replica sharing needs a multi-GPU SKU (g6e.12xlarge+) to validate cross-pod L2 reuse.
3. Optional Stage 7 FSx L3 tier.
