# Wide-EP Fault Tolerance — Prior-Art Research Brief (Grove vs Ray vs SGLang)

**Status**: Research input. Not a spec. No blueprint committed.
**Date**: 2026-06-05
**Scope**: Survey the orchestration- and engine-level fault-tolerance options for serving large MoE models with wide expert-parallelism (WideEP/DeepEP), where one logical model is sharded across a 16–128-rank data-parallel (DP) group and a single dead rank breaks the group's all-to-all dispatch/combine collectives.

> **Scope caveat — this is frontier-only.** PD disaggregation and wide-EP DP groups only pay off at frontier GPU scale; they are not default recommendations for the general customer base. This brief exists as background for *if/when* a frontier-tier wide-EP-on-K8s/Ray deployment is on the table — not as a near-term blueprint. See MEMORY `project-pd-disagg-frontier-only`.

---

## 1. The problem

WideEP shards experts across a DP group (typically 16–128 ranks; DeepSeek-V3 = 256 experts/layer, 8 activated, ~8 experts/rank over 32 ranks). The dispatch (all-to-all to expert-holding ranks) and combine (reverse all-to-all) collectives require **every rank present and healthy**. One dead rank makes the entire group — up to 128 GPUs — non-operational, and by default the router keeps sending traffic to the dead group, so all those queries fail. Spreading experts wider lowers weights/rank (bigger KV cache, larger batches) but enlarges the blast radius.

**Design guidance (vLLM large-scale serving post, via Anyscale):** decode throughput/GPU stays roughly flat across EP sizes 32/72/96 — so tune EP width to the *smallest* value that maxes throughput, because smaller groups = smaller blast radius.

## 2. Three layers, complementary not competing

| | **Grove (NVIDIA / ai-dynamo)** | **Ray Serve LLM (Ray 2.55+)** | **SGLang elastic EP** |
|---|---|---|---|
| Layer | K8s orchestration | Ray/engine control plane | Engine (in-process) |
| FT mechanism | Atomic delete + recreate of whole gang (PCS replica) after `TerminationDelay` debounce on `MinAvailableBreached` | DP group = gang; group marked unhealthy → router stops routing → torn down → healthy group rejoins; gang-aware autoscaling in group-sized increments | In-engine *elastic* EP: shrink/regrow topology without full restart |
| Recovery cost | Full cold restart of the DP group | Full group recreate (orchestration-level) | Finest-grained; no full restart |
| WideEP-specific validation | **Not confirmed** (see §3) | Yes — the blog's headline use case | Yes — engine-level |
| Generality | Engine-agnostic (vLLM/SGLang/any) | Ray Serve LLM deployments | SGLang only |

The honest read: orchestration FT (Grove / Ray gang) decides *which complete groups receive traffic*; engine FT (vLLM Elastic EP / SGLang) handles *recovery inside a group*. A real frontier deployment wants both. Ray↔vLLM Elastic EP integration is on the Ray Serve LLM roadmap, not shipped.

## 3. Grove — current state (mid-2026)

**What it is:** an open-source Kubernetes **API + operator** under `github.com/ai-dynamo/grove` (Apache-2.0, ~219 stars). **Not a scheduler itself** — it emits a `PodGang` object and delegates placement to an external scheduler; **the only supported backend today is KAI-scheduler** (Topology-Aware Scheduling needs KAI v0.14.0+). You describe a whole serving system (prefill/decode/router) as one CR and Grove handles hierarchical gang scheduling, topology-aware placement, multi-level autoscaling, startup ordering, rolling updates.

**Runtime fate-sharing — exists, generically:** in `operator/.../podcliquesetreplica/gangterminate.go`, when any PodClique / PodCliqueScalingGroup in a replica has `MinAvailableBreached=true` longer than `TerminationDelay`, the controller **deletes all pods of that PCS replica and recreates it** — atomic gang teardown + recreate. This is exactly the "one rank dies → restart the whole group" primitive, and it closes the "scheduling-time-only gang" gap of plain K8s.

**The WideEP caveat (could NOT confirm closed):** the Grove repo has **zero references to "WideEP", "DeepEP", or "expert parallel"**. The *mechanism* is generic and present, but there is **no published WideEP-specific integration or validation**. So the Anyscale framing ("Grove adds runtime gang guarantees but not yet applied to WideEP fault tolerance") still holds at the level of published evidence. Also note: Grove's recovery is a **full gang restart**, not in-place/elastic rank replacement.

**Key primitives (confirmed in repo):**
- **PodClique (`pclq`)** — pods for one role (leader/worker/frontend); like a ReplicaSet but with gang-termination + per-clique scaling.
- **PodCliqueScalingGroup (`pcsg`)** — set of PodCliques that scale/schedule as one gang; the unit `MinAvailable` is evaluated against.
- **PodCliqueSet (`pcs`)** — top-level CR; whole system, autoscaling, topology-aware spread. The PCS *replica* is the gang-terminate/recreate unit.
- **PodGang (`pg`, `scheduler.grove.io`)** — scheduler-facing object; pod groups each with a guaranteed minimum replica count.
- Hierarchical multi-level gangs (clique → scaling-group → set), explicit startup ordering, topology constraints (`pack.required`, auto multi-node-NVLink "auto-mnnvl").

**Maturity:** **pre-1.0 alpha.** Latest `v0.1.0-alpha.10-rc1` (2026-05-24); alpha.1 was Sep 2025. Apache-2.0. 2026 roadmap = resource-optimized rolling updates, topology spread constraints, automatic topology detection.

**Dynamo relationship:** Grove + KAI are an **optional** install in the Dynamo K8s Platform; Dynamo exposes topology via `topologyConstraint` on `DynamoGraphDeployment`, and Grove "is aligning its release schedule with NVIDIA Dynamo." Whether Dynamo runs on K8s *without* Grove was not definitively confirmed.

## 4. Fit for our stack

- **Grove** fits if serving is K8s-native and Dynamo-leaning — we already have `dynamo-hyperpod` and `llmd-hyperpod` blueprints, so Grove is the natural scheduler pairing there. Gives gang scheduling + runtime full-gang-restart fate-sharing today; **not** elastic rank recovery.
- **Ray 2.55 gang FT** fits if serving goes through Ray Serve LLM. Our `ray-serve-ft` blueprint is TP/GCS-FT — a *different* axis (single-replica TP fault tolerance via ElastiCache), not DP-group gang FT.
- **SGLang elastic EP** is the finest-grained (no full restart) but SGLang-only.

## 5. Open questions / unverified

- No WideEP/DeepEP-specific FT support found in Grove code or docs — the "not yet applied to WideEP" gap is **not confirmed closed**; only the generic mechanism is confirmed.
- Whether Grove's recreate preserves placement/topology on retry, and the `TerminationDelay` defaults, were not traced.
- Whether Dynamo can run on K8s without Grove was not definitively confirmed.
- Anyscale blog claims are relayed from a summary fetch, not a verified quote of the underlying tables.

**Primary sources:** `github.com/ai-dynamo/grove` (README, `docs/installation.md`, `operator/.../gangterminate.go`, `operator/api/core/v1alpha1/podcliqueset.go`, releases); `github.com/ai-dynamo/dynamo` `docs/kubernetes/grove.md`; Anyscale blog "DP group fault tolerance for vLLM WideEP on Ray Serve LLM" (Ray 2.55).
