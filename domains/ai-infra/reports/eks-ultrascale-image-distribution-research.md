# EKS Ultra-Scale & Large-Scale Image Distribution for an Agent-Sandbox Workload

**Research date:** 2026-06-27. **Knowledge-cutoff caveat:** assistant training cutoff is January 2026; several primary sources below (EKS Provisioned Control Plane, SOCI v0.13+, Kubernetes 1.35 scheduler batching) are at or past that cutoff and were fetched live — re-verify version-specific behavior on the actual cluster version before relying on it. Tone is deliberately skeptical: where a number is a vendor benchmark rather than a contractual SLA, it is flagged.

## Workload context (the thing being designed for)

Periodic batch "experiment" runs. Each run creates **~80,000 short-lived sandbox pods now, ~100,000 soon** — CPU task pods, **not GPU training, not the k8s `Sandbox` project**. Three-layer image model:

- **L1** — OS + system deps (shared by nearly everything).
- **L2** — toolchain version: **a few hundred distinct base images** across all languages.
- **L3** — per-task deps + source: **the 80k, each ≤200 MB**, each built **FROM one of the few hundred L2 bases**.

Budget: **sub-second pod ("sandbox") creation**. Images prepared **hours in advance**. Between runs mostly only L3 changes; previous-run images can likely be deleted when prepping the next run. They are considering **sharding across multiple EKS clusters by language** to reduce images-per-cluster.

---

## 1. Executive summary

- **EKS ultra-scale = a re-architected etcd, not a new scheduler and not a pod-throughput product.** AWS offloaded etcd consensus from Raft to an internal "journal," moved BoltDB to in-memory tmpfs, and partitioned the keyspace across multiple etcd clusters. This raises **control-plane / datastore headroom** (node count, object count, write QPS), targeting 100K-node accelerator fleets. It does **not** rewrite kube-scheduler. (AWS Containers blog, 2025-07-16.)
- **It barely touches your real concern.** The "500 pods/sec" figure is a **scheduler-tuning** result (plugin tailoring + node-scoring params), achieved *alongside* the etcd work, not *because* of it. The general-purpose, non-GPU-gated form is **Provisioned Control Plane** (re:Invent 2025, 2025-11-27), whose top published tier commits **400 pods/sec** of scheduling. At 400/sec, 80k–100k pods take **~3.3–4.2 minutes to bind** regardless of how fast any single pod starts. Scheduling rate, not etcd, is your binding constraint — and it's a *throughput* problem distinct from the *per-pod sub-second start* problem.
- **SOCI probably does NOT solve your L3 problem, by AWS's own reasoning.** SOCI lazy-load only wins when *most image bytes are never read at startup* (Slacker FAST '16: ~6.4% of bytes touched; independent studies span 1–40%). Your L3 = per-task deps + source that the task reads immediately → high byte-access ratio → lazy-load adds FUSE + on-demand-fetch latency and tends to break even or regress. AWS explicitly steers AI/ML-style "bundled code+deps" images to **Parallel Pull** (a full, parallelized download), not lazy-load. (AWS EKS blog, 2025-08-27.)
- **The real bottleneck is registry pull fan-out for the 80k DISTINCT L3 images, and no cache/P2P mechanism reduces it.** Distinct content (by digest) must transit from origin exactly once; caches and P2P only de-dupe *repeated* pulls of the *same* content. ECR's documented pull-API quotas (BatchGetImage 2,000/s, GetDownloadUrlForLayer 3,000/s, GetAuthorizationToken 500/s — all per-region, per-account, adjustable) will throttle a near-simultaneous 80k-distinct-pull burst unless quotas are raised **and** pulls are staggered.
- **Recommendation: split by layer.** Pre-warm the bounded **L2 base set** onto every node before the run (EBS-snapshot Bottlerocket data volume via Karpenter, or Spegel/P2P) so the bulk of each image's bytes are already local. Pull only the small **distinct L3 delta** just-in-time via **SOCI parallel-pull-unpack** to NVMe, with **raised ECR quotas + staggered scale-up**. Don't expect P2P or pull-through caching to help the distinct tail. Multi-cluster sharding helps the *control-plane/scheduler* axis, not image bytes (see §7).

---

## 2. EKS ultra-scale — facts, GA status, what it changes vs doesn't

**Primary sources:** "Under the hood: Amazon EKS ultra scale clusters," AWS Containers blog, **2025-07-16** (https://aws.amazon.com/blogs/containers/under-the-hood-amazon-eks-ultra-scale-clusters/); "Amazon EKS introduces Provisioned Control Plane," **2025-11-27** (https://aws.amazon.com/blogs/containers/amazon-eks-introduces-provisioned-control-plane/); InfoQ, **2025-09-03** (https://www.infoq.com/news/2025/09/aws-eks-kubernetes-ultrascale/).

**What was re-architected — etcd internals, NOT a move off etcd.** AWS kept etcd's API semantics (so upstream Kubernetes is unchanged) and changed three things:

1. **Consensus offloaded from Raft to an internal "journal"** — described as a component "we've been building at AWS for more than a decade," removing the quorum requirement and peer-to-peer etcd communication so replicas scale freely. (Contrast GKE, which genuinely replaced etcd with a Spanner-based store to reach 65K–130K nodes — per etcd maintainer commentary, GitHub etcd-io discussion #20687, Sep 2025, and Google's 130K-node blog, 2025-11-21.)
2. **BoltDB moved from EBS to in-memory tmpfs**, max DB size doubled to 20 GB (≈32 GB aggregate across partitions in test).
3. **Partitioned keyspace** — hot resource types (nodes, pods, leases, events) split into separate etcd clusters → "up to 5× write throughput."

**A large share of the headline gain is upstream, not AWS-proprietary.** KEP-2340 "consistent reads from cache" (beta in K8s 1.31) serves consistent LISTs from the apiserver watch cache instead of etcd quorum reads (KEP cites 2–10× CPU reduction, 20–50× latency reduction; >80% of LISTs served from cache in production). Snapshottable API server cache (beta, default-on in **1.34**, blog 2025-09-09) further offloads paginated/historical LISTs. The etcd maintainer (GKE-employed) argued in discussion #20687 that the Raft replacement "played a smaller role than headlines suggest."

**GA / gating status — two distinct products:**

- **Ultra-scale clusters** (the 100K-node capability): **new-cluster-only, opt-in, not retrofittable**. Framed as standard/full-conformance but no literal "GA" stamp; oriented to accelerator fleets (Trainium/GPU, HyperPod). Not *formally* GPU-restricted, but the entire framing is AI/ML.
- **Provisioned Control Plane** (re:Invent 2025, 2025-11-27): the **generally usable, non-GPU-gated** form. Opt-in with tiered pricing, **available for new AND existing clusters on K8s 1.29+** (one parameter, no migration), provisionable via Console/CLI/eksctl/CFN/Terraform. Explicitly for "all types of workloads" (multi-tenant SaaS, web apps cited). **This is the path a CPU sandbox-pod workload would actually use.**

**Demonstrated scale (benchmarks on test clusters — NOT contractual SLAs):** >10M Kubernetes objects, 100K nodes, 900K pods; DNS 1.5M QPS p99 <1s; 1,000 degraded nodes replaced in <5 min; API latencies within the standard Kubernetes SLOs (1s get/write, 30s list). AWS explicitly calls the comparison numbers "directional indicators rather than absolute limits."

**Provisioned Control Plane published tiers** (closest thing to commitments — they carry pricing):

| Tier | API concurrency | Pod scheduling | etcd DB | Price |
|------|-----------------|----------------|---------|-------|
| XL | 1,700 | 167 pods/s | 16 GB | $1.65/hr |
| 2XL | 3,400 | 283 pods/s | 16 GB | $3.40/hr |
| 4XL | 6,800 | 400 pods/s | 16 GB | $6.90/hr |

Above 4XL → "contact your AWS account team." A Standard-vs-4XL comparison shows Standard caps ~5,000 nodes / 80,000 pods / 500 deployments / 500 jobs; 4XL ~40,000 nodes / 640,000 pods / 40,000 deployments / 40,000 jobs.

**Is 100K nodes even the right axis?** No — your workload is **pod-count- and pod-churn-heavy, not node-count-heavy**. Upstream Kubernetes' tested envelope is 5,000 nodes / **150,000 pods cluster-wide** / 150,000 objects per resource type. So **80k–100k concurrent pods is within stock upstream limits on a 5,000-node-class cluster** — you do not need the 100K-node ultra-scale path for pod *count*. What you might need beyond Standard EKS is headroom for **pod churn** (short-lived pods at high create/delete rate stress etcd writes on the pods/leases/events keyspaces and stress the scheduler) — exactly where Provisioned Control Plane's higher scheduling rate + partitioned/in-memory etcd, plus the upstream cache KEPs, help.

**Bottom line:** ultra-scale raises datastore/control-plane headroom; it does **not** change kube-scheduler's serial-binding model. For this workload, **Provisioned Control Plane (for churn/QPS headroom) is relevant; the 100K-node capability is not.**

---

## 3. Scheduler throughput — the "500 pods/sec" number explained, levers to exceed it

**Source of "500 pods/sec":** it is **AWS's tuned-scheduler result at 100K nodes**, from the EKS ultra-scale blog (2025-07-16), achieved by "tailoring scheduler plugins" and "optimizing node filtering/scoring parameters." Corroborated by InfoQ (2025-09-03). It is **not** a generic kube-scheduler constant.

**What it measures:** the **kube-scheduler component's binding rate** — the rate at which the scheduler makes and commits pod→node decisions. The canonical upstream metric is `SchedulingThroughput` from `scheduler_perf` ("scheduled pods per second"). It is **not** end-to-end pod-start latency and **not** raw control-plane/etcd QPS, though it is coupled to them: each bind is a mutating API write that must persist to etcd under the 1s mutating-latency SLO.

**Honest baseline for an untuned upstream scheduler: ~200–450 pods/sec**, not 500:
- Godel/kubewharf benchmark (KWOK, 30,000 nodes, K8s 1.29): standard kube-scheduler **~300 pods/s avg, ~430 peak** (Godel with 3 shards hit ~1,000). (DeepWiki, indexed 2025-04-28.)
- Preferred Networks `scheduler_perf` (2025-12-08): standard scheduler **~234–267 pods/s**.
- Integration-bench numbers (~100 pods/s) are an **API-server-QPS-capped harness artifact** (the bench caps client QPS at 100), not the algorithmic limit — don't conflate.

**Why it's serial:** the Scheduling Framework runs **scheduling cycles serially** (pick a node) but **binding cycles concurrently** (apply the decision). AWS: the scheduler "processes pods serially… making its throughput inherently latency-bound," and worse on large clusters because there are more nodes to score.

**Levers to raise/bypass it (for an 80k–100k short-lived-pod burst):**

| Lever | Effect | Honest verdict |
|-------|--------|----------------|
| `percentageOfNodesToScore` ↓ | Score fewer nodes/pod → higher throughput, worse placement | Most-cited knob. Default linear (50% @100 nodes → 5% floor). Docs warn not below 10% unless throughput is critical and scoring quality isn't. **This is how AWS got to 500.** |
| Scheduler `parallelism` ↑ | Parallelizes Filter/Score *within one pod's cycle*, not across pods | Helps large clusters; does **not** break the serial-pod ceiling. |
| Opportunistic Batching | Caches filter/score across cycles (0.5s expiry) | Beta in **K8s 1.35**, default-on — past cutoff, verify on your version. |
| Multiple scheduler **profiles** | Policy mechanism (`schedulerName` per pod) | **Not a throughput lever** — still one serial scheduler. |
| Multiple/**sharded** schedulers | Aggregate throughput ↑ (Godel: 3 shards ≈ 1,000/s) | **Not a supported upstream feature.** Safe only if partitioned by disjoint node pools / pod sets; two default schedulers over the same nodes race → overcommit. |
| **Kueue** | Job/workload **admission** gating (quota, fair-share, gang) | Does **not** raise binding throughput — it delegates binding to kube-scheduler. GKE's 1,000 pods/s run used Kueue *on top of* the default scheduler. |
| **Volcano** | Replacement batch scheduler, gang semantics | Vendor claims "up to 1,000 pods/s" (Huawei CCE — unverified marketing). Neutral AKS test: gang adds only ~7–9s end-to-end. **But gang mode REDUCES raw throughput** (`scheduler_perf`: ~3–20 pods/s gang vs ~234 plain). Wrong lever for independent short-lived pods. |
| **Pre-assign `pod.spec.nodeName`** | Bypasses the scheduler entirely | **Highest throughput, but bypasses everything**: no resource-fit, no taints/tolerations, no affinity, no binding cycle. You implement placement yourself. Bad nodes → OutOfcpu/OutOfmemory; documented failure where K8s "continually tries to spin up thousands of pods" on a pressured node. Admission webhooks + the pod CREATE etcd write still fire, so you escape the scoring loop, not the control-plane write ceiling. This is effectively what custom batch schedulers do internally. |

**Does ultra-scale change the number?** No — scheduler is orthogonal to the etcd re-architecture. The etcd work *raises the headroom* so binds aren't choked by the datastore; the 500 is a scheduler-tuning result. Provisioned Control Plane lists pod-scheduling rate (167/283/400) as a **separate provisioned dimension** from API concurrency and etcd size, confirming the independence. (Anthropic, per the re:Invent CNS429 session, runs a second in-house Rust scheduler "Cartographer" that "scales by number of workloads, not pods" — a customer-side optimization, not part of EKS.)

**Is the scheduler even your bottleneck? And is sub-second realistic?** Be skeptical of the sub-second budget. The SIG-scalability **pod-startup-latency SLO is 99p ≤ 5s per cluster-day, and it explicitly EXCLUDES image pull and init-container time, covers only stateless pods, and assumes nodes are already Ready** (https://github.com/kubernetes/community/blob/master/sig-scalability/slos/pod_startup_latency.md; mirrored at https://docs.aws.amazon.com/eks/latest/best-practices/kubernetes_upstream_slos.html). GKE's 130K-node run reported **p99 pod startup ≈ 10s** even at hyperscale. So:
- **Sub-second per-pod start is unrealistic if it includes image pull** — the SLO is 5s and explicitly excludes pull, because pull dominates and is environment-dependent. Sub-second is only plausible when the image is already local (pre-warmed) and the container is small/fast-starting.
- The **throughput** problem (binding 80k–100k pods at ~300–500/s = minutes) is separate from the **per-pod latency** problem.
- For high-churn pods the binding constraints, in order: (1) image pull / kubelet startup — usually dominant; (2) control-plane mutating-write rate (every CREATE + Bind is an etcd write; AWS's own scalability test used 50 pods/s churn to stress 5,000 nodes / 170,000 pods, blog 2024-01-31); (3) scheduler binding rate.

---

## 4. SOCI — mechanics, EKS support, when it helps/doesn't, the byte-access dependency

**Sources:** awslabs/soci-snapshotter README + docs (fetched 2026-06-27); "Under the hood: Lazy loading container images with SOCI and AWS Fargate," 2023-07-18; "Introducing Seekable OCI Parallel Pull mode for Amazon EKS," **2025-08-27**; Slacker, USENIX FAST '16.

**Mechanics (lazy-load mode):** SOCI is a containerd remote snapshotter. Given a **SOCI index** (built at image-prep time), it mounts a **FUSE filesystem per layer**, and on a file read not yet cached it maps the file offset → compressed **spans** (via per-layer zTOC: a TOC of file offsets + zInfo gzip checkpoints, classic seekable-gzip), issues **HTTP range GETs** for only those spans, decompresses just those, and caches them. A **background prefetch** then pulls the full image anyway. No index → graceful fallback to a normal full download.

**SOCI v2 (the big 2024–2026 change):** Index Manifest v2 landed in **soci-snapshotter v0.10.0 (2025-06-27)** and is now default. It **reverses the original "no conversion" design** — it requires a lightweight `soci convert` step, but layers are *shared* with the original image (index adds a few KB). Correction to a common misconception: **referrers/`subject` is the v1 discovery model**; v2 deliberately moves *away* from referrers, bundling the index into a single immutable multi-arch image. A separate **Parallel Pull/Unpack** mode (v0.11.0, 2025) does a *full* parallelized download — **no index, no FUSE, no enumeration**. Latest release v0.14.1 (2026-06-12); still 0.x, no formal GA.

**EKS support — DIY and rough for lazy-load; partial AMI integration for parallel-pull:**
- The project itself says "SOCI is not included in EKS-optimized AMIs — it must be installed DIY," via a launch-template boot script + systemd service, **AL2023 only** (uses `nodeadm`). Containerd needs a `proxy_plugins.soci` snapshotter, `snapshotter = "soci"`, `disable_snapshot_annotations = false`, a CRI keychain, containerd ≥1.7.16, FUSE. K8s isn't snapshotter-aware → SOCI must be used for **all** containers on the node (incl. `pause`).
- AWS's 2025-08-27 blog claims "recent versions of the AL2023 and Bottlerocket EKS-optimized AMIs have SOCI **Parallel Pull** Mode built in" — but gives no AMI versions and it is **not on by default** (enable via NodeConfig/Bottlerocket settings). **Native Bottlerocket support is recent — v1.44.0 (2025-08-04)** — and is **parallel-pull, not lazy-load**. There is **no EKS managed add-on**.
- Fargate has had true auto-detected lazy-load since 2023, but that's a different runtime than EKS-on-EC2.

**The byte-access-ratio dependency (the whole game):**
- **Slacker (FAST '16):** pulling packages is 76% of container start time but only **6.4% of that data is read** at startup (median ~20 MB of a 329 MB image). This single 2016 study is the load-bearing assumption behind every lazy-loading system.
- Independent corroboration spans **1–40%, not a clean 6.4%**: Starlight (NSDI '22) "<1% of files, 1–39% of data"; FaaSNet (ATC '21) ~16% fetched.
- AWS SOCI-specific speedups (PyTorch 129s→60s ~50%; Flywire >60% on >750 MB; Autodesk 50%) are **all vendor/customer figures on large, read-light images** — no independent published SOCI benchmark exists.

**When SOCI does NOT help (this is your case):**
- AWS's own thresholds: "**large (>250 MB)** images see the greatest benefit"; below that "the initial lazy loading overhead may be greater"; "**no benefit if the workload needs to access all the image data quickly**." Your L3 ≤200 MB sits near/below the lazy-load threshold, and per-task deps+source are read immediately.
- AWS's explicit AI/ML position (2025-08-27): bundled code+deps+data "makes the complete image download inevitable regardless of lazy loading" → **use Parallel Pull, not lazy-load.**
- Independent negatives: a 2026-02-08 benchmark — lazy-pull made nginx readiness **20× slower** (a plain local registry beat it on total readiness); Grab's production eval (2026-01-21) — SOCI merely **matched** overlayFS (no gain). FUSE overhead: "To FUSE or Not to FUSE" (FAST '17) measured up to −83% / ~3× slower; SOCI snapshotter has reported 2.6+ GB cgroup memory with no containers running (issue #1850) and an OPEN regression tracker (#555) naming images SOCI makes *slower*.
- **Registry becomes a runtime hot path:** every uncached file access is a live HTTP request to the registry — a container "can fail an hour after start when a rare code path hits an uncached file." For 80k–100k images lazy-fetching from ECR *during* the run, this is read-amplification and an operational risk, not just a pull-time concern.

**Index-generation cost (you have hours):** defaults `--min-layer-size` 10 MiB, `--span-size` 4 MiB; `soci create` builds sparse zTOCs (offset/checkpoint tables, not content copies), parallelizable across layers, dedupes existing digests. **No official per-image timing benchmark.** AWS's automated **SOCI Index Builder** (EventBridge→Lambda per ECR push) caps at **Lambda 900s / 1024 MB / 10 GB ephemeral, ≤6 GB compressed image**; for larger, run `soci create` in CodeBuild. No published "thousands of images" case study — you'd have to measure.

**Verdict:** SOCI **lazy-load is likely the wrong tool for your L3** (≤200 MB, read-heavy). SOCI **parallel-pull-unpack** (eager, parallel, no index/enumeration, shipped in the AMIs) is the relevant SOCI feature — gate any lazy-load consideration on first measuring your actual startup byte-access ratio; if it's not single-to-low-double-digit percent, don't.

---

## 5. Image distribution at scale — ECR limits, Spegel, Dragonfly, pre-pull, comparison

**The load-bearing fact:** the workload splits into two layer populations demanding **opposite** mechanisms:
- **L2 shared bases** (few hundred, near-universal fan-out): pre-warm and P2P both work great.
- **L3 distinct tops** (80k–100k, ~1 consumer each, ~16–20 TB aggregate): P2P and pull-through caching do essentially **nothing** — distinct content must transit origin once regardless. Only lazy-load or just-in-time distinct pulls help, and **registry/transport throughput is the constraint**.

**ECR pull-API quotas** (https://docs.aws.amazon.com/AmazonECR/latest/userguide/service-quotas.html, 2026-06-27 — per-region, per-account, **adjustable**): BatchGetImage **2,000/s** (manifest, 1×/pull), GetDownloadUrlForLayer **3,000/s** (1× per uncached layer), GetAuthorizationToken **500/s** (only on token expiry; tokens valid 12h). Crucial mechanic: `GetDownloadUrlForLayer` returns a **pre-signed S3 URL — the layer bytes are served from S3 and do NOT count against ECR API quotas.** So API pressure scales with **(pulls × uncached layers), not bytes**. For distinct images, every layer is "uncached" → full layer-count `GetDownloadUrlForLayer` per image → a near-simultaneous 80k-distinct-pull burst (e.g. 20k nodes × 8 layers in ~1–2s) trivially exceeds 2,000/s and 3,000/s → **429 ThrottlingException** on the API calls (not the S3 download). Spreading the same total over more seconds stays under quota. **ECR's "max pulls/sec" is never published as a single number** — only the per-API quotas. **ECR pull-through cache** removes the *upstream* registry's limits but does **NOT** raise your own ECR quotas (cached repos count like any other), and caps at **50 rules/registry (hard limit)**.

**Spegel** (P2P, layer-level, cache-only; spegel.dev, github.com/spegel-org/spegel, latest v0.7.2 2026-06-18): stateless per-node DaemonSet, uses containerd's existing content store + libp2p Kademlia DHT to advertise digests already present on a node; containerd points at `localhost` mirror, miss → fallback to upstream. **No prefetch.** EKS needs config (AL2023: disable `discard_unpacked_layers` via nodeadm; Bottlerocket ≥1.56: bootstrap mirror config). Verdict: **big win for L2 shared bases** (pulled once, served P2P to all peers), **~zero for distinct L3** (no second consumer). Skeptical flag: project self-describes as "evolving API… home lab and individual contributor use case" — not enterprise-hardened; CNCF-sandbox status unverified; performance claims qualitative only.

**Dragonfly** (CNCF **Graduated** 2025-10-28; dragonflyoss/dragonfly, v2.5.0 2026-06-25 — very fresh): Manager + Scheduler + Seed Peers + dfdaemon peers; piece-level multi-source swarming; **proactive preheat** API (`single_seed_peer`/`all_seed_peers`/`all_peers`, targetable by ips/percentage); **Nydus lazy-load** integration. Verdict: reactive P2P has the **same L3 limitation as Spegel**. Its one differentiator is **targeted preheat** — could pre-stage a specific distinct image onto the specific nodes that will run it — but only if you know placement in advance (couples your pipeline to the scheduler), still pulls each distinct image from origin once, and is wasteful for low-fan-out. Heavyweight (scheduler/manager). Nydus lazy-load overlaps with SOCI. **Conditional and operationally heavy for distinct images.**

**In-cluster mirrors (registry:2 / Harbor):** pull-through cache only helps on the **2nd+ pull of the same digest**; each distinct digest is fetched upstream exactly once. For L3: barely helps, **adds a hop**, concentrates fan-in (SPOF on RWO EBS for registry:2), and if it retains everything that's **~16–20 TB** (80k–100k × 200 MB ≈ $1.6–2k/mo at $0.10/GB-mo). What it *does* buy for L3: co-locating fan-out in-VPC/in-AZ (same-region ECR transfer $0.00; cross-AZ ~$0.02/GB) and absorbing retry/rate-limit storms — not byte reduction.

**Pre-pull / pre-warm patterns:**
- **EBS-snapshot-with-baked-images (Bottlerocket data volume)** — "Reduce container startup time on Amazon EKS with Bottlerocket data volume," AWS, **2023-10-19**: bake images into the containerd store on `/dev/xvdb`, snapshot it, boot nodes with a volume restored from the snapshot → images present at boot. Reported **49s → 3s** for a 4.93 GB image. Requires `imagePullPolicy: IfNotPresent`. Wired via **Karpenter `EC2NodeClass.spec.blockDeviceMappings[].ebs.snapshotID`** (+ `volumeInitializationRate` knob for the snapshot first-touch penalty) or MNG launch template. Tooling: `aws-samples/bottlerocket-images-cache`. AWS recommends this in the EKS AI/ML best-practices page.
- **Custom AMI baking:** works but AWS least-recommends for large/frequently-updated content (couples to AMI patch cycle).
- **kube-fledged:** ImageCache CRD, pre-pulls via Jobs — **ABANDONED** (last release 2022-10-21, no commits 2024–2026). Don't adopt.
- **DaemonSet pre-pull:** documented by AWS but has scale-out races and GC eviction, and **requires enumerating images** → unworkable for 80k distinct.

### Comparison table — mechanism × layer

| Mechanism | Optimizes | Helps SHARED L2 | Helps DISTINCT L3 | EKS support | Cost / caveats |
|-----------|-----------|-----------------|-------------------|-------------|----------------|
| ECR quota ↑ + staggered pulls | Avoid 429 on pull APIs | n/a | **Yes — the direct lever** for JIT distinct pulls | Native | Service Quotas request; stagger; tune kubelet `registryPullQPS`/`registryBurst` (default 5/10); no byte reduction |
| ECR pull-through cache | Remove *upstream* limits | Yes | No (miss per distinct image; ~16–20 TB) | Native | 50 rules/registry (hard); doesn't raise own ECR quotas |
| In-cluster mirror (registry:2/Harbor) | De-dupe repeats; co-locate fan-out | Big win (N:1) | No byte reduction; adds hop; SPOF; ~16–20 TB | Yes (hosts.toml) | registry:2 RWO EBS single-writer; Harbor ~8 services |
| Spegel (P2P, layer) | Reduce upstream egress for shared layers | **Big win, distributed** | **~Zero** | Requires config | Stateless DaemonSet; "home lab" self-framing; CNCF status unverified |
| Dragonfly (P2P piece + preheat) | Egress reduction + preheat + Nydus | Big win | Reactive: no. Targeted preheat: conditional/heavy | Yes (dfdaemon) | CNCF Graduated; heavyweight; v2.5.0 very fresh |
| EBS snapshot data volume (Bottlerocket) | Images present at boot | **Strong win** (bounded) | **Impractical** (~16–20 TB; per-node bloat) | Native (Karpenter snapshotID) | Re-bake per base-set change; must enumerate; IfNotPresent |
| Custom AMI baking | Images present at boot | Yes (bounded) | Impractical | Native | Couples to AMI patch cycle |
| SOCI parallel-pull | Faster full pull (concurrent) | Marginal | **Yes — fast pull, no enumeration, no pre-bake** | Built into recent AL2023+BR AMIs (2025-08), off by default | Doesn't eliminate transfer; needs CPU/NVMe headroom; 0.x |
| SOCI lazy-load | Fetch only touched bytes | Marginal | **Only if startup touches few bytes** (not your case) | DIY snapshotter config | Only >250 MB; degenerates to full pull; registry becomes runtime hot path |
| kube-fledged | Pre-pull enumerated list | Yes | No (can't enumerate 80k) | Yes | **ABANDONED since 2023** |
| DaemonSet pre-pull | Pre-pull enumerated list | Yes | No (can't enumerate 80k) | Yes | Scale-out race; GC eviction; must enumerate |

---

## 6. The L2 pre-warm vs L3 lazy-pull split — concrete recommendation

The decision is **not one mechanism — it's a layer split**:

1. **L2 shared bases → pre-warm onto every node before the run.** Strongest fit: **EBS-snapshot Bottlerocket data volume** via Karpenter `blockDeviceMappings.snapshotID`, so the bulk of every image's bytes are already local at node boot. Use `volumeInitializationRate` (and/or EBS Fast Snapshot Restore — FSR pairing is *inferred*, not in an AWS doc, flag) to kill the first-touch penalty. Hours of lead time is ample to rebuild this bounded snapshot. **Alternative:** Spegel for L2 if you'd rather not manage snapshots — same dedup without a central store, but pays the cold-first-pull and has no cross-AZ sharing.
2. **L3 distinct deltas → SOCI parallel-pull-unpack** (shipped in AL2023/Bottlerocket EKS AMIs, enable in NodeConfig), ideally onto **NVMe instance store**. With L2 already local, only the small per-task delta (layers not in the baked base) crosses the network — the combination most likely to approach the sub-second budget. **Do NOT use SOCI lazy-load here** unless a measured byte-access ratio justifies it.
3. **Registry path for L3 deltas → raise ECR pull-API quotas** (BatchGetImage / GetDownloadUrlForLayer / GetAuthorizationToken), **stagger node scale-up**, reach ECR via **PrivateLink + S3 gateway endpoint** (layer bytes come from S3). This is the irreducible part: 80k–100k distinct deltas transit ECR/S3 once; the levers are quota headroom, time-spreading, and minimizing bytes per delta (which step 1 does).
4. **Avoid** kube-fledged (abandoned), plain DaemonSet pre-pull for L3 (can't enumerate), and any expectation that cache/P2P reduces total upstream egress for the distinct tail (physically once-per-digest).

The hard ceiling that survives all of this: even with images local, **binding 80k–100k pods at ~300–500 pods/s = ~3–6 minutes**. If that's outside budget, the lever is sharded/custom schedulers or `nodeName` pre-assignment (§3) — and per-pod sub-second start is only achievable when the image is already on the node (step 1+2).

---

## 7. Multi-cluster sharding by language — does it help, and on which axis?

Sharding by language **helps the control-plane/scheduler axis, not the image-bytes axis.**

- **Helps:** fewer pods/objects per cluster → less etcd write pressure, less scheduler queue depth, more aggregate scheduler binding throughput (each cluster has its own kube-scheduler → N× the ~300–500 pods/s ceiling), smaller blast radius, and it sidesteps needing a single 100K-node ultra-scale cluster. If the per-run pod count per cluster drops below the ~150k-pod / Provisioned-Control-Plane-tier envelope, this is a legitimate way to stay on simpler control planes. This is the same logic as running multiple sharded schedulers, but at cluster granularity (and safely partitioned).
- **Does NOT help:** total image bytes. The 80k distinct L3 images still exist and still transit ECR/S3 once each; sharding just routes each language's images to that language's cluster. It *does* reduce **images-per-cluster** (so a per-cluster registry mirror / pre-warm set is smaller and more cacheable), and it improves **L2 locality** — a Go-only cluster pre-warms only Go toolchain bases, raising the shared-fraction per node and the P2P/pre-warm hit rate. So sharding **amplifies the L2 pre-warm win** and shrinks per-cluster cache footprint, without changing aggregate egress.
- **Costs:** more control planes to run/pay for, cross-cluster orchestration of runs, and ECR quotas are **per-region per-account** — multiple clusters in the same region/account **share the same pull-API quotas**, so sharding does *not* multiply your ECR throughput unless clusters are in different accounts/regions. This is a common trap: assuming N clusters = N× registry throughput. They don't, in one account+region.

**Verdict:** shard by language if it keeps per-cluster pod-count/churn within a simpler control-plane tier and you want N× scheduler throughput and better L2 locality. Don't expect it to relieve the ECR pull-API ceiling unless you also split accounts/regions.

---

## 8. Open questions / what they'd need to benchmark

1. **L2-vs-L3 byte split per image.** What fraction of each ≤200 MB image is shared L2 base vs distinct L3 delta? This determines how much the EBS-snapshot pre-warm offloads and how small the JIT pull is. (Workload-dependent; not knowable from docs.)
2. **Startup byte-access ratio for L3** (only if SOCI lazy-load is still on the table). Measure what fraction of the image the task actually reads in the first seconds. If it's not single-to-low-double-digit percent, lazy-load loses to pre-pull/parallel-pull.
3. **Actual per-pod start latency with L2 pre-warmed + L3 parallel-pulled to NVMe.** Whether "sub-second" is even reachable given the SLO is 5s *excluding pull*.
4. **ECR pull-API throttling under a realistic staggered ramp.** What scale-up cadence keeps GetDownloadUrlForLayer/BatchGetImage/GetAuthorizationToken under (raised) quotas. What quota increase AWS will actually grant.
5. **Scheduler drain time at target tier.** At 167/283/400 pods/s (or a sharded/custom scheduler), how long to bind a full run, and whether that's acceptable.
6. **`soci create` / convert cost for 80k–100k images** within the hours of lead time, given the Lambda Index Builder caps (6 GB / 15 min) and the CodeBuild fallback.
7. **EBS snapshot restore + FSR economics** for booting many nodes from a baked snapshot fast (FSR credit limits, `volumeInitializationRate` tuning).
8. **Provisioned Control Plane fit** — whether an XL/2XL/4XL tier (or sharded Standard clusters) matches the run's pod-churn profile, and the $/hr during the run window.

### Access / sourcing caveats
- The re:Invent 2025 CNS429 session (AWS re:post page) returned **HTTP 403** to the fetcher; its numbers (e.g., 7,500 read req/s) reached only via a third-party AI-translated writeup (zenn.dev) — treat as indicative, not quoted-from-AWS.
- AWS never literally stamps "GA" on ultra-scale or Provisioned Control Plane in the posts read; GA-grade status is inferred from pricing + IaC support + existing-cluster availability.
- Ultra-scale 100K-node / 900K-pod / 1.5M-QPS figures are **benchmark demonstrations explicitly hedged as "directional," not SLAs.**
- "500 pods/s" is a **tuned AWS vendor result**; honest untuned baseline is ~200–450 pods/s.
- SOCI 50–60% speedups are **vendor/customer figures on large, read-light images** — no independent published SOCI benchmark; negatives concentrate in AWS's own issue tracker (#555, #1648, #1850) and small-image/read-heavy analyses.
- Spegel CNCF-sandbox status is **unverified**; Dragonfly v2.5.0 and SOCI 0.x are very fresh with little field hardening.
- The EBS-FSR + image-cache pairing is **inferred**, not in an AWS doc.
- ECR has **no published single "max pulls/sec"** — only the per-API quotas; layer bytes are S3-served and uncounted.
