# Spec C-EBS — Compile Cache via EBS Snapshot, Weights on Instance NVMe

## Status: DRAFT

This is a sister spec to `compile-cache-strategies.md` (Spec C). It tests the AWS-native architecture for persistent compile-cache across pod and node lifecycles using an EBS snapshot, while keeping weights on instance NVMe (where bandwidth wins).

## Hypothesis

A hybrid storage tier — **EBS snapshot for compile cache, instance NVMe for weights** — delivers the AOT-compile-cache HIT benefit on *any* fresh node (not just the same node where the cache was originally populated), without paying the cost of EBS bandwidth for the weight-load stage.

Specifically:
- **Cache hit on cold cluster**: any new pod sees a populated `/root/.cache/vllm` from EBS snapshot, hits AOT cache on first compile attempt. Saves the ~17-30 s torch.compile work + Inductor warmup that would otherwise happen on cache MISS.
- **Multi-variant coexistence**: different vLLM versions, different `cudagraph_capture_sizes`, different attention backends each have their own hash-keyed subdir in `/root/.cache/vllm/torch_compile_cache/<hash>/`. vLLM picks the right one automatically. No image rebuild per variant.
- **Snapshot restore overhead**: < 60 s for cold attach + lazy-load of 500 MB of cache files via Fast Snapshot Restore (FSR).

## Falsification criteria

- EBS-mounted cache fails to HIT (vLLM doesn't find the directory or hash mismatch) → cache invalidation rules need refinement.
- Snapshot attach overhead > 90 s consistently → not faster than just rebuilding the cache (~30-60 s for compile work).
- Multi-variant coexistence breaks (variants step on each other's hashes) → operational flexibility lost.
- io2 Block Express required for acceptable read throughput → cost (>$120/mo) outweighs cache benefit (~17 s saved per replica-1).

## Why this matters

Spec C-image (compile cache baked into Docker image, deferred from current Spec C plan):
- Each variant requires a new image build + push + retag
- Image grows by 500 MB per cache, multiplying for variants
- Cache update couples to image deploy

Spec C-EBS (this spec):
- Snapshot once, copy-on-write for new pods
- Multiple variants share one volume
- Update the cache without touching the image
- Same on-disk format vLLM uses natively (`compile_cache_save_format='binary'`)

For lab use specifically: **operational flexibility matters more than ~5% absolute speedup difference**. We need to test variants quickly without rebuilds.

## Stage-budget claim

For Kimi K2.6 FP8 TP=8 on B200, **fresh node** (cold cluster, no prior cache):

| Stage | Without EBS cache | With EBS cache (this spec) | Δ |
|---|---|---|---|
| Node provision | 60-120 s | 60-120 s | 0 |
| EBS snapshot attach + mount | n/a | 30-60 s | +30-60 s |
| Image pull | 30 s | 30 s | 0 |
| S3 → NVMe weight sync | 18 min | 18 min | 0 |
| Container start | 5 s | 5 s | 0 |
| Weight load | 245 s | 245 s | 0 |
| **JIT compile + Inductor warmup** | **~17 s + ~5 s warmup** | **~5 s warmup only** | **-17 s (cache HIT)** |
| CUDA graph capture | 50 s | 50 s | 0 |
| **Total fresh-node cold start** | **~30 min** | **~29.5 min** | **-17 s, +30-60 s overhead = roughly neutral** |

For Kimi K2.6 FP8 TP=8 on B200, **warm-NVMe replica restart** (weights cached, compile cache lost — e.g., pod replaced on same node):

| Stage | Without EBS cache | With EBS cache (this spec) |
|---|---|---|
| Snapshot attach + mount | n/a | 30-60 s |
| All else identical | | |
| JIT compile | ~140 s (no cache) | **~17 s (HIT)** |
| **Total warm-NVMe** | ~7 min | **~5.7 min (-1.3 min)** |

The win is concentrated in the **second-and-subsequent replicas on cold nodes**, where weights are cold but the compile work would have been duplicated.

For *small* models where weight-load is fast (e.g., Qwen3 8B at ~5 s weight load), the cache HIT savings dominate proportionally — could be 30-50% of total cold start.

Replica index: target is **any cold-cluster fresh node**, not just N≥2 same-host. This is the differentiator from ModelExpress.

## Matrix

| Axis | Values |
|------|--------|
| Cache state | (a) cold (no EBS attach), (b) EBS-attached fresh (this run will populate), (c) EBS-attached pre-populated (this is the test cell) |
| Variant within cache dir | (i) cudagraph trim 10-sizes, (ii) cudagraph default 51-sizes, (iii) different vLLM version (out of scope this run) |
| Weight source | NVMe-warm only (the EBS-bandwidth question is dodged by keeping weights on instance NVMe) |
| Hardware | p6-b200.48xlarge us-east-2b (matches our existing runs) |

Run set:
1. **Bake**: cold pod, run to ready, copy `/root/.cache/vllm/` → EBS volume, snapshot it
2. **Validate cold-cluster restore**: fresh node, attach EBS-from-snapshot, mount as `/root/.cache/vllm`, run pod → confirm cache HIT logs
3. **Multi-variant**: same EBS, different cudagraph config → confirm separate hash dir created, original variant still works

3 cells minimum, can be done in one B200 session of ~90 min.

## Baseline

Existing run-3 measurement (`b200-kimi-stats/results/run-3-cache-hit.json`): 384 s main → ready with cache HIT on the *same node* via hostPath. Spec C-EBS targets the same number on a *different node*, with the cache restored from snapshot.

## Measurement

Reuse `shared/profiler.py` events. Additional instrumentation:

- **EBS attach time**: from `aws ec2 attach-volume` API call to filesystem-mounted (host-side, captured separately)
- **Cache HIT vs MISS detection**: grep vLLM logs for `Directly load AOT compilation from path` (HIT) vs `Compiling .*with torch.compile` for an extended duration (MISS)
- **Per-rank HIT confirmation**: 8 TP ranks should each log HIT for their `rank_N_0/model` path

## Architecture

### Storage layout

```
EBS volume (700 GB gp3, snapshot-backed):
  /mnt/persistent/
    compile-cache/                          ← mounted into pods as /root/.cache/vllm
      torch_compile_cache/
        <model-config-hash-A>/              ← e.g., kimi-tp8-cu130-trim10
          rank_0_0/
            backbone/                       (vLLM compile cache)
            eagle_head/
          rank_1_0/ ... rank_7_0/
        torch_aot_compile/
          <aot-hash-A>/
            rank_0_0/model
            rank_1_0/model
            ...
        <model-config-hash-B>/              ← e.g., kimi-tp8-cu130-default51
        <model-config-hash-C>/              ← e.g., qwen3-8b-tp1
      modelinfos/                           (vLLM model registry cache)
    metadata.json                           (tracking which variants exist + last update timestamps)

Instance NVMe (3.5 TB ephemeral):
  /mnt/nvme/
    Kimi-K2.6/                              ← weights, populated from S3 (init container)
    qwen3-8b/                               ← other models
    .staged                                  ← marker file
```

### Pod manifest pattern

```yaml
spec:
  volumes:
    - name: compile-cache-pv
      persistentVolumeClaim:
        claimName: ai-infra-compile-cache
    - name: nvme
      hostPath:
        path: /mnt/nvme
        type: Directory
  containers:
    - name: vllm
      volumeMounts:
        - name: compile-cache-pv
          mountPath: /root/.cache/vllm
        - name: nvme
          mountPath: /mnt/nvme
      args:
        - "--model=/mnt/nvme/Kimi-K2.6"
        # vLLM auto-uses /root/.cache/vllm; no extra arg needed
```

### EBS volume lifecycle

| Operation | When | Cost |
|---|---|---|
| Create from snapshot | On pod scheduling (CSI driver attach) | ~30 s |
| Mount | Pod init | <1 s |
| Read cache files | First compile-cache load | lazy-fetch from S3 if FSR off, immediate if FSR on |
| Write new cache (cache MISS) | First run of new variant | ~500 MB write |
| Detach | Pod terminate | <10 s |
| Snapshot (after each cache update) | Manual or scheduled job | incremental, ~minutes |
| Delete volume | When done | $0 — only snapshot persists |

### FSR (Fast Snapshot Restore) decision

- **FSR ON**: ~$0.75/hour per snapshot per AZ. Eliminates first-touch latency. Worth it for production, overkill for lab.
- **FSR OFF**: lazy-load via S3, ~10-30 ms first-touch per block. For 500 MB cache that's <5 s overhead. **Default for lab.**
- **Switch to FSR ON** if measurements show first-touch latency bleeds into compile-stage timings.

## Fixtures

- `kimi-k2.6-fixture` from `staging/manifests/kimi-k2.6-fixture.yaml` — copy and add the PVC mount
- New manifest: `staging/manifests/kimi-k2.6-ebs-cache-fixture.yaml`
- Reuses existing instance NVMe data path; only adds the PVC mount

## Rule the experiment would produce

> **Persistent compile cache via EBS snapshot**: bake compile artifacts into a versioned EBS snapshot, mount via PVC at `/root/.cache/vllm` in serving pods. Multi-variant: vLLM auto-keys cache by config hash, so multiple `cudagraph_capture_sizes`, `tp_size`, `dtype` variants coexist on one volume. Refresh snapshot after major version bumps. Use Fast Snapshot Restore in production environments where cold-cluster cold-start SLO matters; default off for lab/staging.
>
> **What this rule does NOT cover**: weight load (still requires instance NVMe + S3 sync, or ModelExpress P2P for replica-N≥2). Use this in combination with `staging/manifests/kimi-k2.6-fixture.yaml`'s init-container weight stage.

## Out of scope

- Weight loading (covered by Spec B and existing init-container patterns)
- Image-pull stage (covered by Spec A)
- Cross-region snapshot replication (operational concern, not lab)
- KMS-signed cache artifacts (steering rule from Spec C; not testable in this experiment)
- io2 Block Express tier exploration — only relevant if gp3 read throughput is the bottleneck for cache load (we expect not)

## Cost estimate

| Item | Cost |
|---|---|
| EBS volume 700 GB gp3 (when attached) | $56/month full-time, prorated |
| EBS snapshot storage | ~$30/month (700 GB × $0.05/GB-month) |
| Fast Snapshot Restore (optional) | ~$0.75/hour × 24 × 30 = $540/month per AZ — **disabled for lab** |
| B200 time for bake + validate | ~$25 (15 min bake + 30 min validate) |
| **Total to run this spec once** | **~$25 + $5 EBS for the first month** |
| **Steady-state if kept on** | **~$30/month idle storage** |

## References

- Modal "Truly Serverless GPUs" (`spec-c-compile-cache/references/modal-truly-serverless-gpus.md`) — 4 cache tiers, custom filesystem, the architecture this spec emulates with AWS primitives
- Modal Gemma4 AOT log (`spec-c-compile-cache/references/modal-gemma4-aot-h200.md`) — measured AOT cache HIT timings (13.96 s + 2.82 s = 22 s combined for 26B-class)
- Spec C parent (`compile-cache-strategies.md`) — image-baked variant comparison
- Spec B (`model-decoupling-and-load.md`) — weight load mechanisms (orthogonal to this spec)
- Existing measurements: `b200-kimi-stats/results/run-3-cache-hit.json` — same-node hostPath baseline (384 s main→ready)
- AWS docs on EBS snapshots and FSR: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-fast-snapshot-restore.html
