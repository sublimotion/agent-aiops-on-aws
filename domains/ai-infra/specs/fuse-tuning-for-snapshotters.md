# Spec E — FUSE Tuning for Lazy Snapshotters

## Status: DRAFT

## Hypothesis

Default FUSE settings on stargz-snapshotter and SOCI-snapshotter leave ≥3× throughput on the table. Applying Modal's published tunables (`read_ahead=32M`, `max_pages=32M`, `congestion_threshold=256`, deferred-write heap management) improves lazy-loaded image read throughput from ~800 MiB/s to ~2.5 GiB/s, matching Modal's published numbers — without requiring proprietary infrastructure.

## Falsification criteria

- Tuned throughput < 1.5 GiB/s on commodity AWS NVMe-backed instances → tunables don't transfer; either a snapshotter implementation gap or AWS-specific bottleneck.
- Improvement < 1.5× over default → tuning isn't worth the userspace patch maintenance cost.
- Tuning causes regressions on small-file-heavy workloads (Python imports) → conditional rule, not blanket default.

## Why this matters

This spec produces the **input config** for Spec A's "SOCI + Modal-style tuning" variant. Without it, the SOCI variants in Spec A run at default tunables and the comparison to EBS prebake is unfair.

It's also the cheapest spec to run — no model, no GPU, just a node, a snapshotter, and a stopwatch.

## Stage-budget claim

| Stage | Baseline (sec) | Predicted with FUSE tuning | Why |
|---|---|---|---|
| Image pull (lazy) | 90-180 | 30-60 | inner-loop bandwidth 3× |
| Container start | 5-30 | 5-10 | fewer userspace round-trips |
| All others | unchanged | unchanged | stage-local change |

This experiment is purely **inner-loop performance** of the lazy-loading mechanism. Other stages unaffected.

## Matrix

| Axis | Values |
|------|--------|
| Snapshotter | stargz-snapshotter, SOCI-snapshotter |
| Tunable set | (a) default, (b) `read_ahead` only, (c) `max_pages` only, (d) `congestion_threshold` only, (e) all-Modal-tunables, (f) all + deferred writes |
| Image profile | (i) bandwidth-bound (few large layers, e.g. CUDA toolkit), (ii) ops-bound (many small files, e.g. Python site-packages) |
| Hardware | g7e.24xlarge (NVMe-backed), m7i.4xlarge (EBS gp3, control) |

2 × 6 × 2 × 2 = 48 cells. Run all — each cell is a few minutes.

## Baseline

stargz-snapshotter and SOCI-snapshotter at upstream-default config. Same image, same instance type.

## Measurement

- **Bandwidth**: `dd if=/mnt/lazy/<largefile> of=/dev/null bs=4M` after FUSE mount. Median of 5 runs.
- **Ops/s**: `find /mnt/lazy -type f | xargs stat` over the entire image. Wall clock + syscall count.
- **Cold-start proxy**: time `python -c "import torch; import vllm"` against the lazy mount. Captures realistic mixed read+stat workload.
- **Negative-lookup latency**: `time test -e /nonexistent/path` repeated 1000×.

Output: per-cell results table + recommended config snippet for each snapshotter.

## Fixtures

No model, no GPU. Use a fresh g7e or m7i node, install snapshotter, point at a representative image (e.g. `vllm/vllm-openai:latest`).

## Rule the experiment would produce

> **FUSE tuning for stargz/SOCI on EKS**: bootstrap nodes (Karpenter EC2NodeClass userData) with the following snapshotter config:
> ```toml
> [fuse]
> read_ahead_kb = 32768       # 32 MiB
> max_pages = 8192            # 32 MiB worth of 4KiB pages
> congestion_threshold = 256
> ```
> Apply universally for image profiles (i) and (ii). Negative-lookup fast-path required (verify snapshotter version supports it).

If falsified, the rule is the inverse: default config is fine; don't invest in patches.

## Out of scope

- Snapshotter selection (stargz vs SOCI vs Nydus) — that's Spec A.
- AZ-local cache server architecture — separate steering decision.
- Modal's proprietary content-addressed registry — not portable to EKS.

## Cost estimate

< $50. Quick experiment.

## References

- Modal: [Fast lazy container loading](https://tinfoil-knight.github.io/notes/fast,-lazy-container-loading-in-modal-2024) — primary tunables source
- stargz-snapshotter: https://github.com/containerd/stargz-snapshotter
- SOCI-snapshotter: https://github.com/awslabs/soci-snapshotter
- Linux FUSE docs: `man 4 fuse`
