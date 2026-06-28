# Steering Additions from minimax-m2 compound step (2026-06-28)

## To insert after L93 (under Deployment Conventions, after "scale to 0" rule)

```markdown
#### Pin kubectl context explicitly in unattended runners with destructive actions

When running detached scripts that scale nodes (`aws eks update-nodegroup-config ... desiredSize=0`) or terminate instances, NEVER rely on ambient `kubectl` context. Pin the context explicitly in every kubectl call and gate any destructive action behind a positive confirmation that you are operating the intended cluster.

```bash
# Pattern: context-pinned kubectl
KCTX=(kubectl --context "$EXPECT_CONTEXT")
"${KCTX[@]}" get nodes

# Preflight before any scaledown
if ! "${KCTX[@]}" get nodes -l blueprint=<name> | grep -q Ready; then
  echo "ERROR: Expected node not Ready in context $EXPECT_CONTEXT"
  exit 1
fi
PREFLIGHT_OK=true

# Scaledown gate
if [[ "$PREFLIGHT_OK" == "true" ]]; then
  aws eks update-nodegroup-config ... --desired-size 0
fi
```

**Why**: 2026-06-27, on minimax-m2, kubectl context drifted from `qn-bench-use2` (us-east-2) to `qn-sglang-usw2` (us-west-2) in another shell. A relaunched detached sweep's trap-scaledown (`pkill` → EXIT trap) fired `aws --region us-east-2` scaledown while all kubectl calls targeted us-west-2 → drained+terminated the live B200 node in us-east-2. Lost ~60min rebuild. Context-pinning + preflight interlock prevents the wrong-cluster trap from executing.
```

---

## To insert after L121 (after "Always document benchmark execution location before running")

```markdown
#### vLLM cache-hit measurement is server-side only (Prometheus counters), not per-request
<!-- stack: vllm=0.19.1rc1-0.23.0 | validated: 2026-06-27 -->

vLLM 0.19.x through 0.23.x reports prefix cache hits in **SERVER COUNTERS** on the `/metrics` endpoint (`vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total`), NOT in the per-request response's `cached_tokens` field (which returns `None` on these builds). Client-side cache-hit measurement reads 0; you must harvest the Prometheus counter snapshot before/after the batch to compute the delta.

```python
# WRONG (reads 0 on vLLM 0.19-0.23)
cached_tokens = response.usage.cached_tokens  # None

# RIGHT
# Snapshot Prometheus vllm:prefix_cache_hits_total before/after batch, take delta
hit_rate = (hits_after - hits_before) / (queries_after - queries_before)
```

The per-request `cached_tokens` field exists in the OpenAI API spec but is not populated by these engine versions. Verify the observability pod (Prometheus scraper) is running and scraping before trusting a "cache=0" reading — a missing scraper is an instrumentation gap, not a cache miss.

#### Always validate the output artifact, not just the exit code

A runner exiting `rc=0` does NOT mean it produced its deliverable. After any unattended run, validate the output artifact exists and is non-empty before trusting the "success" signal.

```bash
# WRONG — trusting exit code alone
if ./run-benchmark.sh; then
  echo "Success"
fi

# RIGHT — validate the artifact
if ./run-benchmark.sh && [[ -s results/pareto.json ]] && jq empty results/pareto.json 2>/dev/null; then
  echo "Success — artifact validated"
else
  echo "FAILED — missing or invalid output"
fi
```

Prefer an **append-only ledger + rebuild-at-end** pattern: write results to a `.jsonl` as they're produced, then rebuild the headline artifact from the ledger at the end. A single bad incremental write can't zero the deliverable.

**Why**: 2026-06-27, minimax-m2 VALIDATE_ONLY run completed `rc=0` with real per-config data in the trajectory ledger, but the headline `pareto-<date>.json` was `[]`. Root cause: `python3 - <<'HEREDOC' ... json.load(sys.stdin)` — the heredoc redirected stdin, so `json.load(sys.stdin)` parsed the consumed script text → `JSONDecodeError` → point never appended. The trajectory survived only because it used `>>` append (no stdin conflict). The JSONDecodeError tracebacks in the log were mis-attributable. Only validating the pareto artifact caught it before the full grid.
```

---

## To update the cold-start table at L132-136 (add a new row)

Add this row to the table after the existing "vLLM DeepGEMM / B200 sm_100f" row:

```
| vLLM DeepGEMM / B200 sm_100 (MiniMax-M2 FP8) | ~35-40 min | Weight load 25.5 min (130 shards @ ~13s/shard, 214GB FP8) + torch.compile + CUDA-graph. Cache `/root/.cache/vllm/` saves JIT/compile tail only, NOT weight load (pure NVMe read BW). | 2026-06-27 |
```
