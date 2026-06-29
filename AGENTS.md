# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

> **Codex/Bedrock: never view images.** The Bedrock vision path inlines images as raw base64 text (~1 token/char), so a single screenshot (≈500KB) becomes ~520K tokens and overflows the context window on the next call. Do NOT screenshot rendered HTML and feed it to `view_image`, and do not call any image-viewing tool. To inspect HTML/reports, read the source with `sed`/`rg`/`cat` instead.

## Workflow

1. Write spec in the correct domain location (see Domain Routing below)
2. Run `mdc get <model> --engine <engine>` to load the deployment card before deploying
3. Run `/ralph-loop:ralph-loop Deploy <spec-path>`
4. Claude selects the deployer agent automatically based on spec path (see Domain Routing)
5. Claude iterates until deployment succeeds — **capture failures and fixes to `lessons.md` as they happen** (mid-conversation lesson capture, see infra-deployer agent)
6. Run compound step: invoke `compound-learner` agent — it elevates cross-cutting lessons to `.claude/steering/*.md` **and generates the YAML frontmatter** for `lessons.md` (model, engine, hardware, outcome, failure_categories, mdc_learn_commands, gpu_infra_learn_commands)
7. Run `scripts/fe.sh learn <blueprint-path>` to apply the learn commands from the frontmatter
8. Optionally run `scripts/fe.sh contribute <blueprint-path>` to open a community contribution issue

### Version Refresh

When a stack component upgrades (new vLLM, SGLang, Ray, etc.), run the version refresh protocol before the next deployment that uses it. This scans `tech-stack.md` for rules tagged with the old version and validates each one. See `compound-learner.md` for the full protocol. Steering rules older than 90 days without a refresh are flagged as stale.

## Context Loading

Read these files **on demand** based on what you're doing:

| When you are... | Read |
|-----------------|------|
| Writing or modifying Terraform | `.claude/steering/tech-stack.md` |
| Making architectural or product decisions | `.claude/steering/product.md` |
| Creating new files, modules, or blueprints | `.claude/steering/project-structure.md` |
| About to deploy a model | `mdc get <model> [--engine <engine>]` and `mdc prs <model>` for upstream deployment card + recent PRs |
| Deploying or modifying a GPU-serving blueprint | `domains/gpu-serving/specs/<matching-spec>.md` |
| About to deploy a HyperPod blueprint (spec ending in `-hyperpod`) | `.claude/steering/tech-stack.md` §SageMaker HyperPod Inference Operator release tracking, then WebFetch the release-notes URL to confirm the pinned version is current |
| Deploying an agent-runtime blueprint | `domains/agent-runtime/specs/<name>.md` |
| Running an autoresearch experiment | `domains/autoresearch/specs/<name>.md` |
| Designing a verifier, model router, cascade, or test-time selection | `docs/verifier-router-mechanism-selection.md` — measure regime first (routing headroom, cost-spread, verify-cost, verifier portability) → decision tree picks the mechanism (single model / verifier-gated cascade / trained router). Don't build before measuring |
| Running the compound step after deployment | `domains/<domain>/blueprints/<name>/lessons.md` + `.claude/steering/*.md`; compound-learner must generate YAML frontmatter (see `docs/card-format.md`) as its final step, then run `scripts/fe.sh learn <path>` |
| Fixing lint, checkov, tflint, or pre-commit failures | `.claude/steering/lint-remediation.md` |
| Running pre-flight or post-deploy checks | Use the `deployment-orchestrator` skill |
| Launching or managing a detached agent run on EKS (agent-runner CLI) | Use the `agent-runtime` skill |
| Selecting a serving config / filling the Stage 0b lever ledger | `.claude/steering/inference-first-principles.md` (predict the regime) → `docs/optimization-stack.md` (T0–T6 lever catalog: priority per regime, typical Δ, conflicts). Account for every tier; defer with a reason, don't skip silently |
| Tuning a serving config past the first working deploy (in-spec optimization loop) | `standards/benchmark-commons/OPTIMIZATION-LOOP.md` — declared SLO objective, lineage/dead-end trajectory record, plateau/budget exit, reward-hacking guard. The loop reads `optimization-stack.md` as its search space + ordering prior |
| Gating a serving config before deploy (Stage 0c) | `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar <benchmark.yaml> --corpus-root .` — fail-closed; exit 2 blocks. See `standards/serving-commons/README.md` |
| Validating GPU hardware before serving (Stage 4a) | Use `gpu-infra` MCP tools: `discover_cluster`, `check_gpu_health`, `run_nccl_test` |
| Diagnosing GPU hardware issues | Use `gpu-infra` MCP tools: `explain_xid`, `get_gpu_metrics`, `check_gpu_health` |
| Running or designing benchmarks | `standards/benchmark-commons/PROPOSAL.md` + `docs/inference-optimization-guide.md` |
| Choosing benchmark workloads | `standards/benchmark-commons/workloads/*.yaml` (7 standard workload cards) |
| Writing Stage 6 benchmark criteria | `domains/gpu-serving/specs/_template.md` (Stage 6 section) |
| Interpreting or comparing benchmark results | `.claude/steering/benchmark-analysis.md` (verify-before-assert; engine/output-length matching; bottleneck-class evidence) + `docs/inference-optimization-guide.md` (Sections 9-12: cost, methodology, KV cache, kernels) |
| Building or deploying the benchmark container | `standards/benchmark-commons/container/` (Dockerfile, enrich-benchmark.py, benchmark-job.yaml) |
| Writing a benchmark.yaml sidecar for a blueprint | `standards/benchmark-commons/examples/nemotron-super/benchmark.yaml` (reference example) |

**Spec routing** — match blueprints to specs by name:

| Blueprint | Spec |
|-----------|------|
| `domains/gpu-serving/blueprints/ministral-3b/` | `domains/gpu-serving/specs/ministral-3b.md` |
| `domains/gpu-serving/blueprints/kimi-k2.5/` | `domains/gpu-serving/specs/kimi-k2.5.md` |
| `domains/gpu-serving/blueprints/qwen3-next/` | `domains/gpu-serving/specs/qwen3-next.md` |
| `domains/gpu-serving/blueprints/qwen3-next-g7e/` | `domains/gpu-serving/specs/qwen3-next-g7e.md` |
| `domains/gpu-serving/blueprints/qwen3-next-custbench/` | `domains/gpu-serving/specs/qwen3-next-custbench.md` |
| `domains/gpu-serving/blueprints/qwen3-next-sglang/` | `domains/gpu-serving/specs/qwen3-next-sglang.md` |
| `domains/gpu-serving/blueprints/devstral-sera/` | `domains/gpu-serving/specs/devstral-sera.md` |
| `domains/gpu-serving/blueprints/glm5/` | `domains/gpu-serving/specs/glm5.md` |
| `domains/gpu-serving/blueprints/glm5.2/` | `domains/gpu-serving/specs/glm5.2.md` |
| `domains/gpu-serving/blueprints/glm5-hyperpod/` | `domains/gpu-serving/specs/glm5-hyperpod.md` |
| `domains/gpu-serving/blueprints/glm5-lmcache/` | `domains/gpu-serving/specs/glm5-lmcache.md` |
| `domains/gpu-serving/blueprints/glm5-llmd/` | `domains/gpu-serving/specs/glm5-llmd.md` |
| `domains/gpu-serving/blueprints/nemotron-super/` | `domains/gpu-serving/specs/nemotron-super.md` |
| `domains/gpu-serving/blueprints/nemotron-ultra/` | `domains/gpu-serving/specs/nemotron-ultra.md` |
| `domains/gpu-serving/blueprints/fin-rag-answer/` | `domains/gpu-serving/specs/fin-rag-answer.md` |
| `domains/gpu-serving/blueprints/fin-rag-answer-g7e/` | `domains/gpu-serving/specs/fin-rag-answer-g7e.md` |
| `domains/gpu-serving/blueprints/fin-rag-answer-h200/` | `domains/gpu-serving/specs/fin-rag-answer-h200.md` |
| `domains/gpu-serving/blueprints/ray-serve-ft/` | `domains/gpu-serving/specs/ray-serve-ft.md` |
| `domains/gpu-serving/blueprints/ray-serve-video/` | `domains/gpu-serving/specs/ray-serve-video.md` |
| `domains/gpu-serving/blueprints/llmd-hyperpod/` | `domains/gpu-serving/specs/llmd-hyperpod.md` |
| `domains/gpu-serving/blueprints/dynamo-hyperpod/` | `domains/gpu-serving/specs/dynamo-hyperpod.md` |
| `domains/gpu-serving/blueprints/gemma4-hyperpod/` | `domains/gpu-serving/specs/gemma4-hyperpod.md` |
| `domains/gpu-serving/blueprints/gemma4-4b-hyperpod/` | `domains/gpu-serving/specs/gemma4-4b-hyperpod.md` |
| `domains/gpu-serving/blueprints/mistral-small-4-hyperpod/` | `domains/gpu-serving/specs/mistral-small-4-hyperpod.md` |
| `domains/gpu-serving/blueprints/kimi-k2-thinking/` | `domains/gpu-serving/specs/kimi-k2-thinking.md` |
| `domains/gpu-serving/blueprints/kimi-k2.6-speculative/` | `domains/gpu-serving/specs/kimi-k2.6-speculative.md` |
| `domains/gpu-serving/blueprints/kimi-k2.6-lmcache-smoke/` | `domains/gpu-serving/specs/kimi-k2.6-lmcache-smoke.md` |
| `domains/gpu-serving/blueprints/qwen3-235b-b300/` | `domains/gpu-serving/specs/qwen3-235b-b300.md` |
| `domains/gpu-serving/blueprints/qwen3-235b-speculative/` | `domains/gpu-serving/specs/qwen3-235b-speculative.md` |
| `domains/gpu-serving/blueprints/kimi-k2.6/` | `domains/gpu-serving/specs/kimi-k2.6.md` |
| `domains/gpu-serving/blueprints/kimi-k2.6-nvfp4/` | `domains/gpu-serving/specs/kimi-k2.6-nvfp4.md` |
| `domains/gpu-serving/blueprints/kimi-k2.6-cutedsl-moe/` | `domains/gpu-serving/specs/kimi-k2.6-cutedsl-moe.md` |
| `domains/gpu-serving/blueprints/deepseek-v4-flash/` | `domains/gpu-serving/specs/deepseek-v4-flash.md` |
| `domains/gpu-serving/blueprints/deepseek-ocr-2-eks/` | `domains/gpu-serving/specs/deepseek-ocr-2-eks.md` |
| `domains/gpu-serving/blueprints/fin-attribute-extraction/` | `domains/gpu-serving/specs/fin-attribute-extraction.md` |
| `domains/gpu-serving/blueprints/minimax-m2/` | `domains/gpu-serving/specs/minimax-m2.md` |
| `domains/gpu-serving/blueprints/minimax-m2-kv-tiering/` | `domains/gpu-serving/specs/minimax-m2-kv-tiering.md` |
| `domains/gpu-serving/blueprints/qwen3-embedding-8b-hyperpod/` | `domains/gpu-serving/specs/qwen3-embedding-8b-hyperpod.md` |
| `domains/agent-runtime/blueprints/research-agent/` | `domains/agent-runtime/specs/research-agent.md` |
| `domains/agent-runtime/blueprints/managed-agent-runner/` | `domains/agent-runtime/specs/managed-agent-runner.md` |
| `domains/autoresearch/blueprints/training-recipes/` | `domains/autoresearch/specs/training-recipes.md` |
| `domains/autoresearch/blueprints/agent-harness/` | `domains/autoresearch/specs/agent-harness.md` |
| `domains/autoresearch/blueprints/finetuning-recipes/` | `domains/autoresearch/specs/finetuning-recipes.md` |
| `domains/autoresearch/blueprints/finetuning-recipes-1.7b/` | `domains/autoresearch/specs/finetuning-recipes-1.7b.md` |
| `domains/autoresearch/blueprints/agent-swarm/` | `domains/autoresearch/specs/agent-swarm.md` |
| `domains/autoresearch/blueprints/verifier-reward/` | `domains/autoresearch/specs/verifier-reward.md` |
| `domains/autoresearch/blueprints/coderforge-eval/` | `domains/autoresearch/specs/coderforge-eval.md` |
| `domains/autoresearch/blueprints/verification-primitives/` | `domains/autoresearch/specs/verification-primitives.md` |
| `domains/autoresearch/blueprints/verification-primitives-swebench/` | `domains/autoresearch/specs/verification-primitives-swebench.md` |
| `domains/autoresearch/blueprints/debate-verification/` | `domains/autoresearch/specs/debate-verification.md` |
| `domains/autoresearch/blueprints/pivot-analysis/` | `domains/autoresearch/specs/pivot-analysis.md` |
| `domains/autoresearch/blueprints/self-coding-agent-loop/` | `domains/autoresearch/specs/self-coding-agent-loop.md` |
| `domains/autoresearch/blueprints/svg-ece-measurement/` | `domains/autoresearch/specs/svg-ece-measurement.md` |
| `domains/autoresearch/blueprints/tiny-judge/` | `domains/autoresearch/specs/tiny-judge.md` |
| `domains/autoresearch/blueprints/verification-flywheel/` | `domains/autoresearch/specs/verification-flywheel.md` |
| `domains/autoresearch/blueprints/rejection-sampling-sft/` | `domains/autoresearch/specs/rejection-sampling-sft.md` |
| `domains/autoresearch/blueprints/trinity-coordinator/` | `domains/autoresearch/specs/trinity-coordinator.md` |
| `domains/autoresearch/blueprints/vla-cv-distillation/` | `domains/autoresearch/specs/vla-cv-distillation.md` |
| `domains/autoresearch/blueprints/kernel-optimization-agent/` | `domains/autoresearch/specs/kernel-optimization-agent.md` |
| `domains/autoresearch/blueprints/mooncake-kv-tiering/` | `domains/autoresearch/specs/mooncake-kv-tiering.md` |
| `domains/autoresearch/blueprints/rl-conductor/` | `domains/autoresearch/specs/rl-conductor-repro.md` |
| `domains/autoresearch/blueprints/cost-aware-routing/` | `domains/autoresearch/specs/cost-aware-routing.md` |
| `domains/autoresearch/blueprints/epd-disaggregation/` | `domains/autoresearch/specs/epd-disaggregation.md` |
| `domains/autoresearch/blueprints/e-fin1-finqa-skill-verifier/` | `domains/autoresearch/specs/e-fin1-finqa-skill-verifier.md` |
| `domains/autoresearch/blueprints/e-fin2-finqa-behavioral-features/` | `domains/autoresearch/specs/e-fin2-finqa-behavioral-features.md` |
| `domains/autoresearch/blueprints/e-harness1-harness-behavioral-interaction/` | `domains/autoresearch/specs/e-harness1-harness-behavioral-interaction.md` |
| `domains/autoresearch/blueprints/e-harness2-jit-vs-offline-authoring/` | `domains/autoresearch/specs/e-harness2-jit-vs-offline-authoring.md` |
| `domains/autoresearch/blueprints/e-harness3-reward-regime-x-locus/` | `domains/autoresearch/specs/e-harness3-reward-regime-x-locus.md` |
| `domains/autoresearch/blueprints/e-trace-profile-mechanism/` | `domains/autoresearch/specs/e-trace-profile-mechanism.md` |
| `domains/autoresearch/blueprints/cheaper-verifier-cascade/` | `domains/autoresearch/specs/cheaper-verifier-cascade.md` |

**Blueprint-local context** — for operational details (lessons, results, plans), look inside the blueprint directory itself rather than in specs.

## Domain Routing

The repo is organized into domains. **Infer the deployer agent automatically from the spec path** — no need to specify it explicitly in the RALPH loop command.

| Domain | Spec path prefix | Blueprint location | Deployer agent |
|--------|------------------|--------------------|----------------|
| GPU Serving | `domains/gpu-serving/specs/` | `domains/gpu-serving/blueprints/` | `infra-deployer` |
| Agent Runtime | `domains/agent-runtime/specs/` | `domains/agent-runtime/blueprints/` | `agentcore-deployer` |
| Autoresearch | `domains/autoresearch/specs/` | `domains/autoresearch/blueprints/` | `autoresearch-runner` |
| AI Infra | `domains/ai-infra/specs/` | `domains/ai-infra/blueprints/` | `infra-deployer` |

**Auto-detection rule**: all specs live under `domains/<name>/specs/`. Use `domains/gpu-serving/` → `infra-deployer`, `domains/agent-runtime/` → `agentcore-deployer`, `domains/autoresearch/` → `autoresearch-runner`, `domains/ai-infra/` → `infra-deployer`.

**ai-infra is the platform tooling + experimentation lab**: hosts shared infrastructure (slim serving images, cold-start profiler, build host) and isolated technique experiments (image pull, weight load, compile cache) that produce steering rules rather than persistent deployments. See `domains/ai-infra/README.md`.

Examples:
```
/ralph-loop:ralph-loop Deploy domains/agent-runtime/specs/research-agent.md   → agentcore-deployer
/ralph-loop:ralph-loop Deploy domains/gpu-serving/specs/kimi-k2.5.md          → infra-deployer
/ralph-loop:ralph-loop Run domains/autoresearch/specs/training-recipes.md     → autoresearch-runner
```

## External Tools

### fe CLI

Unified entry point for card lookup, field note lifecycle, and community contribution. Wraps `mdc` and `gpu-infra`.

```bash
# Before deploying — look up cards
fe card <model> --engine <engine>    # e.g. fe card glm-5 --engine sglang
fe card --hardware <instance>        # e.g. fe card --hardware p6-b200

# After compound step — apply learn commands from lessons.md frontmatter
fe learn domains/gpu-serving/blueprints/<name>/

# Contribute field note to community (opens GitHub Issue template)
fe contribute domains/gpu-serving/blueprints/<name>/
```

### Model Deployment Cards (mdc)

Curated deployment recipes with upstream PR tracking and tribal knowledge. Repo: `../model-deployment-card/`

```bash
# Before deploying — load the deployment card (also via: fe card)
mdc get <model> --engine <engine>   # e.g. mdc get glm-4.5 --engine sglang
mdc prs <model>                     # check recent upstream PRs

# After deploying — feed lessons back (also via: fe learn)
mdc learn <model> <engine> "<note>"
mdc learn <model> <engine> --from domains/gpu-serving/blueprints/<name>/lessons.md
```

### GPU Infrastructure (gpu-infra)

Live GPU diagnostics via MCP server + CLI for feedback. Repo: `../gpu-infra-troubleshooting/`. MCP config: `.mcp.json`

**MCP tools** (live diagnostics):

| Tool | When to use |
|------|-------------|
| `discover_cluster` | Stage 4a — enumerate GPUs, topology, driver, EFA before serving |
| `check_gpu_health` | Stage 4a — validate ECC, row remap, thermals, PCIe link |
| `run_nccl_test` | Stage 4a — verify NCCL collectives for multi-GPU TP |
| `explain_xid` | Any time — look up Xid error codes from dmesg |
| `get_gpu_metrics` | Benchmarking — pull DCGM metrics from Prometheus |

**CLI** (cards + feedback):

```bash
# Before deploying — load the GPU architecture card
gpu-infra card g7e                        # full card for instance type
gpu-infra cards --interconnect NVSwitch   # filter available cards

# After deploying — feed hardware lessons back
gpu-infra learn -c nccl "NCCL 2.25.1 broken on Blackwell PCIe (sm_120)"
gpu-infra learn -c platform --from path/to/lessons.md

# Review pending notes
gpu-infra inbox
```

## Commands

```bash
# Start RALPH loop
/ralph-loop:ralph-loop <task description>

# Cancel loop
/ralph-loop:cancel-ralph

# Validate
pre-commit run -a
checkov -d .
terraform fmt -recursive
```
