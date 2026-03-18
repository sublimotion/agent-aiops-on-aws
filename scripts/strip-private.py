#!/usr/bin/env python3
"""
Strip private tool dependencies from the repo before syncing to the public OSS repo.

Removes references to:
  - mdc (model-deployment-card, private repo)
  - gpu-infra CLI (gpu-infra-troubleshooting, private repo)

Retains:
  - The RALPH loop framework, spec-driven development, deployer agents
  - The fe CLI (fe learn, fe contribute — card-system agnostic)
  - The field note schema (docs/card-format.md)
  - GPU MCP tool interface (the concept, not the private implementation)
  - The compound-learner agent (steering elevation only)

Run from the repo root before pushing to the public repo.
"""

import re
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(path, content):
    full = os.path.join(ROOT, path)
    with open(full, "w") as f:
        f.write(content)
    print(f"  stripped: {path}")


def read(path):
    full = os.path.join(ROOT, path)
    with open(full) as f:
        return f.read()


# ---------------------------------------------------------------------------
# CLAUDE.md
# ---------------------------------------------------------------------------

claude_md = """\
# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

## Workflow

1. Write spec in the correct domain location (see Domain Routing below)
2. Load the deployment card for your model and hardware (see `docs/card-format.md`)
3. Run `/ralph-loop:ralph-loop Deploy <spec-path>`
4. Claude selects the deployer agent automatically based on spec path (see Domain Routing)
5. Claude iterates until deployment succeeds
6. Capture lessons in the blueprint's `lessons.md` using the field note schema (see `docs/card-format.md` and `domains/gpu-serving/blueprints/LESSONS-TEMPLATE.md`)
7. Run compound step: invoke `compound-learner` agent — it elevates cross-cutting lessons to `.claude/steering/*.md` **and generates the YAML frontmatter** for `lessons.md` (model, engine, hardware, outcome, failure_categories, learn_commands)
8. Run `scripts/fe.sh learn <blueprint-path>` to apply the learn commands from the frontmatter
9. Optionally run `scripts/fe.sh contribute <blueprint-path>` to open a community contribution issue

## Context Loading

Read these files **on demand** based on what you're doing:

| When you are... | Read |
|-----------------|------|
| Writing or modifying Terraform | `.claude/steering/tech-stack.md` |
| Making architectural or product decisions | `.claude/steering/product.md` |
| Creating new files, modules, or blueprints | `.claude/steering/project-structure.md` |
| About to deploy a model | Load the deployment card for your model + hardware (see `docs/card-format.md`) |
| Deploying or modifying a GPU-serving blueprint | `domains/gpu-serving/specs/<matching-spec>.md` |
| Deploying an agent-runtime blueprint | `domains/agent-runtime/specs/<name>.md` |
| Running an autoresearch experiment | `domains/autoresearch/specs/<name>.md` |
| Running the compound step after deployment | `domains/<domain>/blueprints/<name>/lessons.md` + `.claude/steering/*.md`; compound-learner must generate YAML frontmatter (see `docs/card-format.md`) as its final step, then run `scripts/fe.sh learn <path>` |
| Running pre-flight or post-deploy checks | Use the `deployment-orchestrator` skill |
| Validating GPU hardware before serving (Stage 4a) | Use a GPU diagnostics MCP server — see `External Tools` below |
| Diagnosing GPU hardware issues | Use a GPU diagnostics MCP server — see `External Tools` below |

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
| `domains/gpu-serving/blueprints/glm5-hyperpod/` | `domains/gpu-serving/specs/glm5-hyperpod.md` |
| `domains/gpu-serving/blueprints/glm5-lmcache/` | `domains/gpu-serving/specs/glm5-lmcache.md` |
| `domains/gpu-serving/blueprints/glm5-llmd/` | `domains/gpu-serving/specs/glm5-llmd.md` |
| `domains/gpu-serving/blueprints/nemotron-super/` | `domains/gpu-serving/specs/nemotron-super.md` |
| `domains/agent-runtime/blueprints/research-agent/` | `domains/agent-runtime/specs/research-agent.md` |
| `domains/autoresearch/blueprints/training-recipes/` | `domains/autoresearch/specs/training-recipes.md` |
| `domains/autoresearch/blueprints/agent-harness/` | `domains/autoresearch/specs/agent-harness.md` |
| `domains/autoresearch/blueprints/finetuning-recipes/` | `domains/autoresearch/specs/finetuning-recipes.md` |

**Blueprint-local context** — for operational details (lessons, results, plans), look inside the blueprint directory itself rather than in specs.

## Domain Routing

The repo is organized into domains. **Infer the deployer agent automatically from the spec path** — no need to specify it explicitly in the RALPH loop command.

| Domain | Spec path prefix | Blueprint location | Deployer agent |
|--------|------------------|--------------------|----------------|
| GPU Serving | `domains/gpu-serving/specs/` | `domains/gpu-serving/blueprints/` | `infra-deployer` |
| Agent Runtime | `domains/agent-runtime/specs/` | `domains/agent-runtime/blueprints/` | `agentcore-deployer` |
| Autoresearch | `domains/autoresearch/specs/` | `domains/autoresearch/blueprints/` | `autoresearch-runner` |

**Auto-detection rule**: all specs live under `domains/<name>/specs/`. Use `domains/gpu-serving/` → `infra-deployer`, `domains/agent-runtime/` → `agentcore-deployer`, `domains/autoresearch/` → `autoresearch-runner`.

Examples:
```
/ralph-loop:ralph-loop Deploy domains/agent-runtime/specs/research-agent.md   → agentcore-deployer
/ralph-loop:ralph-loop Deploy domains/gpu-serving/specs/kimi-k2.5.md          → infra-deployer
/ralph-loop:ralph-loop Run domains/autoresearch/specs/training-recipes.md     → autoresearch-runner
```

## External Tools

### fe CLI

Field note lifecycle and community contribution. Run from the repo root.

```bash
# After compound step — apply learn commands from lessons.md frontmatter
fe learn domains/gpu-serving/blueprints/<name>/

# Contribute field note to community (opens GitHub Issue template)
fe contribute domains/gpu-serving/blueprints/<name>/
```

### Deployment Cards (bring your own)

Before deploying, load the deployment card for your model and hardware.
The field note schema is defined in `docs/card-format.md` — it's card-system agnostic.

Plug in your card system of choice (e.g., a local markdown directory, a shared registry,
or the community card library at https://github.com/field-engineer-ai/cards).

### GPU Infrastructure Diagnostics (MCP)

For GPU hardware validation (Stage 4a), connect a GPU diagnostics MCP server that exposes:

| Tool | When to use |
|------|-------------|
| `discover_cluster` | Stage 4a — enumerate GPUs, topology, driver, EFA before serving |
| `check_gpu_health` | Stage 4a — validate ECC, row remap, thermals, PCIe link |
| `run_nccl_test` | Stage 4a — verify NCCL collectives for multi-GPU TP |
| `explain_xid` | Any time — look up Xid error codes from dmesg |
| `get_gpu_metrics` | Benchmarking — pull DCGM metrics from Prometheus |

Configure your MCP server in `.mcp.json`. See `.mcp.json.example` for the expected interface.

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
"""

write("CLAUDE.md", claude_md)


# ---------------------------------------------------------------------------
# scripts/fe.sh — remove fe card (wraps private mdc + gpu-infra CLIs)
# ---------------------------------------------------------------------------

fe_sh = read("scripts/fe.sh")

# Remove the cmd_card function
fe_sh = re.sub(
    r"\ncmd_card\(\) \{.*?\n\}\n",
    "\n",
    fe_sh,
    flags=re.DOTALL,
)

# Remove card from usage
fe_sh = fe_sh.replace(
    "  fe card <model> [--engine <engine>]   Look up model deployment card\n"
    "  fe card --hardware <instance>         Look up GPU infrastructure card\n",
    "",
)

# Remove card from examples
fe_sh = fe_sh.replace(
    "  fe card glm-5 --engine sglang\n"
    "  fe card --hardware p6-b200\n",
    "",
)

# Remove card dispatch
fe_sh = fe_sh.replace(
    "  card)        shift; cmd_card \"$@\" ;;\n",
    "",
)

write("scripts/fe.sh", fe_sh)


# ---------------------------------------------------------------------------
# .mcp.json — replace with example stub (no local paths)
# ---------------------------------------------------------------------------

mcp_example = """\
{
  "mcpServers": {
    "gpu-diagnostics": {
      "command": "your-gpu-mcp-server",
      "args": [],
      "env": {}
    }
  }
}
"""

write(".mcp.json", mcp_example)


# ---------------------------------------------------------------------------
# docs/card-format.md — remove mdc/gpu-infra specific CLI references
# ---------------------------------------------------------------------------

card_format = read("docs/card-format.md")

# Replace the mdc learn commands section
card_format = card_format.replace(
    "## Card format (model deployment cards)\n\n"
    "Model deployment cards live in the `model-deployment-card` repo (`mdc`). Format:\n\n"
    "```yaml\n"
    "---\n"
    "model: glm-5\n"
    "engine: sglang\n"
    "hardware_validated: [p6-b200.48xlarge]\n"
    "tp_degree: 8\n"
    "last_validated: 2026-03-10\n"
    "validation_count: 3\n"
    "---\n"
    "```\n\n"
    "Card body contains: launch command, known issues, upstream PR tracking, benchmark reference.\n\n"
    "## Card format (GPU infrastructure cards)\n\n"
    "GPU infra cards live in the `gpu-infra-troubleshooting` repo. Format:\n\n"
    "```yaml\n"
    "---\n"
    "instance_type: p6-b200.48xlarge\n"
    "gpu: B200\n"
    "gpu_arch: sm_120\n"
    "gpu_count: 8\n"
    "interconnect: NVSwitch\n"
    "efa_interfaces: 4\n"
    "driver_min: \"575.x\"\n"
    "nccl_min: \"2.26.2\"\n"
    "ami_required: al2023-nvidia\n"
    "last_validated: 2026-03-10\n"
    "---\n"
    "```\n\n"
    "Card body contains: architecture notes, EFA config, NCCL requirements, AMI requirements, known failure patterns.\n",

    "## Card format (model deployment cards)\n\n"
    "Model deployment cards cover model-specific deployment knowledge. Format:\n\n"
    "```yaml\n"
    "---\n"
    "model: glm-5\n"
    "engine: sglang\n"
    "hardware_validated: [p6-b200.48xlarge]\n"
    "tp_degree: 8\n"
    "last_validated: 2026-03-10\n"
    "validation_count: 3\n"
    "---\n"
    "```\n\n"
    "Card body contains: launch command, known issues, benchmark reference.\n\n"
    "## Card format (GPU infrastructure cards)\n\n"
    "GPU infra cards cover hardware-specific deployment knowledge. Format:\n\n"
    "```yaml\n"
    "---\n"
    "instance_type: p6-b200.48xlarge\n"
    "gpu: B200\n"
    "gpu_arch: sm_120\n"
    "gpu_count: 8\n"
    "interconnect: NVSwitch\n"
    "efa_interfaces: 4\n"
    "driver_min: \"575.x\"\n"
    "nccl_min: \"2.26.2\"\n"
    "ami_required: al2023-nvidia\n"
    "last_validated: 2026-03-10\n"
    "---\n"
    "```\n\n"
    "Card body contains: architecture notes, EFA config, NCCL requirements, AMI requirements, known failure patterns.\n\n"
    "Community cards are available at https://github.com/field-engineer-ai/cards\n",
)

# Remove private repo references from learn commands description
card_format = card_format.replace(
    "# Ready-to-run commands — copy-paste or let `fe learn` run them automatically\n"
    "mdc_learn_commands: []\n"
    "  # - 'mdc learn <model> <engine> \"<lesson>\"'\n"
    "  # - 'mdc learn <model> <engine> --from domains/<domain>/blueprints/<name>/lessons.md'\n"
    "\n"
    "gpu_infra_learn_commands: []\n"
    "  # - 'gpu-infra learn -c nccl \"<lesson>\"'\n"
    "  # - 'gpu-infra learn -c platform \"<lesson>\"'\n"
    "  # - 'gpu-infra learn -c ami \"<lesson>\"'\n",

    "# Ready-to-run commands — copy-paste or let `fe learn` run them automatically\n"
    "# Use whatever card system you've plugged in (see docs/card-format.md)\n"
    "learn_commands: []\n"
    "  # - 'my-card-tool learn <model> <engine> \"<lesson>\"'\n",
)

write("docs/card-format.md", card_format)


# ---------------------------------------------------------------------------
# domains/gpu-serving/blueprints/LESSONS-TEMPLATE.md — generalize learn commands
# ---------------------------------------------------------------------------

template = read("domains/gpu-serving/blueprints/LESSONS-TEMPLATE.md")

template = template.replace(
    "mdc_learn_commands: []\ngpu_infra_learn_commands: []",
    "learn_commands: []",
)

write("domains/gpu-serving/blueprints/LESSONS-TEMPLATE.md", template)


# ---------------------------------------------------------------------------
# .claude/agents/compound-learner.md — strip mdc/gpu-infra feedback sections
# ---------------------------------------------------------------------------

learner = read(".claude/agents/compound-learner.md")

# Remove the "GPU hardware lessons → gpu-infra repo" subsection
learner = re.sub(
    r"### GPU hardware lessons → gpu-infra repo\n.*?(?=### |\Z)",
    "",
    learner,
    flags=re.DOTALL,
)

# Remove "How to feed back" (gpu-infra learn commands block)
learner = re.sub(
    r"### How to feed back\n\nFor each hardware/platform lesson.*?```\n\n",
    "",
    learner,
    flags=re.DOTALL,
)

# Simplify steering file targets table — remove gpu-infra and mdc rows
learner = learner.replace(
    "| GPU hardware: new Xid patterns, NCCL bugs, driver issues, pass/fail thresholds | Flag for `gpu-infra` repo | See below |\n"
    "| Model-specific: configs, flags, known issues, Docker images, sizing rules | Feed back to `mdc learn` | See \"Feedback to mdc\" section |\n",
    "| GPU hardware: new Xid patterns, NCCL bugs, driver issues | Capture in `lessons.md` body + `learn_commands` frontmatter field |\n"
    "| Model-specific: configs, flags, known issues, Docker images, sizing rules | Capture in `lessons.md` body + `learn_commands` frontmatter field |\n",
)

# Remove "Feedback to Model Deployment Cards (mdc)" section
learner = re.sub(
    r"## Feedback to Model Deployment Cards \(mdc\)\n.*?## What not to do",
    "## What not to do",
    learner,
    flags=re.DOTALL,
)

# Update compound summary template — remove mdc/gpu-infra fed-back sections
learner = learner.replace(
    "### Fed back to mdc\n"
    "| Card | Engine | Note | Source |\n"
    "|------|--------|------|--------|\n"
    "\n"
    "### Fed back to gpu-infra\n"
    "| Category | Note | Target |\n"
    "|----------|------|--------|\n",
    "### Learn commands generated\n"
    "| Type | Command | Source |\n"
    "|------|---------|--------|\n",
)

# Update "Fed back to mdc" section reference in compound summary output
learner = learner.replace(
    "### Fed back to mdc\n```\n| Card | Engine | Note | Source |\n|------|--------|------|--------|\n```\n\n"
    "Add a `### Fed back to gpu-infra` section to the compound summary:\n\n"
    "```\n### Fed back to gpu-infra\n| Category | Note | Target |\n|----------|------|--------|\n```\n\n"
    "If no hardware lessons were found, omit this section.",
    "### Learn commands generated\n```\n| Type | Command | Source |\n|------|---------|--------|\n```",
)

# Update frontmatter section — remove mdc/gpu-infra specific command fields
learner = learner.replace(
    "mdc_learn_commands: []  # ready-to-run commands you've identified above\ngpu_infra_learn_commands: []  # ready-to-run commands you've identified above",
    "learn_commands: []      # ready-to-run commands you've identified above",
)

learner = learner.replace(
    'For `mdc_learn_commands` and `gpu_infra_learn_commands`: populate these with the exact commands from the "Fed back to mdc" and "Fed back to gpu-infra" sections of your compound summary. This lets `scripts/fe.sh learn` run them automatically without the operator needing to copy them manually.',
    'For `learn_commands`: populate these with the exact card system commands from your compound summary. This lets `scripts/fe.sh learn` run them automatically.',
)

write(".claude/agents/compound-learner.md", learner)


# ---------------------------------------------------------------------------
# Strip mdc/gpu-infra learn commands from existing lessons.md frontmatter
# Replaces mdc_learn_commands + gpu_infra_learn_commands with learn_commands
# across all blueprint lessons.md files
# ---------------------------------------------------------------------------

import glob

for lessons_path in glob.glob(os.path.join(ROOT, "domains/*/blueprints/*/lessons.md")):
    content = open(lessons_path).read()
    if "mdc_learn_commands" not in content and "gpu_infra_learn_commands" not in content:
        continue

    # Collect all mdc + gpu-infra learn commands into a single learn_commands list
    mdc_block = re.search(r"mdc_learn_commands:(.*?)(?=\ngpu_infra_learn_commands:|\n\w|\Z)", content, re.DOTALL)
    gpu_block = re.search(r"gpu_infra_learn_commands:(.*?)(?=\n---|\n\w|\Z)", content, re.DOTALL)

    combined_items = []
    for block in [mdc_block, gpu_block]:
        if block:
            items = re.findall(r"^\s+- (.+)$", block.group(1), re.MULTILINE)
            combined_items.extend(items)

    if combined_items:
        new_field = "learn_commands:\n" + "\n".join(f"  - {i}" for i in combined_items) + "\n"
    else:
        new_field = "learn_commands: []\n"

    # Replace both old fields with the new combined one
    content = re.sub(
        r"mdc_learn_commands:.*?(?=\ngpu_infra_learn_commands:)",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"gpu_infra_learn_commands:.*?(?=\n---)",
        new_field.rstrip(),
        content,
        flags=re.DOTALL,
    )

    rel = os.path.relpath(lessons_path, ROOT)
    write(rel, content)
    print(f"  updated frontmatter: {rel}")


print("\nDone. OSS-safe files written.")
