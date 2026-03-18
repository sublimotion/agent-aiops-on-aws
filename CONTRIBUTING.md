# Contributing to Field Engineer AI

Field Engineer AI gets better through community field notes — structured records of what worked, what failed, and how it was fixed on real GPU deployments.

**No telemetry. No automated data collection.** Contribution is explicit and GitHub-native.

---

## How to Contribute a Field Note

### 1. Run a blueprint

Clone the repo, pick a blueprint, and run the RALPH loop:

```bash
git clone https://github.com/field-engineer-ai/agent-aiops-on-aws
cd agent-aiops-on-aws

# Load cards before deploying
fe card <model> --engine <engine>
fe card --hardware <instance>

# Deploy
/ralph-loop:ralph-loop Deploy domains/gpu-serving/specs/<spec>.md
```

### 2. Fill in your lessons.md

After deployment, the compound-learner generates a `lessons.md` with YAML frontmatter. Fill in or verify the auto-generated fields:

```bash
# See the template
cat domains/gpu-serving/blueprints/LESSONS-TEMPLATE.md

# The compound-learner fills most fields automatically.
# Review and correct: outcome, failure_categories, card_helped, benchmark results.
```

See `docs/card-format.md` for the full schema and failure_categories enum.

### 3. Apply your lessons locally

```bash
scripts/fe.sh learn domains/gpu-serving/blueprints/<name>/
```

This runs the `mdc_learn_commands` and `gpu_infra_learn_commands` from your frontmatter — contributing to your local card knowledge.

### 4. Contribute to the community (optional)

```bash
scripts/fe.sh contribute domains/gpu-serving/blueprints/<name>/
```

This generates a pre-filled GitHub Issue template. Open the issue on the community card repo to share your field note with the community. The FE team reviews and converts it into a card update.

Alternatively, open a PR directly to the [community card repo](https://github.com/field-engineer-ai/cards) if you want to propose a specific card addition or update.

---

## What Makes a Good Field Note

**High signal:**
- New model × hardware combination not yet covered by existing cards
- Failure with a specific fix (failure_category + what fixed it)
- Benchmark results at a specific concurrency level
- "The card was wrong about X, here's what actually works"

**Low signal (still useful, but less so):**
- Successful deployment with no issues (confirms a card is still accurate)
- Failure without a resolution

**Please omit:**
- AWS account IDs, instance IDs, IP addresses
- Proprietary model weights or internal configs
- Team or company names (unless you want attribution)

---

## Contributing a New Card

If you've developed knowledge about a model/hardware combination not in the community library, you can propose a new card directly:

1. Use the card format in `docs/card-format.md`
2. Open a PR to the [community card repo](https://github.com/field-engineer-ai/cards)
3. Include at least one field note as evidence (link to your issue or paste the frontmatter)

The FE team validates new cards against known hardware behavior before merging.

---

## Reporting Issues

**Stale card:** If a card's recommendation no longer works, open an issue on the community card repo with:
- Which card (model + engine + hardware)
- What the card said
- What actually happened
- Your field note frontmatter (just the YAML block, no prose needed)

**Framework bug:** Open an issue on this repo (`agent-aiops-on-aws`) with steps to reproduce.

---

## Code Contributions

For contributions to the framework itself (deployer agents, compound-learner, RALPH loop integration):

1. Open an issue first to discuss significant changes
2. Run `pre-commit install` after cloning
3. Validate locally: `pre-commit run -a`
4. Include evidence: test output or deployment results in PRs
5. Security scan: `checkov -d .` must pass
