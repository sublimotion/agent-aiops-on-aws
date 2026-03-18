#!/usr/bin/env bash
# fe — Field Engineer AI CLI
# Wraps mdc + gpu-infra and adds field note lifecycle commands.
# Usage: fe <command> [args]

set -euo pipefail

FE_VERSION="0.1.0"
CARD_FORMAT_DOC="docs/card-format.md"
COMMUNITY_REPO="https://github.com/field-engineer-ai/cards"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
fe $FE_VERSION — Field Engineer AI CLI

Commands:
  fe card <model> [--engine <engine>]   Look up model deployment card
  fe card --hardware <instance>         Look up GPU infrastructure card
  fe learn <blueprint-path>             Run learn commands from lessons.md frontmatter
  fe contribute <blueprint-path>        Generate community contribution PR template
  fe help                               Show this help

Examples:
  fe card glm-5 --engine sglang
  fe card --hardware p6-b200
  fe learn domains/gpu-serving/blueprints/glm5/
  fe contribute domains/gpu-serving/blueprints/glm5/

Card format spec: $CARD_FORMAT_DOC
EOF
}

# Parse YAML frontmatter from lessons.md using python3 + yaml
# Returns each value of a top-level key on its own line
parse_frontmatter() {
  local file="$1"
  local key="$2"
  python3 - "$file" "$key" <<'PYEOF'
import sys

try:
    import yaml
except ImportError:
    sys.exit(0)

filepath, key = sys.argv[1], sys.argv[2]
with open(filepath) as f:
    content = f.read()

# Extract between first --- and second ---
parts = content.split('---')
if len(parts) < 3:
    sys.exit(0)

try:
    data = yaml.safe_load(parts[1])
except Exception:
    sys.exit(0)

if not isinstance(data, dict):
    sys.exit(0)

val = data.get(key)
if val is None:
    sys.exit(0)
elif isinstance(val, list):
    for item in val:
        if item is not None:
            print(str(item))
else:
    print(str(val))
PYEOF
}

cmd_card() {
  local hardware_mode=false
  local model=""
  local engine=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --hardware) hardware_mode=true; shift; hardware="$1"; shift ;;
      --engine)   shift; engine="$1"; shift ;;
      *)          model="$1"; shift ;;
    esac
  done

  if $hardware_mode; then
    echo "→ GPU infrastructure card: $hardware"
    echo ""
    gpu-infra card "$hardware"
  elif [[ -n "$model" ]]; then
    echo "→ Model deployment card: $model${engine:+ ($engine)}"
    echo ""
    if [[ -n "$engine" ]]; then
      mdc get "$model" --engine "$engine"
    else
      mdc get "$model"
    fi
  else
    echo "Usage: fe card <model> [--engine <engine>]" >&2
    echo "       fe card --hardware <instance>" >&2
    exit 1
  fi
}

cmd_learn() {
  local blueprint_path="${1:-}"
  if [[ -z "$blueprint_path" ]]; then
    echo "Usage: fe learn <blueprint-path>" >&2
    exit 1
  fi

  local lessons_file="$REPO_ROOT/$blueprint_path/lessons.md"
  if [[ ! -f "$lessons_file" ]]; then
    # Try as absolute path
    lessons_file="$blueprint_path/lessons.md"
  fi
  if [[ ! -f "$lessons_file" ]]; then
    echo "Error: lessons.md not found at $blueprint_path" >&2
    exit 1
  fi

  echo "→ Reading field note: $lessons_file"
  echo ""

  # Show frontmatter summary
  local model engine hardware outcome
  model=$(parse_frontmatter "$lessons_file" "model" | head -1)
  engine=$(parse_frontmatter "$lessons_file" "engine" | head -1)
  hardware=$(parse_frontmatter "$lessons_file" "hardware" | head -1)
  outcome=$(parse_frontmatter "$lessons_file" "outcome" | head -1)

  echo "  Model:    ${model:-(not set)}"
  echo "  Engine:   ${engine:-(not set)}"
  echo "  Hardware: ${hardware:-(not set)}"
  echo "  Outcome:  ${outcome:-(not set)}"
  echo ""

  # Run mdc learn commands
  local mdc_cmds_raw mdc_count=0
  mdc_cmds_raw=$(parse_frontmatter "$lessons_file" "mdc_learn_commands")
  if [[ -n "$mdc_cmds_raw" ]]; then
    echo "→ Running mdc learn commands:"
    while IFS= read -r cmd; do
      if [[ -n "$cmd" ]]; then
        echo "  $ $cmd"
        eval "$cmd" || echo "  (skipped — mdc not available)"
        mdc_count=$((mdc_count + 1))
      fi
    done <<< "$mdc_cmds_raw"
    echo ""
  else
    echo "  No mdc_learn_commands in frontmatter."
  fi

  # Run gpu-infra learn commands
  local gpu_cmds_raw gpu_count=0
  gpu_cmds_raw=$(parse_frontmatter "$lessons_file" "gpu_infra_learn_commands")
  if [[ -n "$gpu_cmds_raw" ]]; then
    echo "→ Running gpu-infra learn commands:"
    while IFS= read -r cmd; do
      if [[ -n "$cmd" ]]; then
        echo "  $ $cmd"
        eval "$cmd" || echo "  (skipped — gpu-infra not available)"
        gpu_count=$((gpu_count + 1))
      fi
    done <<< "$gpu_cmds_raw"
    echo ""
  else
    echo "  No gpu_infra_learn_commands in frontmatter."
  fi

  echo "→ Done. To share this field note with the community:"
  echo "  fe contribute $blueprint_path"
}

cmd_contribute() {
  local blueprint_path="${1:-}"
  if [[ -z "$blueprint_path" ]]; then
    echo "Usage: fe contribute <blueprint-path>" >&2
    exit 1
  fi

  local lessons_file="$REPO_ROOT/$blueprint_path/lessons.md"
  if [[ ! -f "$lessons_file" ]]; then
    lessons_file="$blueprint_path/lessons.md"
  fi
  if [[ ! -f "$lessons_file" ]]; then
    echo "Error: lessons.md not found at $blueprint_path" >&2
    exit 1
  fi

  # Extract frontmatter fields for the PR template
  local model engine hardware gpu_arch outcome card_helped
  model=$(parse_frontmatter "$lessons_file" "model" | head -1)
  engine=$(parse_frontmatter "$lessons_file" "engine" | head -1)
  hardware=$(parse_frontmatter "$lessons_file" "hardware" | head -1)
  gpu_arch=$(parse_frontmatter "$lessons_file" "gpu_arch" | head -1)
  outcome=$(parse_frontmatter "$lessons_file" "outcome" | head -1)
  card_helped=$(parse_frontmatter "$lessons_file" "card_helped" | head -1)

  local failure_categories
  failure_categories=$(parse_frontmatter "$lessons_file" "failure_categories" | tr '\n' ' ' | xargs)

  echo "→ Community contribution for: $blueprint_path"
  echo ""
  echo "This will open a GitHub Issue on the community card repo."
  echo "Repo: $COMMUNITY_REPO"
  echo ""
  echo "Field note summary:"
  echo "  Model:             ${model:-(not set)}"
  echo "  Engine:            ${engine:-(not set)}"
  echo "  Hardware:          ${hardware:-(not set)}"
  echo "  GPU arch:          ${gpu_arch:-(not set)}"
  echo "  Outcome:           ${outcome:-(not set)}"
  echo "  Card helped:       ${card_helped:-(not set)}"
  if [[ -n "$failure_categories" ]]; then
    echo "  Failure categories: $failure_categories"
  fi
  echo ""

  # Generate issue body
  local issue_title="Field note: $model ($engine) on $hardware — $outcome"
  local issue_body
  issue_body=$(cat <<ISSUE
## Field Note

| Field | Value |
|-------|-------|
| Model | \`$model\` |
| Engine | \`$engine\` |
| Hardware | \`$hardware\` |
| GPU arch | \`$gpu_arch\` |
| Outcome | \`$outcome\` |
| Card helped | \`$card_helped\` |
| Failure categories | \`${failure_categories:-none}\` |

## Frontmatter

\`\`\`yaml
$(python3 -c "
import sys
with open('$lessons_file') as f:
    content = f.read()
parts = content.split('---')
if len(parts) >= 3:
    print('---' + parts[1] + '---')
")
\`\`\`

## Lessons

*(paste relevant sections from your lessons.md)*

---
*Generated by \`fe contribute\` from [agent-aiops-on-aws](https://github.com/field-engineer-ai/agent-aiops-on-aws)*
ISSUE
)

  echo "GitHub Issue title:"
  echo "  $issue_title"
  echo ""
  echo "To open the issue, run:"
  echo ""
  echo "  gh issue create \\"
  echo "    --repo field-engineer-ai/cards \\"
  echo "    --title \"$issue_title\" \\"
  echo "    --body \"\$(fe contribute $blueprint_path --body-only)\""
  echo ""
  echo "Or open the repo and create an issue manually:"
  echo "  $COMMUNITY_REPO/issues/new"
}

# Main dispatch
case "${1:-help}" in
  card)        shift; cmd_card "$@" ;;
  learn)       shift; cmd_learn "$@" ;;
  contribute)  shift; cmd_contribute "$@" ;;
  version)     echo "fe $FE_VERSION" ;;
  help|--help|-h) usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
esac
