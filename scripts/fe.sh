#!/usr/bin/env bash
# fe — Field Engineer AI CLI
# Wraps mdc + gpu-infra and adds field note lifecycle commands.
# Usage: fe <command> [args]

set -euo pipefail

FE_VERSION="0.1.0"
CARD_FORMAT_DOC="docs/card-format.md"
COMMUNITY_REPO="https://github.com/field-engineer-ai/cards"
# Resolve through symlinks (fe is typically symlinked onto PATH, e.g. ~/.local/bin/fe)
# so REPO_ROOT points at the real repo, not the symlink's directory.
_fe_src="${BASH_SOURCE[0]}"
while [[ -L "$_fe_src" ]]; do
  _fe_dir="$(cd -P "$(dirname "$_fe_src")" && pwd)"
  _fe_src="$(readlink "$_fe_src")"
  [[ "$_fe_src" != /* ]] && _fe_src="$_fe_dir/$_fe_src"
done
REPO_ROOT="$(cd -P "$(dirname "$_fe_src")/.." && pwd)"

usage() {
  cat <<EOF
fe $FE_VERSION — Field Engineer AI CLI

Commands:
  fe learn <blueprint-path>             Run learn commands from lessons.md frontmatter
  fe contribute <blueprint-path>        Generate community contribution PR template
  fe agent <launch|status|logs|attach|stop|ls> [args]
                                        Drive the agent-runner runtime (auto-resolves AGENT_RUNNER_* env)
  fe help                               Show this help

Examples:
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

# fe agent — thin wrapper over the agent-runner CLI (sibling repo).
# Resolves AGENT_RUNNER_* env once (so you don't re-export 5 vars per shell), then
# passes the verb through unchanged. Env resolution order:
#   1. already-exported AGENT_RUNNER_* vars win (no override)
#   2. else source the saved env file if present
#   3. else derive from `terraform output -raw cli_env` in the blueprint (+ run_role_arn), and cache it
# Locates the agent-runner CLI at ../agent-runner/bin/agent-runner relative to this repo.
cmd_agent() {
  local ar_bin="$REPO_ROOT/../agent-runner/bin/agent-runner"
  [[ -x "$ar_bin" ]] || { echo "agent-runner CLI not found at $ar_bin (clone github.com/sublimotion/agent-runner alongside this repo)" >&2; exit 1; }

  local bp="$REPO_ROOT/domains/agent-runtime/blueprints/managed-agent-runner"
  local env_file="${AGENT_RUNNER_ENV_FILE:-$HOME/.config/agent-runner/env}"

  # (2) saved env file — fills vars; already-exported ones still win because the file uses plain exports
  if [[ -f "$env_file" ]]; then
    set -a; source "$env_file"; set +a
  fi

  # (3) derive from terraform if core vars still unset and a state exists
  if [[ -z "${AGENT_RUNNER_STATE_TABLE:-}" || -z "${AGENT_RUNNER_ARTIFACT_BUCKET:-}" ]] \
     && { [[ -d "$bp/.terraform" ]] || [[ -f "$bp/terraform.tfstate" ]]; } && command -v terraform >/dev/null 2>&1; then
    local cli_env role
    cli_env="$(cd "$bp" && terraform output -raw cli_env 2>/dev/null)" || cli_env=""
    if [[ -n "$cli_env" ]]; then
      eval "$cli_env"
      role="$(cd "$bp" && terraform output -raw run_role_arn 2>/dev/null)" || role=""
      [[ -n "$role" ]] && export AGENT_RUNNER_RUN_ROLE_ARN="$role"
      mkdir -p "$(dirname "$env_file")"
      # Preserve operator-set extras (cluster default, base image) across a regen —
      # carry forward any line for a var the terraform output doesn't provide.
      local extras=""
      [[ -f "$env_file" ]] && extras="$(grep -E '^export (AGENT_RUNNER_DEFAULT_CLUSTER|AGENT_RUNNER_BASE_IMAGE|AGENT_RUNNER_NODE_POOL)=' "$env_file" 2>/dev/null || true)"
      { echo "$cli_env"; [[ -n "$role" ]] && echo "export AGENT_RUNNER_RUN_ROLE_ARN=$role"; [[ -n "$extras" ]] && echo "$extras"; } > "$env_file"
      echo "fe agent: cached env → $env_file (preserved operator extras)" >&2
    fi
  fi

  : "${AGENT_RUNNER_NAMESPACE:=agent-runner}"; export AGENT_RUNNER_NAMESPACE
  exec "$ar_bin" "$@"
}

# Main dispatch
case "${1:-help}" in
  learn)       shift; cmd_learn "$@" ;;
  contribute)  shift; cmd_contribute "$@" ;;
  agent)       shift; cmd_agent "$@" ;;
  version)     echo "fe $FE_VERSION" ;;
  help|--help|-h) usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
esac
