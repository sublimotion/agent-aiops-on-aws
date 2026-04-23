#!/usr/bin/env bash
# setup-droid.sh — One-time bootstrap to derive .factory/ from .claude/
#
# Creates:
#   .factory/droids/*.md    — converted from .claude/agents/*.md (tool name translation)
#   .factory/skills/        — symlinks to .claude/skills/*
#   .factory/mcp.json       — symlink to .mcp.json
#   .factory/settings.json  — preserved if it exists, created otherwise
#
# Safe to re-run: overwrites droids, re-creates symlinks, skips settings.json if present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAUDE_DIR="$REPO_ROOT/.claude"
FACTORY_DIR="$REPO_ROOT/.factory"

# ---------- helpers ----------

log()  { printf "  %-50s" "$1"; }
ok()   { echo "[ok]"; }
skip() { echo "[skip] $1"; }

# Convert Claude Code agent frontmatter tools to Factory Droid tools.
# Mapping: Bash → Execute, Write → Edit + Create
# Factory droids use a JSON array for explicit tool lists.
map_tools() {
  local claude_tools="$1"
  local result=()

  IFS=', ' read -ra parts <<< "$claude_tools"
  for t in "${parts[@]}"; do
    t="$(echo "$t" | xargs)"  # trim whitespace
    case "$t" in
      Bash)   result+=("Execute") ;;
      Write)  result+=("Edit" "Create") ;;
      Read|Glob|Grep) result+=("$t") ;;
      *)      result+=("$t") ;;  # pass through unknown tools
    esac
  done

  # Deduplicate
  local seen=()
  local unique=()
  for item in "${result[@]}"; do
    local found=false
    for s in "${seen[@]+"${seen[@]}"}"; do
      [[ "$s" == "$item" ]] && found=true && break
    done
    if ! $found; then
      unique+=("$item")
      seen+=("$item")
    fi
  done

  # Format as JSON array: ["Read", "Grep", ...]
  local json="["
  for i in "${!unique[@]}"; do
    [[ $i -gt 0 ]] && json+=", "
    json+="\"${unique[$i]}\""
  done
  json+="]"
  echo "$json"
}

# Convert Claude Code model family to Factory model identifier.
# Uses "inherit" for most cases; specific model IDs can be mapped if needed.
map_model() {
  local claude_model="$1"
  case "$claude_model" in
    opus)    echo "inherit" ;;  # maps to parent session; change to specific ID if needed
    sonnet)  echo "inherit" ;;
    haiku)   echo "inherit" ;;
    inherit) echo "inherit" ;;
    *)       echo "inherit" ;;
  esac
}

# ---------- main ----------

echo "=== Droid bootstrap: .claude/ → .factory/ ==="
echo ""

# 1. Ensure directories exist
mkdir -p "$FACTORY_DIR/droids"
mkdir -p "$FACTORY_DIR/skills"

# 2. Convert agents → droids
echo "--- Agents → Droids ---"

for agent_file in "$CLAUDE_DIR"/agents/*.md; do
  [[ ! -f "$agent_file" ]] && continue
  basename="$(basename "$agent_file")"
  droid_file="$FACTORY_DIR/droids/$basename"
  log "  $basename"

  # Parse YAML frontmatter (between first two --- lines)
  local_name=""
  local_desc=""
  local_tools=""
  local_model=""
  in_frontmatter=false
  frontmatter_done=false
  has_frontmatter=false
  body=""

  while IFS= read -r line; do
    if ! $frontmatter_done; then
      if [[ "$line" == "---" ]]; then
        if $in_frontmatter; then
          frontmatter_done=true
          has_frontmatter=true
        else
          in_frontmatter=true
        fi
        continue
      fi
      if $in_frontmatter; then
        case "$line" in
          name:*)        local_name="$(echo "$line" | sed 's/^name:[[:space:]]*//')" ;;
          description:*) local_desc="$(echo "$line" | sed 's/^description:[[:space:]]*//')" ;;
          tools:*)       local_tools="$(echo "$line" | sed 's/^tools:[[:space:]]*//')" ;;
          model:*)       local_model="$(echo "$line" | sed 's/^model:[[:space:]]*//')" ;;
        esac
      else
        # First non-frontmatter line and we never saw opening ---
        # This file has no frontmatter; treat entire content as body
        frontmatter_done=true
        body+="$line"$'\n'
      fi
    else
      body+="$line"$'\n'
    fi
  done < "$agent_file"

  # For files without frontmatter, derive name from filename
  if ! $has_frontmatter; then
    local_name="$(echo "$basename" | sed 's/\.md$//')"
    # Extract description from first markdown heading or first line
    local_desc="$(head -1 "$agent_file" | sed 's/^#* *//')"
  fi

  # Map tools and model
  mapped_tools=""
  if [[ -n "$local_tools" ]]; then
    mapped_tools="$(map_tools "$local_tools")"
  fi
  mapped_model="$(map_model "${local_model:-inherit}")"

  # Write the droid file
  {
    echo "---"
    echo "name: $local_name"
    echo "description: $local_desc"
    echo "model: $mapped_model"
    if [[ -n "$mapped_tools" ]]; then
      echo "tools: $mapped_tools"
    fi
    echo "---"
    echo ""
    printf '%s' "$body"
  } > "$droid_file"

  ok
done

echo ""

# 3. Symlink skills
echo "--- Skills (symlinks) ---"

for skill_dir in "$CLAUDE_DIR"/skills/*/; do
  [[ ! -d "$skill_dir" ]] && continue
  skill_name="$(basename "$skill_dir")"

  # Skip the tests directory (not a skill)
  [[ "$skill_name" == "tests" ]] && continue

  target="$FACTORY_DIR/skills/$skill_name"
  log "  $skill_name"

  # Remove existing symlink or directory before creating
  if [[ -L "$target" ]]; then
    rm "$target"
  elif [[ -d "$target" ]]; then
    rm -rf "$target"
  fi

  # Compute relative path for symlink
  ln -s "../../.claude/skills/$skill_name" "$target"
  ok
done

echo ""

# 4. Symlink MCP config
echo "--- MCP config ---"
log "  .mcp.json → .factory/mcp.json"

mcp_target="$FACTORY_DIR/mcp.json"
if [[ -L "$mcp_target" ]]; then
  rm "$mcp_target"
fi
if [[ -f "$REPO_ROOT/.mcp.json" ]]; then
  ln -s "../.mcp.json" "$mcp_target"
  ok
else
  skip "no .mcp.json at repo root"
fi

echo ""

# 5. Settings (preserve existing, create default if missing)
echo "--- Settings ---"
log "  .factory/settings.json"

settings_file="$FACTORY_DIR/settings.json"
if [[ -f "$settings_file" ]]; then
  skip "already exists"
else
  cat > "$settings_file" <<'EOF'
{
  "enabledPlugins": {
    "core@factory-plugins": true
  }
}
EOF
  ok
fi

echo ""

# 6. Ensure .factory/ is in .gitignore (except settings.json pattern)
echo "--- Gitignore ---"
log "  .factory/ in .gitignore"

gitignore="$REPO_ROOT/.gitignore"
if grep -qxF '.factory/' "$gitignore" 2>/dev/null; then
  skip "already present"
else
  {
    echo ""
    echo "# Derived from .claude/ via scripts/setup-droid.sh"
    echo ".factory/"
  } >> "$gitignore"
  ok
fi

echo ""
echo "=== Done. .factory/ is ready for Droid CLI. ==="
echo ""
echo "Next steps:"
echo "  1. Run 'droid' in this repo — it will pick up droids, skills, and MCP config"
echo "  2. Type /droids to see imported agents"
echo "  3. Type /mcp to verify MCP servers"
echo "  4. Re-run this script any time you update .claude/agents/ or .claude/skills/"
