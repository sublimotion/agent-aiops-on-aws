#!/usr/bin/env bash
# progress.sh — Reconstruct or generate results/progress.md for a blueprint
#
# Usage:
#   scripts/progress.sh <blueprint-path>
#   scripts/progress.sh domains/gpu-serving/blueprints/ray-serve-ft
#   scripts/progress.sh all    # regenerate for all blueprints
#
# Parses existing artifacts (deployment logs, audits, lessons, spec, results)
# to reconstruct the progress state.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- helpers ----------

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Detect domain from blueprint path
detect_domain() {
  local bp="$1"
  if [[ "$bp" == *"gpu-serving"* ]]; then echo "gpu-serving"
  elif [[ "$bp" == *"agent-runtime"* ]]; then echo "agent-runtime"
  elif [[ "$bp" == *"autoresearch"* ]]; then echo "autoresearch"
  else echo "unknown"
  fi
}

# Get blueprint name from path
blueprint_name() {
  basename "$1"
}

# Find matching spec for a blueprint
find_spec() {
  local bp_dir="$1"
  local domain="$2"
  local name
  name="$(blueprint_name "$bp_dir")"
  local spec_path="$REPO_ROOT/domains/$domain/specs/$name.md"
  if [[ -f "$spec_path" ]]; then
    echo "domains/$domain/specs/$name.md"
  else
    echo ""
  fi
}

# Extract deployer stages for a domain
deployer_stages() {
  local domain="$1"
  case "$domain" in
    gpu-serving)
      cat <<'EOF'
stage-0|Deployment card lookup
stage-1|Foundation
stage-2|Build machine
stage-3|Storage and model staging
stage-4|Capacity reservation and GPU node
stage-4a|GPU health validation
stage-5|Serving stack deployment
stage-6|Pre-benchmark validation
stage-7|Readiness audit
stage-8|Compound
EOF
      ;;
    agent-runtime)
      cat <<'EOF'
stage-1|Foundation (Terraform)
stage-2|Container Build
stage-3|AgentCore Runtime
stage-4|Auth Wiring (Cognito)
stage-5|WebSocket Proxy
stage-6|Integration Test
stage-7|Readiness Audit
stage-8|Compound
EOF
      ;;
    autoresearch)
      cat <<'EOF'
stage-1|Read Spec
stage-2|Validate Environment
stage-3|Setup Codebase
stage-4|Configure Loop
stage-5|Run Baseline
stage-6|Execute Loop
stage-7|Analyze Results
stage-8|Capture Lessons
EOF
      ;;
    *)
      echo ""
      ;;
  esac
}

# Extract spec-defined phases (T1, T2, ... or Phase 0, Phase 1, ...)
extract_spec_phases() {
  local spec_file="$1"
  [[ ! -f "$spec_file" ]] && return

  # Match lines like: ### T1: Replica Crash Recovery
  #                    ### Phase 0: Infrastructure (2 hrs)
  grep -E '^### (T[0-9]+|Phase [0-9]+):' "$spec_file" 2>/dev/null | while IFS= read -r line; do
    local id name
    id="$(echo "$line" | sed -E 's/^### (T[0-9]+|Phase [0-9]+):.*/\1/' | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
    name="$(echo "$line" | sed -E 's/^### (T[0-9]+|Phase [0-9]+): //')"
    echo "$id|$name"
  done
}

# Detect stage status from deployment log content
# Looks for patterns like "Stage N" + "COMPLETE|PASS|FAIL|SKIPPED"
detect_stage_status_from_logs() {
  local bp_dir="$1"
  local stage_id="$2"

  # Extract stage number for matching
  local stage_num
  stage_num="$(echo "$stage_id" | sed 's/stage-//')"

  local status="not_started"

  # Search deployment logs for stage references
  for log_file in "$bp_dir"/results/deployment-log-*.md; do
    [[ ! -f "$log_file" ]] && continue

    # Match patterns: "Stage N:" with "COMPLETE", "PASS", "FAIL", "SKIPPED"
    if grep -qiE "(stage\s*${stage_num}[^0-9]|stage\s*${stage_num}$)" "$log_file" 2>/dev/null; then
      if grep -iE "(stage\s*${stage_num})" "$log_file" 2>/dev/null | grep -qiE "(COMPLETE|PASS|\[ok\]|✅)" 2>/dev/null; then
        status="complete"
      elif grep -iE "(stage\s*${stage_num})" "$log_file" 2>/dev/null | grep -qiE "(SKIP)" 2>/dev/null; then
        status="skipped"
      elif grep -iE "(stage\s*${stage_num})" "$log_file" 2>/dev/null | grep -qiE "(FAIL|BLOCK|ERROR)" 2>/dev/null; then
        status="blocked"
      else
        status="in_progress"
      fi
    fi
  done

  echo "$status"
}

# Detect phase status from results files
detect_phase_status() {
  local bp_dir="$1"
  local phase_id="$2"

  local status="not_started"

  # Check ft_summary.md, benchmark-report.md, or results/*.jsonl for phase references
  local phase_upper
  phase_upper="$(echo "$phase_id" | tr '[:lower:]' '[:upper:]' | tr '-' ' ')"

  for result_file in "$bp_dir"/results/ft_summary.md "$bp_dir"/results/benchmark-report.md "$bp_dir"/results/benchmark-report-*.md; do
    [[ ! -f "$result_file" ]] && continue
    if grep -qi "$phase_upper" "$result_file" 2>/dev/null; then
      status="complete"
      break
    fi
  done

  # Also check for JSONL traffic files (ray-serve-ft pattern: traffic_T1_*.jsonl)
  local phase_tag
  phase_tag="$(echo "$phase_id" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
  for jsonl in "$bp_dir"/results/traffic_${phase_tag}_*.jsonl "$bp_dir"/results/*_${phase_tag}*.jsonl; do
    [[ -f "$jsonl" ]] && status="complete" && break
  done

  echo "$status"
}

# Collect artifact metadata
collect_artifacts() {
  local bp_dir="$1"

  local has_lessons="false"
  [[ -f "$bp_dir/lessons.md" ]] && has_lessons="true"

  local audits=()
  for f in "$bp_dir"/results/readiness-audit-*.md; do
    [[ -f "$f" ]] && audits+=("$(basename "$f" .md | sed 's/^readiness-audit-//')")
  done

  local logs=()
  for f in "$bp_dir"/results/deployment-log-*.md; do
    [[ -f "$f" ]] && logs+=("$(basename "$f" .md | sed 's/^deployment-log-//')")
  done

  local compounds=()
  for f in "$bp_dir"/results/compound-*.md; do
    [[ -f "$f" ]] && compounds+=("$(basename "$f" .md | sed 's/^compound-//')")
  done

  local has_benchmark="false"
  [[ -f "$bp_dir/results/benchmark-report.md" ]] && has_benchmark="true"
  for f in "$bp_dir"/results/benchmark-report-*.md; do
    [[ -f "$f" ]] && has_benchmark="true" && break
  done

  echo "lessons:$has_lessons"
  echo "audits:${audits[*]:-}"
  echo "logs:${logs[*]:-}"
  echo "compounds:${compounds[*]:-}"
  echo "benchmark:$has_benchmark"
}

# Check spec status line (e.g., "## Status: COMPLETE")
detect_spec_status() {
  local spec_file="$1"
  [[ ! -f "$spec_file" ]] && echo "unknown" && return

  local status_line
  status_line="$(grep -iE '^## Status:' "$spec_file" 2>/dev/null | head -1)"
  if [[ -z "$status_line" ]]; then
    echo "in_progress"
  elif echo "$status_line" | grep -qi "COMPLETE"; then
    echo "complete"
  elif echo "$status_line" | grep -qi "BLOCKED\|FAIL"; then
    echo "blocked"
  else
    echo "in_progress"
  fi
}

# ---------- main: generate progress for one blueprint ----------

generate_progress() {
  local bp_dir="$1"

  # Normalize path
  bp_dir="${bp_dir%/}"
  if [[ ! "$bp_dir" = /* ]]; then
    bp_dir="$REPO_ROOT/$bp_dir"
  fi

  if [[ ! -d "$bp_dir" ]]; then
    echo "Error: blueprint directory not found: $bp_dir" >&2
    return 1
  fi

  local name domain spec_path spec_file spec_status
  name="$(blueprint_name "$bp_dir")"
  domain="$(detect_domain "$bp_dir")"
  spec_path="$(find_spec "$bp_dir" "$domain")"
  spec_file=""
  [[ -n "$spec_path" ]] && spec_file="$REPO_ROOT/$spec_path"
  spec_status="$(detect_spec_status "$spec_file")"

  mkdir -p "$bp_dir/results"
  local output="$bp_dir/results/progress.md"

  # Collect artifacts
  local art_lessons art_audits art_logs art_compounds art_benchmark
  while IFS= read -r line; do
    case "$line" in
      lessons:*)    art_lessons="${line#lessons:}" ;;
      audits:*)     art_audits="${line#audits:}" ;;
      logs:*)       art_logs="${line#logs:}" ;;
      compounds:*)  art_compounds="${line#compounds:}" ;;
      benchmark:*)  art_benchmark="${line#benchmark:}" ;;
    esac
  done <<< "$(collect_artifacts "$bp_dir")"

  # Format YAML arrays
  fmt_yaml_arr() {
    local items="$1"
    if [[ -z "$items" ]]; then
      echo "[]"
    else
      local result="["
      local first=true
      for item in $items; do
        $first || result+=", "
        result+="\"$item\""
        first=false
      done
      result+="]"
      echo "$result"
    fi
  }

  # Build stages
  local stages_yaml=""
  local last_completed_stage=""
  local any_in_progress=false
  local all_complete=true

  while IFS='|' read -r sid sname; do
    [[ -z "$sid" ]] && continue
    local sstatus
    sstatus="$(detect_stage_status_from_logs "$bp_dir" "$sid")"

    # Heuristic overrides for stages detectable from artifacts
    case "$sid" in
      stage-7)
        [[ -n "$art_audits" ]] && sstatus="complete"
        ;;
      stage-8)
        [[ -n "$art_compounds" ]] && sstatus="complete"
        ;;
    esac

    stages_yaml+="  - id: \"$sid\""$'\n'
    stages_yaml+="    name: \"$sname\""$'\n'
    stages_yaml+="    status: \"$sstatus\""$'\n'

    if [[ "$sstatus" == "complete" || "$sstatus" == "skipped" ]]; then
      last_completed_stage="$sid"
    fi
    if [[ "$sstatus" == "in_progress" ]]; then
      any_in_progress=true
      all_complete=false
    fi
    if [[ "$sstatus" != "complete" && "$sstatus" != "skipped" ]]; then
      all_complete=false
    fi
  done <<< "$(deployer_stages "$domain")"

  # Build phases from spec
  local phases_yaml=""
  if [[ -n "$spec_file" && -f "$spec_file" ]]; then
    while IFS='|' read -r pid pname; do
      [[ -z "$pid" ]] && continue
      local pstatus
      pstatus="$(detect_phase_status "$bp_dir" "$pid")"
      phases_yaml+="  - id: \"$pid\""$'\n'
      phases_yaml+="    name: \"$pname\""$'\n'
      phases_yaml+="    status: \"$pstatus\""$'\n'

      if [[ "$pstatus" == "complete" ]]; then
        last_completed_stage="$pid"
      fi
      if [[ "$pstatus" != "complete" && "$pstatus" != "skipped" && "$pstatus" != "not_started" ]]; then
        any_in_progress=true
      fi
    done <<< "$(extract_spec_phases "$spec_file")"
  fi

  # Determine overall status
  local overall_status
  if [[ "$spec_status" == "complete" ]]; then
    overall_status="complete"
  elif $all_complete; then
    overall_status="complete"
  elif $any_in_progress || [[ -n "$last_completed_stage" ]]; then
    overall_status="in_progress"
  else
    overall_status="not_started"
  fi

  # Write the file
  cat > "$output" <<FRONTMATTER
---
blueprint: "$name"
domain: "$domain"
spec: "$spec_path"
status: "$overall_status"
last_updated: "$(timestamp)"
last_stage: "$last_completed_stage"

stages:
${stages_yaml}
phases:
${phases_yaml}
artifacts:
  lessons: $art_lessons
  readiness_audit: $(fmt_yaml_arr "$art_audits")
  deployment_log: $(fmt_yaml_arr "$art_logs")
  compound: $(fmt_yaml_arr "$art_compounds")
  benchmark_report: $art_benchmark
---

# Progress: $name

FRONTMATTER

  # Markdown body: human-readable stage table
  echo "## Deployer Stages" >> "$output"
  echo "" >> "$output"
  echo "| Stage | Name | Status |" >> "$output"
  echo "|-------|------|--------|" >> "$output"

  while IFS='|' read -r sid sname; do
    [[ -z "$sid" ]] && continue
    local sstatus
    sstatus="$(detect_stage_status_from_logs "$bp_dir" "$sid")"
    case "$sid" in
      stage-7) [[ -n "$art_audits" ]] && sstatus="complete" ;;
      stage-8) [[ -n "$art_compounds" ]] && sstatus="complete" ;;
    esac

    local icon
    case "$sstatus" in
      complete)    icon="DONE" ;;
      skipped)     icon="SKIP" ;;
      in_progress) icon="WIP" ;;
      blocked)     icon="BLOCKED" ;;
      *)           icon="--" ;;
    esac
    echo "| $sid | $sname | $icon |" >> "$output"
  done <<< "$(deployer_stages "$domain")"

  # Spec phases table (if any)
  if [[ -n "$phases_yaml" ]]; then
    echo "" >> "$output"
    echo "## Spec Phases" >> "$output"
    echo "" >> "$output"
    echo "| Phase | Name | Status |" >> "$output"
    echo "|-------|------|--------|" >> "$output"

    while IFS='|' read -r pid pname; do
      [[ -z "$pid" ]] && continue
      local pstatus
      pstatus="$(detect_phase_status "$bp_dir" "$pid")"
      local icon
      case "$pstatus" in
        complete)    icon="DONE" ;;
        in_progress) icon="WIP" ;;
        blocked)     icon="BLOCKED" ;;
        *)           icon="--" ;;
      esac
      echo "| $pid | $pname | $icon |" >> "$output"
    done <<< "$(extract_spec_phases "$spec_file")"
  fi

  # Artifacts section
  echo "" >> "$output"
  echo "## Artifacts" >> "$output"
  echo "" >> "$output"
  echo "| Artifact | Present |" >> "$output"
  echo "|----------|---------|" >> "$output"
  echo "| lessons.md | $art_lessons |" >> "$output"
  echo "| readiness audits | ${art_audits:-(none)} |" >> "$output"
  echo "| deployment logs | ${art_logs:-(none)} |" >> "$output"
  echo "| compound summaries | ${art_compounds:-(none)} |" >> "$output"
  echo "| benchmark report | $art_benchmark |" >> "$output"

  echo "  Generated: $output"
}

# ---------- dispatch ----------

if [[ "${1:-}" == "all" ]]; then
  echo "=== Generating progress for all blueprints ==="
  for domain_dir in "$REPO_ROOT"/domains/*/blueprints/*/; do
    [[ ! -d "$domain_dir" ]] && continue
    generate_progress "$domain_dir"
  done
elif [[ -n "${1:-}" ]]; then
  generate_progress "$1"
else
  echo "Usage: scripts/progress.sh <blueprint-path>" >&2
  echo "       scripts/progress.sh all" >&2
  exit 1
fi
