#!/usr/bin/env python3
"""Skill & subagent effectiveness report over Claude Code JSONL traces.

Answers "how is our agentic framework actually working": which skills/subagents
get invoked, which are dead weight, where tokens go, and where the agent thrashes.

Works on local JSONL only (~/.claude/projects). Subagent *internal* trajectories
are not written to JSONL by current Claude Code versions, so subagent analysis is
invocation + surrounding-cost level; deep trajectory analysis needs the OTel path.

Usage:
  ./analyze.py                       # scan ~/.claude/projects, print markdown
  ./analyze.py --root PATH           # scan a different trace root
  ./analyze.py --project SUBSTR      # only sessions whose project dir matches SUBSTR
  ./analyze.py --repo-root PATH      # where to look for .claude/skills + agents (default: cwd)
  ./analyze.py --json OUT.json       # also dump raw stats as JSON
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Inventory discovery — what skills/agents COULD be invoked, to find dead ones.
# ---------------------------------------------------------------------------

def discover_inventory(repo_root, include_other_project_plugins=False):
    """Return (skills, agents) sets of names actually IN SCOPE for repo_root.

    Scope matters for honest dead-skill detection: a skill installed as a plugin
    scoped to a *different* project doesn't load in this repo, so counting it as
    "dead here" is a false positive (this inflated the original 43-dead-skills
    number — most were `document-skills` scoped to another project). By default
    we include only:
      - project skills/agents under repo_root/.claude
      - user-global skills/agents under ~/.claude
      - plugins whose install scope is `user`, or local/project scoped to repo_root
    Pass include_other_project_plugins=True to restore global-cache behavior.
    """
    skills, agents = set(), set()
    home = os.path.expanduser("~")

    # Project + user skills (directory name == skill name) — always in scope
    for base in (os.path.join(repo_root, ".claude", "skills"),
                 os.path.join(home, ".claude", "skills")):
        if os.path.isdir(base):
            for n in os.listdir(base):
                if not n.startswith(".") and n != "tests":
                    skills.add(n)

    # Plugin skills — filter by install scope so out-of-repo project plugins
    # aren't counted as dead here.
    in_scope = _in_scope_plugin_paths(home, repo_root, include_other_project_plugins)
    for sm in glob.glob(os.path.join(home, ".claude", "plugins", "**", "skills", "*", "SKILL.md"),
                        recursive=True):
        if in_scope is not None and not any(sm.startswith(p) for p in in_scope):
            continue
        parts = sm.split(os.sep)
        try:
            plugin = parts[parts.index("plugins") + 2]
            name = parts[-2]
            skills.add(f"{plugin}:{name}")
            skills.add(name)  # also match bare name, some invocations drop the namespace
        except (ValueError, IndexError):
            pass

    # Agents (project + user); file <name>.md
    for base in (os.path.join(repo_root, ".claude", "agents"),
                 os.path.join(home, ".claude", "agents")):
        if os.path.isdir(base):
            for n in os.listdir(base):
                if n.endswith(".md"):
                    agents.add(n[:-3])

    return skills, agents


def _in_scope_plugin_paths(home, repo_root, include_other_project_plugins):
    """Return set of plugin installPath prefixes in scope for repo_root, or None
    to disable filtering (include everything). Reads installed_plugins.json."""
    if include_other_project_plugins:
        return None
    manifest = os.path.join(home, ".claude", "plugins", "installed_plugins.json")
    try:
        with open(manifest) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None  # can't read manifest → don't over-filter, fall back to global
    repo_root = os.path.realpath(repo_root)
    paths = set()
    for _plugin, installs in (data.get("plugins") or {}).items():
        for inst in installs:
            ip = inst.get("installPath")
            if not ip:
                continue
            if inst.get("scope") == "user":
                paths.add(ip)
            else:  # local/project → in scope only if projectPath == repo_root
                pp = inst.get("projectPath")
                if pp and os.path.realpath(pp) == repo_root:
                    paths.add(ip)
    return paths


# ---------------------------------------------------------------------------
# Correction detection (topic clustering) — reused from friction.py, kept in sync.
# ---------------------------------------------------------------------------

CORRECTION_PATTERNS = [
    r"\bno,? (don't|do not|not|stop|that)", r"\bdon'?t\b", r"\bstop\b", r"\bundo\b",
    r"\brevert\b", r"\bthat'?s (wrong|not)", r"\bactually,?\b", r"\binstead\b",
    r"\byou (missed|forgot|didn'?t)", r"\bdoesn'?t work\b", r"\bstill (broken|failing|wrong|not)",
    r"\bwhy (did|are) you\b", r"\bi (said|asked|told)\b", r"\bnot what i\b",
    r"\bread the\b", r"\bbefore (you|doing)\b", r"\bwait,?\b",
]
CORR_RE = re.compile("|".join(CORRECTION_PATTERNS), re.I)

TOPIC_TAGS = {
    "read-before-acting": [r"\bread\b", r"\bcheck\b", r"\blook at\b", r"\bcontext\b", r"before"],
    "scope-overreach": [r"\bonly\b", r"\bjust\b", r"too (much|many)", r"\bdon'?t (add|create|change)", r"\bscope\b", r"\bminimal\b"],
    "wrong-approach": [r"\binstead\b", r"\bwrong\b", r"\bnot (how|what|the)\b", r"\bdifferent\b", r"\bbetter way\b"],
    "verify-first": [r"\bverify\b", r"\bconfirm\b", r"\bmake sure\b", r"\btest\b", r"\bcheck (it|that|the)\b"],
    "stop-thrashing": [r"\bstop\b", r"\bwait\b", r"\bundo\b", r"\brevert\b", r"same (mistake|thing)"],
    "commit-git": [r"\bcommit\b", r"\bpush\b", r"\bgit\b", r"\bstage\b"],
    "dont-assume": [r"\bassum", r"\bguess", r"\bdon'?t (make up|invent)", r"\bhallucinat"],
    "formatting-style": [r"\bformat\b", r"\bstyle\b", r"\btone\b", r"\bshorter\b", r"\bconcise\b", r"\bverbose\b"],
}


def user_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def is_synthetic(txt):
    s = txt.strip()
    return (not s) or s.startswith("<") or s.startswith("Caveat:") or len(s) < 4


def classify_correction(txt):
    tl = txt.lower()
    for tag, pats in TOPIC_TAGS.items():
        if any(re.search(p, tl) for p in pats):
            return tag
    return "other"


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan(root, project_filter):
    files = glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True)
    if project_filter:
        files = [f for f in files if project_filter in f]

    stats = {
        "sessions": 0,
        "main_lines": 0,
        "side_lines": 0,
        "tok_in": 0, "tok_out": 0, "cache_read": 0, "cache_create": 0,
        "tool_calls": Counter(),
        "tool_errors": Counter(),
        "tool_err_samples": defaultdict(list),
        "skill_calls": Counter(),
        "subagent_calls": Counter(),
        "subagent_out_tok": Counter(),   # output tokens on sidechain lines, attributed to last-spawned agent (best-effort)
        "corrections": [],               # (topic, snippet)
        "correction_by_project": Counter(),
        "retry_loops": 0,
        "sessions_with_skill": 0,
        "per_session_tokens": [],        # (project, out_tokens) for thrash ranking
    }

    for fp in files:
        stats["sessions"] += 1
        proj = os.path.basename(os.path.dirname(fp))
        id2tool = {}
        prev_error_tool = None
        session_out = 0
        session_used_skill = False

        for line in open(fp, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            m = d.get("message", {})
            side = bool(d.get("isSidechain"))
            if side:
                stats["side_lines"] += 1
            else:
                stats["main_lines"] += 1
            if not isinstance(m, dict):
                continue

            if t == "assistant":
                u = m.get("usage") or {}
                oi = u.get("output_tokens", 0) or 0
                stats["tok_in"] += u.get("input_tokens", 0) or 0
                stats["tok_out"] += oi
                stats["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
                stats["cache_create"] += u.get("cache_creation_input_tokens", 0) or 0
                session_out += oi
                for b in (m.get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        name = b.get("name", "?")
                        inp = b.get("input") or {}
                        stats["tool_calls"][name] += 1
                        id2tool[b.get("id")] = name
                        if name == "Skill":
                            stats["skill_calls"][inp.get("skill", "?")] += 1
                            session_used_skill = True
                        if name in ("Task", "Agent"):
                            stats["subagent_calls"][inp.get("subagent_type", "?")] += 1

            if t == "user":
                c = m.get("content")
                if isinstance(c, list):
                    for b in c:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "tool_result" and b.get("is_error"):
                            tn = id2tool.get(b.get("tool_use_id"), "?")
                            stats["tool_errors"][tn] += 1
                            content = b.get("content")
                            if isinstance(content, str):
                                etxt = content
                            elif isinstance(content, list):
                                etxt = " ".join(x.get("text", "") for x in content if isinstance(x, dict))
                            else:
                                etxt = ""
                            stats["tool_err_samples"][tn].append(etxt.strip()[:160])
                            if prev_error_tool == tn:
                                stats["retry_loops"] += 1
                            prev_error_tool = tn
                        elif b.get("type") == "tool_result":
                            prev_error_tool = None
                txt = user_text(m)
                if not is_synthetic(txt) and CORR_RE.search(txt[:400]):
                    topic = classify_correction(txt)
                    stats["corrections"].append((topic, re.sub(r"\s+", " ", txt.strip())[:140]))
                    stats["correction_by_project"][proj] += 1

        if session_used_skill:
            stats["sessions_with_skill"] += 1
        stats["per_session_tokens"].append((proj, session_out))

    return stats


def top_error_patterns(errs, n=3):
    pats = Counter()
    for e in errs:
        key = re.sub(r"[0-9]+", "N", e)[:70]
        pats[key] += 1
    return pats.most_common(n)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(stats, skills_inv, agents_inv):
    out = []
    p = out.append

    p("# Skill & Subagent Effectiveness — Claude Code Traces\n")
    total_lines = stats["main_lines"] + stats["side_lines"]
    side_pct = 100 * stats["side_lines"] / total_lines if total_lines else 0
    p(f"Sessions scanned: **{stats['sessions']}**  |  trace lines: {total_lines:,} "
      f"({side_pct:.0f}% subagent sidechain)\n")

    # ---- Token economy ----
    cr, cc, ti, to = stats["cache_read"], stats["cache_create"], stats["tok_in"], stats["tok_out"]
    hit = 100 * cr / (cr + cc + ti) if (cr + cc + ti) else 0
    p("## Token economy\n")
    p(f"- output: **{to:,}**  |  fresh input: {ti:,}  |  cache read: {cr:,}  |  cache create: {cc:,}")
    p(f"- cache-hit ratio: **{hit:.1f}%** (higher = cheaper; low ratio means prompts aren't reused)")
    p("> Caveat: usage fields in JSONL are subject to the Claude Code token-accounting "
      "bug (#25941) that friction.py deliberately avoids. Treat these as order-of-magnitude "
      "trends, not exact accounting. The skill/subagent/error/correction signals below are unaffected.\n")

    # ---- Skill effectiveness + dead skills ----
    p("## 1. Skill invocations (adoption + dead-skill map)\n")
    used_skills = stats["skill_calls"]
    total_skill = sum(used_skills.values())
    sess_skill = stats["sessions_with_skill"]
    p(f"Total skill invocations: **{total_skill}** across {stats['sessions']} sessions "
      f"(only **{sess_skill}** sessions used any skill = "
      f"{100*sess_skill/stats['sessions'] if stats['sessions'] else 0:.0f}%).\n")
    if used_skills:
        p("| Skill | Invocations |")
        p("|---|---|")
        for n, c in used_skills.most_common():
            p(f"| {n} | {c} |")
    # dead skills: in inventory, never invoked (normalize namespace)
    invoked_norm = set()
    for n in used_skills:
        invoked_norm.add(n)
        if ":" in n:
            invoked_norm.add(n.split(":", 1)[1])
    dead = sorted(s for s in skills_inv if s not in invoked_norm and ":" not in s) if skills_inv else []
    # only report bare-name dead skills to keep the list readable
    p(f"\n**Dead / never-invoked skills** ({len(dead)} of {len([s for s in skills_inv if ':' not in s])} bare-name skills in inventory):\n")
    if dead:
        p("> " + ", ".join(f"`{s}`" for s in dead))
        p("\nEach is a skill you authored/installed but the agent never chose. Either the "
          "trigger description isn't matching, or the capability is redundant with a subagent/tool.\n")
    else:
        p("> (none — every catalogued skill has been used)\n")

    # ---- Subagent effectiveness ----
    p("## 2. Subagent invocations\n")
    subs = stats["subagent_calls"]
    p(f"Total subagent spawns: **{sum(subs.values())}** "
      f"(vs {total_skill} skill invocations — where your real delegation happens).\n")
    if subs:
        p("| Subagent | Spawns | In inventory? |")
        p("|---|---|---|")
        for n, c in subs.most_common():
            known = "yes" if n in agents_inv else ("built-in" if n in ("general-purpose", "Explore", "Plan", "?") else "**unknown**")
            p(f"| {n} | {c} | {known} |")
    unused_agents = sorted(a for a in agents_inv if a not in subs)
    if unused_agents:
        p(f"\n**Never-spawned agents** ({len(unused_agents)}): "
          + ", ".join(f"`{a}`" for a in unused_agents))
    p("\n> Note: subagent *internal* trajectories are not in JSONL on current Claude Code "
      "versions (sidechain lines ~0 in recent sessions). Per-subagent token cost and "
      "tool-use quality need the OTel trace path. This section is invocation-level only.\n")

    # ---- Tool error drill-down ----
    p("## 3. Tool error rates\n")
    p("| Tool | Errors | Total calls | Error rate |")
    p("|---|---|---|---|")
    for tn, ec in stats["tool_errors"].most_common():
        tot = stats["tool_calls"].get(tn, 0)
        rate = f"{100*ec/tot:.1f}%" if tot else "n/a"
        p(f"| {tn} | {ec} | {tot} | {rate} |")
    p("\n### Top recurring error patterns (normalized)\n")
    for tn, _ in stats["tool_errors"].most_common(4):
        p(f"**{tn}:**")
        for pat, cnt in top_error_patterns(stats["tool_err_samples"][tn]):
            p(f"  - ({cnt}x) `{pat}`")
        p("")

    # ---- Corrections (skill-gap map) ----
    p("## 4. User corrections by topic (skill-gap map)\n")
    by_topic = Counter(c[0] for c in stats["corrections"])
    p(f"Total correction turns: **{len(stats['corrections'])}** "
      f"(user turns that steer/negate the agent — a soft negative-reward signal).\n")
    MEANING = {
        "read-before-acting": "Acted without reading context first",
        "scope-overreach": "Did more than asked",
        "wrong-approach": "Chose the wrong method",
        "verify-first": "Didn't verify before claiming done",
        "stop-thrashing": "Looping/repeating a mistake",
        "commit-git": "Git/commit workflow correction",
        "dont-assume": "Assumed/guessed/hallucinated",
        "formatting-style": "Output format/tone/length",
        "other": "Uncategorized steer",
    }
    p("| Topic | Count | Meaning |")
    p("|---|---|---|")
    for topic, cnt in by_topic.most_common():
        p(f"| {topic} | {cnt} | {MEANING.get(topic,'')} |")
    p("\n### Sample corrections per top topic\n")
    for topic, _ in by_topic.most_common(5):
        if topic == "other":
            continue
        samples = [c[1] for c in stats["corrections"] if c[0] == topic][:3]
        p(f"**{topic}:**")
        for s in samples:
            p(f"  - \"{s}\"")
        p("")

    # ---- Thrash signals ----
    p("## 5. Thrash signals\n")
    p(f"- Consecutive same-tool errors (retry loops): **{stats['retry_loops']}**")
    top_sessions = sorted(stats["per_session_tokens"], key=lambda x: -x[1])[:5]
    p("- Highest output-token sessions (candidates for context/thrash review):")
    for proj, tok in top_sessions:
        p(f"  - {tok:,} out-tok — `{proj}`")
    p("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects/"))
    ap.add_argument("--project", default=None, help="only sessions whose path contains this substring")
    ap.add_argument("--repo-root", default=os.getcwd(), help="repo to discover .claude/skills + agents from")
    ap.add_argument("--all-plugins", action="store_true",
                    help="count plugin skills from ALL projects as in-scope (default: only user-global + this repo's project plugins)")
    ap.add_argument("--json", default=None, help="also dump raw stats to this JSON path")
    args = ap.parse_args()

    skills_inv, agents_inv = discover_inventory(args.repo_root,
                                                include_other_project_plugins=args.all_plugins)
    stats = scan(args.root, args.project)
    print(report(stats, skills_inv, agents_inv))

    if args.json:
        dumpable = {k: (dict(v) if isinstance(v, (Counter, defaultdict)) else v)
                    for k, v in stats.items()
                    if k not in ("tool_err_samples", "corrections", "per_session_tokens")}
        dumpable["inventory_skills"] = sorted(skills_inv)
        dumpable["inventory_agents"] = sorted(agents_inv)
        with open(args.json, "w") as f:
            json.dump(dumpable, f, indent=2)


if __name__ == "__main__":
    main()
