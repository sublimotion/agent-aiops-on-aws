#!/usr/bin/env python3
"""Artifact lineage + consistency-drift audit over Claude Code JSONL traces.

Reconstructs, per session, the graph of file interactions the MAIN agent
performed — Read/Grep/Glob (disclosure), Write/Edit (creation/mutation) — with
timestamps and ordering, then runs a drift query on top:

  DRIFT = an artifact A was depended on (read, or written-then-informed later
  work), a related artifact B changed AFTER A was last touched, and A was never
  re-read or re-edited before the session ended.

This targets the "forgot to go back and update X for consistency" failure.

Scope (deliberate, see spec trace-effectiveness.md): MAIN-agent artifact lineage
only. Subagent-internal file ops are not in JSONL on current Claude Code
versions, so a subagent that edited a file won't appear here. Reasoning-text
lineage (linking prose claims to files) is a future layer, not built here.

Usage:
  ./lineage.py --session PATH.jsonl            # one session → markdown graph + drift
  ./lineage.py --project SUBSTR                # newest session in matching project
  ./lineage.py --project SUBSTR --last N       # audit N newest sessions
  ./lineage.py --session PATH --json OUT.json  # dump graph+drift as JSON (for visual-explainer)
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit"}
SEARCH_TOOLS = {"Grep", "Glob"}


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def norm_path(inp, tool):
    """Extract the file path a file-tool acted on."""
    if not isinstance(inp, dict):
        return None
    for k in ("file_path", "notebook_path", "path"):
        if inp.get(k):
            return inp[k]
    return None


def load_session(fp):
    """Return ordered list of file/search events for the MAIN agent.

    Each event: {seq, ts, action(read|write|edit|search), path, dir, turn_uuid}
    Sidechain (subagent) lines are skipped — they carry no main-agent lineage
    and JSONL doesn't include their internal file ops anyway.
    """
    events = []
    seq = 0
    for line in open(fp, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("isSidechain"):
            continue
        if d.get("type") != "assistant":
            continue
        m = d.get("message", {})
        if not isinstance(m, dict):
            continue
        ts = parse_ts(d.get("timestamp"))
        for b in (m.get("content") or []):
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            name = b.get("name", "")
            inp = b.get("input") or {}
            if name in FILE_TOOLS:
                path = norm_path(inp, name)
                if not path:
                    continue
                action = {"Read": "read", "Write": "write",
                          "Edit": "edit", "NotebookEdit": "edit"}[name]
                seq += 1
                events.append({"seq": seq, "ts": ts, "action": action,
                               "path": path, "uuid": d.get("uuid")})
            elif name in SEARCH_TOOLS:
                seq += 1
                events.append({"seq": seq, "ts": ts, "action": "search",
                               "path": inp.get("pattern") or inp.get("glob") or "?",
                               "uuid": d.get("uuid")})
    return events


def build_lineage(events):
    """Per-file timeline of actions, plus a co-directory dependency notion."""
    per_file = defaultdict(list)  # path -> [(seq, action, ts)]
    for e in events:
        if e["action"] == "search":
            continue
        per_file[e["path"]].append((e["seq"], e["action"], e["ts"]))
    return per_file


_MAX_COUPLE_BYTES = 512 * 1024  # don't scan huge/binary files for references


def _read_text(path):
    try:
        if os.path.getsize(path) > _MAX_COUPLE_BYTES:
            return None
        with open(path, errors="ignore") as f:
            return f.read()
    except OSError:
        return None


# Basenames too common to identify a file on their own — a bare mention doesn't
# mean *this* one. Require a disambiguating parent-dir/basename suffix instead.
_GENERIC_BASENAMES = {
    "README.md", "main.tf", "variables.tf", "outputs.tf", "__init__.py",
    "index.html", "index.ts", "index.js", "setup.py", "Dockerfile",
    "requirements.txt", "package.json", "config.yaml", "config.yml",
    "CLAUDE.md", "AGENTS.md", "lessons.md", "spec.md", "_template.md",
}


def _references(container_text, target_path):
    """Does container_text reference target_path? For distinctive basenames a
    bare mention counts; for ubiquitous ones (README.md, main.tf …) require a
    parent-dir/basename suffix so we don't couple every README to every other."""
    if not container_text:
        return False
    base = os.path.basename(target_path)
    if not base:
        return False
    if base in _GENERIC_BASENAMES:
        parent = os.path.basename(os.path.dirname(target_path))
        suffix = f"{parent}/{base}"
        return bool(parent) and suffix in container_text
    if base in container_text:
        return True
    stem, ext = os.path.splitext(base)
    # for importable code, allow bare-stem references (import stem / from stem)
    if ext in (".py", ".ts", ".js", ".tf", ".go") and len(stem) >= 4:
        for pat in (f"import {stem}", f"from {stem}", f'"{stem}"', f"'{stem}'"):
            if pat in container_text:
                return True
    return False


def build_coupling(per_file):
    """Return edges {B: set(A)} meaning 'B references A' — B should stay
    consistent with A. Content-based: reads each touched file's CURRENT content
    and looks for references to other touched files. Beats the old same-dir
    heuristic (no false edge for unrelated new neighbors; direction-aware)."""
    paths = list(per_file)
    text = {p: _read_text(p) for p in paths}
    edges = defaultdict(set)
    for b in paths:
        tb = text[b]
        if not tb:
            continue
        for a in paths:
            if a != b and _references(tb, a):
                edges[b].add(a)
    return edges


def detect_drift(per_file):
    """Flag a file B as possibly stale when: B references A (content coupling),
    A was MUTATED (write/edit) after B was last touched (any action), and B was
    never revisited afterward. Targets "edited A, forgot to update B that
    documents/depends on A."
    """
    last_any = {}     # path -> max seq of ANY action
    last_mut = {}     # path -> (seq, action) of last write/edit
    for path, hist in per_file.items():
        last_any[path] = max(s for s, _, _ in hist)
        muts = [(s, a) for s, a, _ in hist if a in ("write", "edit")]
        if muts:
            last_mut[path] = max(muts)

    edges = build_coupling(per_file)  # B -> set(A) : B references A

    flags = []
    for b, targets in edges.items():
        b_last = last_any[b]
        stale_against = []
        for a in targets:
            if a not in last_mut:
                continue  # A never mutated this session → nothing to drift from
            a_mut_seq = last_mut[a][0]
            if a_mut_seq > b_last:   # A changed after B was last touched
                stale_against.append((a_mut_seq, a))
        if stale_against:
            latest = max(s for s, _ in stale_against)
            b_action = last_mut[b][1] if b in last_mut else "read"
            flags.append({
                "path": b,
                "last_action": b_action,
                "last_seq": b_last,
                "references_changed": sorted({a for _, a in stale_against}),
                "latest_change_seq": latest,
            })
    return sorted(flags, key=lambda f: f["last_seq"])


def render(fp, events, per_file, flags, as_json=False):
    if as_json:
        return json.dumps({
            "session": fp,
            "event_count": len(events),
            "files": {p: [{"seq": s, "action": a} for s, a, _ in h]
                      for p, h in per_file.items()},
            "drift_flags": flags,
        }, indent=2)

    out = []
    p = out.append
    sid = os.path.basename(fp)
    p(f"# Lineage & consistency audit — `{sid}`\n")
    n_read = sum(1 for e in events if e["action"] == "read")
    n_write = sum(1 for e in events if e["action"] == "write")
    n_edit = sum(1 for e in events if e["action"] == "edit")
    n_search = sum(1 for e in events if e["action"] == "search")
    p(f"Main-agent file events: **{len(events)-n_search}** "
      f"({n_read} read, {n_write} write, {n_edit} edit) + {n_search} searches. "
      f"{len(per_file)} distinct files touched.\n")

    # ---- Drift flags: the payoff ----
    p("## Consistency-drift flags\n")
    if not flags:
        p("> None. No file was left un-revisited after a sibling in its "
          "directory changed later in the session.\n")
    else:
        p(f"**{len(flags)}** file(s) may be stale — they reference a file that "
          "changed later, and were never revisited before session end:\n")
        for f in flags:
            refs = ", ".join(f"`{os.path.basename(s)}`" for s in f["references_changed"])
            p(f"- **`{f['path']}`** — last {f['last_action']} at step {f['last_seq']}; "
              f"it references {refs} which changed afterward (step {f['latest_change_seq']}); "
              f"never re-read/re-edited. *Check it's still consistent.*")
        p("")

    # ---- Per-file timeline (the graph, as a readable table) ----
    p("## Per-file timeline\n")
    p("Ordered actions per file (r=read, w=write, e=edit). A `w`/`e` after "
      "another file's later change is the drift signal above.\n")
    # sort files by first-touch order
    ordered = sorted(per_file.items(), key=lambda kv: min(s for s, _, _ in kv[1]))
    for path, hist in ordered:
        seq_str = " → ".join(f"{a[0]}{s}" for s, a, _ in
                             sorted(hist, key=lambda x: x[0]))
        flagged = " ⚠️" if any(fl["path"] == path for fl in flags) else ""
        p(f"- `{path}`{flagged}: {seq_str}")
    p("")
    return "\n".join(out)


def pick_sessions(root, project, session, last):
    if session:
        return [session]
    files = glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True)
    if project:
        files = [f for f in files if project in f]
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[:last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects/"))
    ap.add_argument("--session", default=None, help="explicit .jsonl path")
    ap.add_argument("--project", default=None, help="pick newest session(s) whose path matches")
    ap.add_argument("--last", type=int, default=1, help="how many newest sessions to audit (with --project)")
    ap.add_argument("--json", default=None, help="dump graph+drift JSON (single session only)")
    args = ap.parse_args()

    sessions = pick_sessions(args.root, args.project, args.session, args.last)
    if not sessions:
        print("No matching sessions found.")
        return

    for i, fp in enumerate(sessions):
        events = load_session(fp)
        per_file = build_lineage(events)
        flags = detect_drift(per_file)
        if args.json and len(sessions) == 1:
            with open(args.json, "w") as f:
                f.write(render(fp, events, per_file, flags, as_json=True))
            print(f"Wrote {args.json}")
        else:
            if i:
                print("\n---\n")
            print(render(fp, events, per_file, flags))


if __name__ == "__main__":
    main()
