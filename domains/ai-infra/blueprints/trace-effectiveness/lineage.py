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
                               "path": path, "uuid": d.get("uuid"),
                               "old_string": inp.get("old_string"),
                               "new_string": inp.get("new_string")})
            elif name in SEARCH_TOOLS:
                seq += 1
                events.append({"seq": seq, "ts": ts, "action": "search",
                               "path": inp.get("pattern") or inp.get("glob") or "?",
                               "uuid": d.get("uuid")})
    return events


def load_session_streamjson(fp):
    """Adapter for agent-runner's `run.log` (Claude Code --output-format stream-json).

    Same tool_use shape as projects-JSONL, but: no `isSidechain`/`timestamp`
    fields, some events nest `type` under `msg`, and the file has non-JSON lines
    (git clone / install-deps output) that must be skipped. This is the
    execution-substrate path (agent-runtime); the projects-JSONL path is local.
    """
    events = []
    seq = 0
    for line in open(fp, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue  # raw non-JSON log line (git/install output) — not a trace event
        if not isinstance(d, dict):
            continue
        t = d.get("type") or (d.get("msg") or {}).get("type")
        if t not in ("assistant", "agent_message", "message"):
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            m = d.get("msg") if isinstance(d.get("msg"), dict) else {}
        ts = parse_ts(d.get("timestamp"))  # usually None in stream-json — seq order is what matters
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
                               "path": path, "uuid": b.get("id"),
                               "old_string": inp.get("old_string"),
                               "new_string": inp.get("new_string")})
            elif name in SEARCH_TOOLS:
                seq += 1
                events.append({"seq": seq, "ts": ts, "action": "search",
                               "path": inp.get("pattern") or inp.get("glob") or "?",
                               "uuid": b.get("id")})
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
                "kind": "reference",
            })
    return sorted(flags, key=lambda f: f["last_seq"])


# Only treat distinctive value replacements as coupling signals — a 1-char or
# ultra-common change ("a"→"b", "0"→"1") would false-positive everywhere.
_MIN_VALUE_LEN = 2


def _is_wordish(c):
    return c.isalnum() or c in "._-"


def _minimal_change(old, new):
    """Reduce a full-line Edit (old_string/new_string carry surrounding context)
    to the differing token, snapped to WORD boundaries so e.g. `"v1"`→`"v2"`
    yields `v1`/`v2` (not the bare `1`/`2`, which would be uselessly noisy).
    Returns (old_token, new_token) or (None, None)."""
    if not old or not new or old == new:
        return None, None
    # minimal differing span (strip common prefix/suffix)
    i = 0
    while i < len(old) and i < len(new) and old[i] == new[i]:
        i += 1
    j = 0
    while (j < len(old) - i and j < len(new) - i
           and old[-1 - j] == new[-1 - j]):
        j += 1
    # expand left/right over word-ish chars so the token isn't cut mid-word
    lo = i
    while lo > 0 and _is_wordish(old[lo - 1]):
        lo -= 1
    ro = len(old) - j
    while ro < len(old) and _is_wordish(old[ro]):
        ro += 1
    rn = len(new) - j
    while rn < len(new) and _is_wordish(new[rn]):
        rn += 1
    old_tok = old[lo:ro].strip()
    new_tok = new[lo:rn].strip()
    return old_tok or None, new_tok or None


def _iter_repo_files(root, max_files=2000):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       (".git", "node_modules", "__pycache__", ".claude")]
        for fn in filenames:
            yield os.path.join(dirpath, fn)
            max_files -= 1
            if max_files <= 0:
                return


def detect_value_drift(events, repo_root=None):
    """Flag files that still contain a value the agent REPLACED elsewhere.

    The agent's Edit events carry old_string/new_string. When a distinctive
    token was consistently swapped old→new in the files it touched, any OTHER
    file under repo_root that still contains `old` (and not `new`) is likely a
    forgotten coupled site — value-level drift (constant/config propagation),
    complementary to reference-level drift. Mechanical: substring check only.

    repo_root defaults to the common directory of the edited files.
    """
    # Collect (old_token -> new_token) pairs the agent actually applied.
    edited_paths = set()
    pairs = {}  # old -> new (last wins; seeded tasks use one token/task)
    for e in events:
        if e.get("action") != "edit":
            continue
        edited_paths.add(os.path.realpath(e["path"]))
        # old/new_string carry full-line context; reduce to the minimal changed
        # token so we match the VALUE, not surrounding text.
        old_tok, new_tok = _minimal_change(e.get("old_string"), e.get("new_string"))
        if old_tok and new_tok and old_tok != new_tok and len(old_tok) <= 40:
            pairs[old_tok] = new_tok

    pairs = {o: n for o, n in pairs.items()
             if len(o) >= _MIN_VALUE_LEN and o not in ("", " ")}
    if not pairs or not edited_paths:
        return []

    if repo_root is None:
        repo_root = os.path.commonpath(list(edited_paths)) if len(edited_paths) > 1 \
            else os.path.dirname(next(iter(edited_paths)))
    if not os.path.isdir(repo_root):
        return []

    flags = []
    for fpath in _iter_repo_files(repo_root):
        rp = os.path.realpath(fpath)
        if rp in edited_paths:
            continue  # agent already touched it
        content = _read_text(fpath)
        if not content:
            continue
        stale_tokens = [o for o, n in pairs.items()
                        if o in content and n not in content]
        if stale_tokens:
            flags.append({
                "path": fpath,
                "last_action": "untouched",
                "stale_values": sorted(set(stale_tokens)),
                "propagated_elsewhere_to": sorted({pairs[o] for o in stale_tokens}),
                "kind": "value",
            })
    return flags


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
        p(f"**{len(flags)}** file(s) may be stale (never revisited before session end):\n")
        for f in flags:
            if f.get("kind") == "value":
                vals = ", ".join(f"`{v}`" for v in f["stale_values"])
                to = ", ".join(f"`{v}`" for v in f["propagated_elsewhere_to"])
                p(f"- **`{f['path']}`** (untouched) still contains {vals}, which was "
                  f"changed to {to} in files the agent edited. *Likely a forgotten "
                  f"coupled site — propagate or confirm it should differ.*")
            else:
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
    ap.add_argument("--stream-json", action="store_true",
                    help="parse agent-runner run.log (Claude Code stream-json) instead of projects-JSONL. Use with --session <run.log>.")
    ap.add_argument("--json", default=None, help="dump graph+drift JSON (single session only)")
    ap.add_argument("--repo-root", default=None,
                    help="root to scan for value-drift (default: common dir of edited files)")
    ap.add_argument("--no-value-drift", action="store_true",
                    help="disable value-level drift detection (reference-coupling only)")
    ap.add_argument("--no-reference-drift", action="store_true",
                    help="disable reference-coupling drift (value-level only). Use for value-propagation tasks where reference-drift is noisy.")
    args = ap.parse_args()

    sessions = pick_sessions(args.root, args.project, args.session, args.last)
    if not sessions:
        print("No matching sessions found.")
        return

    loader = load_session_streamjson if args.stream_json else load_session
    for i, fp in enumerate(sessions):
        events = loader(fp)
        per_file = build_lineage(events)
        flags = [] if args.no_reference_drift else detect_drift(per_file)
        if not args.no_value_drift:
            flags = flags + detect_value_drift(events, repo_root=args.repo_root)
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
