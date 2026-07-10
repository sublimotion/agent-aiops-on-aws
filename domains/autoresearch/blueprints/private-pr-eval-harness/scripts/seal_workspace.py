#!/usr/bin/env python3
"""Phase 2 — Gate 1: prepare a git-SEALED workspace at a task's base_commit.

Invariant (spec Gate 1): from inside the workspace, `git log`, `git show`,
`git diff <base>..HEAD`, and the packed refs of the fixing commit MUST be
unreachable. The agent must not be able to read the solution diff from history.

Mechanism (one of several allowed): clone at base_commit, then RE-INIT git —
delete `.git` entirely and create a fresh repo with a SINGLE commit containing
the base tree. Result: history has exactly one commit (the base snapshot), no
future commits, no remotes, no refs to the PR/fix. The agent still has a working
`git` (can diff its own edits) but cannot time-travel to the answer.

Usage:
  python3 seal_workspace.py --repo pydantic/pydantic --base <sha> --dir /tmp/ws
  python3 seal_workspace.py --probe --dir /tmp/ws   # Gate-1 acceptance probe
"""
import argparse, subprocess, sys, pathlib, os


def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"cmd failed: {' '.join(cmd)}\n{r.stderr[:600]}")
    return r


def seal(repo, base, wsdir):
    ws = pathlib.Path(wsdir)
    if ws.exists():
        run(["rm", "-rf", str(ws)])
    ws.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    # full clone (need the base_commit; it may be old) then hard-reset to base
    run(["git", "clone", "--quiet", url, str(ws)])
    run(["git", "checkout", "--quiet", base], cwd=str(ws))

    # --- SEAL: destroy history, re-init with a single base-snapshot commit ---
    run(["rm", "-rf", str(ws / ".git")])
    run(["git", "init", "--quiet"], cwd=str(ws))
    run(["git", "config", "user.email", "eval@harness.local"], cwd=str(ws))
    run(["git", "config", "user.name", "eval-harness"], cwd=str(ws))
    run(["git", "add", "-A"], cwd=str(ws))
    run(["git", "commit", "--quiet", "-m", "base snapshot (sealed)"], cwd=str(ws))
    print(f"[seal] {repo}@{base[:8]} -> {ws} (history reduced to 1 commit)")


def probe(wsdir):
    """Gate-1 acceptance: history must be exactly one commit; base sha absent."""
    ws = str(wsdir)
    log = run(["git", "log", "--oneline"], cwd=ws, check=False).stdout.strip()
    n = len([l for l in log.splitlines() if l.strip()])
    remotes = run(["git", "remote"], cwd=ws, check=False).stdout.strip()
    # try to reach any commit other than HEAD
    allrefs = run(["git", "rev-list", "--all", "--count"], cwd=ws, check=False).stdout.strip()
    ok = (n == 1 and remotes == "" and allrefs == "1")
    print(f"[probe] commits={n} remotes={remotes or 'none'} rev-list-all={allrefs} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(2)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--base")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        probe(args.dir)
    else:
        if not (args.repo and args.base):
            sys.exit("seal mode needs --repo and --base")
        seal(args.repo, args.base, args.dir)
        probe(args.dir)  # always self-verify after sealing


if __name__ == "__main__":
    main()
