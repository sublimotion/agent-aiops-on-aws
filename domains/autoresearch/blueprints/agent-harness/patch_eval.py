#!/usr/bin/env python3
"""
Offline patch evaluator for SWE-bench Lite.

Applies patches to clean workspaces and runs FAIL_TO_PASS tests.
Does NOT use Docker — runs tests directly in the repo.

Usage:
    python3 patch_eval.py --harness langgraph --output results/eval_langgraph.jsonl
    python3 patch_eval.py --harness sera --output results/eval_sera.jsonl
    python3 patch_eval.py --phase1 --output results/eval_phase1.jsonl
    python3 patch_eval.py --report results/eval_*.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset

WORKSPACE_ROOT = Path("/mnt/nvme/sera-workspaces")
REPO_CACHE = WORKSPACE_ROOT / "_repo_cache"
PATCHES_DIR = Path("/mnt/nvme/agent-harness/patches")
RESULTS_DIR = Path("/mnt/nvme/agent-harness/results")


def setup_workspace(instance):
    """Clone repo at base_commit."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    workspace = WORKSPACE_ROOT / (instance_id.replace("/", "__") + "_eval")
    if workspace.exists():
        subprocess.run(["rm", "-rf", str(workspace)], check=True)

    cache_path = REPO_CACHE / repo.replace("/", "__")
    if cache_path.exists():
        check = subprocess.run(["git", "cat-file", "-t", base_commit],
                               cwd=cache_path, capture_output=True)
        if check.returncode == 0:
            subprocess.run(["git", "clone", str(cache_path), str(workspace)],
                           capture_output=True, check=True)
        else:
            subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
                           capture_output=True, check=True, timeout=300)
    else:
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
                       capture_output=True, check=True, timeout=300)

    subprocess.run(["git", "checkout", base_commit], cwd=workspace,
                   capture_output=True, check=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=workspace, capture_output=True)
    return workspace


def apply_patch(workspace, patch_file):
    """Apply a git diff patch to workspace."""
    result = subprocess.run(
        ["git", "apply", "--allow-empty", str(patch_file)],
        cwd=workspace, capture_output=True, text=True
    )
    if result.returncode != 0:
        # Try with --reject to apply what we can
        result2 = subprocess.run(
            ["git", "apply", "--reject", "--allow-empty", str(patch_file)],
            cwd=workspace, capture_output=True, text=True
        )
        if result2.returncode != 0:
            return False, result.stderr + result2.stderr
    return True, ""


def run_tests(workspace, fail_to_pass, repo, timeout=180):
    """Run FAIL_TO_PASS tests and check if they pass."""
    if not fail_to_pass:
        return False, "No FAIL_TO_PASS tests defined"

    test_ids = json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass

    passed = 0
    total = len(test_ids)
    details = []

    is_django = "django" in repo.lower()

    # Build environment with PYTHONPATH for the repo
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)

    if is_django:
        # Django's runtests.py sets its own DJANGO_SETTINGS_MODULE
        env.pop("DJANGO_SETTINGS_MODULE", None)

    # Group all Django tests by module to run together
    if is_django:
        import re
        # Collect unique test labels for runtests.py
        # Format: "test_method (module.path.TestClass)"
        # runtests.py expects: "module.path.TestClass" (dotted path to class or module)
        test_modules = set()
        for test_id in test_ids:
            m = re.match(r"(\S+)\s+\(([^)]+)\)", test_id)
            if m:
                test_name, module_path = m.groups()
                # module_path = "utils_tests.test_lazyobject.SimpleLazyObjectTestCase"
                # runtests.py label = "utils_tests.test_lazyobject.SimpleLazyObjectTestCase"
                test_modules.add(module_path)

        # Run all test modules at once
        labels = " ".join(sorted(test_modules))
        cmd = f"python tests/runtests.py {labels} --parallel 1 --verbosity 0"

        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(workspace),
                capture_output=True, text=True, timeout=timeout, env=env
            )
            output = result.stdout + result.stderr

            if result.returncode == 0 and "FAILED" not in output:
                # All tests passed
                passed = total
                details = [{"test": t, "pass": True} for t in test_ids]
            else:
                details = [{"test": t, "pass": False, "output": output[-500:]} for t in test_ids]
        except subprocess.TimeoutExpired:
            details = [{"test": t, "pass": False, "output": "timeout"} for t in test_ids]
        except Exception as e:
            details = [{"test": t, "pass": False, "output": str(e)} for t in test_ids]

    else:
        # Non-Django: run each test individually with pytest
        # Ensure workspace is first in PYTHONPATH so patched code takes priority
        env["PYTHONPATH"] = str(workspace) + ":" + env.get("PYTHONPATH", "")
        for test_id in test_ids:
            if "::" in test_id:
                cmd = f"python -m pytest {test_id} -x --tb=short --no-header -q"
            else:
                cmd = f"python -m pytest -x --tb=short -k '{test_id}'"

            try:
                result = subprocess.run(
                    cmd, shell=True, cwd=str(workspace),
                    capture_output=True, text=True, timeout=timeout, env=env
                )
                output = result.stdout + result.stderr

                if result.returncode == 0:
                    passed += 1
                    details.append({"test": test_id, "pass": True})
                else:
                    details.append({"test": test_id, "pass": False, "output": output[-500:]})
            except subprocess.TimeoutExpired:
                details.append({"test": test_id, "pass": False, "output": "timeout"})
            except Exception as e:
                details.append({"test": test_id, "pass": False, "output": str(e)})

    all_pass = passed == total
    return all_pass, details


def install_deps(workspace, repo):
    """Install repo-specific test dependencies."""
    repo_lower = repo.lower()

    if "django" in repo_lower:
        # Django: just needs PYTHONPATH set (handled in run_tests via env)
        return

    # For sympy, just PYTHONPATH is enough (pure Python)
    if "sympy" in repo_lower:
        return

    # For other repos, try pip install -e .
    try:
        result = subprocess.run(
            "pip install -e . --quiet --no-build-isolation 2>&1 | tail -3",
            shell=True, cwd=str(workspace), capture_output=True, text=True, timeout=180
        )
    except Exception:
        pass


def evaluate_patches(harness_prefix, output_file, phase1=False):
    """Evaluate all patches for a given harness."""
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Build lookup
    instance_map = {}
    for item in dataset:
        instance_map[item["instance_id"]] = item

    # Find patches
    if phase1:
        # Phase 1 patches don't have harness prefix
        patches = sorted(PATCHES_DIR.glob("*.patch"))
        patches = [p for p in patches if "__" in p.stem and not p.stem.startswith(("langgraph__", "sera__", "aider__"))]
    else:
        patches = sorted(PATCHES_DIR.glob(f"{harness_prefix}__*.patch"))

    print(f"Found {len(patches)} patches for {'phase1' if phase1 else harness_prefix}")

    results = []
    for i, patch_file in enumerate(patches):
        # Extract instance_id from filename
        # Filenames: "langgraph__django__django-11001.patch" or "django__django-11001.patch"
        # Instance IDs: "django__django-11001" (double underscore preserved)
        stem = patch_file.stem
        if not phase1:
            # Remove harness prefix: "langgraph__django__django-11001" -> "django__django-11001"
            instance_id = stem[len(harness_prefix) + 2:]  # skip "harness__"
        else:
            instance_id = stem

        if instance_id not in instance_map:
            print(f"  [{i+1}/{len(patches)}] {instance_id}: SKIP (not in dataset)")
            continue

        instance = instance_map[instance_id]
        print(f"  [{i+1}/{len(patches)}] {instance_id}...", end="", flush=True)

        t0 = time.time()
        try:
            workspace = setup_workspace(instance)
            applied, err = apply_patch(workspace, patch_file)

            if not applied:
                print(f" PATCH FAILED ({err[:100]})")
                results.append({
                    "instance_id": instance_id,
                    "harness": harness_prefix if not phase1 else "phase1",
                    "patch_applied": False,
                    "tests_pass": False,
                })
                continue

            # Apply gold test_patch (adds FAIL_TO_PASS test definitions)
            test_patch = instance.get("test_patch", "")
            if test_patch:
                tp_file = workspace / "_test_patch.diff"
                tp_file.write_text(test_patch)
                tp_result = subprocess.run(
                    ["git", "apply", "--allow-empty", str(tp_file)],
                    cwd=workspace, capture_output=True, text=True
                )
                if tp_result.returncode != 0:
                    # Try with --reject
                    subprocess.run(
                        ["git", "apply", "--reject", "--allow-empty", str(tp_file)],
                        cwd=workspace, capture_output=True, text=True
                    )
                tp_file.unlink(missing_ok=True)

            # Install deps
            install_deps(workspace, instance["repo"])

            # Run FAIL_TO_PASS tests
            fail_to_pass = instance.get("FAIL_TO_PASS", instance.get("fail_to_pass", "[]"))
            all_pass, details = run_tests(workspace, fail_to_pass, instance["repo"])

            elapsed = time.time() - t0
            status = "PASS" if all_pass else "FAIL"
            print(f" {status} ({elapsed:.1f}s)")

            results.append({
                "instance_id": instance_id,
                "harness": harness_prefix if not phase1 else "phase1",
                "patch_applied": True,
                "tests_pass": all_pass,
                "test_details": details,
                "elapsed": round(elapsed, 1),
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f" ERROR: {e}")
            results.append({
                "instance_id": instance_id,
                "harness": harness_prefix if not phase1 else "phase1",
                "patch_applied": False,
                "tests_pass": False,
                "error": str(e),
                "elapsed": round(elapsed, 1),
            })

        # Write incrementally
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

        # Cleanup eval workspace
        eval_ws = WORKSPACE_ROOT / (instance_id.replace("/", "__") + "_eval")
        if eval_ws.exists():
            subprocess.run(["rm", "-rf", str(eval_ws)], capture_output=True)

    # Summary
    total = len(results)
    applied = sum(1 for r in results if r.get("patch_applied"))
    passed = sum(1 for r in results if r.get("tests_pass"))
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Patches: {total}")
    print(f"Applied: {applied}/{total} ({100*applied/max(total,1):.0f}%)")
    print(f"Tests pass: {passed}/{total} ({100*passed/max(total,1):.0f}%)")
    print(f"Pass rate: {passed}/{total} ({100*passed/max(total,1):.0f}%)")

    return results


def generate_report(result_files):
    """Generate comparison report."""
    print("\n=== PATCH EVALUATION REPORT ===\n")
    for fpath in result_files:
        with open(fpath) as f:
            results = [json.loads(l) for l in f if l.strip()]
        if not results:
            continue
        harness = results[0].get("harness", Path(fpath).stem)
        total = len(results)
        applied = sum(1 for r in results if r.get("patch_applied"))
        passed = sum(1 for r in results if r.get("tests_pass"))
        print(f"{harness}: {passed}/{total} pass ({100*passed/max(total,1):.0f}%), {applied}/{total} applied")
        if passed > 0:
            for r in results:
                if r.get("tests_pass"):
                    print(f"  PASS: {r['instance_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", help="Harness name (langgraph, sera, aider)")
    parser.add_argument("--phase1", action="store_true", help="Evaluate Phase 1 patches")
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", nargs="+", help="Generate report from eval files")
    args = parser.parse_args()

    if args.report:
        generate_report(args.report)
    elif args.phase1:
        output = args.output or str(RESULTS_DIR / "eval_phase1.jsonl")
        evaluate_patches("phase1", output, phase1=True)
    elif args.harness:
        output = args.output or str(RESULTS_DIR / f"eval_{args.harness}.jsonl")
        evaluate_patches(args.harness, output)
    else:
        parser.print_help()
