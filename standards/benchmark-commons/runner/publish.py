#!/usr/bin/env python3
"""
Publish benchmark artifacts from a blueprint to community repos.

Converts blueprint artifacts + infrastructure + lessons into a publishable
bundle conforming to target repo conventions, then creates a PR.

Usage:
    ./publish.py --target ai-on-eks --blueprint domains/gpu-serving/blueprints/kimi-k2.6/ \
                 --repo ~/repos/ai-on-eks --dry-run

    ./publish.py --target hyperpod-recipes --blueprint domains/gpu-serving/blueprints/glm5-hyperpod/ \
                 --repo ~/repos/sagemaker-hyperpod-recipes
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# Target repo conventions
TARGETS = {
    "ai-on-eks": {
        "base_path": "blueprints/inference",
        "structure": {
            "benchmarks/": "*.json artifacts + benchmark.yaml sidecar",
            "manifests/": "K8s YAMLs",
            "terraform/": "IaC files",
            "scripts/": "Setup + benchmark scripts",
            "docker/": "Dockerfiles (if custom)",
            "README.md": "Generated from lessons + spec + results",
        },
    },
    "hyperpod-recipes": {
        "base_path": "inference",
        "structure": {
            "benchmarks/": "*.json artifacts + benchmark.yaml sidecar",
            "config/": "Hydra YAML configs",
            "scripts/": "Launcher + benchmark scripts",
            "README.md": "Generated from lessons + spec + results",
        },
    },
}


def find_artifacts(blueprint_path: Path) -> list[Path]:
    """Find all common artifact JSONs in a blueprint's results directory."""
    results_dir = blueprint_path / "results"
    if not results_dir.exists():
        return []

    artifacts = []
    for f in results_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "schema_version" in data and "metrics" in data:
                artifacts.append(f)
        except (json.JSONDecodeError, KeyError):
            continue
    return artifacts


def find_sidecar(blueprint_path: Path) -> Path | None:
    """Find benchmark.yaml sidecar in blueprint."""
    candidates = [
        blueprint_path / "benchmark.yaml",
        blueprint_path / "results" / "benchmark.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_manifests(blueprint_path: Path) -> list[Path]:
    """Find K8s manifests."""
    manifests = []
    for pattern in ["*.yaml", "*.yml"]:
        manifests.extend(blueprint_path.glob(f"manifests/{pattern}"))
        manifests.extend(blueprint_path.glob(f"k8s/{pattern}"))
    return manifests


def find_terraform(blueprint_path: Path) -> list[Path]:
    """Find Terraform files."""
    tf_files = []
    for pattern in ["*.tf", "*.tfvars"]:
        tf_files.extend(blueprint_path.glob(f"terraform/{pattern}"))
        tf_files.extend(blueprint_path.glob(f"infra/{pattern}"))
    return tf_files


def find_scripts(blueprint_path: Path) -> list[Path]:
    """Find operational scripts."""
    scripts = []
    for pattern in ["*.sh", "*.py"]:
        scripts.extend(blueprint_path.glob(f"scripts/{pattern}"))
    return scripts


def generate_readme(blueprint_path: Path, artifacts: list[Path]) -> str:
    """Generate a README from lessons, spec, and results."""
    name = blueprint_path.name
    lessons_file = blueprint_path / "lessons.md"

    readme = f"# {name}\n\n"

    # Summary from lessons if available
    if lessons_file.exists():
        with open(lessons_file) as f:
            lessons_content = f.read()
        # Extract first section as summary
        lines = lessons_content.split("\n")
        summary_lines = []
        for line in lines[2:20]:  # Skip title, grab first ~18 lines
            if line.startswith("## ") and summary_lines:
                break
            summary_lines.append(line)
        readme += "## Overview\n\n" + "\n".join(summary_lines).strip() + "\n\n"

    # Benchmark results summary
    if artifacts:
        readme += "## Benchmark Results\n\n"
        readme += "| Workload | Engine | Throughput | TTFT p50 | TPOT p50 | SLO |\n"
        readme += "|----------|--------|-----------|----------|----------|-----|\n"

        for art_path in sorted(artifacts):
            with open(art_path) as f:
                art = json.load(f)
            workload = art.get("workload", {}).get("catalog_id", "custom")
            engine = art.get("engine", {}).get("name", "?")
            metrics = art.get("metrics", {})
            throughput = f"{metrics.get('output_toks_per_s', 0):.0f} tok/s"
            ttft = f"{metrics.get('ttft_ms', {}).get('p50', 0):.0f} ms"
            tpot = f"{metrics.get('tpot_ms', {}).get('p50', 0):.1f} ms"
            slo = art.get("slo", {}).get("overall_pass", "N/A")
            readme += f"| {workload} | {engine} | {throughput} | {ttft} | {tpot} | {slo} |\n"

        readme += "\nFull artifacts in `benchmarks/` directory.\n\n"

    # Deployment
    readme += "## Deployment\n\n"
    readme += "See `manifests/` for K8s resources and `terraform/` for infrastructure.\n\n"

    # Lessons
    if lessons_file.exists():
        readme += "## Operational Lessons\n\n"
        readme += f"See [{lessons_file.name}]({lessons_file.name}) for detailed operational knowledge.\n"

    return readme


def build_publication(
    blueprint_path: Path,
    target: str,
    output_dir: Path,
) -> Path:
    """Build the publication bundle."""
    target_config = TARGETS[target]
    pub_name = blueprint_path.name
    pub_dir = output_dir / target_config["base_path"] / pub_name

    # Clean and create
    if pub_dir.exists():
        shutil.rmtree(pub_dir)
    pub_dir.mkdir(parents=True)

    # Artifacts
    artifacts = find_artifacts(blueprint_path)
    if artifacts:
        benchmarks_dir = pub_dir / "benchmarks"
        benchmarks_dir.mkdir()
        for art in artifacts:
            shutil.copy2(art, benchmarks_dir / art.name)

    # Sidecar
    sidecar = find_sidecar(blueprint_path)
    if sidecar:
        shutil.copy2(sidecar, pub_dir / "benchmarks" / "benchmark.yaml")

    # Manifests
    manifests = find_manifests(blueprint_path)
    if manifests:
        manifests_dir = pub_dir / "manifests"
        manifests_dir.mkdir()
        for m in manifests:
            shutil.copy2(m, manifests_dir / m.name)

    # Terraform
    tf_files = find_terraform(blueprint_path)
    if tf_files:
        tf_dir = pub_dir / "terraform"
        tf_dir.mkdir()
        for tf in tf_files:
            shutil.copy2(tf, tf_dir / tf.name)

    # Scripts
    scripts = find_scripts(blueprint_path)
    if scripts:
        scripts_dir = pub_dir / "scripts"
        scripts_dir.mkdir()
        for s in scripts:
            shutil.copy2(s, scripts_dir / s.name)

    # Lessons
    lessons = blueprint_path / "lessons.md"
    if lessons.exists():
        shutil.copy2(lessons, pub_dir / "lessons.md")

    # README
    readme_content = generate_readme(blueprint_path, artifacts)
    (pub_dir / "README.md").write_text(readme_content)

    return pub_dir


def create_pr(repo_path: Path, pub_dir: Path, blueprint_name: str, target: str, dry_run: bool):
    """Create a PR in the target repo."""
    branch = f"benchmark/{blueprint_name}-{datetime.now().strftime('%Y%m%d')}"

    if dry_run:
        print(f"\n[DRY RUN] Would create PR:")
        print(f"  Repo:   {repo_path}")
        print(f"  Branch: {branch}")
        print(f"  Title:  Add {blueprint_name} inference benchmark")
        print(f"  Files:  {sum(1 for _ in pub_dir.rglob('*') if _.is_file())} files")
        return

    # Copy publication to target repo
    target_dir = repo_path / pub_dir.relative_to(pub_dir.parent.parent)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(pub_dir, target_dir)

    # Git operations
    os.chdir(repo_path)
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", str(target_dir.relative_to(repo_path))], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add {blueprint_name} inference benchmark\n\nIncludes deployment artifacts, benchmark results (common artifact format),\nand operational lessons from production deployment."],
        check=True,
    )

    # Create PR
    result = subprocess.run(
        ["gh", "pr", "create",
         "--title", f"Add {blueprint_name} inference benchmark",
         "--body", f"## Summary\n\nAdds inference benchmark results and deployment artifacts for {blueprint_name}.\n\nAll benchmark results conform to the common benchmark artifact spec (schema v1.0.0).\n\n## Contents\n\n- Benchmark artifacts (common format)\n- K8s manifests / Terraform\n- Operational lessons\n- README with results summary"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"PR created: {result.stdout.strip()}")
    else:
        print(f"PR creation failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Publish benchmark artifacts to community repos")
    parser.add_argument("--target", required=True, choices=list(TARGETS.keys()))
    parser.add_argument("--blueprint", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path, help="Path to target repo clone")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blueprint_path = args.blueprint.resolve()
    if not blueprint_path.exists():
        print(f"Error: blueprint not found: {blueprint_path}")
        sys.exit(1)

    print(f"=== Publish to {args.target} ===")
    print(f"Blueprint: {blueprint_path}")
    print(f"Target:    {args.repo}")
    print()

    # Build publication bundle in temp location
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pub_dir = build_publication(blueprint_path, args.target, tmp_path)

        # Show what would be published
        print("Files to publish:")
        for f in sorted(pub_dir.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f.relative_to(pub_dir)} ({size:,} bytes)")

        artifacts = find_artifacts(blueprint_path)
        print(f"\nArtifacts: {len(artifacts)}")
        print(f"Manifests: {len(find_manifests(blueprint_path))}")
        print(f"Terraform: {len(find_terraform(blueprint_path))}")
        print(f"Scripts:   {len(find_scripts(blueprint_path))}")

        # Create PR
        create_pr(args.repo, pub_dir, blueprint_path.name, args.target, args.dry_run)


if __name__ == "__main__":
    main()
