#!/usr/bin/env python3
"""
Validates enriched benchmark artifacts against the JSON Schema.

Usage:
  validate-artifact results/*.json
  validate-artifact --strict artifact.json  # fail on missing optional fields
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("ERROR: jsonschema not installed. Run: pip install jsonschema>=4.20")


SCHEMA_PATHS = [
    Path("/etc/aiperf/schema/enriched-artifact.json"),
    Path(__file__).parent / "schema" / "enriched-artifact.json",
]


def load_schema() -> dict:
    for path in SCHEMA_PATHS:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    sys.exit("ERROR: schema not found at any expected location")


def validate_one(artifact_path: Path, schema: dict, strict: bool = False) -> tuple[bool, str]:
    """Validate a single artifact. Returns (pass, message)."""
    try:
        with open(artifact_path) as f:
            artifact = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    # Schema validation
    try:
        jsonschema.validate(artifact, schema)
    except jsonschema.ValidationError as e:
        return False, f"Schema error: {e.message} at {'.'.join(str(p) for p in e.absolute_path)}"

    # Semantic checks
    errors = []

    # Check required fields exist with non-null values
    if not artifact.get("artifact_id"):
        errors.append("missing artifact_id")
    if not artifact.get("model", {}).get("id"):
        errors.append("missing model.id")
    if not artifact.get("engine", {}).get("container_image"):
        errors.append("missing engine.container_image")

    # Metrics sanity
    metrics = artifact.get("metrics", {})
    if metrics.get("error_rate", 0) > 0.5:
        errors.append(f"error_rate={metrics['error_rate']:.2f} — more than 50% requests failed")
    if metrics.get("completed", 0) < 10:
        errors.append(f"only {metrics.get('completed', 0)} completed requests — insufficient for statistical confidence")

    # Strict mode: check optional but recommended fields
    if strict:
        if not artifact.get("slo"):
            errors.append("[strict] missing SLO evaluation")
        if not artifact.get("extensions"):
            errors.append("[strict] missing extensions block")
        ttft = metrics.get("ttft_ms", {})
        if ttft and ttft.get("p99") is None:
            errors.append("[strict] missing ttft_ms.p99")

    if errors:
        return False, "; ".join(errors)
    return True, "PASS"


def main():
    parser = argparse.ArgumentParser(description="Validate enriched benchmark artifacts")
    parser.add_argument("files", nargs="+", help="Artifact JSON files to validate")
    parser.add_argument("--strict", action="store_true", help="Require optional fields")

    args = parser.parse_args()
    schema = load_schema()

    passed = 0
    failed = 0

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"SKIP {path} — file not found")
            continue

        ok, msg = validate_one(path, schema, args.strict)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {path.name} — {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
