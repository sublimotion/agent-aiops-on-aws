#!/usr/bin/env bash
# Run once the burn-in has completed.
# - Appends burn-in results to the main report
# - Scales HyperPod instance groups back to 0
# - Marks spec COMPLETED
set -euo pipefail

BLUEPRINT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$BLUEPRINT_DIR/../../../../.." && pwd)"

FINAL_JSON="$BLUEPRINT_DIR/results/burn-in/burn-in-final.json"
if [[ ! -f "$FINAL_JSON" ]]; then
    echo "Burn-in not finished yet ($FINAL_JSON missing)"
    echo "Progress file contents:"
    cat "$BLUEPRINT_DIR/results/burn-in/burn-in-progress.json" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Slices: {len(d[\"slices\"])}/12')
"
    exit 1
fi

echo "=== Burn-in complete ==="
python3 <<PYEOF
import json
d = json.load(open("$FINAL_JSON"))
s = d.get("stability", {})
print(f"duration: {d['duration_s']}s")
print(f"concurrency: {d['concurrency']}")
print(f"hour_1 throughput: {s.get('hour_1_throughput', 0):.2f} req/s")
print(f"final throughput:  {s.get('final_throughput', 0):.2f} req/s")
print(f"drift:             {s.get('throughput_drift_pct', 0):+.2f}%")
print(f"errors:            {s.get('unrecoverable_errors', 0)}")
print(f"gate:              {'PASS' if s.get('drift_gate_passed') else 'FAIL'}")
print()
print("Per-slice:")
for sl in d["slices"]:
    print(f"  {sl['slice_idx']+1}: {sl['output_throughput']:6.1f} req/s  p50={sl['latency_p50']:5.0f}ms  p99={sl['latency_p99']:5.0f}ms  err={sl['failed']}")
PYEOF

echo
echo "=== Scaling HyperPod instance groups to 0 ==="
python3 - <<PYEOF
import json, subprocess
existing = json.loads(subprocess.check_output([
    "aws","sagemaker","describe-cluster","--cluster-name","finetune-g5-cluster",
    "--region","us-east-1","--query","InstanceGroups","--output","json"]))
LIFECYCLE = {"SourceS3Uri": "s3://hyperpod-eks-bucket-615299764834-us-east-1", "OnCreate": "on_create.sh"}
EXEC_ROLE = "arn:aws:iam::615299764834:role/hyperpod-eks-ExecutionRole-us-east-1"
groups = []
for g in existing:
    groups.append({
        "InstanceGroupName": g["InstanceGroupName"],
        "InstanceType": g["InstanceType"],
        "InstanceCount": 0,
        "LifeCycleConfig": g["LifeCycleConfig"],
        "ExecutionRole": g["ExecutionRole"],
        "ThreadsPerCore": g["ThreadsPerCore"],
    })
with open("/tmp/scale-down.json", "w") as f:
    json.dump(groups, f, indent=2)
PYEOF
aws sagemaker update-cluster --cluster-name finetune-g5-cluster --region us-east-1 \
  --instance-groups file:///tmp/scale-down.json | head -3

echo
echo "=== Done ==="
echo "1. Review $BLUEPRINT_DIR/results/burn-in/burn-in-final.json"
echo "2. Append burn-in numbers to results/benchmark-report-20260513.md"
echo "3. Update spec status to COMPLETED"
