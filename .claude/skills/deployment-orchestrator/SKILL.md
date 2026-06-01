---
name: deployment-orchestrator
description: Pre-flight validation, post-deployment verification, and failure recovery for AWS infrastructure deployments. Use when user says "pre-flight check", "readiness audit", "deployment checklist", "validate deployment", or "recover from failed deploy". Do NOT use for writing Terraform code (use terraform-automation) or for benchmark analysis (use benchmark-analyst agent).
---

# Deployment Orchestrator Skill

Encodes the operational playbooks learned from GPU-serving and AgentCore Runtime deployments. This skill provides pre-flight validation checklists, post-deployment verification, and failure recovery procedures.

## When to Use

- Before a capacity block session — run the pre-flight checklist
- After `terraform apply` — run post-deployment verification
- When a deployment stage fails — consult failure recovery playbooks
- Before benchmarks — validate serving stack readiness

## Pre-Flight Checklists

### GPU Serving Pre-Flight

Run before every capacity block. Each item must be PASS before proceeding.

#### EKS Cluster
```bash
# Cluster reachable
kubectl cluster-info
# System nodes ready
kubectl get nodes -l node-role.kubernetes.io/system
# CoreDNS running
kubectl get pods -n kube-system -l k8s-app=kube-dns
# kube-proxy running
kubectl get pods -n kube-system -l k8s-app=kube-proxy
```

#### Storage (FSx Lustre)
```bash
# FSx lifecycle AVAILABLE
aws fsx describe-file-systems --file-system-ids <id> --query 'FileSystems[0].Lifecycle'
# PV/PVC bound
kubectl get pv,pvc -n <namespace>
# FSx CSI driver running
kubectl get pods -n kube-system -l app=fsx-csi-controller
# NVMe RAID mounted (on GPU node)
kubectl exec -n <ns> <pod> -- df -h /mnt/nvme
# Model files present
kubectl exec -n <ns> <pod> -- ls /mnt/nvme/model/
```

#### Container Images (ECR)
```bash
# For every Dockerfile, verify ECR repo exists
for repo in $(grep -r 'FROM\|ECR_REPO' configs/ docker/ | grep -oP '\d+\.dkr\.ecr\.[^/]+/[^:]+' | sort -u); do
  aws ecr describe-repositories --repository-names "$(basename $repo)" 2>/dev/null && echo "PASS: $repo" || echo "FAIL: $repo"
done
# Cross-reference stage-images-ecr.sh
bash -n scripts/stage-images-ecr.sh  # Syntax check
```

#### GPU / Accelerator Plugins
Items marked PENDING are expected if the GPU node hasn't joined yet — they self-heal on node join. Do not investigate PENDING items unless the node has joined and they are still not running.
```bash
kubectl get ds -n kube-system nvidia-device-plugin-daemonset
kubectl get ds -n kube-system aws-efa-k8s-device-plugin-daemonset
kubectl get ds -n kube-system dcgm-exporter
```

#### Config Scripts & Benchmark Wiring
```bash
# All configs pass syntax check
for f in configs/*.sh; do bash -n "$f" && echo "PASS: $f" || echo "FAIL: $f"; done
# Benchmark script references all configs
python3 -c "
import ast, glob
configs = [f.split('/')[-1].replace('.sh','') for f in glob.glob('configs/*.sh')]
print('Configs found:', configs)
"
```

#### Serving-Config Resolver (fail-closed)
Run the deterministic resolver over the blueprint's sidecar. Exit code 2 = hard-rule FAIL (must fix the sidecar before deploying); 0 = clean. WARN/INFO and `prior-failure:*` corpus findings do not block but must be read.
```bash
python3 standards/serving-commons/resolver/validate-serving-config.py \
  --sidecar domains/gpu-serving/blueprints/<name>/benchmark.yaml \
  --corpus-root .   # replays prior lessons.md failure_categories for this model/engine
# exit 2 => FAIL: apply the printed fix (TP, max_model_len, AMI, ...) and re-run.
```
This codifies the FP8/TP, max-model-len, AMI, LMCache/MLA, HiCache, and spec-decode rules — see `standards/serving-commons/README.md`. The model-load recovery notes below describe the *symptoms*; this gate catches them from the declared config first.

### AgentCore Runtime Pre-Flight

#### Foundation
```bash
# VPC endpoints present
aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=<vpc_id>" \
  --query 'VpcEndpoints[].ServiceName'
# Required: bedrock-runtime, bedrock-agent-runtime, ecr.api, ecr.dkr, s3, dynamodb, secretsmanager
```

#### AgentCore Status
```bash
aws bedrock-agent get-agent --agent-id <id> --query 'agent.agentStatus'
# Must be PREPARED
aws bedrock-agent list-agent-aliases --agent-id <id>
# At least one alias must exist
```

#### Auth (Cognito)
```bash
aws cognito-idp describe-user-pool --user-pool-id <pool_id> --query 'UserPool.Status'
aws cognito-idp admin-get-user --user-pool-id <pool_id> --username test-user \
  --query 'UserStatus'
# Must be CONFIRMED
```

#### Proxy (ECS)
```bash
aws ecs describe-services --cluster <cluster> --services <service> \
  --query 'services[0].{desired:desiredCount,running:runningCount}'
# desired == running
```

## Post-Deployment Verification

After each deployment stage, verify:

1. **Terraform outputs captured** — record VPC ID, subnet IDs, cluster name, FSx DNS, ECR URIs
2. **Health endpoints responding** — `curl <endpoint>/health` returns 200
3. **Test request succeeds** — single inference request returns valid output
4. **Logs flowing** — check CloudWatch (ECS) or OTEL collector (AgentCore) for recent entries
5. **Monitoring active** — Prometheus scraping targets, Grafana dashboards loading

## Failure Recovery Playbooks

### Capacity Block: InsufficientInstanceCapacity

**Symptom**: `aws ec2 run-instances` fails with InsufficientInstanceCapacity
**Recovery**:
1. Shotgun launch across multiple regions: us-east-1, us-east-2, us-west-2
2. Do NOT trust dry-run success as a capacity signal — it only validates permissions/quotas
3. For scarce instance types (g7e), use on-demand or spot — capacity blocks may not be supported
4. Consider bare EC2 over EKS when capacity is unpredictable

### EKS: Node Fails to Join Cluster

**Symptom**: GPU instance launched but not visible in `kubectl get nodes`
**Recovery**:
1. Check AMI type — AL2023 uses `nodeadm` not `bootstrap.sh`
2. Verify EKS access entry exists for the instance role
3. Check security group allows communication with EKS API server
4. For capacity block instances, create access entry manually:
```bash
aws eks create-access-entry --cluster-name <cluster> \
  --principal-arn <instance-role-arn> --type EC2_LINUX
```

### Container: nerdctl/containerd Issues on Bare EC2

**Symptom**: Container commands fail on EKS-optimized AMI outside EKS
**Recovery**:
1. Start containerd: `sudo systemctl start containerd`
2. Use nerdctl, not docker: `nerdctl --gpus 4 run ...`
3. Do NOT combine `--rm` with `-d` — nerdctl doesn't support this
4. Expand root partition if needed: `growpart /dev/nvme0n1 1 && xfs_growfs /`

### Serving Stack: Model Fails to Load

**Symptom**: vLLM/SGLang pod crashes or hangs during model loading
**Recovery**:
1. **FP8 + MoE TP incompatibility**: Verify all weight dimensions divisible by `block_k=128` at target TP. TP=8 on shared experts with `input_size=512` → `64/partition` → ValueError. Fall back to TP=4. **This is now codified** — the serving-config resolver (`rule fp8-moe-tp-divisibility`) catches it deterministically from the sidecar before deploy, given `model.moe_intermediate_size`. Reaching this recovery step means the pre-flight gate was skipped or the sidecar lacked `moe_intermediate_size`.
2. **Missing model in framework registry**: Check `transformers.AutoConfig.from_pretrained()` → `architectures` field. Verify model is in SGLang/vLLM registry.
3. **Hybrid architecture**: Models with mamba layers trigger different cache mode. Prefix caching may conflict with MTP. Test with `--no-enable-prefix-caching` if MTP is needed.
4. **JIT compilation startup**: First launch with DeepGEMM/TensorRT can take 10-15 min. Wait for health endpoint, don't restart prematurely.

### AgentCore: Runtime Status FAILED

**Symptom**: `get-agent` returns status FAILED
**Recovery**:
1. Get failure reason: `aws bedrock-agent get-agent --agent-id <id> --query 'agent.failureReasons'`
2. Common causes:
   - Missing VPC endpoint for `bedrock-runtime`
   - IAM role missing `bedrock:InvokeModel`
   - Foundation model not available in region
3. Fix the cause, then re-apply: `terraform apply -target=module.agentcore_runtime`
4. After update, pin endpoint version:
```bash
aws bedrock-agent update-agent-runtime-endpoint \
  --agent-runtime-id <id> --agent-runtime-version <N>
```

### Terraform: State Lock Conflict

**Symptom**: `Error acquiring the state lock`
**Recovery**:
1. Find orphaned processes: `ps aux | grep terraform`
2. Kill them: `pkill -9 terraform`
3. Force unlock: `terraform force-unlock <LOCK_ID>`
4. Prevention: never run terraform in parallel background tasks

## Operational Artifacts

Every deployment must produce these artifacts in the blueprint's directory:

| Artifact | Path | When |
|----------|------|------|
| Readiness audit | `results/readiness-audit-<YYYYMMDD>.md` | Before every capacity block or deployment |
| Deployment log | `results/deployment-log-<YYYYMMDD>.md` | During deployment (timestamped) |
| Lessons learned | `lessons.md` | After deployment (append-only) |
| Compound summary | `results/compound-<YYYYMMDD>.md` | After compound-learner runs |
| Benchmark report | `results/benchmark-report.md` | After benchmarks (if applicable) |

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Skill triggers on deployment/checklist requests | 90%+ | Test with "pre-flight check", "readiness audit", "deployment failed" |
| Does NOT trigger on Terraform authoring | 0% false triggers | Test with "create VPC", "write terraform" |
| Pre-flight catches blocking issues | 100% | Run checklist against a known-broken deployment |
| Recovery playbook resolves issue | 80%+ | Track whether suggested fix works without user correction |
| All artifacts generated per deployment | 100% | Verify all 4 artifact files exist after deployment |
