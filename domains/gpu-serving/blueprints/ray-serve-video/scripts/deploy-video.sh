#!/usr/bin/env bash
###############################################################################
# deploy-video.sh — Deploy Ray Serve Video Pipeline on EKS
#
# Reuses existing EKS cluster + ElastiCache from ray-serve-ft.
# Deploys Kafka + multi-framework Ray Serve pipeline in ray-video namespace.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLUEPRINT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
K8S_DIR="$BLUEPRINT_DIR/k8s"
TF_DIR="$BLUEPRINT_DIR/../ray-serve-ft/terraform"

EKS_CLUSTER="${EKS_CLUSTER:-qn-sglang-eks-cluster}"
AWS_REGION="${AWS_REGION:-us-west-2}"
NAMESPACE="ray-video"
GPU_NODE_ROLE="gpu-eks-node-group-20260303162535678600000025"

# Which pipeline to deploy (video_pipeline or video_pipeline_s3)
PIPELINE_FILE="${PIPELINE_FILE:-video_pipeline.py}"

echo "============================================================"
echo " Ray Serve Video Pipeline Deployment"
echo " Cluster: $EKS_CLUSTER | Region: $AWS_REGION"
echo " Pipeline: $PIPELINE_FILE"
echo "============================================================"

# --- Step 1/8: kubeconfig ---
echo ""
echo "[1/8] Updating kubeconfig..."
aws eks update-kubeconfig --name "$EKS_CLUSTER" --region "$AWS_REGION" --alias "$EKS_CLUSTER"

# --- Step 2/8: KubeRay operator (idempotent) ---
echo ""
echo "[2/8] Ensuring KubeRay operator..."
if helm list -n kuberay-system 2>/dev/null | grep -q kuberay-operator; then
    echo "  KubeRay operator already installed"
else
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
    helm repo update kuberay
    helm install kuberay-operator kuberay/kuberay-operator \
        --version 1.3.0 \
        --namespace kuberay-system \
        --create-namespace \
        --wait --timeout 120s
fi

# --- Step 3/8: Free GPU nodes (scale down yolo-ft if running) ---
echo ""
echo "[3/8] Checking for existing RayService on GPU nodes..."
if kubectl get rayservice yolo-ft -n ray-ft 2>/dev/null; then
    echo "  Deleting yolo-ft RayService to free GPU nodes..."
    kubectl delete rayservice yolo-ft -n ray-ft --timeout=120s
    echo "  Waiting for GPU pods to terminate..."
    kubectl wait --for=delete pod -l ray.io/cluster -n ray-ft --timeout=120s 2>/dev/null || true
    sleep 5
    echo "  GPU nodes freed"
else
    echo "  No yolo-ft RayService found, GPU nodes available"
fi

# --- Step 4/8: Add S3 access to GPU node role ---
echo ""
echo "[4/8] Ensuring S3 access on GPU node role..."
if aws iam list-attached-role-policies --role-name "$GPU_NODE_ROLE" 2>/dev/null | grep -q AmazonS3ReadOnlyAccess; then
    echo "  S3 read policy already attached"
else
    echo "  Attaching AmazonS3ReadOnlyAccess..."
    aws iam attach-role-policy \
        --role-name "$GPU_NODE_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
    echo "  Attached (may take ~30s to propagate)"
fi

# Set IMDS hop limit to 2 so pods can access instance metadata (S3 credentials)
echo "  Ensuring IMDS hop limit = 2 on all nodes..."
for INSTANCE_ID in $(aws ec2 describe-instances \
    --filters "Name=tag:eks:cluster-name,Values=$EKS_CLUSTER" "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].InstanceId" --output text --region "$AWS_REGION" 2>/dev/null); do
    HOP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
        --query "Reservations[0].Instances[0].MetadataOptions.HttpPutResponseHopLimit" --output text 2>/dev/null)
    if [ "$HOP" != "2" ]; then
        aws ec2 modify-instance-metadata-options --instance-id "$INSTANCE_ID" \
            --http-put-response-hop-limit 2 --region "$AWS_REGION" --output text > /dev/null 2>&1
        echo "    $INSTANCE_ID: hop limit set to 2"
    fi
done

# --- Step 5/8: Create namespace + stunnel ConfigMap ---
echo ""
echo "[5/8] Setting up namespace and stunnel..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Patch stunnel endpoint from terraform output
STUNNEL_CONNECT=""
if [ -d "$TF_DIR" ]; then
    STUNNEL_CONNECT=$(cd "$TF_DIR" && terraform output -raw stunnel_connect 2>/dev/null || echo "")
fi
if [ -n "$STUNNEL_CONNECT" ]; then
    echo "  Patching stunnel endpoint: $STUNNEL_CONNECT"
    sed "s|ray-ft-gcs-.*\.serverless\..*\.cache\.amazonaws\.com:6379|${STUNNEL_CONNECT}|" \
        "$K8S_DIR/stunnel.yaml" | kubectl apply -f -
else
    echo "  WARNING: Could not get ElastiCache endpoint from terraform, using default"
    kubectl apply -f "$K8S_DIR/stunnel.yaml"
fi

# --- Step 6/8: Deploy Kafka ---
echo ""
echo "[6/8] Deploying Kafka..."
kubectl apply -f "$K8S_DIR/kafka.yaml"

echo "  Waiting for Kafka broker to be ready..."
for i in $(seq 1 60); do
    if kubectl get pod kafka-broker-0 -n "$NAMESPACE" 2>/dev/null | grep -q "Running"; then
        # Check if container is actually ready
        if kubectl get pod kafka-broker-0 -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null | grep -q "true"; then
            echo "  Kafka broker ready (attempt $i)"
            break
        fi
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: Kafka broker not ready after 120s"
        kubectl describe pod kafka-broker-0 -n "$NAMESPACE" 2>/dev/null | tail -20
        exit 1
    fi
    sleep 2
done

echo "  Waiting for topic creation job..."
kubectl wait --for=condition=complete job/kafka-topic-init -n "$NAMESPACE" --timeout=120s 2>/dev/null || {
    echo "  Topic init job may still be running, continuing..."
}

# --- Step 7/8: Deploy Ray Serve application ---
echo ""
echo "[7/8] Deploying Ray Serve video pipeline..."

# Create ConfigMap from Python app
kubectl create configmap video-pipeline-app \
    --from-file=video_pipeline.py="$SCRIPT_DIR/$PIPELINE_FILE" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$K8S_DIR/ray-service-video.yaml"

# --- Step 8/8: Poll for readiness ---
echo ""
echo "[8/8] Waiting for Ray Serve to become ready..."
echo "  (This may take 3-5 minutes — pip install of torch/tensorflow is slow)"

for i in $(seq 1 90); do
    APP_STATUS=$(kubectl get rayservice video-pipeline -n "$NAMESPACE" \
        -o jsonpath='{.status.activeServiceStatus.applicationStatuses.video.status}' 2>/dev/null || echo "")

    HEAD_READY=$(kubectl get pods -n "$NAMESPACE" -l ray-node=head \
        -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="ray-head")].ready}' 2>/dev/null || echo "")

    WORKER_COUNT=$(kubectl get pods -n "$NAMESPACE" -l ray-node=worker \
        --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')

    echo "  [$i/90] app=$APP_STATUS head_ready=$HEAD_READY workers=$WORKER_COUNT/2"

    if [ "$APP_STATUS" = "RUNNING" ] && [ "$HEAD_READY" = "true" ] && [ "$WORKER_COUNT" -ge 2 ]; then
        echo ""
        echo "============================================================"
        echo " DEPLOYMENT SUCCESSFUL"
        echo "============================================================"
        echo ""
        echo "Dashboard:"
        echo "  kubectl port-forward svc/video-pipeline-head-svc -n $NAMESPACE 8265:8265"
        echo "  open http://localhost:8265"
        echo ""
        echo "Status endpoint:"
        echo "  kubectl port-forward svc/video-pipeline-head-svc -n $NAMESPACE 8000:8000"
        echo "  curl http://localhost:8000/"
        echo ""
        echo "Produce test messages:"
        echo "  kubectl cp scripts/produce_test.py $NAMESPACE/<head-pod>:/tmp/produce_test.py"
        echo "  kubectl exec -n $NAMESPACE <head-pod> -c ray-head -- python /tmp/produce_test.py"
        echo ""
        exit 0
    fi
    sleep 10
done

echo ""
echo "ERROR: Deployment did not become ready within 15 minutes"
echo "Debug:"
kubectl get rayservice video-pipeline -n "$NAMESPACE" -o yaml 2>/dev/null | tail -30
kubectl get pods -n "$NAMESPACE" -o wide
kubectl logs -n "$NAMESPACE" -l ray-node=head -c ray-head --tail=50 2>/dev/null || true
exit 1
