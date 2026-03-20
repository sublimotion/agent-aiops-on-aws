#!/usr/bin/env bash
# Deploy Ray Serve FT blueprint:
#   KubeRay operator + stunnel ConfigMap + YOLO RayService (ElastiCache via stunnel)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLUEPRINT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$BLUEPRINT_DIR/k8s"
TF_DIR="$BLUEPRINT_DIR/terraform"
NAMESPACE="ray-ft"
CLUSTER_NAME="${EKS_CLUSTER:-qn-sglang-eks-cluster}"
REGION="${AWS_REGION:-us-west-2}"

echo "=== Ray Serve FT Deployment ==="

# 1. Update kubeconfig
echo "[1/6] Configuring kubectl for $CLUSTER_NAME..."
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"

# 2. Install KubeRay operator
echo "[2/6] Installing KubeRay operator..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
helm repo update kuberay
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --version 1.3.0 \
    --namespace kuberay-system \
    --create-namespace \
    --wait --timeout 120s

# 3. Create namespace + patch stunnel ConfigMap with Terraform endpoint
echo "[3/6] Setting up namespace and stunnel TLS proxy config..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Get ElastiCache endpoint from Terraform
ELASTICACHE_ENDPOINT=$(cd "$TF_DIR" && terraform output -raw stunnel_connect 2>/dev/null || echo "")
if [ -z "$ELASTICACHE_ENDPOINT" ]; then
    echo "  WARNING: Could not read Terraform output. Using existing stunnel.yaml as-is."
    echo "  Run 'cd terraform && terraform output -raw stunnel_connect' to get the endpoint."
else
    echo "  ElastiCache endpoint: $ELASTICACHE_ENDPOINT"
    # Patch the stunnel ConfigMap with the actual endpoint
    sed "s|ray-ft-gcs-.*\.serverless\..*\.cache\.amazonaws\.com:6379|${ELASTICACHE_ENDPOINT}|g" \
        "$K8S_DIR/stunnel.yaml" | kubectl apply -f -
fi
kubectl apply -f "$K8S_DIR/stunnel.yaml"

# 4. Create YOLO serve app ConfigMap
echo "[4/6] Creating YOLO serve ConfigMap..."
kubectl create configmap yolo-serve-app \
    --from-file=yolo_serve.py="$SCRIPT_DIR/yolo_serve.py" \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -

# 5. Deploy RayService with GCS FT
echo "[5/6] Deploying YOLO RayService (GCS FT enabled, stunnel → ElastiCache)..."
kubectl apply -f "$K8S_DIR/ray-service.yaml"

# 6. Wait for RayService to be ready
echo "[6/6] Waiting for RayService to be ready..."
echo "  (This may take 3-5 minutes for stunnel init + pip install + model download)"

for i in $(seq 1 60); do
    STATUS=$(kubectl get rayservice yolo-ft -n "$NAMESPACE" \
        -o jsonpath='{.status.activeServiceStatus.applicationStatuses.yolo.status}' 2>/dev/null || echo "Pending")
    HEAD_READY=$(kubectl get pods -n "$NAMESPACE" -l ray-node=head \
        -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="ray-head")].ready}' 2>/dev/null || echo "false")
    WORKER_COUNT=$(kubectl get pods -n "$NAMESPACE" -l ray-node=worker \
        --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l | tr -d ' ')

    echo "  [$i/60] app=$STATUS head_ready=$HEAD_READY workers=$WORKER_COUNT/2"

    if [ "$STATUS" = "RUNNING" ] && [ "$WORKER_COUNT" -ge 2 ]; then
        echo ""
        echo "=== Deployment Complete ==="
        echo "  RayService: yolo-ft (GCS FT ENABLED)"
        echo "  Namespace:  $NAMESPACE"
        echo "  Redis:      ElastiCache Serverless via stunnel (:6380 → TLS)"
        echo ""
        echo "  Dashboard:  kubectl port-forward -n $NAMESPACE svc/yolo-ft-head-svc 8265:8265"
        echo "  Serve:      kubectl port-forward -n $NAMESPACE svc/yolo-ft-serve-svc 8000:8000"
        echo ""
        echo "  Test:"
        echo "    python3 -c \""
        echo "    import base64, json, urllib.request"
        echo "    from PIL import Image; import io"
        echo "    img = Image.new('RGB', (100,100), 'red')"
        echo "    buf = io.BytesIO(); img.save(buf, 'PNG')"
        echo "    b64 = base64.b64encode(buf.getvalue()).decode()"
        echo "    req = urllib.request.Request('http://localhost:8000/', json.dumps({'image':b64}).encode(), {'Content-Type':'application/json'})"
        echo "    print(urllib.request.urlopen(req, timeout=30).read().decode())"
        echo "    \""
        exit 0
    fi
    sleep 10
done

echo "WARNING: RayService not fully ready after 10 minutes. Check with:"
echo "  kubectl get rayservice -n $NAMESPACE"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -n $NAMESPACE \$(kubectl get pods -n $NAMESPACE -l ray-node=head -o jsonpath='{.items[0].metadata.name}') -c ray-head"
exit 1
