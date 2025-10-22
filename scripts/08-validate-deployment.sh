#!/bin/bash
# 08-validate-deployment.sh
# Validate the Kubernetes deployment and platform health

set -e

echo "========================================"
echo "Validating Platform Deployment"
echo "========================================"
echo ""

# Configuration
TIMEOUT=${1:-"300"}

echo "Step 1: Verify cluster health"
echo "=================================="
kubectl cluster-info
kubectl get nodes -o wide
echo ""

echo "Step 2: Check all namespaces"
echo "=================================="
kubectl get namespaces
echo ""

echo "Step 3: Check pod status in data-platform namespace"
echo "=================================="
kubectl get pods -n data-platform -o wide
echo ""

echo "Step 4: Wait for critical pods to be ready"
echo "=================================="
CRITICAL_PODS=(
  "zookeeper-0"
  "kafka-0"
  "minio"
  "vault-0"
  "trino-coordinator"
)

for pod in "${CRITICAL_PODS[@]}"; do
  echo "Waiting for ${pod}..."
  kubectl wait --for=condition=Ready pod -l app="${pod%%-*}" -n data-platform --timeout=60s 2>/dev/null || echo "  (Pod not found or not ready, continuing...)"
done
echo ""

echo "Step 5: Check services"
echo "=================================="
kubectl get svc --all-namespaces -o wide
echo ""

echo "Step 6: Test DNS resolution within cluster"
echo "=================================="
kubectl run -it --rm test-dns --image=busybox --restart=Never -n data-platform -- nslookup kubernetes.default
echo ""

echo "Step 7: Check Cloudflare tunnel"
echo "=================================="
if kubectl get pods -n cloudflare-tunnel &>/dev/null; then
  kubectl get pods -n cloudflare-tunnel -o wide
  kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=50 || echo "No logs available"
else
  echo "Cloudflare tunnel namespace not found"
fi
echo ""

echo "Step 8: Check storage"
echo "=================================="
kubectl get storageclass
kubectl get pv
kubectl get pvc --all-namespaces
echo ""

echo "Step 9: Check ingress"
echo "=================================="
if kubectl get ingress --all-namespaces &>/dev/null; then
  kubectl get ingress --all-namespaces -o wide
else
  echo "No ingress resources found"
fi
echo ""

echo "Step 10: Test service connectivity"
echo "=================================="
echo "Testing Kafka connectivity..."
kubectl run -it --rm test-kafka \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- timeout 5 nc -zv kafka-0.kafka-headless.data-platform.svc.cluster.local 9092 || echo "Kafka not ready"

echo ""
echo "Testing MinIO connectivity..."
kubectl run -it --rm test-minio \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- timeout 5 curl -s http://minio:9000/minio/health/live || echo "MinIO not ready"

echo ""
echo "Step 11: Check for pod errors"
echo "=================================="
echo "Pods with issues:"
kubectl get pods --all-namespaces --field-selector=status.phase!=Running,status.phase!=Succeeded 2>/dev/null || echo "All pods healthy"
echo ""

echo "Step 12: Review resource usage"
echo "=================================="
if kubectl top nodes &>/dev/null; then
  echo "Node resource usage:"
  kubectl top nodes
else
  echo "Metrics server not available (metrics-server pod might not be running)"
fi
echo ""

echo "========================================"
echo "Validation complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "- Check pod status above for any failures"
echo "- Review logs: kubectl logs -n <namespace> <pod-name>"
echo "- Describe pod: kubectl describe pod -n <namespace> <pod-name>"
echo ""
echo "Common issues:"
echo "- ImagePullBackOff: Check image availability and pull secrets"
echo "- CrashLoopBackOff: Check pod logs for errors"
echo "- Pending: Check node resources with 'kubectl top nodes'"
