#!/bin/bash
# 07-deploy-platform.sh
# Deploy 254Carbon platform services in correct dependency order

set -e

echo "========================================"
echo "Deploying 254Carbon Platform"
echo "========================================"

# Configuration
REPO_DIR=${1:-"/home/m/tff/254CARBON/HMCo"}
DRY_RUN=${2:-"false"}

echo "Configuration:"
echo "  Repository: ${REPO_DIR}"
echo "  Dry Run: ${DRY_RUN}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
  echo "Error: kubectl not found. Is Kubernetes installed?"
  exit 1
fi

# Verify connectivity
echo "Step 1: Verify cluster connectivity"
kubectl cluster-info
kubectl get nodes
echo ""

# Apply command based on dry run
if [ "${DRY_RUN}" = "true" ]; then
  KUBECTL_CMD="kubectl apply -f --dry-run=client"
else
  KUBECTL_CMD="kubectl apply -f"
fi

echo "Step 2: Create namespaces"
${KUBECTL_CMD} "${REPO_DIR}/k8s/namespaces/"
kubectl wait --for=condition=Active namespace/data-platform --timeout=30s || true
kubectl wait --for=condition=Active namespace/monitoring --timeout=30s || true

echo ""
echo "Step 3: Deploy core infrastructure"
echo "  - RBAC"
${KUBECTL_CMD} "${REPO_DIR}/k8s/rbac/"

echo "  - Secrets and ConfigMaps"
if [ -d "${REPO_DIR}/k8s/secrets" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/secrets/"
fi

echo "  - Networking"
if [ -d "${REPO_DIR}/k8s/networking" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/networking/"
fi

echo ""
echo "Step 4: Deploy data platform services"
echo "  - Zookeeper"
if [ -f "${REPO_DIR}/k8s/shared/zookeeper/zookeeper.yaml" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/shared/zookeeper/"
  sleep 10
fi

echo "  - Kafka"
if [ -f "${REPO_DIR}/k8s/shared/kafka/kafka.yaml" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/shared/kafka/"
  sleep 10
fi

echo "  - MinIO"
if [ -d "${REPO_DIR}/k8s/data-lake/minio" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/data-lake/minio/"
  sleep 10
fi

echo "  - LakeFS"
if [ -d "${REPO_DIR}/k8s/data-lake/lakefs" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/data-lake/lakefs/"
  sleep 10
fi

echo "  - Iceberg REST Catalog"
if [ -d "${REPO_DIR}/k8s/data-lake/iceberg-rest-catalog" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/data-lake/iceberg-rest-catalog/"
  sleep 10
fi

echo "  - Trino"
if [ -d "${REPO_DIR}/k8s/compute/trino" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/compute/trino/"
  sleep 10
fi

echo "  - Spark"
if [ -d "${REPO_DIR}/k8s/compute/spark" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/compute/spark/"
  sleep 10
fi

echo "  - Doris"
if [ -f "${REPO_DIR}/doris-cluster-data-platform.yaml" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/doris-cluster-data-platform.yaml"
  sleep 15
fi

echo ""
echo "Step 5: Deploy supporting services"
echo "  - SeaTunnel"
if [ -d "${REPO_DIR}/k8s/seatunnel" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/seatunnel/"
  sleep 10
fi

echo "  - Monitoring"
if [ -d "${REPO_DIR}/k8s/monitoring" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/monitoring/"
  sleep 10
fi

echo "  - Vault"
if [ -d "${REPO_DIR}/k8s/vault" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/vault/"
  sleep 10
fi

echo "  - Visualization (Superset)"
if [ -d "${REPO_DIR}/k8s/visualization" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/visualization/"
  sleep 10
fi

echo "  - DataHub"
if [ -d "${REPO_DIR}/k8s/datahub" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/datahub/"
  sleep 10
fi

echo "  - Dolphin Scheduler"
if [ -d "${REPO_DIR}/k8s/dolphinscheduler" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/dolphinscheduler/"
  sleep 10
fi

echo ""
echo "Step 6: Deploy Cloudflare tunnel"
if [ -d "${REPO_DIR}/k8s/cloudflare" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/cloudflare/"
  sleep 10
fi

echo "  - Ingress"
if [ -d "${REPO_DIR}/k8s/ingress" ]; then
  ${KUBECTL_CMD} "${REPO_DIR}/k8s/ingress/"
fi

echo ""
echo "========================================"
echo "Platform deployment complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Verify all pods are running:"
echo "   kubectl get pods --all-namespaces"
echo ""
echo "2. Check service status:"
echo "   kubectl get svc --all-namespaces"
echo ""
echo "3. View pod logs:"
echo "   kubectl logs -f -n data-platform <pod-name>"
echo ""
echo "4. Run 08-validate-deployment.sh to verify"
