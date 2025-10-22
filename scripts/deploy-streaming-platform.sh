#!/bin/bash
# Deploy Real-time Streaming Platform
# Kafka Connect + Apache Flink + Apache Doris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="${REPO_DIR}/k8s"

echo "============================================"
echo "  254Carbon Streaming Platform Deployment"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function wait_for_pods() {
    local namespace=$1
    local label=$2
    local timeout=${3:-300}
    
    log_info "Waiting for pods with label $label in namespace $namespace..."
    kubectl wait --for=condition=ready pod \
        -l "$label" \
        -n "$namespace" \
        --timeout="${timeout}s" || {
        log_warn "Timeout waiting for pods. Continuing..."
    }
}

# Check prerequisites
log_info "Checking prerequisites..."
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubectl."
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster."
    exit 1
fi

log_info "Connected to cluster: $(kubectl config current-context)"
echo ""

# Phase 1: Scale Kafka Infrastructure
echo "============================================"
echo "Phase 1: Scale Kafka Infrastructure (3 brokers)"
echo "============================================"

log_info "Scaling Kafka to 3 brokers..."
kubectl apply -f "${K8S_DIR}/shared/kafka/kafka-production.yaml"

log_info "Waiting for Kafka brokers to be ready..."
sleep 30
wait_for_pods "data-platform" "app=kafka" 600

log_info "Verifying Kafka cluster..."
kubectl get pods -n data-platform -l app=kafka
echo ""

# Phase 2: Deploy Flink Operator
echo "============================================"
echo "Phase 2: Deploy Apache Flink Operator"
echo "============================================"

log_info "Creating Flink Operator namespace..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: flink-operator
  labels:
    app: flink-operator
EOF

log_info "Installing Flink CRDs..."
kubectl apply -f "${K8S_DIR}/streaming/flink/flink-crds.yaml"

log_info "Deploying Flink Operator..."
kubectl apply -f "${K8S_DIR}/streaming/flink/flink-operator.yaml"

log_info "Waiting for Flink Operator to be ready..."
wait_for_pods "flink-operator" "app=flink-operator" 300

log_info "Creating Flink RBAC in data-platform namespace..."
kubectl apply -f "${K8S_DIR}/streaming/flink/flink-rbac.yaml"

log_info "Flink Operator deployed successfully!"
kubectl get pods -n flink-operator
echo ""

# Phase 3: Deploy Kafka Connect
echo "============================================"
echo "Phase 3: Deploy Kafka Connect Cluster"
echo "============================================"

log_info "Deploying Kafka Connect workers..."
kubectl apply -f "${K8S_DIR}/streaming/kafka-connect/kafka-connect.yaml"

log_info "Waiting for Kafka Connect workers to be ready..."
wait_for_pods "data-platform" "app=kafka-connect" 300

log_info "Kafka Connect cluster deployed successfully!"
kubectl get pods -n data-platform -l app=kafka-connect
echo ""

# Phase 4: Deploy Apache Doris
echo "============================================"
echo "Phase 4: Deploy Apache Doris Cluster"
echo "============================================"

log_info "Deploying Doris Frontend (FE) nodes..."
kubectl apply -f "${K8S_DIR}/streaming/doris/doris-fe.yaml"

log_info "Waiting for Doris FE nodes to be ready..."
wait_for_pods "data-platform" "app=doris,component=fe" 600

log_info "Deploying Doris Backend (BE) nodes..."
kubectl apply -f "${K8S_DIR}/streaming/doris/doris-be.yaml"

log_info "Waiting for Doris BE nodes to be ready..."
wait_for_pods "data-platform" "app=doris,component=be" 600

log_info "Initializing Doris schema..."
kubectl apply -f "${K8S_DIR}/streaming/doris/doris-init.yaml"

log_info "Waiting for schema initialization..."
kubectl wait --for=condition=complete job/doris-init -n data-platform --timeout=300s || {
    log_warn "Schema init job did not complete. Check logs: kubectl logs -n data-platform job/doris-init"
}

log_info "Apache Doris cluster deployed successfully!"
kubectl get pods -n data-platform -l app=doris
echo ""

# Phase 5: Deploy Connectors
echo "============================================"
echo "Phase 5: Deploy Kafka Connect Connectors"
echo "============================================"

log_info "Deploying connector configurations..."
kubectl apply -f "${K8S_DIR}/streaming/connectors/"

log_info "Connectors will be registered via REST API..."
log_info "Use scripts/register-connectors.sh to register connectors"
echo ""

# Phase 6: Deploy Flink Applications
echo "============================================"
echo "Phase 6: Deploy Flink Streaming Applications"
echo "============================================"

log_info "Deploying Flink applications..."
kubectl apply -f "${K8S_DIR}/streaming/flink/flink-applications/data-enricher.yaml"
sleep 10
kubectl apply -f "${K8S_DIR}/streaming/flink/flink-applications/price-aggregator.yaml"
sleep 10
kubectl apply -f "${K8S_DIR}/streaming/flink/flink-applications/anomaly-detector.yaml"

log_info "Waiting for Flink jobs to start..."
sleep 30

log_info "Flink applications deployed successfully!"
kubectl get flinkdeployment -n data-platform
echo ""

# Phase 7: Deploy Monitoring
echo "============================================"
echo "Phase 7: Deploy Streaming Monitoring"
echo "============================================"

log_info "Deploying Service Monitors..."
kubectl apply -f "${K8S_DIR}/streaming/monitoring/streaming-servicemonitors.yaml"

log_info "Deploying Alert Rules..."
kubectl apply -f "${K8S_DIR}/streaming/monitoring/streaming-alerts.yaml"

log_info "Deploying Grafana Dashboards..."
kubectl apply -f "${K8S_DIR}/streaming/monitoring/grafana-dashboards.yaml"

log_info "Monitoring configured successfully!"
echo ""

# Phase 8: Deploy Use Cases
echo "============================================"
echo "Phase 8: Configure Real-time Commodity Monitoring"
echo "============================================"

log_info "Deploying commodity monitoring use case..."
kubectl apply -f "${K8S_DIR}/streaming/use-cases/realtime-commodity-monitoring.yaml"

log_info "Waiting for use case setup..."
kubectl wait --for=condition=complete job/setup-commodity-monitoring -n data-platform --timeout=300s || {
    log_warn "Use case setup did not complete. Check logs: kubectl logs -n data-platform job/setup-commodity-monitoring"
}

log_info "Real-time commodity monitoring configured!"
echo ""

# Verification
echo "============================================"
echo "Deployment Verification"
echo "============================================"

log_info "Verifying deployment status..."
echo ""

echo "Kafka Brokers:"
kubectl get pods -n data-platform -l app=kafka -o wide
echo ""

echo "Kafka Connect Workers:"
kubectl get pods -n data-platform -l app=kafka-connect -o wide
echo ""

echo "Flink Operator:"
kubectl get pods -n flink-operator -o wide
echo ""

echo "Flink Deployments:"
kubectl get flinkdeployment -n data-platform
echo ""

echo "Doris Frontend:"
kubectl get pods -n data-platform -l app=doris,component=fe -o wide
echo ""

echo "Doris Backend:"
kubectl get pods -n data-platform -l app=doris,component=be -o wide
echo ""

# Summary
echo "============================================"
echo "Deployment Summary"
echo "============================================"
echo ""
log_info "Streaming platform deployed successfully!"
echo ""
echo "Access Points:"
echo "  - Kafka Connect API: kubectl port-forward -n data-platform svc/kafka-connect-service 8083:8083"
echo "  - Flink UI: kubectl port-forward -n data-platform svc/<flink-jobmanager> 8081:8081"
echo "  - Doris Query (MySQL): kubectl port-forward -n data-platform svc/doris-fe-service 9030:9030"
echo "  - Doris Web UI: kubectl port-forward -n data-platform svc/doris-fe-service 8030:8030"
echo ""
echo "Next Steps:"
echo "  1. Register Kafka Connect connectors: ./scripts/register-connectors.sh"
echo "  2. Verify Flink jobs: kubectl get flinkdeployment -n data-platform"
echo "  3. Check Doris Routine Load: mysql -h 127.0.0.1 -P 9030 -uroot -e 'SHOW ROUTINE LOAD;'"
echo "  4. View monitoring: Access Grafana dashboards in monitoring namespace"
echo ""
log_info "Deployment complete! 🎉"


