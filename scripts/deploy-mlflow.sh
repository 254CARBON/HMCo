#!/bin/bash
# Deploy MLflow to 254Carbon Data Platform
# Complete deployment automation script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLFLOW_DIR="$(dirname "$SCRIPT_DIR")/k8s/compute/mlflow"
NAMESPACE="data-platform"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
cat << 'EOF'
  __  __ _     _____ _                 
 |  \/  | |   |  ___| | _____      __ 
 | |\/| | |   | |_  | |/ _ \ \ /\ / / 
 | |  | | |___|  _| | | (_) \ V  V /  
 |_|  |_|_____|_|   |_|\___/ \_/\_/   
                                       
     Model Management Deployment
EOF
echo -e "${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MLflow Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl found${NC}"

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Kubernetes cluster accessible${NC}"

if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${RED}✗ Namespace '$NAMESPACE' not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Namespace exists${NC}"

echo ""
echo -e "${GREEN}All prerequisites met!${NC}"
echo ""

# Deployment steps
echo -e "${BLUE}Deployment Steps:${NC}"
echo "  1. Initialize PostgreSQL database"
echo "  2. Create MinIO bucket"
echo "  3. Deploy MLflow secrets and config"
echo "  4. Deploy MLflow tracking server"
echo "  5. Deploy ingress and monitoring"
echo "  6. Verify deployment"
echo ""
echo -e "${YELLOW}Estimated time: ~5 minutes${NC}"
echo ""

read -p "Continue? (Y/n): " confirm
if [[ "$confirm" == "n" ]] || [[ "$confirm" == "N" ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""

# Step 1: Initialize PostgreSQL
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 1: Initialize PostgreSQL${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

bash "$SCRIPT_DIR/init-mlflow-postgres.sh"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ PostgreSQL initialization failed${NC}"
    exit 1
fi

echo ""

# Step 2: Create MinIO bucket
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 2: Create MinIO Bucket${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

bash "$SCRIPT_DIR/create-mlflow-minio-bucket.sh"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ MinIO bucket creation failed${NC}"
    exit 1
fi

echo ""

# Step 3: Deploy secrets and config
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 3: Deploy Secrets & Config${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Applying secrets...${NC}"
kubectl apply -f "$MLFLOW_DIR/mlflow-secrets.yaml"
echo -e "${GREEN}✓ Secrets applied${NC}"

echo -e "${BLUE}Applying configuration...${NC}"
kubectl apply -f "$MLFLOW_DIR/mlflow-configmap.yaml"
echo -e "${GREEN}✓ Configuration applied${NC}"

echo ""

# Step 4: Deploy MLflow
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 4: Deploy MLflow Tracking Server${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Applying service...${NC}"
kubectl apply -f "$MLFLOW_DIR/mlflow-service.yaml"
echo -e "${GREEN}✓ Service applied${NC}"

echo -e "${BLUE}Applying deployment...${NC}"
kubectl apply -f "$MLFLOW_DIR/mlflow-deployment.yaml"
echo -e "${GREEN}✓ Deployment applied${NC}"

echo ""
echo -e "${BLUE}Waiting for MLflow pods to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=mlflow -n "$NAMESPACE" --timeout=300s 2>/dev/null || {
    echo -e "${YELLOW}⚠ Pods not ready yet, continuing anyway...${NC}"
}

READY_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=mlflow --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
echo -e "${GREEN}✓ $READY_PODS MLflow pod(s) running${NC}"

echo ""

# Step 5: Deploy ingress and monitoring
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 5: Deploy Ingress & Monitoring${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Applying ingress...${NC}"
kubectl apply -f "$MLFLOW_DIR/mlflow-ingress.yaml"
echo -e "${GREEN}✓ Ingress applied${NC}"

echo -e "${BLUE}Applying ServiceMonitor...${NC}"
kubectl apply -f "$MLFLOW_DIR/mlflow-servicemonitor.yaml" 2>/dev/null || echo -e "${YELLOW}⚠ ServiceMonitor not applied (prometheus-operator may not be installed)${NC}"

echo ""

# Step 6: Verify deployment
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 6: Verify Deployment${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check pods
echo -e "${BLUE}Checking pods...${NC}"
kubectl get pods -n "$NAMESPACE" -l app=mlflow

# Check service
echo ""
echo -e "${BLUE}Checking service...${NC}"
kubectl get svc -n "$NAMESPACE" mlflow

# Check ingress
echo ""
echo -e "${BLUE}Checking ingress...${NC}"
kubectl get ingress -n "$NAMESPACE" mlflow-ingress

# Test health endpoint
echo ""
echo -e "${BLUE}Testing health endpoint...${NC}"
MLFLOW_POD=$(kubectl get pod -n "$NAMESPACE" -l app=mlflow -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$MLFLOW_POD" ]; then
    kubectl exec -n "$NAMESPACE" "$MLFLOW_POD" -- curl -s http://localhost:5000/health > /dev/null 2>&1 && \
        echo -e "${GREEN}✓ Health check passed${NC}" || \
        echo -e "${YELLOW}⚠ Health check failed (server may still be starting)${NC}"
fi

echo ""
echo -e "${CYAN}"
cat << 'EOF'
  ____                 _                                  _   
 |  _ \  ___ _ __  ___| | ___  _   _ _ __ ___   ___ _ __ | |_ 
 | | | |/ _ \ '_ \/ __| |/ _ \| | | | '_ ` _ \ / _ \ '_ \| __|
 | |_| |  __/ |_) \__ \ | (_) | |_| | | | | | |  __/ | | | |_ 
 |____/ \___| .__/|___/_|\___/ \__, |_| |_| |_|\___|_| |_|\__|
            |_|                |___/                           
   ____                      _      _       _ 
  / ___|___  _ __ ___  _ __ | | ___| |_ ___| |
 | |   / _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \ |
 | |__| (_) | | | | | | |_) | |  __/ ||  __/_|
  \____\___/|_| |_| |_| .__/|_|\___|\__\___(_)
                      |_|                      
EOF
echo -e "${NC}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MLflow Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Access MLflow:${NC}"
echo ""
echo -e "  ${CYAN}Internal URL:${NC}   http://mlflow.data-platform.svc.cluster.local:5000"
echo -e "  ${CYAN}External URL:${NC}   https://mlflow.254carbon.com"
echo -e "  ${CYAN}Health Check:${NC}  https://mlflow.254carbon.com/health"
echo ""
echo -e "${BLUE}Connection Details:${NC}"
echo ""
echo -e "  ${CYAN}Tracking URI:${NC}   http://mlflow.data-platform.svc.cluster.local:5000"
echo -e "  ${CYAN}Backend:${NC}        PostgreSQL (postgres-shared-service)"
echo -e "  ${CYAN}Artifacts:${NC}      MinIO S3 (minio-service/mlflow-artifacts)"
echo ""
echo -e "${BLUE}Python Usage:${NC}"
echo ""
echo -e "  ${YELLOW}import mlflow${NC}"
echo -e "  ${YELLOW}mlflow.set_tracking_uri('http://mlflow.data-platform.svc.cluster.local:5000')${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Test MLflow tracking: python3 examples/simple_tracking.py"
echo "  2. Integrate with DolphinScheduler workflows"
echo "  3. View experiments at https://mlflow.254carbon.com"
echo "  4. Check monitoring in Grafana"
echo ""
echo -e "${YELLOW}Note:${NC} If using Cloudflare tunnel, ensure DNS record is configured"
echo ""

exit 0


