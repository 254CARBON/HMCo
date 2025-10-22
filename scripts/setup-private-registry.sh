#!/bin/bash
#
# Setup Private Container Registry for 254Carbon Platform
# Resolves Docker Hub rate limiting issues
#
# Supports:
#   - Harbor (self-hosted, recommended)
#   - AWS ECR (cloud-based)
#   - GCP GCR (cloud-based)
#   - Azure ACR (cloud-based)
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
REGISTRY_TYPE="${1:-harbor}"
NAMESPACE="registry"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}254Carbon Private Registry Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Show usage
if [[ "$REGISTRY_TYPE" == "-h" ]] || [[ "$REGISTRY_TYPE" == "--help" ]]; then
    cat << 'EOF'
Usage: ./setup-private-registry.sh [REGISTRY_TYPE]

Supported Registry Types:
  harbor    - Self-hosted Harbor registry (recommended)
  ecr       - AWS Elastic Container Registry
  gcr       - Google Container Registry
  acr       - Azure Container Registry

Examples:
  ./setup-private-registry.sh harbor
  ./setup-private-registry.sh ecr
  ./setup-private-registry.sh gcr
  ./setup-private-registry.sh acr

For detailed instructions, see docs/readiness/production-readiness.md
EOF
    exit 0
fi

# Harbor Setup
setup_harbor() {
    echo -e "${YELLOW}Setting up Harbor Registry...${NC}"
    echo ""
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Add Harbor Helm repo
    helm repo add harbor https://helm.goharbor.io
    helm repo update
    
    echo -e "${YELLOW}Harbor Helm chart added.${NC}"
    echo ""
    
    # Create values file template
    cat > /tmp/harbor-values.yaml << 'HARBOR_CONFIG'
# Harbor minimal configuration for 254Carbon
expose:
  type: clusterIP
  ingress:
    hosts:
      core: harbor.254carbon.local
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: "letsencrypt-prod"
    tls:
      enabled: true
      certSource: cert-manager
externalURL: https://harbor.254carbon.local

harborAdminPassword: "ChangeMe123!"
secretKey: "not-a-secure-key"

persistence:
  enabled: true
  storageClass: default
  
registry:
  storage:
    s3:
      enabled: false

database:
  type: internal
  
redis:
  type: internal

metrics:
  enabled: false
HARBOR_CONFIG
    
    echo -e "${YELLOW}Harbor values template created at /tmp/harbor-values.yaml${NC}"
    echo -e "${YELLOW}Edit this file before deploying: nano /tmp/harbor-values.yaml${NC}"
    echo ""
    echo -e "${BLUE}To deploy Harbor:${NC}"
    echo "  helm install harbor harbor/harbor -n $NAMESPACE --values /tmp/harbor-values.yaml"
    echo ""
    echo -e "${YELLOW}After Harbor is running:${NC}"
    echo "  1. Access at https://harbor.254carbon.local"
    echo "  2. Login with admin / ChangeMe123!"
    echo "  3. Create project for 254carbon"
    echo "  4. Configure docker credentials"
    echo ""
}

# AWS ECR Setup
setup_ecr() {
    echo -e "${YELLOW}Setting up AWS ECR...${NC}"
    echo ""
    
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}AWS CLI not found. Install it first:${NC}"
        echo "  pip install awscli"
        exit 1
    fi
    
    REGISTRY_NAME="254carbon"
    
    # Create ECR repositories
    aws ecr create-repository \
        --repository-name "$REGISTRY_NAME" \
        --region us-east-1 || echo "Repository may already exist"
    
    # Get login token
    aws ecr get-login-password --region us-east-1 | docker login \
        --username AWS \
        --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
    
    echo -e "${GREEN}✓ ECR configured${NC}"
    echo ""
    echo "Repository: $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/254carbon"
    echo ""
    
    # Create Kubernetes image pull secret
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret docker-registry ecr-credentials \
        -n data-platform \
        --docker-server="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com" \
        --docker-username=AWS \
        --docker-password="$(aws ecr get-login-password --region us-east-1)" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo -e "${GREEN}✓ Kubernetes image pull secret created${NC}"
}

# GCP GCR Setup
setup_gcr() {
    echo -e "${YELLOW}Setting up Google Container Registry...${NC}"
    echo ""
    
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}Google Cloud SDK not found. Install it first${NC}"
        exit 1
    fi
    
    PROJECT_ID=$(gcloud config get-value project)
    REGISTRY="gcr.io/$PROJECT_ID"
    
    # Enable Container Registry API
    gcloud services enable containerregistry.googleapis.com
    
    # Configure docker authentication
    gcloud auth configure-docker
    
    echo -e "${GREEN}✓ GCR configured${NC}"
    echo ""
    echo "Registry: $REGISTRY"
    echo ""
    
    # Create Kubernetes image pull secret
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret docker-registry gcr-credentials \
        -n data-platform \
        --docker-server=gcr.io \
        --docker-username=_json_key \
        --docker-password="$(cat ~/.config/gcloud/application_default_credentials.json)" \
        --docker-email=user@example.com \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo -e "${GREEN}✓ Kubernetes image pull secret created${NC}"
}

# Azure ACR Setup
setup_acr() {
    echo -e "${YELLOW}Setting up Azure Container Registry...${NC}"
    echo ""
    
    if ! command -v az &> /dev/null; then
        echo -e "${RED}Azure CLI not found. Install it first${NC}"
        exit 1
    fi
    
    RESOURCE_GROUP="254carbon"
    REGISTRY_NAME="254carbon"
    LOCATION="eastus"
    
    # Create resource group
    az group create --name $RESOURCE_GROUP --location $LOCATION || true
    
    # Create ACR
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $REGISTRY_NAME \
        --sku Basic || true
    
    LOGIN_SERVER="$REGISTRY_NAME.azurecr.io"
    
    # Get credentials
    az acr login --name $REGISTRY_NAME
    
    echo -e "${GREEN}✓ ACR configured${NC}"
    echo ""
    echo "Registry: $LOGIN_SERVER"
    echo ""
    
    # Create Kubernetes image pull secret
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    CREDENTIALS=$(az acr credential show --name $REGISTRY_NAME)
    USERNAME=$(echo $CREDENTIALS | jq -r '.username')
    PASSWORD=$(echo $CREDENTIALS | jq -r '.passwords[0].value')
    
    kubectl create secret docker-registry acr-credentials \
        -n data-platform \
        --docker-server=$LOGIN_SERVER \
        --docker-username=$USERNAME \
        --docker-password=$PASSWORD \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo -e "${GREEN}✓ Kubernetes image pull secret created${NC}"
}

# Main execution
case "$REGISTRY_TYPE" in
    harbor)
        setup_harbor
        ;;
    ecr)
        setup_ecr
        ;;
    gcr)
        setup_gcr
        ;;
    acr)
        setup_acr
        ;;
    *)
        echo -e "${RED}Unknown registry type: $REGISTRY_TYPE${NC}"
        echo "Supported types: harbor, ecr, gcr, acr"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Registry setup complete${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Verify registry connectivity"
echo "2. Run: ./scripts/mirror-images.sh"
echo "3. Update deployments with new registry URLs"
echo ""
