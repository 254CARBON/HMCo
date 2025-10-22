#!/bin/bash
# Create MinIO Bucket for MLflow Artifacts
# Creates mlflow-artifacts bucket with versioning enabled

set -e

NAMESPACE="data-platform"
BUCKET_NAME="mlflow-artifacts"
MINIO_POD=""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MLflow MinIO Bucket Creation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Find MinIO pod
echo -e "${BLUE}Finding MinIO pod...${NC}"
MINIO_POD=$(kubectl get pods -n "$NAMESPACE" -l app=minio --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$MINIO_POD" ]; then
    echo -e "${RED}✗ MinIO pod not found${NC}"
    echo "Please ensure MinIO is running in the $NAMESPACE namespace"
    exit 1
fi

echo -e "${GREEN}✓ Found MinIO pod: $MINIO_POD${NC}"
echo ""

# Check if bucket exists
echo -e "${BLUE}Checking if bucket exists...${NC}"
BUCKET_EXISTS=$(kubectl exec -n "$NAMESPACE" "$MINIO_POD" -- \
    mc ls local/$BUCKET_NAME 2>/dev/null && echo "exists" || echo "not-exists")

if [[ "$BUCKET_EXISTS" == *"exists"* ]]; then
    echo -e "${YELLOW}⚠ Bucket '$BUCKET_NAME' already exists${NC}"
    echo ""
    echo -e "${GREEN}✓ Bucket is ready${NC}"
    exit 0
fi

echo -e "${BLUE}Creating bucket...${NC}"

# Configure mc alias (in case not configured)
kubectl exec -n "$NAMESPACE" "$MINIO_POD" -- \
    mc alias set local http://localhost:9000 minioadmin minioadmin > /dev/null 2>&1 || true

# Create bucket
kubectl exec -n "$NAMESPACE" "$MINIO_POD" -- \
    mc mb local/$BUCKET_NAME 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bucket created: $BUCKET_NAME${NC}"
else
    echo -e "${RED}✗ Failed to create bucket${NC}"
    exit 1
fi

# Enable versioning
echo -e "${BLUE}Enabling versioning...${NC}"
kubectl exec -n "$NAMESPACE" "$MINIO_POD" -- \
    mc version enable local/$BUCKET_NAME 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Versioning enabled${NC}"
else
    echo -e "${YELLOW}⚠ Could not enable versioning${NC}"
fi

# Set lifecycle policy (optional - keep last 30 versions)
echo -e "${BLUE}Setting lifecycle policy...${NC}"
kubectl exec -n "$MINIO_POD" -n "$NAMESPACE" -- sh -c "cat > /tmp/lifecycle.json <<'EOF'
{
  \"Rules\": [
    {
      \"ID\": \"ExpireOldVersions\",
      \"Status\": \"Enabled\",
      \"NoncurrentVersionExpiration\": {
        \"NoncurrentDays\": 30
      }
    }
  ]
}
EOF
mc ilm import local/$BUCKET_NAME < /tmp/lifecycle.json
" 2>&1 || echo -e "${YELLOW}⚠ Could not set lifecycle policy (optional)${NC}"

# Verify bucket
echo ""
echo -e "${BLUE}Verifying bucket...${NC}"
kubectl exec -n "$NAMESPACE" "$MINIO_POD" -- \
    mc ls local/$BUCKET_NAME > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bucket verified successfully${NC}"
else
    echo -e "${RED}✗ Bucket verification failed${NC}"
    exit 1
fi

# Show bucket info
echo ""
echo -e "${BLUE}Bucket Information:${NC}"
kubectl exec -n "$NAMESPACE" "$MINIO_POD" -- \
    mc stat local/$BUCKET_NAME 2>&1 | grep -E "Name|Date|Versioning" || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MinIO Bucket Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Bucket Details:"
echo "  - Bucket Name: $BUCKET_NAME"
echo "  - Endpoint: http://minio-service.data-platform.svc.cluster.local:9000"
echo "  - Versioning: Enabled"
echo "  - Lifecycle: 30 days retention for old versions"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Deploy MLflow: kubectl apply -f k8s/compute/mlflow/"
echo "  2. Verify deployment: kubectl get pods -n data-platform -l app=mlflow"
echo ""


