#!/bin/bash
# Deploy Automation Scripts ConfigMap
# Updates the ConfigMap with the latest automation scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="data-platform"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Deploying automation scripts ConfigMap...${NC}"

kubectl create configmap automation-scripts \
  --from-file=import-dolphinscheduler-workflows.py="${SCRIPT_DIR}/import-dolphinscheduler-workflows.py" \
  --from-file=import-superset-dashboards.py="${SCRIPT_DIR}/import-superset-dashboards.py" \
  --namespace="${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ConfigMap deployed successfully${NC}"
    echo ""
    echo "The following scripts are now available in the ConfigMap:"
    echo "  - import-dolphinscheduler-workflows.py"
    echo "  - import-superset-dashboards.py"
    echo ""
    echo "To use in Kubernetes Jobs:"
    echo "  kubectl apply -f k8s/dolphinscheduler/workflow-import-job.yaml"
    echo "  kubectl apply -f k8s/visualization/dashboard-import-job.yaml"
else
    echo "Failed to deploy ConfigMap"
    exit 1
fi


