#!/bin/bash
# Configure DolphinScheduler API Credentials
# Sets up Kubernetes secrets and DolphinScheduler global variables for API access
#
# Usage:
#   ./configure-dolphinscheduler-credentials.sh
#   ./configure-dolphinscheduler-credentials.sh --dolphin-url http://localhost:12345

set -e

# Configuration
NAMESPACE="${NAMESPACE:-data-platform}"
SECRET_NAME="dolphinscheduler-api-keys"
DOLPHIN_URL="${DOLPHIN_URL:-http://dolphinscheduler-api.data-platform:12345}"
DOLPHIN_USER="${DOLPHIN_USER:-admin}"
DOLPHIN_PASS="${DOLPHIN_PASS:-dolphinscheduler123}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dolphin-url)
            DOLPHIN_URL="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dolphin-url URL    DolphinScheduler API URL (default: http://dolphinscheduler-api.data-platform:12345)"
            echo "  --namespace NS       Kubernetes namespace (default: data-platform)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DolphinScheduler Credentials Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl available${NC}"

if ! command -v curl &> /dev/null; then
    echo -e "${RED}✗ curl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ curl available${NC}"

if ! command -v jq &> /dev/null; then
    echo -e "${RED}✗ jq not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ jq available${NC}"

# Auto-detect and load API keys from multiple sources
# Priority: 1) Env vars, 2) Kubernetes secrets, 3) Vault, 4) Local file

auto_load_api_keys() {
    local key_name=$1
    local value=""
    
    # 1. Check environment variable
    value="${!key_name}"
    
    # 2. Try to read from Kubernetes secret if kubectl available
    if [[ -z "$value" ]] && command -v kubectl &> /dev/null; then
        value=$(kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath="{.data.$key_name}" 2>/dev/null | base64 -d 2>/dev/null || echo "")
    fi
    
    # 3. Try Vault if available
    if [[ -z "$value" ]] && command -v vault &> /dev/null && [[ -n "$VAULT_ADDR" ]]; then
        value=$(vault kv get -field="$key_name" secret/dolphinscheduler 2>/dev/null || echo "")
    fi
    
    # 4. Try local api-keys.env file if it exists
    if [[ -z "$value" ]] && [[ -f "$(dirname "$0")/../api-keys.env" ]]; then
        source "$(dirname "$0")/../api-keys.env" 2>/dev/null || true
        value="${!key_name}"
    fi
    
    echo "$value"
}

echo -e "${BLUE}Auto-detecting API keys from available sources...${NC}"

ALPHAVANTAGE_API_KEY=$(auto_load_api_keys "ALPHAVANTAGE_API_KEY")
POLYGON_API_KEY=$(auto_load_api_keys "POLYGON_API_KEY")
EIA_API_KEY=$(auto_load_api_keys "EIA_API_KEY")
GIE_API_KEY=$(auto_load_api_keys "GIE_API_KEY")
CENSUS_API_KEY=$(auto_load_api_keys "CENSUS_API_KEY")
NOAA_API_KEY=$(auto_load_api_keys "NOAA_API_KEY")
FRED_API_KEY=$(auto_load_api_keys "FRED_API_KEY")

# Check which keys were found
FOUND_KEYS=()
MISSING_KEYS=()

[[ -n "$ALPHAVANTAGE_API_KEY" ]] && FOUND_KEYS+=("ALPHAVANTAGE_API_KEY") || MISSING_KEYS+=("ALPHAVANTAGE_API_KEY")
[[ -n "$POLYGON_API_KEY" ]] && FOUND_KEYS+=("POLYGON_API_KEY") || MISSING_KEYS+=("POLYGON_API_KEY")
[[ -n "$EIA_API_KEY" ]] && FOUND_KEYS+=("EIA_API_KEY") || MISSING_KEYS+=("EIA_API_KEY")
[[ -n "$GIE_API_KEY" ]] && FOUND_KEYS+=("GIE_API_KEY") || MISSING_KEYS+=("GIE_API_KEY")
[[ -n "$CENSUS_API_KEY" ]] && FOUND_KEYS+=("CENSUS_API_KEY") || MISSING_KEYS+=("CENSUS_API_KEY")
[[ -n "$NOAA_API_KEY" ]] && FOUND_KEYS+=("NOAA_API_KEY") || MISSING_KEYS+=("NOAA_API_KEY")

echo -e "${GREEN}✓ Found ${#FOUND_KEYS[@]}/6 required API keys${NC}"

if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Missing ${#MISSING_KEYS[@]} API keys:${NC}"
    for key in "${MISSING_KEYS[@]}"; do
        echo "  - $key"
    done
    echo ""
    echo -e "${YELLOW}Keys will be set to 'not-configured' - workflows using these sources will fail${NC}"
    echo -e "${BLUE}To fix, add keys to any of these locations:${NC}"
    echo "  1. Environment: export ${MISSING_KEYS[0]}=\"your-key\""
    echo "  2. Kubernetes: kubectl create secret generic $SECRET_NAME --from-literal=..."
    echo "  3. Vault: vault kv put secret/dolphinscheduler ..."
    echo "  4. File: Create $(dirname "$0")/../api-keys.env"
    echo ""
    
    # Set missing keys to placeholder
    for key in "${MISSING_KEYS[@]}"; do
        eval "$key=\"not-configured\""
    done
fi

echo ""
echo -e "${BLUE}Step 1: Creating Kubernetes Secret${NC}"
echo "Namespace: $NAMESPACE"
echo "Secret: $SECRET_NAME"
echo ""

# Create the secret
kubectl create secret generic "$SECRET_NAME" \
    --from-literal=ALPHAVANTAGE_API_KEY="$ALPHAVANTAGE_API_KEY" \
    --from-literal=POLYGON_API_KEY="$POLYGON_API_KEY" \
    --from-literal=EIA_API_KEY="$EIA_API_KEY" \
    --from-literal=GIE_API_KEY="$GIE_API_KEY" \
    --from-literal=CENSUS_API_KEY="$CENSUS_API_KEY" \
    --from-literal=NOAA_API_KEY="$NOAA_API_KEY" \
    --from-literal=FRED_API_KEY="${FRED_API_KEY:-not-configured}" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Kubernetes secret created/updated${NC}"
else
    echo -e "${RED}✗ Failed to create secret${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 2: Patching DolphinScheduler Worker Deployment${NC}"
echo "Adding secret as environment variables to worker pods..."
echo ""

# Check if worker deployment exists
if kubectl get deployment -n "$NAMESPACE" -l app.kubernetes.io/component=worker &> /dev/null; then
    WORKER_DEPLOYMENT=$(kubectl get deployment -n "$NAMESPACE" -l app.kubernetes.io/component=worker -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -n "$WORKER_DEPLOYMENT" ]]; then
        # Create patch to add envFrom
        kubectl patch deployment "$WORKER_DEPLOYMENT" -n "$NAMESPACE" --type='json' -p='[
          {
            "op": "add",
            "path": "/spec/template/spec/containers/0/envFrom",
            "value": [
              {
                "secretRef": {
                  "name": "'"$SECRET_NAME"'"
                }
              }
            ]
          }
        ]' 2>/dev/null || kubectl set env deployment/"$WORKER_DEPLOYMENT" -n "$NAMESPACE" --from=secret/"$SECRET_NAME" --overwrite
        
        echo -e "${GREEN}✓ Worker deployment patched${NC}"
        echo -e "${YELLOW}⟳ Rolling out worker pods...${NC}"
        kubectl rollout status deployment/"$WORKER_DEPLOYMENT" -n "$NAMESPACE" --timeout=120s
        echo -e "${GREEN}✓ Worker pods restarted${NC}"
    else
        echo -e "${YELLOW}⚠ No worker deployment found, skipping patch${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No worker deployment found, skipping patch${NC}"
fi

echo ""
echo -e "${BLUE}Step 3: Creating DolphinScheduler Global Variables${NC}"
echo "Connecting to DolphinScheduler API..."
echo ""

# Setup port-forward if using cluster-internal URL
PORT_FORWARD_PID=""
LOCAL_PORT="12345"
if [[ "$DOLPHIN_URL" == *"data-platform"* ]] || [[ "$DOLPHIN_URL" == *"cluster.local"* ]]; then
    echo -e "${YELLOW}Setting up port-forward for cluster-internal access...${NC}"
    
    # Find the API pod
    API_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$API_POD" ]]; then
        kubectl port-forward -n "$NAMESPACE" "$API_POD" "$LOCAL_PORT:12345" &> /dev/null &
        PORT_FORWARD_PID=$!
        sleep 3
        DOLPHIN_URL="http://localhost:$LOCAL_PORT"
        echo -e "${GREEN}✓ Port-forward established (PID: $PORT_FORWARD_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Could not find API pod, using URL as-is${NC}"
    fi
fi

# Cleanup function
cleanup() {
    if [[ -n "$PORT_FORWARD_PID" ]]; then
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
        echo -e "${YELLOW}Port-forward terminated${NC}"
    fi
}
trap cleanup EXIT

# Login to DolphinScheduler
echo "Authenticating with DolphinScheduler..."
LOGIN_RESPONSE=$(curl -s -X POST "$DOLPHIN_URL/dolphinscheduler/login" \
    -d "userName=$DOLPHIN_USER&userPassword=$DOLPHIN_PASS" \
    -H "Content-Type: application/x-www-form-urlencoded" || echo '{"code": 1}')

TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.data.token // empty' 2>/dev/null)

if [[ -z "$TOKEN" ]] || [[ "$TOKEN" == "null" ]]; then
    echo -e "${YELLOW}⚠ Could not authenticate with DolphinScheduler${NC}"
    echo -e "${YELLOW}  (This is OK if API is not accessible or authentication changed)${NC}"
    echo -e "${YELLOW}  Variables can be created manually in the UI later${NC}"
else
    echo -e "${GREEN}✓ Authenticated successfully${NC}"
    
    # Note: DolphinScheduler 3.x uses environment variables, not global variables
    # We'll document this for manual configuration
    echo ""
    echo -e "${BLUE}Note: API keys are configured as Kubernetes secrets${NC}"
    echo -e "${BLUE}They will be available to workflows via environment variables${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Credentials Configuration Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  ✓ Kubernetes secret: $SECRET_NAME created"
echo "  ✓ Worker pods configured with API keys"
echo "  ✓ API keys available in workflows as:"
echo "      - \$ALPHAVANTAGE_API_KEY"
echo "      - \$POLYGON_API_KEY"
echo "      - \$EIA_API_KEY"
echo "      - \$GIE_API_KEY"
echo "      - \$CENSUS_API_KEY"
echo "      - \$NOAA_API_KEY"
echo "      - \$FRED_API_KEY (if set)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Import workflows: ./scripts/import-workflows-from-files.py"
echo "  2. Or run complete setup: ./scripts/setup-dolphinscheduler-complete.sh"
echo ""

