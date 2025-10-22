#!/bin/bash
# API Key Configuration Tool for 254Carbon Commodity Platform
# Automates the process of configuring API keys for data sources
#
# Usage:
#   Interactive mode:  ./configure-api-keys.sh
#   Non-interactive:   FRED_API_KEY=xxx EIA_API_KEY=yyy ./configure-api-keys.sh --non-interactive
#   From file:         ./configure-api-keys.sh --from-file api-keys.env

set -e

NAMESPACE="data-platform"
SECRET_NAME="seatunnel-api-keys"
NON_INTERACTIVE=false
FROM_FILE=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        --from-file)
            FROM_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --non-interactive    Use environment variables instead of prompts"
            echo "  --from-file FILE     Load API keys from file"
            echo "  --help               Show this help message"
            echo ""
            echo "Interactive mode example:"
            echo "  $0"
            echo ""
            echo "Non-interactive mode example:"
            echo "  FRED_API_KEY=xxx EIA_API_KEY=yyy $0 --non-interactive"
            echo ""
            echo "From file example:"
            echo "  $0 --from-file /path/to/api-keys.env"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  254Carbon API Key Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check if secret exists
if ! kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" &> /dev/null; then
    echo -e "${RED}Error: Secret $SECRET_NAME not found in namespace $NAMESPACE${NC}"
    echo "Please ensure the platform is deployed first."
    exit 1
fi

# Load from file if specified
if [[ -n "$FROM_FILE" ]]; then
    if [[ ! -f "$FROM_FILE" ]]; then
        echo -e "${RED}Error: File not found: $FROM_FILE${NC}"
        exit 1
    fi
    echo -e "${GREEN}Loading API keys from: $FROM_FILE${NC}"
    source "$FROM_FILE"
    NON_INTERACTIVE=true
fi

# Function to validate API key format (basic check)
validate_key() {
    local key=$1
    local name=$2
    
    if [[ -z "$key" ]] || [[ "$key" == "your-"*"-api-key"* ]] || [[ "$key" == "not-required" ]]; then
        return 1
    fi
    
    # Basic length check (most API keys are at least 20 chars)
    if [[ ${#key} -lt 10 ]]; then
        echo -e "${YELLOW}Warning: $name seems too short (${#key} characters)${NC}"
        return 1
    fi
    
    return 0
}

# Function to prompt for API key
prompt_for_key() {
    local var_name=$1
    local display_name=$2
    local help_url=$3
    local optional=$4
    
    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        eval "key=\$$var_name"
        if [[ -z "$key" ]] && [[ "$optional" != "true" ]]; then
            echo -e "${YELLOW}Warning: $var_name not set${NC}"
        fi
        return
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}$display_name${NC}"
    if [[ -n "$help_url" ]]; then
        echo -e "Get your key: ${YELLOW}$help_url${NC}"
    fi
    if [[ "$optional" == "true" ]]; then
        echo -e "${YELLOW}(Optional - press Enter to skip)${NC}"
    fi
    echo ""
    
    read -p "Enter $display_name: " key
    
    if [[ "$optional" == "true" ]] && [[ -z "$key" ]]; then
        key="not-configured"
    fi
    
    eval "$var_name=\"$key\""
}

# Collect API keys
echo -e "${BLUE}Required API Keys:${NC}"
echo ""

prompt_for_key "FRED_API_KEY" "FRED API Key" "https://fred.stlouisfed.org/docs/api/api_key.html" "false"
prompt_for_key "EIA_API_KEY" "EIA API Key" "https://www.eia.gov/opendata/" "false"

echo ""
echo -e "${BLUE}Optional API Keys:${NC}"
echo ""

prompt_for_key "NOAA_API_KEY" "NOAA API Key" "https://www.ncdc.noaa.gov/cdo-web/token" "true"
prompt_for_key "WORLD_BANK_API_KEY" "World Bank API Key" "" "true"
prompt_for_key "WEATHER_API_KEY" "Weather API Key" "https://www.weatherapi.com/" "true"
prompt_for_key "ICE_API_KEY" "ICE Data API Key" "https://www.ice.com/" "true"
prompt_for_key "API_KEY" "Market Data API Key" "" "true"

# Validate required keys
echo ""
echo -e "${BLUE}Validating API keys...${NC}"

VALID_KEYS=0
INVALID_KEYS=0

if validate_key "$FRED_API_KEY" "FRED_API_KEY"; then
    echo -e "${GREEN}✓ FRED_API_KEY valid${NC}"
    ((VALID_KEYS++))
else
    echo -e "${RED}✗ FRED_API_KEY invalid or missing${NC}"
    ((INVALID_KEYS++))
fi

if validate_key "$EIA_API_KEY" "EIA_API_KEY"; then
    echo -e "${GREEN}✓ EIA_API_KEY valid${NC}"
    ((VALID_KEYS++))
else
    echo -e "${RED}✗ EIA_API_KEY invalid or missing${NC}"
    ((INVALID_KEYS++))
fi

# Validate optional keys (just check if provided)
[[ -n "$NOAA_API_KEY" ]] && [[ "$NOAA_API_KEY" != "not-configured" ]] && echo -e "${GREEN}✓ NOAA_API_KEY provided${NC}"
[[ -n "$WORLD_BANK_API_KEY" ]] && [[ "$WORLD_BANK_API_KEY" != "not-configured" ]] && echo -e "${GREEN}✓ WORLD_BANK_API_KEY provided${NC}"
[[ -n "$WEATHER_API_KEY" ]] && [[ "$WEATHER_API_KEY" != "not-configured" ]] && echo -e "${GREEN}✓ WEATHER_API_KEY provided${NC}"
[[ -n "$ICE_API_KEY" ]] && [[ "$ICE_API_KEY" != "not-configured" ]] && echo -e "${GREEN}✓ ICE_API_KEY provided${NC}"
[[ -n "$API_KEY" ]] && [[ "$API_KEY" != "not-configured" ]] && echo -e "${GREEN}✓ API_KEY provided${NC}"

echo ""
if [[ $INVALID_KEYS -gt 0 ]]; then
    echo -e "${RED}Warning: $INVALID_KEYS required API key(s) are invalid or missing${NC}"
    echo -e "${YELLOW}The platform will work with limited functionality${NC}"
    echo ""
    
    if [[ "$NON_INTERACTIVE" != "true" ]]; then
        read -p "Continue anyway? (y/N): " confirm
        if [[ "$confirm" != "y" ]] && [[ "$confirm" != "Y" ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
fi

# Update the secret
echo ""
echo -e "${BLUE}Updating Kubernetes secret...${NC}"

# Get existing MinIO credentials from secret
MINIO_ACCESS_KEY=$(kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath='{.data.MINIO_ACCESS_KEY}' 2>/dev/null | base64 -d || echo "minioadmin")
MINIO_SECRET_KEY=$(kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath='{.data.MINIO_SECRET_KEY}' 2>/dev/null | base64 -d || echo "minioadmin")

# Create temporary file with all keys
TEMP_FILE=$(mktemp)
cat > "$TEMP_FILE" <<EOF
FRED_API_KEY=${FRED_API_KEY:-not-configured}
EIA_API_KEY=${EIA_API_KEY:-not-configured}
NOAA_API_KEY=${NOAA_API_KEY:-not-configured}
WORLD_BANK_API_KEY=${WORLD_BANK_API_KEY:-not-configured}
WEATHER_API_KEY=${WEATHER_API_KEY:-not-configured}
ICE_API_KEY=${ICE_API_KEY:-not-configured}
API_KEY=${API_KEY:-not-configured}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
EOF

# Update secret
kubectl create secret generic "$SECRET_NAME" \
    --from-env-file="$TEMP_FILE" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -

rm -f "$TEMP_FILE"

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Secret updated successfully${NC}"
else
    echo -e "${RED}✗ Failed to update secret${NC}"
    exit 1
fi

# Restart SeaTunnel pods to pick up new keys
echo ""
echo -e "${BLUE}Restarting SeaTunnel pods to apply changes...${NC}"
kubectl rollout restart deployment -n "$NAMESPACE" -l app=seatunnel 2>/dev/null || true
kubectl delete pods -n "$NAMESPACE" -l app=seatunnel 2>/dev/null || echo "No SeaTunnel pods to restart"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  API Keys Configured Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Valid required keys: $VALID_KEYS / 2"
echo "  - Secret updated: $SECRET_NAME"
echo "  - Namespace: $NAMESPACE"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Import DolphinScheduler workflows: ./scripts/import-dolphinscheduler-workflows.py"
echo "  2. Import Superset dashboards: ./scripts/import-superset-dashboards.py"
echo "  3. Verify platform: ./scripts/verify-platform-complete.sh"
echo ""
echo "Or run the all-in-one setup:"
echo "  ./scripts/setup-commodity-platform.sh"
echo ""


