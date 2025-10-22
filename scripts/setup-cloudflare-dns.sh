#!/bin/bash
#
# Cloudflare DNS Setup Script
# Configures DNS records for 254carbon.com services
#
# Usage:
#   ./setup-cloudflare-dns.sh [-t TOKEN] [-z ZONE_ID]
#
# Environment Variables:
#   CLOUDFLARE_API_TOKEN - API token (or use -t flag)
#   CLOUDFLARE_ZONE_ID   - Zone ID for 254carbon.com (or use -z flag)
#

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="254carbon.com"
# Determine Tunnel endpoint from TUNNEL_ID if provided
TUNNEL_ID="${CLOUDFLARE_TUNNEL_ID:-${TUNNEL_ID:-}}"
if [[ -n "$TUNNEL_ID" ]]; then
  TUNNEL_ENDPOINT="${TUNNEL_ID}.cfargotunnel.com"
else
  TUNNEL_ENDPOINT="254carbon-cluster.cfargotunnel.com"
fi

# Services to expose with their subdomains
declare -A SERVICES=(
  ["portal"]="Portal - Entry Point"
  ["grafana"]="Grafana - Monitoring Dashboards"
  ["superset"]="Superset - Data Visualization"
  ["datahub"]="DataHub - Metadata Platform"
  ["trino"]="Trino - Query Engine"
  ["doris"]="Apache Doris - OLAP Database"
  ["vault"]="Vault - Secrets Management"
  ["minio"]="MinIO - Object Storage Console"
  ["dolphin"]="DolphinScheduler - Workflow Orchestration"
  ["lakefs"]="LakeFS - Data Versioning"
  ["mlflow"]="MLflow - ML Platform"
  ["spark-history"]="Spark History Server"
  ["harbor"]="Harbor - Registry + UI"
)

# Parse arguments
API_TOKEN="${CLOUDFLARE_API_TOKEN:-}"
ZONE_ID="${CLOUDFLARE_ZONE_ID:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--token)
            API_TOKEN="$2"
            shift 2
            ;;
        -z|--zone-id)
            ZONE_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-t TOKEN] [-z ZONE_ID]"
            echo ""
            echo "Configure DNS records for 254carbon.com via Cloudflare API"
            echo ""
            echo "Arguments:"
            echo "  -t, --token TOKEN        Cloudflare API token (or use CLOUDFLARE_API_TOKEN env var)"
            echo "  -z, --zone-id ZONE_ID    Zone ID for 254carbon.com (or use CLOUDFLARE_ZONE_ID env var)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -t 'HsmXB0pAPV7ejbWFrpQt148LoxksjQKxJGRn4J7N' -z 'zone123456'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$API_TOKEN" ]]; then
    echo -e "${RED}Error: Cloudflare API token not provided${NC}"
    echo "Set CLOUDFLARE_API_TOKEN environment variable or use -t flag"
    exit 1
fi

if [[ -z "$ZONE_ID" ]]; then
    echo -e "${YELLOW}Zone ID not provided. Attempting to fetch...${NC}"
    
    # Fetch zone ID if not provided
    ZONE_RESPONSE=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=${DOMAIN}" \
        -H "Authorization: Bearer ${API_TOKEN}" \
        -H "Content-Type: application/json")
    
    ZONE_ID=$(echo "$ZONE_RESPONSE" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [[ -z "$ZONE_ID" ]]; then
        echo -e "${RED}Error: Could not fetch zone ID for ${DOMAIN}${NC}"
        echo "Please check your API token and ensure 254carbon.com is configured in Cloudflare"
        exit 1
    fi
    
    echo -e "${GREEN}Found zone ID: ${ZONE_ID}${NC}"
fi

# Function to create or update DNS record
create_dns_record() {
    local subdomain=$1
    local description=$2
    local type="CNAME"
    local name="${subdomain}.${DOMAIN}"
    local content="${TUNNEL_ENDPOINT}"
    local ttl=1  # Auto TTL (Cloudflare)
    local proxied=true  # Orange cloud - proxied through Cloudflare
    
    echo -n "Configuring ${name}... "
    
    # Check if record already exists
    EXISTING=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?name=${name}" \
        -H "Authorization: Bearer ${API_TOKEN}" \
        -H "Content-Type: application/json" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [[ -n "$EXISTING" ]]; then
        # Update existing record
        RESPONSE=$(curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records/${EXISTING}" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"type\": \"${type}\",
                \"name\": \"${name}\",
                \"content\": \"${content}\",
                \"ttl\": ${ttl},
                \"proxied\": ${proxied}
            }")
    else
        # Create new record
        RESPONSE=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"type\": \"${type}\",
                \"name\": \"${name}\",
                \"content\": \"${content}\",
                \"ttl\": ${ttl},
                \"proxied\": ${proxied}
            }")
    fi
    
    # Check response
    if echo "$RESPONSE" | grep -q '"success":true'; then
        echo -e "${GREEN}✓${NC} (${description})"
    else
        ERROR_MSG=$(echo "$RESPONSE" | grep -o '"message":"[^"]*"' | head -1 | cut -d'"' -f4)
        echo -e "${RED}✗${NC} Failed: ${ERROR_MSG}"
        return 1
    fi
}

# Attempt to create a Page Rule redirect from apex → portal
create_apex_redirect_rule() {
  echo -n "Configuring apex redirect (254carbon.com → portal.254carbon.com)... "
  local payload_template='{
    "targets": [
      { "target": "url", "constraint": { "operator": "matches", "value": "__DOMAIN__/*" } }
    ],
    "actions": [
      { "id": "forwarding_url", "value": { "url": "https://portal.__DOMAIN__/\$1", "status_code": 301 } }
    ],
    "priority": 1,
    "status": "active"
  }'
  local payload
  payload=${payload_template//__DOMAIN__/$DOMAIN}
  local resp
  resp=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/pagerules" \
    -H "Authorization: Bearer ${API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$payload")
  if echo "$resp" | grep -q '"success":true'; then
    echo -e "${GREEN}✓${NC}"
  else
    local msg
    msg=$(echo "$resp" | grep -o '"message":"[^\"]*"' | head -1 | cut -d'"' -f4)
    echo -e "${YELLOW}⚠${NC} Could not create redirect via API (${msg}). Use Cloudflare Redirect Rule: ${DOMAIN}/* → https://portal.${DOMAIN}/\$1 (301)."
  fi
}

# Main execution
echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}Cloudflare DNS Configuration${NC}"
echo -e "${YELLOW}================================${NC}"
echo ""
echo "Domain: ${DOMAIN}"
echo "Tunnel Endpoint: ${TUNNEL_ENDPOINT}"
echo "Zone ID: ${ZONE_ID}"
echo ""
echo -e "${YELLOW}Creating DNS records:${NC}"
echo ""

# Track failures
FAILURES=0

# Create DNS records for each service
for subdomain in "${!SERVICES[@]}"; do
    if ! create_dns_record "$subdomain" "${SERVICES[$subdomain]}"; then
        ((FAILURES++))
    fi
done

# Ensure www alias exists
if ! create_dns_record "www" "WWW Alias"; then
  ((FAILURES++))
fi

# Try to configure apex redirect
create_apex_redirect_rule

# (Apex CNAME not created; redirect configured above. WWW already ensured.)

echo ""
echo -e "${YELLOW}================================${NC}"

if [[ $FAILURES -eq 0 ]]; then
    echo -e "${GREEN}All DNS records configured successfully!${NC}"
    echo ""
    echo "Services are now available at:"
    for subdomain in "${!SERVICES[@]}"; do
        echo "  https://${subdomain}.${DOMAIN}  - ${SERVICES[$subdomain]}"
    done
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Verify tunnel is deployed: kubectl get pods -n cloudflare-tunnel"
    echo "2. Check tunnel is connected: kubectl logs -n cloudflare-tunnel -f"
    echo "3. Test access: curl -v https://grafana.254carbon.com"
else
    echo -e "${RED}Failed to configure ${FAILURES} record(s)${NC}"
    echo "Please check your API token and permissions"
    exit 1
fi
