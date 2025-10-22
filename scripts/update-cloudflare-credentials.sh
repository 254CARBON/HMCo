#!/bin/bash
#
# Update Cloudflare Tunnel Credentials
# Safely updates the tunnel credentials secret in Kubernetes
#
# Usage:
#   ./update-cloudflare-credentials.sh TUNNEL_ID ACCOUNT_TAG AUTH_TOKEN
#

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verify arguments
if [[ $# -ne 3 ]]; then
    echo -e "${RED}Usage: $0 TUNNEL_ID ACCOUNT_TAG AUTH_TOKEN${NC}"
    echo ""
    echo "Update Cloudflare Tunnel credentials in Kubernetes"
    echo ""
    echo "Arguments:"
    echo "  TUNNEL_ID       - Your tunnel UUID from Cloudflare dashboard"
    echo "  ACCOUNT_TAG     - Your account tag from credentials JSON"
    echo "  AUTH_TOKEN      - Base64-encoded auth token from credentials"
    echo ""
    echo "These values are obtained from:"
    echo "  Cloudflare Dashboard → Networks → Tunnels → 254carbon-cluster → Credentials"
    exit 1
fi

TUNNEL_ID="$1"
ACCOUNT_TAG="$2"
AUTH_TOKEN="$3"

# Validate inputs
if [[ -z "$TUNNEL_ID" ]] || [[ -z "$ACCOUNT_TAG" ]] || [[ -z "$AUTH_TOKEN" ]]; then
    echo -e "${RED}Error: All arguments are required${NC}"
    exit 1
fi

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Cloudflare Credentials Update${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Create JSON payload
CREDENTIALS_JSON=$(cat <<EOF
{
  "tunnel_id": "$TUNNEL_ID",
  "account_tag": "$ACCOUNT_TAG",
  "tunnel_name": "254carbon-cluster",
  "auth_token": "$AUTH_TOKEN",
  "tunnel_remote_config": true
}
EOF
)

# Base64 encode credentials
CREDENTIALS_B64=$(echo -n "$CREDENTIALS_JSON" | base64 | tr -d '\n')

echo -e "${YELLOW}Creating/updating secret...${NC}"

# Delete existing secret if it exists (to avoid conflicts)
if kubectl get secret cloudflare-tunnel-credentials -n cloudflare-tunnel &>/dev/null; then
    echo "Deleting existing secret..."
    kubectl delete secret cloudflare-tunnel-credentials -n cloudflare-tunnel
fi

# Create new secret with base64-encoded credentials
kubectl create secret generic cloudflare-tunnel-credentials \
    -n cloudflare-tunnel \
    --from-literal=credentials.json="$CREDENTIALS_JSON" \
    --dry-run=client \
    -o yaml | kubectl apply -f -

echo -e "${GREEN}✓ Secret created successfully${NC}"
echo ""

# Restart cloudflared to pick up new credentials
echo -e "${YELLOW}Restarting cloudflared deployment...${NC}"
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel

echo -e "${YELLOW}Waiting for rollout...${NC}"
kubectl rollout status deployment/cloudflared -n cloudflare-tunnel --timeout=2m

echo ""
echo -e "${GREEN}✓ Credentials updated and tunnel restarted${NC}"
echo ""

# Verify tunnel is connected
echo -e "${YELLOW}Verifying tunnel connection...${NC}"
sleep 5

POD=$(kubectl get pods -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "$POD" ]]; then
    echo -e "${YELLOW}Waiting for pod to be ready...${NC}"
    sleep 10
    POD=$(kubectl get pods -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
fi

if [[ -n "$POD" ]]; then
    echo -e "${YELLOW}Checking logs from pod: $POD${NC}"
    
    # Show last few logs to verify connection
    if kubectl logs -n cloudflare-tunnel "$POD" 2>/dev/null | grep -q "Connection"; then
        echo -e "${GREEN}✓ Tunnel appears to be connected${NC}"
    else
        echo -e "${YELLOW}Tunnel may still be initializing. Check logs:${NC}"
        echo "  kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -f"
    fi
else
    echo -e "${YELLOW}No pod ready yet. Check status:${NC}"
    echo "  kubectl get pods -n cloudflare-tunnel"
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Update Complete${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Next steps:"
echo "1. Verify tunnel is running: kubectl get pods -n cloudflare-tunnel"
echo "2. Check tunnel logs: kubectl logs -n cloudflare-tunnel -f"
echo "3. Test connectivity: curl -v https://grafana.254carbon.com"
