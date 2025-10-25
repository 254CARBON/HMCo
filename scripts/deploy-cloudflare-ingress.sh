#!/bin/bash

set -e

echo "========================================"
echo "254Carbon Cloudflare Ingress Deployment"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
  echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Apply Cloudflare tunnel configuration update
print_status "Applying Cloudflare tunnel configuration..."
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/cloudflare-tunnel-ingress.yaml
print_success "Cloudflare tunnel configuration updated"

# Step 2: Apply all ingress resources
print_status "Applying missing ingress resources..."

INGRESS_FILES=(
  "prometheus-ingress.yaml"
  "alertmanager-ingress.yaml"
  "victoria-metrics-ingress.yaml"
  "loki-ingress.yaml"
  "clickhouse-ingress.yaml"
  "katib-ingress.yaml"
  "kong-admin-ingress.yaml"
)

for file in "${INGRESS_FILES[@]}"; do
  kubectl apply -f "/home/m/tff/254CARBON/HMCo/k8s/ingress/$file"
  print_success "Applied $file"
done

echo ""
print_status "Verifying ingress resources..."
kubectl get ingress -A -o wide

# Step 3: Restart tunnel pods to pick up changes
print_status "Restarting Cloudflare tunnel pods..."
kubectl rollout restart deployment cloudflared -n cloudflare-tunnel
print_success "Tunnel pods restarted"

# Step 4: Wait for tunnel pods to be ready
print_status "Waiting for tunnel pods to be ready (30s)..."
sleep 30
kubectl get pods -n cloudflare-tunnel

echo ""
print_status "Verifying tunnel connectivity..."
kubectl logs -n cloudflare-tunnel deployment/cloudflared --tail=20

echo ""
print_success "Deployment complete!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Verify DNS records in Cloudflare dashboard"
echo "2. Test service access:"
echo "   - https://clickhouse.254carbon.com"
echo "   - https://prometheus.254carbon.com"
echo "   - https://alertmanager.254carbon.com"
echo "   - https://victoria.254carbon.com"
echo "   - https://loki.254carbon.com"
echo "   - https://katib.254carbon.com"
echo "   - https://kong.254carbon.com"
echo "3. Check ingress status: kubectl get ingress -A"
echo "4. View tunnel logs: kubectl logs -f -n cloudflare-tunnel deployment/cloudflared"
echo ""
