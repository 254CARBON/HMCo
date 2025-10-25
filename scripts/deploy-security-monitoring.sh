#!/bin/bash

set -e

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║   254Carbon Security & Monitoring Deployment Script                  ║"
echo "║   Cloudflare Access + WAF + Alerts                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

print_error() {
  echo -e "${RED}[✗]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
  print_error "kubectl not found in PATH"
  exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
  print_error "Cannot connect to Kubernetes cluster"
  exit 1
fi

print_success "Prerequisites verified"
echo ""

# Phase 1: Update Ingress Resources with Cloudflare Access
print_status "PHASE 1: Updating Ingress Resources with Cloudflare Access..."
echo ""

print_status "Updating Vault ingress..."
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/vault-ingress.yaml
print_success "Vault ingress updated"

print_status "Updating Prometheus ingress..."
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/prometheus-ingress.yaml
print_success "Prometheus ingress updated"

print_status "Updating AlertManager ingress..."
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/alertmanager-ingress.yaml
print_success "AlertManager ingress updated"

echo ""

# Verify ingress resources
print_status "Verifying ingress resources..."
kubectl get ingress -A | grep -E "vault|prometheus|alertmanager" || print_warning "Some ingress resources not found"
echo ""

# Phase 2: Deploy Monitoring Configuration
print_status "PHASE 2: Deploying Monitoring Configuration..."
echo ""

print_status "Deploying AlertManager configuration..."
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/monitoring/alertmanager-config.yaml
print_success "AlertManager configuration deployed"

print_status "Deploying Prometheus alert rules..."
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/monitoring/prometheus-alert-rules.yaml
print_success "Prometheus alert rules deployed"

echo ""

# Phase 3: Gmail Configuration Instructions
print_status "PHASE 3: Email Alert Configuration..."
echo ""

echo -e "${YELLOW}MANUAL SETUP REQUIRED:${NC}"
echo ""
echo "1. Generate Gmail App Password:"
echo "   - Go to: https://myaccount.google.com/security/apppasswords"
echo "   - Select App: Mail"
echo "   - Select Device: Windows Computer (or your device)"
echo "   - Generate a 16-character app password"
echo ""
echo "2. Save the password and then run:"
echo ""
echo "   kubectl create secret generic alertmanager-email-secret \\"
echo "     --from-literal=password='YOUR_16_CHAR_APP_PASSWORD' \\"
echo "     -n monitoring \\"
echo "     --dry-run=client -o yaml | kubectl apply -f -"
echo ""
echo "3. Or update the alertmanager ConfigMap directly:"
echo ""
echo "   kubectl edit configmap alertmanager-config -n monitoring"
echo "   Replace \${GMAIL_APP_PASSWORD} with your actual app password"
echo ""

# Phase 4: Cloudflare Dashboard Setup Instructions
echo ""
print_status "PHASE 4: Cloudflare Dashboard Configuration..."
echo ""

echo -e "${YELLOW}MANUAL CLOUDFLARE SETUP REQUIRED:${NC}"
echo ""
echo "1. Create Cloudflare Access Applications:"
echo "   - Log in to: https://dash.cloudflare.com"
echo "   - Navigate to: Zero Trust → Access → Applications"
echo ""
echo "   For each service (vault, prometheus, alertmanager, kong):"
echo "     a) Click 'Create Application'"
echo "     b) Application name: <service>.254carbon.com"
echo "     c) Session duration: 24 hours"
echo "     d) Click 'Configure login rules'"
echo "     e) Add rule: 'Emails ending with @254carbon.com'"
echo "     f) Add rule: 'Emails ending with @project52.org'"
echo "     g) Action: Allow"
echo "     h) Click 'Save application'"
echo ""
echo "2. Enable WAF (Web Application Firewall):"
echo "   - Navigate to: Security → WAF → Managed Rules"
echo "   - Enable: Cloudflare Managed Ruleset"
echo "   - Set Sensitivity: Low (Conservative)"
echo ""
echo "3. Configure IP Whitelist:"
echo "   - Navigate to: Security → WAF → Tools → IP Access Rules"
echo "   - Add Rule:"
echo "     - IP: 192.168.1.0/24"
echo "     - Action: Allow"
echo "     - Priority: High"
echo ""
echo "4. Set Up Rate Limiting:"
echo "   - Navigate to: Security → Rate Limiting"
echo "   - Create Rule:"
echo "     - Path: /api/*"
echo "     - Threshold: 100 requests per 10 minutes"
echo "     - Action: Block for 15 minutes"
echo ""

# Phase 5: Test Configuration
echo ""
print_status "PHASE 5: Testing Configuration..."
echo ""

print_status "Checking ingress with auth headers..."
kubectl describe ingress prometheus-ingress -n monitoring | grep -A 2 "auth-url" && \
  print_success "Auth headers configured" || \
  print_warning "Auth headers not found - verify manual setup"

print_status "Checking AlertManager deployment..."
kubectl get pods -n monitoring -l app=alertmanager 2>/dev/null && \
  print_success "AlertManager pods running" || \
  print_warning "AlertManager not found"

print_status "Checking Prometheus deployment..."
kubectl get pods -n monitoring -l app=prometheus 2>/dev/null && \
  print_success "Prometheus pods running" || \
  print_warning "Prometheus not found"

echo ""

# Phase 6: Post-Deployment Instructions
echo ""
print_status "PHASE 6: Post-Deployment Setup..."
echo ""

echo -e "${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. Generate and save Gmail app password (see Phase 3 above)"
echo ""
echo "2. Update AlertManager with Gmail app password:"
echo "   - Edit ConfigMap with app password"
echo "   - Restart AlertManager pods:"
echo "   kubectl rollout restart deployment alertmanager -n monitoring"
echo ""
echo "3. Configure Cloudflare Access in dashboard (see Phase 4 above)"
echo ""
echo "4. Test Cloudflare Access:"
echo "   - Open: https://vault.254carbon.com"
echo "   - Should redirect to Cloudflare login"
echo "   - Login with @254carbon.com or @project52.org email"
echo ""
echo "5. Test AlertManager:"
echo "   - Wait for AlertManager to restart with password"
echo "   - Monitor logs: kubectl logs -f -n monitoring -l app=alertmanager"
echo ""
echo "6. Test alert delivery:"
echo "   - Wait ~5 minutes for alerts to stabilize"
echo "   - Check email: qagiw3@gmail.com"
echo ""
echo "7. Verify WAF is active:"
echo "   - Cloudflare Dashboard → Analytics → Security"
echo "   - Should see WAF events if attacks are occurring"
echo ""

# Phase 7: Verification Checklist
echo ""
print_success "Deployment Script Complete!"
echo ""
echo -e "${YELLOW}VERIFICATION CHECKLIST:${NC}"
echo ""
echo "  [ ] Gmail app password generated"
echo "  [ ] AlertManager ConfigMap updated with password"
echo "  [ ] AlertManager pods restarted"
echo "  [ ] Cloudflare Access applications created (4 total)"
echo "  [ ] WAF Managed Ruleset enabled"
echo "  [ ] IP whitelist configured (192.168.1.0/24)"
echo "  [ ] Rate limiting rules set"
echo "  [ ] Test login: vault.254carbon.com (redirects, login works)"
echo "  [ ] Test login: prometheus.254carbon.com (redirects, login works)"
echo "  [ ] Test login: alertmanager.254carbon.com (redirects, login works)"
echo "  [ ] Alert received at qagiw3@gmail.com"
echo "  [ ] WAF logs appearing in Cloudflare dashboard"
echo ""

echo "═════════════════════════════════════════════════════════════════════════════"
echo "                    Security & Monitoring Deployment Ready"
echo "═════════════════════════════════════════════════════════════════════════════"
echo ""
echo "For detailed instructions, see:"
echo "  📖 CLOUDFLARE_SECURITY_MONITORING_SETUP.md"
echo ""
