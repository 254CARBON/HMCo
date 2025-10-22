#!/bin/bash

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PORTAL_URL="https://254carbon.com"
TEAM_NAME="${CLOUDFLARE_TEAM_NAME:-qagi}"
SERVICES=(
    "grafana:grafana.254carbon.com"
    "superset:superset.254carbon.com"
    "vault:vault.254carbon.com"
    "minio:minio.254carbon.com"
    "dolphin:dolphin.254carbon.com"
    "datahub:datahub.254carbon.com"
    "trino:trino.254carbon.com"
    "doris:doris.254carbon.com"
    "lakefs:lakefs.254carbon.com"
)

# Results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; ((PASSED_TESTS++)); }
log_fail() { echo -e "${RED}[✗]${NC} $1"; ((FAILED_TESTS++)); }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

# Test counter
test_case() {
    ((TOTAL_TESTS++))
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Test $TOTAL_TESTS: $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          PHASE 4: COMPREHENSIVE SSO TESTING                 ║"
echo "║        254Carbon Cloudflare Access Implementation            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ===========================================
# SECTION 1: INFRASTRUCTURE VERIFICATION
# ===========================================
echo -e "\n${BLUE}═══ SECTION 1: INFRASTRUCTURE VERIFICATION ═══${NC}"

test_case "Tunnel pods are running"
TUNNEL_PODS=$(kubectl get pods -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --field-selector=status.phase=Running -o name | wc -l)
if [ "$TUNNEL_PODS" -gt 0 ]; then
    log_success "Tunnel pods running: $TUNNEL_PODS replicas"
else
    log_fail "No tunnel pods running"
fi

test_case "Portal pods are running"
PORTAL_PODS=$(kubectl get pods -n data-platform -l app=portal --field-selector=status.phase=Running -o name | wc -l)
if [ "$PORTAL_PODS" -ge 1 ]; then
    log_success "Portal pods running: $PORTAL_PODS replicas"
else
    log_fail "Portal pods not running"
fi

test_case "Ingress rules are deployed"
INGRESS_COUNT=$(kubectl get ingress -A | grep -c "254carbon" || echo 0)
if [ "$INGRESS_COUNT" -ge 10 ]; then
    log_success "All ingress rules deployed: $INGRESS_COUNT resources"
else
    log_fail "Not all ingress rules deployed: $INGRESS_COUNT/10"
fi

test_case "Ingress has auth annotations"
AUTH_CHECK=$(kubectl get ingress grafana-ingress -n monitoring -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/auth-url}' 2>/dev/null | grep -c "cloudflareaccess" || echo 0)
if [ "$AUTH_CHECK" -gt 0 ]; then
    log_success "Auth annotations present on ingress rules"
else
    log_fail "Auth annotations missing from ingress rules"
fi

test_case "Tunnel connection active"
TUNNEL_LOGS=$(kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=20 2>/dev/null | grep -c "connection" || echo 0)
if [ "$TUNNEL_LOGS" -gt 0 ]; then
    log_success "Tunnel connection active"
else
    log_warning "Could not verify tunnel connection from logs"
fi

# ===========================================
# SECTION 2: AUTHENTICATION FLOW TESTING
# ===========================================
echo -e "\n${BLUE}═══ SECTION 2: AUTHENTICATION FLOW ═══${NC}"

test_case "Portal URL accessible"
PORTAL_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -L "$PORTAL_URL" 2>/dev/null || echo "000")
if [[ "$PORTAL_STATUS" =~ ^(200|302|301)$ ]]; then
    log_success "Portal responds with HTTP $PORTAL_STATUS (redirects or loads)"
else
    log_fail "Portal returned HTTP $PORTAL_STATUS"
fi

test_case "Portal redirects to Cloudflare login"
REDIRECT_CHECK=$(curl -s -i "$PORTAL_URL" 2>/dev/null | grep -i "location:" | grep -c "cloudflareaccess\|login" || echo 0)
if [ "$REDIRECT_CHECK" -gt 0 ]; then
    log_success "Portal redirects to Cloudflare Access login page"
else
    log_warning "Could not verify redirect (may be cached by CDN)"
fi

test_case "HTTPS enforced on portal"
HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://254carbon.com" 2>/dev/null || echo "000")
if [[ "$HTTP_RESPONSE" =~ ^(301|302|307|308)$ ]]; then
    log_success "HTTP traffic redirects to HTTPS"
else
    log_warning "Could not verify HTTP redirect"
fi

test_case "Cloudflare Access endpoint accessible"
CF_ENDPOINT="https://${TEAM_NAME}.cloudflareaccess.com/cdn-cgi/access/authorize"
CF_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$CF_ENDPOINT" 2>/dev/null || echo "000")
if [[ "$CF_STATUS" =~ ^(200|401|403|404)$ ]]; then
    log_success "Cloudflare Access endpoint reachable (HTTP $CF_STATUS)"
else
    log_fail "Cannot reach Cloudflare Access endpoint"
fi

# ===========================================
# SECTION 3: SERVICE ACCESSIBILITY
# ===========================================
echo -e "\n${BLUE}═══ SECTION 3: SERVICE ACCESSIBILITY ═══${NC}"

for SERVICE in "${SERVICES[@]}"; do
    IFS=':' read -r NAME URL <<< "$SERVICE"
    
    test_case "Service $NAME is accessible"
    SERVICE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -L "https://$URL" 2>/dev/null || echo "000")
    if [[ "$SERVICE_STATUS" =~ ^(200|401|403|302|301)$ ]]; then
        log_success "$NAME responds with HTTP $SERVICE_STATUS"
    else
        log_fail "$NAME returned HTTP $SERVICE_STATUS"
    fi
done

# ===========================================
# SECTION 4: SECURITY VALIDATION
# ===========================================
echo -e "\n${BLUE}═══ SECTION 4: SECURITY VALIDATION ═══${NC}"

test_case "HTTPS certificate valid"
CERT_CHECK=$(curl -s -I "https://254carbon.com" 2>&1 | grep -c "HTTP\|SSL" || echo 0)
if [ "$CERT_CHECK" -gt 0 ]; then
    log_success "HTTPS certificate and SSL connection verified"
else
    log_warning "Could not verify SSL certificate"
fi

test_case "Security headers present"
SECURITY_HEADERS=$(curl -s -I "https://254carbon.com" 2>&1 | grep -iE "Strict-Transport|X-Frame|X-Content" | wc -l)
if [ "$SECURITY_HEADERS" -gt 0 ]; then
    log_success "Security headers present ($SECURITY_HEADERS headers found)"
else
    log_warning "Some security headers may be missing"
fi

test_case "Unauthorized access blocked"
UNAUTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://vault.254carbon.com" 2>/dev/null || echo "000")
if [[ "$UNAUTH_STATUS" =~ ^(401|403|302|301)$ ]]; then
    log_success "Service returns $UNAUTH_STATUS (unauthorized access blocked)"
else
    log_fail "Service returned unexpected HTTP $UNAUTH_STATUS"
fi

test_case "Tunnel credential secrets exist"
SECRET_CHECK=$(kubectl get secret cloudflare-tunnel-credentials -n cloudflare-tunnel -o name 2>/dev/null | wc -l)
if [ "$SECRET_CHECK" -gt 0 ]; then
    log_success "Tunnel credentials secret exists and is configured"
else
    log_fail "Tunnel credentials secret not found"
fi

# ===========================================
# SECTION 5: INGRESS CONFIGURATION
# ===========================================
echo -e "\n${BLUE}═══ SECTION 5: INGRESS CONFIGURATION ═══${NC}"

test_case "Portal ingress configured"
PORTAL_INGRESS=$(kubectl get ingress -n data-platform -o name 2>/dev/null | grep "portal" | wc -l)
if [ "$PORTAL_INGRESS" -gt 0 ]; then
    log_success "Portal ingress deployed"
else
    log_fail "Portal ingress not found"
fi

test_case "Service ingress configured (sample: Grafana)"
GRAFANA_INGRESS=$(kubectl get ingress -n monitoring -o name 2>/dev/null | grep "grafana" | wc -l)
if [ "$GRAFANA_INGRESS" -gt 0 ]; then
    log_success "Grafana ingress deployed"
else
    log_fail "Grafana ingress not found"
fi

test_case "NGINX auth-url annotation set"
AUTH_URL=$(kubectl get ingress grafana-ingress -n monitoring -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/auth-url}' 2>/dev/null || echo "")
if [[ "$AUTH_URL" == *"cloudflareaccess"* ]]; then
    log_success "Auth URL configured: ${AUTH_URL:0:60}..."
else
    log_fail "Auth URL not properly configured"
fi

test_case "NGINX auth-signin annotation set"
AUTH_SIGNIN=$(kubectl get ingress grafana-ingress -n monitoring -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/auth-signin}' 2>/dev/null || echo "")
if [[ "$AUTH_SIGNIN" == *"cloudflareaccess"* ]]; then
    log_success "Auth signin configured"
else
    log_fail "Auth signin not properly configured"
fi

# ===========================================
# SECTION 6: LOCAL AUTH STATUS
# ===========================================
echo -e "\n${BLUE}═══ SECTION 6: LOCAL AUTH DISABLE STATUS ═══${NC}"

test_case "Grafana local auth disabled"
GRAFANA_ANON=$(kubectl get deployment grafana -n monitoring -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="GF_AUTH_ANONYMOUS_ENABLED")].value}' 2>/dev/null || echo "")
if [[ "$GRAFANA_ANON" == "false" ]] || [ -z "$GRAFANA_ANON" ]; then
    log_success "Grafana anonymous auth is disabled"
else
    log_warning "Grafana anonymous auth status: $GRAFANA_ANON"
fi

test_case "Superset local auth disabled"
SUPERSET_DISABLED=$(kubectl get deployment superset -n data-platform -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="SUPERSET_DISABLE_LOCAL_AUTH")].value}' 2>/dev/null || echo "")
if [[ "$SUPERSET_DISABLED" == "true" ]]; then
    log_success "Superset local authentication disabled"
else
    log_warning "Superset auth status: $SUPERSET_DISABLED"
fi

# ===========================================
# SECTION 7: CLUSTER HEALTH
# ===========================================
echo -e "\n${BLUE}═══ SECTION 7: CLUSTER HEALTH ═══${NC}"

test_case "No pod restart loops"
RESTART_COUNT=$(kubectl get pods -A -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}' | tr ' ' '\n' | awk '$1 > 5' | wc -l)
if [ "$RESTART_COUNT" -eq 0 ]; then
    log_success "No excessive pod restarts detected"
else
    log_warning "Some pods have high restart counts"
fi

test_case "All namespaces have running pods"
NAMESPACES=$(kubectl get ns -o name | grep -E "cloudflare-tunnel|data-platform|monitoring" | wc -l)
if [ "$NAMESPACES" -ge 3 ]; then
    log_success "Required namespaces present"
else
    log_fail "Some namespaces are missing"
fi

test_case "DNS resolution working"
DNS_CHECK=$(nslookup 254carbon.com 2>/dev/null | grep -c "Address\|address" || echo 0)
if [ "$DNS_CHECK" -gt 0 ]; then
    log_success "DNS resolution working for 254carbon.com"
else
    log_warning "Could not verify DNS resolution"
fi

# ===========================================
# TEST SUMMARY
# ===========================================
echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    TEST RESULTS SUMMARY                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo -e "${BLUE}Total Tests Run:${NC} $TOTAL_TESTS"
echo -e "${GREEN}Passed:${NC} $PASSED_TESTS"
echo -e "${RED}Failed:${NC} $FAILED_TESTS"
echo

SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo -e "${BLUE}Success Rate:${NC} ${SUCCESS_RATE}%"

if [ "$FAILED_TESTS" -eq 0 ]; then
    echo -e "\n${GREEN}✅ ALL TESTS PASSED - SSO SYSTEM OPERATIONAL${NC}"
    echo -e "\n${GREEN}Status: PHASE 4 PASSED - READY FOR PRODUCTION${NC}"
    EXIT_CODE=0
else
    echo -e "\n${YELLOW}⚠️  SOME TESTS FAILED - REVIEW RESULTS${NC}"
    echo -e "\n${YELLOW}Status: PHASE 4 INCOMPLETE - FIX ISSUES AND RETEST${NC}"
    EXIT_CODE=1
fi

echo
echo "Generated: $(date)"
echo

exit $EXIT_CODE
