#!/bin/bash

# SSO Phase 4: Testing & Validation Script
# This script validates the complete SSO implementation

set -e

echo "🧪 Starting SSO Phase 4: Testing & Validation"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check test result
check_test() {
    local test_name=$1
    local result=$2

    if [ "$result" = "PASS" ]; then
        print_status $GREEN "✅ $test_name: PASS"
    else
        print_status $RED "❌ $test_name: FAIL"
        return 1
    fi
}

echo "📋 Phase 4 Validation Tests:"
echo "1. Portal accessibility and redirect"
echo "2. Authentication flow"
echo "3. Service access after authentication"
echo "4. Session persistence"
echo "5. Security validation"
echo "6. Performance testing"
echo ""

# Test 1: Portal Accessibility
echo "🔍 Test 1: Portal Accessibility..."
echo "Checking if portal redirects to Cloudflare Access..."

# This would need to be done manually or with curl in a real environment
# For now, we'll check if the ingress is configured correctly
if kubectl get ingress portal-ingress -n data-platform >/dev/null 2>&1; then
    check_test "Portal ingress configured" "PASS"
else
    check_test "Portal ingress configured" "FAIL"
fi

# Test 2: Cloudflare Access Applications
echo "🔍 Test 2: Cloudflare Access Applications..."
echo "Checking if Access applications are configured..."

# Check if we can detect the applications exist (this would require API access)
# For now, we'll assume they need to be configured manually
print_status $YELLOW "⚠️  Manual check required: Verify 14 Access applications exist in Cloudflare dashboard"
print_status $YELLOW "   Expected applications: portal, apex, www, grafana, superset, datahub, trino, doris, vault, minio, dolphinscheduler, lakefs, mlflow, spark-history"

# Test 3: Service Authentication Configuration
echo "🔍 Test 3: Service Authentication Configuration..."
echo "Checking if services are configured for SSO..."

# Check if SSO ingress rules are applied
if kubectl get ingress -A -o yaml | grep -q "auth-url"; then
    check_test "SSO ingress rules applied" "PASS"
else
    check_test "SSO ingress rules applied" "FAIL"
fi

# Test 4: Grafana Configuration
echo "🔍 Test 4: Grafana SSO Configuration..."
if kubectl get configmap grafana-config -n monitoring -o yaml | grep -q "disable_login_form"; then
    check_test "Grafana local auth disabled" "PASS"
else
    check_test "Grafana local auth disabled" "FAIL"
fi

# Test 5: Superset Configuration
echo "🔍 Test 5: Superset SSO Configuration..."
if kubectl get configmap superset-config-sso -n data-platform >/dev/null 2>&1; then
    check_test "Superset SSO config created" "PASS"
else
    check_test "Superset SSO config created" "FAIL"
fi

# Test 6: Pod Health
echo "🔍 Test 6: Service Health..."
echo "Checking if all services are running..."

# Check critical services
services=("portal" "grafana" "datahub-frontend" "trino" "doris-fe" "vault" "minio" "superset")
all_healthy=true

for service in "${services[@]}"; do
    # Find pods for this service
    pods=$(kubectl get pods -A --no-headers | grep "$service" | grep -v Completed | awk '{print $2}' || true)

    if [ -n "$pods" ]; then
        while read -r pod; do
            if [ -n "$pod" ]; then
                status=$(kubectl get pod -A | grep "$pod" | awk '{print $4}')
                if [[ "$status" != "Running" ]]; then
                    print_status $RED "❌ Pod $pod status: $status"
                    all_healthy=false
                fi
            fi
        done <<< "$pods"
    else
        print_status $YELLOW "⚠️  Service $service not found"
    fi
done

if $all_healthy; then
    check_test "All services running" "PASS"
else
    check_test "All services running" "FAIL"
fi

# Test 7: Ingress Status
echo "🔍 Test 7: Ingress Status..."
echo "Checking ingress configurations..."

ingress_count=$(kubectl get ingress -A --no-headers | wc -l)
if [ "$ingress_count" -gt 0 ]; then
    check_test "Ingress rules configured" "PASS"
else
    check_test "Ingress rules configured" "FAIL"
fi

# Test 8: TLS Certificates
echo "🔍 Test 8: TLS Certificates..."
echo "Checking certificate status..."

cert_count=$(kubectl get certificate -A --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$cert_count" -gt 0 ]; then
    check_test "TLS certificates configured" "PASS"
else
    check_test "TLS certificates configured" "FAIL"
fi

# Summary
echo ""
echo "📊 Phase 4 Validation Summary:"
echo "=============================="

# Count passed/failed tests
passed=0
failed=0
warnings=0

# These would need to be tracked from the actual test results
# For now, provide manual testing instructions

echo ""
echo "🔧 Manual Testing Required:"
echo "=========================="
echo ""
echo "1. **Portal Access Test**:"
echo "   - Visit: https://254carbon.cloudflareaccess.com"
echo "   - Expected: Redirect to Cloudflare login page"
echo "   - Enter email, receive OTP, complete login"
echo "   - Expected: Portal loads with service catalog"
echo ""
echo "2. **Service Access Test**:"
echo "   - From portal, click any service link"
echo "   - Expected: Access service without additional login"
echo "   - Test multiple services in same session"
echo ""
echo "3. **Direct Service Access Test**:"
echo "   - Try accessing service URLs directly:"
echo "     https://grafana.254carbon.com"
echo "     https://datahub.254carbon.com"
echo "     https://trino.254carbon.com"
echo "   - Expected: Redirect to Cloudflare login"
echo ""
echo "4. **Session Persistence Test**:"
echo "   - Login once, access multiple services"
echo "   - Close browser, reopen, access service"
echo "   - Expected: May need re-authentication after timeout"
echo ""
echo "5. **Security Test**:"
echo "   - Try accessing services with invalid/missing tokens"
echo "   - Expected: Access denied"
echo ""
echo "6. **Performance Test**:"
echo "   - Time portal load (< 100ms target)"
echo "   - Time service access (< 500ms target)"
echo "   - Test with multiple concurrent users"
echo ""
echo "📋 Testing Checklist:"
echo "- [ ] Portal redirects to Cloudflare login"
echo "- [ ] Email OTP authentication works"
echo "- [ ] Portal loads after authentication"
echo "- [ ] Service links work without re-login"
echo "- [ ] Direct service access requires auth"
echo "- [ ] Session persists across services"
echo "- [ ] Performance meets targets"
echo "- [ ] Security policies enforced"
echo ""
echo "🔍 Troubleshooting Commands:"
echo "kubectl get ingress -A (check ingress status)"
echo "kubectl describe ingress <name> (detailed info)"
echo "kubectl logs -n cloudflare-tunnel -f (tunnel logs)"
echo "kubectl logs -n data-platform -l app=portal -f (portal logs)"
echo ""
echo "📖 Documentation References:"
echo "docs/sso/guide.md - Complete implementation guide"
echo "docs/sso/cloudflare-access.md - Configuration details"
echo "docs/sso/checklist.md - Progress tracking"
echo ""
echo "✅ Ready for Production Testing!"

# Save test results
echo "$(date): SSO Phase 4 validation completed" >> /tmp/sso-implementation.log
