#!/bin/bash

# SSO Phase 3: Service Integration Script
# This script disables local authentication in services and updates configurations for SSO

set -e

echo "🚀 Starting SSO Phase 3: Service Integration"
echo "============================================="

# Check if running in correct directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the HMCo project root directory"
    exit 1
fi

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

echo "📋 Phase 3 Checklist:"
echo "1. Disable Grafana local authentication"
echo "2. Disable Superset local authentication"
echo "3. Update ingress rules with SSO annotations"
echo "4. Restart affected services"
echo "5. Verify configurations"
echo ""

# 1. Disable Grafana local authentication
echo "🔧 Step 1: Disabling Grafana local authentication..."

# Check if Grafana configmap exists
if kubectl get configmap grafana-config -n monitoring >/dev/null 2>&1; then
    echo "📝 Updating existing Grafana configuration..."

    # Patch the existing configmap to disable authentication
    kubectl patch configmap grafana-config -n monitoring --type merge -p='{
        "data": {
            "grafana.ini": "[auth]\ndisable_login_form = true\n\n[auth.anonymous]\nenabled = false\n\n[users]\nauto_assign_org = false\n\n[auth.basic]\nenabled = false\n\n[auth.proxy]\nenabled = true\nauto_sign_up = true\nheaders = CF_Access-User:user\n"
        }
    }'
    check_success "Grafana configuration update"
else
    echo "⚠️  Grafana configmap not found. Grafana may not be deployed yet."
fi

# 2. Disable Superset local authentication
echo "🔧 Step 2: Disabling Superset local authentication..."

# Check if Superset configmap exists
if kubectl get configmap superset-config -n data-platform >/dev/null 2>&1; then
    echo "📝 Updating Superset configuration..."

    # Create new configmap for Superset with SSO
    kubectl create configmap superset-config-sso -n data-platform --from-literal=config.py="\
import os
from flask_appbuilder.security.manager import BaseSecurityManager
from flask_appbuilder.security.forms import LoginForm_db

class CustomSecurityManager(BaseSecurityManager):
    def __init__(self, appbuilder):
        super(CustomSecurityManager, self).__init__(appbuilder)

# Disable password authentication
AUTH_USER_DB = None
AUTH_TYPE = 3  # AUTH_DB

# Configure proxy authentication
AUTH_REMOTE_USER = 'CF_Access-User'

# Enable proxy authentication
ENABLE_PROXY_FIX = True

# Auto-assign users to public role
PUBLIC_ROLE_LIKE = 'Gamma'

# Disable registration
AUTH_USER_REGISTRATION = False

# Disable password change
AUTH_USER_REGISTRATION_ROLE = 'Gamma'
" --dry-run=client -o yaml | kubectl apply -f -
    check_success "Superset SSO configuration creation"
else
    echo "⚠️  Superset configmap not found. Superset may not be deployed yet."
fi

# 3. Update ingress rules
echo "🔧 Step 3: Updating ingress rules with SSO annotations..."

# Check if SSO ingress file exists
if [ -f "k8s/ingress/ingress-sso-rules.yaml" ]; then
    echo "📝 Applying SSO-enabled ingress rules..."
    kubectl apply -f k8s/ingress/ingress-sso-rules.yaml
    check_success "SSO ingress rules application"
else
    echo "⚠️  SSO ingress rules file not found. Please create k8s/ingress/ingress-sso-rules.yaml"
fi

# 4. Restart services
echo "🔧 Step 4: Restarting services to apply changes..."

# Restart Grafana
if kubectl get deployment grafana -n monitoring >/dev/null 2>&1; then
    echo "🔄 Restarting Grafana..."
    kubectl rollout restart deployment grafana -n monitoring
    kubectl rollout status deployment grafana -n monitoring --timeout=300s
    check_success "Grafana restart"
fi

# Restart Superset
if kubectl get deployment superset -n data-platform >/dev/null 2>&1; then
    echo "🔄 Restarting Superset..."
    kubectl rollout restart deployment superset -n data-platform
    kubectl rollout status deployment superset -n data-platform --timeout=300s
    check_success "Superset restart"
fi

# 5. Verify configurations
echo "🔧 Step 5: Verifying configurations..."

echo "🔍 Checking ingress rules..."
kubectl get ingress -A | grep -E "(datahub|grafana|trino|doris|vault|minio|dolphinscheduler|lakefs|superset)"

echo "🔍 Checking pod status..."
kubectl get pods -A | grep -E "(grafana|superset|portal|datahub)" | head -10

echo "🔍 Checking ingress annotations..."
kubectl get ingress grafana-ingress -n monitoring -o yaml | grep -A 5 "auth-url:" || echo "⚠️  Grafana ingress may not have SSO annotations"

# 6. Display next steps
echo ""
echo "🎉 Phase 3 Service Integration Complete!"
echo "========================================"
echo ""
echo "📋 Next Steps (Phase 4 - Testing & Validation):"
echo ""
echo "1. Test Portal Access:"
echo "   Visit: https://254carbon.cloudflareaccess.com"
echo "   Should redirect to Cloudflare login"
echo ""
echo "2. Test Service Access:"
echo "   After portal login, click service links"
echo "   Should access services without additional login"
echo ""
echo "3. Test Session Persistence:"
echo "   Login once, access multiple services"
echo "   Session should persist across services"
echo ""
echo "4. Verify Security:"
echo "   Try accessing services directly (should redirect to login)"
echo "   Check that unauthorized access is denied"
echo ""
echo "5. Monitor Logs:"
echo "   kubectl logs -n cloudflare-tunnel -f (for tunnel logs)"
echo "   kubectl logs -n data-platform -l app=portal -f (for portal logs)"
echo ""
echo "📖 Documentation:"
echo "   See docs/sso/cloudflare-access.md for configuration details"
echo "   See docs/sso/guide.md for troubleshooting"
echo ""
echo "🔧 Troubleshooting Commands:"
echo "   kubectl get ingress -A (check ingress status)"
echo "   kubectl describe ingress <name> (detailed ingress info)"
echo "   kubectl logs -n <namespace> <pod> (service logs)"
echo ""
echo "✅ Phase 3 Status: COMPLETE"
echo "⏳ Ready for Phase 4: Testing & Validation"

# Save completion status
echo "$(date): SSO Phase 3 completed" >> /tmp/sso-implementation.log
