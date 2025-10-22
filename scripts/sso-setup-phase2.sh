#!/bin/bash

# SSO Implementation Phase 2 & 3 Setup Script
# Complete Cloudflare Access integration for 254Carbon cluster
# Team: qagi (Cloudflare Zero Trust)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-}"
TEAM_NAME="${CLOUDFLARE_TEAM_NAME:-qagi}"
TUNNEL_ID="${CLOUDFLARE_TUNNEL_ID:-}"
TUNNEL_NAME="254carbon-cluster"
NAMESPACE_DATA="data-platform"
NAMESPACE_MONITORING="monitoring"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    if ! command -v base64 &> /dev/null; then
        log_error "base64 not found."
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Verify Cloudflare credentials
verify_cloudflare_creds() {
    log_info "Verifying Cloudflare credentials..."
    
    # Account ID is no longer needed for ingress annotations, but keep a soft check for compatibility
    if [[ -z "$ACCOUNT_ID" ]]; then
        log_warning "CLOUDFLARE_ACCOUNT_ID not set (ok). Ingress uses Team domain instead."
    fi
    
    if [[ -z "$TUNNEL_ID" ]]; then
        log_error "CLOUDFLARE_TUNNEL_ID environment variable not set"
        log_info "Set it with: export CLOUDFLARE_TUNNEL_ID=<your-tunnel-id>"
        exit 1
    fi
    
    log_success "Cloudflare credentials verified"
}

# Update tunnel credentials in secret
update_tunnel_credentials() {
    log_info "Updating Cloudflare tunnel credentials..."
    
    # Check if secret exists
    if ! kubectl get secret cloudflare-tunnel-credentials -n cloudflare-tunnel &> /dev/null; then
        log_warning "Tunnel secret not found. Creating new secret..."
        
        # Prompt for credentials
        read -sp "Enter Cloudflare tunnel auth token: " TUNNEL_TOKEN
        echo
        
        read -sp "Enter Cloudflare account tag: " ACCOUNT_TAG
        echo
        
        # Create credentials.json
        CREDS_JSON=$(cat <<EOF
{
  "tunnel_id": "$TUNNEL_ID",
  "account_tag": "$ACCOUNT_TAG",
  "tunnel_name": "$TUNNEL_NAME",
  "auth_token": "$TUNNEL_TOKEN",
  "tunnel_remote_config": true
}
EOF
)
        
        # Create the secret
        kubectl create secret generic cloudflare-tunnel-credentials \
            -n cloudflare-tunnel \
            --from-literal=credentials.json="$CREDS_JSON" || true
    else
        log_success "Tunnel secret already exists"
    fi
}

# Update ingress rules with Cloudflare Access annotations
update_ingress_rules() {
    log_info "Updating ingress rules with Cloudflare Access authentication..."
    
    # Update the ingress configuration with Team name (replace default 'qagi' placeholder)
    if [[ -n "$TEAM_NAME" ]]; then
      sed -i "s/qagi\\.cloudflareaccess\\.com/${TEAM_NAME}.cloudflareaccess.com/g" k8s/ingress/ingress-cloudflare-sso.yaml || true
      sed -i "s/qagi\\.cloudflareaccess\\.com/${TEAM_NAME}.cloudflareaccess.com/g" k8s/ingress/ingress-sso-rules.yaml || true
      sed -i "s/qagi\\.cloudflareaccess\\.com/${TEAM_NAME}.cloudflareaccess.com/g" k8s/ingress/portal-deployment.yaml || true
    fi
    
    # Apply ingress rules
    log_info "Applying ingress configuration..."
    kubectl apply -f k8s/ingress/ingress-cloudflare-sso.yaml
    
    # Wait for ingress to be ready
    log_info "Waiting for ingress rules to be ready..."
    sleep 5
    
    # Verify ingress
    if kubectl get ingress -A | grep -q 254carbon; then
        log_success "Ingress rules updated successfully"
        kubectl get ingress -A | grep 254carbon
    else
        log_error "Ingress rules not created properly"
        exit 1
    fi
}

# Disable Grafana local authentication
disable_grafana_auth() {
    log_info "Disabling Grafana local authentication..."
    
    if ! kubectl get deployment grafana -n $NAMESPACE_MONITORING &> /dev/null; then
        log_warning "Grafana deployment not found, skipping..."
        return
    fi
    
    # Update Grafana config to disable anonymous access
    kubectl -n $NAMESPACE_MONITORING patch deployment grafana --type='json' \
        -p='[
            {
              "op": "replace",
              "path": "/spec/template/spec/containers/0/env",
              "value": [
                {
                  "name": "GF_AUTH_ANONYMOUS_ENABLED",
                  "value": "false"
                },
                {
                  "name": "GF_AUTH_BASIC_ENABLED",
                  "value": "false"
                },
                {
                  "name": "GF_USERS_AUTO_ASSIGN_ORG_ROLE",
                  "value": "Viewer"
                },
                {
                  "name": "GF_SECURITY_ADMIN_USER",
                  "value": "admin"
                }
              ]
            }
        ]' || log_warning "Failed to patch Grafana, may already be configured"
    
    # Restart Grafana
    log_info "Restarting Grafana deployment..."
    kubectl rollout restart deployment/grafana -n $NAMESPACE_MONITORING
    kubectl rollout status deployment/grafana -n $NAMESPACE_MONITORING --timeout=5m
    
    log_success "Grafana local authentication disabled"
}

# Disable Superset local authentication
disable_superset_auth() {
    log_info "Disabling Superset local authentication..."
    
    if ! kubectl get deployment superset -n $NAMESPACE_DATA &> /dev/null; then
        log_warning "Superset deployment not found, skipping..."
        return
    fi
    
    # Update Superset environment
    kubectl -n $NAMESPACE_DATA set env deployment/superset \
        SUPERSET_DISABLE_LOCAL_AUTH=true || log_warning "Failed to update Superset env"
    
    # Restart Superset
    log_info "Restarting Superset deployment..."
    kubectl rollout restart deployment/superset -n $NAMESPACE_DATA
    kubectl rollout status deployment/superset -n $NAMESPACE_DATA --timeout=5m
    
    log_success "Superset local authentication disabled"
}

# Verify all services are running
verify_services() {
    log_info "Verifying all services are running..."
    
    SERVICES=(
        "grafana:monitoring"
        "superset:data-platform"
        "vault:data-platform"
        "minio-console:data-platform"
        "dolphinscheduler-api:data-platform"
        "datahub-frontend:data-platform"
        "trino-coordinator:data-platform"
        "doris-fe-service:data-platform"
        "lakefs:data-platform"
        "portal:data-platform"
    )
    
    FAILED=0
    for SERVICE in "${SERVICES[@]}"; do
        IFS=':' read -r NAME NS <<< "$SERVICE"
        
        if kubectl get deployment $NAME -n $NS &> /dev/null; then
            READY=$(kubectl get deployment $NAME -n $NS -o jsonpath='{.status.readyReplicas}')
            DESIRED=$(kubectl get deployment $NAME -n $NS -o jsonpath='{.spec.replicas}')
            
            if [[ "$READY" == "$DESIRED" ]]; then
                log_success "$NAME ($NS): $READY/$DESIRED ready"
            else
                log_warning "$NAME ($NS): $READY/$DESIRED ready (not all replicas ready)"
                FAILED=$((FAILED + 1))
            fi
        else
            log_warning "$NAME not found in $NS"
        fi
    done
    
    if [[ $FAILED -gt 0 ]]; then
        log_warning "$FAILED services not fully ready yet"
    else
        log_success "All services verified and ready"
    fi
}

# Verify DNS records
verify_dns() {
    log_info "Verifying DNS records..."
    
    DOMAINS=(
        "254carbon.com"
        "portal.254carbon.com"
        "grafana.254carbon.com"
        "superset.254carbon.com"
        "vault.254carbon.com"
        "minio.254carbon.com"
        "dolphin.254carbon.com"
        "datahub.254carbon.com"
        "trino.254carbon.com"
        "doris.254carbon.com"
        "lakefs.254carbon.com"
    )
    
    log_info "Testing DNS resolution (may require internet access)..."
    for DOMAIN in "${DOMAINS[@]}"; do
        if nslookup $DOMAIN &> /dev/null; then
            log_success "DNS: $DOMAIN resolved"
        else
            log_warning "DNS: $DOMAIN could not be resolved"
        fi
    done
}

# Generate test report
generate_report() {
    log_info "Generating deployment report..."
    
    cat > /tmp/sso-deployment-report.txt <<EOF
SSO Implementation Phase 2 & 3 Report
Generated: $(date)
Cluster: $(kubectl config current-context)
Account ID: $ACCOUNT_ID
Tunnel ID: $TUNNEL_ID

Services Status:
$(kubectl get pods -A -l app=portal,app=grafana,app=superset,app=vault,app=minio,app=dolphinscheduler 2>/dev/null | head -20)

Ingress Rules:
$(kubectl get ingress -A | grep 254carbon)

Cloudflare Tunnel Status:
$(kubectl get pods -n cloudflare-tunnel 2>/dev/null | head -5)

Notes:
- Ensure all Cloudflare Access applications have been created in Zero Trust dashboard
- Replace <ACCOUNT_ID> with actual account ID before applying ingress rules
- Verify tunnel credentials are correct and tunnel is connected
- Test service access after Cloudflare configuration is complete

Next Steps:
1. Configure Cloudflare Access applications in Zero Trust dashboard
2. Test portal access: https://254carbon.com
3. Test service access with authentication
4. Review audit logs in Cloudflare dashboard
EOF
    
    log_success "Report saved to /tmp/sso-deployment-report.txt"
    cat /tmp/sso-deployment-report.txt
}

# Main execution
main() {
    log_info "Starting SSO Implementation Phase 2 & 3 Setup"
    echo
    
    check_prerequisites
    verify_cloudflare_creds
    update_tunnel_credentials
    update_ingress_rules
    disable_grafana_auth
    disable_superset_auth
    verify_services
    verify_dns
    generate_report
    
    echo
    log_success "SSO Implementation Phase 2 & 3 Setup Complete!"
    echo
    log_info "Next Steps:"
    log_info "1. Go to Cloudflare Zero Trust dashboard (https://dash.cloudflare.com/zero-trust)"
    log_info "2. Create Access applications for all services (see CLOUDFLARE_SSO_SETUP.md)"
    log_info "3. Configure policies for each application"
    log_info "4. Test access: https://254carbon.com"
    echo
}

main "$@"
