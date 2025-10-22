#!/bin/bash
#
# Verify and Fix Cloudflare Tunnel Connectivity
# Diagnoses and resolves tunnel connection issues
#
# Usage: ./verify-tunnel.sh [COMMAND]
#
# Commands:
#   status   - Check tunnel status
#   logs     - Show tunnel pod logs
#   test     - Test portal connectivity
#   restart  - Restart tunnel pods
#   fix      - Run all diagnostics and attempt fixes
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="cloudflare-tunnel"
DOMAIN="254carbon.com"

# Verify pod status
check_pods() {
    echo -e "${YELLOW}1. Checking tunnel pods...${NC}"
    
    PODS=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=cloudflare-tunnel --no-headers 2>/dev/null | wc -l)
    RUNNING=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=cloudflare-tunnel --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    
    if [[ $RUNNING -eq 2 ]]; then
        echo -e "${GREEN}✓ Both tunnel pods running (2/2)${NC}"
        return 0
    else
        echo -e "${RED}✗ Only $RUNNING/2 pods running${NC}"
        kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=cloudflare-tunnel
        return 1
    fi
}

# Verify credentials
check_credentials() {
    echo ""
    echo -e "${YELLOW}2. Checking tunnel credentials...${NC}"
    
    if ! kubectl get secret cloudflare-tunnel-credentials -n $NAMESPACE &>/dev/null; then
        echo -e "${RED}✗ Credentials secret not found${NC}"
        return 1
    fi
    
    # Decode and verify credentials
    CREDS=$(kubectl get secret cloudflare-tunnel-credentials -n $NAMESPACE \
        -o jsonpath='{.data.credentials\.json}' | base64 -d)
    
    # Check for base64-encoded token (bad) vs UUID format (good)
    if echo "$CREDS" | grep -q "auth_token.*[A-Za-z0-9+/=]*[A-Za-z0-9+/=]"; then
        TOKEN=$(echo "$CREDS" | grep -o '"auth_token":"[^"]*"' | cut -d'"' -f4)
        
        # UUID format: 8-4-4-4-12
        if [[ $TOKEN =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
            echo -e "${GREEN}✓ Credentials properly formatted${NC}"
            echo "  Tunnel ID: $(echo "$CREDS" | grep -o '"tunnel_id":"[^"]*"' | cut -d'"' -f4)"
            echo "  Account ID: $(echo "$CREDS" | grep -o '"account_tag":"[^"]*"' | cut -d'"' -f4)"
            return 0
        else
            echo -e "${YELLOW}⚠ Token may be base64-encoded${NC}"
            echo "  Token: $TOKEN"
            return 1
        fi
    else
        echo -e "${RED}✗ Could not parse credentials${NC}"
        return 1
    fi
}

# Check pod logs for connection status
check_logs() {
    echo ""
    echo -e "${YELLOW}3. Checking tunnel pod logs...${NC}"
    
    POD=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=cloudflare-tunnel \
        --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$POD" ]]; then
        echo -e "${RED}✗ No running pods found${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Recent logs from $POD:${NC}"
    kubectl logs -n $NAMESPACE $POD --tail=10 | tail -5
    
    # Check for connection status
    if kubectl logs -n $NAMESPACE $POD | grep -i "connected\|registered" | tail -1; then
        echo -e "${GREEN}✓ Tunnel appears connected${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ No recent connection confirmation${NC}"
        return 1
    fi
}

# Test portal connectivity
test_portal() {
    echo ""
    echo -e "${YELLOW}4. Testing portal accessibility...${NC}"
    
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN 2>/dev/null || echo "000")
    
    case "$HTTP_CODE" in
        302)
            echo -e "${GREEN}✓ Portal responding (HTTP $HTTP_CODE - redirect to login)${NC}"
            echo "  This is correct behavior for unauthenticated access"
            return 0
            ;;
        200)
            echo -e "${YELLOW}⚠ Portal responding (HTTP $HTTP_CODE)${NC}"
            return 0
            ;;
        502|503)
            echo -e "${RED}✗ Portal unreachable (HTTP $HTTP_CODE)${NC}"
            echo "  This typically indicates tunnel connection issue"
            return 1
            ;;
        000)
            echo -e "${RED}✗ Cannot reach portal (connection failed)${NC}"
            echo "  Check DNS resolution and tunnel status"
            return 1
            ;;
        *)
            echo -e "${YELLOW}⚠ Unexpected response (HTTP $HTTP_CODE)${NC}"
            return 1
            ;;
    esac
}

# Test service connectivity
test_services() {
    echo ""
    echo -e "${YELLOW}5. Testing service connectivity...${NC}"
    
    SERVICES=("grafana" "vault" "datahub")
    SUCCESS=0
    
    for SERVICE in "${SERVICES[@]}"; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://$SERVICE.$DOMAIN 2>/dev/null || echo "000")
        
        if [[ $HTTP_CODE == "302" ]] || [[ $HTTP_CODE == "200" ]] || [[ $HTTP_CODE == "401" ]]; then
            echo -e "${GREEN}✓ $SERVICE.$DOMAIN responding (HTTP $HTTP_CODE)${NC}"
            ((SUCCESS++))
        else
            echo -e "${RED}✗ $SERVICE.$DOMAIN unreachable (HTTP $HTTP_CODE)${NC}"
        fi
    done
    
    echo ""
    echo "Services responding: $SUCCESS/${#SERVICES[@]}"
    [[ $SUCCESS -ge 2 ]] && return 0 || return 1
}

# Show tunnel status
show_status() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Cloudflare Tunnel Status Report${NC}"
    echo -e "${BLUE}================================${NC}"
    
    check_pods
    check_credentials
    check_logs
    test_portal
    test_services
    
    echo ""
    echo -e "${BLUE}================================${NC}"
}

# Show logs
show_logs() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Cloudflare Tunnel Logs${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=cloudflare-tunnel -f
}

# Restart tunnel
restart_tunnel() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Restarting Cloudflare Tunnel${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    echo -e "${YELLOW}Restarting deployment...${NC}"
    kubectl rollout restart deployment/cloudflared -n $NAMESPACE
    
    echo -e "${YELLOW}Waiting for rollout...${NC}"
    kubectl rollout status deployment/cloudflared -n $NAMESPACE --timeout=2m
    
    echo ""
    echo -e "${YELLOW}Waiting for pods to stabilize...${NC}"
    sleep 5
    
    echo -e "${GREEN}✓ Tunnel restarted${NC}"
    echo ""
    echo -e "${YELLOW}Checking status...${NC}"
    show_status
}

# Run full diagnostics and fix
run_full_fix() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Full Tunnel Diagnostics & Fix${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    echo -e "${YELLOW}Running diagnostics...${NC}"
    echo ""
    
    # Check all components
    local pods_ok=0
    local creds_ok=0
    local logs_ok=0
    local portal_ok=0
    
    check_pods && ((pods_ok=1)) || ((pods_ok=0))
    check_credentials && ((creds_ok=1)) || ((creds_ok=0))
    check_logs && ((logs_ok=1)) || ((logs_ok=0))
    test_portal && ((portal_ok=1)) || ((portal_ok=0))
    
    echo ""
    echo -e "${BLUE}Diagnostic Summary:${NC}"
    echo "  Pods: $([[ $pods_ok -eq 1 ]] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}")"
    echo "  Credentials: $([[ $creds_ok -eq 1 ]] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}")"
    echo "  Logs: $([[ $logs_ok -eq 1 ]] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}")"
    echo "  Portal: $([[ $portal_ok -eq 1 ]] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}")"
    echo ""
    
    # Attempt fixes based on diagnostics
    if [[ $pods_ok -eq 0 ]]; then
        echo -e "${YELLOW}Attempting to fix pod issues...${NC}"
        restart_tunnel
    fi
    
    if [[ $creds_ok -eq 0 ]]; then
        echo -e "${YELLOW}Credentials appear to have issues.${NC}"
        echo "Use: ./scripts/update-cloudflare-credentials.sh TUNNEL_ID ACCOUNT_TAG AUTH_TOKEN"
        echo ""
        echo "Get credentials from: https://dash.cloudflare.com/zero-trust/networks/tunnels"
        return 1
    fi
    
    echo ""
    echo -e "${BLUE}Final Status Check:${NC}"
    show_status
}

# Main
COMMAND="${1:-status}"

case "$COMMAND" in
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        test_portal
        test_services
        ;;
    restart)
        restart_tunnel
        ;;
    fix)
        run_full_fix
        ;;
    *)
        echo "Usage: $0 [status|logs|test|restart|fix]"
        echo ""
        echo "Commands:"
        echo "  status  - Show tunnel status report"
        echo "  logs    - Show tunnel pod logs (live)"
        echo "  test    - Test portal and service connectivity"
        echo "  restart - Restart tunnel pods"
        echo "  fix     - Run full diagnostics and attempt fixes"
        exit 1
        ;;
esac

echo ""
