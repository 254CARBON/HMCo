#!/bin/bash
#
# Continue Phase 1 Implementation - Quick Status Check and Next Steps
# Run this script to check current status and proceed with remaining Phase 1 tasks
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Phase 1 Continuation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to check pod status
check_pods() {
    local namespace=$1
    local label=$2
    local name=$3
    
    local ready=$(kubectl get pods -n $namespace -l $label 2>/dev/null | grep -c "1/1.*Running" || echo "0")
    local total=$(kubectl get pods -n $namespace -l $label 2>/dev/null | tail -n +2 | wc -l || echo "0")
    
    if [ "$ready" -eq "$total" ] && [ "$total" -gt "0" ]; then
        echo -e "${GREEN}✓${NC} $name: $ready/$total ready"
        return 0
    else
        echo -e "${YELLOW}⏳${NC} $name: $ready/$total ready"
        return 1
    fi
}

# Phase 1.1 & 1.2 Status
print_section "Phase 1.1 & 1.2 Status (Completed)"

echo "PostgreSQL (via Kong):"
kubectl exec -n kong kong-postgres-0 -- psql -U postgres -c "\l" 2>/dev/null | grep -E "(dolphinscheduler|datahub|superset|iceberg)" | sed 's/^/ /'
echo ""

echo "MinIO:"
kubectl get pods -n data-platform -l app=minio | tail -1
echo ""

# Phase 1.3 Status
print_section "Phase 1.3: Core Services Status"

echo "Checking critical services..."
echo ""

# Trino
if check_pods "data-platform" "app=trino" "Trino"; then
    echo "  - Query engine operational"
fi

# DolphinScheduler
echo ""
echo "DolphinScheduler Components:"
check_pods "data-platform" "app.kubernetes.io/name=dolphinscheduler,component=alert" "  Alert"
check_pods "data-platform" "app.kubernetes.io/name=dolphinscheduler,component=master" "  Master"
check_pods "data-platform" "app.kubernetes.io/name=dolphinscheduler,component=worker" "  Worker"

api_ready=$(kubectl get pods -n data-platform -l app.kubernetes.io/name=dolphinscheduler,component=api 2>/dev/null | grep -c "1/1.*Running" || echo "0")
api_total=$(kubectl get pods -n data-platform -l app.kubernetes.io/name=dolphinscheduler,component=api 2>/dev/null | tail -n +2 | wc -l || echo "0")
if [ "$api_ready" -gt "0" ]; then
    echo -e "${YELLOW}⏳${NC}   API: $api_ready/$api_total ready (in progress)"
else
    echo -e "${RED}✗${NC}   API: $api_ready/$api_total ready"
fi

# Iceberg
echo ""
check_pods "data-platform" "app=iceberg-rest-catalog" "Iceberg REST Catalog"

# MinIO
check_pods "data-platform" "app=minio" "MinIO Object Storage"

# Phase 1.4 - Ingress Check
print_section "Phase 1.4: Ingress & External Access"

echo "Checking ingress controller..."
nginx_pods=$(kubectl get pods -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx 2>/dev/null | grep -c "Running" || echo "0")
if [ "$nginx_pods" -gt "0" ]; then
    echo -e "${GREEN}✓${NC} Nginx Ingress Controller: $nginx_pods pod(s) running"
else
    echo -e "${YELLOW}⚠${NC} Nginx Ingress Controller: Not detected"
    echo "   Next action: Deploy nginx-ingress"
fi

echo ""
echo "Checking Cloudflare Tunnel..."
cf_pods=$(kubectl get pods -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel 2>/dev/null | grep -c "Running" || echo "0")
if [ "$cf_pods" -gt "0" ]; then
    echo -e "${GREEN}✓${NC} Cloudflare Tunnel: $cf_pods pod(s) running"
else
    echo -e "${RED}✗${NC} Cloudflare Tunnel: Not running"
fi

# Phase 1.5 - Workflow Import
print_section "Phase 1.5: DolphinScheduler Workflow Import"

if [ "$api_ready" -gt "0" ]; then
    echo -e "${GREEN}✓${NC} DolphinScheduler API ready - Can proceed with workflow import"
    echo ""
    echo "Next action:"
    echo "  ./scripts/setup-dolphinscheduler-complete.sh"
else
    echo -e "${YELLOW}⏳${NC} Waiting for DolphinScheduler API to be fully ready"
    echo "  Current: $api_ready/$api_total API pods ready"
    echo "  Re-run this script in 5-10 minutes"
fi

# Overall Status Summary
print_section "Overall Status Summary"

total_pods=$(kubectl get pods -n data-platform --field-selector status.phase=Running 2>/dev/null | tail -n +2 | wc -l || echo "0")
problem_pods=$(kubectl get pods -n data-platform --field-selector status.phase!=Running,status.phase!=Succeeded 2>/dev/null | tail -n +2 | wc -l || echo "0")

echo "Data Platform Namespace:"
echo "  Running Pods: $total_pods"
echo "  Problematic Pods: $problem_pods"
echo ""

if [ "$problem_pods" -le "5" ]; then
    echo -e "${GREEN}✓${NC} Platform is substantially stable"
    echo ""
    echo "Recommended next steps:"
    echo "  1. Wait 5 minutes for remaining pods to stabilize"
    echo "  2. Proceed with Phase 1.4 (Ingress setup)"
    echo "  3. Then Phase 1.5 (Workflow import)"
else
    echo -e "${YELLOW}⚠${NC} Some pods still stabilizing"
    echo "  Check logs of failing pods for issues"
fi

# Storage Status
print_section "Storage Status"

echo "PersistentVolumeClaims:"
kubectl get pvc -n data-platform | tail -n +2 | while read line; do
    if echo "$line" | grep -q "Bound"; then
        echo -e "${GREEN}✓${NC} $line"
    elif echo "$line" | grep -q "Pending"; then
        echo -e "${YELLOW}⏳${NC} $line"
    else
        echo -e "${RED}✗${NC} $line"
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Status Check Complete${NC}"
echo -e "${BLUE}========================================${NC}"

