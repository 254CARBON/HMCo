#!/bin/bash
# Comprehensive Platform Verification Script
# Verifies all components of the 254Carbon Commodity Platform
#
# Usage: ./verify-platform-complete.sh

set -e

NAMESPACE="data-platform"
EXIT_CODE=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  254Carbon Platform Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
    EXIT_CODE=1
}

check_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

section_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# 1. Check Kubernetes connectivity
section_header "1. Kubernetes Connectivity"

if kubectl cluster-info &> /dev/null; then
    check_pass "Kubernetes cluster is accessible"
else
    check_fail "Cannot connect to Kubernetes cluster"
    exit 1
fi

if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    check_pass "Namespace '$NAMESPACE' exists"
else
    check_fail "Namespace '$NAMESPACE' not found"
    exit 1
fi

# 2. Check pod health
section_header "2. Pod Health Status"

# Count total pods
TOTAL_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
RUNNING_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
PENDING_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l)
FAILED_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed --no-headers 2>/dev/null | wc -l)

echo "Total pods: $TOTAL_PODS"
echo "  Running: $RUNNING_PODS"
echo "  Pending: $PENDING_PODS"
echo "  Failed: $FAILED_PODS"

if [ "$RUNNING_PODS" -gt 30 ]; then
    check_pass "Sufficient pods running ($RUNNING_PODS)"
else
    check_warn "Low pod count ($RUNNING_PODS), expected 30+"
fi

if [ "$FAILED_PODS" -gt 0 ]; then
    check_warn "$FAILED_PODS pod(s) in Failed state"
fi

# Check critical components
check_pod() {
    local label=$1
    local name=$2
    
    if kubectl get pods -n "$NAMESPACE" -l "$label" --field-selector=status.phase=Running &> /dev/null; then
        local count=$(kubectl get pods -n "$NAMESPACE" -l "$label" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            check_pass "$name ($count pod(s) running)"
        else
            check_fail "$name (no running pods)"
        fi
    else
        check_fail "$name (not found or not running)"
    fi
}

check_pod "app=dolphinscheduler-api" "DolphinScheduler API"
check_pod "app=dolphinscheduler-master" "DolphinScheduler Master"
check_pod "app=dolphinscheduler-worker" "DolphinScheduler Worker"
check_pod "app=superset" "Superset"
check_pod "app=trino-coordinator" "Trino Coordinator"
check_pod "app=datahub-frontend" "DataHub Frontend"
check_pod "app=postgres-shared" "PostgreSQL"
check_pod "app=minio" "MinIO"
check_pod "app=kafka" "Kafka"

# 3. Check API keys configuration
section_header "3. API Keys Configuration"

if kubectl get secret seatunnel-api-keys -n "$NAMESPACE" &> /dev/null; then
    check_pass "API keys secret exists"
    
    # Check if keys are configured (not placeholder values)
    FRED_KEY=$(kubectl get secret seatunnel-api-keys -n "$NAMESPACE" -o jsonpath='{.data.FRED_API_KEY}' 2>/dev/null | base64 -d 2>/dev/null || echo "")
    EIA_KEY=$(kubectl get secret seatunnel-api-keys -n "$NAMESPACE" -o jsonpath='{.data.EIA_API_KEY}' 2>/dev/null | base64 -d 2>/dev/null || echo "")
    
    if [[ "$FRED_KEY" != "your-"* ]] && [[ -n "$FRED_KEY" ]] && [[ ${#FRED_KEY} -gt 10 ]]; then
        check_pass "FRED_API_KEY configured"
    else
        check_warn "FRED_API_KEY not configured or invalid"
    fi
    
    if [[ "$EIA_KEY" != "your-"* ]] && [[ -n "$EIA_KEY" ]] && [[ ${#EIA_KEY} -gt 10 ]]; then
        check_pass "EIA_API_KEY configured"
    else
        check_warn "EIA_API_KEY not configured or invalid"
    fi
else
    check_fail "API keys secret not found"
fi

# 4. Check service endpoints
section_header "4. Service Endpoints"

check_service() {
    local service=$1
    local name=$2
    
    if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
        local cluster_ip=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
        check_pass "$name service ($cluster_ip)"
    else
        check_fail "$name service not found"
    fi
}

check_service "dolphinscheduler-api-service" "DolphinScheduler API"
check_service "superset" "Superset"
check_service "trino-coordinator" "Trino"
check_service "datahub-frontend-service" "DataHub"
check_service "postgres-shared-service" "PostgreSQL"
check_service "minio-service" "MinIO"

# 5. Check database connectivity
section_header "5. Database Connectivity"

# Test PostgreSQL connection
if kubectl exec -n "$NAMESPACE" -it $(kubectl get pod -n "$NAMESPACE" -l app=postgres-shared -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) -- psql -U postgres -c "SELECT 1" &> /dev/null 2>&1; then
    check_pass "PostgreSQL connection successful"
else
    check_warn "PostgreSQL connection failed (may require password)"
fi

# Test Trino connection (basic check)
TRINO_POD=$(kubectl get pod -n "$NAMESPACE" -l app=trino-coordinator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$TRINO_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$TRINO_POD" -- curl -s http://localhost:8080/v1/info &> /dev/null; then
        check_pass "Trino API responding"
    else
        check_warn "Trino API not responding"
    fi
else
    check_fail "Trino coordinator pod not found"
fi

# 6. Check DolphinScheduler workflows
section_header "6. DolphinScheduler Workflows"

# Check if workflow ConfigMap exists
if kubectl get configmap dolphinscheduler-commodity-workflows -n "$NAMESPACE" &> /dev/null; then
    check_pass "Workflow ConfigMap exists"
    
    WORKFLOW_COUNT=$(kubectl get configmap dolphinscheduler-commodity-workflows -n "$NAMESPACE" -o jsonpath='{.data}' 2>/dev/null | jq -r 'keys[]' 2>/dev/null | grep -c '\.json$' || echo "0")
    echo "  Workflows defined: $WORKFLOW_COUNT"
    
    if [ "$WORKFLOW_COUNT" -ge 5 ]; then
        check_pass "Expected workflows present ($WORKFLOW_COUNT)"
    else
        check_warn "Expected 5+ workflows, found $WORKFLOW_COUNT"
    fi
else
    check_fail "Workflow ConfigMap not found"
fi

# Check DolphinScheduler API accessibility (simplified check)
DS_API_POD=$(kubectl get pod -n "$NAMESPACE" -l app=dolphinscheduler-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$DS_API_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$DS_API_POD" -- curl -s http://localhost:12345/dolphinscheduler/ui/ &> /dev/null; then
        check_pass "DolphinScheduler API accessible"
    else
        check_warn "DolphinScheduler API not responding"
    fi
fi

# 7. Check Superset dashboards
section_header "7. Superset Dashboards"

# Check if dashboard ConfigMap exists
if kubectl get configmap superset-commodity-dashboards -n "$NAMESPACE" &> /dev/null 2>&1; then
    check_pass "Dashboard ConfigMap exists"
    
    DASHBOARD_COUNT=$(kubectl get configmap superset-commodity-dashboards -n "$NAMESPACE" -o jsonpath='{.data}' 2>/dev/null | jq -r 'keys[]' 2>/dev/null | grep -c '\.json$' || echo "0")
    echo "  Dashboards defined: $DASHBOARD_COUNT"
    
    if [ "$DASHBOARD_COUNT" -ge 1 ]; then
        check_pass "Dashboards present ($DASHBOARD_COUNT)"
    else
        check_warn "No dashboards found in ConfigMap"
    fi
else
    check_warn "Dashboard ConfigMap not found (may not be created yet)"
fi

# Check Superset health
SUPERSET_POD=$(kubectl get pod -n "$NAMESPACE" -l app=superset -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$SUPERSET_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$SUPERSET_POD" -- curl -s http://localhost:8088/health &> /dev/null; then
        check_pass "Superset health check passed"
    else
        check_warn "Superset health check failed"
    fi
fi

# 8. Check data quality components
section_header "8. Data Quality Framework"

# Check if Deequ validation is deployed
if kubectl get deployment deequ-validation -n "$NAMESPACE" &> /dev/null 2>&1; then
    check_pass "Deequ validation deployed"
else
    check_warn "Deequ validation not deployed"
fi

# Check if data quality exporter is running
if kubectl get pods -n "$NAMESPACE" -l app=data-quality-exporter --field-selector=status.phase=Running &> /dev/null 2>&1; then
    check_pass "Data quality exporter running"
else
    check_warn "Data quality exporter not running"
fi

# 9. Check monitoring stack
section_header "9. Monitoring Stack"

MONITORING_NS="monitoring"

if kubectl get namespace "$MONITORING_NS" &> /dev/null; then
    check_pass "Monitoring namespace exists"
    
    # Check Prometheus
    if kubectl get pods -n "$MONITORING_NS" -l app.kubernetes.io/name=prometheus --field-selector=status.phase=Running &> /dev/null 2>&1; then
        check_pass "Prometheus running"
    else
        check_warn "Prometheus not running"
    fi
    
    # Check Grafana
    if kubectl get pods -n "$MONITORING_NS" -l app.kubernetes.io/name=grafana --field-selector=status.phase=Running &> /dev/null 2>&1; then
        check_pass "Grafana running"
    else
        check_warn "Grafana not running"
    fi
    
    # Check AlertManager
    if kubectl get pods -n "$MONITORING_NS" -l app.kubernetes.io/name=alertmanager --field-selector=status.phase=Running &> /dev/null 2>&1; then
        check_pass "AlertManager running"
    else
        check_warn "AlertManager not running"
    fi
else
    check_warn "Monitoring namespace not found"
fi

# 10. Check MLflow
section_header "10. MLflow Model Management"

# Check MLflow deployment
if kubectl get deployment mlflow -n "$NAMESPACE" &> /dev/null; then
    MLFLOW_READY=$(kubectl get deployment mlflow -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    MLFLOW_DESIRED=$(kubectl get deployment mlflow -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    if [ "$MLFLOW_READY" -eq "$MLFLOW_DESIRED" ] && [ "$MLFLOW_READY" -gt 0 ]; then
        check_pass "MLflow deployment ($MLFLOW_READY/$MLFLOW_DESIRED replicas)"
    else
        check_warn "MLflow deployment ($MLFLOW_READY/$MLFLOW_DESIRED replicas)"
    fi
else
    check_warn "MLflow not deployed"
fi

# Check MLflow service
if kubectl get service mlflow -n "$NAMESPACE" &> /dev/null; then
    check_pass "MLflow service exists"
else
    check_warn "MLflow service not found"
fi

# Check MLflow health
MLFLOW_POD=$(kubectl get pod -n "$NAMESPACE" -l app=mlflow --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$MLFLOW_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$MLFLOW_POD" -- curl -s http://localhost:5000/health &> /dev/null; then
        check_pass "MLflow health check passed"
    else
        check_warn "MLflow health check failed"
    fi
fi

# 11. Check Ingress and DNS
section_header "11. Ingress and DNS"

# Check ingress resources
INGRESS_COUNT=$(kubectl get ingress -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
echo "Ingress resources: $INGRESS_COUNT"

if [ "$INGRESS_COUNT" -gt 5 ]; then
    check_pass "Ingress resources configured ($INGRESS_COUNT)"
else
    check_warn "Few ingress resources ($INGRESS_COUNT)"
fi

# Check Cloudflare tunnel
if kubectl get pods -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --field-selector=status.phase=Running &> /dev/null 2>&1; then
    TUNNEL_COUNT=$(kubectl get pods -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    check_pass "Cloudflare tunnel running ($TUNNEL_COUNT replica(s))"
else
    check_warn "Cloudflare tunnel not running"
fi

# 11. Final Summary
section_header "Final Summary"

TOTAL_CHECKS=$((PASSED + FAILED + WARNINGS))

echo ""
echo "Total checks: $TOTAL_CHECKS"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"
echo -e "  ${YELLOW}Warnings: $WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ Platform is fully operational!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}Quick Links:${NC}"
    echo "  - Portal: https://portal.254carbon.com"
    echo "  - DolphinScheduler: https://dolphinscheduler.254carbon.com/dolphinscheduler/ui/"
    echo "  - Superset: https://superset.254carbon.com"
    echo "  - Grafana: https://grafana.254carbon.com"
    echo "  - DataHub: https://datahub.254carbon.com"
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⚠ Platform is operational with warnings${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Review warnings above for optional improvements."
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ✗ Platform has critical issues${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Please address the failed checks above."
fi

echo ""

exit $EXIT_CODE

