#!/bin/bash
# Commodity Data Platform - Deployment Verification Script
# Run this to check all components are operational

set -e

echo "=================================="
echo "Commodity Data Platform Verification"
echo "Date: $(date)"
echo "=================================="
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function check_component() {
    local name=$1
    local namespace=$2
    local label=$3
    local expected=$4
    
    local running=$(kubectl get pods -n $namespace -l $label --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    
    if [ "$running" -eq "$expected" ]; then
        echo -e "${GREEN}✓${NC} $name: $running/$expected running"
        return 0
    elif [ "$running" -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC} $name: $running/$expected running (partial)"
        return 1
    else
        echo -e "${RED}✗${NC} $name: $running/$expected running (FAILED)"
        return 1
    fi
}

echo "## Core Platform Components"
echo
check_component "DolphinScheduler Master" "data-platform" "app=dolphinscheduler-master" 1
check_component "DolphinScheduler API" "data-platform" "app=dolphinscheduler-api" 3
check_component "DolphinScheduler Workers" "data-platform" "app=dolphinscheduler-worker" 2
check_component "DataHub Frontend" "data-platform" "app=datahub-frontend" 3
check_component "DataHub GMS" "data-platform" "app=datahub-gms" 1
check_component "Trino Coordinator" "data-platform" "app=trino-coordinator" 1
check_component "Trino Workers" "data-platform" "app=trino-worker" 2
check_component "Superset Web" "data-platform" "app=superset,component=web" 1
echo

echo "## Commodity Data Platform Components"
echo
check_component "SeaTunnel Engines" "data-platform" "app=seatunnel" 2
check_component "RAPIDS GPU Processor" "data-platform" "app=rapids" 1
check_component "Spark Deequ Validator" "data-platform" "app=spark-deequ" 1
check_component "Data Quality Exporter" "data-platform" "app=data-quality-exporter" 1
echo

echo "## Infrastructure"
echo
check_component "PostgreSQL" "data-platform" "app=postgres-shared" 1
check_component "Kafka" "data-platform" "app=kafka" 1
check_component "MinIO" "data-platform" "app=minio" 1
check_component "Elasticsearch" "data-platform" "app=elasticsearch" 1
echo

echo "## GPU Operator"
echo
gpu_pods=$(kubectl get pods -n gpu-operator --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$gpu_pods" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} GPU Operator: $gpu_pods pods deployed"
    gpu_running=$(kubectl get pods -n gpu-operator --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    echo "  - Running: $gpu_running/$gpu_pods"
else
    echo -e "${YELLOW}⚠${NC} GPU Operator: Not installed"
fi
echo

echo "## API Keys Configuration"
echo
if kubectl get secret seatunnel-api-keys -n data-platform &>/dev/null; then
    echo -e "${GREEN}✓${NC} SeaTunnel API Keys secret exists"
    keys=$(kubectl get secret seatunnel-api-keys -n data-platform -o jsonpath='{.data}' | jq -r 'keys[]' 2>/dev/null | grep -E "FRED|EIA|NOAA" | wc -l)
    echo "  - Configured keys: $keys (FRED, EIA, NOAA)"
else
    echo -e "${RED}✗${NC} SeaTunnel API Keys secret not found"
fi
echo

echo "## Workflows & Dashboards"
echo
if kubectl get configmap dolphinscheduler-commodity-workflows -n data-platform &>/dev/null; then
    workflows=$(kubectl get configmap dolphinscheduler-commodity-workflows -n data-platform -o jsonpath='{.data}' | jq 'keys | length' 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} DolphinScheduler workflows: $workflows configured"
else
    echo -e "${RED}✗${NC} DolphinScheduler workflows ConfigMap not found"
fi

if kubectl get configmap superset-commodity-dashboards -n data-platform &>/dev/null; then
    dashboards=$(kubectl get configmap superset-commodity-dashboards -n data-platform -o jsonpath='{.data}' | jq 'keys | length' 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} Superset dashboards: $dashboards configured"
else
    echo -e "${RED}✗${NC} Superset dashboards ConfigMap not found"
fi
echo

echo "## Service Accessibility"
echo
services=(
    "https://portal.254carbon.com:Portal"
    "https://dolphinscheduler.254carbon.com:DolphinScheduler"
    "https://superset.254carbon.com:Superset"
    "https://grafana.254carbon.com:Grafana"
    "https://datahub.254carbon.com:DataHub"
)

for svc in "${services[@]}"; do
    IFS=':' read -r url name <<< "$svc"
    if curl -k -s -o /dev/null -w "%{http_code}" "$url" --max-time 5 | grep -q "200\|302\|401"; then
        echo -e "${GREEN}✓${NC} $name accessible"
    else
        echo -e "${YELLOW}⚠${NC} $name: Check DNS/Cloudflare"
    fi
done
echo

echo "## Resource Utilization"
echo
kubectl top nodes 2>/dev/null || echo "Metrics server not available"
echo

echo "=================================="
echo "Verification Complete"
echo "=================================="
echo
echo "Next Steps:"
echo "1. Import workflows in DolphinScheduler"
echo "2. Import dashboards in Superset"
echo "3. Run first data ingestion"
echo
echo "Documentation: COMMODITY_QUICKSTART.md"

