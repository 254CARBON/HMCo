#!/bin/bash
# Complete Phase 2 Deployment - Run after urgent remediation
# This script completes the Phase 2 monitoring, logging, and backup setup

set -e

echo "============================================"
echo "  254Carbon Phase 2 Completion Script"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Initialize DolphinScheduler Schema
echo -e "${YELLOW}[1/6] Initializing DolphinScheduler schema...${NC}"
if kubectl wait --for=condition=complete job/dolphinscheduler-full-schema-init -n data-platform --timeout=60s 2>/dev/null; then
  echo -e "${GREEN}✅ Schema already initialized${NC}"
else
  echo "Applying schema manually..."
  wget -q -O /tmp/dolphinscheduler_postgresql.sql \
    https://raw.githubusercontent.com/apache/dolphinscheduler/3.1.9/dolphinscheduler-dao/src/main/resources/sql/dolphinscheduler_postgresql.sql
  
  kubectl exec -n kong $(kubectl get pods -n kong -l app=postgres-temp -o jsonpath='{.items[0].metadata.name}') \
    -- psql -U postgres -d dolphinscheduler -f - < /tmp/dolphinscheduler_postgresql.sql || echo "Schema may already exist"
  
  echo -e "${GREEN}✅ Schema initialization complete${NC}"
fi

# 2. Verify DolphinScheduler Health
echo -e "${YELLOW}[2/6] Verifying DolphinScheduler health...${NC}"
kubectl wait --for=condition=ready pod -l app=dolphinscheduler-api -n data-platform --timeout=120s
echo -e "${GREEN}✅ DolphinScheduler API is ready${NC}"

# 3. Deploy Fluent Bit Logging
echo -e "${YELLOW}[3/6] Deploying Fluent Bit logging...${NC}"
if kubectl get daemonset fluent-bit -n monitoring >/dev/null 2>&1; then
  echo "Fluent Bit already deployed"
else
  kubectl apply -f k8s/logging/fluent-bit-daemonset.yaml
  echo -e "${GREEN}✅ Fluent Bit deployed${NC}"
fi

# 4. Deploy Loki
echo -e "${YELLOW}[4/6] Deploying Loki log aggregation...${NC}"
if kubectl get deployment loki -n monitoring >/dev/null 2>&1; then
  echo "Loki already deployed"
else
  kubectl apply -f k8s/logging/loki-deployment.yaml
  kubectl apply -f k8s/monitoring/loki-datasource.yaml
  echo -e "${GREEN}✅ Loki deployed${NC}"
fi

# 5. Configure Velero Backups
echo -e "${YELLOW}[5/6] Configuring Velero backups...${NC}"
# Create MinIO bucket
POD=$(kubectl get pods -n data-platform -l app=minio -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n data-platform $POD -- mc alias set minio http://localhost:9000 minioadmin minioadmin123 2>/dev/null || true
kubectl exec -n data-platform $POD -- mc mb minio/velero-backups 2>/dev/null || echo "Bucket may already exist"

# Create backup schedule
cat <<EOF | kubectl apply -f -
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-platform-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"
  template:
    includedNamespaces:
    - data-platform
    - monitoring
    - kong
    ttl: 720h0m0s
EOF

echo -e "${GREEN}✅ Velero backup schedule configured${NC}"

# 6. Verify All Services
echo -e "${YELLOW}[6/6] Verifying all services...${NC}"
echo ""
echo "Service Status:"
kubectl get pods -n data-platform -l 'app in (dolphinscheduler-api,dolphinscheduler-master,dolphinscheduler-worker)' --no-headers | \
  awk '{printf "  %-50s %s\n", $1, $3}' | grep Running | wc -l | xargs echo "  DolphinScheduler pods running:"

kubectl get pods -n data-platform -l app=trino-coordinator --no-headers | grep Running | wc -l | xargs echo "  Trino coordinator running:"
kubectl get pods -n data-platform -l app=minio --no-headers | grep Running | wc -l | xargs echo "  MinIO running:"
kubectl get pods -n data-platform -l app=superset --no-headers | grep Running | wc -l | xargs echo "  Superset components running:"
kubectl get pods -n monitoring -l app=grafana --no-headers | grep Running | wc -l | xargs echo "  Grafana running:"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Phase 2 Deployment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Access your services:"
echo "  • DolphinScheduler: https://dolphin.254carbon.com"
echo "  • Trino: https://trino.254carbon.com"
echo "  • Superset: https://superset.254carbon.com"
echo "  • Grafana: https://grafana.254carbon.com"
echo "  • MinIO: https://minio.254carbon.com"
echo ""
echo "Next: Create custom Grafana dashboards and configure alerts"

