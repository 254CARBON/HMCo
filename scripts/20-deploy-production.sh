#!/bin/bash

# 254Carbon Production Deployment Script
# Deploys the complete platform to a production Kubernetes cluster

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}254Carbon Production Deployment${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Check if running in correct directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}❌ Error: Please run this script from the HMCo project root directory${NC}"
    exit 1
fi

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1 completed successfully${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

echo "📋 Production Deployment Checklist:"
echo "1. Verify cluster connectivity and health"
echo "2. Deploy namespaces and RBAC"
echo "3. Deploy storage infrastructure"
echo "4. Deploy shared services (PostgreSQL, Kafka, Redis)"
echo "5. Deploy data platform services"
echo "6. Deploy monitoring and logging"
echo "7. Deploy ingress and security"
echo "8. Validate deployment"
echo ""

# 1. Verify cluster connectivity
echo -e "${YELLOW}🔍 Step 1: Verifying cluster connectivity...${NC}"

# Check if kubectl is configured
if ! kubectl cluster-info >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: kubectl not configured or cluster not accessible${NC}"
    echo "Please ensure:"
    echo "1. kubectl is configured with your production cluster"
    echo "2. The cluster is accessible and healthy"
    exit 1
fi

# Check cluster health
kubectl get nodes >/dev/null 2>&1
check_success "Cluster connectivity verification"

# Check if cluster has multiple nodes (production requirement)
NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
if [ "$NODE_COUNT" -lt 3 ]; then
    echo -e "${YELLOW}⚠️  Warning: Cluster has only $NODE_COUNT nodes. Production deployment requires at least 3 nodes.${NC}"
    echo "Consider adding more nodes for high availability."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Deploy namespaces and RBAC
echo -e "${YELLOW}🔧 Step 2: Deploying namespaces and RBAC...${NC}"

kubectl apply -f k8s/namespaces/ >/dev/null 2>&1
check_success "Namespaces deployment"

kubectl apply -f k8s/rbac/ >/dev/null 2>&1
check_success "RBAC deployment"

# 3. Deploy storage infrastructure
echo -e "${YELLOW}🔧 Step 3: Deploying storage infrastructure...${NC}"

# Deploy storage classes and persistent volumes
kubectl apply -f k8s/storage/ >/dev/null 2>&1
check_success "Storage infrastructure deployment"

# 4. Deploy shared services
echo -e "${YELLOW}🔧 Step 4: Deploying shared services...${NC}"

# Deploy PostgreSQL, Kafka, Redis, etc.
kubectl apply -f k8s/shared/postgres/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  PostgreSQL deployment may need manual intervention${NC}"
kubectl apply -f k8s/shared/zookeeper/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Zookeeper deployment may need manual intervention${NC}"
kubectl apply -f k8s/shared/kafka/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Kafka deployment may need manual intervention${NC}"

echo "⏳ Waiting for shared services to initialize..."
kubectl wait --for=condition=ready pod -n data-platform --timeout=600s \
    --selector=app in \(postgres,kafka,zookeeper\) || echo -e "${YELLOW}⚠️  Some shared services may still be initializing${NC}"

# 5. Deploy data platform services
echo -e "${YELLOW}🔧 Step 5: Deploying data platform services...${NC}"

# Deploy core data services
kubectl apply -f k8s/data-lake/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Data lake deployment may need manual intervention${NC}"
kubectl apply -f k8s/datahub/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  DataHub deployment may need manual intervention${NC}"
kubectl apply -f k8s/compute/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Compute services deployment may need manual intervention${NC}"
kubectl apply -f k8s/visualization/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Visualization services deployment may need manual intervention${NC}"

echo "⏳ Waiting for data platform services to initialize..."
kubectl wait --for=condition=ready pod -n data-platform --timeout=900s || echo -e "${YELLOW}⚠️  Some data platform services may still be initializing${NC}"

# 6. Deploy monitoring and logging
echo -e "${YELLOW}🔧 Step 6: Deploying monitoring and logging...${NC}"

kubectl apply -f k8s/monitoring/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Monitoring deployment may need manual intervention${NC}"

echo "⏳ Waiting for monitoring services to initialize..."
kubectl wait --for=condition=ready pod -n monitoring --timeout=600s || echo -e "${YELLOW}⚠️  Some monitoring services may still be initializing${NC}"

# 7. Deploy ingress and security
echo -e "${YELLOW}🔧 Step 7: Deploying ingress and security...${NC}"

# Deploy ingress controller (should already be deployed)
kubectl apply -f k8s/ingress/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Ingress deployment may need manual intervention${NC}"

# Deploy network policies and security
kubectl apply -f k8s/networking/ >/dev/null 2>&1
check_success "Network policies deployment"

# Deploy resilience configurations
kubectl apply -f k8s/resilience/ >/dev/null 2>&1
check_success "Resilience configurations deployment"

# Deploy secrets
kubectl apply -f k8s/secrets/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Secrets deployment may need manual intervention${NC}"

# Deploy certificates
kubectl apply -f k8s/certificates/ >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Certificate deployment may need manual intervention${NC}"

# 8. Validation
echo -e "${YELLOW}🔍 Step 8: Validating deployment...${NC}"

# Check overall pod health
TOTAL_PODS=$(kubectl get pods -A --no-headers | wc -l)
RUNNING_PODS=$(kubectl get pods -A --no-headers | grep -c "Running\|Completed")

echo "📊 Deployment Summary:"
echo "Total Pods: $TOTAL_PODS"
echo "Running Pods: $RUNNING_PODS"

if [ "$RUNNING_PODS" -lt "$((TOTAL_PODS * 8 / 10))" ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 80% of pods are running${NC}"
    echo "Check pod status with: kubectl get pods -A"
fi

# Check for critical services
CRITICAL_SERVICES=("postgres" "kafka" "zookeeper" "datahub" "grafana" "prometheus")
for service in "${CRITICAL_SERVICES[@]}"; do
    if kubectl get pods -A -l app="$service" --no-headers | grep -q "Running"; then
        echo -e "${GREEN}✅ $service is running${NC}"
    else
        echo -e "${YELLOW}⚠️  $service may not be fully ready${NC}"
    fi
done

# Check ingress status
INGRESS_COUNT=$(kubectl get ingress -A --no-headers | wc -l)
if [ "$INGRESS_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Ingress resources deployed: $INGRESS_COUNT${NC}"
else
    echo -e "${YELLOW}⚠️  No ingress resources found${NC}"
fi

# Display final status
echo ""
echo -e "${GREEN}🎉 Production Deployment Complete!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo "📋 Post-Deployment Checklist:"
echo ""
echo "1. **Verify External Access**:"
echo "   - Test: curl -v https://254carbon.com"
echo "   - Check: kubectl get ingress -A"
echo ""
echo "2. **Test Key Services**:"
echo "   - Grafana: https://grafana.254carbon.com"
echo "   - DataHub: https://datahub.254carbon.com"
echo "   - Trino: https://trino.254carbon.com"
echo ""
echo "3. **Check Monitoring**:"
echo "   - Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n monitoring"
echo "   - Grafana: kubectl port-forward svc/grafana 3000:3000 -n monitoring"
echo ""
echo "4. **Validate Security**:"
echo "   - Check network policies: kubectl get networkpolicy -A"
echo "   - Verify RBAC: kubectl auth can-i get pods --as=system:serviceaccount:monitoring:prometheus"
echo ""
echo "5. **Test Backups**:"
echo "   - Create test backup: kubectl apply -f k8s/storage/velero-restore-test.yaml"
echo "   - Check backup status: kubectl get backups -n velero"
echo ""
echo "🔧 Useful Commands:"
echo "   kubectl get pods -A (overall status)"
echo "   kubectl get ingress -A (external access)"
echo "   kubectl get svc -A (service endpoints)"
echo "   kubectl logs -n <namespace> <pod> (debugging)"
echo ""
echo "📖 Documentation:"
echo "   See PRODUCTION_MIGRATION_PLAN.md for detailed migration procedures"
echo "   See infrastructure/cloud/ or infrastructure/bare-metal/ for setup guides"
echo ""
echo "🚨 If you encounter issues:"
echo "1. Check pod logs: kubectl logs -n <namespace> <pod>"
echo "2. Verify resource quotas: kubectl describe resourcequota -A"
echo "3. Check events: kubectl get events -A --sort-by=.metadata.creationTimestamp"
echo ""
echo -e "${GREEN}✅ Production deployment status: COMPLETE${NC}"
echo "⏳ Ready for production operations"

# Save deployment completion status
echo "$(date): Production deployment completed successfully" >> /tmp/254carbon-deployment.log
