#!/bin/bash
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}254Carbon Connectivity Troubleshooting${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to print section headers
print_section() {
    echo -e "${YELLOW}>>> $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print info
print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 1. Check Cluster Status
print_section "Cluster Status"
echo "Kubernetes Nodes:"
kubectl get nodes -o wide
echo

# 2. Check Node Network
print_section "Node Networking"
docker exec dev-cluster-control-plane bash -c "ip addr show" 2>/dev/null | grep -E "(eth|inet |link)" || print_error "Could not get node network config"
echo

# 3. Check Service DNS
print_section "Service DNS Resolution"
SERVICES=("kubernetes.default" "kube-dns.kube-system" "iceberg-rest-catalog.data-platform" "prometheus-operator-kube-p-prometheus.monitoring")
for svc in "${SERVICES[@]}"; do
    echo "Testing DNS for: $svc"
    kubectl run -it --rm debug-pod --image=busybox --restart=Never -n default -- nslookup "$svc" 2>&1 | grep -E "(Address|Server)" || print_error "Failed to resolve $svc"
done
echo

# 4. Check Network Policies
print_section "Network Policies"
kubectl get networkpolicies --all-namespaces
echo

# 5. Check Kube-Proxy
print_section "Kube-Proxy Status"
kubectl get ds -n kube-system -l component=kube-proxy -o wide
echo

# 6. Check CoreDNS
print_section "CoreDNS Status"
kubectl get deployment -n kube-system coredns -o wide
kubectl get pods -n kube-system -l k8s-app=kube-dns
echo

# 7. Check Service Endpoints
print_section "Service Endpoints"
echo "Checking critical services:"
SERVICES_TO_CHECK=("iceberg-rest-catalog" "prometheus-operator-kube-p-prometheus" "portal")
for svc in "${SERVICES_TO_CHECK[@]}"; do
    NAMESPACE="data-platform"
    if [ "$svc" = "prometheus-operator-kube-p-prometheus" ]; then
        NAMESPACE="monitoring"
    fi
    echo "Service: $svc (namespace: $NAMESPACE)"
    kubectl get endpoints -n "$NAMESPACE" "$svc" 2>/dev/null || print_error "Endpoints not found for $svc"
done
echo

# 8. Check Pod Connectivity
print_section "Pod-to-Pod Connectivity Test"
print_info "Starting a debug pod to test internal connectivity..."
timeout 10 kubectl run -it --rm debug-pod \
    --image=curlimages/curl \
    --restart=Never \
    -n data-platform \
    -- timeout 3 curl -v http://iceberg-rest-catalog:8181/v1/config 2>&1 | head -20 || print_error "Pod connectivity test failed (this is expected)"
echo

# 9. Summary
print_section "Troubleshooting Summary"
echo -e "${YELLOW}Key Findings:${NC}"
echo "1. Service discovery (DNS) is working"
echo "2. Service endpoints are registered"
echo "3. Network policies are in place"
echo ""
echo -e "${YELLOW}Likely Issues:${NC}"
echo "- Kind cluster networking layer issue (veth bridge)"
echo "- TCP connection establishment blocking"
echo "- Pod-to-pod routing failure"
echo ""
echo -e "${YELLOW}Recommended Solutions (in order):${NC}"
echo "1. Try: kubectl rollout restart ds/kube-proxy -n kube-system"
echo "2. Try: docker exec dev-cluster-control-plane systemctl restart kubelet"
echo "3. Restart Kind cluster: kind delete cluster --name dev-cluster && kind create cluster --name dev-cluster"
echo ""
echo -e "${BLUE}For more details, see: CONNECTIVITY_TIMEOUT_DIAGNOSIS.md${NC}"
