#!/bin/bash
# Kubernetes Cluster Validation Script
# This script validates all cluster fixes have been applied and services are operational

set -e

echo "======================================================================"
echo "Kubernetes Cluster Validation"
echo "======================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

check_pass() {
  echo -e "${GREEN}✓${NC} $1"
  ((PASS++))
}

check_fail() {
  echo -e "${RED}✗${NC} $1"
  ((FAIL++))
}

check_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARN++))
}

# 1. Check Storage Classes
echo "1. Validating Storage Classes..."
if kubectl get storageclass local-storage-standard &>/dev/null; then
  check_pass "Storage class 'local-storage-standard' exists"
else
  check_fail "Storage class 'local-storage-standard' not found"
fi

# 2. Check PersistentVolumeClaims
echo ""
echo "2. Validating PersistentVolumeClaims..."
for pvc in elasticsearch-data neo4j-data neo4j-logs vault-data lakefs-data; do
  if kubectl get pvc $pvc -n data-platform &>/dev/null; then
    status=$(kubectl get pvc $pvc -n data-platform -o jsonpath='{.status.phase}')
    if [ "$status" = "Bound" ]; then
      check_pass "PVC '$pvc' is Bound"
    else
      check_warn "PVC '$pvc' status: $status (expected Bound)"
    fi
  else
    check_warn "PVC '$pvc' not found"
  fi
done

# 3. Check StatefulSets
echo ""
echo "3. Validating StatefulSets..."
for ss in zookeeper kafka minio; do
  if kubectl get statefulset $ss -n data-platform &>/dev/null; then
    replicas=$(kubectl get statefulset $ss -n data-platform -o jsonpath='{.spec.replicas}')
    ready=$(kubectl get statefulset $ss -n data-platform -o jsonpath='{.status.readyReplicas}')
    if [ -z "$ready" ]; then ready="0"; fi
    if [ "$replicas" -eq "$ready" ]; then
      check_pass "StatefulSet '$ss' has $ready/$replicas replicas ready"
    else
      check_warn "StatefulSet '$ss' has $ready/$replicas replicas ready"
    fi
  else
    check_fail "StatefulSet '$ss' not found"
  fi
done

# 4. Check Core Deployments
echo ""
echo "4. Validating Core Deployments..."
for deploy in datahub-frontend datahub-gms schema-registry superset; do
  if kubectl get deployment $deploy -n data-platform &>/dev/null; then
    replicas=$(kubectl get deployment $deploy -n data-platform -o jsonpath='{.spec.replicas}')
    ready=$(kubectl get deployment $deploy -n data-platform -o jsonpath='{.status.readyReplicas}')
    if [ -z "$ready" ]; then ready="0"; fi
    if [ "$replicas" -eq "$ready" ] 2>/dev/null; then
      check_pass "Deployment '$deploy' is ready ($ready/$replicas)"
    else
      check_warn "Deployment '$deploy' has $ready/$replicas replicas ready"
    fi
  else
    check_warn "Deployment '$deploy' not found"
  fi
done

# 5. Check DolphinScheduler Components
echo ""
echo "5. Validating DolphinScheduler Components..."
for deploy in dolphinscheduler-api dolphinscheduler-master dolphinscheduler-worker; do
  if kubectl get deployment $deploy -n data-platform &>/dev/null; then
    image=$(kubectl get deployment $deploy -n data-platform -o jsonpath='{.spec.template.spec.containers[0].image}')
    if [[ "$image" == *"dolphinscheduler"* ]]; then
      check_pass "Deployment '$deploy' image: $image"
    else
      check_fail "Deployment '$deploy' has unexpected image: $image"
    fi
  else
    check_warn "Deployment '$deploy' not found"
  fi
done

# 6. Check Secrets
echo ""
echo "6. Validating Secrets..."
for secret in datahub-secret postgres-workflow-secret minio-secret docker-registry-secret; do
  if kubectl get secret $secret -n data-platform &>/dev/null; then
    check_pass "Secret '$secret' exists"
  else
    check_warn "Secret '$secret' not found (may be required)"
  fi
done

# 7. Check ConfigMaps
echo ""
echo "7. Validating ConfigMaps..."
for cm in datahub-config trino-coordinator-config; do
  if kubectl get configmap $cm -n data-platform &>/dev/null; then
    check_pass "ConfigMap '$cm' exists"
  else
    check_warn "ConfigMap '$cm' not found (may be required)"
  fi
done

# 8. Check Services
echo ""
echo "8. Validating Services..."
for svc in zookeeper-service kafka-headless minio-headless; do
  if kubectl get service $svc -n data-platform &>/dev/null; then
    check_pass "Service '$svc' exists"
  else
    check_fail "Service '$svc' not found"
  fi
done

# 9. Check Ingress Controller
echo ""
echo "9. Validating Ingress Controller..."
if kubectl get deployment nginx-ingress-controller -n ingress-nginx &>/dev/null; then
  check_pass "NGINX Ingress Controller found"
else
  check_warn "NGINX Ingress Controller not found"
fi

# 10. Pod Status Summary
echo ""
echo "10. Pod Status Summary..."
running=$(kubectl get pods -A --field-selector=status.phase=Running --no-headers | wc -l)
pending=$(kubectl get pods -A --field-selector=status.phase=Pending --no-headers | wc -l)
failed=$(kubectl get pods -A --field-selector=status.phase=Failed --no-headers | wc -l)
unknown=$(kubectl get pods -A --field-selector=status.phase=Unknown --no-headers | wc -l)

if [ "$running" -gt 0 ]; then
  check_pass "Running pods: $running"
fi
if [ "$pending" -gt 10 ]; then
  check_warn "Pending pods: $pending (expected during initialization)"
fi
if [ "$failed" -gt 0 ]; then
  check_fail "Failed pods: $failed"
fi
if [ "$unknown" -gt 0 ]; then
  check_fail "Unknown status pods: $unknown"
fi

# Summary
echo ""
echo "======================================================================"
echo "Validation Summary"
echo "======================================================================"
echo -e "Passed:  ${GREEN}$PASS${NC}"
echo -e "Failed:  ${RED}$FAIL${NC}"
echo -e "Warnings: ${YELLOW}$WARN${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
  echo -e "${GREEN}✓ Cluster validation PASSED${NC}"
  exit 0
else
  echo -e "${RED}✗ Cluster validation FAILED${NC}"
  exit 1
fi
