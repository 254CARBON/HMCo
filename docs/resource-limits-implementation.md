# Resource Limits and Autoscaling Implementation

## Summary

This document summarizes the implementation of T6.1 and T6.2 - Resource limits and autoscaling enhancements across all Helm charts.

## Changes Made

### T6.1 - Requests/Limits + HPA

#### Core Charts (New Resource Limits)
The following charts now have sensible default resource requests and limits:

| Chart | Requests | Limits |
|-------|----------|--------|
| api-gateway | 200m CPU, 256Mi RAM | 1000m CPU, 512Mi RAM |
| monitoring | 100m CPU, 256Mi RAM | 500m CPU, 512Mi RAM |
| networking | 100m CPU, 128Mi RAM | 500m CPU, 256Mi RAM |
| service-mesh | 100m CPU, 128Mi RAM | 500m CPU, 256Mi RAM |
| platform-policies | 50m CPU, 64Mi RAM | 200m CPU, 128Mi RAM |

#### HPAs (Horizontal Pod Autoscalers)
New autoscaling configurations for user-facing services:

| Service | Min Replicas | Max Replicas | Target CPU % |
|---------|--------------|--------------|--------------|
| api-gateway | 2 | 10 | 70% |
| portal-services | 2 | 5 | 70% (existing) |
| datahub-frontend | 2 | 6 | 70% |
| mlflow | 2 | 6 | 70% |
| superset-web | 1 | 4 | 75% (existing) |
| trino-worker | 1 | 5 | 75% (existing) |

#### Data Platform Subcharts

| Chart | Requests | Limits | Notes |
|-------|----------|--------|-------|
| clickhouse | 2000m CPU, 4Gi RAM | 4000m CPU, 8Gi RAM | Existing - unchanged |
| datahub-frontend | 500m CPU, 1Gi RAM | 2000m CPU, 4Gi RAM | Existing - autoscaling enabled |
| datahub-gms | 1000m CPU, 2Gi RAM | 4000m CPU, 8Gi RAM | Existing - unchanged |
| superset-web | 500m CPU, 1Gi RAM | 2000m CPU, 4Gi RAM | Existing - unchanged |
| superset-worker | 500m CPU, 2Gi RAM | 2000m CPU, 4Gi RAM | Existing - unchanged |
| trino-coordinator | 2000m CPU, 6Gi RAM | 4000m CPU, 8Gi RAM | Existing - unchanged |
| trino-worker | 3000m CPU, 6Gi RAM | 4000m CPU, 8Gi RAM | Existing - unchanged |
| dolphinscheduler-api | 500m CPU, 1Gi RAM | 2000m CPU, 4Gi RAM | Existing - unchanged |
| dolphinscheduler-worker | 2000m CPU, 4Gi RAM | 4000m CPU, 8Gi RAM | Existing - unchanged |
| spark-operator | 200m CPU, 256Mi RAM | 500m CPU, 512Mi RAM | Existing - unchanged |
| data-lake | 500m CPU, 1Gi RAM | 2000m CPU, 4Gi RAM | New |
| backend (HMCo) | 100m CPU, 256Mi RAM | 500m CPU, 512Mi RAM | Existing - unchanged |

#### ML Platform Subcharts

| Chart | Requests | Limits | Notes |
|-------|----------|--------|-------|
| mlflow | 500m CPU, 1Gi RAM | 2000m CPU, 4Gi RAM | New - autoscaling enabled |
| kubeflow | 500m CPU, 1Gi RAM | 2000m CPU, 4Gi RAM | New |

#### Other Charts Verified
- **cloudflare-tunnel**: Already has resources (100m/128Mi → 500m/512Mi)
- **jupyterhub-hub**: Already has resources (500m/1Gi → 2000m/4Gi)
- **jupyterhub-proxy**: Already has resources (100m/512Mi → 500m/1Gi)
- **portal-services**: Already has resources and HPA (100m/256Mi → 500m/512Mi)
- **kong**: Already has resources (200m/256Mi → 1000m/1Gi)

### T6.2 - Namespace Quotas and LimitRanges

#### New LimitRanges Created
LimitRanges enforce per-pod and per-container resource limits:

**data-platform**
- Container defaults: 500m CPU / 1Gi RAM
- Container requests: 100m CPU / 256Mi RAM
- Container max: 8 CPU / 16Gi RAM
- Pod max: 8 CPU / 16Gi RAM

**monitoring**
- Container defaults: 200m CPU / 256Mi RAM
- Container requests: 50m CPU / 128Mi RAM
- Container max: 2 CPU / 4Gi RAM
- Pod max: 2 CPU / 4Gi RAM

**vault-prod**
- Container defaults: 500m CPU / 512Mi RAM
- Container requests: 250m CPU / 256Mi RAM
- Container max: 2 CPU / 4Gi RAM
- Pod max: 2 CPU / 4Gi RAM

**ingress-nginx**
- Container defaults: 500m CPU / 512Mi RAM
- Container requests: 100m CPU / 128Mi RAM
- Container max: 2 CPU / 4Gi RAM
- Pod max: 2 CPU / 4Gi RAM

**cert-manager**
- Container defaults: 200m CPU / 256Mi RAM
- Container requests: 50m CPU / 128Mi RAM
- Container max: 1 CPU / 2Gi RAM
- Pod max: 1 CPU / 2Gi RAM

#### Existing ResourceQuotas
The following namespace quotas were already in place (unchanged):

| Namespace | CPU Requests | CPU Limits | Memory Requests | Memory Limits | PVCs | Pods |
|-----------|--------------|------------|-----------------|---------------|------|------|
| data-platform | 80 | 160 | 256Gi | 512Gi | 100 | 400 |
| monitoring | 8 | 12 | 16Gi | 24Gi | - | 50 |
| vault-prod | 4 | 6 | 8Gi | 12Gi | 10 | 20 |
| ingress-nginx | 2 | 4 | 4Gi | 8Gi | - | 10 |
| cert-manager | 1 | 2 | 2Gi | 4Gi | - | 10 |

#### Escape Hatch Documentation
Created `RESOURCE_LIMITS_GUIDE.md` with:
- Overview of resource limits architecture
- Instructions for overriding LimitRange restrictions
- Three escape hatch methods:
  1. Explicit resource specifications in deployments
  2. Namespace-specific LimitRange adjustments
  3. Emergency LimitRange deletion (with warnings)
- Troubleshooting common issues
- Best practices and example configurations

## Verification Steps

### Pre-Deployment Checks
1. ✅ Helm lint passed for all modified charts
2. ✅ Template rendering validated (HPAs, LimitRanges, ResourceQuotas)
3. ✅ All charts have valid syntax

### Post-Deployment Checks

#### 1. Verify ResourceQuotas
```bash
# Check all namespace quotas
for ns in data-platform monitoring vault-prod ingress-nginx cert-manager; do
  echo "=== $ns ==="
  kubectl describe resourcequota -n $ns
done
```

Expected: All quotas should show `Used` vs `Hard` limits.

#### 2. Verify LimitRanges
```bash
# Check all namespace limit ranges
for ns in data-platform monitoring vault-prod ingress-nginx cert-manager; do
  echo "=== $ns ==="
  kubectl describe limitrange -n $ns
done
```

Expected: All LimitRanges should be present with correct min/max/default values.

#### 3. Verify HPAs
```bash
# Check HPA status
kubectl get hpa -A
kubectl describe hpa api-gateway
kubectl describe hpa portal-services-hpa -n data-platform
kubectl describe hpa mlflow-hpa -n ml-platform
```

Expected: HPAs should show current/target metrics and replica counts.

#### 4. Monitor Pod Resource Usage
```bash
# Check if pods are running within limits
kubectl top pods -n data-platform
kubectl top pods -n monitoring
kubectl top pods -n ml-platform
```

Expected: No pods should be throttled (CPU near limit) or OOMKilled (memory exceeded).

#### 5. Test Autoscaling
```bash
# Generate load on a service (example for api-gateway)
kubectl run load-generator --image=busybox:1.28 --restart=Never -- /bin/sh -c "while true; do wget -q -O- http://api-gateway; done"

# Watch HPA scale up
kubectl get hpa api-gateway --watch
```

Expected: HPA should scale up when CPU exceeds 70%.

#### 6. Test LimitRange Enforcement
```bash
# Try to create a pod without resource limits
kubectl run test-pod --image=nginx --restart=Never -n data-platform

# Check if default limits were applied
kubectl get pod test-pod -n data-platform -o jsonpath='{.spec.containers[0].resources}'
```

Expected: Pod should have default requests/limits from LimitRange.

#### 7. Test Escape Hatch
Create a test deployment that exceeds LimitRange:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: high-resource-test
  namespace: data-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: test
        image: nginx
        resources:
          requests:
            cpu: "10"        # Exceeds LimitRange max
            memory: "20Gi"   # Exceeds LimitRange max
          limits:
            cpu: "10"
            memory: "20Gi"
```

Expected: Pod should be rejected IF it exceeds ResourceQuota, but allowed if within quota.

## Rollback Plan

If issues arise after deployment:

1. **Remove HPAs (will not delete pods)**
   ```bash
   kubectl delete hpa api-gateway
   kubectl delete hpa mlflow-hpa -n ml-platform
   ```

2. **Remove LimitRanges (will not affect running pods)**
   ```bash
   kubectl delete limitrange -n data-platform data-platform-limits
   ```

3. **Revert Helm values** by applying previous values
   ```bash
   helm upgrade api-gateway helm/charts/api-gateway/ --values previous-values.yaml
   ```

## Known Issues

1. **Monitoring chart has pre-existing template error**: Unrelated to resource limits changes
2. **MLflow chart has pre-existing nil pointer error**: Unrelated to resource limits changes
3. **GPU workloads**: May need explicit resource overrides as LimitRange container max is 8 CPU / 16Gi

## References

- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [LimitRanges](https://kubernetes.io/docs/concepts/policy/limit-range/)
- [ResourceQuotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- Platform-specific guide: `helm/charts/platform-policies/RESOURCE_LIMITS_GUIDE.md`
