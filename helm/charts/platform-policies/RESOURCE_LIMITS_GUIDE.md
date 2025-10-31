# Resource Limits and Quotas Guide

## Overview

This chart configures resource limits, quotas, and autoscaling policies for all namespaces in the platform. It provides guardrails to prevent resource exhaustion while allowing flexibility for legitimate high-resource workloads.

## Components

### 1. ResourceQuotas
- Enforce **namespace-level** aggregate resource limits
- Prevent any single namespace from consuming all cluster resources
- Applied to: `data-platform`, `monitoring`, `vault-prod`, `ingress-nginx`, `cert-manager`

### 2. LimitRanges
- Enforce **per-pod and per-container** resource limits
- Provide default resource requests/limits for containers that don't specify them
- Prevent individual pods from requesting excessive resources
- Applied to same namespaces as ResourceQuotas

### 3. Horizontal Pod Autoscalers (HPAs)
- Automatically scale user-facing services based on CPU/memory utilization
- Configured for: API Gateway, Portal Services, DataHub, MLflow, Superset, Trino

### 4. PriorityClasses
- Ensure critical services (databases, brokers) are scheduled before batch jobs
- Three tiers: `critical-services` (1000), `standard-services` (500), `low-priority-batch` (100)

## Escape Hatch: Overriding LimitRange Restrictions

LimitRanges are **guardrails**, not absolute blocks. You can override them when needed:

### Method 1: Explicit Resource Specifications
Set explicit `resources.requests` and `resources.limits` in your Deployment/Pod spec:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-gpu-workload
  namespace: data-platform
spec:
  template:
    spec:
      containers:
      - name: app
        image: my-image:latest
        resources:
          requests:
            cpu: "16"      # Exceeds LimitRange container max of 8 CPU
            memory: "32Gi" # Exceeds LimitRange container max of 16Gi
          limits:
            cpu: "16"
            memory: "32Gi"
```

**Important:** Your explicit requests/limits must still fit within the namespace's **ResourceQuota**. If they don't, contact the platform team.

### Method 2: Namespace-Specific Adjustments
For persistent high-resource workloads (e.g., GPU training, large-scale ETL):

1. Open an issue with the platform team
2. Include:
   - Workload description and justification
   - Required CPU/memory/GPU resources
   - Expected duration (temporary vs. permanent)
3. Platform team will create a custom LimitRange for your namespace

### Method 3: Disable LimitRange (Emergency Only)
In extreme cases, you can temporarily disable LimitRange enforcement:

```bash
kubectl delete limitrange <limitrange-name> -n <namespace>
```

**⚠️ Warning:** This removes all guardrails. Only use in emergencies and restore immediately after.

## Checking Current Limits

### View ResourceQuota
```bash
kubectl describe resourcequota -n data-platform
```

### View LimitRange
```bash
kubectl describe limitrange -n data-platform
```

### Check if Pod was Rejected
```bash
kubectl describe pod <pod-name> -n <namespace>
# Look for events like "exceeded quota" or "minimum memory usage"
```

## Common Issues and Solutions

### Issue: Pod rejected with "minimum memory usage per Container is 32Mi"
**Solution:** Add explicit `resources.requests.memory` to your container spec (minimum 32Mi).

### Issue: Pod rejected with "maximum memory usage per Container is 16Gi"
**Solution:** For legitimate high-memory workloads, set explicit `resources.limits.memory` in your spec. If it exceeds 16Gi, ensure it fits within the namespace ResourceQuota.

### Issue: HPA not scaling up
**Solution:** Ensure your deployment has `resources.requests` set. HPA requires these to calculate utilization percentages.

### Issue: "forbidden: exceeded quota"
**Solution:** The namespace ResourceQuota is full. Either:
1. Scale down other workloads in the namespace
2. Request a quota increase from the platform team

## Best Practices

1. **Always set resources**: Don't rely on LimitRange defaults for production workloads
2. **Right-size your requests**: Set `requests` to typical usage, `limits` to burst capacity
3. **Monitor usage**: Use `kubectl top pods` to validate your resource settings
4. **Use HPAs**: Enable autoscaling for variable-load services
5. **Set PriorityClass**: Use `critical-services` for databases, `low-priority-batch` for jobs

## Example Configurations

### Small Service (API, worker)
```yaml
resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

### Medium Service (web app, cache)
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Large Service (database, query engine)
```yaml
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

### GPU Workload (ML training)
```yaml
resources:
  requests:
    cpu: 8000m
    memory: 32Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 16000m
    memory: 64Gi
    nvidia.com/gpu: 1
```

## Contact

For questions or quota increase requests, contact the platform team via:
- Slack: `#platform-support`
- Email: `platform-team@254carbon.com`
- GitHub Issues: Tag with `resource-quota`
