# Resource Optimization Guide for 254Carbon Data Platform

## Overview

This guide provides best practices and procedures for optimizing resource utilization in the 254Carbon data platform.

## Current Autoscaling Configuration

### Horizontal Pod Autoscalers (HPA)

The following services have autoscaling enabled:

| Service | Min Replicas | Max Replicas | CPU Target | Memory Target |
|---------|-------------|--------------|------------|---------------|
| DataHub GMS | 2 | 5 | 70% | 80% |
| DataHub Frontend | 1 | 3 | 75% | 80% |
| Portal | 2 | 5 | 70% | 75% |
| Superset Web | 1 | 3 | 75% | 80% |
| Trino Worker | 1 | 4 | 70% | 75% |
| DolphinScheduler API | 1 | 3 | 75% | 80% |
| DolphinScheduler Worker | 2 | 6 | 70% | 75% |

### Monitoring Autoscalers

```bash
# Check HPA status
kubectl get hpa -n data-platform

# View detailed HPA metrics
kubectl describe hpa datahub-gms-hpa -n data-platform

# Check current pod resource usage
kubectl top pods -n data-platform
```

## Resource Right-Sizing Procedure

### 1. Collect Metrics

```bash
# Get 7-day average CPU usage
kubectl top pods -n data-platform --use-protocol-buffers

# For detailed historical data, query Prometheus
# Access Grafana at https://grafana.254carbon.com
```

### 2. Analyze Resource Utilization

Use the Data Platform dashboard in Grafana to review:
- CPU usage trends
- Memory usage trends
- Pod restart counts
- OOM (Out of Memory) kills

### 3. Adjust Resource Requests/Limits

#### Example: Updating DataHub GMS Resources

```yaml
resources:
  limits:
    memory: "3Gi"    # Increased from 2Gi
    cpu: "1500m"     # Increased from 1000m
  requests:
    memory: "1.5Gi"  # Increased from 1Gi
    cpu: "750m"      # Increased from 500m
```

Apply changes:
```bash
kubectl edit deployment datahub-gms -n data-platform
# Or update the YAML file and apply
kubectl apply -f k8s/datahub/datahub.yaml
```

### 4. Monitor After Changes

Monitor for at least 24-48 hours after resource changes to ensure stability.

## Best Practices

### Resource Requests vs Limits

1. **Requests**: Set based on typical usage (50-60th percentile)
2. **Limits**: Set to handle peak loads (90-95th percentile)
3. **Ratio**: Maintain a 2:1 or 3:1 ratio between limits and requests

### CPU Optimization

- **Frontend Services**: 250m-500m requests, 500m-1000m limits
- **Backend Services**: 500m-1000m requests, 1000m-2000m limits
- **Data Processing**: 1000m-2000m requests, 2000m-4000m limits

### Memory Optimization

- **Stateless Services**: 512Mi-1Gi requests, 1Gi-2Gi limits
- **Stateful Services**: 1Gi-2Gi requests, 2Gi-4Gi limits
- **Java Services**: Add 25-30% overhead for JVM

### JVM Tuning

For Java-based services (DataHub, DolphinScheduler, Trino):

```yaml
env:
- name: JAVA_OPTS
  value: "-Xms1g -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+UseStringDeduplication"
```

Guidelines:
- `-Xms`: Set to 50% of container memory request
- `-Xmx`: Set to 75-80% of container memory limit
- Use G1GC for containers with >2GB heap

## Vertical Pod Autoscaling (VPA)

### Install VPA (Optional)

```bash
# Install VPA
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# Create VPA for a service
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: datahub-gms-vpa
  namespace: data-platform
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: datahub-gms
  updatePolicy:
    updateMode: "Off"  # Recommendation only
  resourcePolicy:
    containerPolicies:
    - containerName: datahub-gms
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 2000m
        memory: 4Gi
EOF
```

## Cost Optimization

### 1. Identify Overprovisioned Resources

```bash
# Find pods with low utilization
kubectl top pods -n data-platform | awk '{if(NR>1)print $1,$2,$3}'
```

### 2. Implement Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: data-platform-quota
  namespace: data-platform
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "20"
```

### 3. Use Cluster Autoscaler

For cloud environments, enable cluster autoscaler to automatically add/remove nodes based on demand.

## Performance Tuning Checklist

- [ ] Resource requests set based on average usage
- [ ] Resource limits set for peak loads
- [ ] HPA configured for scalable services
- [ ] Pod Disruption Budgets (PDBs) in place
- [ ] Monitoring and alerts configured
- [ ] JVM tuned for Java services
- [ ] Connection pools sized appropriately
- [ ] Database queries optimized
- [ ] Caching implemented where beneficial
- [ ] Regular resource utilization reviews

## Troubleshooting

### Pods Being OOMKilled

1. Check memory usage: `kubectl top pod <pod-name> -n data-platform`
2. Review logs before crash: `kubectl logs <pod-name> -n data-platform --previous`
3. Increase memory limits if consistently hitting ceiling
4. Check for memory leaks

### CPU Throttling

1. Check CPU metrics in Grafana
2. Review throttling: `kubectl describe pod <pod-name> -n data-platform | grep -A 5 "cpu"`
3. Increase CPU limits if consistently throttled
4. Optimize code if inefficient algorithms detected

### HPA Not Scaling

1. Verify metrics-server is running: `kubectl get deployment metrics-server -n kube-system`
2. Check HPA conditions: `kubectl describe hpa <hpa-name> -n data-platform`
3. Ensure resource requests are set on pods
4. Verify targets are realistic

## Maintenance Schedule

- **Daily**: Review pod status and resource alerts
- **Weekly**: Check HPA scaling events and adjust thresholds
- **Monthly**: Full resource utilization review and optimization
- **Quarterly**: Comprehensive performance tuning and cost optimization

## References

- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [HPA Best Practices](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [JVM Performance Tuning](https://docs.oracle.com/en/java/javase/17/gctuning/)


