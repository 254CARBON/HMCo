# Progressive Delivery with Argo Rollouts

## Overview

The HMCo platform implements progressive delivery using Argo Rollouts to enable safe, automated deployments with canary analysis and automatic rollback capabilities.

### Key Features

- **Canary Deployments**: Gradual traffic shifting from stable to new version
- **Automated Analysis**: Metric-based promotion/rollback using Prometheus queries
- **Risk Mitigation**: Automatic rollback on metric failures
- **Zero-Downtime Deployments**: Seamless transitions between versions
- **Manual Controls**: Ability to pause, promote, or abort rollouts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Argo Rollouts Controller                  │
│                  (Manages Rollout Resources)                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─────────────────────────────────────────┐
                 │                                         │
        ┌────────▼────────┐                    ┌──────────▼──────────┐
        │  Stable Version  │                    │   Canary Version    │
        │   ReplicaSet     │                    │    ReplicaSet       │
        │   (Previous)     │                    │      (New)          │
        └────────┬────────┘                    └──────────┬──────────┘
                 │                                         │
                 └──────────┬──────────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   Service       │
                   │ (Traffic Split) │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │   Prometheus    │
                   │  (Metrics)      │
                   └─────────────────┘
                            │
                   ┌────────▼────────┐
                   │ AnalysisRuns    │
                   │  (Validation)   │
                   └─────────────────┘
```

## Components

### 1. Progressive Delivery Chart

Location: `helm/charts/progressive-delivery/`

Deploys:
- Argo Rollouts controller
- CRDs (Rollout, AnalysisTemplate, AnalysisRun, Experiment)
- RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding)
- Analysis templates for common metrics

### 2. Updated Service Charts

Both portal-services and MLflow charts have been updated to support Rollouts:
- `helm/charts/portal-services/` - Portal Services API
- `helm/charts/ml-platform/charts/mlflow/` - MLflow Tracking Server

## Quick Start

### Installation

```bash
# Install via ArgoCD (automatic)
kubectl apply -f k8s/gitops/argocd-applications.yaml

# Or install manually
helm install progressive-delivery ./helm/charts/progressive-delivery \
  --namespace argo-rollouts \
  --create-namespace
```

### Deploy with Canary

```bash
# Portal Services
helm upgrade --install portal-services ./helm/charts/portal-services \
  --set rollout.enabled=true \
  --set image.tag=1.0.1

# MLflow
helm upgrade --install mlflow ./helm/charts/ml-platform/charts/mlflow \
  --set rollout.enabled=true \
  --set global.vault.enabled=false
```

### Monitor Rollout

```bash
# Install kubectl plugin
curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
chmod +x kubectl-argo-rollouts-linux-amd64
sudo mv kubectl-argo-rollouts-linux-amd64 /usr/local/bin/kubectl-argo-rollouts

# Watch rollout
kubectl argo rollouts get rollout portal-services -n data-platform --watch
```

## Configuration

### Canary Strategy

```yaml
rollout:
  enabled: true
  strategy:
    canary:
      maxSurge: "25%"        # Allow 25% extra pods during rollout
      maxUnavailable: 0       # No pods can be unavailable
      steps:
      - setWeight: 20         # Send 20% traffic to canary
      - pause: {duration: 2m} # Wait 2 minutes
      - setWeight: 40         # Send 40% traffic to canary
      - pause: {duration: 2m}
      - setWeight: 60
      - pause: {duration: 2m}
      - setWeight: 80
      - pause: {duration: 2m}
      analysis:
        templates:
        - templateName: success-rate  # Check HTTP 2xx rate
        - templateName: error-rate    # Check HTTP 5xx rate
        - templateName: latency       # Check response time
        args:
        - name: service-name
          value: portal-services
```

### Analysis Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Success Rate | ≥ 95% | Percentage of successful requests (2xx) |
| Error Rate | ≤ 5% | Percentage of server errors (5xx) |
| Latency P95 | ≤ 1000ms | 95th percentile response time |

All metrics are checked every 60 seconds. If 3 consecutive checks fail, the rollout automatically rolls back.

## Usage

### View Rollout Status

```bash
# List all rollouts
kubectl argo rollouts list rollouts -n data-platform

# Get detailed status
kubectl argo rollouts get rollout portal-services -n data-platform

# Watch in real-time
kubectl argo rollouts get rollout portal-services -n data-platform --watch
```

### Manual Control

```bash
# Promote to next step
kubectl argo rollouts promote portal-services -n data-platform

# Abort and rollback
kubectl argo rollouts abort portal-services -n data-platform

# Retry a failed rollout
kubectl argo rollouts retry rollout portal-services -n data-platform
```

### Dashboard

```bash
kubectl argo rollouts dashboard -n data-platform
```

Open http://localhost:3100 in your browser.

## Testing

See detailed test scenarios in:
- `helm/charts/progressive-delivery/TESTING.md`

Quick test:
```bash
# Deploy initial version
helm upgrade --install portal-services ./helm/charts/portal-services \
  --set rollout.enabled=true \
  --set image.tag=1.0.0

# Trigger canary
helm upgrade portal-services ./helm/charts/portal-services \
  --set rollout.enabled=true \
  --set image.tag=1.0.1

# Watch progression
kubectl argo rollouts get rollout portal-services -n data-platform --watch
```

## Troubleshooting

### Rollout Stuck

```bash
# Check rollout events
kubectl describe rollout portal-services -n data-platform

# Check analysis runs
kubectl get analysisrun -n data-platform

# View analysis details
kubectl describe analysisrun <name> -n data-platform
```

### Metrics Not Available

Ensure your application:
1. Exposes Prometheus metrics at `/metrics`
2. Has `http_requests_total` counter with `status` label
3. Has `http_request_duration_milliseconds_bucket` histogram
4. ServiceMonitor is configured

### Automatic Rollback Not Working

Check:
1. Analysis templates exist: `kubectl get analysistemplate -n data-platform`
2. Prometheus is accessible: `kubectl get svc -n monitoring`
3. Metrics are being collected: Query Prometheus UI

## Best Practices

1. **Start Conservative**: Begin with small canary percentages (10-20%)
2. **Adequate Pauses**: Allow 2-5 minutes for metric collection
3. **Test in Staging**: Always test canary deployments in non-prod first
4. **Monitor Business Metrics**: Don't rely solely on technical metrics
5. **Have Rollback Plan**: Be prepared to manually abort if needed

## Integration

### ArgoCD

Rollouts work seamlessly with ArgoCD:
- ArgoCD manages Helm releases
- Argo Rollouts manages deployment strategy
- Both systems coordinate automatically

### CI/CD

Example GitHub Actions:
```yaml
- name: Deploy with Canary
  run: |
    helm upgrade portal-services ./helm/charts/portal-services \
      --set image.tag=${{ github.sha }} \
      --set rollout.enabled=true \
      --wait --timeout=20m
```

## Additional Documentation

- [Architecture Details](./architecture.md) - Coming soon
- [Testing Guide](../../helm/charts/progressive-delivery/TESTING.md)
- [Argo Rollouts Official Docs](https://argo-rollouts.readthedocs.io/)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs: `kubectl logs -n argo-rollouts -l app.kubernetes.io/name=argo-rollouts`
3. Contact platform team
