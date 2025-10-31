# Progressive Delivery with Argo Rollouts

This Helm chart deploys Argo Rollouts controller and CRDs for progressive delivery capabilities including canary deployments with automated analysis.

## Features

- **Argo Rollouts Controller**: Manages progressive delivery strategies
- **Canary Deployments**: Gradual rollout with traffic shifting
- **Automated Analysis**: Metric-based promotion/rollback using Prometheus
- **Analysis Templates**: Pre-configured templates for success rate, error rate, and latency monitoring

## Installation

```bash
helm install progressive-delivery ./helm/charts/progressive-delivery \
  --namespace argo-rollouts \
  --create-namespace
```

## Configuration

Key configuration options in `values.yaml`:

- `argoRollouts.replicaCount`: Number of controller replicas (default: 2)
- `analysisTemplates.successRate.threshold`: Minimum success rate % (default: 95)
- `analysisTemplates.errorRate.threshold`: Maximum error rate % (default: 5)
- `analysisTemplates.latency.threshold`: Maximum p95 latency in ms (default: 1000)
- `prometheus.address`: Prometheus server address for metrics

## Usage

### Converting a Deployment to Rollout

Replace your Deployment with a Rollout resource:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: my-app
spec:
  replicas: 3
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 2m}
      - setWeight: 40
      - pause: {duration: 2m}
      - setWeight: 60
      - pause: {duration: 2m}
      - setWeight: 80
      - pause: {duration: 2m}
      analysis:
        templates:
        - templateName: success-rate
        - templateName: error-rate
        - templateName: latency
        args:
        - name: service-name
          value: my-app
  selector:
    matchLabels:
      app: my-app
  template:
    # ... pod template ...
```

### Monitoring Rollouts

```bash
# Watch rollout status
kubectl argo rollouts get rollout my-app -n data-platform --watch

# List all rollouts
kubectl argo rollouts list rollouts -n data-platform

# Promote a rollout manually
kubectl argo rollouts promote my-app -n data-platform

# Abort a rollout
kubectl argo rollouts abort my-app -n data-platform
```

## Analysis Templates

Three analysis templates are provided:

1. **success-rate**: Monitors HTTP 2xx response rate
2. **error-rate**: Monitors HTTP 5xx error rate
3. **latency**: Monitors p95 response latency

These templates query Prometheus and automatically promote or rollback deployments based on metric thresholds.

## Integration with Existing Services

The portal-services and MLflow charts have been updated to use Rollout resources with canary analysis enabled by default.
