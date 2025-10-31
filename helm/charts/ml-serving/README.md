# ML Serving Helm Chart

This Helm chart deploys a complete ML model serving infrastructure with:

- **KServe**: Model serving platform for machine learning models
- **Argo Rollouts**: Progressive delivery with canary deployments
- **SLO Monitoring**: Latency and availability Service Level Objectives
- **Autoscaling**: HPA with CPU, memory, and custom metrics
- **Metrics**: Prometheus integration for monitoring

## Features

### Model Serving with KServe
- Deploy ML models with industry-standard runtimes (sklearn, pytorch, tensorflow, etc.)
- Automatic scaling based on traffic
- Request/response logging and monitoring

### Canary Deployments with Argo Rollouts
- Progressive traffic shifting (20% → 40% → 60% → 80% → 100%)
- Automated analysis during rollout
- Automatic rollback on failure
- Pause points for manual validation

### Service Level Objectives (SLOs)
- **Latency SLO**: P95 < 500ms, P99 < 1s
- **Availability SLO**: 99.9% uptime
- Multi-window, multi-burn rate alerting
- Error budget tracking

### Autoscaling
- Horizontal Pod Autoscaling based on:
  - CPU utilization (70% target)
  - Memory utilization (80% target)
  - Custom metrics (requests per second)
- Smart scale-up/scale-down policies

## Installation

### Prerequisites

1. Install KServe:
```bash
helm repo add kserve https://kserve.github.io/charts
helm repo update
```

2. Install Argo Rollouts:
```bash
helm repo add argo https://argoproj.github.io/argo-helm
helm install argo-rollouts argo/argo-rollouts -n argo-rollouts --create-namespace
```

3. Install Prometheus Operator (for metrics):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

### Deploy the Chart

```bash
# Install with default example model
helm install ml-serving ./helm/charts/ml-serving -n ml-serving --create-namespace

# Install with custom values
helm install ml-serving ./helm/charts/ml-serving -n ml-serving --create-namespace -f custom-values.yaml

# Upgrade existing deployment
helm upgrade ml-serving ./helm/charts/ml-serving -n ml-serving
```

## Configuration

### Example Model Configuration

The chart includes an example sklearn iris model that demonstrates all features:

```yaml
exampleModel:
  enabled: true
  name: sklearn-iris
  runtime: kserve-sklearn
  storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
```

### Canary Deployment Configuration

```yaml
exampleModel:
  canary:
    enabled: true
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
      enabled: true
      successRate:
        threshold: 99  # 99% success rate required
      latencyP95:
        threshold: 500  # 500ms p95 latency required
```

### SLO Configuration

```yaml
slo:
  enabled: true
  latency:
    enabled: true
    p95Threshold: 0.5  # seconds
    p99Threshold: 1.0  # seconds
  availability:
    enabled: true
    target: 0.999  # 99.9%
    errorBudget: 0.001
```

### Autoscaling Configuration

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
  customMetrics:
    enabled: true
    requestsPerSecond:
      targetValue: 100
```

## Usage

### Testing the Model

Once deployed, you can send inference requests:

```bash
# Get the service endpoint
kubectl get inferenceservice sklearn-iris -n ml-serving

# Send a prediction request
curl -X POST http://sklearn-iris.ml-serving.svc.cluster.local/v1/models/sklearn-iris:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```

### Monitoring

View metrics in Prometheus:
```bash
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Open http://localhost:9090
```

View SLO alerts:
```bash
kubectl get prometheusrules -n ml-serving
```

### Trigger a Canary Deployment

Update the model version or configuration:

```bash
helm upgrade ml-serving ./helm/charts/ml-serving -n ml-serving \
  --set exampleModel.storageUri="gs://kfserving-examples/models/sklearn/2.0/model"
```

Watch the rollout progress:
```bash
kubectl argo rollouts get rollout sklearn-iris-predictor -n ml-serving --watch
```

### Manual Promotion/Abort

```bash
# Promote canary to stable
kubectl argo rollouts promote sklearn-iris-predictor -n ml-serving

# Abort canary and rollback
kubectl argo rollouts abort sklearn-iris-predictor -n ml-serving
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Istio Gateway                        │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              VirtualService (Traffic Split)              │
│           Stable: 80%  │  Canary: 20%                   │
└──────────┬────────────────────────┬─────────────────────┘
           │                        │
           ▼                        ▼
    ┌──────────┐            ┌──────────┐
    │  Stable  │            │  Canary  │
    │  Service │            │  Service │
    └────┬─────┘            └────┬─────┘
         │                       │
         ▼                       ▼
    ┌──────────┐            ┌──────────┐
    │ Stable   │            │ Canary   │
    │ Pods     │            │ Pods     │
    │ (HPA)    │            │ (HPA)    │
    └──────────┘            └──────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Prometheus │
              │  (Metrics)  │
              └─────────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Analysis    │
              │ Template    │
              │ (validates) │
              └─────────────┘
```

## Troubleshooting

### Check deployment status
```bash
kubectl get inferenceservice -n ml-serving
kubectl get rollout -n ml-serving
kubectl get hpa -n ml-serving
```

### View logs
```bash
kubectl logs -n ml-serving -l serving.kserve.io/inferenceservice=sklearn-iris
```

### Check SLO metrics
```bash
kubectl get prometheusrules -n ml-serving ml-serving-slo-alerts -o yaml
```

### Debug canary issues
```bash
kubectl argo rollouts get rollout sklearn-iris-predictor -n ml-serving
kubectl argo rollouts status sklearn-iris-predictor -n ml-serving
```

## Values

See `values.yaml` for all configuration options.

## DoD Checklist

- ✅ One model served (sklearn-iris example)
- ✅ Canary deploy with Argo Rollouts
- ✅ Latency SLO enforced (P95 < 500ms, P99 < 1s)
- ✅ Availability SLO enforced (99.9% uptime)
- ✅ Metrics collection via Prometheus
- ✅ Autoscaling with HPA
