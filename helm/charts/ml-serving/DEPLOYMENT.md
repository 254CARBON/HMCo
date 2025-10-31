# ML Serving Deployment Guide

## Overview

This chart provides a complete ML model serving infrastructure with:
- **KServe** for model deployment and inference
- **Argo Rollouts** for progressive canary deployments
- **Service Level Objectives (SLOs)** for latency and availability monitoring
- **Horizontal Pod Autoscaling** for automatic scaling based on load

## Architecture

The chart is self-contained and does not have external Helm dependencies. Instead, it assumes that the required components (KServe and Argo Rollouts) are already installed in the cluster.

### Components Deployed

1. **Namespace**: `ml-serving` with Istio injection enabled
2. **InferenceService**: KServe CRD for model deployment
3. **Rollout**: Argo Rollouts CRD for canary deployment strategy
4. **Services**: Stable and canary services for traffic splitting
5. **VirtualService**: Istio configuration for traffic routing
6. **AnalysisTemplate**: Metrics-based validation during canary rollout
7. **HorizontalPodAutoscaler**: Auto-scaling based on CPU, memory, and custom metrics
8. **PrometheusRules**: SLO recording rules and alerts
9. **ServiceMonitor**: Prometheus metrics collection

## Prerequisites

Before deploying this chart, ensure the following are installed:

### 1. KServe (v0.11.0 or later)

```bash
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-runtimes.yaml
```

Verify installation:
```bash
kubectl get crd | grep kserve
# Should show: inferenceservices.serving.kserve.io
```

### 2. Argo Rollouts (v1.6.0 or later)

```bash
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
```

Verify installation:
```bash
kubectl get crd | grep rollouts
# Should show: rollouts.argoproj.io
```

### 3. Prometheus Operator

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

### 4. Istio Service Mesh

This platform already has Istio installed. Verify:
```bash
kubectl get pods -n istio-system
```

## Installation

### Quick Start

```bash
# Install with default example model (sklearn-iris)
helm install ml-serving ./helm/charts/ml-serving -n ml-serving --create-namespace

# Check deployment
kubectl get inferenceservice -n ml-serving
kubectl get rollout -n ml-serving
kubectl get hpa -n ml-serving
```

### Production Deployment

```bash
# Install with production values
helm install ml-serving ./helm/charts/ml-serving \
  -n ml-serving \
  --create-namespace \
  -f ./helm/charts/ml-serving/values/prod.yaml

# Or via ArgoCD
kubectl apply -f k8s/gitops/argocd-applications.yaml
```

## Configuration

### Feature Flags

The chart uses feature flags to enable/disable components:

```yaml
features:
  kserve: true              # Enable KServe InferenceService
  canaryDeployment: true    # Enable Argo Rollouts canary
  sloMonitoring: true       # Enable SLO metrics and alerts
  autoscaling: true         # Enable HPA
```

### Example Model

The chart includes an example sklearn iris model:

```yaml
exampleModel:
  enabled: true
  name: sklearn-iris
  runtime: kserve-sklearn
  storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
```

To deploy your own model, disable the example and create your own InferenceService.

### SLO Thresholds

```yaml
slo:
  latency:
    p95Threshold: 0.5  # 500ms
    p99Threshold: 1.0  # 1000ms
  availability:
    target: 0.999      # 99.9%
```

## Validation

After deployment, validate that all components are working:

```bash
# 1. Check all resources
kubectl get all -n ml-serving

# 2. Verify InferenceService is ready
kubectl get inferenceservice sklearn-iris -n ml-serving
# STATUS should be "True"

# 3. Check Rollout status
kubectl argo rollouts get rollout sklearn-iris-predictor -n ml-serving

# 4. Verify HPA
kubectl get hpa -n ml-serving
# Should show current/target metrics

# 5. Check PrometheusRules
kubectl get prometheusrules -n ml-serving

# 6. Check ServiceMonitor
kubectl get servicemonitor -n ml-serving
```

## Testing

Send a test inference request:

```bash
# Port-forward the service
kubectl port-forward -n ml-serving svc/sklearn-iris-predictor-stable 8080:80

# Send prediction request (in another terminal)
curl -X POST http://localhost:8080/v1/models/sklearn-iris:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```

Expected response:
```json
{
  "predictions": [0]
}
```

## Monitoring

### View Metrics in Prometheus

```bash
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Open http://localhost:9090
```

Useful queries:
- Inference request rate: `sum(rate(inference_request_total[5m])) by (service, model)`
- P95 latency: `sli:inference_latency_p95:5m`
- Error rate: `sli:inference_error_rate:5m`

### View Alerts

```bash
kubectl get prometheusrules -n ml-serving ml-serving-slo-alerts -o yaml
```

## Canary Deployment

When you update the model, Argo Rollouts will automatically perform a canary deployment:

```bash
# Update model version
helm upgrade ml-serving ./helm/charts/ml-serving \
  --set exampleModel.storageUri="gs://kfserving-examples/models/sklearn/2.0/model"

# Watch the rollout
kubectl argo rollouts get rollout sklearn-iris-predictor -n ml-serving --watch

# Manually promote if needed
kubectl argo rollouts promote sklearn-iris-predictor -n ml-serving

# Abort and rollback if needed
kubectl argo rollouts abort sklearn-iris-predictor -n ml-serving
```

## Troubleshooting

### InferenceService Not Ready

```bash
kubectl describe inferenceservice sklearn-iris -n ml-serving
kubectl logs -n ml-serving -l serving.kserve.io/inferenceservice=sklearn-iris
```

### Rollout Stuck

```bash
kubectl argo rollouts get rollout sklearn-iris-predictor -n ml-serving
kubectl argo rollouts status sklearn-iris-predictor -n ml-serving
```

### HPA Not Scaling

```bash
kubectl describe hpa sklearn-iris-predictor-hpa -n ml-serving
kubectl top pods -n ml-serving
```

## Cleanup

```bash
# Uninstall the chart
helm uninstall ml-serving -n ml-serving

# Delete the namespace
kubectl delete namespace ml-serving
```

## Definition of Done Checklist

- ✅ One model served (sklearn-iris example included)
- ✅ Canary deployment with Argo Rollouts configured
- ✅ Latency SLO enforced (P95 < 500ms, P99 < 1s)
- ✅ Availability SLO enforced (99.9% uptime)
- ✅ Metrics collection via Prometheus ServiceMonitor
- ✅ Autoscaling with HPA (CPU, memory, custom metrics)
- ✅ Integration tests included
- ✅ Documentation complete

## Next Steps

1. Deploy your own models by creating additional InferenceService resources
2. Customize SLO thresholds based on your requirements
3. Configure alerting destinations in AlertManager
4. Set up Grafana dashboards for visualization
5. Integrate with your CI/CD pipeline for automated model deployments
