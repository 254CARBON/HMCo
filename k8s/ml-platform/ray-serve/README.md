# Ray Serve - Real-time ML Model Serving

**Platform**: 254Carbon Advanced Analytics Platform  
**Component**: Real-time Model Serving  
**Technology**: Ray Serve 2.9.0  
**Status**: Operator-based deployment (KubeRay) — Canonical

---

## Overview

Ray Serve provides scalable, low-latency ML model serving with:

- **Auto-scaling**: Dynamically scale based on traffic
- **MLflow Integration**: Load models directly from MLflow registry
- **Feast Integration**: Real-time feature serving
- **High Availability**: Multi-replica deployment with load balancing
- **GPU Support**: Optional GPU acceleration for inference
- **Metrics**: Prometheus metrics for monitoring

## Architecture

```
External Requests → Kong API Gateway → Ray Serve Head Node
                                            ↓
                                    Ray Serve Workers (3-10 replicas)
                                            ↓
                        ┌───────────────────┴────────────────────┐
                        ↓                                         ↓
                  MLflow Registry                          Feast Feature Store
                 (Model Loading)                         (Feature Fetching)
```

## Components

### Ray Operator
- Manages Ray cluster lifecycle
- Handles scaling and failover
- Monitors cluster health

### Ray Serve Cluster
- **Head Node**: Cluster coordinator and dashboard
- **Worker Nodes**: Model serving replicas (3-10 auto-scaling)
- **Serve Applications**: ML model serving logic

### Integration Points
- **MLflow**: Model registry and artifact storage
- **Feast**: Real-time feature serving
- **MinIO**: Model artifact storage
- **Prometheus**: Metrics and monitoring

## Deployment

### Prerequisites

```bash
# Ensure MLflow is running
kubectl get pods -n data-platform -l app=mlflow

# Ensure MinIO credentials secret exists
kubectl get secret minio-credentials -n data-platform
```

### Deploy Ray Serve

```bash
# 1. Create namespace and RBAC
kubectl apply -f k8s/ml-platform/ray-serve/namespace.yaml

# 2. Deploy Ray Operator (canonical)
kubectl apply -f k8s/ml-platform/ray-serve/namespace.yaml
kubectl apply -f k8s/ml-platform/ray-serve/ray-operator.yaml

# 3. Wait for operator to be ready
kubectl wait --for=condition=ready pod -l app=ray-operator -n ray-system --timeout=300s

# 4. Deploy Ray Serve (RayService CRD)
kubectl apply -f k8s/ml-platform/ray-serve/ray-serve-cluster.yaml

# 5. Verify deployment
kubectl get rayservice -n data-platform
kubectl get pods -n data-platform -l app=ray-serve
```

### Verify Installation

```bash
# Check Ray Serve status
kubectl get rayservice ray-serve-cluster -n data-platform

# Check pods
kubectl get pods -n data-platform -l app=ray-serve

# Port-forward to dashboard
kubectl port-forward -n data-platform svc/ray-serve-service 8265:8265

# Access dashboard at http://localhost:8265
```

## Usage

### Load Model from MLflow

```python
import requests

# Load a model
response = requests.post(
    "http://ray-serve-service.data-platform.svc.cluster.local:8000/serve",
    json={
        "model_name": "commodity_price_predictor",
        "action": "load",
        "model_version": "1"
    }
)
```

### Make Predictions

```python
# Predict with direct data
response = requests.post(
    "http://ray-serve-service.data-platform.svc.cluster.local:8000/serve",
    json={
        "model_name": "commodity_price_predictor",
        "data": {
            "crude_oil_price": 75.5,
            "natural_gas_price": 3.2,
            "exchange_rate": 1.18
        }
    }
)

print(response.json())
# Output: {"prediction": [78.3], "status": "success"}
```

### Predict with Feature Store

```python
# Fetch features from Feast and predict
response = requests.post(
    "http://ray-serve-service.data-platform.svc.cluster.local:8000/serve",
    json={
        "model_name": "commodity_price_predictor",
        "use_feature_store": true,
        "entity_id": "crude_oil_wti",
        "feature_view": "commodity_features"
    }
)
```

### Health Check

```bash
curl http://ray-serve-service.data-platform.svc.cluster.local:8000/health
```

## Monitoring

### Prometheus Metrics

```bash
# Port-forward to dashboard
kubectl port-forward -n data-platform svc/ray-serve-service 8265:8265

# Access metrics at http://localhost:8265/metrics
```

### Key Metrics

- `ray_serve_deployment_request_counter`: Total requests per deployment
- `ray_serve_deployment_latency_ms`: Request latency
- `ray_serve_deployment_replica_count`: Number of active replicas
- `ray_serve_deployment_error_counter`: Error count

### Grafana Dashboard

Import the Ray Serve dashboard (coming in Phase 4).

## Scaling

### Manual Scaling

```yaml
# Edit ray-serve-cluster.yaml
workerGroupSpecs:
- replicas: 5  # Change from 3 to 5
  minReplicas: 2
  maxReplicas: 10
```

### Auto-scaling Configuration

Auto-scaling is configured per deployment in `ray-serve-app.py`:

```python
@serve.deployment(
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
```

## Integration with Kong

Add Ray Serve to Kong API Gateway (Phase 1.2):

```bash
kubectl apply -f k8s/api-gateway/kong-services/ray-serve-service.yaml
kubectl apply -f k8s/api-gateway/kong-routes/ray-serve-routes.yaml
```

External access:
```
https://api.254carbon.com/ml/serve
```

## Troubleshooting

### Pods Not Starting

```bash
# Check operator logs
kubectl logs -n ray-system -l app=ray-operator

# Check head node logs
kubectl logs -n data-platform -l app=ray-serve,component=head

# Check events
kubectl get events -n data-platform --sort-by='.lastTimestamp'
```

### Model Loading Failures

```bash
# Check MLflow connectivity
kubectl exec -n data-platform -it <ray-pod> -- curl http://mlflow.data-platform.svc.cluster.local:5000/health

# Check MinIO access
kubectl exec -n data-platform -it <ray-pod> -- env | grep AWS

# Check logs
kubectl logs -n data-platform -l app=ray-serve,component=worker --tail=100
```

### Performance Issues

```bash
# Check resource usage
kubectl top pods -n data-platform -l app=ray-serve

# Check metrics
curl http://ray-serve-service.data-platform.svc.cluster.local:8265/metrics | grep latency

# Optional: pre-pull heavy Ray images on all nodes to avoid cold starts
kubectl apply -f k8s/ml-platform/ray-serve/ray-image-prepull.yaml -n kube-system
```

## Best Practices

1. **Model Versioning**: Always use explicit model versions in production
2. **Resource Limits**: Set appropriate CPU/memory limits based on model size
3. **Auto-scaling**: Configure based on expected traffic patterns
4. **Monitoring**: Monitor latency and error rates continuously
5. **Feature Store**: Use Feast for consistent features across training/serving
6. **Caching**: Cache frequently accessed models in memory
7. **Batching**: Enable request batching for high-throughput scenarios
8. **Operator Only**: Use KubeRay Operator with RayCluster/RayService CRDs (single source of truth). The standalone YAML is deprecated and kept for reference only.

## Next Steps

- [ ] Deploy Feast feature store (Phase 1.1)
- [ ] Integrate with Kubeflow Pipelines (Phase 2.1)
- [ ] Add A/B testing with Seldon Core (Phase 2.2)
- [ ] Enable GPU support for deep learning models
- [ ] Implement model monitoring and drift detection
 - [ ] Apply cluster policy baseline (Kyverno) for security and reliability
   - File: k8s/security/kyverno-baseline-policies.yaml

## Resources


## Deprecated

- `ray-standalone.yaml` is deprecated. The supported approach is via the KubeRay Operator and CRDs (`RayCluster`, `RayService`). Keep this file for reference only; do not apply it in production environments.

- **Ray Serve Docs**: https://docs.ray.io/en/latest/serve/
- **MLflow Integration**: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
- **Feast Integration**: https://docs.feast.dev/


