# Seldon Core - Advanced Model Serving

**Platform**: 254Carbon Advanced Analytics Platform  
**Component**: Production Model Serving  
**Technology**: Seldon Core 1.17  
**Status**: Implementation Phase 2.2

---

## Overview

Seldon Core provides production-grade model serving with advanced deployment patterns:

- **A/B Testing**: Compare multiple model versions
- **Canary Deployments**: Gradual rollout of new models
- **Shadow Mode**: Test models without impacting production
- **Multi-Model Ensembles**: Combine multiple models
- **Explainability**: Integrated SHAP/LIME explanations
- **MLflow Integration**: Deploy directly from MLflow registry

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Kong API Gateway / Istio Service Mesh                     │
└─────────────────────┬──────────────────────────────────────┘
                      │
                      │ Traffic Routing
                      ↓
┌────────────────────────────────────────────────────────────┐
│  Seldon Deployment (Kubernetes CRD)                        │
│  ┌──────────────────┬──────────────────┬──────────────┐  │
│  │  Predictor A     │  Predictor B     │  Shadow      │  │
│  │  (90% traffic)   │  (10% traffic)   │  (no traffic)│  │
│  └────────┬─────────┴────────┬─────────┴──────┬───────┘  │
└───────────┼──────────────────┼─────────────────┼───────────┘
            │                  │                 │
            ↓                  ↓                 ↓
┌───────────────────┐  ┌───────────────────┐  ┌─────────────┐
│  Model Pods (3x)  │  │  Model Pods (1x)  │  │ Shadow Pod  │
│  MLflow Model v1  │  │  MLflow Model v2  │  │ Exp Model   │
└───────────────────┘  └───────────────────┘  └─────────────┘
            │                  │                 │
            └──────────────────┴─────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────┐
│  Monitoring & Metrics (Prometheus, Grafana)                │
│  - Request latency                                         │
│  - Model accuracy                                          │
│  - Traffic distribution                                    │
└────────────────────────────────────────────────────────────┘
```

## Deployment Patterns

### 1. Simple Single Model Deployment

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: my-model
  namespace: data-platform
spec:
  name: my-model
  replicas: 3
  predictors:
  - name: default
    graph:
      name: classifier
      implementation: MLFLOW_SERVER
      modelUri: models:/my_model/production
```

### 2. A/B Testing (50/50 split)

```yaml
spec:
  predictors:
  - name: model-a
    traffic: 50
    graph:
      modelUri: models:/my_model/1
  - name: model-b
    traffic: 50
    graph:
      modelUri: models:/my_model/2
```

### 3. Canary Deployment (10% canary)

```yaml
spec:
  predictors:
  - name: main
    traffic: 90
    replicas: 3
    graph:
      modelUri: models:/my_model/production
  - name: canary
    traffic: 10
    replicas: 1
    graph:
      modelUri: models:/my_model/staging
```

### 4. Shadow Deployment

```yaml
spec:
  predictors:
  - name: main
    graph:
      modelUri: models:/my_model/production
  - name: shadow
    shadow: true  # Receives traffic but doesn't serve responses
    graph:
      modelUri: models:/my_model/experimental
```

### 5. Model Ensemble

```yaml
spec:
  predictors:
  - name: ensemble
    graph:
      name: combiner
      type: AVERAGE_COMBINER
      children:
      - name: model-1
        implementation: MLFLOW_SERVER
        modelUri: models:/rf_model/production
      - name: model-2
        implementation: MLFLOW_SERVER
        modelUri: models:/gb_model/production
```

## Deployment

### Prerequisites

```bash
# Ensure MLflow is running
kubectl get pods -n data-platform -l app=mlflow

# Ensure Istio is deployed
kubectl get pods -n istio-system

# Create MLflow credentials secret
kubectl create secret generic mlflow-credentials -n data-platform \
  --from-literal=AWS_ACCESS_KEY_ID=minio \
  --from-literal=AWS_SECRET_ACCESS_KEY=minio123 \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=http://minio-service:9000
```

### Deploy Seldon Core

```bash
# 1. Deploy Seldon Operator
kubectl apply -f k8s/ml-platform/seldon-core/seldon-operator.yaml

# 2. Wait for operator to be ready
kubectl wait --for=condition=ready pod -l app=seldon-controller-manager -n seldon-system --timeout=300s

# 3. Deploy example models
kubectl apply -f k8s/ml-platform/seldon-core/seldon-deployment-example.yaml

# 4. Verify deployments
kubectl get sdep -n data-platform
```

### Verify Installation

```bash
# Check Seldon operator
kubectl get pods -n seldon-system

# Check model deployments
kubectl get sdep -n data-platform

# Check model pods
kubectl get pods -n data-platform -l seldon-deployment-id

# Get deployment status
kubectl describe sdep commodity-price-predictor -n data-platform
```

## Usage

### Deploy Model from MLflow

```bash
# 1. Train and register model in MLflow
python train_model.py  # Registers model in MLflow

# 2. Create Seldon deployment
cat <<EOF | kubectl apply -f -
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: my-commodity-model
  namespace: data-platform
spec:
  name: my-commodity-model
  predictors:
  - name: default
    replicas: 2
    graph:
      name: classifier
      implementation: MLFLOW_SERVER
      modelUri: models:/my_commodity_model/production
      envSecretRefName: mlflow-credentials
EOF
```

### Make Predictions

```bash
# Port-forward to model service
kubectl port-forward -n data-platform svc/commodity-price-predictor-default 8080:8000

# Make prediction
curl -X POST http://localhost:8080/api/v1.0/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "ndarray": [[75.5, 3.2, 1.18]]
    }
  }'

# Response
{
  "data": {
    "names": [],
    "ndarray": [78.3]
  },
  "meta": {}
}
```

### Python Client

```python
from seldon_core.seldon_client import SeldonClient
import numpy as np

# Connect to Seldon deployment
client = SeldonClient(
    gateway="istio",
    gateway_endpoint="seldon-gateway.istio-system:80",
    namespace="data-platform",
    deployment_name="commodity-price-predictor"
)

# Make prediction
data = np.array([[75.5, 3.2, 1.18]])
response = client.predict(data=data)

print(f"Prediction: {response.response['data']['ndarray']}")
```

### Gradual Canary Rollout

```bash
# Start with 5% canary
kubectl patch sdep commodity-predictor-canary -n data-platform --type='json' \
  -p='[{"op": "replace", "path": "/spec/predictors/1/traffic", "value": 5}]'

# Monitor metrics, then increase to 25%
kubectl patch sdep commodity-predictor-canary -n data-platform --type='json' \
  -p='[{"op": "replace", "path": "/spec/predictors/1/traffic", "value": 25}]'

# If all good, promote to 100%
kubectl patch sdep commodity-predictor-canary -n data-platform --type='json' \
  -p='[{"op": "replace", "path": "/spec/predictors/1/traffic", "value": 100}]'
```

## Model Explainability

Enable SHAP explanations:

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: explainable-model
  namespace: data-platform
spec:
  name: explainable-model
  predictors:
  - name: default
    graph:
      name: model
      implementation: MLFLOW_SERVER
      modelUri: models:/commodity_price_predictor/production
    explainer:
      type: AnchorTabular
      containerSpec:
        name: explainer
        image: seldonio/alibi-explain-server:1.17.1
```

Get explanation:

```bash
curl -X POST http://localhost:8080/api/v1.0/explanations \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "ndarray": [[75.5, 3.2, 1.18]]
    }
  }'
```

## Monitoring

### Prometheus Metrics

Seldon exposes metrics:
- `seldon_api_executor_server_requests_seconds`: Request latency
- `seldon_api_executor_client_requests_seconds`: Model latency
- `seldon_deployment_prediction_count`: Prediction count

### Grafana Dashboard

Import the Seldon Core dashboard:

```bash
# Dashboard ID: 14634 (Seldon Core Analytics)
```

Key metrics:
- Request rate per model
- Prediction latency (P50, P95, P99)
- Error rate
- Traffic distribution (for A/B tests)

### Model Performance Tracking

```python
from prometheus_client import Counter, Histogram

# Track predictions
prediction_counter = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_version', 'outcome']
)

# Track accuracy (requires ground truth)
def track_prediction(model_version, prediction, actual):
    is_correct = abs(prediction - actual) < threshold
    prediction_counter.labels(
        model_version=model_version,
        outcome='correct' if is_correct else 'incorrect'
    ).inc()
```

## Integration with Kong

Add Seldon deployments to Kong for external access:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongService
metadata:
  name: commodity-predictor
  namespace: kong
spec:
  host: commodity-price-predictor-default.data-platform.svc.cluster.local
  port: 8000
  protocol: http
---
apiVersion: configuration.konghq.com/v1
kind: KongRoute
metadata:
  name: ml-predictions
  namespace: kong
spec:
  service: commodity-predictor
  paths:
  - /ml/predict/commodity-prices
```

External access:
```
https://api.254carbon.com/ml/predict/commodity-prices
```

## Best Practices

1. **Start Conservative**: Begin with 5-10% canary traffic
2. **Monitor Closely**: Track latency, errors, and business metrics
3. **Automate Rollback**: Set up alerts for automatic rollback on errors
4. **Version Everything**: Tag models with git commits and data versions
5. **Load Testing**: Test canary under production load before full rollout
6. **Documentation**: Document model versions and deployment history
7. **Explainability**: Enable explanations for high-stakes predictions

## Troubleshooting

### Model Not Loading

```bash
# Check Seldon deployment logs
kubectl logs -n data-platform -l seldon-deployment-id=commodity-price-predictor

# Check MLflow connectivity
kubectl exec -n data-platform <pod-name> -- \
  curl http://mlflow.data-platform.svc.cluster.local:5000/health

# Verify model exists in MLflow
kubectl exec -n data-platform <pod-name> -- \
  curl http://mlflow.data-platform.svc.cluster.local:5000/api/2.0/mlflow/registered-models/get?name=commodity_price_predictor
```

### High Latency

```bash
# Check resource usage
kubectl top pods -n data-platform -l seldon-deployment-id

# Check Prometheus metrics
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Query: rate(seldon_api_executor_server_requests_seconds_sum[5m])

# Scale up replicas
kubectl patch sdep commodity-price-predictor -n data-platform --type='json' \
  -p='[{"op": "replace", "path": "/spec/replicas", "value": 5}]'
```

### Traffic Not Splitting

```bash
# Check Istio virtual service
kubectl get virtualservice -n data-platform

# Verify predictor traffic weights
kubectl get sdep commodity-predictor-ab-test -n data-platform -o yaml | grep traffic

# Check Istio routing
istioctl proxy-config routes <pod-name> -n data-platform
```

## Next Steps

- [ ] Implement automated canary analysis
- [ ] Add model drift detection
- [ ] Create deployment pipelines from Kubeflow
- [ ] Set up automated rollback on errors
- [ ] Integrate with DataHub for model lineage
- [ ] Build model comparison dashboards

## Resources

- **Seldon Core Docs**: https://docs.seldon.io/
- **MLflow Integration**: https://docs.seldon.io/projects/seldon-core/en/latest/servers/mlflow.html
- **Deployment Patterns**: https://docs.seldon.io/projects/seldon-core/en/latest/graph/annotations.html



