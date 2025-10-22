# MLflow Quick Reference

## Access MLflow

**Internal** (from within cluster):
```
http://mlflow.data-platform.svc.cluster.local:5000
```

**External** (via Cloudflare):
```
https://mlflow.254carbon.com
```

**Port Forward** (for testing):
```bash
kubectl port-forward -n data-platform svc/mlflow 5000:5000
# Then access: http://localhost:5000
```

---

## Quick Start

### Python Client

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri('http://mlflow.data-platform.svc.cluster.local:5000')

# Create experiment
mlflow.set_experiment("my_experiment")

# Track run
with mlflow.start_run():
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("metric1", 0.85)
    mlflow.sklearn.log_model(model, "model")
```

### DolphinScheduler Integration

```python
from mlflow_client import setup_mlflow_for_dolphinscheduler

client = setup_mlflow_for_dolphinscheduler("production_model")
client.start_run("training_run")
client.log_params({"lr": 0.01})
client.log_metrics({"accuracy": 0.95})
client.log_model(model, "model", flavor="sklearn")
client.end_run()
```

---

## Deployment Info

**Status**: ✅ 2/2 pods Running  
**Backend**: PostgreSQL (postgres-shared-service)  
**Artifacts**: MinIO S3 (mlflow-artifacts bucket)  
**Monitoring**: Prometheus ServiceMonitor active

---

## Test Deployment

```bash
# Run example tracking script
python3 examples/mlflow/simple_tracking.py

# Check pod status
kubectl get pods -n data-platform -l app=mlflow

# View logs
kubectl logs -n data-platform -l app=mlflow

# Verify health
kubectl port-forward -n data-platform svc/mlflow 5000:5000 &
curl http://localhost:5000/health
```

---

## Files & Scripts

**Deployment Scripts**:
- `scripts/deploy-mlflow.sh` - One-command deployment
- `scripts/init-mlflow-postgres.sh` - Database setup
- `scripts/create-mlflow-minio-bucket.sh` - Bucket creation

**K8s Manifests**:
- `k8s/compute/mlflow/mlflow-deployment.yaml` - Main deployment
- `k8s/compute/mlflow/mlflow-service.yaml` - Service
- `k8s/compute/mlflow/mlflow-ingress.yaml` - External access
- `k8s/compute/mlflow/mlflow-servicemonitor.yaml` - Monitoring
- `k8s/compute/mlflow/mlflow-secrets.yaml` - Credentials
- `k8s/compute/mlflow/mlflow-configmap.yaml` - Configuration

**Integration**:
- `services/mlflow-orchestration/mlflow_client.py` - DolphinScheduler client
- `examples/mlflow/simple_tracking.py` - Test example

---

## Troubleshooting

**Pods not ready?**
```bash
kubectl logs -n data-platform -l app=mlflow
```

**Connection issues?**
```bash
kubectl exec -n data-platform mlflow-7cc888cf5b-vkhq4 -- \
  python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:5000/health').read())"
```

**Re-deploy?**
```bash
kubectl delete deployment mlflow -n data-platform
./scripts/deploy-mlflow.sh
```

---

## Next Steps

1. Access UI: https://mlflow.254carbon.com (once tunnel is healthy)
2. Run test: `python3 examples/mlflow/simple_tracking.py`
3. Create ML workflows in DolphinScheduler with MLflow tracking
4. View experiments and models in MLflow UI
5. Set up model deployment pipelines

---

**Documentation**: See `k8s/compute/mlflow/README.md` for full details


