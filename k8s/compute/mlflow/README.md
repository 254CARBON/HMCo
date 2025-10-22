# MLFlow Deployment Guide

**Status**: ✅ Deployed and Operational (October 22, 2025)

This directory contains Kubernetes manifests for deploying MLFlow tracking server as part of the 254Carbon data platform.

## Deployment Status

✅ **Backend**: PostgreSQL database initialized and connected  
✅ **Artifacts**: MinIO bucket (mlflow-artifacts) with versioning  
✅ **Deployment**: 2/2 pods running (HA configuration)  
✅ **Service**: ClusterIP on port 5000  
✅ **Ingress**: mlflow.254carbon.com configured  
✅ **Monitoring**: ServiceMonitor active (Prometheus)  
✅ **DNS**: Cloudflare CNAME record configured

## Overview

MLFlow provides a unified platform for tracking ML experiments, managing models, and implementing model registry functionality. The deployment includes:

- **Backend Store**: PostgreSQL database for metadata ✅ Deployed
- **Artifact Store**: MinIO S3-compatible storage for models and artifacts ✅ Deployed
- **HA Configuration**: 2 replicas with pod anti-affinity ✅ Running
- **External Access**: Cloudflare tunnel routing ✅ Configured
- **Monitoring**: Prometheus metrics exposure and Grafana dashboards ✅ Active

## Components

### mlflow-secrets.yaml
Kubernetes Secrets containing sensitive credentials:
- PostgreSQL connection details (mlflow user/password)
- MinIO S3 access keys
- S3 endpoint configuration

**Prerequisites**: These secrets reference the shared PostgreSQL instance running in data-platform namespace.

### mlflow-configmap.yaml
Configuration map for non-sensitive settings:
- Backend and artifact URIs
- Gunicorn server settings
- Logging configuration
- AWS/S3 endpoint URLs

### mlflow-service.yaml
ClusterIP service exposing the tracking server on port 5000 internally.

### mlflow-deployment.yaml
Main deployment manifest with:
- 2 replicas for high availability
- Pod anti-affinity for node distribution
- Health checks (liveness and readiness probes)
- Resource limits (500m-1000m CPU, 1Gi-2Gi memory)
- Security context (non-root user)
- Volume mounts for temporary files

### mlflow-backend-db.sql
PostgreSQL initialization script that:
- Creates `mlflow` user
- Creates `mlflow` database
- Sets up proper schema permissions
- Grants necessary privileges

## Setup Instructions

### Step 1: Initialize PostgreSQL Backend

Connect to the shared PostgreSQL instance and run the initialization script:

```bash
# Option A: Using kubectl exec
kubectl exec -it -n data-platform postgres-shared-<pod-id> -- \
  psql -U datahub -d postgres -f - < mlflow-backend-db.sql

# Option B: Manual execution
kubectl exec -it -n data-platform postgres-shared-<pod-id> -- psql -U datahub -d postgres
# Then paste contents of mlflow-backend-db.sql and execute
```

Verify the setup:
```bash
kubectl exec -it -n data-platform postgres-shared-<pod-id> -- \
  psql -U mlflow -d mlflow -c "\dt"
```

### Step 2: Create MinIO Bucket

Create the MLFlow artifacts bucket in MinIO:

```bash
# Forward MinIO console
kubectl port-forward -n data-platform svc/minio 9001:9001 &

# Access MinIO Console at http://localhost:9001
# Login with default credentials (minioadmin/minioadmin)
# Create bucket: mlflow-artifacts
# Set versioning: Enable
```

Alternatively, use MinIO CLI:
```bash
# Access MinIO CLI in MinIO pod
kubectl exec -it -n data-platform minio-<pod-id> -- /bin/sh
mc alias set local https://localhost:9000 minioadmin minioadmin
mc mb local/mlflow-artifacts
mc version enable local/mlflow-artifacts
```

### Step 3: Deploy MLFlow

Apply the Kubernetes manifests in order:

```bash
# Create secrets
kubectl apply -f mlflow-secrets.yaml

# Create configuration
kubectl apply -f mlflow-configmap.yaml

# Create service
kubectl apply -f mlflow-service.yaml

# Deploy tracking server
kubectl apply -f mlflow-deployment.yaml
```

### Step 4: Verify Deployment

```bash
# Check pod status
kubectl get pods -n data-platform -l app=mlflow

# Check logs
kubectl logs -n data-platform -l app=mlflow

# Port forward to test locally
kubectl port-forward -n data-platform svc/mlflow 5000:5000 &
curl http://localhost:5000/health
```

## Configuration

### Backend Store URI
The backend store uses PostgreSQL for storing experiment metadata:
```
postgresql://mlflow:mlflow-secure-password-change-me@postgres-shared-service.data-platform.svc.cluster.local:5432/mlflow
```

**Note**: Change the password in both `mlflow-secrets.yaml` and PostgreSQL setup before production deployment.

### Artifact Store URI
Artifacts are stored in MinIO S3-compatible storage:
```
s3://mlflow-artifacts
```

### S3 Configuration
- **Endpoint**: `https://minio.minio.svc.cluster.local:9000`
- **Verify SSL**: Disabled (for self-signed certs)
- **Credentials**: MinIO default admin credentials

## Production Considerations

### Security
- [ ] Change PostgreSQL password from default `mlflow-secure-password-change-me`
- [ ] Change MinIO credentials from default `minioadmin/minioadmin`
- [ ] Enable SSL verification in production
- [ ] Use Vault for credential management
- [ ] Enable network policies to restrict pod communication

### Scaling
- Increase replicas for higher load
- Monitor resource usage and adjust limits
- Use horizontal pod autoscaler (HPA)

### Backup & Recovery
- PostgreSQL: Implement automated backups
- MinIO: Enable versioning on artifacts bucket
- Disaster recovery: Document restore procedures

## Monitoring

### Prometheus Metrics
MLFlow exposes metrics on the same port (5000) at `/metrics`. Configure Prometheus scrape job:

```yaml
- job_name: 'mlflow'
  kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
          - data-platform
  relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: mlflow
```

### Health Checks
- **Liveness probe**: `/health` (pod alive?)
- **Readiness probe**: `/health` (pod ready to serve?)

## Troubleshooting

### Pod won't start
```bash
# Check logs
kubectl logs -n data-platform <mlflow-pod-name>

# Common issues:
# 1. PostgreSQL connection: Verify credentials and network
# 2. MinIO connection: Verify S3 endpoint and credentials
# 3. Permission issues: Check pod security context

# Check events
kubectl describe pod -n data-platform <mlflow-pod-name>
```

### Experiment tracking fails
```bash
# Verify MLFlow is accessible
kubectl port-forward -n data-platform svc/mlflow 5000:5000
curl http://localhost:5000/health

# Check tracking URI in client:
# MLFLOW_TRACKING_URI=http://mlflow.data-platform.svc.cluster.local:5000
```

### S3 upload errors
```bash
# Verify MinIO bucket exists
kubectl exec -it -n data-platform minio-<pod-id> -- \
  mc ls local/mlflow-artifacts

# Check S3 credentials in MLFlow pod
kubectl exec -n data-platform <mlflow-pod-name> -- env | grep AWS
```

## Integration Points

### DolphinScheduler
Set tracking URI in workflow jobs:
```python
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow.data-platform.svc.cluster.local:5000'
```

### DataHub
MLFlow models can be ingested as DataHub assets via scheduled recipes.

### Portal
MLFlow is registered in the service catalog and accessible via `mlflow.254carbon.com`.

## Next Steps

1. Deploy ingress rule for external access
2. Configure Cloudflare Access SSO
3. Set up DolphinScheduler integration
4. Create DataHub ingestion recipes
5. Deploy monitoring dashboards
6. Document ML workflow best practices

## Files

| File | Purpose |
|------|---------|
| `mlflow-backend-db.sql` | PostgreSQL schema initialization |
| `mlflow-secrets.yaml` | Credentials for backend and artifact storage |
| `mlflow-configmap.yaml` | Configuration settings |
| `mlflow-service.yaml` | Kubernetes service exposure |
| `mlflow-deployment.yaml` | Main deployment with HA config |
| `README.md` | This file |
