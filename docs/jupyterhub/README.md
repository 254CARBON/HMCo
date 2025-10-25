# JupyterHub for Kubernetes on 254Carbon Platform

A comprehensive JupyterHub deployment for Kubernetes providing interactive Jupyter notebooks with full integration to all 254Carbon platform services.

## Overview

This deployment provides:

- **Multi-tenant Jupyter environment**: Individual notebook servers per user
- **Platform service integration**: Pre-configured access to Trino, MinIO, MLflow, PostgreSQL, DataHub, Ray, and Kafka
- **Cloud-native design**: Kubernetes-native spawner, persistent storage, resource quotas
- **Cloudflare Access authentication**: SSO integration for secure access
- **Monitoring & observability**: Prometheus metrics, Grafana dashboards, logging

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Cloudflare Access (SSO)                 │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│    NGINX Ingress (jupyter.254carbon.com)        │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│   JupyterHub Proxy (configurable-http-proxy)    │
│        Routes to Hub & User Servers             │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
    ┌───▼──┐    ┌────▼───┐    ┌──▼──┐
    │  Hub │    │ User 1 │    │User 2│
    │      │    │ Server │    │Server│
    └─┬──┬─┘    └────┬───┘    └──┬───┘
      │ │ │         │            │
      │ │ └─────────┼────────────┴─► Trino (SQL Queries)
      │ │           │              MinIO (Storage)
      │ │           │              MLflow (ML Tracking)
      │ │           │              PostgreSQL (Data)
      │ │           │              DataHub (Metadata)
      │ │           │              Ray (Distributed Computing)
      │ │           │              Kafka (Streaming)
      │ │           │
      │ └─────┬─────┘
      │       │
      │   Kubernetes Services
      │   - etcd (session data)
      │   - PersistentVolumes (user data)
      │   - ConfigMaps (config)
      │   - Secrets (credentials)
      │
```

## Features

### Multi-User Support
- Individual notebook servers spawned per user on demand
- Automatic cleanup when users log out
- Persistent home directories
- Resource limits per user (CPU, memory, storage)

### Platform Integration
- **Trino**: Query OLAP data with pre-configured connection
- **MinIO**: S3-compatible object storage access
- **MLflow**: Track experiments and manage ML models
- **PostgreSQL**: Relational database queries
- **DataHub**: Data governance and lineage
- **Ray**: Distributed computing and ML training
- **Kafka**: Event streaming and messaging

### Security
- Cloudflare Access SSO authentication
- Kubernetes RBAC for fine-grained permissions
- Network policies for pod-to-pod communication
- Resource quotas to prevent runaway workloads
- Istio mTLS for encrypted service communication

### Observability
- Prometheus metrics export
- Grafana dashboard for JupyterHub monitoring
- User pod metrics collection
- Audit logging for all access events

## Installation

### Prerequisites

- Kubernetes cluster (v1.24+)
- Helm 3.x
- NGINX Ingress Controller
- Cloudflare tunnel configured for 254carbon.com
- Existing platform services (Trino, MinIO, etc.)

### Deploy JupyterHub

1. **Update Helm values** (if needed)
   ```bash
   # Edit configuration
   vi helm/charts/jupyterhub/values.yaml
   ```

2. **Deploy via ArgoCD**
   ```bash
   # The application is already defined in k8s/gitops/argocd-applications.yaml
   argocd app sync jupyterhub
   ```

   Or deploy directly with Helm:
   ```bash
   cd helm/charts/jupyterhub
   helm install jupyterhub . \
     --namespace jupyter \
     --create-namespace \
     --values values.yaml
   ```

3. **Verify deployment**
   ```bash
   kubectl get all -n jupyter
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=jupyterhub -n jupyter --timeout=300s
   ```

4. **Access JupyterHub**
   - Navigate to: https://jupyter.254carbon.com
   - Authenticate via Cloudflare Access
   - Start your first notebook

## Configuration

### Customize User Resources

Edit `values.yaml` to change default user pod resources:

```yaml
singleuser:
  cpu:
    request: 2
    limit: 4
  memory:
    request: "8Gi"
    limit: "16Gi"
  storage:
    capacity: 10Gi
```

### Add Admin Users

Update `hub.config.JupyterHub.admin_users`:

```yaml
hub:
  config:
    JupyterHub:
      admin_users:
        - admin
        - user1@254carbon.com
```

### Configure Platform Service Credentials

Update secrets in `templates/secrets.yaml`:

```bash
kubectl create secret generic jupyterhub-secrets \
  -n jupyter \
  --from-literal=minio-access-key=<key> \
  --from-literal=minio-secret-key=<secret> \
  --from-literal=postgres-password=<password> \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Usage

### Starting a Notebook

1. Log in at https://jupyter.254carbon.com
2. Click "Start My Server" (if not already running)
3. Wait for JupyterHub to spawn your notebook
4. You'll be redirected to JupyterLab interface

### Accessing Platform Services

All platform services are pre-configured and available via Python:

#### Trino Example
```python
from connect_trino import get_connection

conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT * FROM iceberg.default.my_table LIMIT 10")
df = pd.DataFrame(cursor.fetchall())
```

#### MinIO Example
```python
from connect_minio import get_client

client = get_client()
objects = client.list_objects("data-lake", prefix="my-data/")
for obj in objects:
    print(obj.object_name)
```

#### MLflow Example
```python
import mlflow
from connect_mlflow import set_experiment

set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_artifact("model.pkl")
```

### Persistent Storage

Your home directory (`/home/jovyan`) is automatically backed by persistent storage. Files created here will persist across sessions.

Shared data is available at `/mnt/shared-data` for datasets accessible to all users.

## Monitoring

### View Metrics

Access Grafana dashboard: https://grafana.254carbon.com

Look for "JupyterHub Monitoring" dashboard showing:
- Active user count
- Memory usage trends
- CPU utilization
- Storage consumption

### View Logs

```bash
# Hub logs
kubectl logs -n jupyter deployment/jupyterhub-hub -f

# Proxy logs
kubectl logs -n jupyter deployment/jupyterhub-proxy -f

# User server logs
kubectl logs -n jupyter <user-pod-name> -f
```

### Check Health

```bash
# Hub health
curl https://jupyter.254carbon.com/hub/health

# Hub API status
curl https://jupyter.254carbon.com/hub/api/info
```

## Troubleshooting

### Users can't spawn notebooks

Check KubeSpawner logs:
```bash
kubectl logs -n jupyter deployment/jupyterhub-hub | grep -i spawn
```

Common issues:
- Insufficient cluster resources
- PVC storage full
- Service account permissions

### Connection refused to platform services

Verify service DNS is accessible:
```bash
kubectl exec -it pod/jupyterhub-hub -- nslookup trino.data-platform
```

Check network policies:
```bash
kubectl get networkpolicies -n jupyter
```

### Cloudflare Access not working

Verify OAuth2 credentials in secrets:
```bash
kubectl get secret jupyterhub-secrets -n jupyter -o yaml
```

Check ingress configuration:
```bash
kubectl get ingress -n jupyter
kubectl describe ingress jupyterhub -n jupyter
```

### High memory usage

Users may be running memory-intensive workloads. Options:
1. Increase memory limit in values.yaml
2. Set memory request/limit in user spawner options
3. Monitor with: `kubectl top pods -n jupyter`

## Advanced Configuration

### Custom Notebook Image

Build and push custom image:
```bash
cd docker/jupyter-notebook
docker build -t <registry>/jupyter-notebook:latest .
docker push <registry>/jupyter-notebook:latest
```

Update values.yaml:
```yaml
singleuser:
  image:
    name: <registry>/jupyter-notebook
    tag: latest
```

### Authentication with Keycloak

Instead of Cloudflare Access, you can use Keycloak:

```yaml
auth:
  type: keycloak
  oauthlib:
    client_id: jupyterhub
    client_secret: <secret>
    oauth_callback_url: "https://jupyter.254carbon.com/hub/oauth_callback"
    authorize_url: "https://auth.254carbon.com/auth/realms/master/protocol/openid-connect/auth"
    token_url: "https://auth.254carbon.com/auth/realms/master/protocol/openid-connect/token"
    userdata_url: "https://auth.254carbon.com/auth/realms/master/protocol/openid-connect/userinfo"
```

### Enable JupyterLab Extensions

Add to Dockerfile or use spawner environment:

```yaml
singleuser:
  environment:
    JUPYTER_ENABLE_LAB: "1"
    # Extensions will be auto-installed from requirements.txt
```

## Uninstallation

```bash
# Via Helm
helm uninstall jupyterhub -n jupyter

# Via ArgoCD
argocd app delete jupyterhub

# Clean up namespace
kubectl delete namespace jupyter
```

## Documentation Links

- [JupyterHub Documentation](https://jupyterhub.readthedocs.io/)
- [Zero to JupyterHub with Kubernetes](https://zero-to-jupyterhub.readthedocs.io/)
- [KubeSpawner Documentation](https://kubespawner.readthedocs.io/)
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/)

## Support

For issues or questions:
1. Check logs: `kubectl logs -n jupyter <pod-name>`
2. Review troubleshooting section above
3. Consult platform documentation: https://docs.254carbon.com
4. Contact platform team: platform@254carbon.com
