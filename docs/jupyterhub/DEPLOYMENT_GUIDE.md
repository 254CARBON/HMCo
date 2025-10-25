# JupyterHub Deployment Guide

Step-by-step guide to deploy JupyterHub on Kubernetes with full platform integration.

## Table of Contents

1. [Pre-Deployment Checks](#pre-deployment-checks)
2. [Build Custom Notebook Image](#build-custom-notebook-image)
3. [Configure Secrets](#configure-secrets)
4. [Deploy JupyterHub](#deploy-jupyterhub)
5. [Configure Cloudflare Access](#configure-cloudflare-access)
6. [Update Cloudflare Tunnel](#update-cloudflare-tunnel)
7. [Verify Deployment](#verify-deployment)
8. [Post-Deployment Configuration](#post-deployment-configuration)

## Pre-Deployment Checks

### Verify Cluster Resources

```bash
# Check available nodes
kubectl get nodes

# Check available resources
kubectl top nodes
kubectl describe nodes | grep -A 5 "Allocated resources"

# Verify required storage class exists
kubectl get storageclass
# Should show: local-path (or similar)
```

### Verify Platform Services Are Running

```bash
# Check critical services
for service in trino minio postgres-shared datahub-gms mlflow ray-cluster-head kafka-cluster-kafka-bootstrap; do
  echo "Checking $service..."
  kubectl get svc $service -n data-platform 2>/dev/null || kubectl get svc $service -n ml-platform 2>/dev/null || echo "  ⚠️  $service not found"
done
```

### Verify NGINX Ingress Controller

```bash
# Check ingress controller is running
kubectl get deployment -n ingress-nginx

# Verify it's ready
kubectl get pods -n ingress-nginx
```

## Build Custom Notebook Image

The custom notebook image includes all data science libraries and platform SDKs.

### Prerequisites

- Docker or Podman installed
- Container registry (Docker Hub, Harbor, etc.)
- Push credentials configured

### Build Steps

```bash
# Navigate to Dockerfile directory
cd docker/jupyter-notebook

# Build image
docker build -t 254carbon/jupyter-notebook:4.0.0 .

# Tag for your registry
docker tag 254carbon/jupyter-notebook:4.0.0 <your-registry>/jupyter-notebook:4.0.0

# Push to registry
docker push <your-registry>/jupyter-notebook:4.0.0

# Verify push
docker pull <your-registry>/jupyter-notebook:4.0.0
```

### Alternative: Use Public Image

If not building custom image, update `values.yaml`:

```yaml
singleuser:
  image:
    name: jupyter/datascience-notebook
    tag: "latest"
```

## Configure Secrets

### Create JupyterHub Secrets

```bash
# Generate random tokens
API_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
CRYPT_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

echo "API Token: $API_TOKEN"
echo "Crypt Key: $CRYPT_KEY"

# Save for later use
cat > /tmp/jupyterhub-secrets.txt << EOF
API_TOKEN: $API_TOKEN
CRYPT_KEY: $CRYPT_KEY
EOF
```

### Create Kubernetes Secret

```bash
# Get platform service credentials
MINIO_ACCESS_KEY=$(kubectl get secret -n data-platform minio-creds -o jsonpath='{.data.accesskey}' | base64 -d)
MINIO_SECRET_KEY=$(kubectl get secret -n data-platform minio-creds -o jsonpath='{.data.secretkey}' | base64 -d)
POSTGRES_PASSWORD=$(kubectl get secret -n data-platform postgres-secret -o jsonpath='{.data.password}' | base64 -d)

# Create secret
kubectl create secret generic jupyterhub-secrets \
  --namespace=jupyter \
  --from-literal=api-token="${API_TOKEN}" \
  --from-literal=crypt-key="${CRYPT_KEY}" \
  --from-literal=minio-access-key="${MINIO_ACCESS_KEY}" \
  --from-literal=minio-secret-key="${MINIO_SECRET_KEY}" \
  --from-literal=postgres-password="${POSTGRES_PASSWORD}" \
  --from-literal=oauth-client-id="change-to-real-value" \
  --from-literal=oauth-client-secret="change-to-real-value" \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify secret created
kubectl get secret jupyterhub-secrets -n jupyter
```

### Update Cloudflare Access OAuth2 Credentials

If using Cloudflare Access (recommended):

1. In Cloudflare Zero Trust console:
   - Go to Access > Applications > JupyterHub
   - Note the Client ID and Client Secret
   - (If not created yet, create application first)

2. Update secret:
   ```bash
   kubectl patch secret jupyterhub-secrets -n jupyter \
     -p "{\"data\":{\"oauth-client-id\":\"$(echo -n 'YOUR_CLIENT_ID' | base64)\",\"oauth-client-secret\":\"$(echo -n 'YOUR_CLIENT_SECRET' | base64)\"}}"
   ```

## Deploy JupyterHub

### Via ArgoCD (Recommended)

ArgoCD application is already configured in `k8s/gitops/argocd-applications.yaml`.

```bash
# List applications
argocd app list | grep jupyterhub

# Sync application
argocd app sync jupyterhub

# Wait for sync to complete
argocd app wait jupyterhub --sync

# Check deployment status
argocd app get jupyterhub
```

### Via Helm (Manual)

```bash
# Add Helm repository (if needed)
helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
helm repo update

# Install JupyterHub
helm install jupyterhub ./helm/charts/jupyterhub \
  --namespace jupyter \
  --create-namespace \
  --values helm/charts/jupyterhub/values.yaml

# Wait for deployment
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=jupyterhub -n jupyter --timeout=600s

# Check status
helm status jupyterhub -n jupyter
```

### Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n jupyter

# Expected output:
# NAME                                    READY   STATUS    RESTARTS   AGE
# jupyterhub-hub-xxxxxxxxxx-xxxxx         1/1     Running   0          2m
# jupyterhub-proxy-xxxxxxxxxx-xxxxx       1/1     Running   0          2m

# Check services
kubectl get svc -n jupyter

# Check ingress
kubectl get ingress -n jupyter
```

## Configure Cloudflare Access

### Create JupyterHub Application in Cloudflare

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/)
2. Navigate to Access → Applications
3. Click "Add an application"
4. Select "SaaS" application
5. Fill in:
   - **Application name**: JupyterHub
   - **Subdomain**: jupyter
   - **Domain**: 254carbon.com
   - **Application type**: SaaS
6. Click Next
7. Configure OAuth:
   - **Provider**: OAuth 2.0
   - Leave OAuth endpoints as default (will auto-populate)
8. Add policy:
   - **Policy action**: Allow
   - **Rule**: Email domain is `@254carbon.com`
9. Click Save

### Note OAuth2 Credentials

After creating application:
1. Go back to Application List
2. Find JupyterHub application
3. Go to Settings > Configuration
4. Note down:
   - Client ID
   - Client Secret
5. Update Kubernetes secret (see Configure Secrets section)

## Update Cloudflare Tunnel

### Add JupyterHub to Tunnel Configuration

Edit Cloudflare tunnel configuration to include JupyterHub route:

```yaml
tunnel: <tunnel-id>
credentials-file: /etc/cloudflared/config.json

ingress:
  # ... existing routes ...
  - hostname: jupyter.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  # ... rest of routes ...
```

### Via Cloudflare Dashboard

1. Go to Zero Trust → Tunnels
2. Select your tunnel
3. Go to "Public Hostnames" tab
4. Click "Add a public hostname"
5. Configure:
   - **Subdomain**: jupyter
   - **Domain**: 254carbon.com
   - **Type**: HTTP
   - **URL**: http://ingress-nginx-controller.ingress-nginx:80
6. Save

### Verify Tunnel Route

```bash
# Test route
curl https://jupyter.254carbon.com -I

# Expected: 200 OK (or redirect to login)
```

## Verify Deployment

### Check Health Endpoints

```bash
# JupyterHub Hub health
curl -I https://jupyter.254carbon.com/hub/health

# JupyterHub API
curl https://jupyter.254carbon.com/hub/api/info

# Should return JSON with version info
```

### Test User Access

1. Navigate to https://jupyter.254carbon.com
2. Authenticate via Cloudflare Access
3. Should see JupyterHub interface
4. Click "Start My Server"
5. Wait for notebook to spawn
6. Should be redirected to JupyterLab

### Check Pod Logs

```bash
# Hub logs
kubectl logs -n jupyter deployment/jupyterhub-hub -f

# Proxy logs
kubectl logs -n jupyter deployment/jupyterhub-proxy -f

# When spawning user notebook
kubectl logs -n jupyter <user-pod-name> -f
```

## Post-Deployment Configuration

### Create Example Notebooks

```bash
# Copy example notebooks to shared data volume
kubectl cp examples/ jupyter/jupyterhub-hub-0:/opt/notebooks/examples/
```

### Configure Backup

```bash
# Ensure persistent volumes are included in Velero backups
kubectl label pvc jupyter-shared-data -n jupyter velero.io/exclude-from-backup=false
```

### Set Up Monitoring

1. Access Grafana: https://grafana.254carbon.com
2. Add new data source: Prometheus
3. Import JupyterHub dashboard:
   - Dashboard ID: 12114 (JupyterHub)
   - Or use dashboard from ConfigMap: `jupyterhub-dashboard`

### Configure User Quotas

To limit total resources used by all notebooks:

```bash
# Edit ResourceQuota
kubectl edit resourcequota jupyter-quota -n jupyter

# Example limits:
# requests.cpu: "100"
# requests.memory: "200Gi"
# limits.cpu: "200"
# limits.memory: "400Gi"
```

### Enable Audit Logging

```bash
# Check if audit logging is enabled
kubectl get audit -n jupyter

# Or via ArgoCD/Helm, enable in values.yaml:
# monitoring:
#   enabled: true
#   auditLog:
#     enabled: true
```

## Troubleshooting Deployment

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n jupyter

# Check logs
kubectl logs <pod-name> -n jupyter
```

### Ingress Not Working

```bash
# Verify ingress is created
kubectl get ingress -n jupyter

# Describe ingress
kubectl describe ingress jupyterhub -n jupyter

# Check ingress controller
kubectl get pods -n ingress-nginx
```

### Users Can't Connect to Platform Services

```bash
# Test connectivity from user pod
kubectl exec -it <user-pod> -n jupyter -- bash

# Inside pod:
nslookup trino.data-platform
curl http://trino.data-platform:8080

# Check network policies
kubectl get networkpolicies -n jupyter
```

### High Resource Usage

```bash
# Check top pods
kubectl top pods -n jupyter

# Describe specific pod
kubectl describe pod <pod-name> -n jupyter

# Check resource requests/limits
kubectl get pod <pod-name> -n jupyter -o yaml | grep -A 10 resources:
```

## Next Steps

1. [Create example notebooks](./examples/)
2. [Configure additional authentication methods](./authentication.md)
3. [Set up monitoring dashboards](../monitoring/jupyter-dashboards.md)
4. [Configure backup and disaster recovery](../backup/README.md)
5. [Scale to production](./production-scaling.md)
