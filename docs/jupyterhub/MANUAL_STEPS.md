# Manual Steps for JupyterHub Completion

This document outlines the manual steps required to complete the JupyterHub deployment. These steps involve Cloudflare configuration and system-level setup.

## Step 1: Build and Push Custom Notebook Image

**Location**: `docker/jupyter-notebook/`

```bash
# Navigate to Docker directory
cd docker/jupyter-notebook

# Build the image
docker build -t 254carbon/jupyter-notebook:4.0.0 .

# Tag for your registry (example: Docker Hub)
docker tag 254carbon/jupyter-notebook:4.0.0 yourusername/jupyter-notebook:4.0.0

# Push to registry
docker push yourusername/jupyter-notebook:4.0.0

# If using Harbor or private registry:
docker tag 254carbon/jupyter-notebook:4.0.0 registry.254carbon.com/jupyter-notebook:4.0.0
docker push registry.254carbon.com/jupyter-notebook:4.0.0
```

**Update values.yaml if using custom registry**:
```bash
# Edit helm/charts/jupyterhub/values.yaml
# Change:
singleuser:
  image:
    name: yourusername/jupyter-notebook    # or your registry path
    tag: "4.0.0"
```

**Time**: 10-15 minutes (building)

## Step 2: Create Cloudflare Access Application

### 2.1 Access Cloudflare Zero Trust Console

1. Go to: https://one.dash.cloudflare.com/
2. Sign in with your Cloudflare account
3. Select your team

### 2.2 Create OAuth Application

1. Navigate to: **Access → Applications**
2. Click: **Add an application**
3. Select: **SaaS**
4. Fill in application details:
   - **Application name**: JupyterHub
   - **Subdomain**: jupyter
   - **Domain**: 254carbon.com
   - **Application type**: SaaS
5. Click: **Next**

### 2.3 Configure OAuth2

1. Under "Configure OIDC" section:
   - **Provider**: OpenID Connect
   - Keep default OAuth endpoints (auto-populated by Cloudflare)
2. Click: **Next**

### 2.4 Add Access Policy

1. Under "Policies" section:
2. Create rule:
   - **Action**: Allow
   - **Condition**: Email Domain is @254carbon.com
3. Click: **Add** to save policy
4. Click: **Save**

### 2.5 Get OAuth Credentials

1. After creation, you'll see application details
2. Go to: **Settings → Configuration**
3. Note down:
   - **Client ID**: (copy this)
   - **Client Secret**: (copy this)
4. Save these for Step 3

**Time**: 5 minutes

## Step 3: Create Kubernetes Secrets

### 3.1 Generate Random Tokens

```bash
# Generate API token
API_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Generate crypt key
CRYPT_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Generate OAuth secrets
OAUTH_CLIENT_ID="<from-cloudflare-step-2>"
OAUTH_CLIENT_SECRET="<from-cloudflare-step-2>"

echo "API_TOKEN=$API_TOKEN"
echo "CRYPT_KEY=$CRYPT_KEY"
echo "OAUTH_CLIENT_ID=$OAUTH_CLIENT_ID"
echo "OAUTH_CLIENT_SECRET=$OAUTH_CLIENT_SECRET"
```

### 3.2 Get Platform Service Credentials

```bash
# Get MinIO credentials
MINIO_ACCESS_KEY=$(kubectl get secret -n data-platform minio-creds -o jsonpath='{.data.accesskey}' 2>/dev/null | base64 -d)
MINIO_SECRET_KEY=$(kubectl get secret -n data-platform minio-creds -o jsonpath='{.data.secretkey}' 2>/dev/null | base64 -d)

# If above fails, use defaults (change in production!)
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

# Get PostgreSQL password
POSTGRES_PASSWORD=$(kubectl get secret -n data-platform postgres-secret -o jsonpath='{.data.password}' 2>/dev/null | base64 -d)

# If above fails, use default (change in production!)
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"

echo "MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY"
echo "MINIO_SECRET_KEY=$MINIO_SECRET_KEY"
echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
```

### 3.3 Create Secret in Kubernetes

```bash
# Create namespace first (if not using ArgoCD)
kubectl create namespace jupyter --dry-run=client -o yaml | kubectl apply -f -

# Create secret with all credentials
kubectl create secret generic jupyterhub-secrets \
  --namespace=jupyter \
  --from-literal=api-token="$API_TOKEN" \
  --from-literal=crypt-key="$CRYPT_KEY" \
  --from-literal=minio-access-key="$MINIO_ACCESS_KEY" \
  --from-literal=minio-secret-key="$MINIO_SECRET_KEY" \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --from-literal=oauth-client-id="$OAUTH_CLIENT_ID" \
  --from-literal=oauth-client-secret="$OAUTH_CLIENT_SECRET" \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify secret was created
kubectl get secret jupyterhub-secrets -n jupyter
```

**Time**: 5 minutes

## Step 4: Deploy JupyterHub via ArgoCD

### 4.1 Sync Application

```bash
# List applications to verify jupyterhub is present
argocd app list | grep jupyterhub

# Sync the application
argocd app sync jupyterhub

# Wait for sync to complete
argocd app wait jupyterhub --sync

# Check detailed status
argocd app get jupyterhub
```

### 4.2 Monitor Deployment

```bash
# Watch pods come up
kubectl get pods -n jupyter -w

# Expected output:
# NAME                                  READY   STATUS    RESTARTS   AGE
# jupyterhub-hub-xxxxxxxxxx-xxxxx       1/1     Running   0          2m
# jupyterhub-proxy-xxxxxxxxxx-xxxxx     1/1     Running   0          2m

# Check services
kubectl get svc -n jupyter

# Check ingress
kubectl get ingress -n jupyter
```

### 4.3 View Logs

```bash
# Hub logs
kubectl logs -n jupyter deployment/jupyterhub-hub -f

# Proxy logs
kubectl logs -n jupyter deployment/jupyterhub-proxy -f

# Watch for errors
kubectl logs -n jupyter deployment/jupyterhub-hub | grep -i error
```

**Time**: 10 minutes (including wait for pods)

## Step 5: Configure Cloudflare Tunnel

### 5.1 Add JupyterHub Route to Tunnel

1. Go to: https://one.dash.cloudflare.com/
2. Navigate to: **Networks → Tunnels**
3. Select your existing tunnel
4. Click: **Public Hostnames** tab
5. Click: **Add a public hostname**
6. Fill in:
   - **Subdomain**: jupyter
   - **Domain**: 254carbon.com
   - **Service type**: HTTP
   - **URL**: http://ingress-nginx-controller.ingress-nginx:80
7. Click: **Save**

### 5.2 Verify Tunnel Configuration

```bash
# Test DNS resolution
nslookup jupyter.254carbon.com

# Should return Cloudflare IP

# Test HTTP connectivity
curl -I https://jupyter.254carbon.com

# Should return 200 OK or redirect
```

**Time**: 5 minutes

## Step 6: Test JupyterHub Access

### 6.1 Test via Web Browser

1. Open: https://jupyter.254carbon.com
2. You should see Cloudflare Access login page
3. Enter your email: your@254carbon.com
4. Check email for one-time code
5. Enter code
6. You should now see JupyterHub interface

### 6.2 Test Server Spawning

1. Click: **"Start My Server"**
2. Default options should be fine
3. Click: **"Start"**
4. Wait 30-60 seconds for notebook to spawn
5. You should be redirected to JupyterLab

### 6.3 Test Platform Service Connectivity

Create a new notebook and test connectivity:

```python
# Test Trino connection
from connect_trino import get_connection
conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT 1")
print("Trino works!", cursor.fetchall())

# Test MinIO connection
from connect_minio import get_client
client = get_client()
buckets = client.list_buckets()
print("MinIO works!", [b.name for b in buckets.buckets])
```

**Time**: 10 minutes

## Step 7: Configure Monitoring

### 7.1 Import Grafana Dashboard

```bash
# Option 1: Import from dashboard ID
# 1. Go to Grafana: https://grafana.254carbon.com
# 2. Click: + → Import
# 3. Enter ID: 12114 (JupyterHub)
# 4. Select Prometheus datasource
# 5. Click: Import

# Option 2: Use ConfigMap dashboard
# Already created in helm/charts/jupyterhub/templates/grafana-dashboard.yaml
# Grafana will auto-discover and import
```

### 7.2 Verify Metrics Collection

```bash
# Check ServiceMonitor is created
kubectl get servicemonitor -n jupyter

# Check Prometheus is scraping metrics
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090

# Then visit: http://localhost:9090
# Search for: jupyterhub_spawner_cpu_limit
```

**Time**: 5 minutes

## Step 8: Verify Complete Deployment

Run this verification checklist:

```bash
# 1. Check all pods are running
kubectl get pods -n jupyter
# All should be in "Running" state

# 2. Check services are accessible
kubectl get svc -n jupyter
# Should see jupyterhub-hub and jupyterhub-proxy-public

# 3. Check ingress is configured
kubectl get ingress -n jupyter
# Should show jupyter.254carbon.com

# 4. Check DNS resolution
nslookup jupyter.254carbon.com
# Should resolve to Cloudflare IP

# 5. Test HTTPS access
curl -I https://jupyter.254carbon.com
# Should return 200 OK or redirect

# 6. Check storage is accessible
kubectl get pvc -n jupyter
# Should show jupyter-shared-data in Bound state

# 7. Check secrets are created
kubectl get secret -n jupyter
# Should show jupyterhub-secrets

# 8. Check resource quotas
kubectl describe resourcequota jupyter-quota -n jupyter
# Should show resource limits applied

# 9. Verify metrics collection
kubectl logs -n jupyter deployment/jupyterhub-hub | grep -i metric
# Should show metrics being exported

# 10. Test user access via browser
# Visit: https://jupyter.254carbon.com
# Should authenticate and show JupyterHub interface
```

**Time**: 10 minutes

## Troubleshooting

### Issue: Pods Not Starting

```bash
kubectl describe pod <pod-name> -n jupyter
# Check Events section for error messages

# Common causes:
# - Insufficient resources
# - Image pull errors
# - Secret not found

# Solution: Create missing resources or increase resources
```

### Issue: Ingress 502 Bad Gateway

```bash
# Check proxy pod is running
kubectl get pods -n jupyter | grep proxy

# Check proxy logs
kubectl logs -n jupyter deployment/jupyterhub-proxy

# Solution: Restart proxy
kubectl rollout restart deployment/jupyterhub-proxy -n jupyter
```

### Issue: Can't Connect to Platform Services

```bash
# Test DNS from user pod
kubectl exec -it <user-pod> -n jupyter -- bash
nslookup trino.data-platform

# Test connection
curl http://trino.data-platform:8080

# Check network policies
kubectl get networkpolicies -n jupyter
```

### Issue: Cloudflare Returns Error

```bash
# Check tunnel status
cloudflared tunnel list

# Check tunnel logs
cloudflared tunnel logs <tunnel-name>

# Verify DNS is pointing to tunnel
dig jupyter.254carbon.com
```

## Rollback Procedure

If something goes wrong, roll back using:

```bash
# Via ArgoCD
argocd app rollback jupyterhub <revision>

# Via Helm
helm rollback jupyterhub

# Via kubectl (delete and recreate)
kubectl delete all -n jupyter --all
argocd app sync jupyterhub
```

## Post-Deployment Tasks

After successful deployment:

1. **Add users to admin list** (if needed):
   ```bash
   # Edit secret or values
   kubectl edit configmap jupyterhub-hub-config -n jupyter
   # Update admin_users list
   ```

2. **Configure backup**:
   ```bash
   # Ensure PVCs are backed up
   kubectl label pvc jupyter-shared-data -n jupyter velero.io/exclude-from-backup=false
   ```

3. **Set up monitoring alerts**:
   - Go to Grafana
   - Create alert rules for high memory usage
   - Create alert for pod crash loop

4. **Document customizations**:
   - Note any changes to values.yaml
   - Document custom image modifications
   - Record Cloudflare configuration

5. **Train users**:
   - Share QUICKSTART.md
   - Conduct demo session
   - Create internal documentation

## Timeline

| Step | Task | Time |
|------|------|------|
| 1 | Build & push image | 15 min |
| 2 | Create Cloudflare app | 5 min |
| 3 | Create K8s secrets | 5 min |
| 4 | Deploy via ArgoCD | 10 min |
| 5 | Configure tunnel | 5 min |
| 6 | Test access | 10 min |
| 7 | Configure monitoring | 5 min |
| 8 | Verification | 10 min |
| **Total** | | **~65 min** |

## Success Criteria

✅ JupyterHub accessible at https://jupyter.254carbon.com
✅ Users can authenticate via Cloudflare Access
✅ Users can spawn notebook servers
✅ Notebooks can connect to all platform services
✅ Metrics are being collected in Prometheus
✅ Grafana dashboard shows usage data
✅ Shared storage is accessible
✅ Resource quotas are enforced

## Support

For issues or questions:
1. Check DEPLOYMENT_GUIDE.md
2. Review troubleshooting section above
3. Check logs: `kubectl logs -n jupyter <pod-name>`
4. Contact platform team: platform@254carbon.com

---

**Manual Steps Status**: Ready to Execute
**Estimated Time**: ~65 minutes
**Difficulty**: Medium (requires Cloudflare console access)
