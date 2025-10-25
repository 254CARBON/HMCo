# JupyterHub Deployment Checklist

Use this checklist to track the JupyterHub deployment progress. Work through each section systematically.

## Phase 1: Preparation (15 minutes)

### 1.1 Review Documentation
- [ ] Read `JUPYTERHUB_IMPLEMENTATION_SUMMARY.md`
- [ ] Review `docs/jupyterhub/README.md` for architecture
- [ ] Understand resource requirements
- [ ] Check cluster has sufficient resources

### 1.2 Verify Prerequisites
- [ ] Kubernetes cluster running (v1.24+)
- [ ] kubectl configured and working
- [ ] Helm 3.x installed
- [ ] ArgoCD installed and configured
- [ ] NGINX Ingress Controller running
- [ ] Cloudflare tunnel already configured
- [ ] All platform services (Trino, MinIO, MLflow, etc.) running

### 1.3 Assign Roles
- [ ] Identify operators who will perform deployment
- [ ] Identify who has Cloudflare access
- [ ] Identify who will manage monitoring
- [ ] Document contact for support

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 2: Image Building (15 minutes)

### 2.1 Build Custom Image
```bash
# [ ] Navigate to: docker/jupyter-notebook/
cd docker/jupyter-notebook

# [ ] Build image
docker build -t 254carbon/jupyter-notebook:4.0.0 .

# [ ] Tag for registry (example for Docker Hub)
docker tag 254carbon/jupyter-notebook:4.0.0 yourusername/jupyter-notebook:4.0.0

# [ ] Push to registry
docker push yourusername/jupyter-notebook:4.0.0
```

**Operator**: ________________  **Date**: __________ **Time**: __________

### 2.2 Verify Image
```bash
# [ ] Verify image pushed successfully
docker pull yourusername/jupyter-notebook:4.0.0

# [ ] Check image size and layers
docker inspect yourusername/jupyter-notebook:4.0.0
```

### 2.3 Update values.yaml (if using custom registry)
- [ ] Edit `helm/charts/jupyterhub/values.yaml`
- [ ] Update `singleuser.image.name` to your registry path
- [ ] Update `singleuser.image.tag` if not 4.0.0
- [ ] Commit changes to git

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 3: Cloudflare Configuration (10 minutes)

### 3.1 Create Cloudflare Access Application
- [ ] Log in to: https://one.dash.cloudflare.com/
- [ ] Navigate to: Access → Applications
- [ ] Click: Add an application
- [ ] Select: SaaS
- [ ] Set application name: "JupyterHub"
- [ ] Set subdomain: "jupyter"
- [ ] Set domain: "254carbon.com"
- [ ] Click: Next

**Operator**: ________________  **Date**: __________

### 3.2 Configure OAuth2
- [ ] Select Provider: OpenID Connect
- [ ] Accept default OAuth endpoints (auto-populated)
- [ ] Click: Next

### 3.3 Add Access Policy
- [ ] Create policy rule
- [ ] Set action: Allow
- [ ] Set condition: Email domain is @254carbon.com
- [ ] Click: Save

**Operator**: ________________  **Date**: __________

### 3.4 Save Credentials
- [ ] Go to: Settings → Configuration
- [ ] Copy: Client ID
- [ ] Copy: Client Secret
- [ ] Store securely (e.g., password manager)
- [ ] Note credentials for next step

**Client ID**: ___________________________________
**Client Secret**: ________________________________

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 4: Kubernetes Secrets (5 minutes)

### 4.1 Generate Random Tokens
```bash
# [ ] Generate API token
API_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
echo $API_TOKEN

# [ ] Generate crypt key
CRYPT_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
echo $CRYPT_KEY

# [ ] Set OAuth credentials from Cloudflare
OAUTH_CLIENT_ID="<from-cloudflare>"
OAUTH_CLIENT_SECRET="<from-cloudflare>"
```

**Operator**: ________________  **Date**: __________

### 4.2 Get Platform Service Credentials
```bash
# [ ] Get MinIO credentials
MINIO_ACCESS_KEY=$(kubectl get secret -n data-platform minio-creds -o jsonpath='{.data.accesskey}' 2>/dev/null | base64 -d || echo "minioadmin")
MINIO_SECRET_KEY=$(kubectl get secret -n data-platform minio-creds -o jsonpath='{.data.secretkey}' 2>/dev/null | base64 -d || echo "minioadmin")

# [ ] Get PostgreSQL password
POSTGRES_PASSWORD=$(kubectl get secret -n data-platform postgres-secret -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || echo "postgres")

echo "All credentials loaded"
```

### 4.3 Create Kubernetes Secret
```bash
# [ ] Create jupyter namespace
kubectl create namespace jupyter --dry-run=client -o yaml | kubectl apply -f -

# [ ] Create secret
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

# [ ] Verify secret
kubectl get secret jupyterhub-secrets -n jupyter
```

**Operator**: ________________  **Date**: __________

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 5: Deployment (10 minutes)

### 5.1 Sync via ArgoCD
```bash
# [ ] Check JupyterHub app exists
argocd app list | grep jupyterhub

# [ ] Sync application
argocd app sync jupyterhub

# [ ] Wait for sync to complete
argocd app wait jupyterhub --sync

# [ ] Verify status
argocd app get jupyterhub
```

**Operator**: ________________  **Date**: __________  **Time**: __________

### 5.2 Monitor Deployment
```bash
# [ ] Watch pods coming up
kubectl get pods -n jupyter -w

# [ ] (After pods are ready, Ctrl+C to exit)

# [ ] Verify all pods are Running
kubectl get pods -n jupyter
# Expected: 2 hub pods, 2 proxy pods (all Running)

# [ ] Check services
kubectl get svc -n jupyter

# [ ] Check ingress
kubectl get ingress -n jupyter
```

**Operator**: ________________  **Date**: __________

### 5.3 Check Logs
```bash
# [ ] Check hub logs for errors
kubectl logs -n jupyter deployment/jupyterhub-hub | head -50

# [ ] Check proxy logs
kubectl logs -n jupyter deployment/jupyterhub-proxy | head -50

# [ ] Verify no error messages
```

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 6: Cloudflare Tunnel (5 minutes)

### 6.1 Add JupyterHub Route
- [ ] Go to: https://one.dash.cloudflare.com/
- [ ] Navigate to: Networks → Tunnels
- [ ] Select your tunnel
- [ ] Click: Public Hostnames tab
- [ ] Click: Add a public hostname
- [ ] Set subdomain: jupyter
- [ ] Set domain: 254carbon.com
- [ ] Set service: HTTP
- [ ] Set URL: http://ingress-nginx-controller.ingress-nginx:80
- [ ] Click: Save

**Operator**: ________________  **Date**: __________

### 6.2 Verify Route
```bash
# [ ] Test DNS resolution
nslookup jupyter.254carbon.com
# Expected: Cloudflare IP address

# [ ] Test HTTPS connectivity
curl -I https://jupyter.254carbon.com
# Expected: 200 OK or redirect (not 502/503)
```

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 7: User Access Testing (10 minutes)

### 7.1 Test Authentication
- [ ] Open browser
- [ ] Navigate to: https://jupyter.254carbon.com
- [ ] Should see Cloudflare Access login
- [ ] Enter your email: your@254carbon.com
- [ ] Check email for one-time code
- [ ] Enter code
- [ ] Should see JupyterHub interface

**Tester**: ________________  **Date**: __________  **Time**: __________

### 7.2 Test Server Spawning
- [ ] Click: "Start My Server"
- [ ] Default options should be fine
- [ ] Click: "Start"
- [ ] Wait 30-60 seconds
- [ ] Should be redirected to JupyterLab
- [ ] See notebook interface with file browser

**Status**: [ ] Success [ ] Failed

**Issues** (if any): ________________________________

### 7.3 Test Platform Service Access
Open a new notebook and test:

```python
# [ ] Test Trino
from connect_trino import get_connection
conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT 1")
print("Trino works!")

# [ ] Test MinIO
from connect_minio import get_client
client = get_client()
buckets = client.list_buckets()
print(f"MinIO works! {len(buckets.buckets)} buckets found")

# [ ] Test MLflow
import mlflow
from connect_mlflow import get_tracking_uri
mlflow.set_tracking_uri(get_tracking_uri())
print("MLflow works!")
```

**All Tests Status**: [ ] Success [ ] Failed

**Issues** (if any): ________________________________

### 7.4 Test Storage Persistence
```python
# [ ] Create a test file
with open("/home/jovyan/work/test.txt", "w") as f:
    f.write("Test data")

# [ ] Verify file exists
import os
print(os.path.exists("/home/jovyan/work/test.txt"))  # Should be True
```

**Test Result**: [ ] File persisted [ ] File disappeared

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 8: Monitoring Setup (5 minutes)

### 8.1 Import Grafana Dashboard
- [ ] Go to: https://grafana.254carbon.com
- [ ] Click: + → Import
- [ ] Enter Dashboard ID: 12114 (for existing dashboard) or search for "jupyterhub"
- [ ] Select Prometheus data source
- [ ] Click: Import
- [ ] Dashboard should appear

**Operator**: ________________  **Date**: __________

### 8.2 Verify Metrics Collection
```bash
# [ ] Check ServiceMonitor is created
kubectl get servicemonitor -n jupyter

# [ ] Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# (Then visit http://localhost:9090)
# Search for: jupyterhub_
```

**Operator**: ________________  **Date**: __________

### 8.3 View Dashboard
- [ ] Go to Grafana: https://grafana.254carbon.com
- [ ] Find JupyterHub dashboard
- [ ] Should show:
  - [ ] Active users count
  - [ ] Memory usage graph
  - [ ] CPU usage graph
  - [ ] Pod count

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 9: Documentation & Training (varies)

### 9.1 Share Documentation
- [ ] Send QUICKSTART.md to users
- [ ] Send docs/jupyterhub/README.md to operators
- [ ] Create internal wiki/documentation page
- [ ] Update team wiki with JupyterHub info

**Owner**: ________________  **Date**: __________

### 9.2 Conduct Training
- [ ] Schedule user training session
- [ ] Prepare demo (QUICKSTART.md examples)
- [ ] Train operators on management
- [ ] Document Q&A for FAQ

**Owner**: ________________  **Date**: __________

### 9.3 Create Support Documentation
- [ ] Document common issues
- [ ] Create troubleshooting guide
- [ ] Document escalation path
- [ ] Set up support channel (Slack)

**Owner**: ________________  **Date**: __________

**Section Status**: [ ] Incomplete [ ] Complete

---

## Phase 10: Post-Deployment

### 10.1 Backup Configuration
- [ ] Label PVCs for Velero backup
```bash
kubectl label pvc jupyter-shared-data -n jupyter velero.io/exclude-from-backup=false
```
- [ ] Verify backup includes JupyterHub resources

### 10.2 Monitor Production
```bash
# [ ] Day 1: Check logs hourly
# [ ] Week 1: Monitor Grafana dashboard daily
# [ ] Week 2: Gather user feedback
# [ ] Month 1: Full production review
```

### 10.3 Document Any Issues
- [ ] Log any issues encountered
- [ ] Document workarounds
- [ ] Plan fixes/improvements
- [ ] Update documentation

**Owner**: ________________  **Date**: __________

**Section Status**: [ ] Incomplete [ ] Complete

---

## Final Verification Checklist

### 10.1 All Access Working
- [ ] ✅ Users can access https://jupyter.254carbon.com
- [ ] ✅ Authentication via Cloudflare works
- [ ] ✅ Notebook servers spawn successfully
- [ ] ✅ Users can connect to platform services

### 10.2 All Services Integrated
- [ ] ✅ Trino connectivity verified
- [ ] ✅ MinIO connectivity verified
- [ ] ✅ MLflow connectivity verified
- [ ] ✅ PostgreSQL connectivity verified
- [ ] ✅ DataHub connectivity verified
- [ ] ✅ Ray connectivity verified
- [ ] ✅ Kafka connectivity verified

### 10.3 Monitoring Working
- [ ] ✅ Prometheus scraping metrics
- [ ] ✅ Grafana dashboard displays data
- [ ] ✅ Alerts configured (optional)

### 10.4 Security in Place
- [ ] ✅ Network policies applied
- [ ] ✅ Resource quotas enforced
- [ ] ✅ RBAC working
- [ ] ✅ Secrets encrypted

### 10.5 Documentation Complete
- [ ] ✅ User documentation shared
- [ ] ✅ Operator documentation ready
- [ ] ✅ Runbooks created
- [ ] ✅ FAQ documented

---

## Rollout Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Preparation | 15 min | [ ] |
| 2. Image Building | 15 min | [ ] |
| 3. Cloudflare Config | 10 min | [ ] |
| 4. K8s Secrets | 5 min | [ ] |
| 5. Deployment | 10 min | [ ] |
| 6. Tunnel Setup | 5 min | [ ] |
| 7. Testing | 10 min | [ ] |
| 8. Monitoring | 5 min | [ ] |
| 9. Documentation | 30 min | [ ] |
| 10. Post-Deploy | Ongoing | [ ] |
| **Total** | **~100 min** | [ ] |

---

## Sign-Off

**Deployment Completed By**: ________________  **Date**: __________

**Verified By**: ________________  **Date**: __________

**Approved By**: ________________  **Date**: __________

---

## Issues Encountered

| Issue | Resolution | Date |
|-------|-----------|------|
| | | |
| | | |
| | | |

---

## Success Metrics

After 1 week:
- [ ] 0 critical incidents
- [ ] All users able to access
- [ ] No performance issues
- [ ] Positive user feedback

After 1 month:
- [ ] Stable operation
- [ ] Active user usage
- [ ] Platform integration complete
- [ ] Monitoring fully operational

---

**Deployment Status**: [ ] Not Started [ ] In Progress [ ] Complete [ ] Rolled Back

**Overall Status**: _____ / 100% complete

**Ready for Production**: [ ] Yes [ ] No

---

Last Updated: __________
Next Review: __________
