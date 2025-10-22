# MLFlow Troubleshooting Guide

## Quick Diagnostics

### Check MLFlow Service Health

```bash
# Check if pods are running
kubectl get pods -n data-platform -l app=mlflow
kubectl get svc -n data-platform | grep mlflow

# Check pod status in detail
kubectl describe pod -n data-platform -l app=mlflow

# View recent logs
kubectl logs -n data-platform -l app=mlflow --tail=50
```

### Test MLFlow Connectivity

```bash
# Port forward to local
kubectl port-forward -n data-platform svc/mlflow 5000:5000 &

# Test health endpoint
curl -v http://localhost:5000/health

# Test API
curl -s http://localhost:5000/api/2.0/experiments/list | jq .
```

## Common Issues & Solutions

### Issue 1: MLFlow Pods in CrashLoopBackOff

#### Symptoms
- Pods restart continuously
- `kubectl get pods` shows `CrashLoopBackOff` status
- Logs show connection errors

#### Diagnosis

```bash
# Check pod logs
kubectl logs -n data-platform <mlflow-pod-name>

# Check recent events
kubectl describe pod -n data-platform <mlflow-pod-name>

# Common error patterns:
# 1. "could not connect to database" → PostgreSQL issue
# 2. "connection refused" → MinIO issue
# 3. "permission denied" → Credential issue
```

#### Solutions

**PostgreSQL Connection Fails**

```bash
# Verify PostgreSQL pod is running
kubectl get pods -n data-platform -l app=postgres

# Check PostgreSQL logs
kubectl logs -n data-platform postgres-shared-<pod>

# Verify database and user exist
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U datahub -d postgres -c "\du mlflow"

# Recreate if needed
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U datahub -d postgres -f - < k8s/compute/mlflow/mlflow-backend-db.sql
```

**MinIO Connection Fails**

```bash
# Verify MinIO is running
kubectl get pods -n data-platform -l app=minio

# Check MinIO logs
kubectl logs -n data-platform minio-<pod>

# Test MinIO connectivity from MLFlow pod
kubectl exec -it -n data-platform mlflow-<pod> -- \
  curl -k https://minio.minio.svc.cluster.local:9000/minio/health/live

# Verify bucket exists
kubectl exec -it -n data-platform minio-<pod> -- \
  mc ls local/mlflow-artifacts
```

**Credential Issues**

```bash
# Check secrets
kubectl get secrets -n data-platform | grep mlflow

# View secret content (careful with credentials!)
kubectl get secret -n data-platform mlflow-backend-secret -o yaml

# Verify env vars in pod
kubectl exec -n data-platform mlflow-<pod> -- env | grep -E "AWS|MLFLOW|DATABASE"
```

---

### Issue 2: Cannot Access MLFlow UI (401 Unauthorized)

#### Symptoms
- Receiving 401 or 403 errors
- Cloudflare Access login loop
- Cannot see MLFlow UI after authentication

#### Diagnosis

```bash
# Check ingress is configured
kubectl get ingress -n data-platform mlflow-ingress

# Check ingress details
kubectl describe ingress -n data-platform mlflow-ingress

# Verify annotation annotations
kubectl get ingress -n data-platform mlflow-ingress -o yaml | \
  grep -i auth
```

#### Solutions

**Fix Cloudflare Access Configuration**

1. Verify MLFlow application exists in Cloudflare:
   - Go to: https://dash.cloudflare.com/zero-trust/access
   - Check: `mlflow.254carbon.com` application exists
   - Status: Application is active

2. Verify policy configuration:
   - Policy allows your email/domain
   - Session duration is appropriate (8 hours recommended)
   - Policy is enabled

3. Update ingress annotation if account ID changed:
   ```bash
   kubectl patch ingress mlflow-ingress -n data-platform --type json -p \
     '[{"op":"replace","path":"/metadata/annotations/nginx.ingress.kubernetes.io~1auth-url","value":"https://YOUR_TEAM.cloudflareaccess.com/cdn-cgi/access/authorize"}]'
   ```

**Check Ingress Rules**

```yaml
# Required annotations
nginx.ingress.kubernetes.io/auth-url: "https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize"
nginx.ingress.kubernetes.io/auth-signin: "https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$escaped_request_uri"
```

---

### Issue 3: Experiment Tracking Fails in DolphinScheduler

#### Symptoms
- Task runs but MLFlow logging fails
- No experiments appear in MLFlow UI
- Errors about "cannot reach tracking server"

#### Diagnosis

```bash
# Check if mlflow package is installed in worker
# (On DolphinScheduler worker node)
python -c "import mlflow; print(mlflow.__version__)"

# Check environment variable
echo $MLFLOW_TRACKING_URI

# Test connectivity
python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow.data-platform.svc.cluster.local:5000'); print(mlflow.get_tracking_uri())"
```

#### Solutions

**Install MLFlow on Workers**

```bash
# On each DolphinScheduler worker node
pip install mlflow>=2.10.0 boto3>=1.26.0

# Or add to worker Dockerfile
RUN pip install mlflow>=2.10.0 boto3>=1.26.0
```

**Set Tracking URI**

Option A: In task code:
```python
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow.data-platform.svc.cluster.local:5000'
```

Option B: In DolphinScheduler environment:
```bash
export MLFLOW_TRACKING_URI=http://mlflow.data-platform.svc.cluster.local:5000
```

Option C: In mlflow_client.py (already has default):
```python
self.tracking_uri = tracking_uri or os.environ.get(
    'MLFLOW_TRACKING_URI',
    'http://mlflow.data-platform.svc.cluster.local:5000'
)
```

**Verify Network Connectivity**

```bash
# From DolphinScheduler worker pod
kubectl exec -it -n data-platform dolphinscheduler-worker-<pod> -- \
  curl -v http://mlflow.data-platform.svc.cluster.local:5000/health

# Check if network policy is blocking traffic
kubectl get networkpolicies -n data-platform

# If blocked, create allow rule
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mlflow-allow-from-dolphinscheduler
  namespace: data-platform
spec:
  podSelector:
    matchLabels:
      app: mlflow
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: dolphinscheduler
EOF
```

---

### Issue 4: S3 (MinIO) Artifact Upload Fails

#### Symptoms
- Errors like "S3 upload error", "bucket not found", "access denied"
- MLFlow logs show S3 authentication failures
- Artifacts don't appear in MinIO console

#### Diagnosis

```bash
# Check MinIO bucket exists
kubectl exec -it -n data-platform minio-<pod> -- /bin/sh
mc alias set local https://localhost:9000 minioadmin minioadmin
mc ls local | grep mlflow-artifacts
# Exit

# Check S3 credentials in MLFlow pod
kubectl exec -n data-platform mlflow-<pod> -- env | grep AWS

# Test S3 connectivity from MLFlow pod
kubectl exec -n data-platform mlflow-<pod> -- \
  aws s3 ls s3://mlflow-artifacts/ --endpoint-url https://minio.minio.svc.cluster.local:9000

# Check artifact store URI
kubectl get configmap -n data-platform mlflow-config -o yaml | grep artifact
```

#### Solutions

**Create Missing Bucket**

```bash
# Access MinIO pod
kubectl exec -it -n data-platform minio-<pod> -- /bin/sh

# Create bucket
mc alias set local https://localhost:9000 minioadmin minioadmin
mc mb local/mlflow-artifacts

# Enable versioning (recommended)
mc version enable local/mlflow-artifacts

# Set lifecycle policy (optional - cleanup old artifacts)
mc ilm import local/mlflow-artifacts << 'EOF'
{
 "Rules": [
  {
   "ID": "cleanup-old-artifacts",
   "Status": "Enabled",
   "Filter": {},
   "Expiration": {
    "Days": 90
   }
  }
 ]
}
EOF

exit
```

**Fix Credentials**

```bash
# Update secrets with correct MinIO credentials
kubectl patch secret -n data-platform mlflow-artifact-secret --type merge -p \
  '{"stringData":{"aws_access_key_id":"your-key","aws_secret_access_key":"your-secret"}}'

# Restart MLFlow pods to pick up new credentials
kubectl rollout restart deployment -n data-platform mlflow
```

**Disable SSL Verification (dev/test only)**

```bash
# Update config map
kubectl patch configmap -n data-platform mlflow-config --type merge -p \
  '{"data":{"aws_s3_verify":"false"}}'

# Restart pods
kubectl rollout restart deployment -n data-platform mlflow
```

---

### Issue 5: High Memory Usage / Pod OOMKilled

#### Symptoms
- Pods restart with `OOMKilled` status
- `kubectl top pods` shows high memory usage
- Errors in logs before restart

#### Diagnosis

```bash
# Check resource limits
kubectl get deployment -n data-platform mlflow -o yaml | grep -A 5 resources:

# Monitor resource usage
kubectl top pods -n data-platform -l app=mlflow

# Check for memory leaks in logs
kubectl logs -n data-platform -l app=mlflow | grep -i memory
```

#### Solutions

**Increase Memory Limits**

```bash
# Update deployment
kubectl patch deployment -n data-platform mlflow --type json -p \
  '[{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/memory","value":"4Gi"}]'

# Increase requests too
kubectl patch deployment -n data-platform mlflow --type json -p \
  '[{"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/memory","value":"2Gi"}]'

# Verify changes
kubectl get deployment -n data-platform mlflow -o yaml | grep -A 2 resources:
```

**Reduce Concurrent Connections**

```bash
# Edit deployment and adjust gunicorn workers
kubectl edit deployment -n data-platform mlflow

# Change this in env:
env:
  - name: GUNICORN_CMD_ARGS
    value: "--workers=2 --threads=1 --timeout=300"  # Reduced from 4 workers

# Save and exit (kubectl will apply automatically)
```

---

### Issue 6: Slow MLFlow UI / Timeouts

#### Symptoms
- MLFlow UI takes long time to load
- Timeout errors when browsing experiments
- Dashboard feels sluggish

#### Diagnosis

```bash
# Check CPU usage
kubectl top pods -n data-platform -l app=mlflow

# Check response times
# Port forward and use browser dev tools
kubectl port-forward -n data-platform svc/mlflow 5000:5000

# Check for database query issues
# (Run these in PostgreSQL)
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U mlflow -d mlflow -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

#### Solutions

**Scale Up Replicas**

```bash
# Increase replicas for load distribution
kubectl scale deployment -n data-platform mlflow --replicas=3

# Verify scaling
kubectl get pods -n data-platform -l app=mlflow
```

**Increase Gunicorn Workers**

```bash
kubectl set env deployment/mlflow -n data-platform \
  GUNICORN_CMD_ARGS="--workers=8 --threads=2 --timeout=300"
```

**Clean Up Old Data** (if database is very large)

```bash
# Archive old runs (via MLFlow UI or API)
# Or delete old experiments:
# kubectl exec mlflow-<pod> -- mlflow experiments delete --experiment-ids=<id>

# Vacuum PostgreSQL
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U mlflow -d mlflow -c "VACUUM ANALYZE;"
```

---

## General Debugging Steps

### 1. Check All Components

```bash
#!/bin/bash
echo "=== MLFlow Deployment ==="
kubectl get deployment -n data-platform mlflow

echo "=== MLFlow Service ==="
kubectl get svc -n data-platform mlflow

echo "=== MLFlow Pods ==="
kubectl get pods -n data-platform -l app=mlflow -o wide

echo "=== MLFlow Ingress ==="
kubectl get ingress -n data-platform mlflow-ingress

echo "=== PostgreSQL ==="
kubectl get pods -n data-platform -l app=postgres

echo "=== MinIO ==="
kubectl get pods -n data-platform -l app=minio
```

### 2. Collect Logs

```bash
# Current logs
kubectl logs -n data-platform -l app=mlflow

# Previous logs (if pod restarted)
kubectl logs -n data-platform -l app=mlflow --previous

# All containers
kubectl logs -n data-platform -l app=mlflow --all-containers

# Follow logs in real-time
kubectl logs -n data-platform -l app=mlflow -f
```

### 3. Test Network Connectivity

```bash
# Test from MLFlow pod to PostgreSQL
kubectl exec -n data-platform mlflow-<pod> -- \
  nc -zv postgres-shared-service.data-platform.svc.cluster.local 5432

# Test from MLFlow pod to MinIO
kubectl exec -n data-platform mlflow-<pod> -- \
  nc -zv minio.minio.svc.cluster.local 9000

# Test from DolphinScheduler to MLFlow
kubectl exec -n data-platform dolphinscheduler-worker-<pod> -- \
  nc -zv mlflow.data-platform.svc.cluster.local 5000
```

### 4. Check Resource Status

```bash
# Check node status
kubectl get nodes

# Check namespace resources
kubectl describe namespace data-platform

# Check resource quotas
kubectl get resourcequota -n data-platform

# Check PVC status (if using persistent volumes)
kubectl get pvc -n data-platform | grep mlflow
```

## Performance Tuning

### Optimize PostgreSQL

```bash
# Connection pooling in MLFlow config
# (Edit deployment environment)
MLFLOW_BACKEND_STORE_URI="postgresql+psycopg2://mlflow:password@postgres:5432/mlflow?connect_timeout=10&application_name=mlflow"
```

### Optimize MinIO

```bash
# Increase MinIO object cache
kubectl set env deployment/minio -n data-platform \
  MINIO_DRIVE_CACHE_ENABLED="on" \
  MINIO_DRIVE_CACHE_SIZE="512MB"

# Restart MinIO
kubectl rollout restart deployment -n data-platform minio
```

### Enable MLFlow Caching

```yaml
# Add to MLFlow deployment env:
- name: MLFLOW_ARTIFACT_CACHE_ENABLED
  value: "true"
- name: MLFLOW_ARTIFACT_CACHE_SIZE
  value: "536870912"  # 512 MB
```

## Getting Help

### Check Logs First

1. `kubectl logs -n data-platform -l app=mlflow`
2. `kubectl logs -n data-platform postgres-shared-<pod>`
3. `kubectl logs -n data-platform minio-<pod>`

### Check Kubernetes Events

```bash
kubectl get events -n data-platform --sort-by='.lastTimestamp' | tail -20
```

### Review Ingress Configuration

```bash
kubectl describe ingress -n data-platform mlflow-ingress
```

### Contact Support

Provide:
- Output of diagnostic commands above
- Pod descriptions and events
- MLFlow, PostgreSQL, and MinIO logs
- Resource usage info (top output)
- Kubernetes cluster version
