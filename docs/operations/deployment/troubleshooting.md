# Deployment Troubleshooting Guide

## ImagePullBackOff Issues

### Issue: Iceberg REST Catalog Pod in ImagePullBackOff

**Error**: Pod shows `ImagePullBackOff` status  
**Cause**: Docker image cannot be pulled from registry

### Solutions

#### Solution 1: Check Docker Registry Connectivity

```bash
# Check pod event details
kubectl describe pod -n data-platform iceberg-rest-catalog-xxx

# Look for error messages like:
# - "Failed to pull image"
# - "ImagePullBackOff"
# - "Failed to download metadata for ..."
```

#### Solution 2: Docker Hub Rate Limiting

If using Docker Hub, you may hit rate limits. Options:

**Option A: Use Alternative Image Registry**

```bash
# Update Iceberg REST Catalog image to Quay.io mirror
kubectl set image deployment/iceberg-rest-catalog \
  -n data-platform \
  iceberg-rest-catalog=quay.io/tabulario/iceberg-rest:0.6.0
```

**Option B: Configure Docker Credentials**

```bash
# Apply Docker registry secret (already exists)
kubectl apply -f k8s/storage/docker-credentials.yaml

# Update deployment to use imagePullSecrets
kubectl patch deployment iceberg-rest-catalog -n data-platform \
  -p '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"docker-registry-secret"}]}}}}'
```

**Option C: Use Local/Private Registry**

Tag and push image to your private registry:

```bash
# Pull image locally
docker pull tabulario/iceberg-rest:0.6.0

# Tag with your registry
docker tag tabulario/iceberg-rest:0.6.0 \
  your-registry.com/iceberg-rest:0.6.0

# Push to private registry
docker push your-registry.com/iceberg-rest:0.6.0

# Update deployment
kubectl set image deployment/iceberg-rest-catalog \
  -n data-platform \
  iceberg-rest-catalog=your-registry.com/iceberg-rest:0.6.0
```

#### Solution 3: Wait for Rate Limit Reset

```bash
# Docker Hub rate limits reset after ~6 hours
# Monitor pod status:
kubectl get pod -n data-platform -l app=iceberg-rest-catalog --watch

# Once rate limit resets, manually trigger pod restart:
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
```

#### Solution 4: Pre-pull Images (Recommended for Production)

```bash
# Pull all images on cluster nodes
docker pull tabulario/iceberg-rest:0.6.0
docker pull trinodb/trino:436
docker pull acryldata/datahub-gms:head
docker pull acryldata/datahub-frontend-react:head
docker pull apache/seatunnel:2.3.12
docker pull acryldata/datahub-ingestion:latest

# This prevents pull delays and rate limit issues
```

### Verify Fix

```bash
# Check pod status
kubectl get pod -n data-platform -l app=iceberg-rest-catalog

# Should show status: Running

# Verify API is responding
kubectl port-forward svc/iceberg-rest-catalog 8181:8181 &
sleep 2
curl http://localhost:8181/v1/config
```

## RBAC Issues

### Issue: Role/RoleBinding Errors

**Error**: "strict decoding error: unknown field 'spec'"

**Root Cause**: Invalid YAML structure in RBAC resources

### Fix

The problematic RBAC resources have been separated into a dedicated file:

```bash
# Apply corrected RBAC file
kubectl apply -f k8s/rbac/datahub-ingestion-rbac.yaml

# Verify RBAC was created
kubectl get role -n data-platform
kubectl get rolebinding -n data-platform
```

## Common Deployment Errors

### Error: Connection Refused

```bash
# Check if service exists
kubectl get svc -n data-platform iceberg-rest-catalog

# Check if endpoints exist
kubectl get endpoints -n data-platform iceberg-rest-catalog

# Test DNS resolution in pod
kubectl exec -it iceberg-rest-catalog-xxx -n data-platform -- \
  nslookup iceberg-rest-catalog.data-platform.svc.cluster.local
```

### Error: PostgreSQL Connection Failed

```bash
# Verify PostgreSQL is running
kubectl get pod -n data-platform -l app=postgres-shared

# Test connection from pod
kubectl exec -it postgres-shared-xxx -n data-platform -- \
  psql -U iceberg_user -d iceberg_rest -c "SELECT 1"

# Check credentials in secret
kubectl get secret postgres-shared-secret -n data-platform -o jsonpath='{.data.password}' | base64 -d
```

### Error: MinIO Connection Failed

```bash
# Verify MinIO is running
kubectl get pod -n data-platform -l app=minio

# Check MinIO secret
kubectl get secret minio-secret -n data-platform -o yaml

# Test S3 connectivity
kubectl exec -it iceberg-rest-catalog-xxx -n data-platform -- \
  curl -v http://minio-service:9000/minio/health/live
```

## Container Issues

### High Memory Usage

```bash
# Check current memory usage
kubectl top pod -n data-platform iceberg-rest-catalog-xxx

# If > 85% of limit, increase resource limits:
kubectl set resources deployment iceberg-rest-catalog \
  -n data-platform \
  --limits=memory=2Gi,cpu=1000m \
  --requests=memory=1Gi,cpu=500m

# Restart pod to apply changes
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
```

### Crashes/Restarts

```bash
# Check pod logs
kubectl logs -n data-platform iceberg-rest-catalog-xxx

# Check previous logs (if crashed)
kubectl logs -n data-platform iceberg-rest-catalog-xxx --previous

# View last N lines
kubectl logs -n data-platform iceberg-rest-catalog-xxx --tail=100

# Follow logs in real-time
kubectl logs -f -n data-platform deployment/iceberg-rest-catalog
```

## Network Issues

### Cannot Reach Iceberg REST Catalog

```bash
# Verify service is accessible
kubectl get svc -n data-platform iceberg-rest-catalog

# Port-forward for local testing
kubectl port-forward -n data-platform svc/iceberg-rest-catalog 8181:8181 &

# Test API
curl http://localhost:8181/v1/config

# Check network policies (if enabled)
kubectl get networkpolicy -n data-platform

# Verify pod can reach other services
kubectl exec -it iceberg-rest-catalog-xxx -n data-platform -- \
  curl http://postgres-shared-service:5432
```

## Configuration Issues

### Invalid Configuration

```bash
# Check ConfigMap for errors
kubectl get configmap iceberg-rest-catalog-docs -n data-platform -o yaml

# Check environment variables in pod
kubectl exec iceberg-rest-catalog-xxx -n data-platform -- env | sort

# Verify secret values
kubectl get secret minio-secret -n data-platform -o yaml
```

### PostgreSQL Permission Error (`permission denied for schema public`)

- **Symptom**: Iceberg REST Catalog pods stay in `Running`/`NotReady` and logs show `permission denied for schema public` during Flyway or JDBC initialization.
- **Root Cause**: Earlier manifests relied on `SET ROLE iceberg_app` during connection initialization. In-cluster connection pools reuse sessions before the role switch runs, so the database reports missing privileges on the `public` schema.
- **Fix**:
  1. Make sure the `postgres-shared-init` ConfigMap grants schema privileges directly to `iceberg_user` (see `k8s/shared/postgres/postgres-shared.yaml:142`).
  2. Ensure the deployment sets `CATALOG_JDBC_INITIALIZE_SQL="SET search_path TO iceberg_catalog, public"` plus the Flyway schema overrides (see `k8s/data-lake/iceberg-rest.yaml:70`).
  3. Restart the catalog after applying the ConfigMap and deployment updates:
     ```bash
     kubectl rollout restart deployment/postgres-shared -n data-platform
     kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
     ```

## Monitoring Health

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "=== Iceberg Health Check ==="
echo ""

# 1. Check pod status
echo "1. Pod Status:"
kubectl get pod -n data-platform -l app=iceberg-rest-catalog

# 2. Check API
echo ""
echo "2. API Health:"
kubectl port-forward -n data-platform svc/iceberg-rest-catalog 8181:8181 &
PID=$!
sleep 2
if curl -s http://localhost:8181/v1/config > /dev/null; then
  echo "✓ API is responding"
else
  echo "✗ API is not responding"
fi
kill $PID 2>/dev/null

# 3. Check PostgreSQL
echo ""
echo "3. Database Status:"
kubectl get pod -n data-platform -l app=postgres-shared

# 4. Check MinIO
echo ""
echo "4. Storage Status:"
kubectl get pod -n data-platform -l app=minio

# 5. Check recent events
echo ""
echo "5. Recent Events:"
kubectl get events -n data-platform --sort-by='.lastTimestamp' | tail -5
```

## Quick Recovery Steps

### If Iceberg is down but other services are up:

```bash
# 1. Check logs
kubectl logs -n data-platform deployment/iceberg-rest-catalog | tail -50

# 2. Restart deployment
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform

# 3. Monitor restart
kubectl rollout status deployment/iceberg-rest-catalog -n data-platform

# 4. Verify health
kubectl get pod -n data-platform -l app=iceberg-rest-catalog

# 5. Test API
kubectl port-forward svc/iceberg-rest-catalog 8181:8181 &
curl http://localhost:8181/v1/config
```

## Getting Help

If issues persist:

1. **Collect debug information**:
   ```bash
   # Describe pod for detailed status
   kubectl describe pod -n data-platform iceberg-rest-catalog-xxx
   
   # Get all events
   kubectl get events -n data-platform
   
   # Export logs
   kubectl logs -n data-platform deployment/iceberg-rest-catalog > iceberg-logs.txt
   ```

2. **Check logs for error messages**: Look for ERROR or WARN level logs

3. **Review documentation**: See component-specific guides (ICEBERG_DEPLOYMENT.md, etc.)

4. **Contact platform team**: Provide pod description, logs, and steps taken

## References

- [Kubernetes Pod Troubleshooting](https://kubernetes.io/docs/tasks/debug-application-cluster/debug-pods/)
- [Kubectl Cheatsheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Iceberg REST Catalog Documentation](https://github.com/tabular-io/iceberg-rest-image)
