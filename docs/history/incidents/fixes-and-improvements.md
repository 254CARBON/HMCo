# Deployment Fixes and Improvements (Archived)

## Issues Found and Fixed

### 1. RBAC Resource Errors ✅ FIXED

**Issue**: Role and RoleBinding resources had "strict decoding error: unknown field 'spec'"

**Root Cause**: RBAC resources were incorrectly placed in the ingestion recipe ConfigMap file with malformed syntax.

**Fix Applied**:
- Separated RBAC resources into dedicated file: `k8s/rbac/datahub-ingestion-rbac.yaml`
- Fixed YAML structure with proper "rules" field for Role
- Applied corrected resources to cluster

**Status**: ✅ Resolved - RBAC now applied successfully

---

### 2. ImagePullBackOff Error ⚠️ NEEDS ATTENTION

**Issue**: Iceberg REST Catalog pod stuck in ImagePullBackOff

**Likely Causes**:
- Docker Hub rate limiting (very common)
- Network connectivity issues
- Image not available in specified registry

**Solutions Available**:

#### Option A: Use Alternative Registry (Recommended)
```bash
kubectl set image deployment/iceberg-rest-catalog \
  -n data-platform \
  iceberg-rest-catalog=quay.io/tabulario/iceberg-rest:0.6.0 \
  --record

kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
```

#### Option B: Use Fix Script
```bash
chmod +x scripts/fix-image-pullback.sh
./scripts/fix-image-pullback.sh
```

#### Option C: Wait for Rate Limit Reset
Docker Hub rate limits reset after ~6 hours. Then:
```bash
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
```

---

## Improvements Made

### 1. Enhanced Documentation ✅

**New/Updated Files**:
- `DEPLOYMENT_TROUBLESHOOTING.md` - Comprehensive troubleshooting guide
- `FIXES_AND_IMPROVEMENTS.md` - This file
- `scripts/fix-image-pullback.sh` - Automated fix script

**Coverage**:
- ImagePullBackOff issues and solutions
- Connection/networking problems
- Resource constraints
- Configuration issues
- Health check procedures
- Quick recovery steps

### 2. RBAC Organization ✅

**Improvements**:
- Separated RBAC into dedicated file
- Proper YAML structure
- Reusable for other components
- Clear documentation

**File Structure**:
```
k8s/rbac/
└── datahub-ingestion-rbac.yaml
```

### 3. Automated Recovery Script ✅

**New File**: `scripts/fix-image-pullback.sh`

**Features**:
- Tries multiple image registries
- Automatic rollout monitoring
- API connectivity verification
- Detailed error reporting
- 10-minute timeout with fallback

---

## Deployment Status

### Current Status

| Component | Status | Issue | Solution |
|-----------|--------|-------|----------|
| Secrets | ✅ Created | None | N/A |
| PostgreSQL | ✅ Configured | None | N/A |
| MinIO Init Job | ✅ Created | None | N/A |
| Trino | ✅ Configured | None | N/A |
| DataHub Ingestion | ✅ Created | RBAC syntax fixed | Applied fixes |
| SeaTunnel | ✅ Configured | None | N/A |
| **Iceberg REST** | ⚠️ ImagePullBackOff | Image pull fails | See solutions |

### Next Steps

1. **Immediate** (5 minutes):
   ```bash
   # Fix ImagePullBackOff
   kubectl set image deployment/iceberg-rest-catalog \
     -n data-platform \
     iceberg-rest-catalog=quay.io/tabulario/iceberg-rest:0.6.0
   
   kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
   ```

2. **Verify** (2 minutes):
   ```bash
   kubectl get pod -n data-platform -l app=iceberg-rest-catalog --watch
   ```

3. **Test** (5 minutes):
   ```bash
   kubectl port-forward svc/iceberg-rest-catalog 8181:8181 &
   curl http://localhost:8181/v1/config
   ```

---

## Additional Improvements Recommended

### 1. ImagePullPolicy

**Current**: Default (IfNotPresent)  
**Recommended**: Add explicit policy to prevent future issues

Update `k8s/data-lake/iceberg-rest.yaml`:
```yaml
spec:
  containers:
  - name: iceberg-rest-catalog
    image: quay.io/tabulario/iceberg-rest:0.6.0
    imagePullPolicy: IfNotPresent  # Add this
```

### 2. Pre-pull Images

For production, pre-pull images on all nodes:
```bash
docker pull quay.io/tabulario/iceberg-rest:0.6.0
docker pull trinodb/trino:436
docker pull acryldata/datahub-gms:head
docker pull acryldata/datahub-ingestion:latest
```

### 3. Image Registry Mirror

Consider using a private registry mirror:
```bash
# Configure in kubelet for faster, more reliable pulls
# Edit /etc/docker/daemon.json or kubelet config
{
  "registry-mirrors": ["your-mirror.com"]
}
```

### 4. Add Health Monitoring Script

**Created**: Already included in this repo
- Use `scripts/fix-image-pullback.sh` for automated fixes
- Create additional monitoring scripts as needed

### 5. Documentation Enhancements

**Completed**:
- ✅ Troubleshooting guide added
- ✅ Quick recovery procedures documented
- ✅ Automated recovery script provided
- ✅ Issue categorization and solutions mapped

---

## Testing Recommendations

### After ImagePullBackOff Fix

1. **Verify Pod Health**
   ```bash
   kubectl get pod -n data-platform iceberg-rest-catalog-xxx
   # Status should be: Running, Ready: 1/1
   ```

2. **Test API**
   ```bash
   kubectl port-forward svc/iceberg-rest-catalog 8181:8181 &
   curl http://localhost:8181/v1/config
   # Should return: {"defaults": {}, "overrides": {}}
   ```

3. **Test from Trino**
   ```bash
   # Access Trino CLI or UI
   SHOW CATALOGS;  # Should include "iceberg"
   ```

4. **Test from DataHub**
   ```bash
   # Trigger ingestion job
   kubectl delete job datahub-iceberg-ingestion-test -n data-platform
   kubectl apply -f k8s/datahub/iceberg-ingestion-recipe.yaml
   ```

---

## Monitoring Post-Deployment

### Health Check Frequency

- **Critical**: Every 5 minutes (automated)
- **Warning**: Every hour (manual or automated)
- **Info**: Every day (daily review)

### Key Metrics to Monitor

```bash
# Pod status
kubectl get pod -n data-platform -l app=iceberg-rest-catalog

# Resource usage
kubectl top pod -n data-platform iceberg-rest-catalog-xxx

# Recent events
kubectl get events -n data-platform --sort-by='.lastTimestamp'

# API responsiveness
curl -s -o /dev/null -w "%{http_code}" http://localhost:8181/v1/config
```

---

## Documentation Files Created/Updated

### New Files
1. `DEPLOYMENT_TROUBLESHOOTING.md` - 200+ lines
2. `FIXES_AND_IMPROVEMENTS.md` - This file
3. `scripts/fix-image-pullback.sh` - Automated recovery
4. `k8s/rbac/datahub-ingestion-rbac.yaml` - Corrected RBAC

### Updated Files
1. `k8s/datahub/iceberg-ingestion-recipe.yaml` - Removed incorrect RBAC

---

## Quick Reference: Common Commands

```bash
# Check status
kubectl get pods -n data-platform | grep iceberg

# View logs
kubectl logs -f deployment/iceberg-rest-catalog -n data-platform

# Restart if needed
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform

# Scale up
kubectl scale deployment iceberg-rest-catalog -n data-platform --replicas=3

# Check resource usage
kubectl top pod -n data-platform iceberg-rest-catalog-xxx

# Port-forward for testing
kubectl port-forward svc/iceberg-rest-catalog 8181:8181 &
```

---

## Deployment Checklist (Updated)

- [x] Create secrets
- [x] Initialize PostgreSQL
- [x] Initialize MinIO buckets
- [x] Configure Iceberg REST Catalog
- [x] Configure Trino
- [x] Configure DataHub (with RBAC fix)
- [x] Configure SeaTunnel
- [x] Fix RBAC errors
- [ ] **Fix ImagePullBackOff** - Use one of the provided solutions
- [ ] Verify all pods running
- [ ] Run end-to-end tests
- [ ] Complete security hardening
- [ ] Set up monitoring

---

## Support Resources

1. **Troubleshooting**: `DEPLOYMENT_TROUBLESHOOTING.md`
2. **Quick Fixes**: `scripts/fix-image-pullback.sh`
3. **Component Guides**: Individual guide files (Trino, DataHub, SeaTunnel, etc.)
4. **Testing**: `ICEBERG_INTEGRATION_TEST_GUIDE.md`
5. **Operations**: `ICEBERG_OPERATIONS_RUNBOOK.md`

---

**Last Updated**: October 19, 2025  
**Status**: Fixes Applied, ImagePullBackOff Resolution Pending  
**Next Action**: Execute ImagePullBackOff fix (see "Immediate" steps above)
