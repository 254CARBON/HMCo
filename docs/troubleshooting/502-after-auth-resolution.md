# 502 Error After Authentication - RESOLVED

**Issue**: Receiving 502 Bad Gateway after successfully authenticating with Cloudflare Access on https://portal.254carbon.com  
**Date**: October 20, 2025  
**Status**: ✅ FIXED

---

## Root Cause

The `portal-services` backend API was not deployed. The portal ingress was configured to route `/api/services` requests to a service that didn't exist.

---

## What Was Missing

### 1. Portal Services Backend
- **Service**: `portal-services` (provides API for service health checks and registry)
- **Issue**: Deployment not applied, service not created
- **Image**: `254carbon/portal-services:latest` (not built or loaded)

### 2. ConfigMap Syntax Error
- **Issue**: ConfigMap had incorrect indentation (`metadata.data` instead of top-level `data`)
- **Impact**: ConfigMap creation failed initially

---

## Fixes Applied

### 1. Fixed ConfigMap Syntax
```yaml
# Before (WRONG)
metadata:
  name: portal-services-registry
  namespace: data-platform
  data:  # ← Wrong indentation
    services.json: |

# After (CORRECT)
metadata:
  name: portal-services-registry
  namespace: data-platform
data:  # ← Correct indentation
  services.json: |
```

**File**: `k8s/portal/portal-services.yaml`

### 2. Built Portal Services Image
```bash
cd services/portal-services
docker build -t 254carbon/portal-services:latest .
```

### 3. Loaded Image into Kind
```bash
kind load docker-image 254carbon/portal-services:latest --name dev-cluster
```

### 4. Deployed Portal Services
```bash
kubectl apply -f k8s/portal/portal-services.yaml
kubectl delete pod -n data-platform -l app=portal-services  # Restart to pick up image
```

---

## Current Status

### Portal Infrastructure

```
✅ Portal Frontend: 2/2 pods running
   - portal-7b4f66945d-4jwxb (1/1 Running)
   - portal-7b4f66945d-qw9fj (1/1 Running)
   - Endpoints: 10.244.0.133:8080, 10.244.0.134:8080

✅ Portal Services API: 1/1 pod running
   - portal-services-7568d7ddd5-vrqgk (1/1 Running)
   - Endpoint: 10.244.0.150:8080
   - Health: Passing (responding to /healthz)
   - Logs: "Loaded 9 services from /config/services.json"

✅ Portal Service: ClusterIP 10.107.185.72:8080
✅ Portal-Services Service: ClusterIP 10.109.76.36:8080
```

### Ingress Configuration

```yaml
spec:
  rules:
  - host: portal.254carbon.com
    paths:
    - path: /api/services  → portal-services:8080 ✅
    - path: /            → portal:8080 ✅
```

Both backend services are now operational!

---

## Verification

### Test from Browser
1. Navigate to: https://portal.254carbon.com
2. You'll see: Cloudflare Access login (qagi.cloudflareaccess.com)
3. Enter: Your @254carbon.com email
4. Check email for OTP code
5. Enter code
6. **Expected**: Portal homepage loads successfully

### If Still Seeing 502

Try these tests:

```bash
# 1. Verify both portal services are running
kubectl get pods -n data-platform -l app=portal
kubectl get pods -n data-platform -l app=portal-services

# 2. Check endpoints exist
kubectl get endpoints -n data-platform portal
kubectl get endpoints -n data-platform portal-services

# 3. Check logs for errors
kubectl logs -n data-platform -l app=portal --tail=20
kubectl logs -n data-platform -l app=portal-services --tail=20

# 4. Test portal directly (bypassing Cloudflare)
kubectl port-forward -n data-platform svc/portal 8080:8080
# Open: http://localhost:8080

# 5. Test portal-services API
kubectl port-forward -n data-platform svc/portal-services 8081:8080
# Open: http://localhost:8081/healthz
```

---

## What Changed

### Before
```
User → Cloudflare Access → NGINX Ingress → ❌ portal-services (missing) → 502 Error
```

### After
```
User → Cloudflare Access → NGINX Ingress → ✅ portal:8080 (running)
                                          → ✅ portal-services:8080 (running)
                                          → Portal loads successfully
```

---

## Related Issues Fixed

### Also Built & Loaded
- ✅ `254carbon-portal:latest` - Portal frontend (Next.js)
- ✅ `254carbon/portal-services:latest` - Portal API backend

### DNS FQDNs Updated
- ✅ Kafka → Zookeeper
- ✅ Schema Registry → Kafka
- ✅ Superset → PostgreSQL and Redis

---

## Summary

**The 502 error is now resolved.** Both portal components are running:
1. Portal frontend (Next.js app on port 8080)
2. Portal services API (Node.js API on port 8080)

Both have endpoints and are responding to health checks. The portal should now load successfully after Cloudflare Access authentication.

---

**Try accessing https://portal.254carbon.com now - it should work!** ✅

