# SSO Portal Deployment - Debug Fixes

## Issues Encountered and Resolution

### Issue 1: npm ci Error - "package-lock.json not found"
**Error**: `npm ci` command requires an existing `package-lock.json` file

**Root Cause**: Dockerfile used `npm ci` (clean install) which requires a lockfile for reproducible builds

**Fix Applied**:
- Changed Dockerfile to use `npm install` instead of `npm ci`
- This is acceptable when no lockfile exists yet
- Created `.dockerignore` to optimize build context

**File Changed**: `portal/Dockerfile`
```dockerfile
# Before
RUN npm ci

# After
RUN npm install
```

### Issue 2: Module Resolution Error - "@/components/Header"
**Error**: `Module not found: Can't resolve '@/components/Header'`

**Root Cause**: Path aliases defined in `tsconfig.json` weren't being resolved correctly during Docker build

**Fix Applied**:
- Changed all imports from path aliases (`@/...`) to relative imports (`../...`)
- This is more reliable for build environments

**Files Changed**:
- `portal/app/page.tsx` - Changed imports to relative paths
- `portal/components/ServiceGrid.tsx` - Changed `@/lib/services` to `../lib/services`
- `portal/components/ServiceCard.tsx` - Changed `@/lib/services` to `../lib/services`

### Issue 3: globals.css Location
**Error**: `Module not found: Can't resolve './globals.css'`

**Root Cause**: File was created in `styles/` directory but Next.js convention places it in `app/` directory

**Fix Applied**:
- Moved `styles/globals.css` to `app/globals.css`
- Removed empty `styles/` directory

### Issue 4: Kind Cluster Image Not Available
**Error**: `ErrImagePull` / `ErrImageNeverPull`

**Root Cause**: Docker image was built locally but not loaded into the kind Kubernetes cluster

**Fix Applied**:
- Ran `kind load docker-image 254carbon-portal:latest --name dev-cluster`
- Image was then available to Kubernetes pods

**Command Used**:
```bash
kind load docker-image 254carbon-portal:latest --name dev-cluster
```

### Issue 5: Ingress Not Applied
**Issue**: Portal ingress rules weren't created during initial deployment

**Fix Applied**:
- Re-applied ingress manifest: `kubectl apply -f k8s/ingress/portal-ingress.yaml`

## Final Status

### ✅ Deployment Successful

```
NAME                     READY   STATUS    RESTARTS   AGE
portal-8f6d6754b-976dj   1/1     Running   0          2m
portal-8f6d6754b-d4kn6   1/1     Running   0          2m
```

### Service
```
portal   ClusterIP   10.96.166.148   <none>   8080/TCP
```

### Ingress
```
portal-ingress   nginx   254carbon.com,www.254carbon.com,portal.254carbon.com   80, 443
```

## Changes Made Summary

| File | Change | Reason |
|------|--------|--------|
| `portal/Dockerfile` | npm install instead of npm ci | No lock file for clean install |
| `portal/tsconfig.json` | Added `allowJs: true` | Better module resolution |
| `portal/.dockerignore` | Created | Optimize build context |
| `portal/.gitignore` | Created | Standard Node.js ignore patterns |
| `portal/app/page.tsx` | Relative imports | Fix module resolution |
| `portal/components/ServiceGrid.tsx` | Relative imports | Fix module resolution |
| `portal/components/ServiceCard.tsx` | Relative imports | Fix module resolution |
| `portal/styles/globals.css` → `portal/app/globals.css` | Moved file | Next.js convention |
| `k8s/ingress/portal-deployment.yaml` | imagePullPolicy: Never | Use local image |

## Key Lessons

1. **npm ci vs npm install**: Use `npm install` for initial development, `npm ci` for CI/CD with existing lockfile
2. **Path Aliases**: Relative imports are more reliable in build environments than TypeScript path aliases
3. **File Structure**: Follow Next.js conventions (app directory layout)
4. **Kind Cluster**: Local Docker images must be explicitly loaded: `kind load docker-image`
5. **imagePullPolicy**: Set to `Never` for local development, `IfNotPresent` for production

## Verification Commands

```bash
# Check pod status
kubectl get pods -n data-platform -l app=portal

# Check service
kubectl get svc -n data-platform | grep portal

# Check ingress
kubectl get ingress -n data-platform | grep portal

# View logs
kubectl logs -n data-platform -l app=portal -f

# Test connectivity
curl http://10.96.166.148:8080/
```

## Next Steps

1. Configure Cloudflare Access (Phase 2)
2. Set up authentication policies
3. Test end-to-end SSO flow
4. Configure service integrations (Phase 3)

---

**Deployment Completed**: October 19, 2025  
**Portal Status**: ✅ Running and Ready for Phase 2
