# Image Mirroring Guide - Manual Authentication Required

## Overview

The image mirroring process is blocked on Harbor registry authentication. This guide provides step-by-step instructions to complete the image mirroring manually.

**Status**: ⏳ Authentication Required
**Estimated Time**: 2-3 hours
**Impact**: Eliminates Docker Hub dependency and rate limiting

## Problem Identified

The Harbor registry requires authentication but the current setup doesn't have proper credentials configured. The registry responds with "401 Unauthorized" for all requests.

## Solution Steps

### Step 1: Access Harbor Web Interface

**Option A: Port Forward (Current Setup)**
```bash
# Harbor core UI
kubectl port-forward svc/harbor-core 8080:80 -n registry

# Harbor registry API
kubectl port-forward svc/harbor-registry 5000:5000 -n registry
```

**Option B: Direct Access**
- Harbor UI: http://minio.254carbon.com (if DNS configured)
- Registry API: http://harbor-registry.registry:5000 (cluster internal)

### Step 2: Login to Harbor

1. **Access Harbor UI**: http://localhost:8080 (or http://minio.254carbon.com)
2. **Login Credentials**:
   - Username: `admin`
   - Password: `ChangeMe123!`

### Step 3: Create Robot Account

1. **Navigate to Administration**:
   - Click "Administration" in the top menu
   - Go to "Robot Accounts"

2. **Create New Robot Account**:
   - Click "New Robot Account"
   - Name: `docker-mirror-robot`
   - Description: "Robot account for Docker image mirroring"
   - Select Projects: All projects or create "254carbon" project

3. **Generate Token**:
   - Click "Generate Token"
   - Copy the generated token (keep it secure)

### Step 4: Configure Docker Authentication

**Using Robot Account Token**:
```bash
# Login with robot account
docker login localhost:5000 --username docker-mirror-robot --password <ROBOT_TOKEN>

# Or create a secret for Kubernetes
kubectl create secret docker-registry harbor-credentials \
  --docker-server=localhost:5000 \
  --docker-username=docker-mirror-robot \
  --docker-password=<ROBOT_TOKEN> \
  --docker-email=admin@254carbon.com \
  -n data-platform
```

### Step 5: Execute Image Mirroring

**Run the mirroring script**:
```bash
# Ensure you're logged in
docker login localhost:5000 --username docker-mirror-robot --password <ROBOT_TOKEN>

# Execute mirroring
./scripts/mirror-images.sh localhost:5000 harbor
```

### Step 6: Verify Mirroring Success

**Check script output**:
- Should show "Successful: X" where X > 0
- Should show "Skipped: 0" or minimal skips
- No "Failed" entries

**Verify images in Harbor**:
1. Go to Harbor UI → Projects → 254carbon
2. Should see mirrored images in the project

### Step 7: Update Deployment Configurations

**Update image references**:
```bash
# For each deployment, update image URLs
kubectl set image deployment/DEPLOYMENT_NAME -n NAMESPACE \
  IMAGE_NAME=localhost:5000/IMAGE_NAME:latest

# Or update YAML files manually
# Replace docker.io/IMAGE with localhost:5000/IMAGE
```

**Example**:
```bash
kubectl set image deployment/trino -n data-platform \
  trino=localhost:5000/trinodb/trino:424
```

## Troubleshooting

### Authentication Issues

**Problem**: "401 Unauthorized" errors
**Solution**:
1. Verify robot account exists and is active
2. Check token hasn't expired
3. Ensure robot has access to the project

**Problem**: Connection timeout
**Solution**:
1. Ensure port forwards are active
2. Check Harbor pods are running
3. Verify network connectivity

### Registry Issues

**Problem**: Registry not responding
**Solution**:
1. Check Harbor registry pod logs: `kubectl logs -n registry -l app=harbor-registry`
2. Verify registry service: `kubectl get svc -n registry harbor-registry`
3. Restart registry if needed: `kubectl rollout restart deployment/harbor-registry -n registry`

### Docker Issues

**Problem**: Docker login fails
**Solution**:
1. Clear Docker credentials: `docker logout localhost:5000`
2. Retry login with correct credentials
3. Check Docker daemon is running

## Alternative Approaches

### Option 1: Use Existing Harbor Credentials
If you have access to the original Harbor setup, use those credentials instead of creating a robot account.

### Option 2: Configure Registry for Anonymous Access (Development Only)
For development environments, you can configure Harbor to allow anonymous pulls, but this is not recommended for production.

### Option 3: Use Different Registry
If Harbor authentication proves too complex, consider using:
- Docker Hub (current setup)
- AWS ECR
- Google Container Registry
- Azure Container Registry

## Security Considerations

### Robot Account Security
- Use strong, unique passwords for robot accounts
- Limit robot account permissions to specific projects
- Rotate robot account tokens regularly
- Never commit robot account credentials to version control

### Registry Security
- Enable HTTPS for production registries
- Configure proper authentication and authorization
- Monitor registry access logs
- Implement image scanning for vulnerabilities

## Next Steps After Mirroring

1. **Update all deployments** to use mirrored images
2. **Restart affected services** to pick up new images
3. **Verify functionality** of all services
4. **Clean up old images** from Docker Hub (optional)
5. **Set up automated image mirroring** for future updates

## Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `mirror-images.sh` | Main mirroring script | `scripts/` |
| `services.json` | Service registry | Root directory |
| This guide | Complete procedures | `IMAGE_MIRRORING_GUIDE.md` |

## Support

For issues during image mirroring:
1. Check Harbor UI for project and robot account status
2. Verify Docker login credentials
3. Review script logs for specific error messages
4. Check network connectivity and port forwards

---
**Status**: ⏳ Manual Authentication Required
**Estimated Time**: 2-3 hours
**Dependencies**: Harbor UI access and robot account creation
