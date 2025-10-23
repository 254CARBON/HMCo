# Cloudflare Tunnel Configuration - Action Required

**Date**: October 24, 2025  
**Status**: ⚠️ Requires Manual Intervention

---

## Issue Summary

The Cloudflare tunnel deployment is failing because it requires either:
1. The original `cert.pem` (origin certificate) file
2. A valid tunnel token (not an API token)

---

## What Was Attempted

1. ✅ Created nginx ingress controller (working)
2. ✅ Created all service ingresses (working internally)
3. ⚠️ Tried to configure Cloudflare tunnel with API token (failed)

**Error**: `Error decoding origin cert: missing token in the certificate`

---

## Current State

### What Works:
- ✅ Internal service access via ingress controller
- ✅ NodePort access (ports 31317/80, 30512/443)
- ✅ Port-forwarding for testing
- ✅ All service ingresses configured

### What Needs Fixing:
- ⚠️ External access via *.254carbon.com domains
- ⚠️ Cloudflare tunnel authentication

---

## Solution Options

### Option 1: Get Tunnel Token (Recommended)
If you have access to Cloudflare dashboard:

1. Go to: https://one.dash.cloudflare.com/
2. Navigate to: Zero Trust → Networks → Tunnels
3. Find tunnel: `254carbon-cluster` (ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5)
4. Click "Configure" → Get tunnel token
5. The token will be a long string starting with `eyJ...`

Then run:
```bash
kubectl create secret generic cloudflare-tunnel-token \
  --from-literal=token='<ACTUAL_TUNNEL_TOKEN>' \
  -n cloudflare-tunnel --dry-run=client -o yaml | kubectl apply -f -

kubectl patch deployment cloudflared -n cloudflare-tunnel --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": ["tunnel", "--no-autoupdate", "run", "--token", "$(TUNNEL_TOKEN)"]},
  {"op": "replace", "path": "/spec/template/spec/containers/0/env", "value": [
    {"name": "TUNNEL_TOKEN", "valueFrom": {"secretKeyRef": {"name": "cloudflare-tunnel-token", "key": "token"}}}
  ]}
]'

kubectl scale deployment cloudflared -n cloudflare-tunnel --replicas=2
```

### Option 2: Recreate Tunnel
If you can't get the token:

```bash
# Using cloudflared CLI locally with API token
export CLOUDFLARE_API_TOKEN=xCY-2jiRPLgmjLSjg3ThTQbZDAKJxtked0yu0O9k

# Create new tunnel
cloudflared tunnel create 254carbon-cluster-new

# This will generate credentials.json
# Copy it to K8s secret
kubectl create secret generic cloudflare-tunnel-credentials \
  --from-file=credentials.json=~/.cloudflared/<tunnel-id>.json \
  -n cloudflare-tunnel --dry-run=client -o yaml | kubectl apply -f -

# Update tunnel configuration in Cloudflare dashboard
# Point *.254carbon.com DNS to new tunnel
```

### Option 3: Use Alternative Ingress (Temporary)
Until Cloudflare is fixed, use NodePort:

```bash
# Access services via NodePort
curl -H "Host: dolphin.254carbon.com" http://<NODE-IP>:31317
curl -H "Host: trino.254carbon.com" http://<NODE-IP>:31317
```

Or use port-forward for testing:
```bash
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
curl -H "Host: dolphin.254carbon.com" http://localhost:8080
```

---

## Existing Tunnel Configuration

**Tunnel ID**: 291bc289-e3c3-4446-a9ad-8e327660ecd5  
**Tunnel Name**: 254carbon-cluster  
**Account ID**: 0c93c74d5269a228e91d4bf91c547f56

**Configured Domains**:
- portal.254carbon.com
- www.254carbon.com
- grafana.254carbon.com
- superset.254carbon.com
- datahub.254carbon.com
- trino.254carbon.com
- doris.254carbon.com
- dolphin.254carbon.com
- minio.254carbon.com
- mlflow.254carbon.com
- spark-history.254carbon.com
- harbor.254carbon.com
- lakefs.254carbon.com

All configured to route to: `http://ingress-nginx-controller.ingress-nginx:80`

---

## Why API Token Didn't Work

The API token `xCY-2jiRPLgmjLSjg3ThTQbZDAKJxtked0yu0O9k` is for:
- Managing Cloudflare resources via API
- Creating/modifying DNS records
- Managing firewall rules

**NOT for**:
- Authenticating tunnel connections
- Running cloudflared daemon

**Tunnel authentication requires**:
- Tunnel-specific token (JWT format, starts with `eyJ...`)
- Or origin certificate (cert.pem)
- Or credentials.json with proper tunnel credentials

---

## Impact Assessment

### Current Impact:
- **Internal Services**: ✅ Fully functional
- **External Access**: ⚠️ Not available via custom domains
- **Critical Services**: ✅ Accessible via NodePort/port-forward
- **Data Ingestion**: ✅ Not blocked - DolphinScheduler works internally

### Priority:
- **For Development/Testing**: Low (NodePort works fine)
- **For Production**: High (need proper external access)

---

## Workaround for Immediate Use

Until Cloudflare tunnel is fixed, access services using:

1. **Port-Forward** (recommended for testing):
   ```bash
   # Terminal 1: Forward ingress controller
   kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
   
   # Terminal 2: Access services
   curl -H "Host: dolphin.254carbon.com" http://localhost:8080
   ```

2. **NodePort** (if you have node access):
   ```bash
   # Get node IP
   NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
   
   # Access services
   curl -H "Host: dolphin.254carbon.com" http://$NODE_IP:31317
   ```

3. **Direct Service Access** (internal only):
   ```bash
   kubectl port-forward -n data-platform svc/dolphinscheduler-api 12345:12345
   curl http://localhost:12345
   ```

---

## Next Steps

1. **Immediate**: Proceed with DolphinScheduler workflow import (doesn't require external access)
2. **Short-term**: Get proper tunnel token from Cloudflare dashboard
3. **Long-term**: Consider backup ingress solution (LoadBalancer, metallb, etc.)

---

## Files Modified

- Scaled cloudflared deployment to 0 replicas
- All ingress resources remain configured and functional
- Nginx ingress controller operational on NodePort

---

**Created**: October 24, 2025 00:45 UTC  
**Status**: Documented, awaiting tunnel token  
**Blocking**: External domain access only  
**Non-blocking**: Development, testing, workflow import

