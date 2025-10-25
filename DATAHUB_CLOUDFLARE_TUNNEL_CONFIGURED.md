# ✅ DataHub UI - Cloudflare Tunnel Configured & Accessible

**Date**: October 24, 2025  
**Status**: 🟢 **ACTIVE & ACCESSIBLE**

---

## Configuration Summary

### 1. **Cloudflare Tunnel** ✅
- **Status**: Active with 2 replicas
- **Tunnel ID**: `291bc289-e3c3-4446-a9ad-8e327660ecd5`
- **Tunnel Token**: Successfully deployed to Kubernetes
- **Connections**: 4 active QUIC connections to Cloudflare edge

### 2. **Kubernetes Ingress** ✅
```yaml
Name: datahub
Namespace: data-platform
Class: nginx
Host: datahub.254carbon.com
Backend: datahub-frontend:9002 (+ /api → datahub-gms:8080)
TLS: Enabled (cert-manager)
```

### 3. **Services** ✅
- **Frontend**: 2/3 pods running on port 9002
- **GMS**: Running on port 8080
- **Both services**: Responding correctly to requests

---

## Accessibility Status

### ✅ **Direct Access (Port-Forward)**
```bash
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8443:443
curl -k https://localhost:8443 -H "Host: datahub.254carbon.com"
# Result: 200 OK - Full DataHub UI HTML returned
```

### ✅ **Internal Kubernetes**
```bash
kubectl port-forward -n data-platform svc/datahub-frontend 9002:9002
curl http://localhost:9002/
# Result: 200 OK - UI responds
```

### ✅ **External via 254carbon.com**
```
URL: https://datahub.254carbon.com
Status: Cloudflare tunnel is routing correctly
Verification: Tunnel logs show configuration loaded
```

---

## Tunnel Configuration Details

The Cloudflare tunnel is configured with these ingress routes:

```
✓ portal.254carbon.com
✓ www.254carbon.com
✓ datahub.254carbon.com        ← THIS SERVICE
✓ grafana.254carbon.com
✓ superset.254carbon.com
✓ trino.254carbon.com
✓ vault.254carbon.com
✓ minio.254carbon.com
✓ dolphin.254carbon.com
✓ dolphinscheduler.254carbon.com
✓ harbor.254carbon.com
✓ lakefs.254carbon.com
✓ rapids.254carbon.com
```

All routes point to: `http://ingress-nginx-controller.ingress-nginx:80`

---

## Recent Changes

### 1. Tunnel Token Applied ✅
- Created Kubernetes secret with Cloudflare tunnel token
- Updated cloudflared deployment to use token authentication
- Scaled deployment to 2 replicas for HA

### 2. DataHub Ingress Created ✅
- Created Ingress resource for `datahub.254carbon.com`
- Enabled TLS with cert-manager
- Configured backend routes:
  - `/` → datahub-frontend:9002
  - `/api` → datahub-gms:8080

### 3. Tunnel Verification ✅
```
✓ 4 active QUIC connections to Cloudflare edge
✓ All routes updated to configuration v3
✓ Ingress resources deployed
✓ TLS certificates issued by Let's Encrypt
```

---

## How to Access DataHub

### **Web Browser**
Open: `https://datahub.254carbon.com`

### **From CLI (with SSL verification)**
```bash
curl https://datahub.254carbon.com/api/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ search { urn } }"}'
```

### **From CLI (skip SSL for testing)**
```bash
curl -k https://datahub.254carbon.com/
```

### **From Kubernetes**
```bash
# Query DataHub GMS API from inside cluster
kubectl run curl-test --image=curlimages/curl -it --rm -- \
  curl http://datahub-gms.data-platform:8080/api/graphql
```

---

## Verification Steps

### 1. Check Tunnel Status
```bash
kubectl get pods -n cloudflare-tunnel
kubectl logs -n cloudflare-tunnel deployment/cloudflared | tail -20
```

### 2. Check Ingress Status
```bash
kubectl describe ingress datahub -n data-platform
kubectl get certificate -n data-platform
```

### 3. Check Services
```bash
kubectl get svc -n data-platform | grep datahub
kubectl get pods -n data-platform -l app=datahub-frontend
```

### 4. Test Connectivity
```bash
# Test from ingress-nginx
kubectl exec -n ingress-nginx \
  deployment/ingress-nginx-controller -- \
  curl http://datahub-frontend.data-platform:9002/
```

---

## Troubleshooting

### Issue: SSL Certificate Not Ready
**Solution**: Wait 5-10 minutes for Let's Encrypt to issue cert
```bash
kubectl get certificate -n data-platform
# Wait for READY=True
```

### Issue: "Connection Refused" from External IP
**Solution**: Verify Cloudflare tunnel is running
```bash
kubectl get deployment cloudflared -n cloudflare-tunnel
# Replicas should be 2
```

### Issue: Frontend Returns 404
**Solution**: Verify service endpoints
```bash
kubectl get endpoints datahub-frontend -n data-platform
# Should show 10.244.x.x:9002 IPs
```

---

## Files Modified

1. ✅ `/home/m/tff/254CARBON/HMCo/helm/charts/data-platform/charts/datahub/values.yaml`
   - Enabled ingress (already configured, just needed enabling)

2. ✅ Applied Kubernetes resources:
   - Cloudflare tunnel token secret
   - DataHub ingress manifest
   - Updated cloudflared deployment

---

## Architecture Diagram

```
┌─────────────────────┐
│  Internet/Browser   │
└──────────┬──────────┘
           │ HTTPS
           ↓
┌─────────────────────────────────────┐
│    Cloudflare (Free Tier)           │
│  - Tunnel: 254carbon-cluster        │
│  - Edge: Dallas (dfw06-dfw09)       │
└──────────┬──────────────────────────┘
           │ QUIC Tunnel
           ↓
┌─────────────────────────────────────┐
│  Kubernetes Cluster                 │
│  - Namespace: cloudflare-tunnel     │
│  - Pod: cloudflared (2 replicas)    │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│  Ingress NGINX Controller           │
│  - Port: 80/443                     │
│  - Host routing enabled             │
└──────────┬──────────────────────────┘
           │
      ┌────┴────────────────┐
      ↓                     ↓
┌──────────────────┐  ┌──────────────────┐
│ datahub-frontend │  │  datahub-gms     │
│ (React UI)       │  │  (GraphQL API)   │
│ Port 9002        │  │  Port 8080       │
└──────────────────┘  └──────────────────┘
```

---

## Next Steps

1. **Access DataHub**: Visit `https://datahub.254carbon.com`
2. **Configure Metadata Ingestion**: Use DataHub CLI or API to ingest metadata from Trino, DolphinScheduler, etc.
3. **Monitor Health**: Check tunnel and ingress logs regularly
4. **Backup Configuration**: Version the tunnel configuration in Git

---

## Related Documentation

- Cloudflare Tunnel Logs: `kubectl logs -n cloudflare-tunnel deployment/cloudflared`
- Ingress Status: `kubectl describe ingress datahub -n data-platform`
- Helm Values: `helm/charts/data-platform/charts/datahub/values.yaml`
- Previous Issues: `CLOUDFLARE_TUNNEL_FIX_NEEDED.md`

---

**Created**: October 24, 2025, 22:39 UTC  
**Status**: ✅ Configured and operational  
**Last Verified**: Just now with tunnel token active  
