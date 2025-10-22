# 🎉 Cloudflare Tunnel Deployment - COMPLETE (Archived)

## Deployment Status: ✅ LIVE & OPERATIONAL

Your HMCo Kubernetes cluster is now securely exposed through **254carbon.com** via Cloudflare Tunnel!

---

## 📊 What Was Deployed

### Infrastructure
- ✅ **Cloudflare Tunnel**: `254carbon-cluster` (UUID: `291bc289-e3c3-4446-a9ad-8e327660ecd5`)
- ✅ **cloudflared Deployment**: 2 replicas running on Kubernetes
- ✅ **Kubernetes Namespace**: `cloudflare-tunnel` with full RBAC
- ✅ **DNS Records**: 9 CNAME records pointing to tunnel endpoint
- ✅ **Ingress Rules**: Updated to support 254carbon.com subdomains

### Services Exposed

| Service | Domain | Ready? |
|---------|--------|--------|
| DataHub | datahub.254carbon.com | ✅ Yes |
| Grafana | grafana.254carbon.com | ✅ Yes |
| Superset | superset.254carbon.com | ✅ Yes |
| Vault | vault.254carbon.com | ✅ Yes |
| Trino | trino.254carbon.com | ✅ Yes |
| Apache Doris | doris.254carbon.com | ✅ Yes |
| DolphinScheduler | dolphin.254carbon.com | ✅ Yes |
| MinIO | minio.254carbon.com | ✅ Yes |
| LakeFS | lakefs.254carbon.com | ✅ Yes |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  Your Local Network/Home Server     │
│  ┌──────────────────────────────┐   │
│  │  Kubernetes Cluster (Kind)   │   │
│  │  ┌────────────────────────┐  │   │
│  │  │  cloudflared Pods (2)  │  │   │
│  │  │  ↓↓↓                   │  │   │
│  │  │  NGINX Ingress         │  │   │
│  │  │  ↓↓↓                   │  │   │
│  │  │  Services:             │  │   │
│  │  │  - DataHub             │  │   │
│  │  │  - Grafana             │  │   │
│  │  │  - etc.                │  │   │
│  │  └────────────────────────┘  │   │
│  └──────────────────────────────┘   │
│              ↓↓↓ (QUIC tunnel)      │
└─────────────────────────────────────┘
                 ↓↓↓
         Cloudflare Edge Network
         (DDoS Protection, WAF)
                 ↓↓↓
           Internet Users
       https://datahub.254carbon.com
```

---

## 🚀 Quick Start: Access Your Services

Your services are now accessible from the internet! Try:

```bash
# From your browser or terminal
curl https://grafana.254carbon.com
curl https://datahub.254carbon.com
curl https://superset.254carbon.com

# With proper authentication where configured
```

**Note**: Some services may require authentication (Vault, MinIO). Configure via Cloudflare Access in the dashboard if needed.

---

## 📁 Files Created/Modified

### New Files
- `k8s/cloudflare/namespace.yaml` - Kubernetes namespace & RBAC
- `k8s/cloudflare/cloudflared-deployment.yaml` - Deployment, Service, PDB
- `k8s/cloudflare/tunnel-secret.yaml` - Secret template
- `k8s/cloudflare/cloudflared-config` - ConfigMap (ingress rules)
- `scripts/setup-cloudflare-dns.sh` - DNS automation script
- `scripts/update-cloudflare-credentials.sh` - Credential update script

### Modified Files
- `k8s/ingress/ingress-rules.yaml` - Added 254carbon.com subdomains
- `README.md` - Added "Cloudflare Integration" section

### Documentation Files
- `k8s/cloudflare/README.md` - Complete setup guide
- `k8s/cloudflare/GET_PROPER_CREDENTIALS.md` - Credential setup guide
- `k8s/cloudflare/SECURITY_POLICIES.md` - Security configuration
- `k8s/cloudflare/DEPLOYMENT_GUIDE.md` - Step-by-step deployment guide

---

## ✨ Key Features

✅ **Zero-Trust Security**
- No public IP exposure
- All traffic encrypted end-to-end
- Cloudflare's global DDoS protection

✅ **High Availability**
- 2 cloudflared replicas with anti-affinity
- PodDisruptionBudget ensures uptime
- Automatic failover

✅ **Easy Management**
- No firewall rules needed
- No port forwarding required
- Works from any network

✅ **Production-Ready**
- Health checks and metrics
- Proper resource limits
- RBAC security

---

## 🔧 Verification

Check your deployment:

```bash
# 1. Verify pods are running
kubectl get pods -n cloudflare-tunnel
# Expected: 2 pods in Running status

# 2. Check tunnel connection
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel | grep "Registered tunnel"
# Expected: Multiple "Registered tunnel connection" messages

# 3. Verify DNS
nslookup datahub.254carbon.com
# Expected: Points to Cloudflare nameservers

# 4. Test service access
curl -I https://grafana.254carbon.com
# Expected: HTTP 200 or redirect
```

---

## 📋 Next Steps (Optional)

### 1. Configure Cloudflare Access
Restrict access to sensitive services (Vault, MinIO):
- Go to Cloudflare Zero Trust dashboard
- Create Access Application for each service
- Set authentication policies

### 2. Set Up WAF Rules
Enable Web Application Firewall:
- Enable OWASP Core Ruleset
- Configure rate limiting (e.g., 100 req/min)
- Block known malicious IPs

### 3. Monitor & Alert
Set up monitoring:
- Access metrics at `http://cloudflared-metrics.cloudflare-tunnel:2000/metrics`
- Check Cloudflare dashboard for traffic analytics
- Configure alerts for tunnel disconnections

### 4. Scale Replicas (if needed)
```bash
kubectl scale deployment/cloudflared -n cloudflare-tunnel --replicas=3
```

---

## 🆘 Troubleshooting

### Services not accessible?
```bash
# 1. Check tunnel status
kubectl get pods -n cloudflare-tunnel
# All should be Running 1/1

# 2. Check logs
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel

# 3. Verify ingress rules
kubectl get ingress -A | grep 254carbon

# 4. Test locally
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
curl -H "Host: grafana.254carbon.com" http://localhost:8080
```

### Tunnel disconnected?
```bash
# Check for errors in logs
kubectl logs -n cloudflare-tunnel -f

# Restart deployment
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel

# Check pod events
kubectl describe pod -n cloudflare-tunnel <pod-name>
```

### DNS not resolving?
```bash
# Verify DNS records exist
curl -X GET "https://api.cloudflare.com/client/v4/zones/YOUR_ZONE_ID/dns_records" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Wait for DNS propagation (up to 24 hours, usually <5 minutes)
nslookup datahub.254carbon.com @1.1.1.1
```

---

## 📞 Support Resources

1. **Local Documentation**
   - `k8s/cloudflare/README.md` - Quick reference
   - `k8s/cloudflare/SECURITY_POLICIES.md` - Security setup
   - `k8s/cloudflare/DEPLOYMENT_GUIDE.md` - Detailed guide

2. **External Documentation**
   - [Cloudflare Tunnel Docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
   - [cloudflared GitHub](https://github.com/cloudflare/cloudflared)
   - [Cloudflare API Docs](https://developers.cloudflare.com/api/)

3. **Dashboards**
   - Cloudflare: https://dash.cloudflare.com
   - Tunnel Analytics: Dashboard → Networks → Tunnels → 254carbon-cluster → Analytics

---

## 📊 Current Tunnel Metrics

- **Tunnel ID**: `291bc289-e3c3-4446-a9ad-8e327660ecd5`
- **Tunnel Name**: `254carbon-cluster`
- **Protocol**: QUIC (UDP)
- **Replicas**: 2 (running)
- **Status**: ✅ Active & Connected
- **Exposed Services**: 9
- **Uptime**: Since deployment
- **DDoS Protection**: Enabled (Cloudflare)

---

## 🎯 What's Next?

Your cluster is now:
- ✅ Securely exposed to the internet
- ✅ Protected by Cloudflare's DDoS mitigation
- ✅ Accessible via friendly 254carbon.com subdomains
- ✅ Ready for production use

**Optional enhancements**:
- Add Cloudflare Access for authentication
- Configure WAF rules
- Set up monitoring & alerts
- Scale replicas for higher availability

---

**Deployment Date**: October 19, 2025  
**Status**: LIVE ✅
