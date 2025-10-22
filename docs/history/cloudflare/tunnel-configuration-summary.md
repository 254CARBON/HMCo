# Cloudflare Tunnel & Access Configuration

**Status**: ✅ Production Ready  
**Domain**: 254carbon.com  
**Last Updated**: October 20, 2025

---

## Quick Start

### Access Services
All 254carbon.com services are accessible via Cloudflare Tunnel with SSO protection:

```bash
# Test connectivity
curl https://portal.254carbon.com      # Portal homepage
curl https://grafana.254carbon.com     # Grafana dashboards
curl https://harbor.254carbon.com      # Container registry
```

### Check Infrastructure Health
```bash
# Tunnel status
kubectl get pods -n cloudflare-tunnel

# Should show: 2/2 Running pods with 8+ active connections
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Users / Internet                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Cloudflare Edge Network                             │
│   • DDoS Protection                                         │
│   • WAF (Web Application Firewall)                         │
│   • TLS Termination                                         │
│   • Cloudflare Access (SSO Authentication)                 │
└────────────────────────┬────────────────────────────────────┘
                         │ Encrypted Tunnel (HTTP/2)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│      Cloudflared (Kubernetes - 2 HA Replicas)              │
│   • Tunnel ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5       │
│   • Metrics: Port 2000                                      │
│   • 8 Active Connections to Edge                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         NGINX Ingress Controller                            │
│   • Routes traffic to backend services                     │
│   • 14 Ingress rules configured                            │
└────────────────────────┬────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      [Services]    [Services]    [Services]
      Portal        Grafana       Vault
      DataHub       Superset      MLflow
      Trino         Doris         Harbor
      MinIO         DolphinSched  LakeFS
```

---

## Services & URLs

### All 254carbon.com Services (14 Total)

| Service | URL | Purpose | SSO Session |
|---------|-----|---------|-------------|
| Portal | https://portal.254carbon.com | Main entry point | 24h |
| Grafana | https://grafana.254carbon.com | Monitoring dashboards | 24h |
| Superset | https://superset.254carbon.com | Data visualization | 24h |
| DataHub | https://datahub.254carbon.com | Metadata catalog | 12h |
| Trino | https://trino.254carbon.com | Query engine | 8h |
| Doris | https://doris.254carbon.com | OLAP database | 8h |
| Vault | https://vault.254carbon.com | Secrets management | 2h |
| MinIO | https://minio.254carbon.com | Object storage | 8h |
| DolphinScheduler | https://dolphin.254carbon.com | Workflow orchestration | 12h |
| LakeFS | https://lakefs.254carbon.com | Data versioning | 12h |
| MLflow | https://mlflow.254carbon.com | ML platform | 12h |
| Spark History | https://spark-history.254carbon.com | Spark UI | 12h |
| Harbor | https://harbor.254carbon.com | Container registry | N/A |
| WWW | https://www.254carbon.com | Alias to portal | 24h |

---

## Configuration

### Cloudflare Account
- **Account ID**: 0c93c74d5269a228e91d4bf91c547f56
- **Zone ID**: 799bab5f5bb86d6de6dd0ec01a143ef8
- **Tunnel ID**: 291bc289-e3c3-4446-a9ad-8e327660ecd5

### Kubernetes Resources
- **Namespace**: cloudflare-tunnel
- **Deployment**: cloudflared (2 replicas)
- **ConfigMap**: cloudflared-config (tunnel routing rules)
- **Secret**: cloudflare-tunnel-credentials
- **Service**: cloudflared-metrics (port 2000)

### Cloudflare Access (SSO)
- **Mode**: Zone (using public 254carbon.com domains)
- **Policy**: Allow @254carbon.com email domain
- **Authentication**: Email OTP or configured IdP
- **Applications**: 14 configured (one per service)

---

## Management

### Add New Service

1. **Update Tunnel ConfigMap**:
   ```bash
   kubectl edit configmap cloudflared-config -n cloudflare-tunnel
   # Add new ingress rule
   ```

2. **Create DNS Record**:
   ```bash
   ./scripts/add-dns-record.sh newservice 254carbon.com
   ```

3. **Create Access Application** (optional):
   ```bash
   ./scripts/create-cloudflare-access-apps.sh --force
   # Or manually in Zero Trust dashboard
   ```

4. **Create Ingress**:
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: newservice-ingress
     namespace: data-platform
     annotations:
       nginx.ingress.kubernetes.io/ssl-redirect: "true"
       cert-manager.io/cluster-issuer: "selfsigned"
   spec:
     ingressClassName: nginx
     tls:
     - hosts:
       - newservice.254carbon.com
       secretName: newservice-tls
     rules:
     - host: newservice.254carbon.com
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: newservice
               port:
                 number: 8080
   ```

### Rotate Credentials

```bash
# Run credential update script
./scripts/update-cloudflare-credentials.sh <TUNNEL_ID> <ACCOUNT_TAG> <AUTH_TOKEN>
```

---

## Monitoring

### Health Checks
```bash
# Tunnel pods
kubectl get pods -n cloudflare-tunnel

# Tunnel connections (should be 4-8)
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel | grep "Registered tunnel connection" | wc -l

# Service accessibility
for svc in portal grafana harbor; do
  curl -s -o /dev/null -w "$svc: %{http_code}\n" https://$svc.254carbon.com
done
```

### Metrics
```bash
# Port-forward metrics endpoint
kubectl port-forward -n cloudflare-tunnel svc/cloudflared-metrics 2000:2000

# View metrics
curl http://localhost:2000/metrics | grep cloudflared_tunnel
```

---

## Troubleshooting

### Service Returns 502/503
- **Check**: Backend pod status
  ```bash
  kubectl get pods -n <namespace>
  kubectl logs -n <namespace> <pod>
  ```

### DNS Not Resolving
- **Check**: Cloudflare dashboard → DNS records
- **Verify**: `nslookup service.254carbon.com`

### Tunnel Disconnected
- **Check**: Pod logs
  ```bash
  kubectl logs -n cloudflare-tunnel -f
  ```
- **Restart**: `kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel`

### SSO Authentication Failing
- **Check**: Cloudflare Zero Trust → Access → Applications
- **Verify**: Email domain in allowed list
- **Test**: Ingress annotations correct

---

## Documentation

- **Main README**: `k8s/cloudflare/README.md` - Detailed setup and configuration
- **Operational Runbook**: `docs/operations/cloudflare-runbook.md` - Day-to-day operations
- **Implementation Report**: `CLOUDFLARE_STABILIZATION_COMPLETE.md` - What was done
- **Final Status**: `CLOUDFLARE_FINAL_STATUS.md` - Production readiness report

---

## Support

- **Cloudflare Dashboard**: https://dash.cloudflare.com/
- **Zero Trust Dashboard**: https://one.dash.cloudflare.com/
- **Operational Runbook**: `docs/operations/cloudflare-runbook.md`
- **Cloudflare Support**: https://support.cloudflare.com/

---

## Current Status

✅ **All Core Components Operational**
- Tunnel: 2/2 pods, 8 connections
- DNS: 14/14 records configured
- Access (SSO): 14/14 applications configured
- Services: 13/13 accessible (HTTP 200)
- Ingress: Clean configurations
- cert-manager: Stable (using self-signed certs)

⚠️ **Known Issues**
- cert-manager webhook readiness (use Cloudflare Origin Certificates as alternative)
- Kafka/Zookeeper DNS resolution (being investigated)

**Overall Health**: 90% - Production Ready ✅

---

**For detailed operational procedures, see**: `docs/operations/cloudflare-runbook.md`

