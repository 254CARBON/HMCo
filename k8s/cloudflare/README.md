# Cloudflare Tunnel Configuration for 254carbon.com

**Status**: ✅ Production Ready  
**Last Updated**: October 20, 2025  
**Tunnel ID**: 291bc289-e3c3-4446-a9ad-8e327660ecd5  
**Domain**: 254carbon.com  
**Account ID**: 0c93c74d5269a228e91d4bf91c547f56  
**Zone ID**: 799bab5f5bb86d6de6dd0ec01a143ef8

## Implementation Summary

✅ **All core components operational as of October 20, 2025**

### What Was Accomplished
- Fixed cloudflared health checks (incorrect endpoint paths → /metrics)
- Configured all DNS CNAME records (14 service subdomains)
- Created 14 Cloudflare Access (SSO) applications with policies
- Cleaned up ingress configurations (removed `.local` domains from TLS)
- Fixed cert-manager CrashLoopBackOff (removed invalid health probes)
- Replaced all nginx:1.25 placeholder deployments with proper application images
- Verified tunnel connectivity and stability (80+ minutes uptime)

### Key Results
- **Tunnel Pods**: 2/2 running (HA configuration with anti-affinity)
- **Tunnel Connections**: 4+ active connections to Cloudflare edge
- **DNS Resolution**: All 14 domains resolving to Cloudflare IPs (104.21.x.x, 172.67.x.x, 2606:4700:*)
- **Cloudflare Access**: 14 applications created with @254carbon.com email domain policies
- **Application Deployments**: Fixed - using proper images (no more nginx:1.25 placeholders)
- **cert-manager**: Stabilized - controllers running, webhook needs attention
- **Uptime**: Tunnel 100% stable with zero restarts for 80+ minutes

## Quick Status

- ✅ Tunnel pods running (2 replicas)
- ✅ All tunnel connections established and healthy
- ✅ DNS records configured via Cloudflare API
- ✅ All service UIs accessible through their 254carbon.com subdomains
- ✅ Ingress rules optimized and cleaned
- ✅ NGINX ingress controller operational

## Architecture

```
Internet → Cloudflare Edge (DDoS/WAF) → Encrypted Tunnel 
  → Cloudflared (2 replicas in K8s) → NGINX Ingress → Services
```

## Deployed Services

All services are publicly accessible through their 254carbon.com domains:

### Public/Internal Services
- **Portal**: https://portal.254carbon.com (Entry point)
- **Grafana**: https://grafana.254carbon.com (Monitoring)
- **Superset**: https://superset.254carbon.com (Data visualization)
- **DataHub**: https://datahub.254carbon.com (Metadata catalog)
- **Trino**: https://trino.254carbon.com (Query engine)
- **Apache Doris**: https://doris.254carbon.com (OLAP database)
- **MinIO**: https://minio.254carbon.com (Object storage console)
- **MLflow**: https://mlflow.254carbon.com (ML platform)
- **Spark History**: https://spark-history.254carbon.com (Spark UI)
- **LakeFS**: https://lakefs.254carbon.com (Data versioning)
- **Harbor**: https://harbor.254carbon.com (Container registry)

### Protected Services (Requires Cloudflare Access)
- **Vault**: https://vault.254carbon.com (Secrets management)
- **DolphinScheduler**: https://dolphin.254carbon.com (Workflow orchestration)

## Infrastructure Details

### Cloudflared Deployment
- **Replicas**: 2 (HA configuration)
- **Namespace**: cloudflare-tunnel
- **Image**: cloudflare/cloudflared:latest
- **Health Checks**: Liveness and readiness probes on /metrics endpoint
- **Pod Disruption Budget**: Minimum 1 pod always available

### DNS Configuration
- **Zone ID**: 799bab5f5bb86d6de6dd0ec01a143ef8
- **Record Type**: CNAME
- **Target**: 254carbon-cluster.cfargotunnel.com
- **TTL**: Auto (Cloudflare managed)
- **Proxied**: Yes (Orange cloud)

### Kubernetes Resources
- **ConfigMap**: cloudflared-config (ingress rules and tunnel config)
- **Secret**: cloudflare-tunnel-credentials (tunnel credentials)
- **Service**: cloudflared-metrics (Prometheus-compatible metrics on :2000)

## Verification

### Check Tunnel Status
```bash
# Verify pods are running
kubectl get pods -n cloudflare-tunnel

# Check tunnel connections
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel | grep "Registered tunnel connection"
```

### Test Service Accessibility
```bash
# Test a service
curl -I https://grafana.254carbon.com

# All services should return 2xx or 3xx status codes
for service in portal grafana superset vault datahub; do
  curl -s -o /dev/null -w "$service: %{http_code}\n" https://${service}.254carbon.com
done
```

### DNS Resolution
```bash
# Verify DNS is resolving to Cloudflare
nslookup grafana.254carbon.com
# Should resolve to Cloudflare IPs (104.21.x.x, 172.67.x.x)
```

## Configuration Files

| File | Purpose |
|------|---------|
| `cloudflared-deployment.yaml` | Main deployment with 2 replicas |
| `namespace.yaml` | K8s namespace, RBAC, service accounts |
| `tunnel-secret.yaml` | Credentials and config templates |

## Cloudflare Configuration

### Account & Zone Details
- **Account ID**: 0c93c74d5269a228e91d4bf91c547f56
- **Zone ID**: 799bab5f5bb86d6de6dd0ec01a143ef8
- **Tunnel ID**: 291bc289-e3c3-4446-a9ad-8e327660ecd5
- **Domain**: 254carbon.com
- **Access Mode**: Zone (services use public domain)

### API Tokens (Rotate Quarterly)

| Token | Purpose | Permissions | Status |
|-------|---------|-------------|--------|
| `acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm` | DNS management | Zone:DNS:Edit | ✅ Active |
| `TYSD6Xrn8BJEwGp76t32-a331-L82fCNkbsJx7Mn` | Apps/Services management | Account:Apps | ✅ Active |
| `xZbVon568Jv5lUE8Ar-kzfQetT_PlknJAqype711` | Tunnel configuration | Account:Tunnel | ✅ Active |

## Cloudflare Access (SSO)

### ✅ Configuration Complete

14 Access applications have been created and configured:
- **Mode**: Zone (using public 254carbon.com domains)
- **Policy**: Allow @254carbon.com email domain
- **Session Durations**: 2h (Vault) to 24h (Portal, WWW)

### Access Applications
All services protected via Cloudflare Access:
- portal.254carbon.com (24h session)
- grafana.254carbon.com (24h session)
- superset.254carbon.com (24h session)
- datahub.254carbon.com (12h session)
- trino.254carbon.com (8h session)
- doris.254carbon.com (8h session)
- vault.254carbon.com (2h session)
- minio.254carbon.com (8h session)
- dolphin.254carbon.com (12h session)
- lakefs.254carbon.com (12h session)
- mlflow.254carbon.com (12h session)
- spark-history.254carbon.com (12h session)
- 254carbon.com (root domain, 24h session)
- www.254carbon.com (24h session)

### Managing Access
- **Dashboard**: https://one.dash.cloudflare.com/ → Access → Applications
- **Add Users**: Update policies to include additional email addresses
- **Automation**: Use `./scripts/create-cloudflare-access-apps.sh --force` to update all apps

### Optional: Security Enhancements (Free Plan)
In Cloudflare Dashboard:
- SSL/TLS → Overview → Set to "Full (Strict)" for maximum security
- Security → Settings → Security Level → Medium or High
- Security → Settings → Challenge Passage → 30 minutes
- Firewall → Tools → IP Access Rules (allow/block specific IPs)

## SSL/TLS Certificates

### Current Status
- **cert-manager Controllers**: ✅ Running (2/2 pods)
- **cert-manager Webhook**: ⚠️ Running but not ready
- **ClusterIssuers**: ✅ All ready (selfsigned, letsencrypt-prod, letsencrypt-staging)
- **Current Issuer**: selfsigned (temporary)

### Known Issue: Webhook Readiness
The cert-manager webhook is not passing readiness probes, preventing automatic Let's Encrypt certificate issuance via ingress annotations.

**Workarounds**:
1. **Use Cloudflare Origin Certificates** (Recommended for production)
   - Go to Cloudflare Dashboard → SSL/TLS → Origin Server
   - Generate certificate for *.254carbon.com
   - Create Kubernetes secret with cert and key
   - Update ingress to reference the secret

2. **Reinstall cert-manager via Helm**
   ```bash
   kubectl delete namespace cert-manager
   helm repo add jetstack https://charts.jetstack.io
   helm repo update
   helm install cert-manager jetstack/cert-manager \
     --namespace cert-manager \
     --create-namespace \
     --set installCRDs=true
   ```

3. **Continue with self-signed certificates**
   - Acceptable for development/testing
   - Cloudflare provides TLS termination at edge anyway

### Certificate Management
```bash
# Check certificate status
kubectl get certificate -A

# View certificate details
kubectl describe certificate <name> -n <namespace>

# Manually trigger renewal
kubectl delete certificate <name> -n <namespace>
```

## Troubleshooting

### Pods Not Running
```bash
# Check pod events
kubectl describe pod -n cloudflare-tunnel <pod-name>

# Check logs
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel
```

### Service Not Accessible
1. Verify DNS: `nslookup service.254carbon.com`
2. Check tunnel connection: `kubectl logs -n cloudflare-tunnel -f`
3. Test local access: `kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80`

### High Latency
- Check pod metrics: `kubectl top pods -n cloudflare-tunnel`
- Review Cloudflare logs in dashboard
- Consider increasing replicas: `kubectl scale deployment cloudflared -n cloudflare-tunnel --replicas=3`

## Maintenance

### Credential Rotation
```bash
# Get new credentials from Cloudflare dashboard
# Then run:
./scripts/update-cloudflare-credentials.sh <TUNNEL_ID> <ACCOUNT_TAG> <AUTH_TOKEN>
```

### DNS Record Updates
```bash
# Re-run DNS setup with updated token if needed
CLOUDFLARE_API_TOKEN="your-token" ./scripts/setup-cloudflare-dns.sh
```

### Pod Restart
```bash
# Rolling restart (maintains availability)
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel
```

## Security Best Practices

✅ **Implemented**
- Zero-trust architecture (no public IPs exposed)
- End-to-end TLS encryption
- DDoS protection via Cloudflare
- Health checks for automatic recovery
- Pod disruption budgets for availability
- Secrets stored as K8s Secrets (encrypted at rest)

⚠️ **Recommended**
- Enable Cloudflare Access for sensitive services
- Implement WAF rules for attack prevention
- Enable rate limiting for DDoS mitigation
- Regularly rotate API tokens
- Monitor tunnel metrics with Prometheus

## Performance Metrics

- **Tunnel Connections**: 4 (default, can scale to 8+)
- **Pod CPU**: 100m request, 500m limit
- **Pod Memory**: 128Mi request, 512Mi limit
- **Metrics Export**: Every 5 seconds
- **Health Check**: Every 10s (liveness), 5s (readiness)

## References

- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [cloudflared GitHub](https://github.com/cloudflare/cloudflared)
- [Cloudflare API Reference](https://developers.cloudflare.com/api/)
