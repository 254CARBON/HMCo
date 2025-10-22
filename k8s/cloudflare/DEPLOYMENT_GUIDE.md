# Cloudflare Tunnel Deployment & Configuration Guide

**Status**: ✅ **COMPLETE & OPERATIONAL**  
**Date Completed**: October 20, 2025  
**Implementation Time**: ~1.5 hours

## What Has Been Completed

### Phase 1: Cloudflared Deployment ✅
- [x] Namespace and RBAC configured
- [x] Cloudflared deployment with 2 replicas
- [x] Health checks fixed (using /metrics endpoint)
- [x] Pod disruption budget for HA
- [x] Metrics service for monitoring
- [x] Anti-affinity rules for pod distribution

### Phase 2: DNS Configuration ✅
- [x] All service CNAME records created
- [x] DNS pointing to 291bc289-e3c3-4446-a9ad-8e327660ecd5.cfargotunnel.com
- [x] Cloudflare proxying enabled (orange cloud)
- [x] Apex redirect configured (254carbon.com → portal.254carbon.com)

### Phase 3: Ingress Configuration ✅
- [x] All ingress resources configured for 254carbon.com domains
- [x] Duplicate SSO annotations removed
- [x] Portal service discovery annotations added
- [x] TLS certificates configured
- [x] NGINX ingress controller operational

### Phase 4: Tunnel Connectivity ✅
- [x] Tunnel pods running and healthy (2/2 ready)
- [x] All tunnel connections established (4+ connections to Cloudflare)
- [x] Service accessibility verified (all services responding)
- [x] DNS resolution verified (resolving to Cloudflare IPs)

## Current Infrastructure Status

```
┌─────────────────────────────────────────────────────────────────┐
│                    254carbon.com                                │
│                   (Cloudflare Domain)                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTPS (Encrypted via Cloudflare)
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              Cloudflare Edge Network                            │
│  - DDoS Protection                                              │
│  - WAF (Optional)                                               │
│  - Rate Limiting (Optional)                                     │
│  - Global Caching                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Encrypted Tunnel (QUIC)
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│        cloudflared (Kubernetes Deployment)                      │
│  Namespace: cloudflare-tunnel                                   │
│  Replicas: 2 (High Availability)                                │
│  Status: Running ✅                                             │
│  Connections: 4 active                                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│        NGINX Ingress Controller                                 │
│  Namespace: ingress-nginx                                       │
│  Status: Running ✅                                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬──────────────┐
        │             │             │              │
   ┌────▼───┐    ┌────▼───┐  ┌────▼────┐  ┌─────▼────┐
   │ Grafana │    │DataHub │  │ Superset│  │  Trino   │
   │  (3000) │    │ (9002) │  │ (8088)  │  │ (8080)   │
   └────────┘    └────────┘  └─────────┘  └──────────┘
        │             │             │              │
        └─────────────┼─────────────┴──────────────┘
                      │ etc...
            More services
```

## Deployment Checklist

### Prerequisites (Already Done)
- [x] Kubernetes cluster running (bare-metal, all nodes functional)
- [x] NGINX ingress controller deployed
- [x] Cloudflare account with 254carbon.com configured
- [x] Tunnel created in Cloudflare dashboard
- [x] API tokens generated

### Tunnel Deployment
- [x] Create namespace: `kubectl apply -f k8s/cloudflare/namespace.yaml`
- [x] Configure credentials: Updated in tunnel-secret.yaml
- [x] Deploy tunnel: `kubectl apply -f k8s/cloudflare/cloudflared-deployment.yaml`
- [x] Verify pods: `kubectl get pods -n cloudflare-tunnel` → 2 Running

### DNS Configuration
- [x] Run DNS script: `CLOUDFLARE_API_TOKEN="acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm" ./scripts/setup-cloudflare-dns.sh`
- [x] Verify all 13+ CNAME records created
- [x] Test DNS resolution: `nslookup grafana.254carbon.com` → Returns Cloudflare IPs

### Ingress Configuration
- [x] Apply ingress rules: `kubectl apply -f k8s/ingress/ingress-rules.yaml`
- [x] Verify all ingresses: `kubectl get ingress -A` → All showing configured
- [x] Clean duplicate annotations: ✅ Completed

### Verification
- [x] Tunnel connectivity: `kubectl logs -n cloudflare-tunnel | grep "Registered tunnel connection"`
- [x] Service accessibility: All services returning 2xx/3xx responses
- [x] DNS resolution: All domains resolving to Cloudflare IPs
- [x] NGINX operational: Ingress controller running and ready

## API Credentials Used

```
Account ID:                    0c93c74d5269a228e91d4bf91c547f56
Tunnel ID:                     291bc289-e3c3-4446-a9ad-8e327660ecd5
Zone ID (254carbon.com):       799bab5f5bb86d6de6dd0ec01a143ef8
DNS API Token:                 acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm
Apps API Token:                TYSD6Xrn8BJEwGp76t32-a331-L82fCNkbsJx7Mn
Tunnel Edit API Token:         xZbVon568Jv5lUE8Ar-kzfQetT_PlknJAqype711
```

## Deployed Services (13 Total)

| Service | Domain | Status | Type |
|---------|--------|--------|------|
| Portal | portal.254carbon.com | ✅ | Public |
| Grafana | grafana.254carbon.com | ✅ | Public |
| Superset | superset.254carbon.com | ✅ | Public |
| DataHub | datahub.254carbon.com | ✅ | Public |
| Trino | trino.254carbon.com | ✅ | Internal |
| Apache Doris | doris.254carbon.com | ✅ | Internal |
| MinIO Console | minio.254carbon.com | ✅ | Protected |
| Vault | vault.254carbon.com | ✅ | Protected |
| DolphinScheduler | dolphin.254carbon.com | ✅ | Protected |
| MLflow | mlflow.254carbon.com | ✅ | Public |
| Spark History | spark-history.254carbon.com | ✅ | Public |
| Harbor | harbor.254carbon.com | ✅ | Public |
| LakeFS | lakefs.254carbon.com | ✅ | Public |

## Monitoring & Health Checks

### Tunnel Health
```bash
# Check pod status (should be 2/2 Running)
kubectl get pods -n cloudflare-tunnel

# Check tunnel connections (should show 4+ registered connections)
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel | grep "Registered tunnel connection"

# Check metrics endpoint
kubectl port-forward -n cloudflare-tunnel svc/cloudflared-metrics 2000:2000
# Then access http://localhost:2000/metrics
```

### Service Accessibility
```bash
# Test all services
for service in portal grafana superset vault datahub; do
  curl -s -o /dev/null -w "$service: %{http_code}\n" https://${service}.254carbon.com
done
```

### DNS Health
```bash
# Verify DNS resolution
nslookup grafana.254carbon.com
# Should resolve to Cloudflare IPs (104.21.x.x, 172.67.x.x)
```

## Troubleshooting

### Pods Not Running
**Symptom**: CrashLoopBackOff status
**Solution**:
```bash
# Check logs
kubectl logs -n cloudflare-tunnel <pod-name>

# Verify credentials are correct
kubectl get secret cloudflare-tunnel-credentials -n cloudflare-tunnel -o yaml

# Verify config is valid
kubectl get configmap cloudflared-config -n cloudflare-tunnel -o yaml
```

### Service Not Accessible
**Symptom**: curl returns connection timeout or 502
**Solution**:
1. Verify DNS: `nslookup service.254carbon.com`
2. Check tunnel is connected: `kubectl logs -n cloudflare-tunnel -f | grep "Registered tunnel"`
3. Verify NGINX: `kubectl get pods -n ingress-nginx`
4. Test locally: `kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80`

### High Latency
**Symptom**: Services are slow to respond
**Solution**:
```bash
# Check pod resource usage
kubectl top pods -n cloudflare-tunnel

# Scale to 3 replicas for better distribution
kubectl scale deployment cloudflared -n cloudflare-tunnel --replicas=3

# Check Cloudflare dashboard for edge issues
```

## Maintenance Tasks

### Daily
- Monitor tunnel health: `kubectl get pods -n cloudflare-tunnel`
- Verify services are accessible

### Weekly
- Review tunnel logs for errors
- Check Cloudflare dashboard for blocked requests
- Monitor tunnel metrics: http://localhost:2000/metrics

### Monthly
- Review and rotate API tokens if needed
- Update cloudflared image to latest version
- Audit ingress rules and DNS records
- Test disaster recovery procedures

### Annually
- Review security policies and WAF rules
- Optimize tunnel configuration
- Plan capacity upgrades

## Next Steps

### Optional: Enable Advanced Security
1. **Cloudflare Access**: Add authentication to sensitive services
   - See k8s/cloudflare/SECURITY_POLICIES.md

2. **WAF Rules**: Enable OWASP protection
   - Dashboard → Security → WAF → Enable ruleset

3. **Rate Limiting**: Protect against abuse
   - Dashboard → Security → Rate Limiting

### Optional: Performance Optimization
1. **Enable Caching**: Cache static content at Cloudflare edge
2. **Increase Replicas**: Scale to 3-4 pods for higher throughput
3. **Monitor Metrics**: Set up Prometheus to scrape :2000/metrics

## Support Resources

- **Cloudflare Tunnel Docs**: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- **cloudflared GitHub**: https://github.com/cloudflare/cloudflared
- **Cloudflare API**: https://developers.cloudflare.com/api/
- **Kubernetes Ingress**: https://kubernetes.io/docs/concepts/services-networking/ingress/

## Success Criteria Met

✅ All pods running without restarts  
✅ All tunnel connections established (4+ active)  
✅ All DNS records resolving to Cloudflare  
✅ All services accessible through 254carbon.com domains  
✅ No duplicate or conflicting configurations  
✅ High availability with 2 replicas and pod disruption budget  
✅ Metrics available for monitoring  
✅ Proper RBAC and security context configured  
✅ Documentation complete and up-to-date  

**Status**: ✅ **PRODUCTION READY**
