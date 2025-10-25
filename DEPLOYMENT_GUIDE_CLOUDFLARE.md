# 254Carbon Cloudflare DNS/Tunnel/Ingress - Deployment Guide

## Quick Start (< 5 minutes)

### One-Command Deployment
```bash
chmod +x /home/m/tff/254CARBON/HMCo/scripts/deploy-cloudflare-ingress.sh
/home/m/tff/254CARBON/HMCo/scripts/deploy-cloudflare-ingress.sh
```

---

## Step-by-Step Deployment (Manual)

### Prerequisites Checklist
- [ ] kubectl configured and authenticated to cluster
- [ ] Cloudflare account with tunnel credentials
- [ ] cert-manager installed in cluster (`kubectl get ns cert-manager`)
- [ ] nginx-ingress-controller running (`kubectl get pods -n ingress-nginx`)

### Step 1: Verify Tunnel Pods Are Running
```bash
kubectl get pods -n cloudflare-tunnel
```

**Expected Output**:
```
NAME                           READY   STATUS    RESTARTS   AGE
cloudflared-7f7fbf867b-m26f7   1/1     Running   0          15h
cloudflared-7f7fbf867b-njmzk   1/1     Running   0          15h
```

If pods are not running, deploy the tunnel first:
```bash
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/cloudflare-tunnel-ingress.yaml
```

### Step 2: Update Tunnel Configuration
```bash
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/cloudflare-tunnel-ingress.yaml
```

**What this does**:
- Updates ConfigMap with all service hostnames
- Adds routes for: Prometheus, AlertManager, Victoria Metrics, Loki, ClickHouse, Katib, Kong
- Maintains existing service routes

### Step 3: Deploy Ingress Resources
Deploy each ingress resource:

```bash
# Monitoring services
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/prometheus-ingress.yaml
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/alertmanager-ingress.yaml

# Time-series and logging
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/victoria-metrics-ingress.yaml
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/loki-ingress.yaml

# Data platform
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/clickhouse-ingress.yaml

# ML & Infrastructure
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/katib-ingress.yaml
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/kong-admin-ingress.yaml
```

Or deploy all at once:
```bash
kubectl apply -f /home/m/tff/254CARBON/HMCo/k8s/ingress/
```

**Verify ingress resources**:
```bash
kubectl get ingress -A
```

**Expected**: Should see all new ingress resources in their respective namespaces

### Step 4: Restart Tunnel Pods
```bash
kubectl rollout restart deployment cloudflared -n cloudflare-tunnel
```

Wait for pods to restart:
```bash
kubectl rollout status deployment cloudflared -n cloudflare-tunnel
```

**Expected Output**:
```
deployment "cloudflared" successfully rolled out
```

### Step 5: Verify Deployment

**Check ingress resources**:
```bash
kubectl get ingress -A -o wide
```

**Check tunnel pods**:
```bash
kubectl get pods -n cloudflare-tunnel -o wide
```

**Check tunnel logs**:
```bash
kubectl logs -n cloudflare-tunnel deployment/cloudflared --tail=50
```

**Expected log indicators**:
- "Connection established"
- "Tunnel is running"
- No error messages about hostnames

### Step 6: Test Service Access

**DNS Resolution**:
```bash
nslookup prometheus.254carbon.com
```

Should resolve to Cloudflare's edge IP.

**HTTPS Connectivity**:
```bash
# Test HTTPS (should get 200 or redirect)
curl -I https://prometheus.254carbon.com
curl -I https://alertmanager.254carbon.com
curl -I https://clickhouse.254carbon.com

# Follow redirects
curl -L https://prometheus.254carbon.com 2>&1 | head -20
```

**Browser Access**:
Open these URLs in a browser:
- https://prometheus.254carbon.com
- https://alertmanager.254carbon.com
- https://victoria.254carbon.com
- https://loki.254carbon.com
- https://clickhouse.254carbon.com
- https://katib.254carbon.com
- https://kong.254carbon.com

All should load successfully (may prompt for authentication).

---

## Troubleshooting

### Issue: "Service Unavailable" / 502 Errors

**Diagnosis**:
```bash
# Check ingress resource
kubectl describe ingress prometheus-ingress -n monitoring

# Check if service exists
kubectl get svc prometheus -n monitoring

# Check service endpoints
kubectl get endpoints prometheus -n monitoring
```

**Solution**:
1. Verify service is running: `kubectl get pods -n monitoring -l app=prometheus`
2. Verify ingress backend service name matches actual service
3. Check service port configuration

### Issue: Certificate Not Valid

**Diagnosis**:
```bash
# Check certificate status
kubectl get certificates -A
kubectl describe certificate prometheus-tls -n monitoring
```

**Solution**:
```bash
# Check cert-manager status
kubectl get pods -n cert-manager
kubectl logs -n cert-manager deployment/cert-manager

# Force certificate renewal if needed
kubectl delete secret prometheus-tls -n monitoring
kubectl delete certificate prometheus-tls -n monitoring
kubectl apply -f k8s/ingress/prometheus-ingress.yaml
```

### Issue: Tunnel Connection Failed

**Diagnosis**:
```bash
# Check tunnel pod logs
kubectl logs -n cloudflare-tunnel deployment/cloudflared

# Check if tunnel pods are ready
kubectl describe pod -n cloudflare-tunnel -l app=cloudflared
```

**Solution**:
```bash
# Verify tunnel credentials
kubectl get secret -n cloudflare-tunnel -o yaml cloudflare-tunnel-token

# Restart tunnel
kubectl rollout restart deployment cloudflared -n cloudflare-tunnel
kubectl rollout status deployment cloudflared -n cloudflare-tunnel
```

### Issue: DNS Not Resolving

**Diagnosis**:
```bash
# Check Cloudflare DNS
dig prometheus.254carbon.com
nslookup prometheus.254carbon.com
```

**Solution**:
1. Log into Cloudflare dashboard
2. Verify DNS records are created (should be CNAME or A records)
3. Allow 5-10 minutes for DNS propagation
4. Clear DNS cache if needed: `sudo systemctl restart systemd-resolved`

### Issue: Ingress Shows No Address

**Diagnosis**:
```bash
kubectl get ingress prometheus-ingress -n monitoring -o wide
```

Should show HOSTS and ADDRESS.

**Solution**:
1. Wait a few minutes for ingress controller to process
2. Check ingress controller logs: `kubectl logs -n ingress-nginx deployment/nginx-ingress-controller --tail=50`
3. Verify ingress class name matches: `kubectl get ingressclass`

---

## Post-Deployment Configuration

### Cloudflare Dashboard Setup

1. **Log in to Cloudflare** → Account → Tunnels
2. **Select the tunnel**: `254carbon` (ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5)
3. **Verify DNS records are created**:
   - *.254carbon.com → CNAME → tunnel-id.cfargotunnel.com
   - 254carbon.com → A record → Cloudflare IP
   - www.254carbon.com → CNAME → 254carbon.com

4. **Optional: Configure Cloudflare Access** (for authentication):
   - Add Cloudflare Access policy
   - Restrict access to Vault, Prometheus, Kong to authorized users
   - Set up SSO integration

5. **Optional: Enable WAF** (Web Application Firewall):
   - Enable managed rules
   - Add custom rules as needed

6. **Optional: Configure Rate Limiting**:
   - Add rate limiting rules to prevent abuse
   - Set thresholds for API endpoints

---

## Verification Checklist

After deployment, verify:

- [ ] All tunnel pods are running (2 replicas)
  ```bash
  kubectl get pods -n cloudflare-tunnel
  ```

- [ ] All ingress resources exist
  ```bash
  kubectl get ingress -A | grep -E "prometheus|alertmanager|victoria|loki|clickhouse|katib|kong"
  ```

- [ ] All TLS certificates are issued
  ```bash
  kubectl get certificates -A | grep -E "prometheus|alertmanager|victoria|loki|clickhouse|katib|kong"
  ```

- [ ] All services are accessible via HTTPS
  ```bash
  for svc in prometheus alertmanager victoria loki clickhouse katib kong; do
    echo "Testing $svc.254carbon.com:"
    curl -I https://$svc.254carbon.com 2>&1 | head -1
  done
  ```

- [ ] Tunnel logs show no errors
  ```bash
  kubectl logs -n cloudflare-tunnel deployment/cloudflared --tail=20 | grep -i error
  ```

---

## Services Exposed

### After Successful Deployment

| Service | URL | Port | Namespace |
|---------|-----|------|-----------|
| Prometheus | https://prometheus.254carbon.com | 9090 | monitoring |
| AlertManager | https://alertmanager.254carbon.com | 9093 | monitoring |
| Victoria Metrics | https://victoria.254carbon.com | 8428 | victoria-metrics |
| Loki | https://loki.254carbon.com | 3100 | victoria-metrics |
| ClickHouse | https://clickhouse.254carbon.com | 8123 | data-platform |
| Katib UI | https://katib.254carbon.com | 80 | kubeflow |
| Kong Admin | https://kong.254carbon.com | 8001 | kong |

Plus all existing services:
- Portal, DataHub, Grafana, Superset, Trino, Vault, MinIO, DolphinScheduler, LakeFS, Harbor, JupyterHub

---

## File Reference

### Modified Files
- `k8s/cloudflare-tunnel-ingress.yaml` - Updated with new hostnames

### New Files Created
- `k8s/ingress/prometheus-ingress.yaml`
- `k8s/ingress/alertmanager-ingress.yaml`
- `k8s/ingress/victoria-metrics-ingress.yaml`
- `k8s/ingress/loki-ingress.yaml`
- `k8s/ingress/clickhouse-ingress.yaml`
- `k8s/ingress/katib-ingress.yaml`
- `k8s/ingress/kong-admin-ingress.yaml`
- `scripts/deploy-cloudflare-ingress.sh` - Deployment automation
- `docs/cloudflare/INGRESS_SETUP.md` - Detailed documentation

### Updated Files
- `services.json` - Added 6 new services to catalog

---

## Support & Documentation

- **Quick Reference**: This file
- **Detailed Setup**: `docs/cloudflare/INGRESS_SETUP.md`
- **Deployment Script**: `scripts/deploy-cloudflare-ingress.sh`
- **Implementation Summary**: `CLOUDFLARE_IMPLEMENTATION_SUMMARY.md`

---

## Rollback Procedure

If something goes wrong, follow these steps to rollback:

```bash
# 1. Delete new ingress resources
kubectl delete -f k8s/ingress/prometheus-ingress.yaml
kubectl delete -f k8s/ingress/alertmanager-ingress.yaml
kubectl delete -f k8s/ingress/victoria-metrics-ingress.yaml
kubectl delete -f k8s/ingress/loki-ingress.yaml
kubectl delete -f k8s/ingress/clickhouse-ingress.yaml
kubectl delete -f k8s/ingress/katib-ingress.yaml
kubectl delete -f k8s/ingress/kong-admin-ingress.yaml

# 2. Restore original tunnel configuration
git checkout k8s/cloudflare-tunnel-ingress.yaml
kubectl apply -f k8s/cloudflare-tunnel-ingress.yaml

# 3. Restart tunnel pods
kubectl rollout restart deployment cloudflared -n cloudflare-tunnel

# 4. Verify rollback
kubectl get ingress -A
kubectl logs -n cloudflare-tunnel deployment/cloudflared --tail=20
```

---

## Performance & Monitoring

### Monitor Tunnel Health
```bash
# Real-time logs
kubectl logs -f -n cloudflare-tunnel deployment/cloudflared

# Resource usage
kubectl top pods -n cloudflare-tunnel

# Connection statistics
kubectl exec -it -n cloudflare-tunnel deployment/cloudflared -- /app/cloudflared tunnel info
```

### Monitor Certificate Status
```bash
# Check all certificates
kubectl get certificates -A

# Watch certificate renewal
watch kubectl get certificates -A
```

---

## Next Steps

After successful deployment:

1. **Configure authentication** for sensitive services (Vault, Kong, Prometheus)
2. **Set up monitoring** for tunnel health and certificate expiration
3. **Configure Cloudflare WAF** for additional security
4. **Test with production traffic** and monitor for issues
5. **Document in runbooks** for operational procedures

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section above
2. Review detailed documentation: `docs/cloudflare/INGRESS_SETUP.md`
3. Check Kubernetes events: `kubectl describe ingress <name> -n <namespace>`
4. Review tunnel logs: `kubectl logs -f -n cloudflare-tunnel deployment/cloudflared`
