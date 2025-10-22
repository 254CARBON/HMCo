# Cloudflare Operations Runbook

**Version**: 1.0  
**Last Updated**: October 20, 2025  
**Owner**: Platform Team  
**Domain**: 254carbon.com

---

## Overview

This runbook provides operational procedures for managing the Cloudflare Tunnel, DNS, and Access (SSO) infrastructure for 254carbon.com services.

## Quick Reference

### Critical Information
- **Account ID**: `0c93c74d5269a228e91d4bf91c547f56`
- **Zone ID**: `799bab5f5bb86d6de6dd0ec01a143ef8`
- **Tunnel ID**: `291bc289-e3c3-4446-a9ad-8e327660ecd5`
- **Tunnel Name**: 254carbon-cluster
- **Namespace**: cloudflare-tunnel

### Access Points
- **Cloudflare Dashboard**: https://dash.cloudflare.com/
- **Zero Trust Dashboard**: https://one.dash.cloudflare.com/
- **DNS Management**: https://dash.cloudflare.com/[account]/254carbon.com/dns

### API Tokens (Rotate Regularly)
- **DNS Token**: Manage DNS records
- **Apps Token**: Manage Access applications  
- **Tunnel Token**: Modify tunnel configuration

---

## Daily Health Checks

### 1. Tunnel Status Check
```bash
# Check tunnel pods are running
kubectl get pods -n cloudflare-tunnel

# Expected output: 2/2 pods Running
# If any pod is not Running, investigate with:
kubectl describe pod <pod-name> -n cloudflare-tunnel
kubectl logs -n cloudflare-tunnel <pod-name>
```

### 2. Connection Verification
```bash
# Check tunnel has active connections
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=50 | grep "Connection"

# Expected: Should see "Registered tunnel connection" messages
# Healthy tunnel maintains 4+ connections
```

### 3. DNS Resolution Test
```bash
# Test a few key services
for svc in portal grafana vault harbor; do
  echo -n "$svc.254carbon.com: "
  nslookup $svc.254carbon.com | grep Address | tail -1
done

# Expected: All should resolve to Cloudflare IPs (104.21.x.x or 172.67.x.x)
```

### 4. Service Accessibility
```bash
# Quick HTTP status check
for svc in portal grafana superset harbor; do
  echo -n "$svc.254carbon.com: "
  curl -s -o /dev/null -w "%{http_code}\n" -L --max-time 10 https://$svc.254carbon.com
done

# Expected: 200 (OK), 302 (Redirect), or 401 (Auth required)
# 502/503 indicates backend service issues
# Connection timeout indicates tunnel/DNS issues
```

---

## Common Operational Tasks

### Restart Tunnel

**When**: Tunnel connections dropping, configuration changes, credential updates

```bash
# Rolling restart (maintains availability)
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel

# Wait for rollout to complete
kubectl rollout status deployment/cloudflared -n cloudflare-tunnel

# Verify connections re-established
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=20
```

### Update Tunnel Configuration

**When**: Adding/removing services, changing routing rules

```bash
# Edit the configmap
kubectl edit configmap cloudflared-config -n cloudflare-tunnel

# Or apply from file
kubectl apply -f k8s/cloudflare/tunnel-config.yaml

# Restart tunnel to pick up changes
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel
```

### Add New DNS Record

**When**: Deploying new service that needs public DNS

```bash
# Option 1: Use automation script
export CLOUDFLARE_API_TOKEN="<your-token>"
export CLOUDFLARE_ZONE_ID="799bab5f5bb86d6de6dd0ec01a143ef8"
export CLOUDFLARE_TUNNEL_ID="291bc289-e3c3-4446-a9ad-8e327660ecd5"

# Add single service
./scripts/add-cloudflare-dns-record.sh newservice 254carbon.com

# Option 2: Manual via Dashboard
# 1. Go to dash.cloudflare.com → 254carbon.com → DNS
# 2. Add CNAME record:
#    - Name: <subdomain>
#    - Target: 291bc289-e3c3-4446-a9ad-8e327660ecd5.cfargotunnel.com
#    - Proxied: Yes (orange cloud)
```

### Create Cloudflare Access Application

**When**: New service needs SSO protection

```bash
# Use automation script
./scripts/create-cloudflare-access-apps.sh \
  -t "<APPS_TOKEN>" \
  -a "0c93c74d5269a228e91d4bf91c547f56" \
  --mode zone \
  --zone-domain "254carbon.com" \
  --allowed-email-domains "254carbon.com" \
  --force

# Or manually in Dashboard:
# 1. Go to one.dash.cloudflare.com → Access → Applications
# 2. Click "Add an application" → Self-hosted
# 3. Configure:
#    - Application name: Service Name
#    - Application domain: service.254carbon.com
#    - Session duration: As needed (2h-24h)
# 4. Create policy allowing @254carbon.com
```

### Update Ingress SSO Annotations

**When**: Changing Zero Trust team domain, enabling/disabling SSO

```bash
# Edit ingress
kubectl edit ingress <service>-ingress -n <namespace>

# Add/update annotations:
nginx.ingress.kubernetes.io/auth-url: "https://<team>.cloudflareaccess.com/cdn-cgi/access/authorize"
nginx.ingress.kubernetes.io/auth-signin: "https://<team>.cloudflareaccess.com/cdn-cgi/access/login"
nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"

# Apply changes
kubectl apply -f k8s/ingress/<service>-ingress.yaml
```

---

## Incident Response

### Scenario 1: All Services Inaccessible

**Symptoms**: All *.254carbon.com domains return connection errors or timeouts

**Diagnosis**:
```bash
# 1. Check DNS resolution
nslookup portal.254carbon.com

# 2. Check tunnel pods
kubectl get pods -n cloudflare-tunnel

# 3. Check tunnel logs
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=100

# 4. Check Cloudflare Dashboard for tunnel status
```

**Resolution**:
- If DNS not resolving: Check Cloudflare DNS settings in dashboard
- If tunnel pods not running: Check pod events and restart
- If no tunnel connections: Rotate credentials
- If Cloudflare reports issues: Check status.cloudflare.com

### Scenario 2: Single Service Inaccessible

**Symptoms**: One service returns 502/503, others work fine

**Diagnosis**:
```bash
# 1. Check backend pod status
kubectl get pods -n <namespace> | grep <service>

# 2. Check pod logs
kubectl logs -n <namespace> <pod-name>

# 3. Check ingress configuration
kubectl get ingress <service>-ingress -n <namespace> -o yaml

# 4. Test from ingress controller
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
curl -H "Host: service.254carbon.com" http://localhost:8080
```

**Resolution**:
- If backend pod not running: Check pod events, resources, image pull
- If ingress misconfigured: Fix and reapply ingress YAML
- If service not responding: Check service selector matches pod labels

### Scenario 3: SSO Authentication Failing

**Symptoms**: Redirected to Cloudflare Access but login fails or loops

**Diagnosis**:
```bash
# 1. Check Access application exists
# Go to one.dash.cloudflare.com → Access → Applications

# 2. Check ingress annotations
kubectl get ingress <service>-ingress -n <namespace> -o yaml | grep auth-url

# 3. Check Access policy
# Verify allowed email domains/users in dashboard

# 4. Check browser network tab for auth errors
```

**Resolution**:
- If application doesn't exist: Create via script or dashboard
- If auth-url wrong: Update ingress annotations
- If policy rejects user: Update policy in dashboard
- If JWT validation fails: Check service configuration accepts cf-access-jwt-assertion header

### Scenario 4: Tunnel Connection Drops

**Symptoms**: Intermittent connection failures, tunnel logs show reconnections

**Diagnosis**:
```bash
# 1. Check tunnel metrics
kubectl port-forward -n cloudflare-tunnel svc/cloudflared-metrics 2000:2000
curl http://localhost:2000/metrics | grep cloudflared_tunnel

# 2. Check pod resource usage
kubectl top pods -n cloudflare-tunnel

# 3. Check for pod restarts
kubectl get pods -n cloudflare-tunnel
```

**Resolution**:
- If resource constrained: Increase CPU/memory limits
- If network issues: Check cluster networking, node connectivity
- If credentials expired: Rotate tunnel credentials
- If persistent: Scale up replicas temporarily

---

## Maintenance Procedures

### Credential Rotation

**Frequency**: Quarterly or after suspected compromise

**Procedure**:
```bash
# 1. Generate new credentials in Cloudflare Dashboard
# Networks → Tunnels → 254carbon-cluster → Configure → Regenerate credentials

# 2. Extract new credentials JSON

# 3. Update Kubernetes secret
./scripts/update-cloudflare-credentials.sh <TUNNEL_ID> <ACCOUNT_TAG> <AUTH_TOKEN>

# 4. Verify tunnel reconnects
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -f
```

### Certificate Renewal

**Frequency**: Automatic (Let's Encrypt: every 60 days, Cloudflare: every 15 years)

**Manual Trigger** (if needed):
```bash
# Delete certificate to trigger renewal
kubectl delete certificate <cert-name> -n <namespace>

# Check cert-manager creates new certificate
kubectl get certificate -A -w

# Verify new secret created
kubectl get secret <cert-name>-tls -n <namespace>
```

### DNS Record Audit

**Frequency**: Monthly

**Procedure**:
```bash
# 1. List all current DNS records
curl -X GET "https://api.cloudflare.com/client/v4/zones/799bab5f5bb86d6de6dd0ec01a143ef8/dns_records" \
  -H "Authorization: Bearer <DNS_TOKEN>" \
  | jq '.result[] | {name: .name, type: .type, content: .content}'

# 2. Compare with expected services (14 subdomains)
# 3. Remove stale records
# 4. Add missing records for new services
```

### Access Policy Review

**Frequency**: Quarterly

**Procedure**:
```bash
# 1. List all Access applications
curl -X GET "https://api.cloudflare.com/client/v4/accounts/0c93c74d5269a228e91d4bf91c547f56/access/apps" \
  -H "Authorization: Bearer <APPS_TOKEN>" \
  | jq '.result[] | {name: .name, domain: .domain}'

# 2. For each application:
#    - Review policies (allowed users/groups)
#    - Check session duration appropriateness
#    - Verify application still in use
# 3. Remove unused applications
# 4. Tighten overly permissive policies
```

---

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Tunnel Connectivity**
   - Metric: `cloudflared_tunnel_connections_registered`
   - Threshold: < 2 connections (alert)
   - Source: `kubectl port-forward -n cloudflare-tunnel svc/cloudflared-metrics 2000:2000`

2. **Pod Health**
   - Metric: Pod restarts, crash loops
   - Threshold: > 3 restarts in 1 hour (alert)
   - Source: `kubectl get pods -n cloudflare-tunnel`

3. **DNS Resolution Time**
   - Metric: Time to resolve *.254carbon.com
   - Threshold: > 500ms (warning), > 2s (alert)
   - Tool: `dig +stats portal.254carbon.com`

4. **Service Response Time**
   - Metric: HTTP response time through tunnel
   - Threshold: > 2s (warning), > 5s (alert)
   - Tool: `curl -w "%{time_total}\n" -o /dev/null -s https://portal.254carbon.com`

### Recommended Grafana Dashboards

```yaml
# Cloudflare Tunnel Dashboard
- Panel: Active Tunnel Connections (gauge)
- Panel: Request Rate (graph)
- Panel: Error Rate (graph)
- Panel: Pod Memory/CPU Usage (graph)
- Panel: Restart Count (stat)
```

### Alert Rules

```yaml
# Example Prometheus alert rules
groups:
  - name: cloudflare_tunnel
    rules:
      - alert: TunnelConnectionsLow
        expr: cloudflared_tunnel_connections_registered < 2
        for: 5m
        annotations:
          summary: "Cloudflare Tunnel has insufficient connections"
          
      - alert: TunnelPodDown
        expr: kube_deployment_status_replicas_available{deployment="cloudflared"} < 2
        for: 2m
        annotations:
          summary: "Cloudflare Tunnel pod(s) not available"
```

---

## Troubleshooting Guide

### DNS Not Resolving

**Check**:
1. Record exists in Cloudflare DNS
2. Nameservers point to Cloudflare (louis.ns.cloudflare.com, reza.ns.cloudflare.com)
3. DNS propagation completed (check from multiple locations)
4. No conflicting local DNS caching

**Tools**:
```bash
dig portal.254carbon.com
nslookup portal.254carbon.com 8.8.8.8
whois 254carbon.com | grep "Name Server"
```

### 502 Bad Gateway

**Causes**:
- Backend pod not running
- Service selector mismatch
- Backend not listening on expected port
- Backend taking too long to respond

**Debug**:
```bash
kubectl get pods -n <namespace>
kubectl logs -n <namespace> <pod-name>
kubectl describe svc <service-name> -n <namespace>
```

### 503 Service Unavailable

**Causes**:
- No backend pods available
- All pods failing health checks
- Service has no endpoints

**Debug**:
```bash
kubectl get endpoints <service-name> -n <namespace>
kubectl describe pod <pod-name> -n <namespace>
```

### SSL/TLS Errors

**Causes**:
- Certificate expired
- Certificate for wrong domain
- Cert-manager webhook not working
- Missing TLS secret

**Debug**:
```bash
kubectl get certificate -A
kubectl describe certificate <cert-name> -n <namespace>
kubectl get secret <cert-name>-tls -n <namespace>
openssl s_client -connect portal.254carbon.com:443 -servername portal.254carbon.com
```

---

## Rollback Procedures

### Disable Cloudflare Tunnel

**When**: Emergency - tunnel causing widespread issues

```bash
# 1. Scale tunnel to 0 replicas
kubectl scale deployment cloudflared -n cloudflare-tunnel --replicas=0

# 2. Services become inaccessible via *.254carbon.com
# 3. Use port-forward for emergency access:
kubectl port-forward -n <namespace> svc/<service> 8080:<port>

# 4. Re-enable when ready
kubectl scale deployment cloudflared -n cloudflare-tunnel --replicas=2
```

### Disable Cloudflare Access (SSO)

**When**: SSO blocking legitimate access

```bash
# Remove auth annotations from ingress
kubectl edit ingress <service>-ingress -n <namespace>

# Delete these lines:
# nginx.ingress.kubernetes.io/auth-url
# nginx.ingress.kubernetes.io/auth-signin  
# nginx.ingress.kubernetes.io/auth-response-headers

# Apply changes immediately
kubectl apply -f k8s/ingress/<service>-ingress.yaml
```

### Revert to Previous Tunnel Configuration

```bash
# 1. Get previous configmap version
kubectl rollout history deployment/cloudflared -n cloudflare-tunnel

# 2. Rollback
kubectl rollout undo deployment/cloudflared -n cloudflare-tunnel

# 3. Verify
kubectl rollout status deployment/cloudflared -n cloudflare-tunnel
```

---

## Contact Information

### Escalation Path
1. **L1**: Platform Team (restart pods, check logs)
2. **L2**: DevOps Lead (credential rotation, configuration changes)
3. **L3**: Cloudflare Support (tunnel/DNS issues beyond our control)

### External Support
- **Cloudflare Support**: https://support.cloudflare.com/
- **Status Page**: https://www.cloudflarestatus.com/
- **Community**: https://community.cloudflare.com/

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-10-20 | Initial runbook creation | AI Agent |
| 2025-10-20 | Added incident response procedures | AI Agent |
| 2025-10-20 | Added monitoring section | AI Agent |

