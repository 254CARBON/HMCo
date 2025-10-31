# Cloudflare Tunnel Deployment Guide (Canonical)

## Complete Step-by-Step Implementation

This guide walks through the complete process of deploying Cloudflare Tunnel integration for your Kubernetes cluster.

## Quick Start: One-Command Bootstrap

For rapid deployment with pre-configured values:

```bash
# Set environment variables
export CLOUDFLARE_API_TOKEN=<your-api-token>
export CLOUDFLARE_ACCOUNT_ID=<your-account-id>
export CLOUDFLARE_TUNNEL_ID=<your-tunnel-id>
export CLOUDFLARE_ZONE_NAME=254carbon.com

# One-command bootstrap (creates tunnel config, DNS records, and Access apps)
./scripts/bootstrap-cloudflare.sh
```

This will:
1. Configure Cloudflare tunnel token in Kubernetes
2. Create all DNS records pointing to the tunnel
3. Set up Cloudflare Access applications and policies
4. Deploy cloudflared to Kubernetes
5. Verify the setup

**Prerequisites for one-command bootstrap:**
- Cloudflare tunnel already created (see Phase 1 below if needed)
- API token with permissions: Tunnel:Read, DNS:Edit, Access:Edit
- kubectl configured and connected to your cluster
- Tunnel credentials available

For detailed step-by-step instructions or troubleshooting, continue with Phase 1 below.

---

## Phase 1: Cloudflare Setup (Dashboard)

### Step 1.1: Create Tunnel in Cloudflare

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Navigate to **Networks** → **Tunnels** (requires Zero Trust subscription)
3. Click **Create a tunnel**
4. Select **Cloudflared** as the connector
5. Enter tunnel name: `254carbon-cluster`
6. Click **Save tunnel**
7. On the next screen, you'll see installation instructions - **ignore these** (we'll use Kubernetes)

### Step 1.2: Get Tunnel Credentials

1. From the tunnel configuration page, click **Credentials**
2. You'll see a credentials JSON file - download or copy it
3. Extract these values:
   - `tunnel_id` - UUID format
   - `account_tag` - Numeric ID
   - `auth_token` - Long base64 string
4. **Save these securely** - you'll need them soon

### Step 1.3: Verify 254carbon.com is Active

1. In Cloudflare dashboard, ensure `254carbon.com` is added as a domain
2. Note the **Zone ID** for your domain (you can find in Domain → Overview → Zone ID)
3. This is needed for the DNS setup script

### Step 1.4: Generate API Token (if using DNS script)

If you want to automate DNS record creation:

1. Go to **My Profile** → **API Tokens**
2. Click **Create Token**
3. Use template **Edit zone DNS** or create custom with:
   - **Zone Resources**: Include `254carbon.com`
   - **Permissions**: `Zone:DNS:Edit`
4. Copy the token (this is what you provided: `HsmXB0pAPV7ejbWFrpQt148LoxksjQKxJGRn4J7N`)

---

## Phase 2: Kubernetes Deployment

### Step 2.1: Create Namespace and RBAC

```bash
cd /home/m/tff/254CARBON/HMCo

kubectl apply -f k8s/cloudflare/namespace.yaml

# Verify
kubectl get namespace cloudflare-tunnel
kubectl get serviceaccount -n cloudflare-tunnel cloudflared
```

Expected output:
```
NAME                STATUS   AGE
cloudflare-tunnel   Active   2s
```

### Step 2.2: Create Tunnel Credentials Secret

**Option A: Using the helper script (Recommended)**

```bash
# Make sure you have the three values from Cloudflare
./scripts/update-cloudflare-credentials.sh TUNNEL_ID ACCOUNT_TAG AUTH_TOKEN

# Example:
./scripts/update-cloudflare-credentials.sh \
  "abc123def456" \
  "1234567890" \
  "eyJhIjp..."
```

**Option B: Manual edit**

```bash
# First, deploy the template
kubectl apply -f k8s/cloudflare/tunnel-secret.yaml

# Then edit it
kubectl edit secret cloudflare-tunnel-credentials -n cloudflare-tunnel
```

Update the `credentials.json` field with your actual values.

### Step 2.3: Deploy cloudflared

```bash
kubectl apply -f k8s/cloudflare/cloudflared-deployment.yaml

# Wait for pods to be running
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=cloudflare-tunnel \
  -n cloudflare-tunnel \
  --timeout=120s

# Verify deployment
kubectl get pods -n cloudflare-tunnel -w

# Check tunnel connection in logs
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=50
```

Expected log output:
```
INF Connection established...
INF Registered tunnel with Cloudflare
INF Connected successfully
```

### Step 2.4: Update Ingress Rules

```bash
kubectl apply -f k8s/ingress/ingress-rules.yaml

# Verify ingress is created
kubectl get ingress -A | grep 254carbon

# Check ingress details
kubectl describe ingress datahub-ingress -n data-platform
```

---

## Phase 3: DNS Configuration

### Step 3.1: Automated DNS Setup (Recommended)

```bash
# Using the API token from Step 1.4
./scripts/setup-cloudflare-dns.sh \
  -t "HsmXB0pAPV7ejbWFrpQt148LoxksjQKxJGRn4J7N" \
  -z "your_zone_id"

# If Zone ID not provided, script will fetch it automatically
./scripts/setup-cloudflare-dns.sh \
  -t "HsmXB0pAPV7ejbWFrpQt148LoxksjQKxJGRn4J7N"
```

Expected output:
```
================================
Cloudflare DNS Configuration
================================

Domain: 254carbon.com
Tunnel Endpoint: 254carbon-cluster.cfargotunnel.com

Creating DNS records:

Configuring datahub.254carbon.com... ✓ (DataHub - Metadata Platform)
Configuring grafana.254carbon.com... ✓ (Grafana - Monitoring Dashboards)
...
All DNS records configured successfully!
```

### Step 3.2: Manual DNS Setup (if script fails)

In Cloudflare dashboard, create these CNAME records:

| Subdomain | CNAME Target | Proxy | TTL |
|-----------|---|---|---|
| datahub | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| grafana | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| superset | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| vault | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| trino | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| clickhouse | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| dolphin | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| minio | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |
| lakefs | 254carbon-cluster.cfargotunnel.com | ✅ Orange (Proxied) | Auto |

---

## Phase 4: Verification

### Step 4.1: Verify Tunnel is Connected

```bash
# Check pod status
kubectl get pods -n cloudflare-tunnel

# Should show 2 running pods (deployment replicas: 2)
NAME                           READY   STATUS    RESTARTS   AGE
cloudflared-7f6d9c8f94-abc12   1/1     Running   0          2m
cloudflared-7f6d9c8f94-xyz98   1/1     Running   0          2m

# View tunnel logs
kubectl logs -n cloudflare-tunnel -f -l app.kubernetes.io/name=cloudflare-tunnel
```

### Step 4.2: Test DNS Resolution

```bash
# From outside the cluster (your laptop)
nslookup grafana.254carbon.com

# Should resolve to Cloudflare IP (not your local IP)
# Example output:
# Name:   grafana.254carbon.com
# Address: 104.16.132.229  (this is Cloudflare, not your IP)
```

### Step 4.3: Test Service Access

```bash
# Test each service (from outside your network)
curl -I https://grafana.254carbon.com
curl -I https://superset.254carbon.com
curl -I https://datahub.254carbon.com

# Should return 200 OK or 302 (if redirect) with Cloudflare headers:
# cf-ray: <some-id>
# cf-cache-status: MISS
```

### Step 4.4: Verify in Cloudflare Dashboard

1. Go to **Networks** → **Tunnels** → **254carbon-cluster**
2. Status should show **HEALTHY**
3. Click **Status** - should show connected connectors
4. View **Analytics** to see traffic

---

## Phase 5: Security Configuration

### Step 5.1: Enable Cloudflare Access (Optional)

For protected services (Vault, MinIO, DolphinScheduler):

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/)
2. Navigate to **Applications** → **Create Application** → **Self-hosted**
3. Follow instructions in [ACCESS_APPS_AND_POLICIES.md](ACCESS_APPS_AND_POLICIES.md)

### Step 5.2: Enable WAF Rules

1. Go to **Security** → **WAF** → **Manage Rules**
2. Enable these rule sets:
   - Cloudflare Managed Ruleset
   - OWASP ModSecurity Core Ruleset
3. Go to **Security** → **Bots** → Enable **Super Bot Fight Mode**

### Step 5.3: Configure Rate Limiting

In Cloudflare dashboard:

1. **Security** → **Rate Limiting**
2. Add rules:
   - Standard APIs: 100 req/min per IP
   - Auth endpoints: 10 req/min per IP
3. Action: Block or Challenge

---

## Troubleshooting

### Tunnel Not Connecting

```bash
# 1. Check credentials are correct
kubectl get secret -n cloudflare-tunnel cloudflare-tunnel-credentials -o yaml

# 2. Verify tunnel exists in Cloudflare dashboard
# 3. Check pod logs for auth errors
kubectl logs -n cloudflare-tunnel deployment/cloudflared | grep -i "error\|auth"

# 4. Try restarting
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel
```

### Services Not Accessible

```bash
# 1. Verify ingress is created
kubectl get ingress -A | grep 254carbon

# 2. Test locally (from cluster)
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
curl -H "Host: grafana.254carbon.com" http://localhost:8080

# 3. Verify DNS is resolving
nslookup grafana.254carbon.com

# 4. Check NGINX logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -f
```

### Certificate Errors

```bash
# Check cert-manager status
kubectl get certificate -A | grep 254carbon

# Describe specific certificate
kubectl describe certificate datahub-tls -n data-platform

# Check cert-manager logs
kubectl logs -n cert-manager -l app.kubernetes.io/name=cert-manager -f
```

---

## Maintenance

### Update Credentials

When rotating credentials:

```bash
# Get new credentials from Cloudflare dashboard
# Then run:
./scripts/update-cloudflare-credentials.sh NEW_TUNNEL_ID NEW_ACCOUNT_TAG NEW_AUTH_TOKEN
```

### Monitor Tunnel Health

```bash
# View metrics
kubectl port-forward -n cloudflare-tunnel svc/cloudflared-metrics 2000:2000

# In another terminal
curl http://localhost:2000/metrics | grep cloudflared

# Monitor in Cloudflare dashboard
# Networks → Tunnels → 254carbon-cluster → Analytics
```

### Backup Configuration

```bash
# Backup all configurations
kubectl get configmap,secret -n cloudflare-tunnel -o yaml > cloudflare-backup-$(date +%Y%m%d).yaml

# Backup Access applications and policies
./scripts/export-cloudflare-access-apps.sh --output docs/cloudflare/access-baseline.json

# Backup DNS records (via Cloudflare API)
curl -X GET "https://api.cloudflare.com/client/v4/zones/ZONE_ID/dns_records" \
  -H "Authorization: Bearer YOUR_API_TOKEN" > dns-backup.json

# Version control the backups
git add docs/cloudflare/access-baseline.json
git commit -m "Backup Cloudflare configuration"
```

### Drift Detection and Remediation

The infrastructure should be managed as code. To detect and fix configuration drift:

```bash
# Detect DNS drift - compare current DNS with desired state
# Option 1: Dry run to see what would change
./scripts/create-cloudflare-dns-records.sh --dry-run

# Option 2: Apply changes to fix drift
./scripts/create-cloudflare-dns-records.sh --force

# Detect Access policy drift
# Export current state and compare with baseline
./scripts/export-cloudflare-access-apps.sh --output /tmp/access-current.json
diff docs/cloudflare/access-baseline.json /tmp/access-current.json

# Fix Access drift - reconcile to desired state
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com \
  --force

# Full infrastructure reconciliation
./scripts/bootstrap-cloudflare.sh --skip-deploy
```

**Recommended**: Run drift detection weekly or as part of CI/CD:

```bash
#!/bin/bash
# drift-check.sh - Add to cron or CI

# Check DNS
./scripts/create-cloudflare-dns-records.sh --dry-run > /tmp/dns-drift.log
if grep -q "Would create\|Would update" /tmp/dns-drift.log; then
    echo "DNS drift detected!"
    # Send alert or auto-remediate
    ./scripts/create-cloudflare-dns-records.sh --force
fi

# Check Access
./scripts/export-cloudflare-access-apps.sh --output /tmp/access-current.json
if ! diff -q docs/cloudflare/access-baseline.json /tmp/access-current.json; then
    echo "Access policy drift detected!"
    # Send alert or auto-remediate
    ./scripts/create-cloudflare-access-apps.sh --force
fi
```

---

## Support

For issues, check:

1. Tunnel logs: `kubectl logs -n cloudflare-tunnel -f`
2. Ingress logs: `kubectl logs -n ingress-nginx -f`
3. Cloudflare dashboard → Networks → Tunnels → Analytics
4. [Troubleshooting Guide](troubleshooting.md)
5. [Access Apps & Policies](ACCESS_APPS_AND_POLICIES.md)
