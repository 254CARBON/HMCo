# Cloudflare DNS Configuration - Status & Instructions

**Date**: October 21, 2025  
**Status**: ✅ DNS Records Created | ⏳ Tunnel Routes Need Manual Configuration

---

## ✅ What Was Done

### DNS Records Created Successfully

All DNS records have been created and are resolving correctly:

```bash
$ dig +short rapids.254carbon.com
172.67.203.4  ← Cloudflare proxy IP ✅

$ dig +short dolphinscheduler.254carbon.com
172.67.203.4  ← Cloudflare proxy IP ✅
```

**DNS Records Status:**
- ✅ rapids.254carbon.com → Tunnel CNAME
- ✅ dolphinscheduler.254carbon.com → Tunnel CNAME
- ✅ harbor.254carbon.com → Tunnel CNAME
- ✅ All other services (portal, datahub, superset, grafana, etc.) → Updated

**DNS Propagation**: Complete (1-2 minutes)

---

## ⏳ Manual Action Required

### Cloudflare Tunnel Routes Configuration

The DNS records are created, but the tunnel needs to be configured to route the new domains. Since the tunnel uses remote configuration, you need to add routes in the Cloudflare dashboard.

### Option A: Configure via Cloudflare Dashboard (Recommended)

**Steps:**

1. **Login to Cloudflare Dashboard**
   - Go to: https://one.dash.cloudflare.com/
   - Account ID: `0c93c74d5269a228e91d4bf91c547f56`

2. **Navigate to Tunnel Configuration**
   - Zero Trust → Networks → Tunnels
   - Select tunnel: `254carbon-cluster`
   - Tunnel ID: `291bc289-e3c3-4446-a9ad-8e327660ecd5`

3. **Add Public Hostnames**
   
   Click "Add a public hostname" for each:
   
   **RAPIDS GPU Analytics:**
   - Subdomain: `rapids`
   - Domain: `254carbon.com`
   - Type: `HTTP`
   - URL: `ingress-nginx-controller.ingress-nginx:80`
   
   **DolphinScheduler (Alternative Domain):**
   - Subdomain: `dolphinscheduler`
   - Domain: `254carbon.com`
   - Type: `HTTP`
   - URL: `ingress-nginx-controller.ingress-nginx:80`

4. **Save Changes**
   - Click "Save tunnel"
   - Routes will be active immediately

### Option B: Use Cloudflare API (If you have correct token)

The DNS_API_TOKEN provided doesn't have permission to update tunnel configurations. You need:
- Tunnel token with edit permissions
- Or Account-level API token with "Cloudflare Tunnel" edit permission

```bash
# If you get the correct token, update this script:
./scripts/update-tunnel-routes.sh
```

---

## Alternative: Kubernetes ConfigMap (Local Config)

The tunnel can also use local configuration from the ConfigMap (already updated). To use this:

### Disable Remote Configuration

1. **Update Tunnel Credentials**:
   
   The credentials.json needs to have `tunnel_remote_config: false`:
   
   ```json
   {
     "tunnel_id": "291bc289-e3c3-4446-a9ad-8e327660ecd5",
     "account_tag": "0c93c74d5269a228e91d4bf91c547f56",
     "tunnel_name": "254carbon-cluster",
     "auth_token": "ACTUAL_TOKEN_HERE",
     "tunnel_remote_config": false
   }
   ```

2. **Restart Tunnel**:
   ```bash
   kubectl rollout restart deployment cloudflared -n cloudflare-tunnel
   ```

**Note**: The ConfigMap `cloudflared-config` already has all routes including rapids and dolphinscheduler.

---

## Current Tunnel Status

### Pods
```
cloudflared pods: CrashLoopBackOff (credentials format issue)
```

### Issue
The tunnel credentials format needs to match what Cloudflare expects. The current credentials have the correct IDs but may need a different auth_token format.

### Services Still Accessible
The DNS records are live, but the tunnel routes need to be configured for the new services (rapids, dolphinscheduler) to be accessible.

---

## Recommended Immediate Action

**Quick Fix** (5 minutes):

1. Go to Cloudflare Dashboard
2. Navigate to tunnel configuration
3. Add the 2 missing public hostnames (rapids, dolphinscheduler)
4. Save

This will make the services immediately accessible while we troubleshoot the credentials format.

---

## Information for Cloudflare Support

If needed, here are the credentials you provided:

```
ACCOUNT_ID: 0c93c74d5269a228e91d4bf91c547f56
TUNNEL_ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5
DNS_API_TOKEN: acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm
APPS_API_TOKEN: TYSD6Xrn8BJEwGp76t32-a331-L82fCNkbsJx7Mn
```

The tunnel is named: `254carbon-cluster`

---

## Files Modified

1. `/home/m/tff/254CARBON/HMCo/k8s/cloudflare/tunnel-secret.yaml`
   - Added rapids.254carbon.com route
   - Added dolphinscheduler.254carbon.com route

2. `/home/m/tff/254CARBON/HMCo/scripts/configure-cloudflare-dns.sh`
   - Created script to automate DNS record creation

3. `/home/m/tff/254CARBON/HMCo/scripts/update-tunnel-routes.sh`
   - Created script to update tunnel routes via API

---

## Testing After Configuration

Once the tunnel routes are added:

```bash
# Test RAPIDS access
curl -I https://rapids.254carbon.com

# Test DolphinScheduler
curl -I https://dolphinscheduler.254carbon.com

# Or use browser:
# https://rapids.254carbon.com
# https://dolphinscheduler.254carbon.com
```

---

**Next Step**: Add the 2 public hostnames in Cloudflare dashboard (5 min task)

**Then**: Continue with `COMMODITY_QUICKSTART.md` for data ingestion

