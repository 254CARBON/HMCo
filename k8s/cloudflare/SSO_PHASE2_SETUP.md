# SSO Phase 2: Cloudflare Access Configuration Guide (Moved — see docs/sso/guide.md)

Complete step-by-step guide for configuring Cloudflare Access (Zero Trust team: qagi) for 254Carbon SSO implementation.

**Duration**: 1-2 hours  
**Prerequisites**: Cloudflare Teams/Enterprise subscription, 254carbon.com domain in Cloudflare  
**Team Name**: qagi

## Overview

Phase 2 involves creating Cloudflare Access applications for the portal and all 9 services. After Phase 2 is complete, users will authenticate once through Cloudflare and have single-session access to all services.

## Prerequisites Checklist

Before starting Phase 2, verify:

- [ ] Cloudflare account with Teams/Enterprise subscription active
- [ ] 254carbon.com domain configured in Cloudflare
- [ ] Cloudflare Tunnel (cloudflared) deployed and running in cluster
- [ ] Portal deployed to Kubernetes (2 replicas running)
- [ ] All 9 services deployed and running
- [ ] Tunnel is connected and showing "Connected" status in dashboard

**Verify Tunnel Status**:
```bash
# In Cloudflare Dashboard:
# Go to: Infrastructure → Tunnels → Your Tunnel
# Status should show "Connected" with green icon

# Or check cluster logs:
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -f | grep "Registered tunnel"
```

## Step 1: Access Cloudflare Zero Trust Dashboard

### 1.1 Navigate to Zero Trust

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Select **Zero Trust** from the left sidebar
3. Or go directly to: https://dash.cloudflare.com/zero-trust

### 1.2 Verify Team Setup

Once in Zero Trust:

1. Look for **Team name** in top-right corner
   - Should show "qagi" or your team name
2. If this is first time:
   - Accept terms and conditions
   - Complete team setup wizard
3. Go to **Settings** → **Team** to verify team is properly configured

### 1.3 Get Your Account ID

**Critical**: You'll need your Account ID for configuring ingress rules later.

1. Go to **Settings** → **Account**
2. Look for **Account ID** field (32-character alphanumeric string)
3. Copy and save this ID - you'll use it in Phase 3
4. Example format: `abc123def456ghi789jkl012mno345pq`

**Save This For Later**:
```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id-here"
```

## Step 2: Configure Authentication Method

### 2.1 Enable Email OTP

1. Go to **Settings** → **Authentication**
2. Under **Login methods**, ensure **Email** is enabled
3. Look for **One-time PIN (Email)** - ensure it's toggled ON
4. This will allow users to receive one-time codes via email

### 2.2 Session Configuration

1. Go to **Settings** → **Session**
2. Configure default session duration:
   - **Session duration**: 24 hours (can be adjusted per application)
   - **Idle timeout**: 8 hours
3. Enable **Require re-authentication to access account settings**
4. Click **Save**

### 2.3 Security Settings

1. Go to **Settings** → **Security**
2. Enable **Require proof of presence**
3. Enable **Limit countries allowed to access Access applications**
4. Select appropriate countries/regions
5. Click **Save**

## Step 3: Create Portal Application

This is the main entry point where users authenticate.

### Automation (Optional)

Prefer to create everything via API? Use `scripts/create-cloudflare-access-apps.sh` to provision the portal and all nine service apps in one run. Supply a Cloudflare API token with **Access: Apps** write scope, your account ID, and the base domain you want Cloudflare Access to protect (example below uses `254carbon.com`):
```bash
CLOUDFLARE_API_TOKEN=cf_api_token_here \
CLOUDFLARE_ACCOUNT_ID=0c93c74d5269a228e91d4bf91c547f56 \
./scripts/create-cloudflare-access-apps.sh --base-domain 254carbon.com
# Use --force to reconcile settings if apps already exist
```

### 3.1 Create New Application

1. Go to **Access** → **Applications**
2. Click **Add an application** button
3. Select **Self-hosted** (NOT Saas)

### 3.2 Configure Application Details

**Application name**: `254Carbon Portal`  
**Subdomain**: `254carbon`  
**Domain**: `cloudflareaccess.com` (from dropdown)  
**Application type**: `Web`  
**Session duration**: `24 hours`

Click **Next** after filling in details.

### 3.3 Configure Policy

On the **Policies** tab:

1. Click **+ Add a policy**
2. Create **Policy 1: Allow Portal Access**
   - **Policy name**: `Allow Portal Access`
   - **Decision**: `Allow`
   - **Include**: 
     - Selector: `Everyone`
   - Click **Save policy**

The default "Deny All Others" will automatically apply as fallback.

### 3.4 Configure Settings

Click **Settings** tab:

1. **Session duration**: Keep as 24 hours
2. **HTTP-Only Cookies**: Toggle ON (for security)
3. **Auto-redirect to identity provider**: Leave OFF (let users see login page)
4. Click **Save application**

**Result**: Portal now accessible at `https://254carbon.cloudflareaccess.com`

## Step 4: Create Service Applications

Create 9 applications for the services. Use the same process for each.

### 4.1 Grafana Application

1. Go to **Access** → **Applications**
2. Click **Add an application** → **Self-hosted**

**Details**:
- **Name**: `Grafana.254Carbon`
- **Subdomain**: `grafana`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `24 hours`

Click **Next**.

**Policy**:
- **Name**: `Allow Grafana Access`
- **Decision**: `Allow`
- **Include**: `Everyone`

Click **Save application**.

### 4.2 Superset Application

**Details**:
- **Name**: `Superset.254Carbon`
- **Subdomain**: `superset`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `24 hours`

**Policy**: Allow Everyone

### 4.3 Vault Application (Sensitive)

**Details**:
- **Name**: `Vault.254Carbon`
- **Subdomain**: `vault`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `2 hours` (shorter for sensitive service)

**Policy**: Allow Everyone

### 4.4 MinIO Application

**Details**:
- **Name**: `MinIO.254Carbon`
- **Subdomain**: `minio`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `8 hours`

**Policy**: Allow Everyone

### 4.5 DolphinScheduler Application

**Details**:
- **Name**: `DolphinScheduler.254Carbon`
- **Subdomain**: `dolphin`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `12 hours`

**Policy**: Allow Everyone

### 4.6 DataHub Application

**Details**:
- **Name**: `DataHub.254Carbon`
- **Subdomain**: `datahub`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `12 hours`

**Policy**: Allow Everyone

### 4.7 Trino Application

**Details**:
- **Name**: `Trino.254Carbon`
- **Subdomain**: `trino`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `8 hours`

**Policy**: Allow Everyone

### 4.8 Doris Application

**Details**:
- **Name**: `Doris.254Carbon`
- **Subdomain**: `doris`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `8 hours`

**Policy**: Allow Everyone

### 4.9 LakeFS Application

**Details**:
- **Name**: `LakeFS.254Carbon`
- **Subdomain**: `lakefs`
- **Domain**: `cloudflareaccess.com`
- **Type**: `Web`
- **Session duration**: `12 hours`

**Policy**: Allow Everyone

## Step 5: Verify All Applications Created

After creating all 10 applications:

1. Go to **Access** → **Applications**
2. Should see list of all applications:
   - 254Carbon Portal
   - Grafana.254Carbon
   - Superset.254Carbon
   - DataHub.254Carbon
   - Trino.254Carbon
   - Doris.254Carbon
   - Vault.254Carbon
   - MinIO.254Carbon
   - DolphinScheduler.254Carbon
   - LakeFS.254Carbon

3. Verify each has:
   - CNAME record (shown in application details)
   - Policy configured
   - Correct session duration

## Step 6: Configure Advanced Settings (Optional)

### 6.1 JWT Token Configuration

To enable JWT token validation in services:

1. Go to **Access** → **Applications**
2. Select an application
3. Go to **Advanced** tab
4. Note down **JWT Token Signing Key**
5. Save for later use if services need token validation

### 6.2 mTLS Configuration (Optional)

For additional security with service-to-service communication:

1. Go to **Settings** → **mTLS**
2. Enable if needed for your environment
3. Configure per application if required

## Step 7: Set Up Audit Logging

### 7.1 Enable Audit Logs

1. Go to **Access** → **Logs**
2. Verify logs are being collected
3. Check for any authentication attempts

### 7.2 Configure Alerts

1. Go to **Settings** → **Notifications**
2. Configure email alerts for:
   - Failed authentications (threshold: > 3 in 1 hour)
   - Policy changes
   - Application modifications
3. Enter email addresses for alerts
4. Save configuration

## Step 8: Test Portal Access

### 8.1 Basic Connectivity Test

From your terminal:
```bash
# Test portal is accessible through Cloudflare tunnel
curl -v https://254carbon.cloudflareaccess.com 2>&1 | head -30
# Should show 200 OK or 302 redirect to login
```

### 8.2 Browser Test (Important!)

This confirms the Cloudflare Access integration is working:

1. Open new **private/incognito** browser window
2. Navigate to `https://254carbon.com`
3. **Expected**: Should redirect to Cloudflare login page
4. Page should show:
   - "Sign in with your email"
   - Email input field
   - "Send me a code" button

### 8.3 Authentication Test

1. At Cloudflare login page, enter your email
2. Click "Send me a code"
3. Check email for one-time code
4. Enter code on login page
5. Should redirect to portal
6. Portal should display all 9 service cards

**If this works**, Phase 2 is complete!

## Troubleshooting Phase 2

### Problem: Can't access Zero Trust dashboard

**Solution**:
1. Verify Cloudflare Teams/Enterprise subscription is active
2. Check that you have admin privileges on the Cloudflare account
3. Try logging out and logging back in
4. Clear browser cache

### Problem: Applications not appearing in list

**Solution**:
1. Refresh the page (F5)
2. Check browser console for errors (F12)
3. Verify you're in correct team (top-right corner)
4. Try creating application again

### Problem: CNAME records not created

**Solution**:
1. CNAME records are created automatically after application save
2. Wait 5-10 minutes for Cloudflare to propagate
3. Check DNS records in Cloudflare dashboard
4. If still missing, try deleting and recreating application

### Problem: Portal redirect doesn't work

**Solution**:
1. Verify portal ingress is deployed: `kubectl get ingress -n data-platform | grep portal`
2. Verify Cloudflare tunnel is connected
3. Check tunnel logs: `kubectl logs -n cloudflare-tunnel -f`
4. Verify tunnel configuration includes portal routes
5. Restart tunnel if needed: `kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel`

### Problem: Email OTP not received

**Solution**:
1. Check spam/junk email folder
2. Verify email address is correct
3. Try again after a few seconds
4. If persistent, check Cloudflare application logs

## Next Steps: Phase 3

After Phase 2 is complete:

1. Proceed to Phase 3: Service Integration
   - Update ingress rules with Cloudflare auth annotations
   - Update tunnel credentials
   - Disable local authentication in Grafana and Superset

2. Run the setup script:
   ```bash
   export CLOUDFLARE_ACCOUNT_ID="your-account-id"
   export CLOUDFLARE_TUNNEL_ID="your-tunnel-id"
   bash scripts/sso-setup-phase2.sh
   ```

3. Continue with Phase 4: Testing & Validation

## Documentation Links

- **Cloudflare Access Docs**: https://developers.cloudflare.com/cloudflare-one/
- **Application Configuration**: https://developers.cloudflare.com/cloudflare-one/applications/
- **Policy Documentation**: https://developers.cloudflare.com/cloudflare-one/policies/access/

## Completion Checklist

- [ ] Accessed Cloudflare Zero Trust dashboard with team "qagi"
- [ ] Got and saved Account ID
- [ ] Configured email OTP authentication
- [ ] Created portal application (254Carbon Portal)
- [ ] Created 9 service applications (all verified in list)
- [ ] Set appropriate session durations for each service
- [ ] Verified CNAME records for all applications
- [ ] Configured audit logging
- [ ] Tested portal access and authentication
- [ ] Email OTP delivery working
- [ ] Portal loads after authentication
- [ ] All 9 service cards visible on portal

**Phase 2 Status**: COMPLETE ✓

Proceed to Phase 3 when all items are checked.
