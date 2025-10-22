# Phase 2: Cloudflare Access Applications - Execution Checklist (Archived)

**Status**: READY FOR MANUAL CONFIGURATION  
**Date**: October 19, 2025  
**Duration**: 1-2 hours estimated  
**Team**: qagi (Cloudflare Zero Trust)

---

## Prerequisites - VERIFY BEFORE STARTING

```bash
# ✅ Verify tunnel connection
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=5 | grep -i "tunnel\|registered"

# ✅ Verify ingress deployed
kubectl get ingress -A | grep 254carbon | wc -l  # Should show 10

# ✅ Verify account credentials
echo "Account ID: 0c93c74d5269a228e91d4bf91c547f56"
echo "Tunnel ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5"
```

---

## Step 1: Verify Tunnel Status in Cloudflare Dashboard

**Time**: 5 minutes

### Tasks:
1. [ ] Go to https://dash.cloudflare.com/zero-trust/networks/tunnels
2. [ ] Look for tunnel named **"254carbon-cluster"**
3. [ ] Verify status shows **"Connected"** (green icon)
4. [ ] If not connected, wait 1-2 minutes and refresh
5. [ ] If still not connected, check tunnel logs (see troubleshooting)

### Expected Result:
- Tunnel shows GREEN connected status
- Connection time recent (within last few minutes)
- No error messages

---

## Step 2: Create Portal Application

**Time**: 5 minutes

### Configuration:
```
Name: 254Carbon Portal
Subdomain: 254carbon
Domain: cloudflareaccess.com
Type: Self-hosted
Application Type: Web
```

### Steps:
1. [ ] Go to https://dash.cloudflare.com/zero-trust/access/applications
2. [ ] Click "Add an application"
3. [ ] Select "Self-hosted"
4. [ ] Fill in above configuration
5. [ ] Click "Next"

### Policy Configuration:
```
Policy Name: Allow Portal Access
Decision: Allow
Include: Everyone
```

1. [ ] Click "+ Add a policy"
2. [ ] Enter policy name
3. [ ] Set decision to "Allow"
4. [ ] Set include to "Everyone"
5. [ ] Click "Save policy"

### Application Settings:
1. [ ] Click "Settings" tab
2. [ ] Session duration: 24 hours
3. [ ] HTTP-Only Cookies: Toggle ON
4. [ ] Click "Save application"

### Verify:
- [ ] Portal application appears in application list
- [ ] CNAME record created (shown in details)

---

## Step 3: Create Service Applications (9 total)

**Time**: 45 minutes (5 minutes each)

> Automation option: Run `scripts/create-cloudflare-access-apps.sh` with an API token that has **Access: Apps** write permissions to create or refresh all 10 applications programmatically.
```bash
# Example (replace with your values)
CLOUDFLARE_API_TOKEN=cf_api_token_here \
CLOUDFLARE_ACCOUNT_ID=0c93c74d5269a228e91d4bf91c547f56 \
./scripts/create-cloudflare-access-apps.sh --base-domain 254carbon.com
# Add --force to reconcile settings if apps already exist
```

### 3.1 Grafana
```
Name: Grafana.254Carbon
Subdomain: grafana
Session Duration: 24 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.2 Superset
```
Name: Superset.254Carbon
Subdomain: superset
Session Duration: 24 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.3 Vault (SENSITIVE - shorter session)
```
Name: Vault.254Carbon
Subdomain: vault
Session Duration: 2 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.4 MinIO
```
Name: MinIO.254Carbon
Subdomain: minio
Session Duration: 8 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.5 DolphinScheduler
```
Name: DolphinScheduler.254Carbon
Subdomain: dolphin
Session Duration: 12 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.6 DataHub
```
Name: DataHub.254Carbon
Subdomain: datahub
Session Duration: 12 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.7 Trino
```
Name: Trino.254Carbon
Subdomain: trino
Session Duration: 8 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.8 Doris
```
Name: Doris.254Carbon
Subdomain: doris
Session Duration: 8 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

### 3.9 LakeFS
```
Name: LakeFS.254Carbon
Subdomain: lakefs
Session Duration: 12 hours
Policy: Allow Everyone
```
- [ ] Created
- [ ] Policy configured
- [ ] CNAME verified

---

## Step 4: Verify All Applications Created

**Time**: 5 minutes

### Verification Steps:

1. [ ] Go to https://dash.cloudflare.com/zero-trust/access/applications
2. [ ] Verify you see all 10 applications:
   - [ ] 254Carbon Portal
   - [ ] Grafana.254Carbon
   - [ ] Superset.254Carbon
   - [ ] Vault.254Carbon
   - [ ] MinIO.254Carbon
   - [ ] DolphinScheduler.254Carbon
   - [ ] DataHub.254Carbon
   - [ ] Trino.254Carbon
   - [ ] Doris.254Carbon
   - [ ] LakeFS.254Carbon

3. [ ] For each application, verify:
   - [ ] Policy is enabled (shows green checkmark)
   - [ ] CNAME record is listed
   - [ ] Session duration is correct

### Expected Application List View:
```
254Carbon Portal       | cloudflareaccess.com | Allow  | ✓
Grafana.254Carbon      | cloudflareaccess.com | Allow  | ✓
Superset.254Carbon     | cloudflareaccess.com | Allow  | ✓
DataHub.254Carbon      | cloudflareaccess.com | Allow  | ✓
Trino.254Carbon        | cloudflareaccess.com | Allow  | ✓
Doris.254Carbon        | cloudflareaccess.com | Allow  | ✓
Vault.254Carbon        | cloudflareaccess.com | Allow  | ✓
MinIO.254Carbon        | cloudflareaccess.com | Allow  | ✓
DolphinScheduler.254Carbon | cloudflareaccess.com | Allow | ✓
LakeFS.254Carbon       | cloudflareaccess.com | Allow  | ✓
```

---

## Step 5: Configure Audit Logging (Optional but Recommended)

**Time**: 5 minutes

### Steps:
1. [ ] Go to https://dash.cloudflare.com/zero-trust/access/logs
2. [ ] Verify logs page loads
3. [ ] Go to Settings → Notifications
4. [ ] Enable email alerts for:
   - [ ] Failed authentications (>3 per hour)
   - [ ] Policy changes
5. [ ] Enter notification email addresses
6. [ ] Save configuration

---

## Step 6: Quick Smoke Test

**Time**: 10 minutes

### Test Portal Access:
1. [ ] Open new **private/incognito** browser window
2. [ ] Navigate to https://254carbon.com
3. [ ] Expected result: Redirects to Cloudflare Access login page

### Test Login Page:
1. [ ] Should show: "Sign in with your email"
2. [ ] Email input field visible
3. [ ] "Send me a code" button visible

### (Optional) Test Email Authentication:
1. [ ] Enter your email address
2. [ ] Click "Send me a code"
3. [ ] Check email for one-time code
4. [ ] Enter code on login page
5. [ ] Should redirect back to portal
6. [ ] Portal should display all 9 service cards

---

## Troubleshooting

### Problem: Portal doesn't redirect to login

**Diagnosis:**
- [ ] Is tunnel connected? Check dashboard
- [ ] Are Access applications created?
- [ ] Is portal application enabled?

**Solution:**
1. Verify tunnel is Connected (green status)
2. Verify portal application exists and is enabled
3. Wait 2-3 minutes for DNS propagation
4. Try incognito/private browser mode
5. Check browser console for errors

### Problem: 254carbon.com won't load at all

**Diagnosis:**
- [ ] Check if domain is configured in Cloudflare
- [ ] Check if tunnel has correct routes

**Solution:**
```bash
# Check if domain resolves
nslookup 254carbon.com

# Check tunnel routes
kubectl get configmap cloudflared-config -n cloudflare-tunnel -o jsonpath='{.data.config\.yaml}' | grep -A 20 "ingress:"
```

### Problem: CNAME records not created

**Diagnosis:**
- [ ] Check Access applications page
- [ ] Look at application details

**Solution:**
1. CNAME records create automatically after app save
2. Wait 5-10 minutes for propagation
3. Try creating application again if missing
4. Verify domain is active in Cloudflare DNS

---

## DNS Verification (Optional)

**Time**: 5 minutes

### Check DNS Records:
```bash
# Test portal DNS
nslookup 254carbon.cloudflareaccess.com

# Test service DNS
nslookup grafana.cloudflareaccess.com
nslookup vault.cloudflareaccess.com
```

### Expected Result:
- All resolve to Cloudflare tunnel endpoint
- CNAME records point to cfargotunnel.com

---

## Completion Checklist

### Phase 2 Sign-Off:
- [ ] Tunnel connected (green status)
- [ ] All 10 applications created
- [ ] All policies configured
- [ ] All CNAME records verified
- [ ] Portal redirects to Cloudflare login
- [ ] Email OTP works (if tested)
- [ ] Audit logging enabled (optional)

### Phase 2 Status:
**READY FOR PHASE 4 TESTING**: YES ✓

---

## Next Steps

After completing Phase 2:

1. **Phase 4 Testing** (2-3 hours)
   - Follow: `k8s/cloudflare/SSO_VALIDATION_GUIDE.md`
   - Run 30+ test procedures
   - Verify security and performance

2. **Monitor Deployment** (Ongoing)
   - Watch tunnel logs: `kubectl logs -n cloudflare-tunnel -f`
   - Review Cloudflare audit logs daily
   - Monitor portal availability

3. **User Rollout** (After Phase 4)
   - Announce SSO to team
   - Provide access instructions
   - Support user onboarding

---

## Documentation References

- **Cloudflare Access Docs**: https://developers.cloudflare.com/cloudflare-one/
- **Application Setup**: https://developers.cloudflare.com/cloudflare-one/applications/
- **Policy Guide**: https://developers.cloudflare.com/cloudflare-one/policies/access/

---

## Important Notes

- **Session Durations**: Cannot be changed after application creation - plan carefully
- **Policies**: Can be updated anytime without service restarts
- **Permissions**: Requires Zero Trust admin access to Cloudflare dashboard
- **Tunnel**: Must remain connected for authentication to work

---

**Time to Complete Phase 2**: 1-2 hours  
**Difficulty Level**: Easy to Medium  
**Cloudflare Experience Needed**: Basic (can follow UI wizard)

---

Generated: October 19, 2025  
Status: READY FOR EXECUTION
