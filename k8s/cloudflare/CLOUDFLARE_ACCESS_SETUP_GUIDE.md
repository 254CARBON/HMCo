# Cloudflare Access Setup Guide - Dashboard Configuration

**Date**: October 20, 2025  
**Team**: qagi  
**Protected Services**: 3 (Vault, MinIO, DolphinScheduler)  
**Status**: Kubernetes ingress rules updated ✅ | Dashboard configuration: PENDING

---

## Overview

This guide provides step-by-step instructions for configuring Cloudflare Access in the Cloudflare Zero Trust dashboard. The Kubernetes ingress rules have already been updated to expect this configuration.

**What's Already Done:**
- ✅ Kubernetes ingress rules updated with auth-url annotations
- ✅ Vault ingress configured for Cloudflare Access
- ✅ MinIO ingress configured for Cloudflare Access
- ✅ DolphinScheduler ingress configured for Cloudflare Access

**What You Need to Do:**
- Configure identity providers (Google, GitHub, Email)
- Create 3 Access applications
- Set up access policies
- Test authentication flows

---

## Phase 1: Configure Identity Providers (15-30 minutes)

### Step 1.1: Access Cloudflare Zero Trust Dashboard

1. Navigate to: https://dash.cloudflare.com/zero-trust
2. Log in with your Cloudflare account
3. Select team: **qagi**
4. Verify you see "Zero Trust" in the left sidebar

### Step 1.2: Set up Google OAuth Provider

1. Go to **Settings** → **Authentication**
2. Click **Add** to add a new provider
3. Select **Google** from the list
4. Follow the OAuth consent screen setup:
   - You'll be directed to Google Cloud Console
   - Create a new OAuth consent screen if you haven't already
   - Create OAuth 2.0 Credentials (Authorized Redirect URI):
     - `https://qagi.cloudflareaccess.com/cdn-cgi/access/callback`
5. In Cloudflare, enter your **Client ID** and **Client Secret**
6. Configure settings:
   - Session Duration: **24 hours**
   - Save and enable

### Step 1.3: Set up GitHub OAuth Provider

1. Go to **Settings** → **Authentication**
2. Click **Add** to add a new provider
3. Select **GitHub** from the list
4. Create GitHub OAuth App:
   - Go to: https://github.com/settings/developers
   - Click **OAuth Apps** → **New OAuth App**
   - Fill in:
     - Application name: `254Carbon Access`
     - Homepage URL: `https://254carbon.com`
     - Authorization callback URL: `https://qagi.cloudflareaccess.com/cdn-cgi/access/callback`
   - Generate **Client ID** and **Client Secret**
5. In Cloudflare, enter your GitHub **Client ID** and **Client Secret**
6. Configure settings:
   - Session Duration: **24 hours**
   - Save and enable

### Step 1.4: Set up Email/OTP Provider

1. Go to **Settings** → **Authentication**
2. Click **Add** to add a new provider
3. Select **One-time PIN** (Email OTP)
4. Configure settings:
   - Allow one-time PIN authentication: **Yes**
   - Session Duration: **24 hours**
   - Save and enable

**Result**: You should now have 3 authentication methods available

---

## Phase 2: Create Access Applications (20-30 minutes)

### Step 2.1: Create Vault Access Application

1. Go to **Access** → **Applications**
2. Click **Create Application**
3. Select **Self-hosted** application type

**Application Details:**
- Application name: `Vault - Secrets Management`
- Subdomain: `vault`
- Domain: `254carbon.com`
- Session duration: `24 hours`
- Application logo (optional): 🔐
- Auto-launch: Off

4. Click **Next** to configure policies

**Add Policy 1: Allow @254carbon.com Domain**

1. Policy name: `254carbon.com Domain`
2. Action: **Allow**
3. Click **Add rule**
4. Configure rule:
   - Include: Select **Email**
   - In the field, enter: `@254carbon.com`
5. Click **Save rule**
6. Click **Add another policy** if not finished

**Add Policy 2: Allow @project52.org Domain**

1. Policy name: `project52.org Domain`
2. Action: **Allow**
3. Click **Add rule**
4. Configure rule:
   - Include: Select **Email**
   - In the field, enter: `@project52.org`
5. Click **Save rule**
6. Click **Add another policy**

**Add Policy 3: Block Everyone Else**

1. Policy name: `Block All Others`
2. Action: **Block**
3. Click **Add rule**
4. Configure rule:
   - Include: Select **Everyone**
5. Click **Save rule**
6. Click **Save application**

**Result**: Vault Access application created with 3 policies

---

### Step 2.2: Create MinIO Access Application

Repeat the same process as Vault with these changes:

**Application Details:**
- Application name: `MinIO - Object Storage Console`
- Subdomain: `minio`
- Domain: `254carbon.com`
- Everything else: same as Vault

**Policies**: Use the same 3 policies as Vault:
1. Allow @254carbon.com
2. Allow @project52.org
3. Block everyone else

---

### Step 2.3: Create DolphinScheduler Access Application

Repeat the same process as Vault with these changes:

**Application Details:**
- Application name: `DolphinScheduler - Workflow Orchestration`
- Subdomain: `dolphin`
- Domain: `254carbon.com`
- Everything else: same as Vault

**Policies**: Use the same 3 policies as Vault:
1. Allow @254carbon.com
2. Allow @project52.org
3. Block everyone else

---

## Phase 3: Verify Kubernetes Configuration

The Kubernetes ingress rules have already been configured. Verify with:

```bash
# Check Vault ingress has auth-url annotation
kubectl get ingress vault-ingress -n data-platform -o yaml | grep auth-url
# Should show: nginx.ingress.kubernetes.io/auth-url: https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize

# Check MinIO ingress has auth-url annotation
kubectl get ingress minio-console-ingress -n data-platform -o yaml | grep auth-url

# Check DolphinScheduler ingress has auth-url annotation
kubectl get ingress dolphinscheduler-ingress -n data-platform -o yaml | grep auth-url
```

---

## Phase 4: Testing & Verification (15-20 minutes)

### Test 4.1: Test Unauthenticated Access

**Expected Result**: Should redirect to Cloudflare Access login page

```bash
# Test Vault
curl -I https://vault.254carbon.com
# Should return 302 with Location header pointing to qagi.cloudflareaccess.com

# Test MinIO
curl -I https://minio.254carbon.com

# Test DolphinScheduler
curl -I https://dolphin.254carbon.com
```

### Test 4.2: Manual Browser Testing

1. Open browser to: https://vault.254carbon.com
2. You should see Cloudflare Access login page
3. Choose authentication method:
   - **Google** - Log in with your Google account
   - **GitHub** - Log in with your GitHub account
   - **One-time PIN** - Enter your email and click "Send me a code"
4. You should be redirected back to vault.254carbon.com
5. Vault UI should display (may require internal auth)

### Test 4.3: Test with Authorized Email (@254carbon.com)

If you have access to a @254carbon.com email:
1. Use email-based OTP authentication
2. Should successfully authenticate
3. Should see service UI

### Test 4.4: Test with Authorized Email (@project52.org)

If you have access to a @project52.org email:
1. Use email-based OTP authentication
2. Should successfully authenticate
3. Should see service UI

### Test 4.5: Test with Unauthorized Email

Use an email NOT from @254carbon.com or @project52.org (e.g., gmail.com):
1. Try to authenticate
2. Should be denied with "Access Denied" message
3. Should NOT see service UI

### Test 4.6: Verify Public Services Still Work

Public services should NOT require authentication:

```bash
# These should work WITHOUT redirecting to login
curl -I https://grafana.254carbon.com      # Should return 302
curl -I https://superset.254carbon.com     # Should return 302
curl -I https://datahub.254carbon.com      # Should return 302
```

---

## Success Criteria Checklist

After completing all phases, verify:

- [ ] Google OAuth provider configured in Zero Trust
- [ ] GitHub OAuth provider configured in Zero Trust
- [ ] Email/OTP provider configured in Zero Trust
- [ ] Vault Access application created with 3 policies
- [ ] MinIO Access application created with 3 policies
- [ ] DolphinScheduler Access application created with 3 policies
- [ ] Kubernetes ingress rules deployed (vault, minio, dolphin)
- [ ] Unauthenticated access redirects to login page
- [ ] @254carbon.com users can authenticate
- [ ] @project52.org users can authenticate
- [ ] Other email domains are denied
- [ ] Public services (Grafana, Superset) still accessible without auth
- [ ] All 3 protected services show correct UI after authentication
- [ ] Session duration is 24 hours (verify by staying logged in)

---

## Troubleshooting

### Issue: "Bad Gateway" Error

**Cause**: Ingress configuration issue

**Solution**:
```bash
# Check NGINX logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx | tail -30

# Verify auth-url endpoint is reachable
curl -I https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize

# Restart ingress controller if needed
kubectl rollout restart deployment/nginx-ingress-controller -n ingress-nginx
```

### Issue: "Access Denied" for @254carbon.com Users

**Cause**: Policy not configured correctly

**Solution**:
1. Go to Zero Trust → Applications → [Service]
2. Check that policy has:
   - Action: **Allow**
   - Include: **Email**
   - Value: **@254carbon.com** (NOT @example.com)
3. Verify policy order (Allow policies before Deny policy)
4. Save changes

### Issue: Session Expires Too Quickly

**Cause**: Session duration not set to 24 hours

**Solution**:
1. Go to Zero Trust → Applications → [Service]
2. Edit application settings
3. Set Session Duration to **24 hours**
4. Save

### Issue: Redirect Loop

**Cause**: NGINX configuration issue

**Solution**:
1. Verify auth-url is exactly: `https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize`
2. Verify auth-signin is exactly: `https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$scheme://$host$request_uri`
3. Clear browser cookies for vault.254carbon.com
4. Try again

---

## Files Modified

- ✅ `k8s/ingress/ingress-rules.yaml` - Updated 3 ingress resources with auth annotations

---

## Next Steps After Implementation

1. **Monitor Access Events**
   - Go to Zero Trust → Logs → Access
   - Watch for authentication events

2. **Configure User Groups** (Optional)
   - Set up Teams/Groups in Cloudflare for fine-grained access control

3. **Enable Audit Logging**
   - Enable detailed audit logs in each Access application

4. **Set up Alerts** (Optional)
   - Configure alerts for failed authentication attempts

---

## Additional Resources

- Cloudflare Zero Trust Dashboard: https://dash.cloudflare.com/zero-trust
- Cloudflare Access Documentation: https://developers.cloudflare.com/cloudflare-one/identity/
- NGINX Ingress Auth Documentation: https://kubernetes.github.io/ingress-nginx/user-guide/nginx-configuration/annotations/#external-authentication
