# Cloudflare Access Implementation Plan - Comprehensive Guide

**Status**: Plan Mode - Ready for Implementation  
**Date**: October 20, 2025  
**Protected Services**: 3 (Vault, MinIO, DolphinScheduler)  
**Identity Providers**: Multiple (Google OAuth, GitHub OAuth, Email OTP)  
**Authorized Domains**: @254carbon.com, @project52.org

---

## Overview

This plan implements Cloudflare Access (Zero Trust) authentication for three sensitive services while allowing users from two email domains to access them. All authenticated users from authorized domains get access to all three services with 24-hour sessions.

## Current State

- ✅ Cloudflare Tunnel deployed and operational
- ✅ DNS records configured for all services
- ✅ Ingress rules cleaned and standardized
- ⏳ Cloudflare Access configuration: NOT YET IMPLEMENTED

## Implementation Scope

### Protected Services (Require Authentication)
1. **Vault** (vault.254carbon.com) - Secrets management
2. **MinIO** (minio.254carbon.com) - Object storage console
3. **DolphinScheduler** (dolphin.254carbon.com) - Workflow orchestration

### Identity Providers to Configure
1. **Google OAuth** - For users with Google accounts
2. **GitHub OAuth** - For developer access
3. **Email/OTP** - Fallback for other users

### Access Rules
- **Allow**: Any user with @254carbon.com email OR @project52.org email
- **Deny**: All others
- **Session**: 24 hours default
- **MFA**: Optional enrollment (not required)

---

## Phase 1: Cloudflare Zero Trust Dashboard Setup

### 1.1 Access Zero Trust Dashboard
1. Log in to Cloudflare Dashboard: https://dash.cloudflare.com
2. Navigate to **Zero Trust** in left sidebar
3. Or directly: https://dash.cloudflare.com/zero-trust
4. Team name should show: **qagi**

### 1.2 Configure Identity Providers

#### Step A: Set up Google OAuth
1. Go to **Settings** → **Authentication**
2. Click **Add** for Google Workspace/Gmail
3. Select **Google** as provider
4. Follow OAuth setup wizard (requires Google Cloud Console access)
5. Configure:
   - Allowed domains: @gmail.com (optional), or all Google accounts
   - Session duration: 24 hours
6. Save and note the Provider ID

#### Step B: Set up GitHub OAuth
1. In **Settings** → **Authentication**, click **Add**
2. Select **GitHub**
3. Follow GitHub OAuth setup:
   - Create GitHub OAuth App: https://github.com/settings/developers
   - App name: "254Carbon Access"
   - Homepage: https://254carbon.com
   - Authorization callback: https://qagi.cloudflareaccess.com/cdn-cgi/access/callback
4. Copy Client ID and Secret into Cloudflare
5. Configure:
   - Allowed organizations: (optional)
   - Session duration: 24 hours
6. Save and note the Provider ID

#### Step C: Set up Email/OTP (One-Time Passcode)
1. In **Settings** → **Authentication**, click **Add**
2. Select **One-time PIN**
3. Configure:
   - Email OTP enabled: Yes
   - Session duration: 24 hours
4. This allows any email address to use OTP (no external provider needed)

---

## Phase 2: Create Access Applications

### 2.1 Create Vault Application

**Location**: Zero Trust Dashboard → **Access** → **Applications**

**Application Configuration:**
```
Name:                    Vault - Secrets Management
Subdomain:               vault
Domain:                  254carbon.com
Session Duration:        24 hours
Auto-launch:            No
CORS Headers:           Enabled
```

**Policy 1: Allow @254carbon.com Email**
```
Policy Name:             "254carbon.com Domain Access"
Action:                  Allow
Rule Type:               Email
Target:                  Emails ending with @254carbon.com
Require:                 (leave empty for OR logic)
```

**Policy 2: Allow @project52.org Email**
```
Policy Name:             "project52.org Domain Access"
Action:                  Allow
Rule Type:               Email
Target:                  Emails ending with @project52.org
Require:                 (leave empty for OR logic)
```

**Policy 3: Deny All Others**
```
Policy Name:             "Block Everyone Else"
Action:                  Block
Rule Type:               Everyone
```

### 2.2 Create MinIO Application

**Application Configuration:**
```
Name:                    MinIO Console - Object Storage
Subdomain:               minio
Domain:                  254carbon.com
Session Duration:        24 hours
Auto-launch:            No
CORS Headers:           Enabled
```

**Policies:** Same as Vault (2 allow policies + 1 deny)

### 2.3 Create DolphinScheduler Application

**Application Configuration:**
```
Name:                    DolphinScheduler - Workflow Orchestration
Subdomain:               dolphin
Domain:                  254carbon.com
Session Duration:        24 hours
Auto-launch:            No
CORS Headers:           Enabled
```

**Policies:** Same as Vault (2 allow policies + 1 deny)

---

## Phase 3: Configure Ingress Rules for Access

### 3.1 Update Vault Ingress
**File**: `k8s/ingress/ingress-rules.yaml`

Update the Vault ingress section with Cloudflare Access configuration:

```yaml
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vault-ingress
  namespace: data-platform
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    # Cloudflare Access Configuration
    nginx.ingress.kubernetes.io/auth-url: "https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$scheme://$host$request_uri"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      auth_request_set $cf_access_user $upstream_http_cf_access_authenticated_user_email;
      auth_request_set $cf_access_groups $upstream_http_cf_access_groups;
      proxy_set_header CF-Access-Authenticated-User-Email $cf_access_user;
      proxy_set_header X-FORWARDED-USER $cf_access_user;
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - vault.254carbon.com
    - vault.local
    secretName: vault-tls
  rules:
  - host: vault.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vault
            port:
              number: 8200
  - host: vault.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vault
            port:
              number: 8200
```

### 3.2 Update MinIO Ingress

Apply same pattern as Vault to `minio-console-ingress`

### 3.3 Update DolphinScheduler Ingress

Apply same pattern as Vault to `dolphinscheduler-ingress`

### 3.4 Keep Public Services Unchanged

Grafana, Superset, DataHub, Trino, Doris, etc. should NOT have auth-url annotations.

---

## Phase 4: Deploy Configuration Changes

### 4.1 Apply Updated Ingress Rules
```bash
kubectl apply -f k8s/ingress/ingress-rules.yaml
```

### 4.2 Verify Ingress Deployment
```bash
kubectl get ingress -n data-platform
# Should show vault-ingress, minio-console-ingress, dolphinscheduler-ingress
```

### 4.3 Check Ingress Annotations
```bash
kubectl get ingress vault-ingress -n data-platform -o yaml | grep auth-url
# Should show Cloudflare Access URL
```

---

## Phase 5: Testing & Verification

### 5.1 Test Unauthenticated Access

**Expected Behavior**: Redirected to Cloudflare Access login

```bash
# Should return 302 redirect to login page
curl -I https://vault.254carbon.com
```

**Response Should Include:**
```
Location: https://qagi.cloudflareaccess.com/cdn-cgi/access/login?...
```

### 5.2 Test Authentication Flow

1. Open browser and navigate to: https://vault.254carbon.com
2. You should see Cloudflare Access login page
3. Choose authentication method:
   - **Google** - Authenticate with Google account
   - **GitHub** - Authenticate with GitHub account
   - **Email** - Enter email and receive OTP via email
4. After authentication, you should see Vault UI

### 5.3 Test Email Domain Restrictions

**Authorized Emails** (Should be allowed):
- user@254carbon.com ✅
- developer@project52.org ✅
- admin@254carbon.com ✅
- anyone@project52.org ✅

**Unauthorized Emails** (Should be blocked):
- user@gmail.com ❌
- admin@example.com ❌
- user@other-domain.org ❌

### 5.4 Test All Three Protected Services

```bash
# Test each service
curl -I https://vault.254carbon.com
curl -I https://minio.254carbon.com
curl -I https://dolphin.254carbon.com

# All should redirect to login page
```

### 5.5 Verify Public Services Still Work

```bash
# Public services should NOT redirect to login
curl -I https://grafana.254carbon.com  # Should return 302 (service redirect)
curl -I https://superset.254carbon.com  # Should return 302 (service redirect)
```

---

## Phase 6: Configuration Persistence

### 6.1 Document Applied Settings

Create configuration record file:
```bash
cat > k8s/cloudflare/CLOUDFLARE_ACCESS_APPLIED.md << 'EOF'
# Cloudflare Access Configuration - Applied Settings

## Protected Services
- Vault (vault.254carbon.com)
- MinIO (minio.254carbon.com)
- DolphinScheduler (dolphin.254carbon.com)

## Identity Providers
- Google OAuth
- GitHub OAuth
- Email/OTP

## Access Policy
- Allow: *@254carbon.com
- Allow: *@project52.org
- Deny: All others

## Session Duration
- 24 hours default

## Ingress Updates
- Updated Vault ingress with auth-url
- Updated MinIO ingress with auth-url
- Updated DolphinScheduler ingress with auth-url

Date Applied: [TODAY]
EOF
```

### 6.2 Update Documentation

- Update k8s/cloudflare/README.md with Access configuration
- Update SECURITY_POLICIES.md with implementation notes
- Add troubleshooting section for common issues

---

## Troubleshooting Guide

### Issue 1: "Access Denied" for Authorized Users

**Cause**: User's email domain not matching policy

**Solution**:
1. Verify user email ends with @254carbon.com or @project52.org
2. Check Cloudflare Access policies in Zero Trust dashboard
3. Ensure policies are in correct order (Allow before Deny)

### Issue 2: "Bad Gateway" Error

**Cause**: Ingress annotation misconfiguration

**Solution**:
```bash
# Verify annotation is correct
kubectl get ingress vault-ingress -n data-platform -o yaml

# Check NGINX logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx | tail -20

# Verify auth-url endpoint is reachable
curl -I https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize
```

### Issue 3: Session Timeout Too Soon

**Cause**: Session duration setting

**Solution**:
1. Go to Zero Trust → Access → Applications
2. Edit vault application
3. Increase "Session Duration" to 24 hours

### Issue 4: Users Can't See Logout Button

**Cause**: NGINX proxy headers not passing through

**Solution**:
Check if `CF-Access-Authenticated-User-Email` header is passed in response

---

## Success Criteria

✅ Protected services redirect to Cloudflare Access login  
✅ Authorized users (@254carbon.com, @project52.org) can authenticate  
✅ Unauthorized users are denied access  
✅ Google OAuth works  
✅ GitHub OAuth works  
✅ Email OTP works  
✅ Public services remain accessible without authentication  
✅ Session duration is 24 hours  
✅ All three services share same authentication  

---

## Files to Modify/Create

| File | Change | Status |
|------|--------|--------|
| k8s/ingress/ingress-rules.yaml | Add auth-url to 3 ingresses | PENDING |
| k8s/cloudflare/README.md | Document Access config | PENDING |
| k8s/cloudflare/SECURITY_POLICIES.md | Update with implementation | PENDING |
| k8s/cloudflare/CLOUDFLARE_ACCESS_APPLIED.md | Create record of applied settings | PENDING |

---

## Manual Steps Required in Cloudflare Dashboard

These cannot be automated and must be done manually:

1. ✅ Configure Google OAuth provider
2. ✅ Configure GitHub OAuth provider
3. ✅ Enable Email/OTP provider
4. ✅ Create Vault Access application
5. ✅ Create MinIO Access application
6. ✅ Create DolphinScheduler Access application
7. ✅ Add 2 allow policies + 1 deny policy to each application
8. ✅ Test each application

---

## Estimated Implementation Time

- Dashboard setup: 15-30 minutes
- Create applications: 20-30 minutes
- Update ingress rules: 5 minutes
- Testing & verification: 15-20 minutes
- **Total: 60-90 minutes (~1.5 hours)**

---

## Ready for Implementation?

When ready to proceed, this plan will:

1. Create/update Kubernetes ingress resources with Cloudflare Access annotations
2. Create documentation and troubleshooting guides
3. Provide verification scripts and testing procedures
4. Set up monitoring for Access events

**Current Status**: ✅ PLAN COMPLETE - Ready to implement

---

## Next Steps (After Plan Confirmation)

1. Confirm identity provider preferences in Cloudflare dashboard
2. Update ingress YAML files
3. Deploy updated ingress rules to cluster
4. Test authentication flows
5. Document final configuration
