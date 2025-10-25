# SSO Implementation: Phase 2-4 Complete Guide (Canonical)

**Status**: Phase 1 Complete ✅ | Ready to Begin Phase 2  
**Date**: October 19, 2025  
**Timeline**: 4-7 days to complete all phases

---

## Overview

This guide provides step-by-step instructions to complete the SSO implementation:
- **Phase 2**: Configure Cloudflare Access authentication (1-2 hours)
- **Phase 3**: Disable local authentication in services (2-3 days)
- **Phase 4**: Test and validate the complete SSO system (1-2 days)

For OIDC service-level integration details (Grafana, Superset, DataHub, ArgoCD) see `docs/sso/oidc-apps.md`.

Phase 1 (Portal deployment) is **COMPLETE** and already deployed.

---

## Phase 2: Cloudflare Access Configuration

### Phase 2.1: Prerequisites Checklist

Before starting Phase 2, verify:

- [ ] Cloudflare account with Teams/Enterprise subscription active
- [ ] 254carbon.com domain configured in Cloudflare
- [ ] Cloudflare Tunnel (cloudflared) deployed and running
  ```bash
  kubectl get pods -n cloudflare-tunnel
  # Should show 2 running pods
  ```
- [ ] Portal deployed and accessible
  ```bash
  kubectl get pods -n data-platform -l app=portal
  # Should show 2 running pods
  ```

### Phase 2.2: Enable Cloudflare Teams / Zero Trust

**Time Required: 5 minutes**

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Go to **My Profile** → **Billing**
3. Verify subscription plan includes **Cloudflare Teams** or **Enterprise**
4. Navigate to **Zero Trust** → **Dashboard**
5. Accept Zero Trust terms and conditions
6. You should now see the Zero Trust dashboard with options for Access, Gateway, etc.

**Verification:**
```
Zero Trust Dashboard shows:
- Access section with Applications submenu
- Authentication settings
- Audit logs section
```

### Phase 2.3: Create Portal Application in Cloudflare Access

**Time Required: 15 minutes**

#### Step 1: Add Application

1. Go to **Zero Trust** → **Access** → **Applications**
2. Click **Add an application** button
3. Select **Self-hosted**
4. Fill in application details:
   - **Application name**: `254Carbon Portal`
   - **Subdomain**: `254carbon`
   - **Domain**: `cloudflareaccess.com` (dropdown)
   - **Application type**: `Web`
5. Click **Next**

#### Step 2: Configure Policies

1. Under **Policies** section, click **+ Add a policy**

**Policy 1: Allow All Authenticated Users**
- **Policy name**: `Allow Portal Access`
- **Decision**: `Allow`
- **Include**: `Everyone`
- Click **Save**

**Policy 2: Implicit Deny** (automatically applied if no other policy matches)

#### Step 3: Configure Settings

1. Click **Settings** tab
2. Configure:
   - **Session Duration**: `24 hours`
   - **HTTP-Only Cookies**: `Enable` (toggle on)
   - **Auto-redirect**: `Disable`
3. Click **Save application**

**Result:**
- Portal now protected by Cloudflare Access at `https://254carbon.cloudflareaccess.com`
- Automatically creates CNAME pointing to your Cloudflare Tunnel

### Phase 2.4: Create Service Applications (9 Services)

**Time Required: 45 minutes (5 min per service × 9)**

Create the following applications using the same process as Phase 2.3:

#### Service 1: Vault

- **Application name**: `Vault.254Carbon`
- **Subdomain**: `vault`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `2 hours` (sensitive service - short session)

#### Service 2: MinIO

- **Application name**: `MinIO.254Carbon`
- **Subdomain**: `minio`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `8 hours`

#### Service 3: DolphinScheduler

- **Application name**: `DolphinScheduler.254Carbon`
- **Subdomain**: `dolphin`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `12 hours`

#### Service 4: Grafana

- **Application name**: `Grafana.254Carbon`
- **Subdomain**: `grafana`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `24 hours`

#### Service 5: Superset

- **Application name**: `Superset.254Carbon`
- **Subdomain**: `superset`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `24 hours`

#### Service 6: DataHub

- **Application name**: `DataHub.254Carbon`
- **Subdomain**: `datahub`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `12 hours`

#### Service 7: Trino

- **Application name**: `Trino.254Carbon`
- **Subdomain**: `trino`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `8 hours`

#### Service 8: ClickHouse

- **Application name**: `ClickHouse.254Carbon`
- **Subdomain**: `clickhouse`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `8 hours`

#### Service 9: LakeFS

- **Application name**: `LakeFS.254Carbon`
- **Subdomain**: `lakefs`
- **Domain**: `cloudflareaccess.com`
- **Policy**: Allow → Everyone
- **Session Duration**: `12 hours`

### Phase 2.5: Enable Audit Logging

**Time Required: 10 minutes**

1. Go to **Zero Trust** → **Access** → **Logs**
2. Verify you can see access attempts
3. Configure alerts:
   - Go to **Settings** → **Notifications**
   - Enable email alerts for:
     - Failed authentication (>3 failures per hour)
     - Policy changes
     - New applications added

### Phase 2.6: Verify DNS Records

**Time Required: 5 minutes**

All Cloudflare Access applications automatically create CNAME records.

Verify in Cloudflare DNS dashboard:

```bash
# Test DNS resolution for portal and services
nslookup 254carbon.cloudflareaccess.com
nslookup vault.cloudflareaccess.com
nslookup grafana.cloudflareaccess.com

# All should resolve to your Cloudflare Tunnel endpoint
# Example output: 254carbon.cloudflareaccess.com canonical name = YOUR_TUNNEL_ID.cfargotunnel.com
```

### Phase 2.7: Test Portal Access

**Time Required: 5 minutes**

1. Open https://254carbon.com in browser
2. Should redirect to Cloudflare Access login page
3. Enter your email address
4. Check email for one-time code
5. Enter one-time code
6. Should now see 254Carbon Portal with all 9 service cards

**Success Criteria:**
- ✅ Portal redirects to Cloudflare Access
- ✅ Email authentication works
- ✅ Portal displays correctly after login
- ✅ Session persists

---

## Phase 3: Service Integration

### Phase 3.1: Disable Grafana Local Authentication

**Time Required: 10 minutes**

```bash
# Step 1: Patch Grafana configmap
kubectl -n monitoring patch configmap grafana-config --type merge -p '{
  "data": {
    "grafana.ini": "[auth.anonymous]\nenabled = false\n[users]\nauto_assign_org_role = Viewer"
  }
}'

# Step 2: Restart Grafana
kubectl rollout restart deployment/grafana -n monitoring

# Step 3: Verify restart
kubectl rollout status deployment/grafana -n monitoring
# Should show: Waiting for deployment "grafana" rollout to finish: 1 old replicas, 1 new replicas...

# Wait for pods to be ready
kubectl get pods -n monitoring -l app=grafana -w
# Press Ctrl+C when both pods are Running
```

### Phase 3.2: Disable Superset Local Authentication

**Time Required: 10 minutes**

```bash
# Step 1: Update Superset environment
kubectl -n data-platform set env deployment/superset \
  SUPERSET_DISABLE_LOCAL_AUTH=true \
  SUPERSET_WEBSERVER_TIMEOUT=120

# Step 2: Restart Superset
kubectl rollout restart deployment/superset -n data-platform

# Step 3: Verify
kubectl rollout status deployment/superset -n data-platform
kubectl get pods -n data-platform -l app=superset -w
```

### Phase 3.3: Configure NGINX Ingress Rules with Cloudflare Authentication

**Time Required: 30 minutes**

For services that should use Cloudflare Access authentication, update ingress rules.

#### Update All Service Ingress Rules

Create or update `k8s/ingress/ingress-cloudflare-auth.yaml`:

```yaml
---
# Vault Ingress with Cloudflare Access Authentication
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vault-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    # Cloudflare Access Authentication
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - vault.254carbon.com
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

---
# Grafana Ingress with Cloudflare Access Authentication
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grafana-ingress
  namespace: monitoring
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - grafana.254carbon.com
    secretName: grafana-tls
  rules:
  - host: grafana.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000

---
# Superset Ingress with Cloudflare Access Authentication
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: superset-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - superset.254carbon.com
    secretName: superset-tls
  rules:
  - host: superset.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: superset
            port:
              number: 8088

---
# MinIO Ingress with Cloudflare Access Authentication
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: minio-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - minio.254carbon.com
    secretName: minio-tls
  rules:
  - host: minio.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: minio
            port:
              number: 9001

---
# DolphinScheduler Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dolphin-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - dolphin.254carbon.com
    secretName: dolphin-tls
  rules:
  - host: dolphin.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dolphinscheduler
            port:
              number: 12345

---
# DataHub Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: datahub-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - datahub.254carbon.com
    secretName: datahub-tls
  rules:
  - host: datahub.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: datahub-frontend
            port:
              number: 3000

---
# Trino Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trino-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - trino.254carbon.com
    secretName: trino-tls
  rules:
  - host: trino.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: trino
            port:
              number: 8080

---
# ClickHouse Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: clickhouse-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - clickhouse.254carbon.com
    secretName: clickhouse-tls
  rules:
  - host: clickhouse.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: clickhouse
            port:
              number: 8123

---
# LakeFS Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lakefs-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-url: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<ACCOUNT_ID>.cloudflareaccess.com/cdn-cgi/access/login"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
spec:
  tls:
  - hosts:
    - lakefs.254carbon.com
    secretName: lakefs-tls
  rules:
  - host: lakefs.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: lakefs
            port:
              number: 8000
```

#### Apply Ingress Rules

**IMPORTANT**: Replace `<ACCOUNT_ID>` with your actual Cloudflare Account ID. Find it in:
- Cloudflare Dashboard → Zero Trust → Settings → Account

```bash
# First, get your Cloudflare Account ID
# Go to: https://dash.cloudflare.com/zero-trust/settings/general
# Copy "Account ID" (looks like: 1234567890abcdef)

# Edit the file and replace <ACCOUNT_ID>
sed -i 's/<ACCOUNT_ID>/YOUR_ACCOUNT_ID_HERE/g' k8s/ingress/ingress-cloudflare-auth.yaml

# Apply the ingress rules
kubectl apply -f k8s/ingress/ingress-cloudflare-auth.yaml

# Verify all ingress rules created
kubectl get ingress -A | grep 254carbon
```

### Phase 3.4: Verify Services are Running

**Time Required: 5 minutes**

```bash
# Check all required services are running
kubectl get pods -n monitoring -l app=grafana
kubectl get pods -n data-platform -l app=superset
kubectl get pods -n data-platform -l app=vault
kubectl get pods -n data-platform -l app=minio
kubectl get pods -n data-platform -l app=dolphinscheduler
kubectl get pods -n data-platform -l app=datahub
kubectl get pods -n data-platform -l app=trino
kubectl get pods -n data-platform -l app=clickhouse
kubectl get pods -n data-platform -l app=lakefs

# All should show Running status
```

---

## Phase 4: Testing & Validation

### Phase 4.1: End-to-End Authentication Flow Test

**Time Required: 20 minutes**

#### Test 1: Portal Access

```bash
# Test 1.1: Portal redirects to Cloudflare Access
curl -v https://254carbon.com 2>&1 | grep -i location
# Should see location header pointing to Cloudflare Access login

# Test 1.2: Open browser and test manually
# 1. Open https://254carbon.com
# 2. Should redirect to login page
# 3. Enter email and receive one-time code
# 4. Enter code and see portal with 9 service cards
```

#### Test 2: Service Access

```bash
# After logging in at portal, visit each service:
# (Note: You should remain logged in for all - no re-authentication)

https://grafana.254carbon.com       # Should show Grafana dashboards
https://superset.254carbon.com      # Should show Superset UI
https://vault.254carbon.com         # Should show Vault UI
https://minio.254carbon.com         # Should show MinIO console
https://dolphin.254carbon.com       # Should show DolphinScheduler
https://datahub.254carbon.com       # Should show DataHub
https://trino.254carbon.com         # Should show Trino UI
https://clickhouse.254carbon.com   # Should show ClickHouse interface
https://lakefs.254carbon.com        # Should show LakeFS UI
```

#### Test 3: Session Persistence

```bash
# Verify session persists across services without re-authentication
# Open browser developer tools (F12)
# 1. Visit https://254carbon.com → Authenticate
# 2. Visit https://vault.254carbon.com → Check cookies
#    Should have CF-Access-JWT-Assertion cookie
# 3. Visit https://grafana.254carbon.com → No login required
# 4. Check tab "Applications" shows all 9 services loaded
```

### Phase 4.2: Audit Log Verification

**Time Required: 10 minutes**

1. Go to **Zero Trust** → **Access** → **Logs**
2. Should see entries like:
   ```
   254carbon.cloudflareaccess.com - User authenticated
   grafana.cloudflareaccess.com - User accessed service
   vault.cloudflareaccess.com - User accessed service
   ...
   ```
3. Each service access should show:
   - User email
   - Service name
   - Authentication success/failure
   - Timestamp

### Phase 4.3: Performance Testing

**Time Required: 15 minutes**

```bash
# Test 1: Portal response time
time curl https://254carbon.com -o /dev/null -s
# Expected: <1 second (typically 100-500ms)

# Test 2: Service response time
time curl -H "CF-Access-JWT-Assertion: <token>" https://vault.254carbon.com -o /dev/null -s
# Expected: <1 second

# Test 3: Load testing (requires ab or Apache Bench)
ab -n 100 -c 10 https://254carbon.com
# Should see: Requests/sec (should be >100)

# Test 4: Concurrent connections
ab -n 500 -c 50 https://254carbon.com
# Should handle without errors
```

### Phase 4.4: Security Testing Checklist

**Time Required: 20 minutes**

- [ ] **Test 1: Access without authentication**
  ```bash
  # Should redirect to login, not grant access
  curl -v https://vault.254carbon.com 2>&1 | grep -i "location\|403\|401"
  ```

- [ ] **Test 2: Invalid token rejection**
  ```bash
  curl -v -H "CF-Access-JWT-Assertion: invalid_token" https://vault.254carbon.com
  # Should return 401 Unauthorized
  ```

- [ ] **Test 3: Session timeout**
  - Login at https://254carbon.com
  - Wait (session expires after configured duration)
  - Try accessing https://vault.254carbon.com
  - Should redirect to login

- [ ] **Test 4: Logout functionality**
  - Login at portal
  - Click logout (when implemented)
  - Try accessing service
  - Should require re-authentication

- [ ] **Test 5: HTTPS only**
  ```bash
  # Should not accept HTTP
  curl -v http://254carbon.com 2>&1 | grep -i "301\|302"
  # Should see redirect to HTTPS
  ```

- [ ] **Test 6: Rate limiting**
  ```bash
  # Test rate limiting on authentication endpoint
  for i in {1..100}; do curl -X POST https://254carbon.com/auth; done
  # Should see rate limit response after N attempts
  ```

- [ ] **Test 7: Authorization by policy**
  - Verify Cloudflare Access policy blocks unauthorized users
  - Go to **Zero Trust** → **Access** → **Applications**
  - Verify each application has correct policy configured

### Phase 4.5: Service-Specific Testing

**Time Required: 20 minutes**

#### Grafana

```bash
# After SSO login at portal:
# 1. Visit https://grafana.254carbon.com
# 2. Should show Grafana UI (not login page)
# 3. Verify dashboards are accessible
# 4. Check Grafana logs for user info
kubectl logs -n monitoring -l app=grafana | grep -i user
```

#### Superset

```bash
# After SSO login:
# 1. Visit https://superset.254carbon.com
# 2. Should show Superset UI
# 3. Verify dashboards are accessible
# 4. Try creating new chart
kubectl logs -n data-platform -l app=superset | grep -i user
```

#### Vault

```bash
# After SSO login:
# 1. Visit https://vault.254carbon.com/ui
# 2. Should show Vault UI (not auth method selection)
# 3. Try reading a secret:
VAULT_TOKEN=$(kubectl get secret -n data-platform vault-token -o jsonpath='{.data.token}' | base64 -d)
curl -H "X-Vault-Token: $VAULT_TOKEN" https://vault.254carbon.com/v1/secret/data/myapp
```

### Phase 4.6: Complete Testing Checklist

- [ ] Portal accessible at https://254carbon.com
- [ ] Portal redirects to Cloudflare Access login
- [ ] Email authentication works (receives one-time code)
- [ ] After authentication, portal displays 9 service cards
- [ ] All service cards have correct icons and descriptions
- [ ] Clicking service link redirects to correct service
- [ ] Service page loads without requiring re-authentication
- [ ] Session persists across all services
- [ ] Can visit all 9 services: Grafana, Superset, Vault, MinIO, DolphinScheduler, DataHub, Trino, ClickHouse, LakeFS
- [ ] Response times are <100ms
- [ ] Audit logs show all access attempts
- [ ] Unauthorized access is denied (401/403)
- [ ] Invalid tokens are rejected
- [ ] Rate limiting is working
- [ ] HTTPS enforced (no HTTP fallback)
- [ ] Logout clears session
- [ ] Service-specific features work (dashboards in Grafana, etc.)

---

## Troubleshooting Guide

### Problem: Portal Returns 502 Bad Gateway

**Symptoms**: Accessing https://254carbon.com shows "502 Bad Gateway"

**Solution**:
```bash
# Step 1: Check if pods are running
kubectl get pods -n data-platform -l app=portal

# Step 2: If not running, check why
kubectl describe pods -n data-platform -l app=portal

# Step 3: Check logs
kubectl logs -n data-platform -l app=portal

# Step 4: Restart if needed
kubectl rollout restart deployment/portal -n data-platform
```

### Problem: Cloudflare Access Login Loop

**Symptoms**: Redirects to login repeatedly even after authenticating

**Solution**:
```bash
# Step 1: Verify application exists in Cloudflare UI
# Go to: Zero Trust → Access → Applications → Should see "254Carbon Portal"

# Step 2: Verify policy is enabled
# Application settings → Policies → Should have "Allow" policy

# Step 3: Check DNS record
nslookup 254carbon.cloudflareaccess.com
# Should resolve to your Cloudflare Tunnel endpoint

# Step 4: Restart cloudflared
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel
```

### Problem: Services Show 401 Unauthorized

**Symptoms**: After logging in at portal, accessing services shows "401 Unauthorized"

**Solution**:
```bash
# Step 1: Verify service application exists in Cloudflare
# Go to: Zero Trust → Access → Applications
# Should see applications for: vault, minio, grafana, etc.

# Step 2: Check ingress annotations
kubectl get ingress vault-ingress -n data-platform -o yaml | grep auth-url
# Should show Cloudflare Access auth URL

# Step 3: Verify service is running
kubectl get pods -n data-platform -l app=vault

# Step 4: Check NGINX logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -f
# Look for errors related to auth
```

### Problem: Session Doesn't Persist Across Services

**Symptoms**: Logged in at portal, but services show login page

**Solution**:
```bash
# Step 1: Verify all services use same Cloudflare Account
# Check each ingress: kubectl get ingress -A -o yaml | grep cloudflareaccess.com

# Step 2: Verify cookies are HTTP-only and Secure
# Browser Dev Tools → Application → Cookies
# Should see CF-Access-JWT-Assertion cookie with Secure flag

# Step 3: Verify CNAME for each service
nslookup vault.254carbon.cloudflareaccess.com
nslookup grafana.254carbon.cloudflareaccess.com
# All should point to same Cloudflare Tunnel endpoint

# Step 4: Restart services
kubectl rollout restart deployment -n data-platform
kubectl rollout restart deployment -n monitoring
```

### Problem: Cloudflare Tunnel Connection Issues

**Symptoms**: Services are inaccessible or return "Bad Gateway"

**Solution**:
```bash
# Step 1: Check tunnel pod status
kubectl get pods -n cloudflare-tunnel

# Step 2: Check tunnel logs
kubectl logs -n cloudflare-tunnel -f
# Look for "Registered tunnel" message

# Step 3: Restart tunnel
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel

# Step 4: Verify tunnel configuration
kubectl describe configmap cloudflare-tunnel-config -n cloudflare-tunnel

# Step 5: Check Cloudflare dashboard
# Go to: Infrastructure → Tunnels
# Verify tunnel shows "Connected"
```

---

## Rollback Plan

If critical issues occur, follow this rollback procedure:

### Complete Rollback to Pre-SSO State

```bash
# Step 1: Delete Cloudflare Access applications (via Cloudflare UI)
# Zero Trust → Access → Applications → Delete each application

# Step 2: Remove ingress auth annotations
kubectl edit ingress -A
# Remove all nginx.ingress.kubernetes.io/auth-* annotations

# Step 3: Re-enable local authentication
kubectl -n monitoring patch configmap grafana-config --type merge -p '{
  "data": {
    "grafana.ini": "[auth.anonymous]\nenabled = true"
  }
}'
kubectl rollout restart deployment/grafana -n monitoring

kubectl -n data-platform set env deployment/superset SUPERSET_DISABLE_LOCAL_AUTH=false
kubectl rollout restart deployment/superset -n data-platform

# Step 4: Verify services work with local auth
curl https://grafana.254carbon.com   # Should show login page
curl https://superset.254carbon.com  # Should show login page

# Step 5: Portal remains as is (optional to delete)
# kubectl delete deployment portal -n data-platform
```

---

## Success Metrics

After completing Phase 2-4, verify:

| Metric | Target | Verification |
|--------|--------|--------------|
| Portal availability | 99.9% | Grafana dashboard / Prometheus metrics |
| Authentication success rate | >99% | Cloudflare Access logs |
| Average response time | <100ms | curl timing / APM monitoring |
| Session persistence | 100% | Manual testing across services |
| Failed login rate | <1% | Cloudflare audit logs |
| Audit log completeness | 100% | Review all user actions in logs |
| Service accessibility | 100% (9/9) | Manual verification of each service |
| Security policy enforcement | 100% | Unauthorized access denied |

---

## Final Checklist

Before declaring Phase 2-4 complete:

- [ ] Portal deployed and accessible
- [ ] Cloudflare Teams subscription verified
- [ ] All 9 Cloudflare Access applications created
- [ ] Portal application policy working
- [ ] Service applications policies working
- [ ] Audit logging enabled
- [ ] Grafana local auth disabled
- [ ] Superset local auth disabled
- [ ] All ingress rules updated with Cloudflare auth
- [ ] End-to-end auth flow tested
- [ ] Session persistence verified
- [ ] All 9 services accessible via SSO
- [ ] Security testing passed
- [ ] Performance requirements met
- [ ] Documentation updated
- [ ] Team trained on SSO system

---

## Next Steps After Phase 4

Once Phase 4 is complete:

1. **Documentation**: Update README.md with SSO access instructions for users
2. **Training**: Conduct team training on SSO login process
3. **Monitoring**: Set up alerts for authentication failures
4. **Rotation**: Plan credential rotation schedule (every 90 days)
5. **Audits**: Schedule regular security audits
6. **Enhancement**: Plan Phase 5 enhancements:
   - User profile management
   - Service bookmarking/favorites
   - Advanced search
   - Role-based access control
   - SAML/OIDC integration

---

## Support & Documentation Files

- `portal/README.md` - Portal application documentation
- `k8s/cloudflare/CLOUDFLARE_SSO_SETUP.md` - Detailed SSO configuration
- `k8s/cloudflare/SECURITY_POLICIES.md` - Security configuration details
- `k8s/cloudflare/README.md` - Cloudflare Tunnel documentation
- `README.md` - Main project documentation (update this after Phase 4)

---

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Portal | Done | ✅ Complete |
| Phase 2: Cloudflare Access | 1-2 hours | ⏳ Next |
| Phase 3: Service Integration | 2-3 days | ⏳ After Phase 2 |
| Phase 4: Testing | 1-2 days | ⏳ After Phase 3 |
| **Total Remaining** | **4-7 days** | **Ready to start** |

**Start Date**: October 19, 2025  
**Estimated Completion**: October 26-30, 2025

---

Generated: October 19, 2025
Version: 1.0
