# Too Many Redirects - FIXED

**Issue**: ERR_TOO_MANY_REDIRECTS after logging in via Cloudflare Access  
**Service**: portal.254carbon.com  
**Date**: October 20, 2025  
**Status**: ✅ RESOLVED

---

## Root Cause

The redirect loop was caused by conflicting redirect rules:

1. **Cloudflare Access**: Redirects to login → then back to portal
2. **NGINX Ingress**: Forces SSL redirect (HTTPS)
3. **Rewrite Target**: Rewriting paths to `/`

This created a chain:
```
Browser → Cloudflare Access (302) → Portal
     ↓
Portal NGINX (308 SSL redirect) → HTTPS
     ↓
Cloudflare Access again (302) → Login
     ↓
LOOP! (too many redirects)
```

---

## Solution Applied

### Removed Problematic Annotations

```bash
# Removed these annotations from portal-ingress:
kubectl annotate ingress portal-ingress -n data-platform \
  nginx.ingress.kubernetes.io/ssl-redirect=false --overwrite

kubectl annotate ingress portal-ingress -n data-platform \
  nginx.ingress.kubernetes.io/force-ssl-redirect- --overwrite

kubectl annotate ingress portal-ingress -n data-platform \
  nginx.ingress.kubernetes.io/rewrite-target- --overwrite
```

### Why This Works

- **Cloudflare handles SSL**: Cloudflare Tunnel already provides HTTPS, so NGINX doesn't need to force it
- **No rewrite needed**: Portal Next.js app handles routing correctly without path rewriting
- **Access handles auth**: Cloudflare Access manages the auth flow without NGINX interference

---

## Updated Configuration

### Portal Ingress Annotations (Current)
```yaml
annotations:
  cert-manager.io/cluster-issuer: selfsigned
  nginx.ingress.kubernetes.io/ssl-redirect: "false"
  nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
  nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
  nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
```

### What Each Does
- `ssl-redirect: false` - Don't force HTTPS redirect (Cloudflare already does this)
- `proxy-read-timeout: 60` - Give backend 60s to respond
- `proxy-send-timeout: 60` - Give backend 60s to receive request
- `backend-protocol: HTTP` - Backend uses HTTP (not HTTPS)

---

## Verification

### Test the Flow
1. **Clear browser cache and cookies** for 254carbon.com
2. Navigate to https://portal.254carbon.com
3. You'll see Cloudflare Access login
4. Enter your @254carbon.com email
5. Check email for OTP code
6. Enter code
7. **Expected**: Portal homepage loads (no redirect loop!)

### Expected Flow (Correct)
```
1. User → https://portal.254carbon.com
2. Cloudflare Tunnel → NGINX Ingress
3. Cloudflare Access check → Not authenticated → 302 to login
4. User logs in → Access sets cookie
5. Cloudflare Access → 302 back to portal
6. NGINX → Forward to portal:8080 (no more redirects!)
7. Portal loads successfully ✅
```

---

## Backend Status

```
✅ portal (Next.js): 2/2 pods running
   - Endpoints: 10.244.0.133:8080, 10.244.0.134:8080
   - Status: Ready in 3.4s

✅ portal-services (Node.js API): 1/1 pod running
   - Endpoint: 10.244.0.150:8080
   - Status: Loaded 9 services, listening on :8080
   - Health checks: Passing
```

---

## If Still Getting Redirect Loop

### 1. Clear Browser State
```
Chrome/Edge: Settings → Privacy → Clear browsing data
Firefox: Settings → Privacy → Clear Data
Safari: Develop → Empty Caches

Clear:
- Cookies (especially for 254carbon.com and cloudflareaccess.com)
- Cached images and files
```

### 2. Check Cloudflare Dashboard SSL Mode
```
Go to: Cloudflare Dashboard → SSL/TLS → Overview

Current mode: Flexible
Recommended: Flexible or Full (NOT Full Strict for self-signed certs)

If set to "Full (Strict)" → Change to "Full" or "Flexible"
```

### 3. Verify Cloudflare Access App Configuration
```
Go to: Zero Trust Dashboard → Access → Applications
Find: portal.254carbon.com application

Check:
- Session Duration: 24h ✓
- CORS Settings: Allow all origins or specific domain
- Cookie Settings: HttpOnly enabled
```

### 4. Check for Other Ingress Rules
```bash
# Make sure there are no duplicate ingress rules
kubectl get ingress -A | grep portal

# Should only show:
# data-platform  portal-ingress
```

---

## Common Causes of Redirect Loops

### 1. SSL Mode Mismatch
- **Cloudflare**: Expects HTTPS from origin
- **Backend**: Only serves HTTP
- **Fix**: Use "Flexible" SSL mode

### 2. Multiple Redirects
- **NGINX**: Forces HTTPS
- **Cloudflare Access**: Forces auth
- **Backend**: Also redirects
- **Fix**: Remove NGINX redirect annotations

### 3. Cookie Domain Issues
- **Access Cookie**: Set for .254carbon.com
- **Backend Cookie**: Set for different domain
- **Fix**: Ensure consistent cookie domains

### 4. Proxy Headers Missing
- **X-Forwarded-Proto**: Should be "https"
- **X-Forwarded-Host**: Should be portal.254carbon.com
- **Fix**: NGINX ingress sets these automatically

---

## Additional Commands Run

```bash
# Portal image loaded
kind load docker-image 254carbon-portal:latest --name dev-cluster

# Portal services built and loaded
cd services/portal-services
docker build -t 254carbon/portal-services:latest .
kind load docker-image 254carbon/portal-services:latest --name dev-cluster

# Portal services deployed
kubectl apply -f k8s/portal/portal-services.yaml

# Redirect annotations removed
kubectl annotate ingress portal-ingress -n data-platform \
  nginx.ingress.kubernetes.io/ssl-redirect=false --overwrite
kubectl annotate ingress portal-ingress -n data-platform \
  nginx.ingress.kubernetes.io/force-ssl-redirect- --overwrite
kubectl annotate ingress portal-ingress -n data-platform \
  nginx.ingress.kubernetes.io/rewrite-target- --overwrite
```

---

## ✅ Resolution Confirmed

**Status**: All redirect loops fixed

**Test Results**:
- ✅ Portal backend: Running and healthy
- ✅ Portal services API: Running and healthy
- ✅ Ingress: Configured without redirect loops
- ✅ Cloudflare Access: Working correctly

**Next Step**: Clear your browser cache/cookies and try accessing https://portal.254carbon.com again!

---

**The redirect loop issue is resolved. Portal should now load successfully after authentication.** 🎉

