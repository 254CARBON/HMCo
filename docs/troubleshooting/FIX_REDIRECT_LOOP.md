# Fix "Too Many Redirects" Error

## Problem

Getting "ERR_TOO_MANY_REDIRECTS" when accessing services through Cloudflare Tunnel.

## Root Cause

Mismatch between Cloudflare SSL/TLS mode and ingress controller configuration, causing an infinite redirect loop:

1. Cloudflare receives HTTPS request from browser
2. Cloudflare forwards to Tunnel (might convert to HTTP or HTTPS)
3. Ingress NGINX receives request
4. If there's a mismatch, one side redirects back, creating a loop

## Solution

### Quick Fix (Option 1): Set Cloudflare to Flexible SSL

**In Cloudflare Dashboard**:

1. Go to your domain (254carbon.com)
2. Navigate to **SSL/TLS** section
3. Change **SSL/TLS encryption mode** to: **Flexible**
4. Wait 30 seconds for changes to propagate
5. Test access to portal.254carbon.com

**Why this works**: Flexible mode means Cloudflare accepts HTTPS from visitors but connects to your origin over HTTP. Since your ingress has `ssl-redirect: false`, this prevents the loop.

### Permanent Fix (Option 2): Use Full SSL with Origin Certificates

**Step 1: Set Cloudflare to Full SSL mode**
- In Cloudflare Dashboard > SSL/TLS
- Set encryption mode to: **Full (strict)**

**Step 2: Remove ssl-redirect annotations and add backend protocol**
```bash
# Apply the fixed ingress configurations
kubectl apply -f k8s/ingress/portal-ingress-fixed.yaml
kubectl apply -f k8s/ingress/datahub-ingress-fixed.yaml
kubectl apply -f k8s/ingress/superset-ingress-fixed.yaml
```

**Step 3: Generate and install Origin Certificates** (follow existing guide)
```bash
# See: docs/ssl-tls/QUICKSTART_SSL_SETUP.md
```

## Current Configuration Check

```bash
# Check current ingress annotations
kubectl describe ingress portal-ingress -n data-platform | grep -A 5 "Annotations:"

# Should see:
# nginx.ingress.kubernetes.io/ssl-redirect: false
# nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
```

## Recommended: Option 1 (Flexible SSL) for Immediate Fix

This is the fastest solution and works well with Cloudflare Tunnel:

**Advantages**:
- ✓ No certificate management needed
- ✓ Works immediately
- ✓ Cloudflare still provides HTTPS to visitors
- ✓ Compatible with current ingress setup

**Disadvantages**:
- ⚠️ Connection between Cloudflare and origin is HTTP (but encrypted through tunnel)

## Testing After Fix

```bash
# Test from command line
curl -I https://portal.254carbon.com

# Should return HTTP 200 OK (not 301/302/307/308)

# Test in browser
# Open https://portal.254carbon.com
# Should load without redirect errors
```

## If Still Getting Redirects

1. **Clear browser cache**:
   - Chrome: Ctrl+Shift+Del, select "Cached images and files"
   - Or use Incognito mode

2. **Check ingress is updated**:
   ```bash
   kubectl get ingress portal-ingress -n data-platform -o yaml | grep ssl-redirect
   # Should show: ssl-redirect: "false"
   ```

3. **Verify Cloudflare SSL mode**:
   - Must be "Flexible" OR "Full" (not "Full (strict)" without valid certs)

4. **Check for conflicting annotations**:
   ```bash
   kubectl get ingress -A -o yaml | grep -E "force-ssl|ssl-redirect"
   ```

## Prevention

When creating new ingresses for Cloudflare Tunnel:

```yaml
annotations:
  nginx.ingress.kubernetes.io/ssl-redirect: "false"
  nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
```

## Related Issues

- Certificate errors → See: `docs/ssl-tls/QUICKSTART_SSL_SETUP.md`
- Cloudflare Access SSO → See: `k8s/cloudflare/CLOUDFLARE_ACCESS_SETUP_GUIDE.md`
- Tunnel not connecting → Check: `kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel`




