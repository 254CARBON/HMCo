# Fix Superset 502 Error

## Problem

Accessing https://superset.254carbon.com returns HTTP 502 Bad Gateway

## Root Cause

Superset runs with a base path `/superset` but the ingress is configured for root path `/`. This causes:
1. Browser accesses `/`
2. Superset redirects to `/superset/welcome/`
3. Cloudflare/browser can't handle the redirect loop properly

## Solution: Access the Correct URL

**The simplest solution**: Access Superset at its actual login page:

**https://superset.254carbon.com/superset/login**

Or bookmark these URLs:
- Login: https://superset.254carbon.com/superset/login
- Home: https://superset.254carbon.com/superset/welcome
- Dashboards: https://superset.254carbon.com/superset/dashboard/list/

## Alternative: Update Cloudflare Access Application

1. Go to Cloudflare Dashboard
2. Navigate to **Zero Trust** > **Access** > **Applications**
3. Find "Superset.254Carbon" application
4. Edit the application settings
5. Update **Application domain** from:
   - `superset.254carbon.com` 
   - To: `superset.254carbon.com/superset`
6. Save changes

This will make https://superset.254carbon.com redirect directly to the login page.

## Long-term Fix: Disable Superset Base Path

Edit the Superset deployment to remove the `/superset` prefix:

```bash
# Get current config
kubectl get deployment superset-web -n data-platform -o yaml > superset-backup.yaml

# Edit deployment
kubectl edit deployment superset-web -n data-platform

# Add this environment variable:
env:
- name: SUPERSET_WEBSERVER_BASEPATH
  value: ""

# Or use kubectl set env
kubectl set env deployment/superset-web -n data-platform \
  SUPERSET_WEBSERVER_BASEPATH=""
```

Then restart Superset:
```bash
kubectl rollout restart deployment/superset-web -n data-platform
```

## Current Workaround

**For now, just use**: https://superset.254carbon.com/superset/login

Default credentials:
- Username: `admin`
- Password: `admin`

**⚠️ Change the password after first login!**

## Testing

```bash
# Test that Superset is responding
kubectl run test-superset --image=curlimages/curl:latest --rm -i --restart=Never -- \
  curl -I http://superset.data-platform.svc.cluster.local:8088

# Should return HTTP 302 with Location: /superset/welcome/
```

## Verification Checklist

- [ ] Can access https://superset.254carbon.com/superset/login
- [ ] Cloudflare Access login page appears
- [ ] Can login with admin/admin
- [ ] Superset dashboard loads after login

## Related Issues

- If getting "Too Many Redirects" → Check `docs/troubleshooting/FIX_REDIRECT_LOOP.md`
- If Cloudflare Access not working → Check `k8s/cloudflare/CLOUDFLARE_ACCESS_SETUP_GUIDE.md`
- If login fails → Check Superset pods: `kubectl logs -n data-platform -l app=superset-web`





