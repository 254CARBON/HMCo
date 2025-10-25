# NGINX Configuration Snippet Limitation

## Issue Encountered

Your NGINX Ingress Controller has `configuration-snippet` directives disabled by an administrator for security reasons. This is a common security hardening practice.

**Error Message:**
```
nginx.ingress.kubernetes.io/configuration-snippet annotation cannot be used. 
Snippet directives are disabled by the Ingress administrator
```

## What This Means

- ❌ Cannot use custom NGINX configuration snippets
- ❌ Cannot pass authenticated user info to backend services
- ✅ BUT: Cloudflare Access protection still works at the edge!

## Solution Implemented

We've updated the ingress resources to use ONLY the basic Cloudflare Access annotations that don't require configuration snippets:

**What remains enabled:**
- `auth-url`: Routes requests to Cloudflare Access for authentication
- `auth-signin`: Redirects unauthenticated users to login
- `auth-response-headers`: Passes JWT assertion header

**What was removed:**
- `configuration-snippet`: Custom NGINX directives (disabled)
- Header manipulation logic: No longer passing user email to backend

## How It Still Works

```
User Request
    ↓
Cloudflare Access checks authentication (@254carbon.com, @project52.org)
    ↓
If authenticated → Request passes through
If not authenticated → Redirected to login page
    ↓
NGINX forwards to backend service
```

## Benefits Still Present

✅ **Zero Trust Access** - Only authenticated users can access
✅ **Domain-based Access** - @254carbon.com & @project52.org only
✅ **Session Management** - 24-hour sessions enforced
✅ **No Manual Credentials** - Uses email authentication

## Trade-offs

**Without configuration-snippet:**
- Backend services don't receive X-WEBAUTH-USER header
- Backend services don't know the authenticated user's email
- Services can't implement per-user features based on Cloudflare headers

**Impact:**
- For most services (Vault, Prometheus, AlertManager): No impact
- If services need to track authenticated users: They won't have that info
- Alternative: Services can parse JWT token directly from CF-Access-JWT-Assertion header

## Workaround for Services Needing User Info

If a backend service needs to know who the authenticated user is:

**Option 1: Parse JWT Token**
- Backend receives `CF-Access-JWT-Assertion` header
- Backend can decode/validate JWT token
- Extract user email from JWT claims

**Option 2: Request Administrator Enable Snippets**
- Contact cluster administrator
- Request enabling `configuration-snippet` for specific services
- Security risk trade-off must be evaluated

## Verified Configuration

Fixed ingress files:
- ✅ `k8s/ingress/vault-ingress.yaml`
- ✅ `k8s/ingress/prometheus-ingress.yaml`
- ✅ `k8s/ingress/alertmanager-ingress.yaml`

All now use auth-url without configuration-snippet.

## Deployment Status

Ready to deploy. Run:
```bash
./scripts/deploy-security-monitoring.sh
```

The script will deploy all three ingress resources with the corrected annotations.

---

**This limitation does NOT reduce security.**
It's a cluster-level security policy that's working as intended.
