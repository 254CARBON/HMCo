# Cloudflare Access SSO Setup Guide (Moved — see docs/sso/guide.md)

This guide walks through configuring Cloudflare Access to provide Single Sign-On (SSO) for all cluster services through 254carbon.com.

## Overview

Cloudflare Access provides zero-trust security with centralized authentication. After setup:
- Users access 254carbon.com landing portal
- Cloudflare Access handles authentication
- Authenticated users can access all protected services
- All access is logged and audited

## Architecture

```
Internet User
    ↓
Cloudflare Edge (DDoS, WAF)
    ↓
Cloudflare Access (Authentication)
    ↓
254carbon.com Portal
    ↓
Service Links (Subdomains)
    ↓
NGINX Ingress → Kubernetes Services
```

## Prerequisites

- Cloudflare account with Enterprise or Teams plan (Access requires this)
- 254carbon.com domain configured in Cloudflare
- Cloudflare API token with appropriate permissions
- Kubernetes cluster with NGINX Ingress
- Portal deployed to cluster

## Step 1: Enable Cloudflare Teams

1. Log in to Cloudflare Dashboard
2. Go to **My Profile** → **Billing**
3. Verify you have **Cloudflare Teams** subscription
4. Navigate to **Zero Trust** → **Dashboard**

## Step 2: Create Identity Provider

### Self-Hosted (Email/Password)

1. In Cloudflare Zero Trust, go to **Settings** → **Authentication**
2. Enable **One-time PIN** for backup
3. Click **+ Add** under Login Methods
4. Select **Cloudflare One-time PIN**
5. Test: Send yourself a test OTP

### Recommended: Using Email + Password

1. Go to **Settings** → **Authentication**
2. Scroll to **Generic OIDC**
3. Click **+ Add**
4. Configure local identity provider (optional for advanced setup)

For this implementation, we'll use Cloudflare's built-in email verification.

## Step 3: Create Cloudflare Access Applications

### 3.1 Portal Application (254carbon.com)

1. Go to **Access** → **Applications**
2. Click **Add an application**
3. Select **Self-hosted**
4. Fill in details:
   - **Application name**: 254Carbon Portal
   - **Subdomain**: 254carbon
   - **Domain**: cloudflareaccess.com
   - **Application type**: Web
5. Click **Next**

### 3.2 Configure Portal Policy

1. Under **Policies**, click **+ Add a policy**
2. Create policy rules:

**Rule 1: Allow Anyone with Valid Email**
- **Policy name**: Allow All Users
- **Decision**: Allow
- **Include**: Email Suffix → example.com (customize as needed)

**Rule 2: Deny Everyone Else (Default)**
- **Policy name**: Deny All Others
- **Decision**: Block
- **Include**: Everyone

3. Click **Save policy**

### 3.3 Configure Portal Settings

1. Under **Settings** tab:
   - **Session Duration**: 24 hours
   - **Allowed IDPS**: Email + One-time PIN
   - **Auto-redirect**: Optional (disable for explicit login)

2. Under **Advanced** tab:
   - **Enable service token authentication**: Optional
   - **Custom error page**: Leave default

3. Click **Save**

## Step 4: Configure Service Applications

For each protected service (Vault, MinIO, DolphinScheduler), create similar applications:

### 4.1 Vault Application

1. Create new application:
   - **Name**: Vault.254Carbon
   - **Subdomain**: vault
   - **Domain**: cloudflareaccess.com

2. Add policy:
   - **Name**: Vault Users
   - **Decision**: Allow
   - **Include**: Email Suffix → example.com

3. Settings:
   - **Session Duration**: 24 hours

### 4.2 MinIO Application

1. Create new application:
   - **Name**: MinIO.254Carbon
   - **Subdomain**: minio
   - **Domain**: cloudflareaccess.com

2. Add policy:
   - **Name**: Storage Admins
   - **Decision**: Allow
   - **Include**: Email Suffix → admin.example.com

3. Settings:
   - **Session Duration**: 8 hours (shorter for storage)

### 4.3 DolphinScheduler Application

1. Create new application:
   - **Name**: DolphinScheduler.254Carbon
   - **Subdomain**: dolphin
   - **Domain**: cloudflareaccess.com

2. Add policy:
   - **Name**: Data Engineers
   - **Decision**: Allow
   - **Include**: Email Suffix → example.com

3. Settings:
   - **Session Duration**: 12 hours

### 4.4 Grafana & Superset Applications

1. **Grafana**:
   - **Name**: Grafana.254Carbon
   - **Subdomain**: grafana
   - **Policy**: Allow all authenticated users

2. **Superset**:
   - **Name**: Superset.254Carbon
   - **Subdomain**: superset
   - **Policy**: Allow all authenticated users

## Step 5: Configure Service Tokens (Optional)

For service-to-service authentication:

1. Go to **Access** → **Service Auth** → **Service Tokens**
2. Click **Create**
3. Configure:
   - **Name**: portal-to-vault
   - **Duration**: 365 days
4. Store credentials securely in Vault

## Step 6: Configure DNS in Cloudflare

All Access applications get CNAME records automatically.

### Public DNS (Optional - for non-Access services)

For public services (Grafana, Superset without protection):

1. Go to Cloudflare DNS settings
2. Create CNAME records:
   ```
   grafana.254carbon.com → tunnel-endpoint.cfargotunnel.com
   superset.254carbon.com → tunnel-endpoint.cfargotunnel.com
   ```

## Step 7: Update Nginx Ingress Annotations

Update ingress rules to work with Cloudflare Access:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vault-ingress
  namespace: data-platform
  annotations:
    nginx.ingress.kubernetes.io/auth-url: "https://<account>.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://<account>.cloudflareaccess.com/cdn-cgi/access/login"
spec:
  # ... rest of ingress config
```

## Step 8: Configure JWT Validation (Optional)

For services that need to validate Cloudflare JWT tokens:

1. Go to **Access** → **Applications**
2. Select application
3. Under **Advanced**, enable **JWT token validation**
4. Copy **JWT Public Key**
5. Configure services to validate tokens using this key

### Example: Validate in Application Code

```python
import jwt
from cryptography.hazmat.primitives import serialization

# Get public key from Cloudflare
public_key_pem = """-----BEGIN PUBLIC KEY-----
..."""

public_key = serialization.load_pem_public_key(
    public_key_pem.encode()
)

# Validate token from header
token = request.headers.get('CF-Access-JWT-Assertion')
payload = jwt.decode(token, public_key, algorithms=['RS256'])
user_email = payload['email']
```

## Step 9: Enable Audit Logging

1. Go to **Access** → **Logs**
2. View real-time access logs
3. Configure **Email alerts** for:
   - Failed authentications (>5 failures/hour)
   - Policy changes
   - Service token usage

## Step 10: Test SSO Flow

### Test Portal Access

```bash
# From outside the cluster
curl -v https://254carbon.com

# Should redirect to Cloudflare Access login
# After authentication, should show portal
```

### Test Service Access

```bash
# After authentication at portal
curl -v https://vault.254carbon.com

# Should have CF-Access-JWT-Assertion header
# Should grant access to Vault UI
```

### Test Audit Logs

1. Go to **Access** → **Logs**
2. Verify entries for:
   - Portal access
   - Service access
   - Policy evaluations

## Configuration as Code

### Terraform (Optional)

```hcl
resource "cloudflare_access_application" "portal" {
  zone_id = var.zone_id
  name    = "254Carbon Portal"
  domain  = "254carbon.cloudflareaccess.com"

  allow_authenticate_via_warp = false

  policies = [
    cloudflare_access_policy.allow_all_users.id,
  ]
}

resource "cloudflare_access_policy" "allow_all_users" {
  zone_id = var.zone_id
  application_id = cloudflare_access_application.portal.id
  name = "Allow All Users"
  decision = "allow"

  include {
    email_domain = ["example.com"]
  }
}
```

## Troubleshooting

### Users Can't Access Portal

1. **Check Application Status**:
   - Verify application exists in Cloudflare Zero Trust
   - Confirm policy is enabled
   - Check session duration

2. **Check DNS**:
   ```bash
   nslookup 254carbon.cloudflareaccess.com
   nslookup 254carbon.com
   ```

3. **Review Logs**:
   - Go to **Access** → **Logs**
   - Look for authentication failures
   - Check policy denials

4. **Verify Tunnel**:
   ```bash
   kubectl logs -n cloudflare-tunnel -f
   ```

### Services Behind Access Inaccessible

1. **Check Ingress Annotations**:
   - Verify auth-url is correct
   - Verify auth-signin is configured

2. **Check Service Status**:
   ```bash
   kubectl get pods -n data-platform | grep <service>
   ```

3. **Check NGINX Logs**:
   ```bash
   kubectl logs -n ingress-nginx -f
   ```

### JWT Validation Fails

1. **Verify Token Signing**:
   - Ensure service validates correct public key
   - Check token expiration

2. **Verify Header Forwarding**:
   - Confirm NGINX passes `CF-Access-JWT-Assertion` header
   - Check application receives header

## Security Best Practices

1. **Session Management**:
   - Set appropriate session durations
   - Shorter for sensitive services (Vault: 1-2 hours)
   - Longer for public dashboards (Grafana: 24 hours)

2. **Policy Configuration**:
   - Use email domain restrictions
   - Implement role-based policies
   - Regular policy audits

3. **Token Handling**:
   - Never log tokens
   - Validate token signatures
   - Rotate service tokens regularly

4. **Audit & Monitoring**:
   - Enable all audit logging
   - Set up alerts for suspicious activity
   - Weekly review of access logs

## Maintenance

### Rotate Credentials (Every 90 Days)

1. Go to **Settings** → **Authentication**
2. Regenerate identity provider credentials
3. Update applications to use new credentials

### Update Policies

1. Review **Access** → **Applications**
2. Update policies based on team changes
3. Remove unused applications

### Monitor Usage

1. Check **Access** → **Logs** weekly
2. Verify expected usage patterns
3. Flag unusual access patterns

## References

- [Cloudflare Zero Trust Documentation](https://developers.cloudflare.com/cloudflare-one/)
- [Cloudflare Access Policy Documentation](https://developers.cloudflare.com/cloudflare-one/policies/access/)
- [Cloudflare JWT Documentation](https://developers.cloudflare.com/cloudflare-one/identity/authorization-policy/access-token/)
- [NGINX Cloudflare Access Integration](https://developers.cloudflare.com/cloudflare-one/applications/configure-apps/nginx/)

## Support

For configuration issues:
1. Review application logs: `kubectl logs -n data-platform -l app=portal`
2. Check Cloudflare Access logs in dashboard
3. Verify NGINX ingress configuration
4. Test basic connectivity without authentication
