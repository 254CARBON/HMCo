# Cloudflare Security Policies & Access Control

## Overview

This document outlines the security configuration for exposing services through Cloudflare Tunnel, including authentication, access control, and WAF rules.

## Security Architecture

```
Request → Cloudflare Edge (DDoS + WAF) → Access Control (if enabled) → Tunnel → NGINX → Service
```

## Service Classification

### Tier 1: Public Services (No Authentication)

Services accessible to anyone via 254carbon.com with only Cloudflare DDoS protection:

- **Grafana** (`grafana.254carbon.com`) - Read-only monitoring dashboards
- **Superset** (`superset.254carbon.com`) - Public data visualizations

**Configuration:**
- No Cloudflare Access policy required
- Enable WAF and rate limiting only
- Use Cloudflare caching for static assets

### Tier 2: Protected Services (Cloudflare Access)

Services requiring authentication before access:

- **Vault** (`vault.254carbon.com`) - Secrets management
- **MinIO** (`minio.254carbon.com`) - S3 storage console
- **DolphinScheduler** (`dolphin.254carbon.com`) - Workflow orchestration

**Configuration:**
- Enable Cloudflare Access
- Require multi-factor authentication (MFA)
- Support SSO integration (Google, GitHub, Okta)
- Maintain audit logs

### Tier 3: Internal Services (Not Exposed)

Services for internal cluster use only:

- Kafka, Redis, PostgreSQL, Elasticsearch
- Backend APIs without UIs
- Database services

**Configuration:**
- No ingress rules for 254carbon.com
- Accessible only via port-forward or internal DNS
- NetworkPolicies restrict external access

## Cloudflare Access Setup

### Step 1: Enable Cloudflare Access

1. Log in to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/)
2. Navigate to **Applications** → **Create Application**
3. Select **Self-hosted**

### Step 2: Create Access Policies for Vault

**Application Details:**
- Application name: `vault.254carbon.com`
- Application domain: `vault.254carbon.com`
- Session duration: 24 hours

**Policy Rules:**

```yaml
# Policy 1: Require Email Domain
Name: "Corporate Email"
Action: Allow
Include: Emails matching example.com

---

# Policy 2: Require GitHub
Name: "GitHub Users"
Action: Allow
Include: GitHub teams: dev-team, sec-team

---

# Policy 3: Require SSO + MFA
Name: "Google + MFA"
Action: Allow
Include: Google OAuth
Additional: Require MFA within 24 hours
```

**Deny Rule:**
```yaml
Name: "Deny All Others"
Action: Block
Include: Everyone except above
```

### Step 3: Create Access Policies for MinIO

**Application Details:**
- Application name: `minio.254carbon.com`
- Application domain: `minio.254carbon.com`
- Session duration: 8 hours (shorter for storage admin)

**Policy Rules:**

```yaml
# Policy 1: IP-based restriction
Name: "Internal IPs Only"
Action: Allow
Include: IP Ranges: 192.168.1.0/24, 10.0.0.0/8

---

# Policy 2: Require GitHub + specific teams
Name: "Platform Team"
Action: Allow
Include: GitHub teams: platform-admin, devops
Additional: Require authentication
```

### Step 4: Create Access Policies for DolphinScheduler

**Application Details:**
- Application name: `dolphin.254carbon.com`
- Application domain: `dolphin.254carbon.com`
- Session duration: 12 hours

**Policy Rules:**

```yaml
# Policy 1: GitHub Users
Name: "Data Team"
Action: Allow
Include: GitHub teams: data-platform, analytics

---

# Policy 2: Require MFA
Name: "MFA Required"
Additional: Require MFA enrollment
```

## Cloudflare WAF Rules

### Step 1: Enable OWASP Core Ruleset

1. Navigate to **Security** → **WAF**
2. Go to **Managed Rules** → **Cloudflare Managed Ruleset**
3. Enable the following:
   - ✅ Cloudflare Managed Ruleset
   - ✅ OWASP ModSecurity Core Ruleset
   - ✅ Cloudflare Free Tier Ruleset

### Step 2: Configure Rate Limiting

```bash
# High-volume APIs
curl https://api.example.com/*
Rate limit: 1000 req/min per IP
Action: Challenge

---

# Standard endpoints
https://service.254carbon.com/*
Rate limit: 100 req/min per IP
Action: Block

---

# Authentication endpoints
https://*/api/auth/*
Rate limit: 10 req/min per IP
Action: Block
```

### Step 3: Enable DDoS Protection

Already enabled by default. Configuration:

- **Sensitivity Level**: Medium
- **DDoS Mitigation**: Enabled
- **Challenge Disposition**: Auto

### Step 4: Block Malicious Bots

Navigate to **Security** → **Bots**

```yaml
Super Bot Fight Mode:
  Definitely Automated: Challenge
  Likely Automated: Managed Challenge
  Verified Bots: Allow
  Human Users: Allow
```

## SSL/TLS Configuration

### Recommended: Flexible SSL/TLS

Current setup uses **Flexible** mode (ideal for development):
- Cloudflare ↔ Origin: HTTP (unencrypted)
- Client ↔ Cloudflare: HTTPS (encrypted)

**Enablement:**
Already default in Cloudflare. Ingress uses self-signed certificates.

### Production: Full SSL/TLS

For production environment, use **Full (Strict)**:

1. Generate origin certificate in Cloudflare:
   - **SSL/TLS** → **Origin Server**
   - Create certificate for `*.254carbon.com`
   - Download certificate and key
   
2. Deploy to cluster:
```bash
kubectl create secret tls cloudflare-origin-cert \
  --cert=cert.pem \
  --key=key.pem \
  -n ingress-nginx
```

3. Update NGINX Ingress annotations:
```yaml
nginx.ingress.kubernetes.io/ssl-certificate: ingress-nginx/cloudflare-origin-cert
```

## API Token Security

### Token Permissions

The API token used for tunnel credentials should have **minimal permissions**:

**Recommended Permissions:**
- Account: Tunnel: Edit
- Zone: DNS: Edit (if using DNS script)
- Zone: Cache Purge: Purge

**NEVER grant:**
- ✗ User: Billing Manage
- ✗ Account: Super admin
- ✗ Zone: Unlimited admin

### Token Rotation

Rotate tunnel credentials every 90 days:

```bash
# 1. Generate new credentials in Cloudflare dashboard
# 2. Update secret
kubectl edit secret cloudflare-tunnel-credentials -n cloudflare-tunnel

# 3. Restart tunnels
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel

# 4. Monitor reconnection
kubectl logs -n cloudflare-tunnel -f
```

## Audit Logging

### Enable Audit Logging

1. **Cloudflare Dashboard**: **Logs** → **Access Requests**
2. Monitor who accessed what and when
3. Export logs for compliance

Example query:
```
Application: vault.254carbon.com
Action: Allow
Time: Last 7 days
```

### Kubernetes Audit Logs

Monitor ingress access within cluster:

```bash
# View ingress logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -f

# Watch for 254carbon.com requests
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -f | \
  grep -E "254carbon|vault|minio|dolphin"
```

## Firewall Rules

### Cloudflare Firewall Rules

Advanced rule examples:

```yaml
# Block specific countries
Rule: cf.threat_score > 75
Action: Block
Description: "Block high-threat IPs"

---

# Rate limit by path
Rule: (cf.threat_score >= 50) or (ip.geoip.country == "CN")
Action: Challenge
Description: "Challenge suspicious IPs and specific countries"

---

# Whitelist corporate network
Rule: ip.src in {192.168.0.0/16 10.0.0.0/8}
Action: Allow
Description: "Allow corporate network"
```

## Security Monitoring

### Dashboard Metrics

Monitor in Cloudflare dashboard:

1. **Analytics** → **Overview**
   - Requests by status
   - Top threats blocked
   - Data transfer

2. **Security** → **Events**
   - WAF rule triggers
   - Rate limit blocks
   - Bot protection hits

3. **Access** → **Logs**
   - User logins
   - Failed authentication
   - Policy changes

### Alerting (Cloudflare Notifications)

Set up email alerts for:
- High rate of 4xx errors
- WAF rule triggered > 100 times/hour
- Access policy violations
- DDoS attack detected

Enable in **Notifications** → **Alert builder**

## Incident Response

### If Services Are Under Attack

1. **Immediate**: Enable DDoS protection (already on)
2. **Dashboard**: View real-time analytics
3. **Block**: Add firewall rule to block attacker IP
4. **Escalate**: If needed, enable additional rate limiting
5. **Review**: Post-incident analysis via Cloudflare logs

### If Access Control Fails

1. **Temporary disable**: Remove policy rules (immediate effect)
2. **Check logs**: Review authentication failures
3. **Reset sessions**: Force user re-authentication
4. **Re-enable**: After fixing root cause

## Best Practices

### Development Environment

- Use Cloudflare Access for all Tier 2 services
- Test policies with test domains first
- Keep token rotation frequent (30 days)
- Monitor logs daily

### Production Environment

- Enforce MFA for all Tier 2 services
- Rotate tokens every 30 days
- Daily audit log review
- Quarterly security assessment
- Implement IP allowlisting where possible
- Regular WAF tuning to reduce false positives

## References

- [Cloudflare Access Documentation](https://developers.cloudflare.com/cloudflare-one/identity/idp-integration/)
- [Cloudflare WAF Rules](https://developers.cloudflare.com/waf/)
- [Cloudflare DDoS Protection](https://www.cloudflare.com/ddos/)
- [OWASP ModSecurity Core Ruleset](https://owasp.org/www-project-modsecurity-core-rule-set/)
