# Cloudflare Access, WAF & Monitoring Implementation Guide

**Date**: October 25, 2025
**Configuration Profile**: Enterprise Security + Comprehensive Monitoring
**Status**: Ready for Implementation

---

## Table of Contents
1. [Cloudflare Access Configuration](#cloudflare-access-configuration)
2. [WAF Rules Setup](#waf-rules-setup)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Implementation Steps](#implementation-steps)
5. [Verification Checklist](#verification-checklist)

---

## Cloudflare Access Configuration

### Overview
Configure Cloudflare Access to protect Vault, Prometheus, and Kong Admin behind authentication.

**Configuration Profile:**
- **Allowed Domains**: @254carbon.com, @project52.org
- **No IP/Country Restrictions**: All geographic locations allowed
- **Vault Access**: All authenticated users
- **Prometheus/Kong**: All authenticated users from allowed domains

### Part 1: Update Ingress Resources with Authentication

Update vault-ingress, prometheus-ingress, alertmanager-ingress, and kong-admin-ingress with Cloudflare Access headers.

#### Step 1.1: Update Vault Ingress
**File**: `k8s/ingress/vault-ingress.yaml` (or create if doesn't exist)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vault-ingress
  namespace: vault-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    # Cloudflare Access Authentication
    nginx.ingress.kubernetes.io/auth-url: https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize
    nginx.ingress.kubernetes.io/auth-signin: https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$escaped_request_uri
    nginx.ingress.kubernetes.io/auth-response-headers: cf-access-jwt-assertion
    nginx.ingress.kubernetes.io/configuration-snippet: |
      auth_request_set $cf_email $upstream_http_cf_access_authenticated_user_email;
      auth_request_set $cf_groups $upstream_http_cf_access_groups;
      proxy_set_header X-WEBAUTH-USER $cf_email;
      proxy_set_header X-WEBAUTH-EMAIL $cf_email;
      proxy_set_header X-WEBAUTH-GROUPS $cf_groups;
spec:
  ingressClassName: nginx
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
  tls:
  - hosts:
    - vault.254carbon.com
    secretName: vault-tls
```

#### Step 1.2: Update Prometheus Ingress
**File**: `k8s/ingress/prometheus-ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prometheus-ingress
  namespace: monitoring
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    # Cloudflare Access Authentication
    nginx.ingress.kubernetes.io/auth-url: https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize
    nginx.ingress.kubernetes.io/auth-signin: https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$escaped_request_uri
    nginx.ingress.kubernetes.io/auth-response-headers: cf-access-jwt-assertion
    nginx.ingress.kubernetes.io/configuration-snippet: |
      auth_request_set $cf_email $upstream_http_cf_access_authenticated_user_email;
      auth_request_set $cf_groups $upstream_http_cf_access_groups;
      proxy_set_header X-WEBAUTH-USER $cf_email;
      proxy_set_header X-WEBAUTH-EMAIL $cf_email;
      proxy_set_header X-WEBAUTH-GROUPS $cf_groups;
spec:
  ingressClassName: nginx
  rules:
  - host: prometheus.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
  tls:
  - hosts:
    - prometheus.254carbon.com
    secretName: prometheus-tls
```

#### Step 1.3: Update AlertManager Ingress
**File**: `k8s/ingress/alertmanager-ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: alertmanager-ingress
  namespace: monitoring
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    # Cloudflare Access Authentication
    nginx.ingress.kubernetes.io/auth-url: https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize
    nginx.ingress.kubernetes.io/auth-signin: https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$escaped_request_uri
    nginx.ingress.kubernetes.io/auth-response-headers: cf-access-jwt-assertion
    nginx.ingress.kubernetes.io/configuration-snippet: |
      auth_request_set $cf_email $upstream_http_cf_access_authenticated_user_email;
      auth_request_set $cf_groups $upstream_http_cf_access_groups;
      proxy_set_header X-WEBAUTH-USER $cf_email;
      proxy_set_header X-WEBAUTH-EMAIL $cf_email;
      proxy_set_header X-WEBAUTH-GROUPS $cf_groups;
spec:
  ingressClassName: nginx
  rules:
  - host: alertmanager.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: alertmanager
            port:
              number: 9093
  tls:
  - hosts:
    - alertmanager.254carbon.com
    secretName: alertmanager-tls
```

#### Step 1.4: Update Kong Admin Ingress (if needed)
**File**: Verify/update existing Kong ingress with auth headers

### Part 2: Configure in Cloudflare Dashboard

**Steps to perform in Cloudflare Dashboard**:

1. **Log in** → https://dash.cloudflare.com
2. **Navigate** → Zero Trust → Access → Applications
3. **Create Application for Vault**:
   - **Application name**: vault.254carbon.com
   - **Session duration**: 24 hours
   - **Application type**: SaaS application
   - **Application domain**: vault.254carbon.com
   
4. **Set Up Login Methods**:
   - Click "Configure login rules"
   - Add rule: "Emails ending with @254carbon.com"
   - Add rule: "Emails ending with @project52.org"
   - Action: Allow

5. **Repeat steps 3-4 for**:
   - prometheus.254carbon.com
   - alertmanager.254carbon.com
   - kong.254carbon.com

6. **Optional: Add CORS/Headers**:
   - Add HTTP request header: `X-WEBAUTH-EMAIL: $cf_email`
   - This passes authenticated user info to backend services

---

## WAF Rules Setup

### Configuration Profile
- **Rule Set**: Cloudflare Managed Ruleset (Standard)
- **Threat Level**: Conservative
- **IP Bypass**: Internal IPs allowed to bypass WAF

### Part 1: Enable Managed Ruleset

**Steps in Cloudflare Dashboard**:

1. **Navigate** → Security → WAF → Managed Rules
2. **Enable** Cloudflare Managed Ruleset:
   - Click "Ruleset name: Cloudflare Managed Ruleset"
   - Set **Sensitivity**: Low (Conservative)
   - Action: Block

3. **Configure WAF Rule Actions**:
   - SQL Injection: Block
   - XSS Attacks: Block
   - Bot Detection: Challenge (not block)
   - DDoS Protection: Enabled

### Part 2: IP Whitelist Configuration

**For internal IP bypass** (so internal monitoring tools work):

1. Navigate to **Security → WAF → Tools**
2. Create **IP Access Rule**:
   - IP: `192.168.1.0/24` (your internal cluster IPs)
   - Action: Allow
   - Priority: High

3. Add additional internal IPs as needed

### Part 3: Rate Limiting Rules

1. Navigate to **Security → Rate Limiting**
2. Create rules for API endpoints:
   - **Path**: `/api/*`
   - **Threshold**: 100 requests per 10 minutes
   - **Action**: Block for 15 minutes
   - **Recommended**: 5 violations in 5 minutes (as per your config)

---

## Monitoring & Alerting

### Alert Types to Configure

**All 5 types enabled:**
1. ✅ Certificate expiration warnings (7 & 30 days)
2. ✅ Tunnel connectivity issues
3. ✅ WAF rule violations (5 in 5 minutes)
4. ✅ Service health degradation
5. ✅ Anomaly detection

### Part 1: Set Up Alertmanager Configuration

**File**: Update/create `k8s/monitoring/alertmanager-config.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      slack_api_url: ''  # Leave empty - using email instead
    
    route:
      receiver: 'email'
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      routes:
        # Certificate expiration alerts
        - match:
            alertname: CertificateExpiration
          receiver: 'certificate-alerts'
          group_wait: 5m
        
        # Tunnel connectivity alerts
        - match:
            alertname: TunnelDown
          receiver: 'tunnel-alerts'
          group_wait: 2m
        
        # WAF violation alerts
        - match:
            alertname: WAFViolation
          receiver: 'security-alerts'
          group_wait: 5m
        
        # Service health alerts
        - match:
            alertname: ServiceDown
          receiver: 'platform-alerts'
    
    receivers:
      - name: 'email'
        email_configs:
          - to: 'qagiw3@gmail.com'
            from: 'alertmanager@254carbon.com'
            smarthost: 'smtp.gmail.com:587'
            auth_username: 'your-email@gmail.com'
            auth_password: 'your-app-password'
            headers:
              Subject: '{{ .GroupLabels.alertname }} - {{ .GroupLabels.service }}'
            html: |
              Alert: {{ .GroupLabels.alertname }}
              Service: {{ .GroupLabels.service }}
              Status: {{ .Status }}
              Details: {{ .Alerts }}
      
      - name: 'certificate-alerts'
        email_configs:
          - to: 'qagiw3@gmail.com'
            from: 'alerts@254carbon.com'
            headers:
              Subject: 'URGENT: Certificate Expiring - {{ .GroupLabels.certificate }}'
      
      - name: 'tunnel-alerts'
        email_configs:
          - to: 'qagiw3@gmail.com'
            from: 'alerts@254carbon.com'
            headers:
              Subject: 'CRITICAL: Tunnel Down - Immediate Action Required'
      
      - name: 'security-alerts'
        email_configs:
          - to: 'qagiw3@gmail.com'
            from: 'security@254carbon.com'
            headers:
              Subject: 'Security: WAF Violations Detected'
      
      - name: 'platform-alerts'
        email_configs:
          - to: 'qagiw3@gmail.com'
            from: 'monitoring@254carbon.com'
            headers:
              Subject: 'Alert: Platform Service Down - {{ .GroupLabels.service }}'
```

### Part 2: Create Prometheus Alert Rules

**File**: Create `k8s/monitoring/prometheus-rules.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  alert-rules.yml: |
    groups:
    - name: cloudflare-alerts
      interval: 30s
      rules:
      # Certificate expiration warning (30 days)
      - alert: CertificateExpirationWarning
        expr: certmanager_certificate_expiration_timestamp_seconds - time() < 30 * 24 * 3600
        for: 1h
        labels:
          severity: warning
          service: certificate-management
        annotations:
          summary: "Certificate expiring in 30 days"
          description: "Certificate {{ $labels.name }} will expire in 30 days"
      
      # Certificate expiration critical (7 days)
      - alert: CertificateExpirationCritical
        expr: certmanager_certificate_expiration_timestamp_seconds - time() < 7 * 24 * 3600
        for: 10m
        labels:
          severity: critical
          service: certificate-management
        annotations:
          summary: "Certificate expiring in 7 days"
          description: "Certificate {{ $labels.name }} will expire in 7 days"
      
      # Tunnel health check
      - alert: TunnelDown
        expr: up{job="cloudflare-tunnel"} == 0
        for: 5m
        labels:
          severity: critical
          service: tunnel
        annotations:
          summary: "Cloudflare tunnel is down"
          description: "Tunnel {{ $labels.instance }} has been down for 5 minutes"
      
      # Service uptime for protected services
      - alert: ServiceDown
        expr: up{job=~"prometheus|alertmanager|vault"} == 0
        for: 5m
        labels:
          severity: critical
          service: "{{ $labels.job }}"
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "Service {{ $labels.job }} has been unreachable for 5 minutes"
    
    - name: waf-alerts
      interval: 1m
      rules:
      # WAF violation threshold (5 violations in 5 minutes)
      - alert: WAFViolationThreshold
        expr: rate(cloudflare_waf_violations_total[5m]) * 300 > 5
        for: 2m
        labels:
          severity: warning
          service: waf
        annotations:
          summary: "WAF violation threshold exceeded"
          description: "More than 5 WAF violations detected in 5 minutes"
      
      # DDoS attack detection
      - alert: DDoSAttackDetected
        expr: rate(cloudflare_http_requests_blocked_total[1m]) > 100
        for: 1m
        labels:
          severity: critical
          service: waf
        annotations:
          summary: "Potential DDoS attack detected"
          description: "Blocking > 100 requests/min from DDoS protection"
```

### Part 3: Configure Cloudflare Analytics

**In Cloudflare Dashboard**:

1. Navigate → **Analytics → Traffic**
2. Set up **Custom Dashboards** for:
   - WAF Events
   - Tunnel Status
   - SSL/TLS Certificate Status
   - Bot Management

3. Create **Scheduled Reports**:
   - Recipients: qagiw3@gmail.com
   - Frequency: Weekly
   - Content: Security Summary + Performance Metrics

---

## Implementation Steps

### Step 1: Apply Updated Ingress Resources

```bash
# Apply vault ingress with auth
kubectl apply -f k8s/ingress/vault-ingress.yaml

# Apply prometheus ingress with auth
kubectl apply -f k8s/ingress/prometheus-ingress.yaml

# Apply alertmanager ingress with auth
kubectl apply -f k8s/ingress/alertmanager-ingress.yaml

# Verify all ingress have auth headers
kubectl get ingress -A | grep -E "vault|prometheus|alertmanager"
kubectl describe ingress vault-ingress -n vault-prod | grep -A 5 "auth-url"
```

### Step 2: Configure Cloudflare Access (Manual)

1. Log into https://dash.cloudflare.com
2. **Zero Trust → Access → Applications**
3. Create 3 applications:
   - vault.254carbon.com
   - prometheus.254carbon.com
   - alertmanager.254carbon.com
4. For each:
   - Add login rule: @254carbon.com OR @project52.org
   - Set session duration: 24 hours
   - Click "Save application"

### Step 3: Enable WAF Rules (Manual)

1. **Security → WAF → Managed Rules**
2. Enable Cloudflare Managed Ruleset
3. Set sensitivity to: Low (Conservative)
4. **Security → WAF → Tools**
5. Add IP whitelist rule for internal IPs: 192.168.1.0/24

### Step 4: Deploy Monitoring Configuration

```bash
# Create monitoring configuration
kubectl apply -f k8s/monitoring/alertmanager-config.yaml
kubectl apply -f k8s/monitoring/prometheus-rules.yaml

# Restart alertmanager to pick up config
kubectl rollout restart deployment alertmanager -n monitoring

# Verify alertmanager is running
kubectl get pods -n monitoring | grep alertmanager
```

### Step 5: Configure Email Alerting

Update alertmanager config with Gmail App Password:

1. Generate Gmail App Password:
   - Google Account → Security → App passwords
   - Generate for "Mail" on "Other (custom name) - Kubernetes Alertmanager"
   - Save the 16-char password

2. Update secret in cluster:
```bash
kubectl create secret generic alertmanager-email \
  --from-literal=auth_password='16-char-app-password' \
  -n monitoring --dry-run=client -o yaml | kubectl apply -f -
```

### Step 6: Test Alert Configuration

```bash
# Test email alert by triggering a test alert
kubectl exec -n monitoring alertmanager-0 -- amtool alert add test_alert severity=critical

# Check alertmanager logs
kubectl logs -n monitoring -l app=alertmanager --tail=50

# Verify email received at qagiw3@gmail.com
```

---

## Verification Checklist

- [ ] **Cloudflare Access**
  - [ ] Vault behind authentication (test login)
  - [ ] Prometheus behind authentication (test login)
  - [ ] AlertManager behind authentication (test login)
  - [ ] Login works with @254carbon.com email
  - [ ] Login works with @project52.org email

- [ ] **WAF Rules**
  - [ ] Managed ruleset enabled in Cloudflare dashboard
  - [ ] Sensitivity set to "Low"
  - [ ] Internal IPs whitelisted (192.168.1.0/24)
  - [ ] Rate limiting configured (100 req/10min)

- [ ] **Monitoring & Alerts**
  - [ ] Alertmanager pod running: `kubectl get pods -n monitoring`
  - [ ] Prometheus scraping alertmanager metrics
  - [ ] Test email alert sent to qagiw3@gmail.com
  - [ ] Alert rules loaded: `kubectl get configmap prometheus-rules -n monitoring`
  - [ ] WAF violations logged: `kubectl logs -n monitoring -l app=prometheus`

- [ ] **Certificate Alerts**
  - [ ] Certificate expiration rules active
  - [ ] Alert fires when cert < 30 days to expiry
  - [ ] Email received for critical (< 7 days) expiry

- [ ] **Tunnel Alerts**
  - [ ] Tunnel health monitoring active
  - [ ] Alert fires if tunnel down > 5 minutes
  - [ ] Recovery alert sent when tunnel restored

- [ ] **End-to-End Testing**
  - [ ] Access vault.254carbon.com → redirects to login
  - [ ] Authenticate with @254carbon.com → access granted
  - [ ] Check alertmanager metrics: `curl -s https://prometheus.254carbon.com/api/v1/query?query=up`
  - [ ] Verify WAF logs in Cloudflare dashboard

---

## Troubleshooting

### Cloudflare Access Not Working

```bash
# Check ingress annotations
kubectl describe ingress prometheus-ingress -n monitoring

# Verify auth-url annotation is set
# Should see: nginx.ingress.kubernetes.io/auth-url: https://qagi.cloudflareaccess.com...

# Reload ingress
kubectl rollout restart deployment nginx-ingress-controller -n ingress-nginx
```

### Alerts Not Sending

```bash
# Check alertmanager config
kubectl get configmap alertmanager-config -n monitoring -o yaml

# Check alertmanager logs
kubectl logs -n monitoring -l app=alertmanager --tail=100 | grep -i email

# Test SMTP connection
kubectl exec -n monitoring alertmanager-0 -- \
  telnet smtp.gmail.com 587
```

### WAF Blocking Legitimate Traffic

```bash
# Check WAF logs in Cloudflare dashboard
# If false positives, adjust sensitivity to "Very Low" or add IP to whitelist

# Whitelist specific IPs:
# Security → WAF → Tools → IP Access Rules
# IP: your-ip, Action: Allow, Priority: High
```

---

## Next Steps (Post-Implementation)

1. **Test Access**: Verify login flows for all protected services
2. **Monitor Alerts**: Watch email for initial alert patterns
3. **Refine Rules**: Adjust WAF sensitivity based on false positives
4. **Document**: Add credentials to your ops runbooks
5. **Backup**: Export Cloudflare configuration

---

## Related Documentation

- Cloudflare Access: https://developers.cloudflare.com/cloudflare-one/access/
- Cloudflare WAF: https://developers.cloudflare.com/waf/
- Prometheus Alerting: https://prometheus.io/docs/alerting/latest/overview/
- Alertmanager Configuration: https://prometheus.io/docs/alerting/latest/configuration/
