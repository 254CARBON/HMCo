# Security & Monitoring Implementation - Quick Start

## 🚀 Fast Track Deployment (90 minutes total)

### Phase 1: Automated Deployment (5 minutes)
```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/deploy-security-monitoring.sh
```

This automatically deploys:
- ✅ Vault ingress with Cloudflare Access
- ✅ Prometheus ingress with authentication
- ✅ AlertManager ingress with authentication
- ✅ AlertManager configuration (email routing)
- ✅ Prometheus alert rules (18+ rules)

### Phase 2: Gmail Setup (5 minutes)
1. Go to: https://myaccount.google.com/security/apppasswords
2. App: `Mail`, Device: `Your Device`
3. Generate password (copy the 16 characters)

### Phase 3: Update AlertManager (5 minutes)
```bash
kubectl edit configmap alertmanager-config -n monitoring
# Find: auth_password: '${GMAIL_APP_PASSWORD}'
# Replace with: auth_password: 'YOUR_16_CHAR_PASSWORD'

kubectl rollout restart deployment alertmanager -n monitoring
```

### Phase 4: Cloudflare Access Setup (15 minutes)
**In Cloudflare Dashboard** → https://dash.cloudflare.com

1. Zero Trust → Access → Applications
2. Create 4 applications:
   - `vault.254carbon.com`
   - `prometheus.254carbon.com`
   - `alertmanager.254carbon.com`
   - `kong.254carbon.com`

3. For each application:
   - Session duration: **24 hours**
   - Add rule: **Emails ending with @254carbon.com**
   - Add rule: **Emails ending with @project52.org**
   - Action: **Allow**
   - Save

### Phase 5: Enable WAF (5 minutes)
**In Cloudflare Dashboard** → Security tab

1. **WAF → Managed Rules**
   - Enable: Cloudflare Managed Ruleset
   - Sensitivity: **Low** (Conservative)

2. **WAF → Tools → IP Access Rules**
   - IP: `192.168.1.0/24`
   - Action: **Allow**
   - Priority: **High**

3. **Rate Limiting**
   - Path: `/api/*`
   - Threshold: **100 requests per 10 minutes**
   - Block for: 15 minutes

### Phase 6: Test & Verify (10 minutes)
```bash
# Test Kubernetes deployment
kubectl get ingress -A | grep -E "vault|prometheus|alertmanager"
kubectl describe ingress prometheus-ingress -n monitoring | grep auth-url

# Test Cloudflare Access
# 1. Open: https://vault.254carbon.com
# 2. Should redirect to Cloudflare login
# 3. Login with @254carbon.com or @project52.org
# 4. Verify access granted

# Test Email Alerts
# 1. Wait 5 minutes for alerts to stabilize
# 2. Check email: qagiw3@gmail.com
# 3. Should receive test alerts
```

---

## 📋 What Was Deployed

### Files Created
- `k8s/ingress/vault-ingress.yaml` - Vault with Zero Trust
- `k8s/ingress/prometheus-ingress.yaml` - Updated with auth
- `k8s/ingress/alertmanager-ingress.yaml` - Updated with auth
- `k8s/monitoring/alertmanager-config.yaml` - Email routing
- `k8s/monitoring/prometheus-alert-rules.yaml` - 18+ alert rules
- `scripts/deploy-security-monitoring.sh` - Automation script
- `CLOUDFLARE_SECURITY_MONITORING_SETUP.md` - Complete guide

### Security Layers Implemented
1. **Cloudflare Access** - Zero Trust authentication
2. **Cloudflare WAF** - Advanced threat protection
3. **Monitoring & Alerts** - Proactive incident detection

### Protected Services
- Vault (port 8200)
- Prometheus (port 9090)
- AlertManager (port 9093)
- Kong Admin (port 8001)

### Alert Rules (18 total)
- Certificate expiration (30 days & 7 days)
- Tunnel health (down, latency, instability)
- Service health (Vault, Prometheus, AlertManager)
- WAF violations (threshold, DDoS, SQL injection, XSS)
- Infrastructure (memory, storage)

---

## ✅ Verification

### Quick Status Check
```bash
# Ingress resources
kubectl get ingress -A | grep -E "vault|prometheus|alertmanager"

# Auth headers present
kubectl describe ingress prometheus-ingress -n monitoring | grep -A 2 "auth-url"

# AlertManager running
kubectl get pods -n monitoring -l app=alertmanager

# Prometheus alert rules loaded
kubectl get cm prometheus-alert-rules -n monitoring
```

### Full Verification Checklist
- [ ] All ingress resources created
- [ ] Auth headers configured
- [ ] AlertManager connected to Gmail
- [ ] Cloudflare Access applications created
- [ ] WAF Managed Ruleset enabled
- [ ] IP whitelist configured
- [ ] Test login successful
- [ ] Alert email received

---

## 🔑 Key Credentials & Configuration

**Alert Recipient:** qagiw3@gmail.com
**Allowed Domains:** @254carbon.com, @project52.org
**Session Duration:** 24 hours
**WAF Sensitivity:** Conservative (Low)
**IP Whitelist:** 192.168.1.0/24 (internal cluster)
**Rate Limit:** 100 req/10 min on `/api/*`

---

## 📚 Documentation

- **Complete Guide:** `CLOUDFLARE_SECURITY_MONITORING_SETUP.md`
- **Quick Reference:** `README_CLOUDFLARE.md`
- **Deployment Steps:** `DEPLOYMENT_GUIDE_CLOUDFLARE.md`
- **This File:** `SECURITY_MONITORING_QUICKSTART.md`

---

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Login redirects but doesn't load | Check AlertManager logs: `kubectl logs -n monitoring -l app=alertmanager` |
| Alerts not sending | Verify Gmail password is correct and AlertManager restarted |
| WAF blocks legitimate traffic | Adjust sensitivity to "Very Low" or add IP to whitelist |
| Can't generate Gmail app password | Enable 2FA on Gmail account first |
| Certificate error on browser | Wait 5-10 minutes for DNS propagation |

---

## 📞 Support

For detailed information, see `CLOUDFLARE_SECURITY_MONITORING_SETUP.md`

**External Resources:**
- Cloudflare Access: https://developers.cloudflare.com/cloudflare-one/access/
- Cloudflare WAF: https://developers.cloudflare.com/waf/
- Prometheus Alerting: https://prometheus.io/docs/alerting/

---

**Status:** ✅ Ready for Deployment
**Time Estimate:** 90 minutes (45 min automated + 45 min manual)
**Last Updated:** October 25, 2025
