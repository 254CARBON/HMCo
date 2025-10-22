# SSO Implementation Quick Reference Card (Canonical)

**Status**: Phase 1 ✅ Complete | Phase 2-4 ⏳ Ready to Begin  
**Estimated Time Remaining**: 4-7 days  
**Start Date**: October 19, 2025

---

## Phase 2: Cloudflare Access Setup (1-2 hours)

### Quick Steps

1. **Enable Cloudflare Teams**
   - Dashboard → My Profile → Billing → Verify Teams subscription
   - Zero Trust → Dashboard → Accept terms

2. **Create Portal Application**
   - Zero Trust → Access → Applications → Add application → Self-hosted
   - Name: `254Carbon Portal`
   - Subdomain: `254carbon`
   - Domain: `cloudflareaccess.com`
   - Policy: Allow → Everyone
   - Session: 24 hours

3. **Create 9 Service Applications**
   - Repeat above for: vault, minio, dolphin, grafana, superset, datahub, trino, doris, lakefs
   - Session durations: Vault (2h), MinIO/Trino/Doris (8h), DolphinScheduler/DataHub/LakeFS (12h), Grafana/Superset (24h)

4. **Enable Audit Logging**
   - Zero Trust → Access → Logs → Verify activity
   - Settings → Notifications → Enable alerts

5. **Test Portal Access**
   - Visit https://254carbon.com
   - Should redirect to Cloudflare login
   - Enter email, receive code, login
   - See portal with 9 service cards

---

## Phase 3: Service Integration (2-3 days)

### Quick Steps

1. **Disable Grafana Local Auth**
   ```bash
   kubectl -n monitoring patch configmap grafana-config --type merge -p '{
     "data": {"grafana.ini": "[auth.anonymous]\nenabled = false\n[users]\nauto_assign_org_role = Viewer"}
   }'
   kubectl rollout restart deployment/grafana -n monitoring
   ```

2. **Disable Superset Local Auth**
   ```bash
   kubectl -n data-platform set env deployment/superset SUPERSET_DISABLE_LOCAL_AUTH=true
   kubectl rollout restart deployment/superset -n data-platform
   ```

3. **Update Ingress Rules**
   - Get your Cloudflare Account ID: https://dash.cloudflare.com/zero-trust/settings/general
   - Update k8s/ingress/ingress-cloudflare-auth.yaml
   - Replace `<ACCOUNT_ID>` with actual ID
   - `kubectl apply -f k8s/ingress/ingress-cloudflare-auth.yaml`

---

## Phase 4: Testing (1-2 days)

### Quick Test Suite

```bash
# Test 1: Portal Access
curl -v https://254carbon.com
# Should redirect to Cloudflare login

# Test 2: Manual Testing
# 1. Open https://254carbon.com in browser
# 2. Complete email authentication
# 3. See portal with 9 services
# 4. Click each service - should work without re-login

# Test 3: Security Test
curl -v https://vault.254carbon.com
# Should require authentication

# Test 4: Performance
time curl https://254carbon.com -o /dev/null -s
# Should be <100ms

# Test 5: Session Persistence
# Login once, visit all 9 services without re-login
```

### Success Criteria ✅

- [ ] Portal accessible at 254carbon.com
- [ ] All 9 services accessible via SSO
- [ ] Single login for all services
- [ ] Session persists across services
- [ ] Unauthorized access denied
- [ ] Response time <100ms
- [ ] 99.9% availability
- [ ] Audit logs complete

---

## Key Files to Reference

| File | Purpose | Phase |
|------|---------|-------|
| `guide.md` | **📌 Main guide** - Detailed step-by-step | All |
| `checklist.md` | **✅ Tracking** - All tasks and verification | All |
| `quickstart.md` | Quick commands and overview | All |
| `k8s/cloudflare/CLOUDFLARE_SSO_SETUP.md` | Detailed Cloudflare configuration | Phase 2 |
| `portal/README.md` | Portal application details | Phase 1 ✅ |
| `k8s/cloudflare/README.md` | Cloudflare Tunnel documentation | Reference |

---

## Critical Information

### Cloudflare Account ID
- Go to: https://dash.cloudflare.com/zero-trust/settings/general
- Copy "Account ID" value
- Use in Phase 3 ingress rules

### Session Durations (Configure per service)
- **Vault**: 2 hours (sensitive)
- **MinIO**: 8 hours
- **Trino**: 8 hours
- **Doris**: 8 hours
- **DolphinScheduler**: 12 hours
- **DataHub**: 12 hours
- **LakeFS**: 12 hours
- **Grafana**: 24 hours
- **Superset**: 24 hours

### Service Ports (For ingress configuration)
- Grafana: 3000
- Superset: 8088
- Vault: 8200
- MinIO: 9001
- DolphinScheduler: 12345
- DataHub: 3000
- Trino: 8080
- Doris: 8030
- LakeFS: 8000

---

## Common Issues & Quick Fixes

### Issue: Portal shows 502 Bad Gateway
```bash
kubectl rollout restart deployment/portal -n data-platform
kubectl get pods -n data-platform -l app=portal
```

### Issue: Cloudflare Access login loop
```bash
# Verify application exists in Cloudflare UI
# Check policy is enabled
# Restart tunnel
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel
```

### Issue: Services show 401 Unauthorized
```bash
# Verify ingress annotations
kubectl get ingress vault-ingress -n data-platform -o yaml | grep auth-url
# Check service is running
kubectl get pods -n data-platform -l app=vault
```

### Issue: Session not persisting across services
```bash
# Verify all services use same Account ID in auth-url
kubectl get ingress -A -o yaml | grep cloudflareaccess.com | sort | uniq
# Should all show the same domain
```

---

## Verification Commands

```bash
# Check portal deployment
kubectl get pods -n data-platform -l app=portal

# Check all services running
kubectl get svc -A | grep -E "grafana|vault|minio|superset"

# Check ingress rules
kubectl get ingress -A | grep 254carbon

# Check Cloudflare tunnel
kubectl get pods -n cloudflare-tunnel

# View portal logs
kubectl logs -n data-platform -l app=portal -f --tail=50

# Check NGINX logs for auth errors
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -f | grep -i auth
```

---

## Timeline

**Phase 2 (Cloudflare Setup)**: 1-2 hours hands-on time
- Most of this is in Cloudflare UI (can be done while other tasks run)

**Phase 3 (Service Integration)**: 2-3 days
- Disabling local auth in services
- Updating all 9 ingress rules
- Testing each service configuration

**Phase 4 (Testing & Validation)**: 1-2 days
- End-to-end testing
- Security testing
- Performance validation

**Total Remaining**: 4-7 days

**Estimated Completion**: October 26-30, 2025

---

## Documentation to Update When Complete

After Phase 4 completion:

1. Update `README.md` with:
   - SSO access instructions for users
   - Portal login process
   - Service availability
   - Troubleshooting section

2. Remove old files:
   - Delete superseded documentation
   - Archive Phase 1 summary (content moved to main README)

3. Update service READMEs:
   - Each service README updated to mention SSO access
   - Remove local auth instructions

---

## Support & Escalation

**For Cloudflare Issues**:
- Cloudflare Dashboard: https://dash.cloudflare.com
- Zero Trust Documentation: https://developers.cloudflare.com/cloudflare-one/
- Check `k8s/cloudflare/README.md` for troubleshooting

**For Kubernetes Issues**:
- Check pod logs: `kubectl logs -n <namespace> <pod-name>`
- Check ingress: `kubectl describe ingress <name> -n <namespace>`
- Check services: `kubectl get svc -A | grep <service>`

**For Portal Issues**:
- Portal README: `portal/README.md`
- Check deployment: `kubectl get deployment portal -n data-platform`
- View logs: `kubectl logs -n data-platform -l app=portal -f`

---

## Next Actions

1. **Immediate** (Today):
   - Review `guide.md`
   - Understand each phase requirements
   - Prepare Cloudflare Account

2. **Phase 2** (Tomorrow - 1-2 hours):
   - Follow Phase 2 section in guide
   - Create applications in Cloudflare UI
   - Test portal access

3. **Phase 3** (Next 2-3 days):
   - Follow Phase 3 section in guide
   - Disable local authentication
   - Update ingress rules
   - Verify services

4. **Phase 4** (Next 1-2 days):
   - Follow Phase 4 section in guide
   - Run complete test suite
   - Verify all success criteria
   - Sign-off and deployment

---

## Success! 🎉

Once Phase 4 is complete:
- ✅ Portal operational
- ✅ 9 services accessible via SSO
- ✅ Single login for all
- ✅ Audit logging enabled
- ✅ Security tested
- ✅ Performance verified
- ✅ Team trained
- ✅ Documentation updated

---

**Created**: October 19, 2025  
**Version**: 1.0  
**Last Updated**: October 19, 2025
