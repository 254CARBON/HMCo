# SSO Implementation Checklist (Canonical)

**Project**: 254Carbon SSO Implementation  
**Status**: Phase 1 Complete ✅ | Ready for Phase 2  
**Date**: October 19, 2025  
**Team**: Infrastructure/DevOps

---

## Phase 1: Portal Deployment ✅ COMPLETE

### Core Components
- [x] Next.js portal application created
- [x] Tailwind CSS styling implemented
- [x] Service catalog with 9 services configured
- [x] Docker image built and tested
- [x] Kubernetes deployment created (2 replicas)
- [x] Ingress rules configured
- [x] Portal accessible at https://254carbon.com

### Infrastructure
- [x] Cloudflare Tunnel deployed (cloudflared)
- [x] DNS records created for portal domain
- [x] TLS certificates configured
- [x] NGINX Ingress Controller operational

### Documentation
- [x] Portal README created
- [x] SSO Quickstart guide created
- [x] Implementation summary documented

---

## Phase 2: Cloudflare Access Configuration ⏳ IN PROGRESS

### Prerequisites
- [ ] Cloudflare Teams/Enterprise subscription verified
- [ ] Cloudflare Account ID obtained
- [ ] Cloudflare API token created
- [ ] Portal running in Kubernetes
- [ ] Cloudflare Tunnel connected

### Portal Application in Cloudflare Access
- [ ] Go to Zero Trust Dashboard
- [ ] Create application "254Carbon Portal"
- [ ] Set subdomain to "254carbon"
- [ ] Create "Allow All Users" policy
- [ ] Set session duration to 24 hours
- [ ] Enable HTTP-Only cookies
- [ ] Verify CNAME record created
- [ ] Test portal access with SSO

### Service Applications (9 Total)

#### 1. Vault (vault.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 2 hours
- [ ] CNAME verified

#### 2. MinIO (minio.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 8 hours
- [ ] CNAME verified

#### 3. DolphinScheduler (dolphin.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 12 hours
- [ ] CNAME verified

#### 4. Grafana (grafana.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 24 hours
- [ ] CNAME verified

#### 5. Superset (superset.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 24 hours
- [ ] CNAME verified

#### 6. DataHub (datahub.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 12 hours
- [ ] CNAME verified

#### 7. Trino (trino.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 8 hours
- [ ] CNAME verified

#### 8. Doris (doris.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 8 hours
- [ ] CNAME verified

#### 9. LakeFS (lakefs.254carbon.com)
- [ ] Application created
- [ ] Policy configured: Allow all
- [ ] Session duration: 12 hours
- [ ] CNAME verified

### Audit & Logging
- [ ] Audit logging enabled
- [ ] Email alerts configured
- [ ] Failed auth alerts enabled
- [ ] Policy change alerts enabled

### Phase 2 Verification
- [ ] Portal redirects to Cloudflare login
- [ ] Email authentication works
- [ ] One-time code delivery working
- [ ] Portal accessible after login
- [ ] Session persists
- [ ] Audit logs show activity

---

## Phase 3: Service Integration ⏳ PENDING

### Disable Local Authentication

#### Grafana
- [ ] SSH into cluster or use kubectl
- [ ] Patch grafana configmap
- [ ] Set auth.anonymous.enabled = false
- [ ] Restart Grafana deployment
- [ ] Verify Grafana running (2 pods)
- [ ] Confirm local auth disabled

#### Superset
- [ ] Patch superset deployment
- [ ] Set SUPERSET_DISABLE_LOCAL_AUTH=true
- [ ] Restart Superset deployment
- [ ] Verify Superset running (2 pods)
- [ ] Confirm local auth disabled

### Update NGINX Ingress Rules

#### Prepare Ingress Configuration
- [ ] Get Cloudflare Account ID
- [ ] Create k8s/ingress/ingress-cloudflare-auth.yaml
- [ ] Add Cloudflare Access annotations to all services
- [ ] Replace <ACCOUNT_ID> with actual ID
- [ ] Verify syntax is valid

#### Apply Ingress Rules
- [ ] Apply ingress-cloudflare-auth.yaml
- [ ] Verify all ingress rules created
- [ ] Check ingress status
- [ ] Verify no errors in NGINX

#### Service-Specific Ingress
- [ ] Vault ingress updated
- [ ] MinIO ingress updated
- [ ] DolphinScheduler ingress updated
- [ ] Grafana ingress updated
- [ ] Superset ingress updated
- [ ] DataHub ingress updated
- [ ] Trino ingress updated
- [ ] Doris ingress updated
- [ ] LakeFS ingress updated

### Verify Services Running
- [ ] Check all pods Running status
- [ ] Verify all services accessible internally
- [ ] Check logs for errors

---

## Phase 4: Testing & Validation ⏳ PENDING

### End-to-End Authentication Flow

#### Portal Access Test
- [ ] Visit https://254carbon.com
- [ ] Redirects to Cloudflare login
- [ ] Enter email address
- [ ] Receive one-time code via email
- [ ] Enter code successfully
- [ ] Portal displays all 9 service cards
- [ ] No errors in portal display

#### Session Persistence Test
- [ ] After portal login, visit https://vault.254carbon.com
- [ ] No re-authentication required
- [ ] Visit https://grafana.254carbon.com
- [ ] No re-authentication required
- [ ] Visit all other services without re-login

#### Individual Service Tests
- [ ] https://grafana.254carbon.com → Grafana UI loads
- [ ] https://superset.254carbon.com → Superset UI loads
- [ ] https://vault.254carbon.com → Vault UI loads
- [ ] https://minio.254carbon.com → MinIO console loads
- [ ] https://dolphin.254carbon.com → DolphinScheduler loads
- [ ] https://datahub.254carbon.com → DataHub loads
- [ ] https://trino.254carbon.com → Trino UI loads
- [ ] https://doris.254carbon.com → Doris console loads
- [ ] https://lakefs.254carbon.com → LakeFS UI loads

### Security Testing

#### Unauthorized Access
- [ ] Accessing vault without auth redirects to login
- [ ] Accessing grafana without auth redirects to login
- [ ] Invalid JWT token returns 401

#### Session Management
- [ ] Session expires after configured duration
- [ ] Expired session redirects to login
- [ ] Logout clears all cookies
- [ ] Re-login after logout works

#### Rate Limiting
- [ ] Multiple failed auth attempts are rate-limited
- [ ] Rate limit response shows error
- [ ] Recovery after rate limit window

#### HTTPS Enforcement
- [ ] HTTP requests redirect to HTTPS
- [ ] No mixed content warnings
- [ ] TLS certificates valid

### Audit Log Verification

#### Cloudflare Access Logs
- [ ] Portal access attempts logged
- [ ] Successful authentications logged
- [ ] Failed authentications logged
- [ ] Service access logged
- [ ] Policy evaluations logged
- [ ] Timestamps accurate

#### Log Format
- [ ] User email visible
- [ ] Service name visible
- [ ] Success/failure status visible
- [ ] Timestamp accurate

### Performance Testing

#### Response Time
- [ ] Portal response <100ms
- [ ] Service response <500ms
- [ ] No timeouts

#### Load Testing
- [ ] Portal handles 100 concurrent users
- [ ] Portal handles 1000 concurrent users
- [ ] Service handles 100 concurrent users
- [ ] No degradation under load

#### Resource Usage
- [ ] Portal CPU <500m under load
- [ ] Portal memory <512Mi
- [ ] No pod OOMkill events
- [ ] NGINX memory reasonable

### Compliance Checklist

#### Functionality
- [ ] All 9 services accessible via SSO
- [ ] No service-specific login required
- [ ] Single login session for all services
- [ ] Session persists across subdomains
- [ ] Logout functional

#### Security
- [ ] HTTPS everywhere
- [ ] No insecure headers
- [ ] No exposed credentials
- [ ] Audit logging complete
- [ ] Access policies enforced
- [ ] Rate limiting working

#### Reliability
- [ ] Portal 99.9% uptime
- [ ] No service timeouts
- [ ] Graceful error handling
- [ ] Pod auto-restart working
- [ ] Health checks passing

#### Performance
- [ ] Response time <100ms
- [ ] Handles 1000+ concurrent users
- [ ] No memory leaks
- [ ] CPU usage reasonable

---

## Documentation ⏳ PENDING

### Update Existing Files
- [ ] Update main README.md with SSO section
- [ ] Update k8s/cloudflare/README.md with SSO status
- [ ] Update portal/README.md with SSO configuration
- [ ] Add SSO troubleshooting section

### Create New Files
- [ ] guide.md ✅ Created
- [ ] checklist.md ✅ Creating now
- [ ] User access guide (for team members)

### Remove Old Files
- [ ] Delete PHASE1_COMPLETION_SUMMARY.md (if superseded)
- [ ] Delete old troubleshooting files (if replaced)
- [ ] Archive unnecessary documentation

### Per User Memory
- [ ] Remember: Do NOT write summary/progress markdown
- [ ] Update README.md files only for all services
- [ ] Remove old documentation files after completion

---

## Deployment Operations

### Pre-Deployment
- [ ] Backup current configurations
- [ ] Test changes in staging (if available)
- [ ] Notify team of changes
- [ ] Prepare rollback plan

### During Deployment
- [ ] Monitor pod status
- [ ] Check logs for errors
- [ ] Verify ingress rules
- [ ] Monitor resource usage

### Post-Deployment
- [ ] Verify all services accessible
- [ ] Check audit logs
- [ ] Performance baselines captured
- [ ] Team feedback collected
- [ ] Issues logged and prioritized

---

## Rollback Triggers

Stop and rollback if any of these occur:

- [ ] Portal completely inaccessible for >5 minutes
- [ ] >50% of service authentication failures
- [ ] Performance degradation >50%
- [ ] Security incident or breach
- [ ] Data loss or corruption
- [ ] Critical service down for >10 minutes

---

## Post-Implementation

### Immediate Tasks
- [ ] Celebrate! 🎉
- [ ] Document actual vs. planned timeline
- [ ] Capture lessons learned
- [ ] Gather team feedback
- [ ] Update team documentation
- [ ] Schedule knowledge transfer sessions

### Short Term (1-2 weeks)
- [ ] Monitor for issues
- [ ] Adjust session durations if needed
- [ ] Review audit logs
- [ ] Get team feedback on UX

### Medium Term (1 month)
- [ ] Review security audit results
- [ ] Plan credential rotation
- [ ] Performance review
- [ ] Documentation review

### Long Term (ongoing)
- [ ] Monitor dashboard usage
- [ ] Security audits (quarterly)
- [ ] Performance reviews (monthly)
- [ ] Credential rotation (every 90 days)
- [ ] System updates and maintenance

---

## Sign-Off

### Implementation Team

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Lead | _______________ | ___/___/___ | ________ |
| DevOps | _______________ | ___/___/___ | ________ |
| Security | _______________ | ___/___/___ | ________ |
| QA | _______________ | ___/___/___ | ________ |

### Stakeholders

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Platform Owner | _______________ | ___/___/___ | ________ |
| Security Lead | _______________ | ___/___/___ | ________ |
| Operations Lead | _______________ | ___/___/___ | ________ |

---

## Timeline Summary

| Phase | Target Dates | Actual Dates | Status |
|-------|--------------|--------------|--------|
| Phase 1: Portal | 2-3 days | ✅ 2024-10-19 | Complete |
| Phase 2: Access | 1-2 hours | ___________ | In Progress |
| Phase 3: Integration | 2-3 days | ___________ | Pending |
| Phase 4: Testing | 1-2 days | ___________ | Pending |
| **Total** | **4-7 days** | _________ | **In Progress** |

---

**Document Version**: 1.0  
**Created**: October 19, 2025  
**Last Updated**: October 19, 2025  
**Next Review**: October 26, 2025
