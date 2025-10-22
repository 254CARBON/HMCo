# SSO Validation & Testing Guide (Canonical)

Complete testing procedures for 254Carbon SSO implementation via Cloudflare Access (team: qagi).

## Pre-Testing Checklist

Before testing, verify these prerequisites are complete:

- [ ] Phase 2: Cloudflare Access applications created (portal + 9 services)
- [ ] Phase 3: Ingress rules applied with auth annotations
- [ ] Phase 3: Tunnel credentials updated
- [ ] Phase 3: Grafana local auth disabled
- [ ] Phase 3: Superset local auth disabled
- [ ] All services pods running (kubectl get pods -A | grep -E "portal|grafana|superset|vault|minio|dolphin")
- [ ] Ingress rules deployed (kubectl get ingress -A | grep 254carbon)
- [ ] Cloudflare tunnel connected (kubectl logs -n cloudflare-tunnel -f | grep "Registered tunnel")

## Phase 4.1: Authentication Flow Testing

### Test 1.1: Portal Redirect to Login

**Objective**: Verify portal redirects unauthenticated users to Cloudflare login

**Steps**:
1. Open new private/incognito browser window
2. Navigate to `https://254carbon.com`
3. Observe redirect to Cloudflare Access login page

**Expected Result**:
- Redirect to `https://qagi.cloudflareaccess.com/cdn-cgi/access/login`
- Login page displays email field
- Status: 200 OK

**Verification Command**:
```bash
curl -v https://254carbon.com 2>&1 | grep -i "location"
# Should show redirect location to qagi.cloudflareaccess.com
```

### Test 1.2: Email OTP Authentication

**Objective**: Verify email OTP delivery and validation works

**Steps**:
1. At Cloudflare login page, enter your email address
2. Click "Send me a code"
3. Check email for one-time code
4. Enter code at prompt
5. Should be redirected to portal

**Expected Result**:
- Email received with 6-digit code within 2 minutes
- Code accepted without errors
- Redirected to portal home page
- Portal displays all 9 service cards

**Validation Checklist**:
- [ ] Email received with valid OTP
- [ ] Code accepted on first attempt
- [ ] Portal loads completely
- [ ] No JavaScript console errors
- [ ] All service cards visible

### Test 1.3: Session Creation

**Objective**: Verify session token is created after authentication

**Steps**:
1. After successful login, open browser dev tools (F12)
2. Go to Application → Cookies
3. Look for `CF-Access-JWT-Assertion` cookie
4. Verify cookie properties:
   - Path: `/`
   - Domain: `.254carbon.com`
   - Secure: Yes
   - HttpOnly: Yes
   - SameSite: Lax

**Expected Result**:
- Session cookie present with correct properties
- Cookie value is a valid JWT token
- Expiration matches configured session duration

### Test 1.4: Session Persistence Across Subdomains

**Objective**: Verify single session works across all service subdomains

**Steps** (after authenticating at portal):
1. Visit `https://grafana.254carbon.com`
   - Should show Grafana dashboard (no login required)
2. Visit `https://vault.254carbon.com`
   - Should show Vault UI (no login required)
3. Visit `https://superset.254carbon.com`
   - Should show Superset UI (no login required)
4. Repeat for remaining services

**Expected Result**:
- No re-authentication required for any service
- Session cookie shared across all subdomains
- All services accessible immediately

**Verification Command**:
```bash
# Test session persistence
for service in grafana vault superset minio dolphin datahub trino doris lakefs; do
  echo "Testing $service.254carbon.com"
  curl -s -b "CF-Access-JWT-Assertion=<your-token>" https://$service.254carbon.com | head -50
done
```

## Phase 4.2: Service Access Testing

### Test 2.1: Unauthenticated Access Blocked

**Objective**: Verify unauthenticated users cannot access protected services

**Steps**:
1. Open new private window (different session)
2. Try to access each service directly:
   - `https://vault.254carbon.com`
   - `https://grafana.254carbon.com`
   - `https://superset.254carbon.com`

**Expected Result**:
- Redirect to Cloudflare login page
- 401 Unauthorized or redirect response
- No service content accessible

**Verification Command**:
```bash
# Test without session token
curl -v https://vault.254carbon.com 2>&1 | grep -E "location|401|403"
# Should show redirect to login or error
```

### Test 2.2: Service Functionality Post-SSO

**Objective**: Verify services work correctly with SSO integration

**For Grafana**:
- [ ] Dashboards load
- [ ] Can select time ranges
- [ ] Can view metrics
- [ ] No authentication errors in logs

**For Superset**:
- [ ] Can access dashboards
- [ ] Can run queries
- [ ] Can create new charts
- [ ] Database connections working

**For Vault**:
- [ ] Can view secrets (if authorized)
- [ ] Can view audit logs
- [ ] Can access policy management (if authorized)

**For MinIO**:
- [ ] Can browse buckets
- [ ] Can view object details
- [ ] Can see usage stats

**For DolphinScheduler**:
- [ ] Can see workflows
- [ ] Can view execution logs
- [ ] Can trigger runs (if authorized)

**For DataHub**:
- [ ] Can search data assets
- [ ] Can view lineage
- [ ] Can add tags/descriptions

**For Trino**:
- [ ] Can see catalog
- [ ] Can execute queries
- [ ] Can view query history

**For Doris**:
- [ ] Can see databases/tables
- [ ] Can view query execution plans
- [ ] Can monitor cluster health

**For LakeFS**:
- [ ] Can see repositories
- [ ] Can view branches
- [ ] Can merge/commit changes

### Test 2.3: Token Validation

**Objective**: Verify service-level JWT token validation

**Steps**:
1. Get session token from cookie: `CF-Access-JWT-Assertion`
2. Make request with token in header:
   ```bash
   curl -H "CF-Access-JWT-Assertion: <token>" https://vault.254carbon.com
   ```
3. Verify service receives and processes token

**Expected Result**:
- Service responds with 200 OK
- Service content loaded
- No authentication errors

## Phase 4.3: Security Validation

### Test 3.1: Invalid Token Rejection

**Objective**: Verify invalid tokens are rejected

**Steps**:
1. Construct invalid JWT token
2. Send request with invalid token:
   ```bash
   curl -H "CF-Access-JWT-Assertion: invalid.token.here" \
        https://vault.254carbon.com
   ```

**Expected Result**:
- Response: 401 Unauthorized or 403 Forbidden
- Redirect to login page
- No service content exposed

### Test 3.2: Expired Token Handling

**Objective**: Verify expired sessions require re-authentication

**Steps**:
1. Note session cookie expiration time
2. Wait until after expiration
3. Try to access service
4. Browser should prompt for re-authentication

**Expected Result**:
- Cookie expired or removed
- Redirect to login page
- Must enter email and new OTP

### Test 3.3: Rate Limiting

**Objective**: Verify rate limiting on authentication endpoint

**Steps**:
```bash
# Attempt multiple failed authentications
for i in {1..20}; do
  curl -X POST https://qagi.cloudflareaccess.com/cdn-cgi/access/login \
       -d "email=invalid@example.com" &
done
```

**Expected Result**:
- After N failed attempts, get rate limit response (429 Too Many Requests)
- After backoff period, can retry

### Test 3.4: HTTPS Enforcement

**Objective**: Verify all HTTP traffic redirects to HTTPS

**Steps**:
```bash
# Test HTTP redirect
curl -v http://254carbon.com 2>&1 | grep -i "location"
curl -v http://grafana.254carbon.com 2>&1 | grep -i "location"

# All should redirect to https://
```

**Expected Result**:
- 301/302 redirect from HTTP to HTTPS
- Location header shows https:// URL
- Final response over HTTPS only

### Test 3.5: CORS & Security Headers

**Objective**: Verify security headers are present

**Steps**:
```bash
curl -v https://254carbon.com 2>&1 | grep -E "Strict-Transport-Security|X-Frame-Options|X-Content-Type|Content-Security"
```

**Expected Result**:
- Strict-Transport-Security header present
- X-Frame-Options: DENY or SAMEORIGIN
- X-Content-Type-Options: nosniff
- Content-Security-Policy configured

## Phase 4.4: Audit Logging Verification

### Test 4.1: Access Logs in Cloudflare Dashboard

**Objective**: Verify access attempts are logged in Cloudflare

**Steps**:
1. Log in to Cloudflare Dashboard
2. Go to Zero Trust → Access → Logs
3. Look for recent entries

**Expected Result**:
- Login attempts logged with timestamp
- Successful authentications shown
- Failed attempts visible
- Service access recorded

**Log Entry Format**:
- User email
- Service name
- Action (allow/deny)
- Timestamp
- Source IP (if available)

### Test 4.2: Service Access Logs

**Objective**: Verify service access is recorded

**Steps**:
1. Make authenticated request to service
2. Check Cloudflare logs for access record
3. Verify timestamp and user email match

**Expected Result**:
- Each service access logged
- Timing shows access through Cloudflare tunnel
- Multiple services show single user session

### Test 4.3: Failed Authentication Logging

**Objective**: Verify failed authentication attempts are logged

**Steps**:
1. Try to login with wrong email
2. Check logs for denied access
3. Verify reason code

**Expected Result**:
- Failed attempt logged with reason
- Policy denial recorded
- Multiple failures visible

## Phase 4.5: Performance Testing

### Test 5.1: Portal Response Time

**Objective**: Verify portal loads quickly

**Steps**:
```bash
# Measure portal response time
time curl -s https://254carbon.com -o /dev/null

# Run multiple times for average
for i in {1..10}; do
  curl -s https://254carbon.com -o /dev/null
done
```

**Expected Result**:
- Average response time < 100ms
- No timeout errors
- Consistent performance

### Test 5.2: Service Response Time

**Objective**: Verify service access latency is acceptable

**Steps**:
```bash
# Test each service response time
for service in grafana vault superset minio dolphin datahub trino doris lakefs; do
  echo "Testing $service..."
  time curl -s -H "CF-Access-JWT-Assertion: <token>" \
           https://$service.254carbon.com -o /dev/null
done
```

**Expected Result**:
- Response time < 500ms per service
- No connection timeouts
- Consistent across multiple attempts

### Test 5.3: Load Testing

**Objective**: Verify system handles concurrent users

**Prerequisites**: Apache Bench (`ab`) or similar tool

**Steps**:
```bash
# Test 100 concurrent requests
ab -n 100 -c 10 https://254carbon.com

# Test 1000 requests
ab -n 1000 -c 50 https://254carbon.com
```

**Expected Result**:
- Requests/sec > 100
- Failed requests < 1%
- Response time variance < 50%
- CPU/memory not maxed out

## Phase 4.6: Complete Testing Checklist

### Functionality
- [ ] Portal accessible at https://254carbon.com
- [ ] Portal redirects to Cloudflare login when not authenticated
- [ ] Email OTP authentication works
- [ ] Session token created after successful auth
- [ ] Session persists across service subdomains
- [ ] All 9 services accessible with SSO
- [ ] No service-specific login required
- [ ] Service functionality intact post-SSO

### Security
- [ ] Unauthenticated access blocked (401/403)
- [ ] Invalid tokens rejected
- [ ] Expired sessions require re-authentication
- [ ] Rate limiting prevents brute force
- [ ] HTTPS enforced (no HTTP)
- [ ] Security headers present
- [ ] Cloudflare DDoS protection active
- [ ] WAF rules configured (if applicable)

### Audit & Compliance
- [ ] All access logged in Cloudflare
- [ ] Successful logins recorded
- [ ] Failed attempts logged
- [ ] Service access audit trail complete
- [ ] Logs show user email, service, timestamp
- [ ] Access denied events recorded

### Performance
- [ ] Portal response < 100ms
- [ ] Service response < 500ms
- [ ] Handles 100+ concurrent users
- [ ] Load test success rate > 99%
- [ ] No memory leaks (monitor pod memory)
- [ ] No CPU spikes under load

### Reliability
- [ ] Portal 99.9% uptime over 24 hours
- [ ] No unexpected pod restarts
- [ ] Health checks passing
- [ ] No service timeouts
- [ ] Graceful error handling
- [ ] Tunnel remains connected

## Test Execution Template

Use this template to document test execution:

```
Test Date: _______________
Tester: ___________________
Test Environment: __________

Test Results:

Test 1.1 (Portal Redirect):
- [ ] Pass
- [ ] Fail
- Notes: ________________

Test 1.2 (Email OTP):
- [ ] Pass
- [ ] Fail
- Notes: ________________

[... continue for all tests ...]

Overall Status: ___________
Issues Found: _____________
Recommendations: __________
Sign-off: __________________
```

## Troubleshooting During Testing

### Portal shows "Bad Gateway"
1. Check portal pods: `kubectl get pods -n data-platform -l app=portal`
2. Check logs: `kubectl logs -n data-platform -l app=portal`
3. Verify ingress: `kubectl get ingress -n data-platform`

### Login loop / keeps redirecting
1. Verify Cloudflare Access application exists
2. Check policy is enabled
3. Verify CNAME record points to tunnel

### Services show 401/403
1. Verify ingress auth annotations are correct
2. Check ACCOUNT_ID is correctly set
3. Verify Cloudflare Access application exists for service

### Session not persisting
1. Verify cookies have correct domain (`.254carbon.com`)
2. Check cookie is HttpOnly and Secure
3. Verify tunnel credentials are correct

### Performance issues
1. Monitor pod resource usage: `kubectl top pods -n data-platform`
2. Check NGINX logs: `kubectl logs -n ingress-nginx`
3. Check tunnel logs: `kubectl logs -n cloudflare-tunnel -f`

## Success Criteria

All items must be checked ✓ before declaring SSO implementation complete:

- ✓ All 10 Cloudflare Access applications configured
- ✓ Portal accessible and redirects correctly
- ✓ Email OTP authentication working
- ✓ All 9 services accessible via SSO
- ✓ Session persists across services
- ✓ Unauthenticated access blocked
- ✓ Audit logs complete
- ✓ Performance meets targets
- ✓ No security vulnerabilities
- ✓ Documentation updated
- ✓ Team trained on SSO system

## Sign-Off

- [ ] QA/Tester Sign-off: _________________ Date: _______
- [ ] Operations Sign-off: ________________ Date: _______
- [ ] Security Review: ____________________ Date: _______
- [ ] Platform Owner Approval: _____________ Date: _______

**SSO Implementation Status: COMPLETE**
