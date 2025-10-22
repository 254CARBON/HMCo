# SSO Implementation Quick Start (Canonical)

Get the 254Carbon SSO portal and authentication system up and running.

---

## Phase 1: Portal Deployment (30 minutes)

### 1.1 Build Docker Image

```bash
cd /home/m/tff/254CARBON/HMCo/portal

# Build image
docker build -t 254carbon-portal:latest .

# Optional: Tag and push to registry
docker tag 254carbon-portal:latest your-registry/254carbon-portal:latest
docker push your-registry/254carbon-portal:latest
```

### 1.2 Deploy to Kubernetes

```bash
# Deploy portal deployment and service
kubectl apply -f k8s/ingress/portal-deployment.yaml

# Deploy ingress rules
kubectl apply -f k8s/ingress/portal-ingress.yaml
```

### 1.3 Verify Deployment

```bash
# Check pods are running
kubectl get pods -n data-platform -l app=portal
# Should show 2 pods in Running state

# Check service
kubectl get svc -n data-platform | grep portal

# View logs
kubectl logs -n data-platform -l app=portal -f
```

### 1.4 Access the Portal

Once deployed, access the portal at:
- https://254carbon.com
- https://www.254carbon.com
- https://portal.254carbon.com

(Note: Will show connection error until Cloudflare Access is configured)

---

## Phase 2: Cloudflare Access Configuration (1-2 hours)

### 2.1 Enable Cloudflare Teams

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Go to **My Profile** → **Billing**
3. Verify **Cloudflare Teams** subscription is active
4. Navigate to **Zero Trust** → **Dashboard**

### 2.2 Create Portal Application

1. In Cloudflare Zero Trust, go to **Access** → **Applications**
2. Click **Add an Application** → **Self-hosted**
3. Fill in:
   - **Application name**: 254Carbon Portal
   - **Subdomain**: 254carbon
   - **Domain**: cloudflareaccess.com
   - **Session duration**: 24 hours
4. Click **Next**

### 2.3 Add Portal Policy

1. Click **+ Add a policy**
2. Create:
   - **Name**: Allow All Users
   - **Decision**: Allow
   - **Include**: Email Suffix → your-domain.com
3. Click **Save**

### 2.4 Create Service Applications

For each protected service, repeat steps 2.2-2.3:

**Services to configure:**
- Grafana (grafana.254carbon.com)
- Superset (superset.254carbon.com)
- Vault (vault.254carbon.com) - 1-2 hour session
- MinIO (minio.254carbon.com) - 8 hour session
- DolphinScheduler (dolphin.254carbon.com) - 12 hour session
- DataHub (datahub.254carbon.com)
- Trino (trino.254carbon.com)
- Doris (doris.254carbon.com)
- LakeFS (lakefs.254carbon.com)

### 2.5 Test Portal Access

```bash
# Test portal is now protected by Cloudflare Access
curl -v https://254carbon.com
# Should redirect to Cloudflare Access login

# After authentication, should show portal with service cards
```

---

## Phase 3: Service Integration (Optional - 2-3 days)

### 3.1 Disable Service-Specific Authentication

For each service that has built-in auth, disable it to use SSO only:

**Grafana:**
```bash
kubectl -n monitoring patch configmap grafana-config --type merge -p '
data:
  "grafana.ini": |
    [auth.anonymous]
    enabled = false
'
```

**Superset:**
```bash
kubectl -n data-platform set env deployment/superset \
  SUPERSET_LOAD_EXAMPLES=no
```

### 3.2 Configure JWT Validation

For services that support it, enable JWT token validation:

1. Get JWT public key from Cloudflare
2. Configure service to validate JWT from `CF-Access-JWT-Assertion` header
3. Extract user email from JWT payload

---

## Testing Checklist

- [ ] Portal loads at https://254carbon.com
- [ ] Cloudflare Access login redirects show
- [ ] After login, portal displays service cards
- [ ] All 9 services are visible
- [ ] Clicking service links redirects correctly
- [ ] Session persists across services (same login)
- [ ] Logout clears session
- [ ] Audit logs show access attempts in Cloudflare

---

## Troubleshooting

### Portal Returns 502 Bad Gateway

```bash
# Check if pods are running
kubectl get pods -n data-platform -l app=portal

# View pod logs
kubectl logs -n data-platform -l app=portal -f

# Redeploy if needed
kubectl rollout restart deployment/portal -n data-platform
```

### Cloudflare Access Not Protecting Portal

1. Verify application exists in Cloudflare Zero Trust
2. Check policy is enabled
3. Confirm CNAME record exists:
   ```bash
   nslookup 254carbon.cloudflareaccess.com
   ```

### Can't Access Services After Login

1. Verify service applications exist in Cloudflare
2. Check ingress rules:
   ```bash
   kubectl get ingress -A | grep 254carbon
   ```
3. Verify service is running:
   ```bash
   kubectl get svc -n data-platform | grep <service>
   ```

---

## Files Reference

| File | Purpose |
|------|---------|
| `portal/` | Next.js application source code |
| `portal/Dockerfile` | Container image definition |
| `k8s/ingress/portal-deployment.yaml` | Kubernetes deployment and service |
| `k8s/ingress/portal-ingress.yaml` | Ingress rules for 254carbon.com |
| `k8s/cloudflare/CLOUDFLARE_SSO_SETUP.md` | Detailed Cloudflare configuration |
| `portal/README.md` | Portal documentation |
| `overview.md` | Complete implementation details |

---

## Key Commands

```bash
# Check portal status
kubectl get pods -n data-platform -l app=portal

# View portal logs
kubectl logs -n data-platform -l app=portal -f --tail=100

# Restart portal
kubectl rollout restart deployment/portal -n data-platform

# Check portal service
kubectl get svc portal -n data-platform

# Check ingress
kubectl get ingress -n data-platform | grep portal

# Delete portal (if needed)
kubectl delete deployment portal -n data-platform
kubectl delete svc portal -n data-platform
kubectl delete ingress portal-ingress -n data-platform
```

---

## Next Steps

1. ✅ Phase 1: Portal deployment
2. ⏳ Phase 2: Cloudflare Access setup
3. ⏳ Phase 3: Service integration (optional)
4. ⏳ Phase 4: Testing and validation

---

## Support

For detailed information, see:
- `portal/README.md` - Portal setup and configuration
- `k8s/cloudflare/CLOUDFLARE_SSO_SETUP.md` - SSO setup guide
- `README.md` - Main documentation
- `overview.md` - Complete implementation details

---

## Estimated Timeline

- **Phase 1 (Portal)**: 30 minutes - 1 hour
- **Phase 2 (Cloudflare)**: 1-2 hours
- **Phase 3 (Integration)**: 2-3 days (optional)
- **Phase 4 (Testing)**: 1-2 days
- **Total**: 3-7 days depending on depth

**Total effort to full SSO**: ~4-6 hours hands-on work + setup time
