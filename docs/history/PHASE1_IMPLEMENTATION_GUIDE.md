# Phase 1: Infrastructure Stabilization - Implementation Guide (Archived)

**Status**: Ready for Implementation  
**Duration**: 1-2 days  
**Objective**: Resolve critical blocking issues preventing platform operation

---

## Overview

Phase 1 addresses three critical infrastructure issues:

1. **Container Image Pull Failures** - Resolve Docker Hub rate limiting
2. **Cloudflare Tunnel Authentication** - Verify and fix connectivity
3. **Service Restoration** - Bring scaled services back online

## Critical Issues & Blocking Items

| Issue | Impact | Status | Priority |
|-------|--------|--------|----------|
| ImagePullBackOff (15+ services) | Services cannot start | Blocking | **CRITICAL** |
| Vault not initialized | Secrets management unavailable | Blocking | **CRITICAL** |
| Self-signed certificates | Not production-grade | Important | **HIGH** |
| Tunnel connectivity (Error 1033) | Portal access may fail | Blocking | **CRITICAL** |
| Limited monitoring | No alerting or observability | Important | **HIGH** |

---

## Task 1: Container Registry Setup

### Objective
Eliminate Docker Hub rate limiting by deploying a private registry.

### Prerequisites
- Docker CLI installed locally
- kubectl configured and cluster accessible
- Helm installed (for Harbor option)
- Cloud CLI tools (for cloud registry options)

### Implementation

#### Step 1.1: Choose Registry Type

**Recommended: Harbor (Self-Hosted)**
- ✅ Full control of images
- ✅ Works offline after setup
- ✅ Low ongoing costs
- ⚠️ Requires storage and compute
- ⏱️ ~30 minutes to deploy

**Or: Cloud Registry**
- ✅ Zero maintenance
- ✅ High availability included
- ⚠️ Dependency on cloud provider
- ⚠️ Potential ongoing costs
- ⏱️ ~15 minutes to setup

#### Step 1.2: Deploy Registry

**For Harbor**:
```bash
cd /home/m/tff/254CARBON/HMCo

# Run setup script
./scripts/setup-private-registry.sh harbor

# Follow prompts to deploy Harbor
# Edit and deploy values:
nano /tmp/harbor-values.yaml

helm install harbor harbor/harbor \
  -n registry \
  --values /tmp/harbor-values.yaml

# Wait for Harbor to be ready
kubectl get pods -n registry -w
```

**For AWS ECR**:
```bash
./scripts/setup-private-registry.sh ecr
```

**For GCP GCR**:
```bash
./scripts/setup-private-registry.sh gcr
```

#### Step 1.3: Verify Registry Access

```bash
# Test registry login
docker login harbor.254carbon.local  # For Harbor
# or
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Verify push works
docker tag alpine:latest <registry>/test:latest
docker push <registry>/test:latest
```

### Completion Checklist
- [ ] Registry deployed and accessible
- [ ] Docker login successful
- [ ] Test image pushed successfully
- [ ] Image pull secret created in cluster

---

## Task 2: Mirror Images to Private Registry

### Objective
Copy all required container images to private registry.

### Implementation

```bash
# Run mirroring script
./scripts/mirror-images.sh <REGISTRY_URL> <REGISTRY_TYPE>

# Example for Harbor:
./scripts/mirror-images.sh harbor.254carbon.local harbor

# Example for ECR:
./scripts/mirror-images.sh 123456789.dkr.ecr.us-east-1.amazonaws.com ecr
```

### What Gets Mirrored
- 40+ container images
- Infrastructure (NGINX, cert-manager)
- Monitoring (Prometheus, Grafana, Loki)
- Data platform (Doris, Trino, Superset, DataHub)
- Storage (MinIO, Vault, PostgreSQL)
- Messaging (Kafka, Zookeeper)
- Workflow (DolphinScheduler, Spark)

### Monitoring Progress

```bash
# Watch as images are pulled and pushed
tail -f /tmp/mirror-progress.log

# Or check registry directly
curl -s https://harbor.254carbon.local/api/v2.0/projects \
  -u admin:ChangeMe123! | jq .
```

### Expected Output
```
================================
254Carbon Image Mirroring Script
================================

Registry: harbor.254carbon.local
Type: harbor

Total images to mirror: 40

Mirroring nginx-ingress ... OK
Mirroring prometheus ... OK
Mirroring grafana ... OK
Mirroring doris ... OK
...

================================
Mirroring Summary
================================
Successful: 38
Skipped: 2
Failed: 0
```

### Completion Checklist
- [ ] All 40+ images mirrored successfully
- [ ] No major failures (skips are acceptable)
- [ ] Images accessible from cluster

---

## Task 3: Update Deployments for Private Registry

### Objective
Update all Kubernetes deployments to use private registry images.

### Implementation

**Option A: Automated (Recommended)**

```bash
# Create update script
cat > /tmp/update-registry.sh << 'EOF'
#!/bin/bash
REGISTRY="${1:-harbor.254carbon.local}"

# Update all deployments
kubectl get deployments -A -o json | \
jq -r '.items[] | "\(.metadata.namespace) \(.metadata.name)"' | \
while read ns name; do
    echo "Updating $ns/$name..."
    kubectl set image deployment/$name -n $ns "*=$REGISTRY/\$(basename \$IMAGE)"
done
EOF

chmod +x /tmp/update-registry.sh
/tmp/update-registry.sh harbor.254carbon.local
```

**Option B: Manual per Deployment**

```bash
# Example: Update MinIO
kubectl set image deployment/minio \
  -n data-platform \
  minio=harbor.254carbon.local/minio:latest

# Example: Update Doris  
kubectl set image deployment/doris-be \
  -n data-platform \
  doris=harbor.254carbon.local/doris:latest
```

### Verification

```bash
# Verify image references updated
kubectl get deployments -A -o wide | grep harbor

# Check no old registry references remain
kubectl get deployments -A -o json | grep -i "docker.io\|gcr.io\|quay.io" | wc -l
# Should return 0
```

### Completion Checklist
- [ ] All deployments updated to use private registry
- [ ] No external registry references remain
- [ ] Images pull successfully

---

## Task 4: Verify Cloudflare Tunnel

### Objective
Ensure Cloudflare Tunnel is properly configured and connected.

### Implementation

**Run comprehensive tunnel verification**:

```bash
chmod +x ./scripts/verify-tunnel.sh

# Check tunnel status
./scripts/verify-tunnel.sh status

# This will check:
# 1. Pod status (2/2 running)
# 2. Credentials formatting
# 3. Pod logs for connection confirmation
# 4. Portal connectivity
# 5. Service accessibility
```

### Expected Output (All OK)
```
================================
Cloudflare Tunnel Status Report
================================

1. Checking tunnel pods...
✓ Both tunnel pods running (2/2)

2. Checking tunnel credentials...
✓ Credentials properly formatted
  Tunnel ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5
  Account ID: 0c93c74d5269a228e91d4bf91c547f56

3. Checking tunnel pod logs...
Recent logs from cloudflared-xxx:
INF Registered tunnel connection [UUID]
✓ Tunnel appears connected

4. Testing portal accessibility...
✓ Portal responding (HTTP 302 - redirect to login)

5. Testing service connectivity...
✓ grafana.254carbon.com responding (HTTP 302)
✓ vault.254carbon.com responding (HTTP 302)
✓ datahub.254carbon.com responding (HTTP 302)

Services responding: 3/3
```

### If Issues Found

```bash
# Check tunnel logs for errors
./scripts/verify-tunnel.sh logs

# If credentials issue:
./scripts/update-cloudflare-credentials.sh \
  291bc289-e3c3-4446-a9ad-8e327660ecd5 \
  0c93c74d5269a228e91d4bf91c547f56 \
  0ddc10a6-4f6a-4357-8430-16ec31febeea

# If pod issues:
./scripts/verify-tunnel.sh restart

# Full diagnostics and automatic fixes:
./scripts/verify-tunnel.sh fix
```

### Verification in Cloudflare Dashboard

1. Navigate to: https://dash.cloudflare.com/zero-trust/networks/tunnels
2. Find tunnel: "254carbon-cluster"
3. Verify status shows: **🟢 Connected** (green icon)
4. Verify routes include all services

### Completion Checklist
- [ ] Tunnel pods running (2/2)
- [ ] Credentials properly formatted
- [ ] Tunnel shows "Connected" in dashboard
- [ ] Portal accessible at https://254carbon.com (HTTP 302)
- [ ] Services accessible (Grafana, Vault, DataHub)

---

## Task 5: Initialize and Scale Vault

### Objective
Set up production Vault with proper PostgreSQL backend and initialize for use.

### Implementation

**Step 1: Scale Vault to 1 replica**

```bash
kubectl scale statefulset vault -n vault-prod --replicas=1
kubectl wait pod -n vault-prod -l app=vault --for=condition=Ready --timeout=120s
```

**Step 2: Initialize Vault**

```bash
chmod +x ./scripts/initialize-vault-production.sh

# Initialize (generates unseal keys and root token)
./scripts/initialize-vault-production.sh init

# This will output:
# ✓ Vault initialized
# CRITICAL: Store these keys securely
# Unseal Keys: (3 keys listed)
# Root Token: (token shown)
```

**Step 3: Unseal Vault**

```bash
# Automatically unseals using keys from previous step
./scripts/initialize-vault-production.sh unseal

# Verify unsealed
./scripts/initialize-vault-production.sh status
```

**Step 4: Configure Authentication**

```bash
# Configure Kubernetes auth
./scripts/initialize-vault-production.sh config
```

**Step 5: Scale to 3 replicas**

```bash
# After everything is initialized
kubectl scale statefulset vault -n vault-prod --replicas=3

# Verify all replicas running
kubectl get pods -n vault-prod
```

### Important: Store Unseal Keys Securely

```bash
# Unseal keys are saved to:
/tmp/vault-init-keys-backup.txt

# CRITICAL: Move to secure storage
# - Encrypt the file
# - Store offline (not in version control)
# - Share securely with authorized personnel
# - Keep multiple copies in different locations

# Example: Encrypt and store
gpg --symmetric /tmp/vault-init-keys-backup.txt
# Prompts for passphrase, creates .gpg file
rm /tmp/vault-init-keys-backup.txt  # Delete original
```

### Completion Checklist
- [ ] Vault initialized
- [ ] Unseal keys generated and stored securely
- [ ] Vault unsealed and accessible
- [ ] Kubernetes auth configured
- [ ] 3 replicas running
- [ ] Secret engines enabled

---

## Task 6: Restore Scaled Services

### Objective
Bring services back online now that image registry is working.

### Implementation

**Identify services currently scaled to 0**:

```bash
kubectl get deployments -A --all-namespaces -o wide | grep "0/0\|0/"
```

**Scale services back up**:

```bash
# Scale individual services
kubectl scale deployment minio -n data-platform --replicas=1
kubectl scale deployment superset -n data-platform --replicas=1
kubectl scale deployment trino -n data-platform --replicas=1
kubectl scale deployment doris-fe -n data-platform --replicas=1
kubectl scale deployment doris-be -n data-platform --replicas=3

# Or use script to scale all at once
cat > /tmp/scale-up.sh << 'EOF'
#!/bin/bash
# Scale all services back up
deployments=(
  "minio:1"
  "superset:1"
  "trino:1"
  "doris-fe:1"
  "doris-be:3"
  "dolphinscheduler-api:1"
)

for dep in "${deployments[@]}"; do
    name="${dep%:*}"
    replicas="${dep#*:}"
    echo "Scaling $name to $replicas..."
    kubectl scale deployment/$name -n data-platform --replicas=$replicas
done
EOF

chmod +x /tmp/scale-up.sh
/tmp/scale-up.sh
```

**Monitor startup**:

```bash
# Watch pods come up
watch 'kubectl get pods -n data-platform -o wide'

# Check logs for any startup issues
kubectl logs -n data-platform -l app=minio --tail=20
```

### Verify Services Operational

```bash
# Check all services are running
kubectl get pods -n data-platform | grep -E "Running|Ready"

# Check specific service health
kubectl exec -n data-platform <pod-name> -- /health-check-command

# Test connectivity through portal
curl -v https://grafana.254carbon.com
curl -v https://vault.254carbon.com
```

### Completion Checklist
- [ ] All critical services scaled to appropriate replicas
- [ ] No ImagePullBackOff errors
- [ ] All pods in Running state
- [ ] Health checks passing
- [ ] Services accessible through portal

---

## Validation & Sign-Off

### Run Complete Validation

```bash
# Check cluster health
./scripts/validate-cluster.sh

# Verify all Phase 1 tasks complete
echo "Phase 1 Validation Checklist:"
echo "✓ Container registry deployed"
echo "✓ Images mirrored to registry"
echo "✓ Deployments updated to use registry"
echo "✓ Cloudflare tunnel verified"
echo "✓ Portal accessible"
echo "✓ Services restored and running"
echo "✓ Vault initialized"
echo "✓ All critical pods healthy"
```

### Expected Status After Phase 1

```
Cluster Health: ✅ OPERATIONAL

Pods by Status:
  Running: 45+
  CrashLoop: 0
  ImagePull: 0
  Pending: 0

Services Accessible:
  Portal: ✅ https://254carbon.com
  Grafana: ✅ https://grafana.254carbon.com
  Vault: ✅ https://vault.254carbon.com
  (+ 6 more services)

Infrastructure:
  ✅ Private registry deployed
  ✅ Tunnel connected
  ✅ TLS working (self-signed)
  ✅ SSO configured

Ready for Phase 2: ✅ YES
```

---

## Troubleshooting

### Issue: Images still pulling from Docker Hub

**Solution**:
```bash
# Verify deployment image references
kubectl get deployment minio -n data-platform -o yaml | grep image

# Update image reference if needed
kubectl set image deployment/minio -n data-platform \
  minio=harbor.254carbon.local/minio:latest
```

### Issue: Tunnel pods not connecting

**Solution**:
```bash
# Check logs for specific error
kubectl logs -n cloudflare-tunnel <pod-name>

# Verify credentials
kubectl get secret cloudflare-tunnel-credentials -n cloudflare-tunnel -o yaml

# Update if needed
./scripts/update-cloudflare-credentials.sh TUNNEL_ID ACCOUNT_TAG AUTH_TOKEN
```

### Issue: Vault won't initialize

**Solution**:
```bash
# Check pod logs
kubectl logs -n vault-prod vault-0

# Verify PostgreSQL connection
kubectl exec -n vault-prod vault-0 -- \
  psql -h postgres-shared-service.data-platform \
  -U vault -d vault -c "SELECT version();"

# Scale back to 0 and retry
kubectl scale statefulset vault -n vault-prod --replicas=0
sleep 10
./scripts/initialize-vault-production.sh init
```

---

## Next Steps

After Phase 1 completion, proceed to:

**Phase 2: Security Hardening** (2-3 days)
- TLS certificates (Let's Encrypt production)
- Secrets migration to Vault
- Network policies
- RBAC hardening

See: [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md#phase-2-security-hardening)

---

## Support & Escalation

For issues not covered here:

1. Check logs: `kubectl logs -n NAMESPACE POD_NAME`
2. Review documentation: `PRODUCTION_READINESS.md`
3. Check Cloudflare dashboard: `https://dash.cloudflare.com/zero-trust`

---

**Phase 1 Status**: Ready to Begin  
**Estimated Duration**: 4-8 hours  
**Next Review**: After each major task completion
