# Production Platform Stabilization & Readiness Guide

**Last Updated**: October 19, 2025  
**Status**: Implementation Phase 1 Starting  
**Project Completion Target**: 90% in 2 weeks

---

## Executive Summary

This document provides a comprehensive roadmap to transform the 254Carbon platform from development to production-grade, enterprise deployment. The plan is organized into 8 phases spanning critical infrastructure fixes through advanced capabilities.

### Current State
- ✅ Kubernetes cluster with 9 core services deployed
- ✅ Cloudflare Tunnel integration for external access
- ✅ SSO via Cloudflare Access (Email OTP authentication)
- ✅ Landing portal application running
- ⚠️ **Image pull failures affecting 15+ services**
- ⚠️ **Vault not initialized (scaled to 0)**
- ⚠️ **Self-signed certificates (not production-grade)**
- ⚠️ **Limited monitoring and no alerting**
- ⚠️ **No backup/disaster recovery procedures**

### Target State
- Production-grade TLS certificates
- Fully initialized Vault with dynamic secrets
- Comprehensive monitoring, logging, and alerting
- Automated backups with disaster recovery testing
- Multi-node HA configuration
- Network policies and enhanced security
- GitOps-based deployment automation

---

## Phase 1: Infrastructure Stabilization (Critical - Week 1)

**Objective**: Resolve immediate blocking issues preventing platform operation

### 1.1 Container Registry Issues - CRITICAL

#### Problem
Docker Hub rate limiting causing `ImagePullBackOff` for 15+ services:
- Doris, Trino, Superset, MinIO, Vault, LakeFS, Spark, Iceberg REST, and others

#### Solution: Private Container Registry
Deploy a private registry to mirror all images and eliminate external dependencies.

**Option A: Harbor (Recommended - Self-Hosted)**
```bash
# Install Harbor Helm chart
helm repo add harbor https://helm.goharbor.io
helm repo update

helm install harbor harbor/harbor \
  --namespace harbor \
  --create-namespace \
  --values harbor-values.yaml
```

**Option B: Cloud Registry (Faster)**
- AWS ECR: `aws ecr create-repository --repository-name 254carbon/imagename`
- GCP GCR: `gcloud container images list`
- Azure ACR: `az acr create --resource-group mygroup --name myregistry`

#### Implementation Steps
1. **Choose registry provider** (Harbor or cloud)
2. **Create service accounts and credentials**
3. **Mirror all required images** (see script: `scripts/mirror-images.sh`)
4. **Configure Kubernetes image pull secrets**
5. **Update all deployments** to use private registry
6. **Test image pulls and pod startup**

#### Status
- [ ] Registry deployed and accessible
- [ ] Image mirroring completed
- [ ] Pull secrets configured
- [ ] All services successfully pulling images

**See Also**: `scripts/setup-private-registry.sh`

---

### 1.2 Cloudflare Tunnel Authentication - CRITICAL

#### Problem
Error 1033 was partially fixed but needs full verification and credential management

#### Solution
Verify and document tunnel configuration

**Verification Steps**:
```bash
# 1. Check tunnel pod logs
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -f | grep -i "connected\|registered\|error"

# 2. Verify credentials are properly formatted (not base64)
kubectl get secret -n cloudflare-tunnel cloudflare-tunnel-credentials -o jsonpath='{.data.credentials\.json}' | base64 -d | jq .

# 3. Test connectivity to portal
curl -v https://254carbon.com 2>&1 | grep -i "location\|status"

# 4. Check Cloudflare dashboard
# Navigate to: https://dash.cloudflare.com/zero-trust/networks/tunnels
# Verify tunnel shows: "Connected" with green status
```

#### Current Tunnel Configuration
- **Tunnel ID**: 291bc289-e3c3-4446-a9ad-8e327660ecd5
- **Account ID**: 0c93c74d5269a228e91d4bf91c547f56
- **Tunnel Name**: 254carbon-cluster
- **Auth Token**: 0ddc10a6-4f6a-4357-8430-16ec31febeea (properly decoded)

#### If Tunnel Not Connected
```bash
# 1. Restart tunnel
kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel

# 2. Monitor reconnection
kubectl logs -n cloudflare-tunnel -f

# 3. If still failing, update credentials
./scripts/update-cloudflare-credentials.sh \
  291bc289-e3c3-4446-a9ad-8e327660ecd5 \
  0c93c74d5269a228e91d4bf91c547f56 \
  0ddc10a6-4f6a-4357-8430-16ec31febeea
```

#### Status
- [ ] Tunnel pods running (2/2)
- [ ] Tunnel shows "Connected" in dashboard
- [ ] Portal accessible at 254carbon.com
- [ ] All service domains resolving

**See Also**: `k8s/cloudflare/README.md`

---

### 1.3 Restore Scaled Services

#### Problem
15+ services are scaled to 0 replicas due to image pull issues

#### Services to Restore
1. **Storage**: MinIO, Vault, LakeFS
2. **Query Engines**: Trino, ClickHouse
3. **Visualization**: Superset
4. **Orchestration**: DolphinScheduler
5. **Compute**: Spark components

#### Restoration Process

**Step 1: Update Image References**
```bash
# Find all deployments using old image registry
kubectl get deployments -A -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}' | grep -v "private-registry"

# Update to use private registry
kubectl set image deployment/minio -n data-platform \
  minio=private-registry/minio/minio:RELEASE.2024-01-11T08-13-15Z
```

**Step 2: Scale Services**
```bash
# Create script to scale all services
cat > scripts/restore-services.sh << 'EOF'
#!/bin/bash
services=(
  "minio:data-platform:1"
  "clickhouse:data-platform:1"
  "trino:data-platform:1"
  "superset:data-platform:1"
  "dolphinscheduler-api:data-platform:1"
  "seatunnel:data-platform:1"
  "spark-operator:data-platform:1"
)

for service in "${services[@]}"; do
  IFS=: read -r name namespace replicas <<< "$service"
  echo "Scaling $name in $namespace to $replicas replicas..."
  kubectl scale deployment/$name -n $namespace --replicas=$replicas
done
EOF

chmod +x scripts/restore-services.sh
./scripts/restore-services.sh
```

**Step 3: Verify Health**
```bash
# Monitor pod startup
watch 'kubectl get pods -A | grep -E "CrashLoop|ImagePull|Pending"'

# Check logs for issues
kubectl logs -n data-platform -l app=minio --tail=20
```

#### Status
- [ ] All services restored to appropriate replica counts
- [ ] No ImagePullBackOff errors
- [ ] All services passing health checks
- [ ] Data persistence verified (where applicable)

---

## Phase 2: Security Hardening (Important - Week 1-2)

**Objective**: Secure the platform with proper certificates, secrets management, and access controls

### 2.1 TLS Certificate Management

#### Current State
- Self-signed certificates suitable only for development

#### Target State
- Let's Encrypt production certificates
- Automatic renewal via cert-manager
- HSTS headers enabled

#### Implementation

**Step 1: Ensure cert-manager is installed**
```bash
kubectl get crd certificates.cert-manager.io || helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0
```

**Step 2: Create Cloudflare API token secret**
```bash
kubectl create secret generic cloudflare-api-token \
  -n cert-manager \
  --from-literal=api-token=YOUR_CF_API_TOKEN
```

**Step 3: Deploy production ClusterIssuer**
See: `k8s/certificates/production-issuer.yaml` (to be created)

**Step 4: Update ingress annotations**
```yaml
annotations:
  cert-manager.io/cluster-issuer: "letsencrypt-prod"
  nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
  nginx.ingress.kubernetes.io/hsts: "true"
  nginx.ingress.kubernetes.io/hsts-max-age: "31536000"
```

**Step 5: Verify certificate issuance**
```bash
kubectl get certificate -A
kubectl describe certificate portal-tls -n data-platform
```

#### Status
- [ ] cert-manager deployed
- [ ] Production ClusterIssuer created
- [ ] All ingress certificates issued
- [ ] Certificate renewal working
- [ ] HTTPS verification passing

---

### 2.2 Production Vault Deployment

#### Current State
- Vault scaled to 0 (not initialized)
- Self-signed TLS certificates
- PostgreSQL database created but not used

#### Implementation

**Step 1: Initialize Vault**
```bash
# Scale to 1 replica
kubectl scale statefulset vault -n vault-prod --replicas=1

# Wait for pod to be ready
kubectl wait pod -n vault-prod -l app=vault --for=condition=Ready --timeout=120s

# Initialize Vault (generates unseal keys and root token)
kubectl exec -it vault-0 -n vault-prod -- vault operator init \
  -key-shares=3 \
  -key-threshold=2 \
  -output-curl-format=false > /secure/vault-init-keys.txt

# CRITICAL: Store init keys securely (offline, encrypted)
```

**Step 2: Unseal Vault**
```bash
# Get unseal keys from secure storage
KEY1=$(cat /secure/vault-init-keys.txt | grep "Unseal Key 1" | awk '{print $NF}')
KEY2=$(cat /secure/vault-init-keys.txt | grep "Unseal Key 2" | awk '{print $NF}')

# Unseal all 3 replicas
for i in 0 1 2; do
  kubectl exec vault-$i -n vault-prod -- vault operator unseal $KEY1
  kubectl exec vault-$i -n vault-prod -- vault operator unseal $KEY2
done

# Verify sealed status
kubectl exec vault-0 -n vault-prod -- vault status
```

**Step 3: Configure Kubernetes auth**
```bash
# Login with root token
export VAULT_TOKEN=$(cat /secure/vault-init-keys.txt | grep "Initial Root Token" | awk '{print $NF}')

# Enable Kubernetes auth
kubectl exec -it vault-0 -n vault-prod -- vault auth enable kubernetes

# Configure Kubernetes auth
kubectl exec -it vault-0 -n vault-prod -- vault write auth/kubernetes/config \
  kubernetes_host=https://kubernetes.default.svc.cluster.local:443 \
  kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token
```

**Step 4: Create secret engines and policies**
```bash
# Enable secret engines
vault secrets enable -path=secret kv-v2
vault secrets enable -path=database database
vault secrets enable -path=ssh ssh

# Create policies for services
kubectl apply -f k8s/vault/vault-policies.yaml
```

#### Status
- [ ] Vault initialized with 3 unseal keys
- [ ] Vault unsealed and ready
- [ ] Kubernetes auth method enabled
- [ ] Secret engines configured
- [ ] All 3 replicas running
- [ ] Init keys stored securely offline

**See Also**: `k8s/vault/VAULT-PRODUCTION-DEPLOYMENT.md`

---

### 2.3 Migrate Secrets from ConfigMaps to Vault/Secrets

#### Current Issues
- Credentials stored in plain ConfigMaps
- Hardcoded passwords visible in manifests
- No rotation mechanism

#### Implementation

**Step 1: Create Kubernetes Secrets for sensitive data**
```bash
# Convert ConfigMap credentials to Secrets
kubectl create secret generic postgres-credentials \
  -n data-platform \
  --from-literal=username=vault \
  --from-literal=password=SECURE_PASSWORD_HERE \
  --dry-run=client -o yaml | kubectl apply -f -

# Update deployments to use Secrets
kubectl patch deployment vault -n vault-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"vault","env":[{"name":"VAULT_CACERT","valueFrom":{"secretKeyRef":{"name":"vault-tls","key":"tls.crt"}}}]}]}}}}'
```

**Step 2: Implement Vault integration**
See: `k8s/vault/vault-secret-integration.yaml` (to be created)

**Step 3: Update all deployments**
```bash
# Audit current ConfigMaps for sensitive data
kubectl get configmaps -A -o json | grep -i "password\|token\|api"
```

#### Status
- [ ] All passwords moved to Secrets or Vault
- [ ] No credentials in ConfigMaps
- [ ] No credentials in manifests
- [ ] Services using injected secrets
- [ ] Audit complete

---

### 2.4 Network Policies

#### Implementation

**Step 1: Deploy default deny policies**
```bash
kubectl apply -f k8s/networking/network-policies-deny.yaml
```

**Step 2: Create service-to-service allow rules**
See: `k8s/networking/network-policies-allow.yaml` (to be created)

**Step 3: Test connectivity**
```bash
# Create test pods
kubectl run test-client -n data-platform --image=busybox --rm -it -- /bin/sh

# Test connectivity
wget -O- http://grafana.monitoring:3000/api/health
```

#### Status
- [ ] Default deny policies deployed
- [ ] Service communication rules configured
- [ ] External ingress working
- [ ] Network policies tested

---

## Phase 3: High Availability & Resilience (Important - Week 2)

**Objective**: Enable multi-node deployment with proper HA configuration

### 3.1 Multi-Node Cluster Setup
**Placeholder for multi-node configuration guidance**

### 3.2 Service High Availability
**Placeholder for HA service configuration**

### 3.3 Resource Management
**Placeholder for resource quotas and autoscaling**

---

## Phase 4: Monitoring & Observability (Important - Week 2)

**Objective**: Comprehensive monitoring, logging, and alerting

### Status Tracking
- [ ] Prometheus enhanced with service discovery
- [ ] Grafana dashboards created for all services
- [ ] AlertManager configured with notification channels
- [ ] Loki configured for log aggregation
- [ ] SLO-based alerting rules deployed

---

## Phase 5: Backup & Disaster Recovery (Important - Week 2)

**Objective**: Automated backups with tested recovery procedures

### Status Tracking
- [ ] PostgreSQL backup strategy implemented
- [ ] Velero deployed for Kubernetes backups
- [ ] Backup retention policies configured
- [ ] Recovery procedures documented
- [ ] DR drill executed successfully

---

## Phase 6: Performance Optimization (Nice to Have - Week 3)

**Objective**: Optimize storage, network, and application performance

---

## Phase 7: Operational Procedures (Important - Week 2)

**Objective**: CI/CD automation, documentation, and compliance

### Status Tracking
- [ ] GitOps with ArgoCD deployed
- [ ] Operational runbooks created
- [ ] Compliance scanning enabled
- [ ] Cost tracking configured

---

## Phase 8: Final Integration (Week 3)

**Objective**: End-to-end testing and security audit

### Status Tracking
- [ ] Integration tests passing
- [ ] Security audit completed
- [ ] Performance testing done
- [ ] Documentation complete
- [ ] Ready for production rollout

---

## Implementation Checklist

### Critical Path (Must Complete)
- [ ] Fix image pull issues with private registry
- [ ] Restore all scaled services
- [ ] Deploy production Vault
- [ ] Implement production TLS certificates
- [ ] Migrate secrets from ConfigMaps

### Important Path (Should Complete)
- [ ] Configure comprehensive monitoring
- [ ] Implement network policies
- [ ] Establish backup procedures
- [ ] Multi-node cluster setup
- [ ] Service HA configuration

### Nice to Have
- [ ] Advanced tracing
- [ ] Cost optimization
- [ ] Service mesh
- [ ] GitOps automation

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Availability | 99.9% | TBD |
| API Response Time | <100ms | TBD |
| MTTR (Mean Time to Recover) | <1 hour | TBD |
| Security Vulnerabilities | 0 critical | TBD |
| Backup Success Rate | 100% | TBD |
| Certificate Renewal Rate | 100% | TBD |

---

## Quick Reference

### Common Commands

```bash
# Check cluster health
kubectl get nodes
kubectl get pods -A | grep -E "CrashLoop|ImagePull|Pending"

# Monitor pod startup
watch 'kubectl get pods -n data-platform'

# View tunnel status
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel -f

# Check vault status
kubectl exec -it vault-0 -n vault-prod -- vault status

# Verify certificates
kubectl get certificate -A
kubectl describe certificate portal-tls -n data-platform
```

### Important Files
- Documentation Index: [../index.md](../index.md)
- Cloudflare Setup: [../cloudflare/deployment.md](../cloudflare/deployment.md)
- SSO Setup: [../sso/guide.md](../sso/guide.md)
- Portal: [../../portal/README.md](../../portal/README.md)

---

**Document Version**: 1.0  
**Last Updated**: October 19, 2025  
**Next Review**: October 26, 2025
