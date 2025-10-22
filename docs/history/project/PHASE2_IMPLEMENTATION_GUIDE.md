# Phase 2: Security Hardening - Implementation Guide

**Status**: Ready for Implementation  
**Duration**: 2-3 days  
**Objective**: Secure platform with production certificates, secrets management, and access controls

---

## Overview

Phase 2 addresses critical security gaps:

1. **TLS Certificates** - Replace self-signed with Let's Encrypt production certs
2. **Secrets Management** - Migrate credentials from ConfigMaps to Vault
3. **Network Security** - Implement NetworkPolicies for pod isolation
4. **RBAC Hardening** - Least-privilege service accounts and role bindings

---

## Task 1: Production TLS Certificates

### Objective
Replace self-signed certificates with Let's Encrypt production certificates using cert-manager and Cloudflare DNS validation.

### Current State
- ✅ cert-manager deployed (already in cluster)
- ✅ Cloudflare domain configured
- ❌ Using self-signed certificates (development-only)

### Implementation

**Step 1.1: Create Cloudflare API Token Secret**

```bash
# Create secret for cert-manager to use Cloudflare DNS challenge
kubectl create secret generic cloudflare-api-token \
  -n cert-manager \
  --from-literal=api-token=YOUR_CLOUDFLARE_API_TOKEN \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Step 1.2: Deploy Production ClusterIssuer**

```bash
cat > /tmp/letsencrypt-production-issuer.yaml << 'EOF'
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@254carbon.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
    - dns01:
        cloudflare:
          apiTokenSecretRef:
            name: cloudflare-api-token
            key: api-token
EOF

kubectl apply -f /tmp/letsencrypt-production-issuer.yaml
```

**Step 1.3: Update Ingress Annotations**

For each ingress, update annotations:
```yaml
annotations:
  cert-manager.io/cluster-issuer: "letsencrypt-prod"
  nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
  nginx.ingress.kubernetes.io/hsts: "true"
  nginx.ingress.kubernetes.io/hsts-max-age: "31536000"
  nginx.ingress.kubernetes.io/hsts-include-subdomains: "true"
```

**Step 1.4: Verify Certificate Issuance**

```bash
# Wait for certificates to be issued (can take 5-10 minutes)
kubectl get certificate -A

# Check certificate details
kubectl describe certificate portal-tls -n data-platform

# Verify certificate is valid
kubectl get secret portal-tls -n data-platform -o yaml | grep tls.crt | \
  awk '{print $2}' | base64 -d | openssl x509 -text -noout
```

### Completion Checklist
- [ ] Cloudflare API token secret created
- [ ] Production ClusterIssuer deployed
- [ ] Ingress annotations updated
- [ ] All certificates issued and valid
- [ ] HTTPS working on all domains
- [ ] HSTS headers present

---

## Task 2: Secrets Management - Migration to Vault

### Objective
Move all credentials from Kubernetes ConfigMaps to Vault, enabling dynamic secrets and rotation.

### Current Issues
- ✅ Vault initialized and running
- ❌ Credentials still in ConfigMaps
- ❌ No dynamic secret generation
- ❌ No credential rotation

### Implementation

**Step 2.1: Audit Current Credentials**

```bash
# Find all ConfigMaps with credentials
kubectl get configmaps -A -o json | jq '.items[] | select(.data | tostring | contains("password") or contains("token") or contains("api")) | {namespace: .metadata.namespace, name: .metadata.name}'

# Save results for migration
```

**Step 2.2: Configure Vault Database Secret Engine**

```bash
# Port-forward to Vault
kubectl port-forward -n data-platform vault-d4c9c888b-cdsgz 8200:8200 &

# Export Vault token (if known) or use root token
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN='your-token-here'

# Enable database secret engine
vault secrets enable database

# Configure PostgreSQL connection
vault write database/config/postgresql \
  plugin_name=postgresql-database-plugin \
  allowed_roles="readonly,readwrite" \
  connection_url="postgresql://{{username}}:{{password}}@postgres-shared-service.data-platform:5432/datahub" \
  username="datahub" \
  password="datahub-secure-password"

# Create read-only role
vault write database/roles/readonly \
  db_name=postgresql \
  creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT USAGE ON SCHEMA public TO \"{{name}}\"; GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
  default_ttl="1h" \
  max_ttl="24h"
```

**Step 2.3: Create Kubernetes Auth Method Role for Services**

```bash
# Enable Kubernetes auth (already done in Phase 1)
vault auth enable kubernetes || echo "Kubernetes auth already enabled"

# Create policy for service secrets
vault policy write datahub-policy - << 'EOF'
path "database/creds/readonly" {
  capabilities = ["read"]
}
path "secret/data/datahub/*" {
  capabilities = ["read"]
}
EOF

# Create Kubernetes auth role
vault write auth/kubernetes/role/datahub \
  bound_service_account_names=datahub \
  bound_service_account_namespaces=data-platform \
  policies=datahub-policy \
  ttl=24h
```

**Step 2.4: Update Service Deployments**

Instead of reading ConfigMaps, pods should:
1. Use Vault agent for secret injection
2. Or use init containers to fetch secrets from Vault
3. Or mount volumes with Vault CSI driver

Example with Vault Agent:
```yaml
annotations:
  vault.hashicorp.com/agent-inject: "true"
  vault.hashicorp.com/agent-inject-secret-database: "database/creds/readonly"
  vault.hashicorp.com/agent-inject-template-database: |
    {{- with secret "database/creds/readonly" -}}
    export DB_USER="{{ .Data.data.username }}"
    export DB_PASSWORD="{{ .Data.data.password }}"
    {{- end }}
  vault.hashicorp.com/role: "datahub"
```

### Completion Checklist
- [ ] All ConfigMap credentials audited
- [ ] Vault database engine configured
- [ ] PostgreSQL read-only role created
- [ ] Kubernetes auth roles created
- [ ] Service deployments updated with Vault injection
- [ ] Credentials successfully injected in pods
- [ ] ConfigMaps cleaned up (credentials removed)

---

## Task 3: Network Policies

### Objective
Implement NetworkPolicies to restrict pod-to-pod communication and enforce least-privilege network access.

### Implementation

**Step 3.1: Default Deny All Ingress**

```bash
cat > /tmp/default-deny-all.yaml << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all-ingress
  namespace: data-platform
spec:
  podSelector: {}
  policyTypes:
  - Ingress
EOF

kubectl apply -f /tmp/default-deny-all.yaml
```

**Step 3.2: Allow Specific Service Communication**

```bash
cat > /tmp/allow-service-communication.yaml << 'EOF'
# Allow DataHub API to communicate with PostgreSQL
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-datahub-to-postgres
  namespace: data-platform
spec:
  podSelector:
    matchLabels:
      app: datahub-gms
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432

---
# Allow NGINX ingress to reach services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-to-services
  namespace: data-platform
spec:
  podSelector:
    matchLabels:
      app: datahub-frontend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 9002
EOF

kubectl apply -f /tmp/allow-service-communication.yaml
```

**Step 3.3: Monitor Network Policies**

```bash
# Test connectivity
kubectl exec -n data-platform <pod-name> -- curl http://<target-service>:port

# View applied policies
kubectl get networkpolicies -n data-platform
kubectl describe networkpolicy <name> -n data-platform
```

### Completion Checklist
- [ ] Default deny ingress policies deployed
- [ ] Service-to-service allow rules configured
- [ ] Ingress to service communication allowed
- [ ] Egress to external services configured
- [ ] All pods can still communicate within rules
- [ ] No broken service communication

---

## Task 4: RBAC Enhancement

### Objective
Implement least-privilege access with service accounts and role bindings.

### Implementation

**Step 4.1: Create Service-Specific ServiceAccounts**

```bash
cat > /tmp/service-accounts.yaml << 'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: datahub
  namespace: data-platform

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: grafana
  namespace: monitoring

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: monitoring
EOF

kubectl apply -f /tmp/service-accounts.yaml
```

**Step 4.2: Create Least-Privilege Roles**

```bash
cat > /tmp/rbac-roles.yaml << 'EOF'
# DataHub read-only role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: datahub-reader
  namespace: data-platform
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["datahub-config"]
  verbs: ["get"]

---
# Prometheus scraper role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: prometheus-scraper
  namespace: monitoring
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["prometheus-config"]
  verbs: ["get"]
EOF

kubectl apply -f /tmp/rbac-roles.yaml
```

**Step 4.3: Bind Roles to ServiceAccounts**

```bash
cat > /tmp/rbac-bindings.yaml << 'EOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: datahub-reader-binding
  namespace: data-platform
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: datahub-reader
subjects:
- kind: ServiceAccount
  name: datahub
  namespace: data-platform

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prometheus-scraper-binding
  namespace: monitoring
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: prometheus-scraper
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: monitoring
EOF

kubectl apply -f /tmp/rbac-bindings.yaml
```

**Step 4.4: Update Deployments to Use ServiceAccounts**

```yaml
spec:
  serviceAccountName: datahub
  serviceAccount: datahub
  # ... rest of pod spec
```

### Completion Checklist
- [ ] Service-specific ServiceAccounts created
- [ ] Least-privilege Roles defined
- [ ] RoleBindings created
- [ ] Deployments updated to use ServiceAccounts
- [ ] RBAC policies verified with `auth can-i` checks

---

## Verification & Testing

### Test TLS Certificates
```bash
# Verify HTTPS working
curl -v https://254carbon.com

# Check certificate details
openssl s_client -connect grafana.254carbon.com:443 -showcerts
```

### Test Vault Secret Injection
```bash
# Check secrets mounted in pod
kubectl exec -it <pod-name> -n data-platform -- env | grep DB_
```

### Test Network Policies
```bash
# Verify pods can't reach services they shouldn't
kubectl exec -it <pod-name> -n data-platform -- curl http://unallowed-service:port || echo "Correctly blocked"
```

### Test RBAC
```bash
# Check service account permissions
kubectl auth can-i get configmaps --as=system:serviceaccount:data-platform:datahub -n data-platform
```

---

## Completion Checklist for Phase 2

- [ ] **TLS Certificates**
  - [ ] Let's Encrypt ClusterIssuer deployed
  - [ ] All ingress certificates issued
  - [ ] HTTPS working on all domains
  - [ ] HSTS headers present

- [ ] **Secrets Management**
  - [ ] Vault database engine configured
  - [ ] Kubernetes auth roles created
  - [ ] Services using Vault injection
  - [ ] ConfigMaps cleaned of credentials

- [ ] **Network Policies**
  - [ ] Default deny ingress deployed
  - [ ] Allow rules for service communication
  - [ ] External egress configured
  - [ ] No broken services

- [ ] **RBAC**
  - [ ] Service accounts created
  - [ ] Least-privilege roles defined
  - [ ] RoleBindings in place
  - [ ] Deployments using service accounts

---

## Success Metrics - Phase 2

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Valid TLS | 100% | `curl -v https://254carbon.com` |
| Secrets in Vault | 100% | `kubectl exec pod -- env \| grep DB_` |
| Network isolation | Working | `kubectl get networkpolicies` |
| RBAC enforcement | Least privilege | `kubectl auth can-i` |

---

## Troubleshooting

### TLS Certificate Issues
```bash
# Check certificate status
kubectl describe certificate -n data-platform

# Check cert-manager logs
kubectl logs -n cert-manager -l app.kubernetes.io/name=cert-manager -f

# Check ClusterIssuer status
kubectl describe clusterissuer letsencrypt-prod
```

### Vault Connection Issues
```bash
# Verify Vault is accessible
kubectl exec vault-pod -n data-platform -- vault status

# Check Kubernetes auth
vault auth list
vault read auth/kubernetes/config
```

### Network Policy Blocking Services
```bash
# Test connectivity
kubectl exec -it <source-pod> -n data-platform -- curl <target-service>

# Review policy
kubectl describe networkpolicy <policy-name> -n data-platform

# Temporarily remove policy for debugging
kubectl delete networkpolicy <policy-name> -n data-platform
```

---

## Next Steps After Phase 2

Once security hardening is complete:
1. Monitor for any connectivity issues
2. Document all policies and credentials locations
3. Prepare for Phase 3: High Availability
4. Schedule Phase 2 review meeting

---

**Phase 2 Status**: Ready to Begin  
**Estimated Duration**: 2-3 days  
**Next Review**: After each major task completion
