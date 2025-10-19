# Iceberg Security Hardening Guide

## Overview

This document provides security best practices and hardening procedures for the Iceberg REST Catalog integration.

## 1. Credentials and Secrets Management

### 1.1 MinIO Credentials

**Current State (Development):**
- Username: `minioadmin`
- Password: `minioadmin123`

**Production Changes Required:**

```bash
# 1. Update MinIO credentials via UI or CLI
# Access MinIO console: http://minio-service:9001
# Change admin user password immediately

# 2. Create service accounts for each application
# SeaTunnel account
mc admin user add minio seatunnel <strong-password>
mc admin policy attach minio readwrite --user seatunnel --bucket iceberg-warehouse

# DataHub account
mc admin user add minio datahub <strong-password>
mc admin policy attach minio readwrite --user datahub

# Iceberg account
mc admin user add minio iceberg <strong-password>
mc admin policy attach minio readwrite --user iceberg

# 3. Update Kubernetes secrets
kubectl create secret generic minio-secret-prod \
  -n data-platform \
  --from-literal=access-key=seatunnel \
  --from-literal=secret-key=<strong-password> \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Update deployments to use new secret
# Edit: k8s/data-lake/iceberg-rest.yaml
# Change secretKeyRef.name from minio-secret to minio-secret-prod
```

### 1.2 PostgreSQL Credentials

**Current State:**
```
iceberg_user: "iceberg_password"
```

**Production Changes:**

```bash
# 1. Generate strong password
NEW_PASSWORD=$(openssl rand -base64 32)
echo "Generated password: $NEW_PASSWORD"

# 2. Update PostgreSQL
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -c "ALTER USER iceberg_user WITH PASSWORD '$NEW_PASSWORD';"

# 3. Update Kubernetes secret
kubectl create secret generic iceberg-db-secret \
  -n data-platform \
  --from-literal=password=$NEW_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Update Iceberg REST Catalog deployment
# Edit: k8s/data-lake/iceberg-rest.yaml
# Change CATALOG_JDBC_PASSWORD to use secretKeyRef
```

### 1.3 DataHub Secret

**Update:**

```bash
# Generate secure DataHub secret
DATAHUB_SECRET=$(openssl rand -base64 32)
echo "Generated secret: $DATAHUB_SECRET"

# Update secret
kubectl create secret generic datahub-secret \
  -n data-platform \
  --from-literal=DATAHUB_SECRET=$DATAHUB_SECRET \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 2. Network Security

### 2.1 Network Policies

Create network policies to restrict traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: iceberg-network-policy
  namespace: data-platform
spec:
  podSelector:
    matchLabels:
      app: iceberg-rest-catalog
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: trino-coordinator
    - podSelector:
        matchLabels:
          app: datahub-gms
    - podSelector:
        matchLabels:
          app: seatunnel
    ports:
    - protocol: TCP
      port: 8181
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres-shared
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: minio
    ports:
    - protocol: TCP
      port: 9000
```

### 2.2 TLS/HTTPS Configuration

**Enable TLS for Iceberg REST Catalog:**

```yaml
# File: k8s/data-lake/iceberg-rest-tls.yaml
apiVersion: v1
kind: Secret
metadata:
  name: iceberg-rest-tls
  namespace: data-platform
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
---
# Update Iceberg deployment to use TLS
# Add to deployment spec:
env:
- name: CATALOG_REST_TLS_ENABLED
  value: "true"
- name: CATALOG_REST_TLS_CERT_PATH
  value: "/etc/tls/certs/tls.crt"
- name: CATALOG_REST_TLS_KEY_PATH
  value: "/etc/tls/certs/tls.key"
volumeMounts:
- name: tls-certs
  mountPath: /etc/tls/certs
  readOnly: true
volumes:
- name: tls-certs
  secret:
    secretName: iceberg-rest-tls
```

### 2.3 Service Mesh Integration (Optional)

Consider using Istio for advanced network policies:

```bash
# Install Istio (optional for production)
istioctl install --set profile=production -y

# Enable sidecar injection for data-platform namespace
kubectl label namespace data-platform istio-injection=enabled

# Create VirtualService for Iceberg
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: iceberg-rest-catalog
  namespace: data-platform
spec:
  hosts:
  - iceberg-rest-catalog
  http:
  - match:
    - uri:
        prefix: "/v1"
    route:
    - destination:
        host: iceberg-rest-catalog
        port:
          number: 8181
EOF
```

## 3. Access Control and RBAC

### 3.1 Kubernetes RBAC

Create fine-grained RBAC for services:

```yaml
---
# ServiceAccount for Iceberg REST Catalog
apiVersion: v1
kind: ServiceAccount
metadata:
  name: iceberg-rest-sa
  namespace: data-platform
---
# Role for Iceberg REST
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: iceberg-rest-role
  namespace: data-platform
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["minio-secret", "iceberg-db-secret"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["iceberg-rest-catalog-docs"]
  verbs: ["get"]
---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: iceberg-rest-binding
  namespace: data-platform
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: iceberg-rest-role
subjects:
- kind: ServiceAccount
  name: iceberg-rest-sa
  namespace: data-platform
```

### 3.2 DataHub Access Control

Configure DataHub RBAC for Iceberg metadata:

```yaml
# Enable DataHub authorization
DATAHUB_AUTHORIZATION_DEFAULT_POLICY: "ALLOW"

# Create policies for Iceberg platform
# - Platform admins: Full access
# - Data engineers: Read/write on production tables
# - Analysts: Read-only on analytics schema
```

## 4. Data Encryption

### 4.1 Encryption at Rest

**MinIO Server-Side Encryption:**

```bash
# Enable MinIO KMS (optional)
# Set environment variables:
MINIO_SSE_MASTER_KEY=$(openssl rand -base64 32)

# Or use AWS KMS/Vault for production
```

**PostgreSQL Encryption:**

```bash
# Enable pgcrypto extension
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -d iceberg_rest -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

# Use encrypted columns for sensitive data
CREATE TABLE encrypted_metadata (
  id UUID PRIMARY KEY,
  data BYTEA,
  encrypted_data TEXT
);
```

### 4.2 Encryption in Transit

- **Enable TLS** for all service-to-service communication
- **Use HTTPS** for REST API calls
- **Enable SSL** for PostgreSQL connections

```yaml
# Update Iceberg REST to use SSL for PostgreSQL
CATALOG_URI: "jdbc:postgresql://postgres-shared-service:5432/iceberg_rest?ssl=true&sslmode=require"
```

## 5. Audit Logging

### 5.1 Enable Audit Logs

```yaml
# Kubernetes audit logging (if not already enabled)
# Edit kubelet config to enable audit logging

# Application logging
# Ensure all components log to centralized system:
# - Elasticsearch
# - Loki
# - CloudWatch

# Iceberg REST Catalog logging
env:
- name: LOGGING_LEVEL
  value: "INFO"
- name: AUDIT_LOG_ENABLED
  value: "true"
```

### 5.2 Monitor for Security Events

```bash
# Monitor for suspicious activities:
# 1. Failed authentication attempts
# 2. Unauthorized access attempts
# 3. Unusual data access patterns
# 4. Configuration changes

# Query logs via Kibana/Loki:
# Failed logins: "status=unauthorized"
# Suspicious queries: "SELECT * FROM information_schema"
```

## 6. Secret Management

### 6.1 Use Vault for Secret Storage

**Integrate with Vault (if deployed):**

```bash
# Store Iceberg credentials in Vault
vault kv put secret/iceberg/minio \
  access-key=minioadmin \
  secret-key=minioadmin123

vault kv put secret/iceberg/database \
  username=iceberg_user \
  password=iceberg_password

# Update deployments to fetch from Vault
# Use external-secrets operator for automatic sync
```

### 6.2 Rotate Credentials Regularly

```bash
#!/bin/bash
# Script: rotate-iceberg-credentials.sh

# Rotate MinIO credentials
MC_HOST=minio-service:9000
NEW_PASS=$(openssl rand -base64 32)

# Update MinIO
mc admin user change-password $MC_HOST minioadmin $NEW_PASS

# Update Kubernetes secret
kubectl create secret generic minio-secret \
  -n data-platform \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=$NEW_PASS \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new credentials
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform

echo "Credentials rotated successfully"
```

## 7. Compliance and Governance

### 7.1 Data Classification

```yaml
# Tag data in DataHub with classification
datasets:
  - urn: "urn:li:dataset:urn:li:platform:iceberg:raw.customers"
    tags:
      - "pii"
      - "confidential"
      - "retention:7years"
```

### 7.2 Data Retention Policies

```sql
-- Set Iceberg retention policies
-- Expire old snapshots
CALL iceberg.system.expire_snapshots('raw.customers', INTERVAL '90' DAY);

-- Remove orphan files
CALL iceberg.system.remove_orphan_files('raw.customers');
```

### 7.3 Data Privacy

- **PII Protection**: Mask sensitive columns
- **Data Anonymization**: Apply anonymization rules
- **Access Logging**: Log all data access
- **Compliance Scanning**: Regular data classification audits

## 8. Security Checklist

### Pre-Production

- [ ] Change all default credentials
- [ ] Enable TLS/HTTPS
- [ ] Configure network policies
- [ ] Enable audit logging
- [ ] Set up secret management
- [ ] Configure RBAC
- [ ] Enable data encryption at rest
- [ ] Enable encryption in transit
- [ ] Configure backup encryption
- [ ] Document security procedures

### Ongoing

- [ ] Rotate credentials monthly
- [ ] Review audit logs weekly
- [ ] Patch security vulnerabilities
- [ ] Monitor for unauthorized access
- [ ] Test disaster recovery procedures
- [ ] Conduct security audits quarterly
- [ ] Update security policies

## 9. Security Tools

### Recommended Tools

1. **Secret Management**: Vault, Sealed Secrets
2. **Network Security**: Istio, Calico
3. **RBAC**: Kubernetes native, DataHub policies
4. **Monitoring**: Prometheus, Loki, Elasticsearch
5. **Scanning**: Trivy, Falco, OPA/Gatekeeper

### Implementation

```bash
# Install security tools
helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system
helm install falco falcosecurity/falco -n falco --create-namespace
helm install kyverno kyverno/kyverno -n kyverno --create-namespace
```

## References

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Apache Iceberg Security](https://iceberg.apache.org/docs/nightly/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)

## Support

For security concerns or incidents:
1. Contact security team
2. Review audit logs
3. Implement remediation
4. Document incident
5. Update security policies
