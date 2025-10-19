# HashiCorp Vault Security & Production Setup Guide

## Current Setup Status

✅ **Development Environment**:
- Vault 1.13.3 running in dev mode (data-platform namespace)
- Kubernetes authentication configured
- Audit logging enabled
- Policies and roles created

⚠️ **Not Production-Ready**:
- In-memory storage (data lost on restart)
- No TLS/HTTPS
- Single instance (no HA)
- No persistent backup

---

## Vault Policies Created

### 1. Admin Policy
Full access to all Vault operations. Use sparingly.

```bash
vault policy read admin
```

**Use Case**: Cluster administrators only

### 2. SeaTunnel Policy
Read access to data source and SeaTunnel secrets.

```bash
vault policy read seatunnel
```

**Granted to**:
- SeaTunnel/Flink pods
- Data integration jobs
- Data processing services

### 3. Database Admin Policy
Full management of database credentials.

```bash
vault policy read database-admin
```

**Granted to**:
- Database administrators
- DBA service accounts

### 4. Read-only Policy
Minimal read access to secrets and health checks.

```bash
vault policy read readonly
```

**Granted to**:
- Monitoring systems
- Read-only applications

---

## Authentication Methods

### Kubernetes Authentication (Active)

**Status**: ✅ Enabled and configured

**Service Roles**:

1. **seatunnel**
   - Service Accounts: `default`, `seatunnel`, `flink`
   - Namespace: `data-platform`
   - Policies: `seatunnel`, `readonly`
   - Token TTL: 24 hours

2. **database-admin**
   - Service Accounts: `default`
   - Namespace: `data-platform`
   - Policies: `database-admin`, `readonly`
   - Token TTL: 1 hour

**How it Works**:
1. Pod requests Kubernetes JWT token from ServiceAccount
2. Pod presents JWT to Vault Kubernetes auth method
3. Vault validates JWT with Kubernetes API
4. Vault issues token based on bound policies
5. Pod uses token to access secrets

**Example**:
```bash
# From within a pod in data-platform namespace
JWT=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
curl --request POST \
  --data '{"jwt": "'$JWT'", "role": "seatunnel"}' \
  http://vault:8200/v1/auth/kubernetes/login
```

### AppRole Authentication (Recommended Setup)

For service-to-service communication without pod identity.

```bash
# Create AppRole
vault auth enable approle
vault write auth/approle/role/seatunnel-job \
  token_ttl=1h \
  token_max_ttl=4h \
  policies="seatunnel,readonly"

# Get Role ID and Secret ID
vault read auth/approle/role/seatunnel-job/role-id
vault write -f auth/approle/role/seatunnel-job/secret-id

# Login
curl --request POST \
  --data '{
    "role_id":"<role-id>",
    "secret_id":"<secret-id>"
  }' \
  http://vault:8200/v1/auth/approle/login
```

---

## Secrets Management

### Stored Secrets

**Location**: `secret/datasources/` and `secret/seatunnel/`

```bash
# List all secrets
vault kv list secret/

# Read PostgreSQL credentials
vault kv get secret/datasources/postgres

# Read MySQL credentials
vault kv get secret/datasources/mysql

# Read Kafka configuration
vault kv get secret/seatunnel/kafka

# Read Elasticsearch configuration
vault kv get secret/seatunnel/elasticsearch
```

### Adding New Secrets

```bash
# Store new database credentials
vault kv put secret/datasources/newdb \
  host="service.namespace.svc.cluster.local" \
  port="5432" \
  username="admin" \
  password="secure_password"

# Store API keys
vault kv put secret/seatunnel/api-keys \
  github-token="ghp_..." \
  slack-webhook="https://hooks.slack.com/..."

# Verify storage
vault kv get secret/seatunnel/api-keys
```

### Secret Versioning

KV v2 automatically versions secrets:

```bash
# View version metadata
vault kv metadata get secret/datasources/postgres

# List all versions
vault kv metadata get secret/datasources/postgres

# Restore previous version
vault kv put secret/datasources/postgres @old_version.json

# Permanently delete version
vault kv metadata delete secret/datasources/postgres
```

---

## Production Deployment Guide

### Prerequisites

1. **PostgreSQL Database**
   - Create `vault` database
   - Create `vault` user with password
   - Grant privileges

```sql
CREATE DATABASE vault;
CREATE USER vault WITH PASSWORD 'vault-secure-password-change-me';
GRANT ALL PRIVILEGES ON DATABASE vault TO vault;
```

2. **TLS Certificates**
   - Generate self-signed or use CA certificates
   - Mount as Kubernetes Secret

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout vault.key \
  -out vault.crt \
  -subj "/CN=vault.vault-prod.svc.cluster.local"

# Create secret
kubectl create secret tls vault-tls \
  --cert=vault.crt \
  --key=vault.key \
  -n vault-prod
```

3. **Storage Class**
   - Ensure `local-storage-standard` exists
   - Or modify `vault-production.yaml`

### Deployment Steps

1. **Prepare Configuration**
   - Update `vault-production.yaml`
   - Set PostgreSQL connection string
   - Set secure TLS certificates

2. **Deploy Vault**
   ```bash
   kubectl apply -f k8s/vault/vault-production.yaml
   ```

3. **Unseal Vault**
   ```bash
   kubectl exec -n vault-prod vault-0 -- vault operator init
   # Save unseal keys securely!
   
   # Unseal each pod
   kubectl exec -n vault-prod vault-0 -- vault operator unseal <unseal-key>
   kubectl exec -n vault-prod vault-1 -- vault operator unseal <unseal-key>
   kubectl exec -n vault-prod vault-2 -- vault operator unseal <unseal-key>
   ```

4. **Migrate Data** (if upgrading from dev)
   ```bash
   # Backup dev secrets
   kubectl exec -n data-platform vault-dev -- \
     vault kv list -format=json secret/ > backup.json
   
   # Restore to production
   # Use vault migration tools
   ```

5. **Enable High Availability**
   ```bash
   # HA is configured automatically with PostgreSQL backend
   # Verify:
   kubectl exec -n vault-prod vault-0 -- vault status | grep -i ha
   ```

---

## Monitoring & Maintenance

### Audit Logs

View all authenticated actions:

```bash
# From Vault pod
tail -f /vault/data/audit.log

# Parse JSON audit logs
cat /vault/data/audit.log | jq '.requests[] | select(.type=="WRITE")'

# Monitor specific paths
cat /vault/data/audit.log | jq '.requests[] | select(.path=="secret/data/*")'
```

### Health Checks

```bash
# Check Vault status
kubectl exec -n data-platform deployment/vault -- vault status

# Check sealed status
kubectl exec -n data-platform deployment/vault -- vault status | grep -i sealed

# Check storage health
kubectl exec -n data-platform deployment/vault -- vault debug

# Check cluster status
kubectl exec -n data-platform deployment/vault -- \
  vault read /sys/metrics
```

### Backup & Recovery

**Backup Strategy**:

```bash
# 1. Export all secrets (KV v2)
vault kv list -format=json secret/ > secrets-backup.json

# 2. Export configuration
vault read -format=json /sys/mounts > mounts-backup.json
vault policy list -format=json > policies-backup.json

# 3. Backup PostgreSQL database
pg_dump -h postgres-shared-service vault > vault-db-backup.sql
```

**Recovery Steps**:

1. Restore PostgreSQL database
2. Restart Vault instances
3. Unseal Vault
4. Restore policies
5. Restore secret mounts
6. Restore secrets

---

## Security Best Practices

### 1. Token Management

- Use short TTLs (1-24 hours depending on use case)
- Implement token refresh logic
- Revoke tokens on service termination

```bash
# Set short TTL for sensitive operations
vault token create -ttl=15m -policy=database-admin

# Revoke token
vault token revoke <token>
```

### 2. Secret Rotation

- Implement automatic credential rotation
- Maintain audit trail
- Test rotation before deployment

```bash
# Rotate database password
vault kv patch secret/datasources/postgres \
  password="new_secure_password"

# Verify
vault kv get secret/datasources/postgres
```

### 3. Audit Logging

- Enable and monitor all access
- Ship logs to SIEM
- Alert on suspicious patterns

```bash
# Enable syslog
vault audit enable syslog tag="vault" facility="LOCAL7"

# Enable multiple backends
vault audit enable file file_path=/vault/logs/audit.log
vault audit enable splunk token="<splunk-token>"
```

### 4. Access Control

- Principle of least privilege
- Use separate policies per role
- Implement MFA for sensitive operations

```bash
# Create restrictive policy
vault policy write restricted-user - << 'EOF'
path "secret/data/datasources/postgres" {
  capabilities = ["read"]
}
EOF
```

### 5. Network Security

- Use TLS for all communications
- Implement network policies
- Restrict Vault access by IP

```yaml
# Example Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vault-access
  namespace: vault-prod
spec:
  podSelector:
    matchLabels:
      app: vault
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: data-platform
    ports:
    - protocol: TCP
      port: 8200
```

### 6. TLS Certificate Management

- Use cert-manager for automatic renewal
- Implement certificate pinning
- Monitor expiration

```bash
# Using cert-manager
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: vault-cert
  namespace: vault-prod
spec:
  secretName: vault-tls
  duration: 2160h # 90d
  renewBefore: 360h # 15d
  commonName: vault.vault-prod.svc.cluster.local
  dnsNames:
  - vault.vault-prod.svc.cluster.local
  - vault.vault-prod
  issuerRef:
    name: self-signed
    kind: Issuer
```

---

## Troubleshooting

### Pod Stuck in Init

```bash
# Check logs
kubectl logs -n data-platform deployment/vault

# Check PostgreSQL connectivity
kubectl exec -n data-platform deployment/vault -- \
  pg_isready -h postgres-shared-service -p 5432
```

### Authentication Failing

```bash
# Verify Kubernetes auth config
vault read auth/kubernetes/config

# Check service account
kubectl get serviceaccount -n data-platform vault

# Test JWT
kubectl exec -n data-platform deployment/vault -- \
  cat /var/run/secrets/kubernetes.io/serviceaccount/token
```

### Secrets Not Accessible

```bash
# Verify policy
vault policy read seatunnel

# Check user policies
vault token lookup

# Verify secret path exists
vault kv list secret/datasources/
```

---

## Migration Checklist

- [ ] PostgreSQL database created and accessible
- [ ] TLS certificates generated and stored
- [ ] Production Vault manifest reviewed
- [ ] Policies exported and tested
- [ ] Secrets backed up
- [ ] Network policies configured
- [ ] Monitoring/alerts set up
- [ ] Disaster recovery plan documented
- [ ] Team trained on Vault operations
- [ ] Production deployment approved

---

## References

- [Vault Official Documentation](https://www.vaultproject.io/docs)
- [Vault Kubernetes Auth](https://www.vaultproject.io/docs/auth/kubernetes)
- [PostgreSQL Storage Backend](https://www.vaultproject.io/docs/configuration/storage/postgresql)
- [Production Hardening](https://www.vaultproject.io/docs/internals/security)
- [Disaster Recovery](https://www.vaultproject.io/docs/concepts/disaster-recovery)
