# Vault Production Deployment Guide

**Date**: 2025-10-19
**Status**: Ready for deployment with corrections

---

## Deployment Progress

✅ **Completed**:
- PostgreSQL vault database created in data-platform namespace
- PostgreSQL user 'vault' created with password
- vault-prod namespace created
- TLS self-signed certificates generated and deployed
- Production manifests created
- Backup procedures documented
- Development secrets backed up

⚠️ **Current Status**:
- Production deployment applied but needs connection string fix
- PostgreSQL connection requires correct URL format with encoded password

---

## Prerequisites Checklist

```markdown
- [x] PostgreSQL database 'vault' created
- [x] PostgreSQL user 'vault' created with password
- [x] TLS certificates generated
- [x] TLS secret deployed (vault-tls)
- [x] vault-prod namespace created
- [x] Development vault running with configured policies
- [x] Secrets backed up
- [x] Service accounts and RBAC configured

Next:
- [ ] Fix PostgreSQL connection string in vault-config ConfigMap
- [ ] Initialize Vault (vault operator init)
- [ ] Unseal all 3 instances
- [ ] Migrate policies from development
- [ ] Migrate secrets from development
- [ ] Configure Kubernetes auth in production
- [ ] Verify HA setup
- [ ] Configure monitoring and backups
```

---

## PostgreSQL Connection Fix

### Issue Found
The ConfigMap has placeholder connection string. Update it with actual credentials:

```bash
# Get the password from the secret
kubectl get secret vault-postgres -n vault-prod -o jsonpath='{.data.password}' | base64 -d

# Update ConfigMap with correct connection string:
postgresql://vault:vault-secure-password-123@postgres-shared-service.data-platform:5432/vault?sslmode=disable
```

### Steps to Fix

1. **Delete failed deployment**:
```bash
kubectl delete deployment vault -n vault-prod
kubectl delete service vault vault-internal -n vault-prod
kubectl delete pvc vault-data -n vault-prod
```

2. **Update ConfigMap**:
```bash
kubectl delete configmap vault-config -n vault-prod

kubectl create configmap vault-config -n vault-prod \
  --from-literal=vault.hcl='
storage "postgresql" {
  connection_string = "postgresql://vault:vault-secure-password-123@postgres-shared-service.data-platform:5432/vault?sslmode=disable"
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/tls.crt"
  tls_key_file = "/vault/tls/tls.key"
}

ui = true
log_level = "info"
api_addr = "https://vault.vault-prod.svc.cluster.local:8200"
'
```

3. **Redeploy**:
```bash
kubectl apply -f k8s/vault/vault-production.yaml
```

---

## Initialization Steps

### 1. Initialize Vault

Once pods are running:

```bash
# Initialize Vault (generates unseal keys and root token)
kubectl exec -n vault-prod vault-0 -- vault operator init \
  -key-shares=5 \
  -key-threshold=3 \
  -format=json > vault-keys.json

# **CRITICAL**: Save vault-keys.json securely!
# Store unseal keys separately (different locations/people)
# Store root token securely (e.g., in password manager)
```

### 2. Unseal All Instances

```bash
# Extract first unseal key
UNSEAL_KEY=$(jq -r '.unseal_keys_b64[0]' vault-keys.json)

# Unseal each pod
for i in {0..2}; do
  kubectl exec -n vault-prod vault-$i -- vault operator unseal $UNSEAL_KEY
  kubectl exec -n vault-prod vault-$i -- vault operator unseal $UNSEAL_KEY
  kubectl exec -n vault-prod vault-$i -- vault operator unseal $UNSEAL_KEY
done
```

### 3. Verify HA Status

```bash
# Check HA is enabled
kubectl exec -n vault-prod vault-0 -- vault status | grep -i "ha enabled"

# Should show: HA Enabled             true

# Check leader election
kubectl exec -n vault-prod vault-0 -- vault status | grep -i leader
```

---

## Migration from Development to Production

### Step 1: Backup Development Secrets

```bash
# Export all development secrets
kubectl exec -n data-platform vault-dev -- \
  vault kv list secret/ --format=json > dev-secrets-list.json

# Export each secret
vault kv get secret/datasources/postgres > postgres-creds.json
vault kv get secret/datasources/mysql > mysql-creds.json
vault kv get secret/seatunnel/kafka > kafka-config.json
vault kv get secret/seatunnel/elasticsearch > elasticsearch-config.json
```

### Step 2: Restore Policies to Production

```bash
# Get root token
ROOT_TOKEN=$(jq -r '.root_token' vault-keys.json)

# Set token
export VAULT_TOKEN=$ROOT_TOKEN

# Create policies in production
vault policy write admin << 'EOF'
path "*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}
EOF

vault policy write seatunnel << 'EOF'
path "secret/data/seatunnel/*" {
  capabilities = ["read", "list"]
}
path "secret/data/datasources/*" {
  capabilities = ["read", "list"]
}
path "sys/leases/renew" {
  capabilities = ["update"]
}
EOF

# ... repeat for other policies
```

### Step 3: Restore Secrets to Production

```bash
# Restore PostgreSQL credentials
vault kv put secret/datasources/postgres \
  host="postgres-shared-service.data-platform.svc.cluster.local" \
  port="5432" \
  username="datahub" \
  password="your_password"

# Restore MySQL credentials
vault kv put secret/datasources/mysql \
  host="mysql-seatunnel-service.data-platform.svc.cluster.local" \
  port="3306" \
  username="seatunnel" \
  password="your_password"

# Restore Kafka config
vault kv put secret/seatunnel/kafka \
  bootstrap-servers="kafka-service.data-platform.svc.cluster.local:9092" \
  security-protocol="PLAINTEXT"

# Restore Elasticsearch config
vault kv put secret/seatunnel/elasticsearch \
  host="elasticsearch-service.data-platform.svc.cluster.local" \
  port="9200" \
  protocol="http"
```

### Step 4: Configure Kubernetes Auth in Production

```bash
vault auth enable kubernetes

vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443" \
  kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token

vault write auth/kubernetes/role/seatunnel \
  bound_service_account_names="default,seatunnel,flink" \
  bound_service_account_namespaces="data-platform" \
  policies="seatunnel,readonly" \
  ttl=24h
```

---

## High Availability Setup

### Verify HA is Working

```bash
# Kill the leader pod
kubectl delete pod vault-0 -n vault-prod

# Check new leader is elected
kubectl exec -n vault-prod vault-1 -- vault status | grep -i "is_leader"

# Should show: Is Leader         true
```

### Configure Monitoring

```bash
# Add Prometheus scrape config
# Point to: http://vault.vault-prod.svc.cluster.local:8200/v1/sys/metrics?format=prometheus
```

---

## Backup Strategy

### Automated Backups

```bash
# Create daily backup job
cat > vault-backup-cronjob.yaml << 'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: vault-backup
  namespace: vault-prod
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: vault
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/sh
            - -c
            - |
              PGPASSWORD=vault-secure-password-123 pg_dump \
                -h postgres-shared-service.data-platform \
                -U vault \
                -d vault \
                > /backups/vault-$(date +%Y%m%d).sql
          restartPolicy: OnFailure
EOF
```

### Recovery

```bash
# Restore from backup
PGPASSWORD=vault-secure-password-123 psql \
  -h postgres-shared-service.data-platform \
  -U vault \
  -d vault \
  < vault-20251019.sql
```

---

## Monitoring & Health

### Health Check

```bash
# Check all instances
for i in {0..2}; do
  echo "=== Vault-$i ==="
  kubectl exec -n vault-prod vault-$i -- vault status
done
```

### Metrics Collection

```bash
# Get metrics (Prometheus format)
curl -k https://vault.vault-prod.svc.cluster.local:8200/v1/sys/metrics?format=prometheus
```

### Audit Logging

```bash
# Check audit logs
kubectl exec -n vault-prod vault-0 -- tail -f /vault/data/audit.log

# Or from pod
kubectl logs -n vault-prod vault-0 -f
```

---

## Troubleshooting

### Pods in CrashLoopBackOff

```bash
# Check logs
kubectl logs -n vault-prod vault-0 -p

# Common issues:
# 1. PostgreSQL connection string incorrect
# 2. TLS certificate invalid
# 3. PVC not binding
# 4. Storage class not available
```

### Cannot Unseal

```bash
# Verify unseal key is correct
echo $UNSEAL_KEY | wc -c  # Should be ~88 chars (base64)

# Check pod logs for specific error
kubectl describe pod -n vault-prod vault-0
```

### HA Not Working

```bash
# Check storage backend is PostgreSQL
vault read sys/mounts

# Verify each instance can reach PostgreSQL
kubectl exec -n vault-prod vault-0 -- \
  pg_isready -h postgres-shared-service.data-platform -U vault
```

---

## Switching Applications to Production

### Update Service URLs

Update your applications to point to production Vault:

```
Old (Development): http://vault.data-platform.svc.cluster.local:8200
New (Production): https://vault.vault-prod.svc.cluster.local:8200
```

### Update Kubernetes Auth Bindings

```bash
# If using different service accounts in production
vault write auth/kubernetes/role/production-apps \
  bound_service_account_names="prod-app" \
  bound_service_account_namespaces="production" \
  policies="seatunnel,readonly" \
  ttl=24h
```

---

## Disaster Recovery

### Complete Vault Recovery

1. Restore PostgreSQL database from backup
2. Delete Vault pods (they'll auto-restart)
3. Unseal all instances
4. Verify data is restored

### Corrupted Unseal Keys

If unseal keys are lost:
1. Contact HashiCorp support
2. Or: Use Shamir key recovery procedures
3. Or: Restore from backup + re-initialize

---

## Checklist for Go-Live

- [ ] PostgreSQL database verified working
- [ ] TLS certificates validated
- [ ] All 3 Vault pods running and sealed=false
- [ ] HA verified (leader election works)
- [ ] All policies migrated from development
- [ ] All secrets migrated from development
- [ ] Kubernetes auth configured
- [ ] Application URLs updated to production
- [ ] Backups configured and tested
- [ ] Monitoring/alerts configured
- [ ] Team trained on operations
- [ ] Runbooks documented
- [ ] Disaster recovery tested
- [ ] Sign-off from security team

---

## Support & References

- Official Vault Docs: https://www.vaultproject.io/docs
- PostgreSQL Backend: https://www.vaultproject.io/docs/configuration/storage/postgresql
- Kubernetes Auth: https://www.vaultproject.io/docs/auth/kubernetes
- HA Setup: https://www.vaultproject.io/docs/concepts/ha
- DR Procedures: https://www.vaultproject.io/docs/concepts/disaster-recovery
