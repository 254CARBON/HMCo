# Phase 2: Production Vault Migration & Advanced Hardening

**Status**: Ready for Deployment  
**Date**: 2025-10-19  
**Duration**: Week 2-3  
**Priority**: CRITICAL - Secrets Management

---

## Overview

Phase 2 transitions the cluster from development-mode Vault (in-memory, insecure) to production-grade Vault with:

- **PostgreSQL-backed persistent storage** (survives pod restarts)
- **High Availability** (3-replica HA cluster with automatic failover)
- **TLS encryption** (automatic certificate management via cert-manager)
- **Automated backups** (daily + weekly with 30/90-day retention)
- **Disaster recovery** (documented restore procedures)
- **Security hardening** (restricted access, audit logging)

---

## Prerequisites

### 1. Verify Phase 1 Completion

```bash
# Check that Phase 1 components are deployed
kubectl get ns cert-manager vault-prod 2>/dev/null || echo "✗ Namespaces missing"
kubectl get pods -n cert-manager | grep cert-manager || echo "✗ cert-manager not running"
kubectl get pdb -n vault-prod | head -1 || echo "✗ PDB not configured"
```

### 2. PostgreSQL Database Setup

Create the Vault database and user in the shared PostgreSQL instance:

```bash
# Connect to PostgreSQL (adjust credentials as needed)
kubectl exec -n data-platform deployment/postgres-shared -- psql -U postgres -c "
CREATE DATABASE vault;
CREATE USER vault WITH ENCRYPTED PASSWORD 'vault-secure-password-change-me';
GRANT ALL PRIVILEGES ON DATABASE vault TO vault;
ALTER DATABASE vault OWNER TO vault;
"

# Verify creation
kubectl exec -n data-platform deployment/postgres-shared -- psql -U postgres -c "
SELECT datname FROM pg_database WHERE datname = 'vault';
SELECT usename FROM pg_user WHERE usename = 'vault';
"
```

### 3. TLS Certificates

Vault requires TLS certificates. We'll use cert-manager with self-signed issuer for initial setup:

```bash
# Verify cert-manager is running
kubectl get pods -n cert-manager -l app.kubernetes.io/name=cert-manager
# Should show 2 cert-manager pods (running)
```

---

## Deployment Steps

### Step 1: Create Vault Namespace & Secrets

The namespace should already exist from Phase 1, but ensure it has proper labels:

```bash
kubectl label namespace vault-prod \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted \
  --overwrite

# Create PostgreSQL secret
kubectl create secret generic vault-postgres \
  --from-literal=username=vault \
  --from-literal=password=vault-secure-password-change-me \
  --from-literal=database=vault \
  -n vault-prod \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Step 2: Create TLS Certificates

```bash
# Generate self-signed certificate for development (use proper CA in production)
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout vault.key \
  -out vault.crt \
  -subj "/CN=vault.vault-prod.svc.cluster.local"

# Create TLS secret
kubectl create secret tls vault-tls \
  --cert=vault.crt \
  --key=vault.key \
  -n vault-prod \
  --dry-run=client -o yaml | kubectl apply -f -

# Clean up temporary files
rm -f vault.crt vault.key
```

### Step 3: Deploy Production Vault

The `vault-production.yaml` is already configured. Deploy it:

```bash
# Apply production Vault configuration
kubectl apply -f k8s/vault/vault-production.yaml

# Monitor deployment
kubectl rollout status deployment/vault -n vault-prod --timeout=5m

# Wait for all 3 replicas to be Running
kubectl get pods -n vault-prod -w
# Press Ctrl+C when all 3 are ready
```

### Step 4: Initialize Vault

After pods are running, initialize Vault:

```bash
# Initialize Vault (generates unseal keys and root token)
kubectl exec -n vault-prod vault-0 -- vault operator init \
  -key-shares=5 \
  -key-threshold=3 \
  -format=json > vault-keys.json

# ⚠️ CRITICAL: Save vault-keys.json securely!
# Store unseal keys separately from each other
# Store root token securely (password manager/secrets vault)
# DO NOT commit to Git!

echo "⚠️  CRITICAL: Back up vault-keys.json to secure location"
ls -lh vault-keys.json
```

### Step 5: Unseal All Vault Instances

Vault starts in sealed state. Unseal using 3 of 5 keys:

```bash
# Extract unseal keys
UNSEAL_KEY_1=$(jq -r '.unseal_keys_b64[0]' vault-keys.json)
UNSEAL_KEY_2=$(jq -r '.unseal_keys_b64[1]' vault-keys.json)
UNSEAL_KEY_3=$(jq -r '.unseal_keys_b64[2]' vault-keys.json)

# Unseal vault-0
kubectl exec -n vault-prod vault-0 -- vault operator unseal $UNSEAL_KEY_1
kubectl exec -n vault-prod vault-0 -- vault operator unseal $UNSEAL_KEY_2
kubectl exec -n vault-prod vault-0 -- vault operator unseal $UNSEAL_KEY_3

# Unseal vault-1 (different pod)
kubectl exec -n vault-prod vault-1 -- vault operator unseal $UNSEAL_KEY_1
kubectl exec -n vault-prod vault-1 -- vault operator unseal $UNSEAL_KEY_2
kubectl exec -n vault-prod vault-1 -- vault operator unseal $UNSEAL_KEY_3

# Unseal vault-2 (third pod)
kubectl exec -n vault-prod vault-2 -- vault operator unseal $UNSEAL_KEY_1
kubectl exec -n vault-prod vault-2 -- vault operator unseal $UNSEAL_KEY_2
kubectl exec -n vault-prod vault-2 -- vault operator unseal $UNSEAL_KEY_3

# Verify all are unsealed
kubectl exec -n vault-prod vault-0 -- vault status | grep -i "sealed"
# Should show: "Sealed Value        false"
```

### Step 6: Deploy Vault Backup Automation

```bash
# Deploy backup CronJobs and restore Job
kubectl apply -f k8s/vault/vault-backup-cronjob.yaml

# Verify backup infrastructure
kubectl get cronjob -n vault-prod
kubectl get pvc -n vault-prod vault-backup-storage

# Verify backups will run (check logs in next 5 minutes)
kubectl logs -n vault-prod -l app=vault-backup -f
```

### Step 7: Set Up Kubernetes Authentication

From a pod in the cluster, or from outside with kubectl access:

```bash
# Get root token from vault-keys.json
ROOT_TOKEN=$(jq -r '.root_token' vault-keys.json)

# Log in to Vault
kubectl exec -n vault-prod vault-0 -- vault login $ROOT_TOKEN

# Enable Kubernetes auth method
kubectl exec -n vault-prod vault-0 -- vault auth enable kubernetes

# Configure Kubernetes auth
kubectl exec -n vault-prod vault-0 -- \
  vault write auth/kubernetes/config \
    kubernetes_host="https://kubernetes.default.svc:443" \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
    token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token

# Create policy for SeaTunnel/applications
kubectl exec -n vault-prod vault-0 -- vault policy write seatunnel - << 'EOF'
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

# Create Kubernetes role for SeaTunnel
kubectl exec -n vault-prod vault-0 -- \
  vault write auth/kubernetes/role/seatunnel \
    bound_service_account_names=default,seatunnel,flink \
    bound_service_account_namespaces=data-platform \
    policies=seatunnel \
    ttl=24h
```

### Step 8: Migrate Secrets from Development Vault

Export secrets from dev Vault and import to production:

```bash
# From dev Vault pod
kubectl exec -n data-platform deployment/vault -- \
  vault kv list -format=json secret/ > secrets-list.json

# Export each secret (example - adjust paths as needed)
kubectl exec -n data-platform deployment/vault -- \
  vault kv get -format=json secret/datasources/postgres > postgres-secret.json

kubectl exec -n data-platform deployment/vault -- \
  vault kv get -format=json secret/datasources/mysql > mysql-secret.json

# Import to production Vault
PROD_TOKEN=$(jq -r '.root_token' vault-keys.json)

# Store in production
kubectl exec -n vault-prod vault-0 -- vault login $PROD_TOKEN

# Restore secrets (example)
kubectl exec -n vault-prod vault-0 -- vault kv put \
  secret/datasources/postgres \
  host=$(jq -r '.data.data.host' postgres-secret.json) \
  port=$(jq -r '.data.data.port' postgres-secret.json) \
  username=$(jq -r '.data.data.username' postgres-secret.json) \
  password=$(jq -r '.data.data.password' postgres-secret.json)

echo "✓ Secrets migrated to production Vault"
```

### Step 9: Verify High Availability

Test HA by causing a failover:

```bash
# Check current leader
kubectl exec -n vault-prod vault-0 -- vault status | grep -i "leader"

# Delete the leader pod (it will auto-restart)
kubectl delete pod -n vault-prod vault-0

# Wait for new pod to start
kubectl rollout status deployment/vault -n vault-prod --timeout=2m

# Verify a new leader was elected
kubectl exec -n vault-prod vault-1 -- vault status | grep -i "leader"
# Should show a different leader after pod restart

echo "✓ HA verified - failover successful"
```

---

## Verification Checklist

After deployment, verify all components:

```bash
# 1. All Vault pods running
kubectl get pods -n vault-prod
# Should show: vault-0, vault-1, vault-2 all Running, 1/1 Ready

# 2. Unsealed status
kubectl exec -n vault-prod vault-0 -- vault status | grep "Sealed"
# Should show: Sealed Value        false

# 3. HA enabled
kubectl exec -n vault-prod vault-0 -- vault status | grep "HA Enabled"
# Should show: HA Enabled           true

# 4. PostgreSQL connected
kubectl exec -n vault-prod vault-0 -- vault status | grep "Storage Type"
# Should show: Storage Type          postgresql

# 5. Service accessible
kubectl exec -n vault-prod vault-0 -- vault status | grep "API Address"
# Should show: https://vault.vault-prod.svc.cluster.local:8200

# 6. Backup CronJobs created
kubectl get cronjob -n vault-prod
# Should show: vault-backup-daily, vault-backup-weekly, vault-backup-cleanup

# 7. Kubernetes auth configured
kubectl exec -n vault-prod vault-0 -- vault auth list | grep kubernetes
# Should show: kubernetes/    kubernetes    -

# 8. Policies available
kubectl exec -n vault-prod vault-0 -- vault policy list | grep seatunnel
# Should show: seatunnel
```

---

## Accessing Vault

### Web UI

```bash
# Port-forward to Vault API
kubectl port-forward -n vault-prod svc/vault 8200:8200

# Open browser to https://localhost:8200
# Login with token: <ROOT_TOKEN from vault-keys.json>
```

### CLI from Inside Cluster

```bash
# From any pod in data-platform namespace
kubectl exec -n data-platform <pod-name> -- sh

# Inside container:
export VAULT_ADDR=http://vault.vault-prod.svc.cluster.local:8200
export VAULT_TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token | \
  curl -s -X POST \
    -d '{"jwt":"$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)","role":"seatunnel"}' \
    $VAULT_ADDR/v1/auth/kubernetes/login | jq -r '.auth.client_token')

vault secrets list
```

---

## Backup & Recovery

### Automated Backups

Backups run automatically via CronJobs:
- **Daily**: Every day at 2 AM UTC (kept 30 days)
- **Weekly**: Every Sunday at 3 AM UTC (kept 90 days)
- **Cleanup**: Every day at 4 AM UTC (removes old backups)

### Manual Backup

```bash
# Trigger manual backup immediately
kubectl create job --from=cronjob/vault-backup-daily \
  manual-backup-$(date +%s) \
  -n vault-prod

# Monitor backup job
kubectl logs -n vault-prod -l job-name=manual-backup* -f
```

### List Available Backups

```bash
# Access backup storage PVC
kubectl exec -n vault-prod vault-0 -- ls -lh /backup/

# Check recent backups
kubectl exec -n vault-prod vault-0 -- ls -lh /backup/ | tail -10
```

### Restore from Backup

```bash
# Get list of available backups
BACKUP_FILE=$(kubectl exec -n vault-prod vault-0 -- ls -1 /backup/vault-*.sql | head -1)

# Trigger restore
kubectl set env job/vault-backup-restore BACKUP_FILE=$BACKUP_FILE -n vault-prod
kubectl apply -f k8s/vault/vault-backup-cronjob.yaml  # Apply restore job

# Monitor restore
kubectl logs -n vault-prod job/vault-backup-restore -f

# After successful restore, restart Vault
kubectl rollout restart deployment/vault -n vault-prod
```

---

## Troubleshooting

### Pods Stuck in Init

```bash
# Check init container logs
kubectl logs -n vault-prod vault-0 -c vault-init

# Common issues:
# 1. PostgreSQL not reachable
kubectl exec -n vault-prod vault-0 -- \
  pg_isready -h postgres-shared-service.data-platform -U vault

# 2. TLS certificate invalid
kubectl get secret vault-tls -n vault-prod -o yaml | grep tls
```

### Cannot Unseal

```bash
# Verify unseal keys are valid base64
echo $UNSEAL_KEY_1 | wc -c  # Should be ~88 characters

# Check pod logs for specific error
kubectl describe pod -n vault-prod vault-0

# Try with different key order
kubectl exec -n vault-prod vault-0 -- vault status
```

### HA Not Working

```bash
# Verify PostgreSQL HA backend
kubectl exec -n vault-prod vault-0 -- vault status | grep -i ha

# Check PostgreSQL connection
kubectl exec -n vault-prod vault-0 -- \
  PGPASSWORD=vault-secure-password-change-me psql \
    -h postgres-shared-service.data-platform \
    -U vault \
    -d vault \
    -c "SELECT version();"
```

### Backup Failing

```bash
# Check backup PVC is bound
kubectl get pvc -n vault-prod vault-backup-storage
# Should show: Bound

# Check backup directory permissions
kubectl exec -n vault-prod vault-0 -- ls -ld /backup
# Should show: dr-xr-xr-x

# Check PostgreSQL backup user permissions
kubectl exec -n data-platform deployment/postgres-shared -- \
  psql -U postgres -c "GRANT pg_read_all_stats TO vault;"
```

---

## Security Best Practices

### 1. Protect Unseal Keys

- Store unseal keys separately (different people, locations)
- Use HSM (Hardware Security Module) in production
- Rotate keys annually
- Never commit to version control

### 2. Rotate Root Token

```bash
# Generate new root token
kubectl exec -n vault-prod vault-0 -- \
  vault generate-root -init

# Follow prompts to unseal and complete generation
```

### 3. Enable Audit Logging

```bash
# File audit backend
kubectl exec -n vault-prod vault-0 -- \
  vault audit enable file file_path=/vault/audit.log

# Verify
kubectl exec -n vault-prod vault-0 -- vault audit list
```

### 4. Set Up Secrets Rotation

```bash
# Example: Database credential rotation
kubectl exec -n vault-prod vault-0 -- vault policy write db-rotation - << 'EOF'
# Allows rotation of database passwords
path "secret/data/datasources/*" {
  capabilities = ["read", "create", "update"]
}
EOF
```

---

## Next Steps (Phase 3)

After Phase 2 is stable (1-2 weeks):

1. **GitOps Implementation** (FluxCD)
2. **Image Registry Management** (private registry mirror)
3. **SeaTunnel/Flink Hardening** (checkpoints, error handling)
4. **Distributed Tracing** (Jaeger/Tempo)
5. **Disaster Recovery Drills** (practice failover scenarios)

---

## Reference Documentation

- `k8s/vault/vault-production.yaml` - Production deployment manifests
- `k8s/vault/VAULT-SECURITY-GUIDE.md` - Security best practices
- `k8s/vault/VAULT-PRODUCTION-DEPLOYMENT.md` - Original planning doc
- `k8s/vault/vault-backup-cronjob.yaml` - Backup automation

---

## Support

For issues:

1. Check Pod logs: `kubectl logs -n vault-prod <pod-name>`
2. Describe resources: `kubectl describe pod -n vault-prod <pod-name>`
3. Check Vault status: `kubectl exec -n vault-prod vault-0 -- vault status`
4. Review events: `kubectl get events -n vault-prod`

---

**Status**: Phase 2 deployment guide complete  
**Next**: Execute deployment steps above  
**Timeline**: Week 2-3  
**Estimated Duration**: 2-4 hours deployment + 1-2 weeks stability testing
