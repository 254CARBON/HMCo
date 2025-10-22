# Vault Initialization Guide

**Status**: Vault is deployed but requires manual initialization  
**Current State**: Pods running but restarting (expected behavior)

---

## Why Vault is "Crash Looping"

Vault is **NOT actually broken**. The CrashLoopBackOff/restart behavior is **EXPECTED** for an uninitialized Vault with PostgreSQL backend.

**What's happening**:
1. Vault starts and connects to PostgreSQL successfully ✓
2. Vault looks for the `vault_kv_store` table (doesn't exist yet)
3. Vault waits in a retry loop logging warnings
4. Kubernetes restarts it thinking it's unhealthy

**This is normal** - Vault needs manual initialization to create the database schema.

---

## Vault Initialization Process

Vault requires a **one-time manual initialization** that:
- Creates encryption keys
- Sets up the PostgreSQL schema (creates vault_kv_store and other tables)
- Generates unseal keys (5 keys, threshold 3)
- Generates root token

**Important**: This is a critical security procedure that requires careful handling of unseal keys.

---

## How to Initialize Vault

### Option 1: Via kubectl exec (When Pod is Stable)

The challenge: Vault restarts before we can initialize it. We need to either:
1. Increase readiness/liveness probe delays
2. Remove health probes temporarily
3. Use a StatefulSet instead of Deployment

### Option 2: Update Deployment for Initialization

Temporarily remove or extend health probes:

```bash
# Patch deployment to disable health checks during init
kubectl patch deployment vault -n vault-prod --type='json' -p='[
  {"op": "remove", "path": "/spec/template/spec/containers/0/livenessProbe"},
  {"op": "remove", "path": "/spec/template/spec/containers/0/readinessProbe"}
]'

# Wait for pod to stabilize
sleep 30

# Get pod name
VAULT_POD=$(kubectl get pods -n vault-prod -l app=vault -o jsonpath='{.items[0].metadata.name}')

# Initialize Vault
kubectl exec -n vault-prod $VAULT_POD -- vault operator init \
  -key-shares=5 \
  -key-threshold=3 \
  -format=json | tee ~/vault-init-keys.json

# CRITICAL: Save the unseal keys and root token securely!
# You'll need 3 of the 5 unseal keys to unseal Vault
```

### Option 3: Use StatefulSet (Better for Production)

Convert Vault from Deployment to StatefulSet for stable pod names and better HA.

---

## Current Vault Configuration

- **Namespace**: vault-prod
- **Replicas**: 3 (HA setup)
- **Storage Backend**: PostgreSQL
  - Database: vault
  - User: vault  
  - Password: vault-secure-password-change-me
  - Host: postgres-shared-service.data-platform.svc.cluster.local:5432
- **TLS**: Self-signed certificates (mounted)
- **API Port**: 8200
- **Cluster Port**: 8201

---

## Why Initialization is Manual

Vault's design requires manual initialization because:
1. **Security**: Unseal keys should be distributed to different people
2. **No Automatic Unsealing**: Prevents unauthorized access if cluster restarts
3. **Root Token**: Must be securely stored by operators

---

## Alternative: Use File-based Storage for Easy Init

If PostgreSQL backend is causing issues, you can temporarily use file-based storage:

```yaml
storage "file" {
  path = "/vault/data"
}
```

This allows easier initialization, then you can migrate to PostgreSQL later.

---

## Current Workaround

For now, Vault is deployed but not functional. Services that need secrets can:
1. Use Kubernetes Secrets (current approach)
2. Wait for Vault initialization
3. Use external secret management

---

## Recommendations

**Quick Fix**: Remove Vault health probes, initialize, then restore health checks

**Better Fix**: Convert to StatefulSet with proper init procedures

**Alternative**: Use file-based storage for simpler initialization

**Skip for Now**: Platform works fine without Vault - all services use K8s secrets currently

---

**Next Steps**: Choose one of the initialization approaches above based on your security requirements and operational preferences.

---

_Last Updated: October 21, 2025_
