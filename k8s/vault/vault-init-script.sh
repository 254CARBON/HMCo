#!/bin/bash
# Vault Initialization and Secret Migration Script
# Run this after deploying Vault to initialize and migrate secrets

set -e

VAULT_NAMESPACE="vault-prod"
DATA_NAMESPACE="data-platform"

echo "==================================================================="
echo "         Vault Initialization & Secret Migration"
echo "==================================================================="
echo ""

# Note: Vault deployment needs to be created first
# The services already exist, we need to create the StatefulSet or Deployment

echo "Step 1: Checking Vault status..."
VAULT_RUNNING=$(kubectl get pods -n $VAULT_NAMESPACE -l app=vault --no-headers 2>/dev/null | wc -l)

if [ "$VAULT_RUNNING" -eq "0" ]; then
    echo "⚠️  Vault is not running. Deploy Vault first:"
    echo "   kubectl apply -f k8s/vault/vault-statefulset.yaml"
    echo ""
    echo "Or use Helm:"
    echo "   helm install vault hashicorp/vault --namespace vault-prod"
    exit 1
fi

echo "✓ Vault pod is running"
echo ""

# Port forward to Vault
echo "Step 2: Setting up port-forward to Vault..."
kubectl port-forward -n $VAULT_NAMESPACE svc/vault 8200:8200 >/dev/null 2>&1 &
PORT_FORWARD_PID=$!
sleep 3

export VAULT_ADDR='http://localhost:8200'

# Initialize Vault (if not already)
echo "Step 3: Initializing Vault (if needed)..."
if kubectl exec -n $VAULT_NAMESPACE -it deployment/vault -- vault status 2>&1 | grep -q "not initialized"; then
    echo "Initializing Vault..."
    INIT_OUTPUT=$(kubectl exec -n $VAULT_NAMESPACE -it deployment/vault -- vault operator init -key-shares=5 -key-threshold=3)
    echo "$INIT_OUTPUT" > vault-init-keys.txt
    echo "✓ Vault initialized. Keys saved to vault-init-keys.txt"
    echo "⚠️  IMPORTANT: Store these keys securely!"
    echo ""
else
    echo "✓ Vault already initialized"
fi

# Unseal Vault
echo "Step 4: Unsealing Vault..."
# Note: In production, use auto-unseal or store keys securely
echo "Manual step required: Unseal Vault with keys from vault-init-keys.txt"
echo "kubectl exec -n $VAULT_NAMESPACE -it deployment/vault -- vault operator unseal <key1>"
echo "kubectl exec -n $VAULT_NAMESPACE -it deployment/vault -- vault operator unseal <key2>"
echo "kubectl exec -n $VAULT_NAMESPACE -it deployment/vault -- vault operator unseal <key3>"
echo ""

# Enable Kubernetes auth
echo "Step 5: Enabling Kubernetes authentication..."
kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault auth enable kubernetes 2>/dev/null || echo "✓ Kubernetes auth already enabled"

# Configure Kubernetes auth
echo "Step 6: Configuring Kubernetes auth..."
K8S_HOST="https://kubernetes.default.svc:443"
kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault write auth/kubernetes/config \
    kubernetes_host="$K8S_HOST"

# Create policies
echo "Step 7: Creating Vault policies..."

# Data platform policy
kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault policy write data-platform - <<EOF
path "secret/data/data-platform/*" {
  capabilities = ["read", "list"]
}

path "database/creds/data-platform" {
  capabilities = ["read"]
}
EOF

echo "✓ Policies created"
echo ""

# Create Kubernetes roles
echo "Step 8: Creating Kubernetes roles..."
kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault write auth/kubernetes/role/data-platform \
    bound_service_account_names=default \
    bound_service_account_namespaces=$DATA_NAMESPACE \
    policies=data-platform \
    ttl=24h

echo "✓ Roles created"
echo ""

# Enable KV secrets engine
echo "Step 9: Enabling KV secrets engine..."
kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault secrets enable -path=secret kv-v2 2>/dev/null || echo "✓ KV secrets already enabled"

# Migrate secrets
echo "Step 10: Migrating secrets to Vault..."

# SeaTunnel API keys
echo "Migrating SeaTunnel API keys..."
FRED_KEY=$(kubectl get secret seatunnel-api-keys -n $DATA_NAMESPACE -o jsonpath='{.data.FRED_API_KEY}' | base64 -d)
EIA_KEY=$(kubectl get secret seatunnel-api-keys -n $DATA_NAMESPACE -o jsonpath='{.data.EIA_API_KEY}' | base64 -d)
NOAA_KEY=$(kubectl get secret seatunnel-api-keys -n $DATA_NAMESPACE -o jsonpath='{.data.NOAA_API_KEY}' | base64 -d)

kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault kv put secret/data-platform/api-keys \
    FRED_API_KEY="$FRED_KEY" \
    EIA_API_KEY="$EIA_KEY" \
    NOAA_API_KEY="$NOAA_KEY"

echo "✓ API keys migrated"

# MinIO credentials
echo "Migrating MinIO credentials..."
MINIO_ACCESS=$(kubectl get secret minio-secret -n $DATA_NAMESPACE -o jsonpath='{.data.access-key}' | base64 -d)
MINIO_SECRET=$(kubectl get secret minio-secret -n $DATA_NAMESPACE -o jsonpath='{.data.secret-key}' | base64 -d)

kubectl exec -n $VAULT_NAMESPACE deployment/vault -- vault kv put secret/data-platform/minio \
    access-key="$MINIO_ACCESS" \
    secret-key="$MINIO_SECRET"

echo "✓ MinIO credentials migrated"
echo ""

# Cleanup
echo "Step 11: Cleaning up..."
kill $PORT_FORWARD_PID 2>/dev/null || true

echo "==================================================================="
echo "                   Vault Setup Complete! ✅"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "1. Deploy Vault Agent Injector: kubectl apply -f k8s/vault/vault-agent-injector.yaml"
echo "2. Update deployments with Vault annotations"
echo "3. Test secret injection"
echo ""
echo "Vault UI: https://vault.254carbon.com"
echo "==================================================================="


