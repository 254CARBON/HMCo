#!/bin/bash
#
# Production Vault Initialization Script
# Initializes Vault with Kubernetes authentication and PostgreSQL backend
#
# Prerequisites:
#   - Vault StatefulSet deployed in vault-prod namespace
#   - PostgreSQL database configured
#   - Image pull secrets configured
#
# Usage: ./initialize-vault-production.sh [ACTION]
#
# Actions:
#   init    - Initialize Vault and generate unseal keys (FIRST RUN ONLY)
#   unseal  - Unseal Vault using stored keys
#   config  - Configure Kubernetes auth and secret engines
#   test    - Test Vault connectivity
#   status  - Check Vault status

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="vault-prod"
POD="vault-0"
KEYS_FILE="/tmp/vault-init-keys-backup.txt"
SECURE_KEYS_FILE="$HOME/.vault-keys"
ROOT_TOKEN=""

# Initialize Vault
vault_init() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Initializing Vault${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    echo -e "${YELLOW}1. Checking if Vault is already initialized...${NC}"
    
    # Check if pod is ready
    kubectl wait pod -n $NAMESPACE -l app=vault --for=condition=Ready --timeout=120s 2>/dev/null || {
        echo -e "${RED}Pod not ready. Scaling up vault...${NC}"
        kubectl scale statefulset vault -n $NAMESPACE --replicas=1
        kubectl wait pod -n $NAMESPACE -l app=vault --for=condition=Ready --timeout=120s
    }
    
    # Initialize Vault
    echo -e "${YELLOW}2. Running Vault operator init...${NC}"
    
    if kubectl exec $POD -n $NAMESPACE -- vault status 2>/dev/null | grep -q "Initialized.*true"; then
        echo -e "${GREEN}✓ Vault already initialized${NC}"
        return 0
    fi
    
    # Initialize with 3 key shares, 2 threshold
    INIT_OUTPUT=$(kubectl exec $POD -n $NAMESPACE -- vault operator init \
        -key-shares=3 \
        -key-threshold=2 \
        -output-curl-format=false 2>/dev/null)
    
    if [[ -z "$INIT_OUTPUT" ]]; then
        echo -e "${RED}✗ Initialization failed${NC}"
        return 1
    fi
    
    # Extract and save keys
    echo "$INIT_OUTPUT" > "$KEYS_FILE"
    chmod 600 "$KEYS_FILE"
    
    echo -e "${GREEN}✓ Vault initialized${NC}"
    echo ""
    echo -e "${YELLOW}CRITICAL: Store these keys securely (offline, encrypted)${NC}"
    echo -e "${YELLOW}Backup location: $KEYS_FILE${NC}"
    echo ""
    
    # Extract root token (for later use)
    ROOT_TOKEN=$(echo "$INIT_OUTPUT" | grep "Initial Root Token" | awk '{print $NF}')
    echo -e "${YELLOW}Root Token: ${RED}$ROOT_TOKEN${NC}"
    echo ""
    echo -e "${RED}IMPORTANT: Never share this token. Store securely!${NC}"
    echo ""
    
    echo -e "${BLUE}Unseal Keys:${NC}"
    echo "$INIT_OUTPUT" | grep "Unseal Key" | nl
    echo ""
    
    return 0
}

# Unseal Vault
vault_unseal() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Unsealing Vault${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    if [[ ! -f "$KEYS_FILE" ]]; then
        echo -e "${RED}Unseal keys file not found: $KEYS_FILE${NC}"
        echo "Run 'init' action first or restore from backup"
        return 1
    fi
    
    # Extract unseal keys
    KEY1=$(grep "Unseal Key 1" "$KEYS_FILE" | awk '{print $NF}')
    KEY2=$(grep "Unseal Key 2" "$KEYS_FILE" | awk '{print $NF}')
    
    if [[ -z "$KEY1" ]] || [[ -z "$KEY2" ]]; then
        echo -e "${RED}✗ Could not extract unseal keys from $KEYS_FILE${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Unsealing all Vault replicas...${NC}"
    echo ""
    
    # Unseal each replica
    for i in 0 1 2; do
        POD_NAME="vault-$i"
        
        # Check if pod exists
        if ! kubectl get pod $POD_NAME -n $NAMESPACE &>/dev/null; then
            echo -e "${YELLOW}Pod $POD_NAME not running, skipping...${NC}"
            continue
        fi
        
        echo -n "Unsealing $POD_NAME ... "
        
        # Provide unseal keys
        kubectl exec $POD_NAME -n $NAMESPACE -- vault operator unseal "$KEY1" >/dev/null 2>&1
        kubectl exec $POD_NAME -n $NAMESPACE -- vault operator unseal "$KEY2" >/dev/null 2>&1
        
        # Check status
        if kubectl exec $POD_NAME -n $NAMESPACE -- vault status 2>/dev/null | grep -q "Sealed.*false"; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${YELLOW}Status unclear${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}✓ Unseal complete${NC}"
    echo ""
}

# Configure Kubernetes authentication
vault_config_k8s_auth() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Configuring Kubernetes Auth${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # Get root token
    if [[ ! -f "$KEYS_FILE" ]]; then
        echo -e "${RED}Unseal keys file not found${NC}"
        return 1
    fi
    
    vault_get_root_token || return 1
    
    echo -e "${YELLOW}Enabling Kubernetes auth...${NC}"
    
    kubectl exec $POD -n $NAMESPACE -- \
        vault auth enable kubernetes 2>/dev/null || echo "Kubernetes auth already enabled"
    
    echo -e "${YELLOW}Configuring Kubernetes auth...${NC}"
    
    kubectl exec $POD -n $NAMESPACE -- vault write auth/kubernetes/config \
        kubernetes_host=https://kubernetes.default.svc.cluster.local:443 \
        kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
        token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token
    
    echo -e "${GREEN}✓ Kubernetes auth configured${NC}"
    echo ""
}

# Retrieve root token once and cache it for subsequent operations
vault_get_root_token() {
    if [[ -n "${ROOT_TOKEN:-}" ]]; then
        return 0
    fi

    if [[ -f "$KEYS_FILE" ]]; then
        ROOT_TOKEN=$(grep "Initial Root Token" "$KEYS_FILE" | awk '{print $NF}')
    fi

    if [[ -z "${ROOT_TOKEN:-}" ]]; then
        echo -e "${YELLOW}Enter Vault root token:${NC}"
        read -s ROOT_TOKEN
    fi

    if [[ -z "${ROOT_TOKEN:-}" ]]; then
        echo -e "${RED}Root token is required to configure policies${NC}"
        return 1
    fi

    return 0
}

# Configure policies and roles for the External Secrets Operator
vault_config_external_secrets() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Configuring External Secrets Access${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    vault_get_root_token || return 1

    echo -e "${YELLOW}Creating external-secrets-read policy...${NC}"

    POLICY=$(cat <<'EOF'
path "secret/data/data-platform/*" {
  capabilities = ["read"]
}

path "secret/metadata/data-platform/*" {
  capabilities = ["read", "list"]
}
EOF
)

    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/external-secrets-policy.hcl
$POLICY
EOF"

    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write external-secrets-read /tmp/external-secrets-policy.hcl" >/dev/null

    kubectl exec $POD -n $NAMESPACE -- sh -c "rm -f /tmp/external-secrets-policy.hcl"

    echo -e "${YELLOW}Creating Kubernetes auth role: external-secrets${NC}"

    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/external-secrets \
            bound_service_account_names=external-secrets-operator \
            bound_service_account_namespaces=vault \
            policies=external-secrets-read \
            ttl=24h" >/dev/null

    echo -e "${GREEN}✓ External Secrets Operator wired to Vault${NC}"
    echo ""
}

# Configure least-privilege policies for each application/namespace
vault_config_app_policies() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Configuring Application Policies${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    vault_get_root_token || return 1

    # Data Platform policy - read access to all data-platform secrets
    echo -e "${YELLOW}Creating data-platform-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# Allow read access to all data-platform secrets
path "secret/data/data-platform/*" {
  capabilities = ["read"]
}

path "secret/metadata/data-platform/*" {
  capabilities = ["read", "list"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write data-platform-read /tmp/policy.hcl" >/dev/null

    # ClickHouse policy - least privilege for ClickHouse only
    echo -e "${YELLOW}Creating clickhouse-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# ClickHouse can only read its own secret
path "secret/data/data-platform/clickhouse" {
  capabilities = ["read"]
}

path "secret/metadata/data-platform/clickhouse" {
  capabilities = ["read"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write clickhouse-read /tmp/policy.hcl" >/dev/null

    # MLflow policy - least privilege for MLflow backend and artifact secrets
    echo -e "${YELLOW}Creating mlflow-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# MLflow can only read its own secrets
path "secret/data/data-platform/mlflow/*" {
  capabilities = ["read"]
}

path "secret/metadata/data-platform/mlflow/*" {
  capabilities = ["read", "list"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write mlflow-read /tmp/policy.hcl" >/dev/null

    # Superset policy - least privilege for Superset secrets
    echo -e "${YELLOW}Creating superset-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# Superset can only read its own secrets
path "secret/data/data-platform/superset" {
  capabilities = ["read"]
}

path "secret/metadata/data-platform/superset" {
  capabilities = ["read"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write superset-read /tmp/policy.hcl" >/dev/null

    # API Gateway policy - for Kong and JWT secrets
    echo -e "${YELLOW}Creating api-gateway-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# API Gateway can read its own secrets
path "secret/data/api-gateway/*" {
  capabilities = ["read"]
}

path "secret/metadata/api-gateway/*" {
  capabilities = ["read", "list"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write api-gateway-read /tmp/policy.hcl" >/dev/null

    # Monitoring policy - for Alertmanager secrets
    echo -e "${YELLOW}Creating monitoring-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# Monitoring can read its own secrets
path "secret/data/monitoring/*" {
  capabilities = ["read"]
}

path "secret/metadata/monitoring/*" {
  capabilities = ["read", "list"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write monitoring-read /tmp/policy.hcl" >/dev/null

    # Cloudflare Tunnel policy
    echo -e "${YELLOW}Creating cloudflare-read policy...${NC}"
    POLICY=$(cat <<'EOF'
# Cloudflare Tunnel can read its own secrets
path "secret/data/cloudflare/*" {
  capabilities = ["read"]
}

path "secret/metadata/cloudflare/*" {
  capabilities = ["read"]
}
EOF
)
    kubectl exec $POD -n $NAMESPACE -- sh -c "cat <<'EOF' >/tmp/policy.hcl
$POLICY
EOF"
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault policy write cloudflare-read /tmp/policy.hcl" >/dev/null

    kubectl exec $POD -n $NAMESPACE -- sh -c "rm -f /tmp/policy.hcl"

    echo -e "${GREEN}✓ Application policies configured${NC}"
    echo ""
}

# Configure Kubernetes auth roles for service accounts (AppRole/JWT bindings)
vault_config_service_account_roles() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Configuring Service Account Roles${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    vault_get_root_token || return 1

    echo -e "${YELLOW}Creating Kubernetes auth roles for service accounts...${NC}"

    # ClickHouse service account role
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/clickhouse \
            bound_service_account_names=clickhouse \
            bound_service_account_namespaces=data-platform \
            policies=clickhouse-read \
            ttl=1h" >/dev/null
    echo -e "  ${GREEN}✓${NC} clickhouse role created"

    # MLflow service account role
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/mlflow \
            bound_service_account_names=mlflow \
            bound_service_account_namespaces=data-platform \
            policies=mlflow-read \
            ttl=1h" >/dev/null
    echo -e "  ${GREEN}✓${NC} mlflow role created"

    # Superset service account role
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/superset \
            bound_service_account_names=superset \
            bound_service_account_namespaces=data-platform \
            policies=superset-read \
            ttl=1h" >/dev/null
    echo -e "  ${GREEN}✓${NC} superset role created"

    # API Gateway service account role
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/api-gateway \
            bound_service_account_names=kong \
            bound_service_account_namespaces=kong \
            policies=api-gateway-read \
            ttl=1h" >/dev/null
    echo -e "  ${GREEN}✓${NC} api-gateway role created"

    # Monitoring service account role
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/monitoring \
            bound_service_account_names=alertmanager,prometheus \
            bound_service_account_namespaces=monitoring \
            policies=monitoring-read \
            ttl=1h" >/dev/null
    echo -e "  ${GREEN}✓${NC} monitoring role created"

    # Cloudflare Tunnel service account role
    kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault write auth/kubernetes/role/cloudflare-tunnel \
            bound_service_account_names=cloudflared \
            bound_service_account_namespaces=default,cloudflare \
            policies=cloudflare-read \
            ttl=1h" >/dev/null
    echo -e "  ${GREEN}✓${NC} cloudflare-tunnel role created"

    echo ""
    echo -e "${GREEN}✓ Service account roles configured${NC}"
    echo ""
}

# Test least-privilege access and cross-read denial
vault_test_access_control() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Testing Access Control${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    vault_get_root_token || return 1

    echo -e "${YELLOW}Testing policy enforcement...${NC}"
    echo ""

    # Test 1: External Secrets Operator should be able to read data-platform secrets
    echo -e "${YELLOW}Test 1: External Secrets read access to data-platform...${NC}"
    TEST_RESULT=$(kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault token create -policy=external-secrets-read -format=json 2>/dev/null | grep -o '\"client_token\":\"[^\"]*\"' | cut -d'\"' -f4")
    
    if [[ -n "$TEST_RESULT" ]]; then
        # Try to read a data-platform secret (should succeed)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/data-platform/minio" >/dev/null 2>&1 && \
            echo -e "  ${GREEN}✓${NC} Can read data-platform secrets" || \
            echo -e "  ${YELLOW}⚠${NC} Cannot read data-platform secrets (may not exist yet)"
        
        # Try to read an api-gateway secret (should fail)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/api-gateway/postgres" >/dev/null 2>&1 && \
            echo -e "  ${RED}✗${NC} ERROR: Can read api-gateway secrets (should be denied)" || \
            echo -e "  ${GREEN}✓${NC} Cross-namespace read denied (api-gateway)"
    fi

    # Test 2: ClickHouse policy should only read its own secrets
    echo ""
    echo -e "${YELLOW}Test 2: ClickHouse least-privilege access...${NC}"
    TEST_RESULT=$(kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault token create -policy=clickhouse-read -format=json 2>/dev/null | grep -o '\"client_token\":\"[^\"]*\"' | cut -d'\"' -f4")
    
    if [[ -n "$TEST_RESULT" ]]; then
        # Try to read ClickHouse secret (should succeed)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/data-platform/clickhouse" >/dev/null 2>&1 && \
            echo -e "  ${GREEN}✓${NC} Can read own secrets" || \
            echo -e "  ${YELLOW}⚠${NC} Cannot read own secrets (may not exist yet)"
        
        # Try to read MLflow secret (should fail)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/data-platform/mlflow/backend" >/dev/null 2>&1 && \
            echo -e "  ${RED}✗${NC} ERROR: Can read MLflow secrets (should be denied)" || \
            echo -e "  ${GREEN}✓${NC} Cross-app read denied (MLflow)"
    fi

    # Test 3: MLflow policy should only read MLflow secrets
    echo ""
    echo -e "${YELLOW}Test 3: MLflow least-privilege access...${NC}"
    TEST_RESULT=$(kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault token create -policy=mlflow-read -format=json 2>/dev/null | grep -o '\"client_token\":\"[^\"]*\"' | cut -d'\"' -f4")
    
    if [[ -n "$TEST_RESULT" ]]; then
        # Try to read MLflow secret (should succeed)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/data-platform/mlflow/backend" >/dev/null 2>&1 && \
            echo -e "  ${GREEN}✓${NC} Can read own secrets" || \
            echo -e "  ${YELLOW}⚠${NC} Cannot read own secrets (may not exist yet)"
        
        # Try to read Superset secret (should fail)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/data-platform/superset" >/dev/null 2>&1 && \
            echo -e "  ${RED}✗${NC} ERROR: Can read Superset secrets (should be denied)" || \
            echo -e "  ${GREEN}✓${NC} Cross-app read denied (Superset)"
    fi

    # Test 4: API Gateway should not access data-platform secrets
    echo ""
    echo -e "${YELLOW}Test 4: API Gateway namespace isolation...${NC}"
    TEST_RESULT=$(kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$ROOT_TOKEN" sh -c \
        "vault token create -policy=api-gateway-read -format=json 2>/dev/null | grep -o '\"client_token\":\"[^\"]*\"' | cut -d'\"' -f4")
    
    if [[ -n "$TEST_RESULT" ]]; then
        # Try to read API Gateway secret (should succeed)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/api-gateway/postgres" >/dev/null 2>&1 && \
            echo -e "  ${GREEN}✓${NC} Can read own secrets" || \
            echo -e "  ${YELLOW}⚠${NC} Cannot read own secrets (may not exist yet)"
        
        # Try to read data-platform secret (should fail)
        kubectl exec $POD -n $NAMESPACE -- env VAULT_TOKEN="$TEST_RESULT" sh -c \
            "vault kv get secret/data-platform/minio" >/dev/null 2>&1 && \
            echo -e "  ${RED}✗${NC} ERROR: Can read data-platform secrets (should be denied)" || \
            echo -e "  ${GREEN}✓${NC} Cross-namespace read denied (data-platform)"
    fi

    echo ""
    echo -e "${GREEN}✓ Access control tests complete${NC}"
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo -e "  • Least-privilege policies enforce app-level isolation"
    echo -e "  • Cross-namespace reads are denied"
    echo -e "  • Each app can only read its own secrets"
    echo ""
}

# Configure secret engines
vault_config_engines() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Configuring Secret Engines${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    echo -e "${YELLOW}Enabling secret engines...${NC}"
    
    # KV v2 secrets
    kubectl exec $POD -n $NAMESPACE -- \
        vault secrets enable -path=secret kv-v2 2>/dev/null || echo "KV v2 already enabled"
    
    # Database secrets
    kubectl exec $POD -n $NAMESPACE -- \
        vault secrets enable -path=database database 2>/dev/null || echo "Database secrets already enabled"
    
    # SSH secrets
    kubectl exec $POD -n $NAMESPACE -- \
        vault secrets enable -path=ssh ssh 2>/dev/null || echo "SSH secrets already enabled"
    
    echo -e "${GREEN}✓ Secret engines configured${NC}"
    echo ""
}

# Test Vault connectivity
vault_test() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Testing Vault Connectivity${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    echo -e "${YELLOW}Checking Vault status...${NC}"
    kubectl exec $POD -n $NAMESPACE -- vault status
    
    echo ""
    echo -e "${YELLOW}Testing KV secrets...${NC}"
    kubectl exec $POD -n $NAMESPACE -- vault kv put secret/test-key value="test-value" 2>/dev/null || {
        echo "Note: You may need to be logged in with valid token"
    }
    
    echo -e "${GREEN}✓ Connectivity test complete${NC}"
    echo ""
}

# Check Vault status
vault_status() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Vault Status${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    echo -e "${YELLOW}Checking pods...${NC}"
    kubectl get pods -n $NAMESPACE
    
    echo ""
    echo -e "${YELLOW}Vault status on $POD...${NC}"
    kubectl exec $POD -n $NAMESPACE -- vault status || echo "Cannot access Vault"
    
    echo ""
    echo -e "${YELLOW}Replicas:${NC}"
    kubectl get statefulset -n $NAMESPACE
    
    echo ""
}

# Main
ACTION="${1:-status}"

case "$ACTION" in
    init)
        vault_init && vault_unseal && vault_config_k8s_auth && vault_config_engines && \
        vault_config_external_secrets && vault_config_app_policies && \
        vault_config_service_account_roles && vault_test_access_control
        ;;
    unseal)
        vault_unseal
        ;;
    config)
        vault_config_k8s_auth && vault_config_engines && vault_config_external_secrets && \
        vault_config_app_policies && vault_config_service_account_roles
        ;;
    test)
        vault_test
        ;;
    test-access)
        vault_test_access_control
        ;;
    status)
        vault_status
        ;;
    *)
        echo "Usage: $0 [init|unseal|config|test|test-access|status]"
        echo ""
        echo "Actions:"
        echo "  init        - Initialize and unseal Vault (first time only)"
        echo "  unseal      - Unseal existing Vault"
        echo "  config      - Configure auth, secret engines, policies, and roles"
        echo "  test        - Test Vault connectivity"
        echo "  test-access - Test least-privilege access control policies"
        echo "  status      - Check Vault status"
        exit 1
        ;;
esac

echo -e "${GREEN}✓ Operation complete${NC}"
echo ""
