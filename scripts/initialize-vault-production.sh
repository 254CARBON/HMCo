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
    
    ROOT_TOKEN=$(grep "Initial Root Token" "$KEYS_FILE" | awk '{print $NF}')
    
    if [[ -z "$ROOT_TOKEN" ]]; then
        echo -e "${YELLOW}Enter Vault root token:${NC}"
        read -s ROOT_TOKEN
    fi
    
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
        vault_init && vault_unseal && vault_config_k8s_auth && vault_config_engines
        ;;
    unseal)
        vault_unseal
        ;;
    config)
        vault_config_k8s_auth && vault_config_engines
        ;;
    test)
        vault_test
        ;;
    status)
        vault_status
        ;;
    *)
        echo "Usage: $0 [init|unseal|config|test|status]"
        echo ""
        echo "Actions:"
        echo "  init   - Initialize and unseal Vault (first time only)"
        echo "  unseal - Unseal existing Vault"
        echo "  config - Configure auth and secret engines"
        echo "  test   - Test Vault connectivity"
        echo "  status - Check Vault status"
        exit 1
        ;;
esac

echo -e "${GREEN}✓ Operation complete${NC}"
echo ""
