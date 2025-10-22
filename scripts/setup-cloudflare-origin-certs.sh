#!/bin/bash
# Setup Cloudflare Origin Certificates for Kubernetes
# This script automates the process of creating TLS secrets and updating ingresses

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CERT_NAME="cloudflare-origin-cert"
NAMESPACES=("data-platform" "monitoring" "registry")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cloudflare Origin Certificate Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running from correct directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Step 1: Get certificate and key file paths
echo -e "${YELLOW}Step 1: Certificate and Key Files${NC}"
echo "Please provide the paths to your Cloudflare Origin Certificate files"
echo ""

# Prompt for certificate file
read -p "Enter path to certificate file (e.g., ~/cloudflare-certs/254carbon-origin.pem): " CERT_FILE
CERT_FILE="${CERT_FILE/#\~/$HOME}"

if [ ! -f "$CERT_FILE" ]; then
    echo -e "${RED}Error: Certificate file not found: $CERT_FILE${NC}"
    exit 1
fi

# Prompt for key file
read -p "Enter path to private key file (e.g., ~/cloudflare-certs/254carbon-origin-key.pem): " KEY_FILE
KEY_FILE="${KEY_FILE/#\~/$HOME}"

if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: Private key file not found: $KEY_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Certificate and key files found${NC}"
echo ""

# Step 2: Verify certificate
echo -e "${YELLOW}Step 2: Verifying Certificate${NC}"

# Extract certificate information
CERT_SUBJECT=$(openssl x509 -in "$CERT_FILE" -noout -subject 2>/dev/null || echo "")
CERT_ISSUER=$(openssl x509 -in "$CERT_FILE" -noout -issuer 2>/dev/null || echo "")
CERT_DATES=$(openssl x509 -in "$CERT_FILE" -noout -dates 2>/dev/null || echo "")

if [ -z "$CERT_SUBJECT" ]; then
    echo -e "${RED}Error: Invalid certificate file${NC}"
    exit 1
fi

echo "Certificate Details:"
echo "$CERT_SUBJECT"
echo "$CERT_ISSUER"
echo "$CERT_DATES"
echo ""

# Verify it's a Cloudflare certificate
if ! echo "$CERT_ISSUER" | grep -qi "cloudflare"; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Cloudflare Origin Certificate${NC}"
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo -e "${GREEN}✓ Certificate verified${NC}"
echo ""

# Step 3: Create Kubernetes secrets
echo -e "${YELLOW}Step 3: Creating Kubernetes Secrets${NC}"

for namespace in "${NAMESPACES[@]}"; do
    echo "Creating secret in namespace: $namespace"
    
    # Check if namespace exists
    if ! kubectl get namespace "$namespace" &> /dev/null; then
        echo -e "${YELLOW}  Warning: Namespace $namespace does not exist, skipping...${NC}"
        continue
    fi
    
    # Create or update the secret
    kubectl create secret tls "$CERT_NAME" \
        --cert="$CERT_FILE" \
        --key="$KEY_FILE" \
        -n "$namespace" \
        --dry-run=client -o yaml | kubectl apply -f - \
        && echo -e "${GREEN}  ✓ Secret created/updated in $namespace${NC}" \
        || echo -e "${RED}  ✗ Failed to create secret in $namespace${NC}"
done

echo ""

# Step 4: List all ingresses that need updating
echo -e "${YELLOW}Step 4: Finding Ingresses to Update${NC}"

INGRESSES_TO_UPDATE=()

for namespace in "${NAMESPACES[@]}"; do
    if ! kubectl get namespace "$namespace" &> /dev/null; then
        continue
    fi
    
    # Get all ingresses in namespace
    INGRESS_LIST=$(kubectl get ingress -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "$INGRESS_LIST" ]; then
        for ingress in $INGRESS_LIST; do
            INGRESSES_TO_UPDATE+=("$namespace:$ingress")
            echo "  Found: $ingress (namespace: $namespace)"
        done
    fi
done

if [ ${#INGRESSES_TO_UPDATE[@]} -eq 0 ]; then
    echo -e "${YELLOW}No ingresses found to update${NC}"
else
    echo -e "${GREEN}Found ${#INGRESSES_TO_UPDATE[@]} ingresses to update${NC}"
fi

echo ""

# Step 5: Update ingresses
if [ ${#INGRESSES_TO_UPDATE[@]} -gt 0 ]; then
    echo -e "${YELLOW}Step 5: Updating Ingress Resources${NC}"
    read -p "Update all ingresses to use the new certificate? (y/N): " UPDATE_INGRESSES
    
    if [ "$UPDATE_INGRESSES" = "y" ] || [ "$UPDATE_INGRESSES" = "Y" ]; then
        for item in "${INGRESSES_TO_UPDATE[@]}"; do
            namespace="${item%%:*}"
            ingress="${item##*:}"
            
            echo "Updating $ingress in $namespace..."
            
            # Get current ingress spec
            CURRENT_TLS=$(kubectl get ingress "$ingress" -n "$namespace" -o jsonpath='{.spec.tls}' 2>/dev/null || echo "[]")
            
            if [ "$CURRENT_TLS" != "[]" ] && [ -n "$CURRENT_TLS" ]; then
                # Update existing TLS configuration
                kubectl patch ingress "$ingress" -n "$namespace" --type='json' \
                    -p="[{\"op\": \"replace\", \"path\": \"/spec/tls/0/secretName\", \"value\": \"$CERT_NAME\"}]" \
                    && echo -e "${GREEN}  ✓ Updated $ingress${NC}" \
                    || echo -e "${RED}  ✗ Failed to update $ingress${NC}"
                    
                # Remove cert-manager annotations if present
                kubectl annotate ingress "$ingress" -n "$namespace" \
                    cert-manager.io/cluster-issuer- \
                    cert-manager.io/issuer- \
                    --overwrite &>/dev/null || true
            else
                echo -e "${YELLOW}  ⊙ No TLS configuration found, skipping $ingress${NC}"
            fi
        done
        
        echo -e "${GREEN}✓ Ingress updates complete${NC}"
    else
        echo "Skipped ingress updates"
    fi
fi

echo ""

# Step 6: Verification
echo -e "${YELLOW}Step 6: Verification${NC}"

echo "Verifying secrets..."
for namespace in "${NAMESPACES[@]}"; do
    if kubectl get secret "$CERT_NAME" -n "$namespace" &>/dev/null; then
        echo -e "${GREEN}  ✓ Secret exists in $namespace${NC}"
    else
        echo -e "${YELLOW}  ⊙ Secret not found in $namespace${NC}"
    fi
done

echo ""
echo "Testing SSL/TLS endpoints..."

# Test a few key services
SERVICES=("portal" "grafana" "harbor")

for service in "${SERVICES[@]}"; do
    URL="https://$service.254carbon.com"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL" --max-time 10 || echo "000")
    
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ] || [ "$HTTP_CODE" = "401" ]; then
        echo -e "${GREEN}  ✓ $service.254carbon.com: $HTTP_CODE${NC}"
    elif [ "$HTTP_CODE" = "000" ]; then
        echo -e "${YELLOW}  ⊙ $service.254carbon.com: Connection timeout or unreachable${NC}"
    else
        echo -e "${RED}  ✗ $service.254carbon.com: $HTTP_CODE${NC}"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Verify all services are accessible via HTTPS"
echo "2. Check certificate details in browser (click padlock icon)"
echo "3. Test with: ./scripts/verify-ssl-certificates.sh"
echo "4. Update Cloudflare SSL/TLS mode to 'Full (strict)'"
echo ""
echo "Documentation: docs/cloudflare/origin-certificates-setup.md"


