#!/bin/bash
# Verify SSL/TLS Certificates for all 254carbon.com services

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SSL/TLS Certificate Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# List of services to check
SERVICES=(
    "portal"
    "grafana"
    "superset"
    "datahub"
    "trino"
    "doris"
    "vault"
    "minio"
    "dolphin"
    "lakefs"
    "mlflow"
    "spark-history"
    "harbor"
)

DOMAIN="254carbon.com"

echo -e "${YELLOW}Checking SSL/TLS certificates for all services...${NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

for service in "${SERVICES[@]}"; do
    URL="https://$service.$DOMAIN"
    
    printf "%-30s" "$service.$DOMAIN"
    
    # Check HTTP status
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL" --max-time 10 2>/dev/null || echo "000")
    
    # Check SSL certificate
    if [ "$HTTP_CODE" != "000" ]; then
        # Get certificate details
        CERT_INFO=$(echo | timeout 5 openssl s_client -connect "$service.$DOMAIN:443" -servername "$service.$DOMAIN" 2>/dev/null | openssl x509 -noout -dates -issuer 2>/dev/null || echo "")
        
        if [ -n "$CERT_INFO" ]; then
            # Extract issuer
            ISSUER=$(echo "$CERT_INFO" | grep "issuer" | sed 's/issuer=//')
            
            # Check if Cloudflare certificate
            if echo "$ISSUER" | grep -qi "cloudflare"; then
                if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ] || [ "$HTTP_CODE" = "401" ]; then
                    echo -e "${GREEN}✓ OK${NC} (HTTP $HTTP_CODE, Cloudflare)"
                    ((PASSED++))
                else
                    echo -e "${YELLOW}⊙ Warning${NC} (HTTP $HTTP_CODE, Cloudflare)"
                    ((WARNINGS++))
                fi
            else
                echo -e "${YELLOW}⊙ Warning${NC} (HTTP $HTTP_CODE, Not Cloudflare)"
                ((WARNINGS++))
            fi
        else
            echo -e "${RED}✗ SSL Error${NC} (HTTP $HTTP_CODE)"
            ((FAILED++))
        fi
    else
        echo -e "${RED}✗ Unreachable${NC}"
        ((FAILED++))
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "Summary:"
echo -e "${GREEN}  Passed:   $PASSED${NC}"
echo -e "${YELLOW}  Warnings: $WARNINGS${NC}"
echo -e "${RED}  Failed:   $FAILED${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detailed certificate check for a sample service
echo -e "${YELLOW}Detailed certificate check for portal.254carbon.com:${NC}"
echo ""
echo | openssl s_client -connect portal.254carbon.com:443 -servername portal.254carbon.com 2>/dev/null | openssl x509 -noout -text | grep -A3 "Validity\|Issuer\|Subject:"

echo ""

# Check Kubernetes secrets
echo -e "${YELLOW}Checking Kubernetes TLS secrets:${NC}"
echo ""

NAMESPACES=("data-platform" "monitoring" "registry")

for namespace in "${NAMESPACES[@]}"; do
    printf "%-25s" "$namespace"
    if kubectl get secret cloudflare-origin-cert -n "$namespace" &>/dev/null 2>&1; then
        # Get certificate expiry from secret
        EXPIRY=$(kubectl get secret cloudflare-origin-cert -n "$namespace" -o jsonpath='{.data.tls\.crt}' 2>/dev/null | base64 -d | openssl x509 -noout -enddate 2>/dev/null | sed 's/notAfter=//' || echo "Unknown")
        echo -e "${GREEN}✓ Secret exists${NC} (Expires: $EXPIRY)"
    else
        echo -e "${RED}✗ Secret not found${NC}"
    fi
done

echo ""

# SSL Labs grade check (optional - commented out as it takes time)
# echo -e "${YELLOW}Want to check SSL Labs grade? This takes 2-3 minutes.${NC}"
# read -p "Run SSL Labs test? (y/N): " SSL_LABS
# if [ "$SSL_LABS" = "y" ] || [ "$SSL_LABS" = "Y" ]; then
#     echo "Testing portal.254carbon.com on SSL Labs..."
#     echo "Visit: https://www.ssllabs.com/ssltest/analyze.html?d=portal.254carbon.com"
# fi

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All certificates are working correctly!${NC}"
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}All services reachable but some certificates need attention.${NC}"
    exit 0
else
    echo -e "${RED}Some services have SSL/TLS issues that need attention.${NC}"
    exit 1
fi


