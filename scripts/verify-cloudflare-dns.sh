#!/bin/bash

# Verify Cloudflare DNS Configuration
# This script checks if DNS records are properly configured for the tunnel

set -e

ACCOUNT_ID="0c93c74d5269a228e91d4bf91c547f56"
DNS_API_TOKEN="acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm"
TUNNEL_ID="291bc289-e3c3-4446-a9ad-8e327660ecd5"
ZONE_NAME="254carbon.com"

echo "===== Cloudflare DNS Verification ====="
echo ""

# Get Zone ID
echo "Getting Zone ID for ${ZONE_NAME}..."
ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=${ZONE_NAME}" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json" | jq -r '.result[0].id')

if [ -z "$ZONE_ID" ] || [ "$ZONE_ID" = "null" ]; then
  echo "❌ ERROR: Could not find zone for ${ZONE_NAME}"
  echo "Please check your DNS_API_TOKEN has correct permissions"
  exit 1
fi

echo "✓ Zone ID: ${ZONE_ID}"
echo ""

# List all DNS records
echo "Checking DNS records..."
RECORDS=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json")

echo "$RECORDS" | jq -r '.result[] | select(.type == "CNAME") | "\(.name) -> \(.content) (\(.type))"'
echo ""

# Check specific records we need
DOMAINS=(
  "portal.254carbon.com"
  "datahub.254carbon.com"
  "dolphinscheduler.254carbon.com"
  "superset.254carbon.com"
  "trino.254carbon.com"
  "grafana.254carbon.com"
  "harbor.254carbon.com"
)

echo "Verifying required DNS records..."
TUNNEL_CNAME="${TUNNEL_ID}.cfargotunnel.com"

for domain in "${DOMAINS[@]}"; do
  RECORD=$(echo "$RECORDS" | jq -r ".result[] | select(.name == \"${domain}\") | .content")
  
  if [ -z "$RECORD" ] || [ "$RECORD" = "null" ]; then
    echo "❌ MISSING: ${domain}"
  elif [ "$RECORD" = "$TUNNEL_CNAME" ]; then
    echo "✓ OK: ${domain} -> ${RECORD}"
  else
    echo "⚠️  WRONG: ${domain} -> ${RECORD} (should be ${TUNNEL_CNAME})"
  fi
done

echo ""
echo "===== Next Steps ====="
echo ""
echo "If any records are MISSING or WRONG, run this to fix them:"
echo "./scripts/create-cloudflare-dns-records.sh"
echo ""





