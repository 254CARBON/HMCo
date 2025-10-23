#!/bin/bash

# Create/Update Cloudflare DNS Records for Tunnel
# This script ensures all DNS records point to your Cloudflare Tunnel

set -e

ACCOUNT_ID="0c93c74d5269a228e91d4bf91c547f56"
DNS_API_TOKEN="acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm"
TUNNEL_ID="291bc289-e3c3-4446-a9ad-8e327660ecd5"
ZONE_NAME="254carbon.com"

echo "===== Cloudflare DNS Record Setup ====="
echo ""

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "❌ ERROR: jq is required but not installed"
    echo "Install: sudo apt-get install jq"
    exit 1
fi

# Get Zone ID
echo "Getting Zone ID for ${ZONE_NAME}..."
ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=${ZONE_NAME}" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json" | jq -r '.result[0].id')

if [ -z "$ZONE_ID" ] || [ "$ZONE_ID" = "null" ]; then
  echo "❌ ERROR: Could not find zone for ${ZONE_NAME}"
  exit 1
fi

echo "✓ Zone ID: ${ZONE_ID}"
echo ""

# Tunnel CNAME target
TUNNEL_CNAME="${TUNNEL_ID}.cfargotunnel.com"

# Define all required DNS records
declare -A DOMAINS
DOMAINS=(
  ["portal"]="portal.254carbon.com"
  ["www"]="www.254carbon.com"
  ["datahub"]="datahub.254carbon.com"
  ["dolphinscheduler"]="dolphinscheduler.254carbon.com"
  ["superset"]="superset.254carbon.com"
  ["trino"]="trino.254carbon.com"
  ["grafana"]="grafana.254carbon.com"
  ["harbor"]="harbor.254carbon.com"
  ["minio"]="minio.254carbon.com"
)

echo "Creating/Updating DNS records to point to tunnel..."
echo "Target: ${TUNNEL_CNAME}"
echo ""

for key in "${!DOMAINS[@]}"; do
  DOMAIN="${DOMAINS[$key]}"
  echo "Processing: ${DOMAIN}"
  
  # Check if record exists
  EXISTING=$(curl -s -X GET \
    "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?name=${DOMAIN}" \
    -H "Authorization: Bearer ${DNS_API_TOKEN}" \
    -H "Content-Type: application/json")
  
  RECORD_ID=$(echo "$EXISTING" | jq -r '.result[0].id // empty')
  
  if [ -n "$RECORD_ID" ] && [ "$RECORD_ID" != "null" ]; then
    # Update existing record
    echo "  Updating existing record (ID: ${RECORD_ID})..."
    RESPONSE=$(curl -s -X PUT \
      "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records/${RECORD_ID}" \
      -H "Authorization: Bearer ${DNS_API_TOKEN}" \
      -H "Content-Type: application/json" \
      --data "{
        \"type\": \"CNAME\",
        \"name\": \"${DOMAIN}\",
        \"content\": \"${TUNNEL_CNAME}\",
        \"ttl\": 1,
        \"proxied\": true
      }")
  else
    # Create new record
    echo "  Creating new record..."
    RESPONSE=$(curl -s -X POST \
      "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
      -H "Authorization: Bearer ${DNS_API_TOKEN}" \
      -H "Content-Type: application/json" \
      --data "{
        \"type\": \"CNAME\",
        \"name\": \"${DOMAIN}\",
        \"content\": \"${TUNNEL_CNAME}\",
        \"ttl\": 1,
        \"proxied\": true
      }")
  fi
  
  SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
  if [ "$SUCCESS" = "true" ]; then
    echo "  ✓ Success"
  else
    ERROR=$(echo "$RESPONSE" | jq -r '.errors[0].message // "Unknown error"')
    echo "  ❌ Failed: ${ERROR}"
  fi
  echo ""
done

echo "===== DNS Records Updated ====="
echo ""
echo "DNS propagation may take up to 5 minutes, but usually takes < 30 seconds"
echo ""
echo "Test access:"
echo "  https://portal.254carbon.com"
echo "  https://grafana.254carbon.com"
echo "  https://superset.254carbon.com"
echo ""
echo "If still not working after 5 minutes:"
echo "1. Clear browser cache or use Incognito mode"
echo "2. Verify Cloudflare SSL mode is set to 'Flexible'"
echo "3. Check tunnel is connected: kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel"
echo ""





