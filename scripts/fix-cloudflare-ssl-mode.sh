#!/bin/bash

# Fix Cloudflare SSL Mode for Tunnel Setup
# This script configures Cloudflare SSL/TLS settings to work with Kubernetes Ingress

set -e

ZONE_NAME="254carbon.com"
DNS_API_TOKEN="acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm"

echo "===== Cloudflare SSL/TLS Configuration Fix ====="
echo ""

# Get Zone ID
echo "Getting Zone ID for ${ZONE_NAME}..."
ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=${ZONE_NAME}" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json" | jq -r '.result[0].id')

if [ -z "$ZONE_ID" ] || [ "$ZONE_ID" = "null" ]; then
  echo "❌ ERROR: Could not find zone"
  exit 1
fi

echo "✓ Zone ID: ${ZONE_ID}"
echo ""

# Get current SSL settings
echo "Current SSL/TLS Settings:"
CURRENT_SSL=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/settings/ssl" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json")

CURRENT_MODE=$(echo "$CURRENT_SSL" | jq -r '.result.value')
echo "  SSL Mode: ${CURRENT_MODE}"

CURRENT_HTTPS=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/settings/always_use_https" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json" | jq -r '.result.value')
echo "  Always Use HTTPS: ${CURRENT_HTTPS}"
echo ""

# Set SSL mode to Flexible
echo "Setting SSL/TLS mode to 'flexible'..."
RESULT=$(curl -s -X PATCH "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/settings/ssl" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{"value":"flexible"}')

SUCCESS=$(echo "$RESULT" | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
  echo "✓ SSL mode set to 'flexible'"
else
  ERROR=$(echo "$RESULT" | jq -r '.errors[0].message // "Unknown error"')
  echo "❌ Failed: ${ERROR}"
  exit 1
fi

# Disable automatic HTTPS redirect (let ingress handle it)
echo "Disabling automatic HTTPS redirect..."
RESULT=$(curl -s -X PATCH "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/settings/always_use_https" \
  -H "Authorization: Bearer ${DNS_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{"value":"off"}')

SUCCESS=$(echo "$RESULT" | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
  echo "✓ Automatic HTTPS redirect disabled"
else
  echo "⚠️  Could not disable automatic redirect (may not be critical)"
fi

echo ""
echo "===== Configuration Complete ====="
echo ""
echo "SSL/TLS Mode: flexible"
echo "Always Use HTTPS: off"
echo ""
echo "This configuration works best for Cloudflare Tunnel + Kubernetes:"
echo "- Browser → Cloudflare: HTTPS (encrypted)"
echo "- Cloudflare → Tunnel → Ingress: HTTP (encrypted in tunnel)"
echo "- No redirect loops"
echo ""
echo "Wait 30 seconds for changes to propagate, then test:"
echo "  https://portal.254carbon.com"
echo "  https://superset.254carbon.com/superset/login"
echo "  https://dolphinscheduler.254carbon.com"
echo ""







