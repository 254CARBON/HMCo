#!/bin/bash

# Re-enable Cloudflare Access for Superset
# Run this after confirming Superset works without Access

set -e

ACCOUNT_ID="0c93c74d5269a228e91d4bf91c547f56"
APPS_API_TOKEN="TYSD6Xrn8BJEwGp76t32-a331-L82fCNkbsJx7Mn"

echo "===== Re-enabling Cloudflare Access for Superset ====="
echo ""

# Create new Access application for Superset
echo "Creating Cloudflare Access application..."

RESPONSE=$(curl -s -X POST \
  "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps" \
  -H "Authorization: Bearer ${APPS_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "name": "Superset.254Carbon",
    "domain": "superset.254carbon.com",
    "type": "self_hosted",
    "session_duration": "24h",
    "auto_redirect_to_identity": false,
    "enable_binding_cookie": false,
    "http_only_cookie_attribute": true,
    "same_site_cookie_attribute": "lax",
    "allowed_idps": [],
    "app_launcher_visible": true
  }')

SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
APP_ID=$(echo "$RESPONSE" | jq -r '.result.id')

if [ "$SUCCESS" = "true" ]; then
  echo "✓ Access application created"
  echo "  App ID: ${APP_ID}"
else
  ERROR=$(echo "$RESPONSE" | jq -r '.errors[0].message // "Unknown error"')
  echo "❌ Failed: ${ERROR}"
  exit 1
fi

echo ""
echo "Creating access policy..."

# Create policy allowing @254carbon.com emails
POLICY_RESPONSE=$(curl -s -X POST \
  "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps/${APP_ID}/policies" \
  -H "Authorization: Bearer ${APPS_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "name": "Allow Superset Access",
    "decision": "allow",
    "include": [
      {
        "email_domain": {
          "domain": "254carbon.com"
        }
      }
    ],
    "require": [],
    "exclude": []
  }')

POLICY_SUCCESS=$(echo "$POLICY_RESPONSE" | jq -r '.success')

if [ "$POLICY_SUCCESS" = "true" ]; then
  echo "✓ Access policy created"
  echo ""
  echo "===== Superset Access Re-enabled ====="
  echo ""
  echo "You can now access Superset at:"
  echo "  https://superset.254carbon.com/superset/login"
  echo ""
  echo "Login flow:"
  echo "  1. Cloudflare Access login (GitHub or Email)"
  echo "  2. Superset login (admin/admin)"
  echo ""
else
  ERROR=$(echo "$POLICY_RESPONSE" | jq -r '.errors[0].message // "Unknown error"')
  echo "❌ Policy creation failed: ${ERROR}"
  exit 1
fi





