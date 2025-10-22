#!/bin/bash
# Configure Cloudflare DNS Records for 254Carbon Platform
# All services route through the Cloudflare Tunnel

set -e

# Cloudflare credentials
ACCOUNT_ID="0c93c74d5269a228e91d4bf91c547f56"
DNS_API_TOKEN="acXHRLyetL39qEcd4hIuW1omGxq8cxu65PN5yMAm"
TUNNEL_ID="291bc289-e3c3-4446-a9ad-8e327660ecd5"
ZONE_NAME="254carbon.com"

echo "=================================="
echo "Cloudflare DNS Configuration"
echo "=================================="
echo

# Get Zone ID
echo "1. Fetching Zone ID for $ZONE_NAME..."
ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=$ZONE_NAME" \
  -H "Authorization: Bearer $DNS_API_TOKEN" \
  -H "Content-Type: application/json" | jq -r '.result[0].id')

if [ "$ZONE_ID" == "null" ] || [ -z "$ZONE_ID" ]; then
  echo "ERROR: Could not fetch Zone ID for $ZONE_NAME"
  exit 1
fi

echo "   Zone ID: $ZONE_ID"
echo

# List of DNS records to create/update
# Format: "subdomain:type"
DNS_RECORDS=(
  "rapids:CNAME"
  "dolphinscheduler:CNAME"
  "harbor:CNAME"
  "portal:CNAME"
  "datahub:CNAME"
  "superset:CNAME"
  "grafana:CNAME"
  "trino:CNAME"
  "vault:CNAME"
  "minio:CNAME"
  "lakefs:CNAME"
  "www:CNAME"
  "@:CNAME"  # Root domain
)

# Tunnel CNAME target
TUNNEL_CNAME="${TUNNEL_ID}.cfargotunnel.com"

echo "2. Configuring DNS records..."
echo "   Tunnel CNAME: $TUNNEL_CNAME"
echo

for record in "${DNS_RECORDS[@]}"; do
  IFS=':' read -r subdomain type <<< "$record"
  
  # Build full hostname
  if [ "$subdomain" == "@" ]; then
    hostname="$ZONE_NAME"
    display_name="$ZONE_NAME"
  else
    hostname="$subdomain.$ZONE_NAME"
    display_name="$hostname"
  fi
  
  echo "   Processing: $display_name..."
  
  # Check if record exists
  existing_record=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=$hostname&type=$type" \
    -H "Authorization: Bearer $DNS_API_TOKEN" \
    -H "Content-Type: application/json" | jq -r '.result[0].id')
  
  if [ "$existing_record" != "null" ] && [ -n "$existing_record" ]; then
    # Update existing record
    echo "      → Updating existing record (ID: $existing_record)"
    response=$(curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$existing_record" \
      -H "Authorization: Bearer $DNS_API_TOKEN" \
      -H "Content-Type: application/json" \
      --data "{
        \"type\": \"$type\",
        \"name\": \"$hostname\",
        \"content\": \"$TUNNEL_CNAME\",
        \"ttl\": 1,
        \"proxied\": true
      }")
    
    success=$(echo $response | jq -r '.success')
    if [ "$success" == "true" ]; then
      echo "      ✓ Updated successfully"
    else
      error=$(echo $response | jq -r '.errors[0].message')
      echo "      ✗ Failed: $error"
    fi
  else
    # Create new record
    echo "      → Creating new record"
    response=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
      -H "Authorization: Bearer $DNS_API_TOKEN" \
      -H "Content-Type: application/json" \
      --data "{
        \"type\": \"$type\",
        \"name\": \"$hostname\",
        \"content\": \"$TUNNEL_CNAME\",
        \"ttl\": 1,
        \"proxied\": true
      }")
    
    success=$(echo $response | jq -r '.success')
    if [ "$success" == "true" ]; then
      echo "      ✓ Created successfully"
    else
      error=$(echo $response | jq -r '.errors[0].message')
      echo "      ✗ Failed: $error"
    fi
  fi
done

echo
echo "=================================="
echo "DNS Configuration Complete"
echo "=================================="
echo
echo "All records point to tunnel: $TUNNEL_CNAME"
echo "DNS propagation time: 1-5 minutes"
echo
echo "Verify DNS:"
echo "  dig rapids.254carbon.com"
echo "  dig dolphinscheduler.254carbon.com"
echo

