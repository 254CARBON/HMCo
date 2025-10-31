#!/bin/bash
#
# Create/Update Cloudflare DNS Records for Tunnel
# This script ensures all DNS records point to your Cloudflare Tunnel
# Non-interactive and idempotent - safe to run multiple times
#
# Usage:
#   ./create-cloudflare-dns-records.sh [options]
#
# Environment Variables:
#   CLOUDFLARE_API_TOKEN   - API token with DNS edit permissions
#   CLOUDFLARE_ZONE_ID     - Zone ID (optional, will be fetched if not provided)
#   CLOUDFLARE_TUNNEL_ID   - Tunnel ID

set -euo pipefail

# Configuration from environment or command line
DNS_API_TOKEN="${CLOUDFLARE_API_TOKEN:-}"
ZONE_ID="${CLOUDFLARE_ZONE_ID:-}"
TUNNEL_ID="${CLOUDFLARE_TUNNEL_ID:-}"
ZONE_NAME="${CLOUDFLARE_ZONE_NAME:-254carbon.com}"
DRY_RUN=false
FORCE=false

usage() {
    cat <<EOF
Usage: $0 [options]

Create or update Cloudflare DNS records to point to the tunnel.

Options:
  --token TOKEN         Cloudflare API token (overrides CLOUDFLARE_API_TOKEN)
  --zone-id ID          Zone ID (overrides CLOUDFLARE_ZONE_ID, auto-detected if not provided)
  --tunnel-id ID        Tunnel ID (overrides CLOUDFLARE_TUNNEL_ID)
  --zone-name NAME      Zone name (default: 254carbon.com)
  --dry-run             Show what would be done without making changes
  --force               Update records even if they already exist
  -h, --help            Show this help message

Environment Variables:
  CLOUDFLARE_API_TOKEN  - API token with DNS:Edit permission
  CLOUDFLARE_ZONE_ID    - Optional, will be fetched if not set
  CLOUDFLARE_TUNNEL_ID  - Tunnel UUID
  CLOUDFLARE_ZONE_NAME  - Domain name (default: 254carbon.com)

Examples:
  # Basic usage with environment variables
  export CLOUDFLARE_API_TOKEN=<token>
  export CLOUDFLARE_TUNNEL_ID=<tunnel-id>
  $0

  # With command line arguments
  $0 --token <token> --tunnel-id <tunnel-id> --zone-name 254carbon.com

  # Dry run to preview changes
  $0 --dry-run

  # Force update of existing records
  $0 --force
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --token)
            DNS_API_TOKEN="$2"
            shift 2
            ;;
        --zone-id)
            ZONE_ID="$2"
            shift 2
            ;;
        --tunnel-id)
            TUNNEL_ID="$2"
            shift 2
            ;;
        --zone-name)
            ZONE_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DNS_API_TOKEN" ]]; then
    echo "❌ ERROR: Cloudflare API token not provided" >&2
    echo "Set CLOUDFLARE_API_TOKEN or use --token" >&2
    exit 1
fi

if [[ -z "$TUNNEL_ID" ]]; then
    echo "❌ ERROR: Tunnel ID not provided" >&2
    echo "Set CLOUDFLARE_TUNNEL_ID or use --tunnel-id" >&2
    exit 1
fi

echo "===== Cloudflare DNS Record Setup ====="
echo ""
echo "Zone: ${ZONE_NAME}"
echo "Tunnel ID: ${TUNNEL_ID}"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Mode: DRY RUN (no changes will be made)"
fi
echo ""

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "❌ ERROR: jq is required but not installed"
    echo "Install: sudo apt-get install jq"
    exit 1
fi

# Get Zone ID if not provided
if [[ -z "$ZONE_ID" ]]; then
    echo "Fetching Zone ID for ${ZONE_NAME}..."
    ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=${ZONE_NAME}" \
      -H "Authorization: Bearer ${DNS_API_TOKEN}" \
      -H "Content-Type: application/json" | jq -r '.result[0].id')

    if [[ -z "$ZONE_ID" ]] || [[ "$ZONE_ID" == "null" ]]; then
      echo "❌ ERROR: Could not find zone for ${ZONE_NAME}"
      exit 1
    fi
    echo "✓ Zone ID: ${ZONE_ID}"
else
    echo "✓ Using provided Zone ID: ${ZONE_ID}"
fi
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

CREATED=0
UPDATED=0
SKIPPED=0
FAILED=0

for key in "${!DOMAINS[@]}"; do
  DOMAIN="${DOMAINS[$key]}"
  echo "Processing: ${DOMAIN}"
  
  # Check if record exists
  EXISTING=$(curl -s -X GET \
    "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?name=${DOMAIN}" \
    -H "Authorization: Bearer ${DNS_API_TOKEN}" \
    -H "Content-Type: application/json")
  
  RECORD_ID=$(echo "$EXISTING" | jq -r '.result[0].id // empty')
  CURRENT_CONTENT=$(echo "$EXISTING" | jq -r '.result[0].content // empty')
  CURRENT_PROXIED=$(echo "$EXISTING" | jq -r '.result[0].proxied // false')
  
  # Check if record already points to the correct target
  if [[ -n "$RECORD_ID" ]] && [[ "$CURRENT_CONTENT" == "$TUNNEL_CNAME" ]] && [[ "$CURRENT_PROXIED" == "true" ]]; then
    if [[ "$FORCE" == "true" ]]; then
      echo "  Record already correct, but forcing update..."
    else
      echo "  ✓ Record already exists and is correct (ID: ${RECORD_ID})"
      ((SKIPPED++))
      continue
    fi
  fi
  
  if [[ "$DRY_RUN" == "true" ]]; then
    if [[ -n "$RECORD_ID" ]]; then
      echo "  [DRY RUN] Would update record (ID: ${RECORD_ID})"
      echo "    Current: ${CURRENT_CONTENT} (proxied: ${CURRENT_PROXIED})"
      echo "    New: ${TUNNEL_CNAME} (proxied: true)"
    else
      echo "  [DRY RUN] Would create new record"
      echo "    CNAME ${DOMAIN} -> ${TUNNEL_CNAME} (proxied: true)"
    fi
    echo ""
    continue
  fi
  
  if [[ -n "$RECORD_ID" ]] && [[ "$RECORD_ID" != "null" ]]; then
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
    
    SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
    if [[ "$SUCCESS" == "true" ]]; then
      echo "  ✓ Updated"
      ((UPDATED++))
    else
      ERROR=$(echo "$RESPONSE" | jq -r '.errors[0].message // "Unknown error"')
      echo "  ❌ Failed: ${ERROR}"
      ((FAILED++))
    fi
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
    
    SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
    if [[ "$SUCCESS" == "true" ]]; then
      echo "  ✓ Created"
      ((CREATED++))
    else
      ERROR=$(echo "$RESPONSE" | jq -r '.errors[0].message // "Unknown error"')
      echo "  ❌ Failed: ${ERROR}"
      ((FAILED++))
    fi
  fi
  echo ""
done

echo "===== Summary ====="
echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN completed - no changes were made"
else
    echo "DNS Records:"
    echo "  Created: ${CREATED}"
    echo "  Updated: ${UPDATED}"
    echo "  Skipped (already correct): ${SKIPPED}"
    echo "  Failed: ${FAILED}"
fi
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "⚠️  Some records failed to update. Check the errors above."
    exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Run without --dry-run to apply changes"
    exit 0
fi

if [[ $CREATED -gt 0 ]] || [[ $UPDATED -gt 0 ]]; then
    echo "✓ DNS records updated successfully"
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
    echo "2. Verify Cloudflare SSL mode is set to 'Full (strict)'"
    echo "   WARNING: If currently using 'Flexible' mode, upgrade to 'Full (strict)' may require:"
    echo "   - Valid SSL certificates on origin (see docs/cloudflare/origin-certificates-setup.md)"
    echo "   - NGINX Ingress Controller configured for HTTPS"
    echo "3. Check tunnel is connected: kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel"
else
    echo "✓ All DNS records are already correctly configured"
fi
echo ""





