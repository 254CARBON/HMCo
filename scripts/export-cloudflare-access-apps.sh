#!/bin/bash
#
# Cloudflare Access Application Exporter
# Exports all Access applications and policies to a JSON file for backup and version control
#
# Usage:
#   ./export-cloudflare-access-apps.sh [-t TOKEN] [-a ACCOUNT_ID] [-o OUTPUT_FILE]
#
# Environment Variables:
#   CLOUDFLARE_API_TOKEN   - API token with Access:Apps read scope
#   CLOUDFLARE_ACCOUNT_ID  - Cloudflare account identifier (32 chars)
#

set -euo pipefail

API_TOKEN="${CLOUDFLARE_API_TOKEN:-}"
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-}"
OUTPUT_FILE="${OUTPUT_FILE:-cloudflare-access-apps-$(date +%Y%m%d-%H%M%S).json}"

API_BASE="https://api.cloudflare.com/client/v4"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

usage() {
    cat <<EOF
Usage: $0 [options]

Export Cloudflare Access applications and policies to JSON for backup and version control.

Options:
  -t, --token TOKEN         Cloudflare API token (overrides CLOUDFLARE_API_TOKEN)
  -a, --account-id ID       Cloudflare account ID (overrides CLOUDFLARE_ACCOUNT_ID)
  -o, --output FILE         Output file path (default: cloudflare-access-apps-TIMESTAMP.json)
  -h, --help                Show this help message

Examples:
  CLOUDFLARE_API_TOKEN=xxxxx CLOUDFLARE_ACCOUNT_ID=yyyy $0
  
  $0 --token xxxxx --account-id yyyy --output access-backup.json
EOF
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--token)
            API_TOKEN="$2"
            shift 2
            ;;
        -a|--account-id)
            ACCOUNT_ID="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$API_TOKEN" ]]; then
    log_error "Cloudflare API token not provided."
    log_info "Export CLOUDFLARE_API_TOKEN or pass --token."
    exit 1
fi

if [[ -z "$ACCOUNT_ID" ]]; then
    log_error "Cloudflare account ID not provided."
    log_info "Export CLOUDFLARE_ACCOUNT_ID or pass --account-id."
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    log_error "jq is required but not installed."
    exit 1
fi

api_request() {
    local method=$1
    local endpoint=$2
    local url="${API_BASE}/accounts/${ACCOUNT_ID}${endpoint}"
    local response

    response=$(curl -sS -X "$method" "$url" \
        -H "Authorization: Bearer ${API_TOKEN}" \
        -H "Content-Type: application/json")

    local success
    success=$(echo "$response" | jq -r '.success // empty' 2>/dev/null || echo "false")
    if [[ "$success" != "true" ]]; then
        local message
        message=$(echo "$response" | jq -r '[.errors[]?.message] | join("; ")' 2>/dev/null)
        if [[ -z "$message" || "$message" == "null" ]]; then
            message="Unknown error. Response: $response"
        fi
        log_error "$message"
        exit 1
    fi

    echo "$response"
}

log_info "Exporting Cloudflare Access applications..."
log_info "Account ID: ${ACCOUNT_ID}"
log_info "Output file: ${OUTPUT_FILE}"
echo ""

# Fetch all applications
log_info "Fetching applications..."
apps_response=$(api_request "GET" "/access/apps?page=1&per_page=100")
apps=$(echo "$apps_response" | jq -r '.result')
app_count=$(echo "$apps" | jq -r 'length')
log_success "Found ${app_count} applications"

# Build complete export with apps and their policies
export_data='{"exported_at":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'","account_id":"'${ACCOUNT_ID}'","applications":[]}'

for ((i=0; i<app_count; i++)); do
    app=$(echo "$apps" | jq -r ".[$i]")
    app_id=$(echo "$app" | jq -r '.id')
    app_name=$(echo "$app" | jq -r '.name')
    app_domain=$(echo "$app" | jq -r '.domain')
    
    log_info "Processing: ${app_name} (${app_domain})"
    
    # Fetch policies for this app (with pagination support)
    all_policies='[]'
    page=1
    per_page=50
    
    while true; do
        policies_response=$(api_request "GET" "/access/apps/${app_id}/policies?page=${page}&per_page=${per_page}")
        policies=$(echo "$policies_response" | jq -r '.result')
        policy_count=$(echo "$policies" | jq -r 'length')
        
        # Break if no more policies
        if [[ $policy_count -eq 0 ]]; then
            break
        fi
        
        # Append to all_policies
        all_policies=$(echo "$all_policies" | jq --argjson new "$policies" '. + $new')
        
        # Break if we got fewer than per_page (last page)
        if [[ $policy_count -lt $per_page ]]; then
            break
        fi
        
        ((page++))
    done
    
    policy_count=$(echo "$all_policies" | jq -r 'length')
    log_info "  Found ${policy_count} policies"
    
    # Add policies to app object
    app_with_policies=$(echo "$app" | jq --argjson policies "$all_policies" '. + {policies: $policies}')
    
    # Add to export
    export_data=$(echo "$export_data" | jq --argjson app "$app_with_policies" '.applications += [$app]')
done

# Write to file
echo "$export_data" | jq '.' > "$OUTPUT_FILE"

log_success "Export complete!"
log_info "Exported ${app_count} applications with their policies to: ${OUTPUT_FILE}"
echo ""

# Print summary
log_info "Summary:"
echo "$export_data" | jq -r '.applications[] | "  - \(.name) (\(.domain)) - \(.policies | length) policies"'
echo ""

log_info "To version control this export:"
echo "  git add ${OUTPUT_FILE}"
echo "  git commit -m 'Backup Cloudflare Access configuration'"
echo ""

log_info "To restore/import from this export:"
echo "  Use create-cloudflare-access-apps.sh with --force flag"
echo "  It will update existing apps to match the configuration"
