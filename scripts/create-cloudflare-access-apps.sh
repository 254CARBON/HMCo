#!/bin/bash
#
# Cloudflare Access Application Creator
# Automates creation (or update) of the 10 Phase 2 self-hosted applications
# for the 254Carbon SSO rollout.
#
# Usage:
#   ./create-cloudflare-access-apps.sh [-t TOKEN] [-a ACCOUNT_ID] [--force]
#
# Environment Variables:
#   CLOUDFLARE_API_TOKEN   - API token with Access:Apps write scope
#   CLOUDFLARE_ACCOUNT_ID  - Cloudflare account identifier (32 chars)
#

set -euo pipefail

API_TOKEN="${CLOUDFLARE_API_TOKEN:-}"
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-}"

# Domain mode: "team" (use <sub>.<TEAM_NAME>.cloudflareaccess.com) or "zone" (use <sub>.<ZONE_DOMAIN>)
CLOUDFLARE_ACCESS_MODE="${CLOUDFLARE_ACCESS_MODE:-team}"
TEAM_NAME="${CLOUDFLARE_TEAM_NAME:-}"              # e.g. qagi
TEAM_DOMAIN="${CLOUDFLARE_TEAM_DOMAIN:-}"          # e.g. qagi.cloudflareaccess.com
ZONE_DOMAIN="${CLOUDFLARE_ZONE_DOMAIN:-}"          # e.g. 254carbon.com

# Policy inputs
ALLOWED_EMAIL_DOMAINS="${CLOUDFLARE_ACCESS_ALLOWED_EMAIL_DOMAINS:-254carbon.com}" # comma-separated
ALLOWED_EMAILS="${CLOUDFLARE_ACCESS_ALLOWED_EMAILS:-}"                              # comma-separated
EXCLUDED_EMAILS="${CLOUDFLARE_ACCESS_EXCLUDED_EMAILS:-}"                            # comma-separated
COUNTRIES="${CLOUDFLARE_ACCESS_ALLOWED_COUNTRIES:-}"                                # comma-separated ISO codes
IDP_ID="${CLOUDFLARE_ACCESS_IDP_ID:-}"                                               # optional IdP (login method) UUID

ACME_BYPASS_ENABLED="${CLOUDFLARE_ACME_BYPASS_ENABLED:-true}"
ACME_BYPASS_SESSION="${CLOUDFLARE_ACME_BYPASS_SESSION:-1h}"
BYPASS_POLICY_NAME="Bypass ACME Challenge"

FORCE_UPDATE=false
DRY_RUN=false
SKIP_EXISTING=false

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

Create or refresh the 254Carbon Cloudflare Access applications required in Phase 2.

Options:
  -t, --token TOKEN         Cloudflare API token (overrides CLOUDFLARE_API_TOKEN)
  -a, --account-id ID       Cloudflare account ID (overrides CLOUDFLARE_ACCOUNT_ID)
      --mode MODE           Domain mode: team|zone (default: ${CLOUDFLARE_ACCESS_MODE})
      --team-name NAME      Cloudflare Zero Trust team name (for team mode)
      --team-domain NAME    Cloudflare Zero Trust team domain (e.g. qagi.cloudflareaccess.com)
      --zone-domain NAME    Public DNS zone (e.g. 254carbon.com) for zone mode
      --allowed-email-domains CSV  Allowed email domains (default: ${ALLOWED_EMAIL_DOMAINS})
      --allowed-emails CSV  Allowed individual emails (optional)
      --excluded-emails CSV Blocked individual emails (optional)
      --countries CSV       Allowed ISO country codes (optional)
      --idp-id UUID         Restrict login method to this IdP UUID (optional)
      --force               Update apps/policies if they already exist (reconcile to desired state)
      --skip-existing       Skip updating existing apps/policies (create new ones only)
      --dry-run             Print payloads without calling Cloudflare API
      --skip-acme-bypass    Skip creating ACME HTTP-01 bypass exemptions
  -h, --help                Show this help message

Examples:
  CLOUDFLARE_API_TOKEN=xxxxx CLOUDFLARE_ACCOUNT_ID=yyyy $0 \
    --mode team --team-name qagi --allowed-email-domains 254carbon.com

  $0 --token xxxxx --account-id yyyy --mode zone --zone-domain 254carbon.com --force
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
        --mode)
            CLOUDFLARE_ACCESS_MODE="$2"
            shift 2
            ;;
        --team-name)
            TEAM_NAME="$2"
            shift 2
            ;;
        --team-domain)
            TEAM_DOMAIN="$2"
            shift 2
            ;;
        --zone-domain)
            ZONE_DOMAIN="$2"
            shift 2
            ;;
        -d|--base-domain)
            # Deprecated: treat as zone-domain for backward compatibility
            ZONE_DOMAIN="$2"
            shift 2
            ;;
        --allowed-email-domains)
            ALLOWED_EMAIL_DOMAINS="$2"
            shift 2
            ;;
        --allowed-emails)
            ALLOWED_EMAILS="$2"
            shift 2
            ;;
        --excluded-emails)
            EXCLUDED_EMAILS="$2"
            shift 2
            ;;
        --countries)
            COUNTRIES="$2"
            shift 2
            ;;
        --idp-id)
            IDP_ID="$2"
            shift 2
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-acme-bypass)
            ACME_BYPASS_ENABLED=false
            shift
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

# Validate mutually exclusive flags
if [[ "$FORCE_UPDATE" == "true" ]] && [[ "$SKIP_EXISTING" == "true" ]]; then
    log_error "Cannot use both --force and --skip-existing flags together"
    log_info "Use --force to update all existing apps, or --skip-existing to only create new apps"
    exit 1
fi

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

case "$CLOUDFLARE_ACCESS_MODE" in
  team)
    if [[ -z "$TEAM_NAME" && -z "$TEAM_DOMAIN" ]]; then
        log_error "Team mode selected but neither team name nor team domain provided."
        log_info "Set CLOUDFLARE_TEAM_NAME / CLOUDFLARE_TEAM_DOMAIN or pass --team-name / --team-domain."
        exit 1
    fi
    if [[ -z "$TEAM_DOMAIN" ]]; then
        TEAM_DOMAIN="${TEAM_NAME}.cloudflareaccess.com"
    elif [[ -z "$TEAM_NAME" ]]; then
        TEAM_NAME="${TEAM_DOMAIN%%.cloudflareaccess.com}"
    fi
    ;;
  zone)
    if [[ -z "$ZONE_DOMAIN" ]]; then
        log_error "Zone mode selected but zone domain not provided."
        log_info "Set CLOUDFLARE_ZONE_DOMAIN or pass --zone-domain."
        exit 1
    fi
    ;;
  *)
    log_error "Unknown mode: $CLOUDFLARE_ACCESS_MODE (expected team|zone)"
    exit 1
    ;;
esac

if ! command -v jq >/dev/null 2>&1; then
    log_error "jq is required but not installed."
    exit 1
fi

api_request() {
    local method=$1
    local endpoint=$2
    local data=${3:-}
    local url="${API_BASE}/accounts/${ACCOUNT_ID}${endpoint}"
    local response

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "{"success":true,"result":{}}"
        return 0
    fi

    if [[ -n "$data" ]]; then
        response=$(curl -sS -X "$method" "$url" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$data")
    else
        response=$(curl -sS -X "$method" "$url" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json")
    fi

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

get_existing_app() {
    local domain=$1
    api_request "GET" "/access/apps?page=1&per_page=100" | \
        jq -r --arg domain "$domain" '.result[] | select(.domain == $domain)'
}

build_app_payload() {
    local name=$1
    local domain=$2
    local session=$3
    local launcher_visible=${4:-true}
    local auto_redirect=${5:-false}
    cat <<EOF
{
  "name": "$name",
  "domain": "$domain",
  "type": "self_hosted",
  "session_duration": "$session",
  "app_launcher_visible": ${launcher_visible},
  "http_only_cookie": true,
  "auto_redirect_to_identity": ${auto_redirect}
}
EOF
}

create_app() {
    local name=$1
    local domain=$2
    local session=$3
    local launcher_visible=${4:-true}
    local auto_redirect=${5:-false}
    local payload
    payload=$(build_app_payload "$name" "$domain" "$session" "$launcher_visible" "$auto_redirect")
    api_request "POST" "/access/apps" "$payload" | jq -r '.result.id'
}

update_app() {
    local app_id=$1
    local name=$2
    local domain=$3
    local session=$4
    local launcher_visible=${5:-true}
    local auto_redirect=${6:-false}
    local payload
    payload=$(build_app_payload "$name" "$domain" "$session" "$launcher_visible" "$auto_redirect")
    api_request "PUT" "/access/apps/${app_id}" "$payload" >/dev/null
}

get_existing_policy() {
    local app_id=$1
    local policy_name=$2
    api_request "GET" "/access/apps/${app_id}/policies?page=1&per_page=50" | \
        jq -r --arg name "$policy_name" '.result[] | select(.name == $name)'
}

csv_to_json_rules() {
    local type="$1"   # email_domain|email|ip|geo|login_method
    local csv="$2"
    local key="$3"    # key name inside object (domain/email/ip/country_code/id)
    local IFS=','
    local first=true
    for item in $csv; do
        local trimmed
        trimmed=$(echo "$item" | xargs)
        [[ -z "$trimmed" ]] && continue
        if [[ "$first" == true ]]; then
            first=false
        else
            echo -n ","
        fi
        if [[ "$type" == "login_method" ]]; then
            # login_method only supports single id at a time; handled in require
            echo -n "{\"login_method\":{\"id\":\"$trimmed\"}}"
        else
            echo -n "{\"$type\":{\"$key\":\"$trimmed\"}}"
        fi
    done
}

build_policy_payload() {
    local name=$1

    # Build include rules
    local include_rules=""
    if [[ -n "$ALLOWED_EMAIL_DOMAINS" ]]; then
        include_rules+=$(csv_to_json_rules "email_domain" "$ALLOWED_EMAIL_DOMAINS" "domain")
    fi
    if [[ -n "$ALLOWED_EMAILS" ]]; then
        [[ -n "$include_rules" ]] && include_rules+=" ,"
        include_rules+=$(csv_to_json_rules "email" "$ALLOWED_EMAILS" "email")
    fi
    # Fallback to everyone if nothing specified
    if [[ -z "$include_rules" ]]; then
        include_rules='{ "everyone": {} }'
    fi

    # Build require rules
    local require_rules=""
    if [[ -n "$COUNTRIES" ]]; then
        require_rules+=$(csv_to_json_rules "geo" "$COUNTRIES" "country_code")
    fi
    if [[ -n "$IDP_ID" ]]; then
        [[ -n "$require_rules" ]] && require_rules+=" ,"
        require_rules+=$(csv_to_json_rules "login_method" "$IDP_ID" "id")
    fi

    # Build exclude rules
    local exclude_rules=""
    if [[ -n "$EXCLUDED_EMAILS" ]]; then
        exclude_rules+=$(csv_to_json_rules "email" "$EXCLUDED_EMAILS" "email")
    fi

    cat <<EOF
{
  "name": "$name",
  "precedence": 1,
  "decision": "allow",
  "include": [ $include_rules ],
  "require": [ $require_rules ],
  "exclude": [ $exclude_rules ],
  "approval_required": false
}
EOF
}

create_policy() {
    local app_id=$1
    local policy_name=$2
    local payload
    payload=$(build_policy_payload "$policy_name")
    api_request "POST" "/access/apps/${app_id}/policies" "$payload" >/dev/null
}

update_policy() {
    local app_id=$1
    local policy_id=$2
    local policy_name=$3
    local payload
    payload=$(build_policy_payload "$policy_name")
    api_request "PUT" "/access/apps/${app_id}/policies/${policy_id}" "$payload" >/dev/null
}

build_bypass_policy_payload() {
    local name=$1
    cat <<EOF
{
  "name": "$name",
  "precedence": 0,
  "decision": "bypass",
  "include": [ { "everyone": {} } ],
  "require": [],
  "exclude": [],
  "approval_required": false
}
EOF
}

create_bypass_policy() {
    local app_id=$1
    local policy_name=$2
    local payload
    payload=$(build_bypass_policy_payload "$policy_name")
    api_request "POST" "/access/apps/${app_id}/policies" "$payload" >/dev/null
}

update_bypass_policy() {
    local app_id=$1
    local policy_id=$2
    local policy_name=$3
    local payload
    payload=$(build_bypass_policy_payload "$policy_name")
    api_request "PUT" "/access/apps/${app_id}/policies/${policy_id}" "$payload" >/dev/null
}

ensure_acme_bypass() {
    local app_label=$1
    local base_domain=$2

    if [[ "$base_domain" == *"cloudflareaccess.com" ]] || \
       [[ "$base_domain" == *".local" ]] || \
       [[ "$base_domain" == "localhost" ]] || \
       [[ "$base_domain" == "127.0.0.1" ]]; then
        log_info "Skipping ACME bypass for ${base_domain} (non-public domain)."
        return
    fi

    local bypass_domain="${base_domain}/.well-known/acme-challenge/*"
    local bypass_app_name="ACME Bypass - ${base_domain}"
    local policy_name="${BYPASS_POLICY_NAME}"

    log_info "Ensuring ACME HTTP-01 bypass for ${bypass_domain}"

    local existing_app app_id
    existing_app=$(get_existing_app "$bypass_domain" || echo "")
    app_id=""

    if [[ -n "$existing_app" ]]; then
        app_id=$(echo "$existing_app" | jq -r '.id')
        log_warning "ACME bypass application already exists."
        if [[ "$FORCE_UPDATE" == "true" ]]; then
            log_info "Updating ACME bypass application settings..."
            update_app "$app_id" "$bypass_app_name" "$bypass_domain" "$ACME_BYPASS_SESSION" false false
            log_success "ACME bypass application updated."
        fi
    else
        log_info "Creating ACME bypass application..."
        app_id=$(create_app "$bypass_app_name" "$bypass_domain" "$ACME_BYPASS_SESSION" false false)
        log_success "ACME bypass application created (ID: ${app_id})."
    fi

    if [[ -z "$app_id" || "$app_id" == "null" ]]; then
        log_error "Unable to determine ACME bypass application ID for ${base_domain}."
        exit 1
    fi

    local existing_policy
    existing_policy=$(get_existing_policy "$app_id" "$policy_name" || echo "")

    if [[ -n "$existing_policy" ]]; then
        if [[ "$FORCE_UPDATE" == "true" ]]; then
            local policy_id
            policy_id=$(echo "$existing_policy" | jq -r '.id')
            log_info "Refreshing ACME bypass policy..."
            update_bypass_policy "$app_id" "$policy_id" "$policy_name"
            log_success "ACME bypass policy updated."
        else
            log_success "ACME bypass policy already present."
        fi
    else
        log_info "Creating ACME bypass policy..."
        create_bypass_policy "$app_id" "$policy_name"
        log_success "ACME bypass policy created."
    fi
}

declare -a APPLICATIONS=(
    "254Carbon Portal|portal|24h|Allow Portal Access"
    "Grafana.254Carbon|grafana|24h|Allow Grafana Access"
    "Superset.254Carbon|superset|24h|Allow Superset Access"
    "DataHub.254Carbon|datahub|12h|Allow DataHub Access"
    "Trino.254Carbon|trino|8h|Allow Trino Access"
    "ClickHouse.254Carbon|clickhouse|8h|Allow ClickHouse Access"
    "Vault.254Carbon|vault|2h|Allow Vault Access"
    "MinIO.254Carbon|minio|8h|Allow MinIO Access"
    "DolphinScheduler.254Carbon|dolphin|12h|Allow DolphinScheduler Access"
    "LakeFS.254Carbon|lakefs|12h|Allow LakeFS Access"
    "MLflow.254Carbon|mlflow|12h|Allow MLflow Access"
    "Spark History.254Carbon|spark-history|12h|Allow Spark History Access"
)

# When using public zone domains, also protect the apex and www hostnames
if [[ "$CLOUDFLARE_ACCESS_MODE" == "zone" ]]; then
    APPLICATIONS+=(
        "254Carbon Root|@|24h|Allow Portal Access"
        "254Carbon WWW|www|24h|Allow Portal Access"
    )
fi

log_info "Preparing to configure Cloudflare Access applications..."
log_info "Account ID: ${ACCOUNT_ID}"

# If dry-run, print planned payloads and exit without API calls
if [[ "$DRY_RUN" == "true" ]]; then
    for entry in "${APPLICATIONS[@]}"; do
        IFS='|' read -r name subdomain session policy <<< "$entry"
        case "$CLOUDFLARE_ACCESS_MODE" in
          team)
            if [[ "$subdomain" == "@" ]]; then
                domain="${TEAM_DOMAIN}"
            else
                domain="${subdomain}.${TEAM_DOMAIN}"
            fi
            ;;
          zone)
            if [[ "$subdomain" == "@" ]]; then
                domain="${ZONE_DOMAIN}"
            else
                domain="${subdomain}.${ZONE_DOMAIN}"
            fi
            ;;
        esac

        echo
        log_info "[DRY-RUN] Application: $name"
        echo "Domain: $domain"
        echo "Session: $session"
        echo "App payload:"
        build_app_payload "$name" "$domain" "$session" | jq .
        echo "Policy '$policy' payload:"
        build_policy_payload "$policy" | jq .
    done
    echo
    log_success "[DRY-RUN] Completed. No changes made."
    exit 0
fi

for entry in "${APPLICATIONS[@]}"; do
    IFS='|' read -r name subdomain session policy <<< "$entry"
    case "$CLOUDFLARE_ACCESS_MODE" in
      team)
        if [[ "$subdomain" == "@" ]]; then
            domain="${TEAM_DOMAIN}"
        else
            domain="${subdomain}.${TEAM_DOMAIN}"
        fi
        ;;
      zone)
        if [[ "$subdomain" == "@" ]]; then
            domain="${ZONE_DOMAIN}"
        else
            domain="${subdomain}.${ZONE_DOMAIN}"
        fi
        ;;
    esac

    log_info "Processing ${name} (${domain})"

    existing_app=$(get_existing_app "$domain" || echo "")
    app_id=""

    if [[ -n "$existing_app" ]]; then
        app_id=$(echo "$existing_app" | jq -r '.id')
        log_warning "Application already exists."
        if [[ "$SKIP_EXISTING" == "true" ]]; then
            log_info "Skipping (--skip-existing flag set)."
        elif [[ "$FORCE_UPDATE" == "true" ]]; then
            log_info "Updating existing application settings..."
            update_app "$app_id" "$name" "$domain" "$session"
            log_success "Application updated (reconciled to desired state)."
        else
            log_info "Use --force to update existing app or --skip-existing to skip."
        fi
    else
        log_info "Creating new application..."
        app_id=$(create_app "$name" "$domain" "$session")
        log_success "Application created (ID: ${app_id})."
    fi

    if [[ -z "$app_id" || "$app_id" == "null" ]]; then
        log_error "Unable to determine application ID for ${name}."
        exit 1
    fi

    existing_policy=$(get_existing_policy "$app_id" "$policy" || echo "")

    if [[ -n "$existing_policy" ]]; then
        if [[ "$SKIP_EXISTING" == "true" ]]; then
            log_success "Policy ${policy} already present (skipped)."
        elif [[ "$FORCE_UPDATE" == "true" ]]; then
            policy_id=$(echo "$existing_policy" | jq -r '.id')
            log_info "Refreshing policy ${policy}..."
            update_policy "$app_id" "$policy_id" "$policy"
            log_success "Policy updated (reconciled to desired state)."
        else
            log_success "Policy ${policy} already present."
        fi
    else
        log_info "Creating policy ${policy}..."
        create_policy "$app_id" "$policy"
        log_success "Policy created."
    fi

    if [[ "$ACME_BYPASS_ENABLED" == "true" ]]; then
        ensure_acme_bypass "$name" "$domain"
    fi
done

log_success "All Cloudflare Access application tasks complete."
