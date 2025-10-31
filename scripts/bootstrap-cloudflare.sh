#!/bin/bash
#
# Cloudflare Infrastructure Bootstrap Script
# One-command deployment for Cloudflare Tunnel, DNS, and Access
#
# Usage:
#   export CLOUDFLARE_API_TOKEN=<token>
#   export CLOUDFLARE_ACCOUNT_ID=<account-id>
#   export CLOUDFLARE_TUNNEL_ID=<tunnel-id>
#   ./bootstrap-cloudflare.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
API_TOKEN="${CLOUDFLARE_API_TOKEN:-}"
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-}"
TUNNEL_ID="${CLOUDFLARE_TUNNEL_ID:-}"
ZONE_NAME="${CLOUDFLARE_ZONE_NAME:-254carbon.com}"
ZONE_ID="${CLOUDFLARE_ZONE_ID:-}"

# Access configuration
ACCESS_MODE="${CLOUDFLARE_ACCESS_MODE:-zone}"
ALLOWED_EMAIL_DOMAINS="${CLOUDFLARE_ACCESS_ALLOWED_EMAIL_DOMAINS:-254carbon.com}"

# Flags
DRY_RUN=false
SKIP_TUNNEL=false
SKIP_DNS=false
SKIP_ACCESS=false
SKIP_DEPLOY=false
VERIFY=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

usage() {
    cat <<EOF
Usage: $0 [options]

Bootstrap complete Cloudflare infrastructure: Tunnel, DNS, and Access.

Required Environment Variables:
  CLOUDFLARE_API_TOKEN          API token with Tunnel:Read, DNS:Edit, Access:Edit
  CLOUDFLARE_ACCOUNT_ID         Cloudflare account ID
  CLOUDFLARE_TUNNEL_ID          Tunnel UUID (must be pre-created)

Optional Environment Variables:
  CLOUDFLARE_ZONE_NAME          Zone name (default: 254carbon.com)
  CLOUDFLARE_ZONE_ID            Zone ID (auto-detected if not set)
  CLOUDFLARE_ACCESS_MODE        Access mode: zone or team (default: zone)
  CLOUDFLARE_ACCESS_ALLOWED_EMAIL_DOMAINS  Allowed email domains (default: 254carbon.com)

Options:
  --dry-run                     Show what would be done without making changes
  --skip-tunnel                 Skip tunnel token configuration
  --skip-dns                    Skip DNS record creation
  --skip-access                 Skip Access application creation
  --skip-deploy                 Skip Kubernetes deployment
  --no-verify                   Skip verification steps
  -h, --help                    Show this help message

Examples:
  # Full bootstrap
  export CLOUDFLARE_API_TOKEN=<token>
  export CLOUDFLARE_ACCOUNT_ID=<account-id>
  export CLOUDFLARE_TUNNEL_ID=<tunnel-id>
  $0

  # Dry run to preview
  $0 --dry-run

  # Skip tunnel config (already done)
  $0 --skip-tunnel
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tunnel)
            SKIP_TUNNEL=true
            shift
            ;;
        --skip-dns)
            SKIP_DNS=true
            shift
            ;;
        --skip-access)
            SKIP_ACCESS=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --no-verify)
            VERIFY=false
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

# Validate prerequisites
log_step "Validating Prerequisites"

if [[ -z "$API_TOKEN" ]]; then
    log_error "CLOUDFLARE_API_TOKEN not set"
    exit 1
fi

if [[ -z "$ACCOUNT_ID" ]]; then
    log_error "CLOUDFLARE_ACCOUNT_ID not set"
    exit 1
fi

if [[ -z "$TUNNEL_ID" ]]; then
    log_error "CLOUDFLARE_TUNNEL_ID not set"
    exit 1
fi

log_success "Configuration validated"
log_info "Zone: ${ZONE_NAME}"
log_info "Tunnel ID: ${TUNNEL_ID}"
log_info "Access Mode: ${ACCESS_MODE}"

if [[ "$DRY_RUN" == "true" ]]; then
    log_warning "DRY RUN MODE - No changes will be made"
fi

# Check required tools
log_info "Checking required tools..."
for tool in kubectl jq curl; do
    if ! command -v "$tool" &> /dev/null; then
        log_error "$tool is required but not installed"
        exit 1
    fi
done
log_success "All required tools found"

# Check kubectl connectivity
if ! kubectl cluster-info &> /dev/null; then
    log_error "kubectl is not connected to a cluster"
    exit 1
fi
log_success "kubectl connected to cluster"

# Step 1: Configure Tunnel Token
if [[ "$SKIP_TUNNEL" == "false" ]]; then
    log_step "Step 1: Configure Cloudflare Tunnel Token"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would configure tunnel token in Kubernetes"
    else
        # Verify the script exists
        if [[ ! -f "${SCRIPT_DIR}/configure-cloudflare-tunnel-token.sh" ]]; then
            log_error "configure-cloudflare-tunnel-token.sh not found in ${SCRIPT_DIR}"
            exit 1
        fi
        
        log_info "Configuring tunnel token..."
        "${SCRIPT_DIR}/configure-cloudflare-tunnel-token.sh" \
            --account-id "$ACCOUNT_ID" \
            --tunnel-id "$TUNNEL_ID" \
            --api-token "$API_TOKEN" || {
            log_error "Failed to configure tunnel token"
            exit 1
        }
        log_success "Tunnel token configured"
    fi
else
    log_info "Skipping tunnel token configuration (--skip-tunnel)"
fi

# Step 2: Create DNS Records
if [[ "$SKIP_DNS" == "false" ]]; then
    log_step "Step 2: Create DNS Records"
    
    DNS_ARGS=(
        --token "$API_TOKEN"
        --tunnel-id "$TUNNEL_ID"
        --zone-name "$ZONE_NAME"
    )
    
    if [[ -n "$ZONE_ID" ]]; then
        DNS_ARGS+=(--zone-id "$ZONE_ID")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        DNS_ARGS+=(--dry-run)
    fi
    
    log_info "Creating DNS records..."
    "${SCRIPT_DIR}/create-cloudflare-dns-records.sh" "${DNS_ARGS[@]}" || {
        log_error "Failed to create DNS records"
        exit 1
    }
    log_success "DNS records configured"
else
    log_info "Skipping DNS record creation (--skip-dns)"
fi

# Step 3: Create Access Applications
if [[ "$SKIP_ACCESS" == "false" ]]; then
    log_step "Step 3: Create Cloudflare Access Applications"
    
    ACCESS_ARGS=(
        --token "$API_TOKEN"
        --account-id "$ACCOUNT_ID"
        --mode "$ACCESS_MODE"
        --allowed-email-domains "$ALLOWED_EMAIL_DOMAINS"
        --force
    )
    
    if [[ "$ACCESS_MODE" == "zone" ]]; then
        ACCESS_ARGS+=(--zone-domain "$ZONE_NAME")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        ACCESS_ARGS+=(--dry-run)
    fi
    
    log_info "Creating Access applications..."
    "${SCRIPT_DIR}/create-cloudflare-access-apps.sh" "${ACCESS_ARGS[@]}" || {
        log_error "Failed to create Access applications"
        exit 1
    }
    log_success "Access applications configured"
else
    log_info "Skipping Access application creation (--skip-access)"
fi

# Step 4: Deploy to Kubernetes
if [[ "$SKIP_DEPLOY" == "false" ]]; then
    log_step "Step 4: Deploy Cloudflare Tunnel to Kubernetes"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy cloudflared to Kubernetes"
    else
        log_info "Applying Kubernetes manifests..."
        kubectl apply -f "${REPO_ROOT}/k8s/cloudflare-tunnel-ingress.yaml" || {
            log_error "Failed to apply Kubernetes manifests"
            exit 1
        }
        log_success "Kubernetes deployment complete"
        
        log_info "Waiting for cloudflared pods to be ready..."
        if kubectl wait --for=condition=ready pod \
            -l app.kubernetes.io/name=cloudflare-tunnel \
            -n cloudflare-tunnel \
            --timeout=120s 2>/dev/null; then
            log_success "Cloudflared pods are ready"
        else
            log_warning "Timeout waiting for pods, check status manually"
        fi
    fi
else
    log_info "Skipping Kubernetes deployment (--skip-deploy)"
fi

# Step 5: Verification
if [[ "$VERIFY" == "true" ]] && [[ "$DRY_RUN" == "false" ]]; then
    log_step "Step 5: Verification"
    
    log_info "Verifying SSL certificates..."
    if "${SCRIPT_DIR}/verify-ssl-certificates.sh"; then
        log_success "SSL certificate verification passed"
    else
        log_warning "SSL certificate verification had warnings (this is normal if services aren't fully deployed yet)"
    fi
    
    log_info "Checking tunnel status..."
    kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=20 || true
else
    log_info "Skipping verification (--no-verify or --dry-run)"
fi

# Summary
log_step "Bootstrap Complete!"

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "DRY RUN completed - no actual changes were made"
    log_info "Run without --dry-run to apply changes"
else
    log_success "Cloudflare infrastructure bootstrap successful!"
    echo ""
    log_info "Summary:"
    [[ "$SKIP_TUNNEL" == "false" ]] && echo "  ✓ Tunnel token configured"
    [[ "$SKIP_DNS" == "false" ]] && echo "  ✓ DNS records created"
    [[ "$SKIP_ACCESS" == "false" ]] && echo "  ✓ Access applications created"
    [[ "$SKIP_DEPLOY" == "false" ]] && echo "  ✓ Kubernetes deployment complete"
    echo ""
    log_info "Next steps:"
    echo "  1. Test access: https://portal.${ZONE_NAME}"
    echo "  2. Check tunnel: kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel"
    echo "  3. Verify DNS: nslookup portal.${ZONE_NAME}"
    echo "  4. Export Access config: ./scripts/export-cloudflare-access-apps.sh"
fi

echo ""
