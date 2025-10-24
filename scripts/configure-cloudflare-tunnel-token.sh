#!/usr/bin/env bash
#
# Fetch a Cloudflare tunnel token via API (when supported) or a provided token
# and sync it into Kubernetes. Requires: curl, kubectl, python3.
#
# Usage:
#   ./configure-cloudflare-tunnel-token.sh \
#     --account-id <ACCOUNT_ID> \
#     --tunnel-id <TUNNEL_ID> \
#     --api-token <CF_API_TOKEN> \
#     [--namespace cloudflare-tunnel] \
#     [--secret-name cloudflare-tunnel-token] \
#     [--secret-key token]
#
# The script never prints the tunnel token. It only reports success/failure
# and restarts the cloudflared deployment so that it picks up the new secret.

set -euo pipefail

ACCOUNT_ID=""
TUNNEL_ID=""
API_TOKEN=""
EXPLICIT_TOKEN=""
NAMESPACE="cloudflare-tunnel"
SECRET_NAME="cloudflare-tunnel-token"
SECRET_KEY="token"

usage() {
  cat <<EOF
Usage: $0 --account-id <ACCOUNT_ID> --tunnel-id <TUNNEL_ID> --api-token <API_TOKEN> [options]

Options:
  --account-id     Cloudflare account ID (aka account tag)
  --tunnel-id      Cloudflare tunnel UUID
  --api-token      Cloudflare API token with tunnel:read scope
  --namespace      Kubernetes namespace for the secret (default: cloudflare-tunnel)
  --secret-name    Kubernetes secret name (default: cloudflare-tunnel-token)
  --secret-key     Secret key storing the token (default: token)
  --tunnel-token   Skip API lookup and use this token directly
  -h, --help       Show this help message

Environment:
  Set CLOUDFLARE_API_TOKEN instead of --api-token to avoid putting secrets in the shell history.

Requirements:
  - curl
  - kubectl (configured for target cluster)
  - python3
EOF
}

abort() {
  echo "Error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account-id)
      ACCOUNT_ID="${2:-}"; shift 2 ;;
    --tunnel-id)
      TUNNEL_ID="${2:-}"; shift 2 ;;
    --api-token)
      API_TOKEN="${2:-}"; shift 2 ;;
    --namespace)
      NAMESPACE="${2:-}"; shift 2 ;;
    --secret-name)
      SECRET_NAME="${2:-}"; shift 2 ;;
    --secret-key)
      SECRET_KEY="${2:-}"; shift 2 ;;
    --tunnel-token)
      EXPLICIT_TOKEN="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$ACCOUNT_ID" ]]; then
  abort "Missing --account-id"
fi
if [[ -z "$TUNNEL_ID" ]]; then
  abort "Missing --tunnel-id"
fi

if [[ -z "$EXPLICIT_TOKEN" ]]; then
  if [[ -z "${API_TOKEN:-}" ]]; then
    if [[ -n "${CLOUDFLARE_API_TOKEN:-}" ]]; then
      API_TOKEN="$CLOUDFLARE_API_TOKEN"
    else
      abort "Either provide --tunnel-token or --api-token (or CLOUDFLARE_API_TOKEN env var)"
    fi
  fi
fi

if [[ -z "$EXPLICIT_TOKEN" ]] && ! command -v curl >/dev/null 2>&1; then
  abort "curl is required when fetching token via API"
fi

if ! command -v kubectl >/dev/null 2>&1; then
  abort "kubectl is required and must target the correct cluster"
fi

if ! command -v python3 >/dev/null 2>&1; then
  abort "python3 is required for JSON parsing"
fi

if [[ -n "$EXPLICIT_TOKEN" ]]; then
  TUNNEL_TOKEN="$EXPLICIT_TOKEN"
else
  echo "Requesting tunnel token from Cloudflare API..."
  API_RESPONSE=$(curl -sS \
    -X POST "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/cfd_tunnels/${TUNNEL_ID}/token" \
    -H "Authorization: Bearer ${API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{}') || abort "Failed to contact Cloudflare API"

  API_SUCCESS=$(python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get("success", False))' <<<"$API_RESPONSE" 2>/dev/null || echo "False")
  if [[ "$API_SUCCESS" != "True" ]]; then
    ERRORS=$(python3 -c 'import json,sys; data=json.load(sys.stdin); errs=data.get("errors") or []; print("; ".join(str(e.get("message", e)) for e in errs) or "unknown error")' <<<"$API_RESPONSE" 2>/dev/null || echo "unknown error")
    abort "Cloudflare API error: $ERRORS"
  fi

  TUNNEL_TOKEN=$(python3 -c 'import json,sys; data=json.load(sys.stdin); print(data["result"]["token"])' <<<"$API_RESPONSE" 2>/dev/null) || abort "Failed to parse token from API response"

  if [[ -z "$TUNNEL_TOKEN" ]]; then
    abort "Token was empty in Cloudflare API response"
  fi
fi

echo "Syncing tunnel token to secret ${SECRET_NAME} in namespace ${NAMESPACE}..."

kubectl get secret "${SECRET_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1 && \
  kubectl delete secret "${SECRET_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1 || true

kubectl create secret generic "${SECRET_NAME}" \
  -n "${NAMESPACE}" \
  --from-literal="${SECRET_KEY}=${TUNNEL_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

echo "Secret updated. Restarting cloudflared deployment..."
if kubectl get deployment cloudflared -n "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl rollout restart deployment/cloudflared -n "${NAMESPACE}" >/dev/null
  kubectl rollout status deployment/cloudflared -n "${NAMESPACE}" --timeout=120s >/dev/null
else
  echo "Warning: deployment cloudflared not found in namespace ${NAMESPACE}" >&2
fi

echo "Cloudflare tunnel token configured successfully."
echo "Verify connectivity with: kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/name=cloudflare-tunnel"
