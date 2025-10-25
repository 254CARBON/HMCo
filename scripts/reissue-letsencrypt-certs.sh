#!/bin/bash

# Helper to trigger cert-manager renewals after updating Cloudflare Access policies.

set -euo pipefail

if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl is required in PATH" >&2
    exit 1
fi

CERTS=()
DRY_RUN=false
WAIT_READY=false
WAIT_TIMEOUT=180

usage() {
    cat <<EOF
Usage: $0 --cert <namespace/name> [--cert <namespace/name> ...] [options]

Options:
  --cert NAMESPACE/NAME   Certificate resource to force renew (may be repeated)
  --dry-run               Print planned kubectl commands without executing them
  --wait                  Wait for certificates to reach Ready=true
  --wait-timeout SECONDS  Wait timeout when --wait is enabled (default: ${WAIT_TIMEOUT})
  -h, --help              Show this help text

Example:
  $0 --cert data-platform/spark-history-tls --cert data-platform/graphql-gateway-tls --wait
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cert)
            CERTS+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --wait)
            WAIT_READY=true
            shift
            ;;
        --wait-timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ${#CERTS[@]} -eq 0 ]]; then
    echo "At least one --cert NAMESPACE/NAME argument is required." >&2
    usage
    exit 1
fi

timestamp() {
    date +%s
}

run() {
    if [[ "$DRY_RUN" == "true" ]]; then
        printf "[DRY-RUN]"
        for arg in "$@"; do
            printf " %q" "$arg"
        done
        printf "\n"
    else
        "$@"
    fi
}

for item in "${CERTS[@]}"; do
    if [[ "$item" != */* ]]; then
        echo "Invalid certificate identifier: ${item}. Use namespace/name format." >&2
        exit 1
    fi

    namespace="${item%%/*}"
    name="${item##*/}"

    echo ">> Forcing renewal for certificate ${namespace}/${name}"

    renew_value=$(timestamp)
    run kubectl -n "$namespace" annotate certificate "$name" cert-manager.io/renew-request="$renew_value" --overwrite
    run kubectl -n "$namespace" delete certificaterequest -l cert-manager.io/certificate-name="$name" --ignore-not-found
    run kubectl -n "$namespace" delete orders.acme.cert-manager.io -l cert-manager.io/certificate-name="$name" --ignore-not-found
    run kubectl -n "$namespace" delete challenges.acme.cert-manager.io -l cert-manager.io/certificate-name="$name" --ignore-not-found

    if [[ "$WAIT_READY" == "true" ]]; then
        run kubectl -n "$namespace" wait --for=condition=Ready --timeout="${WAIT_TIMEOUT}s" certificate/"$name"
    fi
done

echo "All requested certificates processed."
