#!/usr/bin/env bash
#
# Rotate Strimzi Kafka client certificates and distribute them to target namespaces.
# Usage:
#   scripts/rotate-kafka-client-secrets.sh [--skip-rotate] [--dry-run] [--wait N]
#
# Requirements:
#   - kubectl (configured for the target cluster)
#   - python3 (for JSON reshaping)
#
# The script performs the following actions for each mapped KafkaUser:
#   1. (Optional) Requests Strimzi to issue a fresh client certificate by forcing renewal.
#   2. Waits for the backing Secret in the Kafka namespace to be recreated/updated.
#   3. Re-applies the secret payload into the consumer namespace with a predictable name.
#
# Secret mappings are defined in the CLIENTS array below. Adjust as needed for new consumers.

set -euo pipefail

KAFKA_NAMESPACE=${KAFKA_NAMESPACE:-kafka}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-180}
ROTATE_CERTS=true
DRY_RUN=false

usage() {
  cat <<'EOF'
Rotate Kafka client certificates issued by Strimzi and sync them to application namespaces.

Options:
  --skip-rotate     Do not trigger certificate renewal, only copy existing secrets.
  --dry-run         Print the rendered secrets without applying them.
  --wait <seconds>  Override wait timeout (default 180s). Can also set WAIT_TIMEOUT env var.
  -h, --help        Show this help message.

Environment variables:
  KAFKA_NAMESPACE   Namespace where the Strimzi Kafka cluster lives (default: kafka).
  WAIT_TIMEOUT      Max seconds to wait for secret rotation (default: 180).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-rotate)
      ROTATE_CERTS=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --wait)
      if [[ $# -lt 2 ]]; then
        echo "error: --wait requires a value" >&2
        exit 1
      fi
      WAIT_TIMEOUT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

command -v kubectl >/dev/null 2>&1 || {
  echo "error: kubectl not found in PATH" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || {
  echo "error: python3 is required" >&2
  exit 1
}

# KafkaUser -> consumer namespace/secret mappings
CLIENTS=(
  "datahub-app:data-platform:kafka-datahub-tls"
  "schema-registry:data-platform:kafka-schema-registry-tls"
  "platform-apps:data-platform:kafka-platform-apps-tls"
)

log() {
  local level=$1; shift
  printf '[%s] %s\n' "$level" "$*" >&2
}

force_rotate() {
  local user=$1
  if [[ "$ROTATE_CERTS" != "true" ]]; then
    return 0
  fi

  # Record current resource version to detect updates.
  local current_version=""
  current_version=$(kubectl get secret "$user" -n "$KAFKA_NAMESPACE" -o jsonpath='{.metadata.resourceVersion}' 2>/dev/null || echo "")

  kubectl annotate kafkauser "$user" -n "$KAFKA_NAMESPACE" \
    "strimzi.io/force-renew=$(date +%s)" --overwrite >/dev/null
  log INFO "Requested certificate rotation for KafkaUser '$user'"
  echo "$current_version"
}

wait_for_secret_update() {
  local user=$1
  local prev_version=$2
  local deadline=$((SECONDS + WAIT_TIMEOUT))

  while (( SECONDS <= deadline )); do
    if secret_json=$(kubectl get secret "$user" -n "$KAFKA_NAMESPACE" -o json 2>/dev/null); then
      local new_version
      new_version=$(printf '%s' "$secret_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"].get("resourceVersion",""))')
      if [[ -z "$prev_version" || "$new_version" != "$prev_version" ]]; then
        printf '%s' "$secret_json"
        return 0
      fi
    fi
    sleep 2
  done

  log ERROR "Timed out waiting for secret '$user' to refresh in namespace '$KAFKA_NAMESPACE'"
  return 1
}

render_secret() {
  local secret_json=$1
  local target_ns=$2
  local target_name=$3
  local source_user=$4

  printf '%s' "$secret_json" | python3 - "$target_ns" "$target_name" "$source_user" <<'PYCODE'
import json
import sys

target_ns, target_name, source_user = sys.argv[1:4]
doc = json.load(sys.stdin)

# Preserve data, but overwrite metadata with clean values.
labels = doc.get("metadata", {}).get("labels", {})
annotations = doc.get("metadata", {}).get("annotations", {})

managed_labels = {
    "app.kubernetes.io/managed-by": "kafka-client-sync",
    "kafka.strimzi.io/user": source_user,
}
managed_labels.update({k: v for k, v in labels.items() if not k.startswith("strimzi.io/")})

managed_annotations = {
    "kafka.strimzi.io/sourceSecret": source_user,
}
managed_annotations.update({k: v for k, v in annotations.items() if k.startswith("strimzi.io/ca-")})

doc["metadata"] = {
    "name": target_name,
    "namespace": target_ns,
    "labels": managed_labels,
    "annotations": managed_annotations,
}
doc.pop("status", None)
doc["type"] = doc.get("type", "Opaque")

print(json.dumps(doc, indent=2))
PYCODE
}

apply_secret() {
  local rendered=$1
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '%s\n' "$rendered"
  else
    printf '%s\n' "$rendered" | kubectl apply -f -
  fi
}

process_client() {
  local entry=$1
  IFS=':' read -r user target_ns target_name <<<"$entry"

  log INFO "Processing KafkaUser '$user' -> secret '${target_ns}/${target_name}'"
  prev_version=$(force_rotate "$user")
  secret_payload=$(wait_for_secret_update "$user" "$prev_version")
  rendered=$(render_secret "$secret_payload" "$target_ns" "$target_name" "$user")
  apply_secret "$rendered"
  log INFO "Secret synced to '${target_ns}/${target_name}'"
}

for client in "${CLIENTS[@]}"; do
  process_client "$client"
done

log INFO "Completed Kafka client secret synchronization."
