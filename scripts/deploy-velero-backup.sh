#!/usr/bin/env bash

# Deploys the Velero backup stack (Helm chart + schedules) for the 254Carbon platform.
# Requirements:
#   * kubectl with cluster-admin permissions
#   * helm (v3)
#   * Environment variables VELERO_S3_ACCESS_KEY and VELERO_S3_SECRET_KEY populated

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly VELERO_NAMESPACE="${VELERO_NAMESPACE:-velero}"
readonly VELERO_RELEASE_NAME="${VELERO_RELEASE_NAME:-velero}"
readonly SECRET_NAME="${VELERO_CREDENTIALS_SECRET:-minio-backup-credentials}"
readonly VALUES_FILE="${REPO_ROOT}/velero-values.yaml"
readonly CONFIG_FILE="${REPO_ROOT}/k8s/storage/velero-backup-config.yaml"
readonly HELM_REPO_NAME="${VELERO_HELM_REPO_NAME:-vmware-tanzu}"
readonly HELM_REPO_URL="${VELERO_HELM_REPO_URL:-https://vmware-tanzu.github.io/helm-charts}"

INFO="[\033[0;34mINFO\033[0m]"
SUCCESS="[\033[0;32mOK\033[0m]"
WARN="[\033[1;33mWARN\033[0m]"
ERROR="[\033[0;31mERROR\033[0m]"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo -e "${ERROR} Missing required command: $1"
    exit 1
  fi
}

banner() {
  echo "==============================================="
  echo "     Velero Backup Deployment (254Carbon)      "
  echo "==============================================="
}

ensure_namespace() {
  echo -e "${INFO} Ensuring namespace '${VELERO_NAMESPACE}' exists"
  kubectl create namespace "${VELERO_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
}

ensure_credentials_secret() {
  local access_key="${VELERO_S3_ACCESS_KEY:-}"
  local secret_key="${VELERO_S3_SECRET_KEY:-}"

  if [[ -z "${access_key}" || -z "${secret_key}" ]]; then
    cat <<EOF
${ERROR} Missing MinIO/S3 credentials.

Please export the following environment variables and re-run:
  export VELERO_S3_ACCESS_KEY="<minio-access-key>"
  export VELERO_S3_SECRET_KEY="<minio-secret-key>"

Optional overrides:
  export VELERO_CREDENTIALS_SECRET="${SECRET_NAME}"
  export VELERO_NAMESPACE="${VELERO_NAMESPACE}"
EOF
    exit 1
  fi

  echo -e "${INFO} Creating/updating cloud credentials secret '${SECRET_NAME}'"
  kubectl -n "${VELERO_NAMESPACE}" create secret generic "${SECRET_NAME}" \
    --from-literal=cloud="[default]
aws_access_key_id = ${access_key}
aws_secret_access_key = ${secret_key}" \
    --dry-run=client -o yaml | kubectl apply -f -
  echo -e "${SUCCESS} Credentials secret ready"
}

ensure_minio_ready() {
  echo -e "${INFO} Verifying MinIO endpoint availability"
  if ! kubectl get pods -n data-platform -l app=minio >/dev/null 2>&1; then
    echo -e "${WARN} MinIO pods not detected in namespace 'data-platform'"
    echo -e "${WARN} Velero may fail without a reachable S3-compatible endpoint"
  fi
}

deploy_helm_release() {
  require_command helm

  echo -e "${INFO} Adding Helm repository '${HELM_REPO_NAME}'"
  helm repo add "${HELM_REPO_NAME}" "${HELM_REPO_URL}" --force-update >/dev/null
  helm repo update >/dev/null

  echo -e "${INFO} Installing/upgrading Helm release '${VELERO_RELEASE_NAME}'"
  helm upgrade --install "${VELERO_RELEASE_NAME}" "${HELM_REPO_NAME}/velero" \
    --namespace "${VELERO_NAMESPACE}" \
    --create-namespace \
    --values "${VALUES_FILE}" \
    --wait

  echo -e "${SUCCESS} Helm release applied"
  echo -e "${INFO} Waiting for Velero controller rollout"
  kubectl -n "${VELERO_NAMESPACE}" rollout status deploy/velero --timeout=240s

  if kubectl -n "${VELERO_NAMESPACE}" get ds node-agent &>/dev/null; then
    kubectl -n "${VELERO_NAMESPACE}" rollout status ds/node-agent --timeout=240s || \
      echo -e "${WARN} Node agent rollout timed out; check daemonset pods manually"
  fi
}

apply_backup_configuration() {
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo -e "${WARN} Backup configuration file '${CONFIG_FILE}' not found; skipping"
    return
  fi

  echo -e "${INFO} Applying Velero backup configuration"
  kubectl apply -f "${CONFIG_FILE}"
  echo -e "${SUCCESS} Backup storage location and schedules configured"
}

show_status() {
  echo
  echo -e "${INFO} Current Velero components"
  kubectl get pods -n "${VELERO_NAMESPACE}"
  echo
  echo -e "${INFO} Backup storage locations"
  kubectl get backupstoragelocations -n "${VELERO_NAMESPACE}"
  echo
  echo -e "${INFO} Scheduled backups"
  kubectl get schedules -n "${VELERO_NAMESPACE}"
}

main() {
  banner
  require_command kubectl
  ensure_minio_ready
  ensure_namespace
  ensure_credentials_secret
  deploy_helm_release
  apply_backup_configuration
  show_status
  echo
  echo -e "${SUCCESS} Velero backup deployment complete."
  echo "Next steps:"
  echo "  1. Verify the first scheduled backups complete successfully."
  echo "  2. Run 'kubectl apply -f k8s/storage/velero-restore-test.yaml' to rehearse restores."
  echo "  3. Configure monitoring/alerting on Velero metrics."
}

main "$@"
