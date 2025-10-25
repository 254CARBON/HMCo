#!/usr/bin/env bash
#
# Helper script to validate Velero restore workflows.
# It can replay the latest completed backup (or a specific backup) into either
# the original namespace or an alternate namespace for rehearsal purposes.
# See documentation: docs/disaster-recovery/VELERO_RESTORE_VALIDATION.md
#
# Examples:
#   # Dry-run: show the command that would execute
#   ./scripts/velero-restore-validate.sh --schedule daily-backup --namespace data-platform --dry-run
#
#   # Restore latest daily backup into a scratch namespace and wait for completion
#   ./scripts/velero-restore-validate.sh \
#     --schedule daily-backup \
#     --namespace data-platform \
#     --restore-namespace data-platform-dr \
#     --wait
#
#   # Restore a specific backup and include persistent volumes
#   ./scripts/velero-restore-validate.sh --backup daily-backup-20251021020000 --restore-pvs
#

set -euo pipefail

VELERO_CLI="${VELERO_CLI:-velero}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"

RESTORE_NAME=""
BACKUP_NAME=""
SCHEDULE_NAME=""
TARGET_NAMESPACE=""
RESTORE_NAMESPACE=""
LABEL_SELECTOR=""
RESTORE_PVS=false
WAIT_FOR_COMPLETION=false
DRY_RUN=false
CLEANUP_AFTER=false
USE_SCHEDULE=false
VERIFY_TRINO=false

usage() {
  cat <<'EOF'
Usage: scripts/velero-restore-validate.sh [options]

Options:
  -b, --backup NAME               Restore from a specific Velero backup.
  -s, --schedule NAME             Use the latest successful backup from the schedule.
  -n, --namespace NAME            Only include the given namespace in the restore.
  -r, --restore-namespace NAME    Restore into an alternate namespace (requires --namespace).
  -l, --selector KEY=VALUE        Restore only resources matching the given label selector.
      --restore-pvs               Restore persistent volumes (default: disabled for safety).
      --no-restore-pvs            Do not restore persistent volumes (default).
      --wait                      Block until the restore completes and print status.
      --dry-run                   Print the Velero command without executing it.
      --name NAME                 Explicit restore object name (otherwise generated).
      --cleanup                   Delete the restore object (and scratch namespace when applicable) after completion.
      --verify-trino              After restore, verify Trino coordinator/worker are healthy.
  -h, --help                      Show this usage information.

Documentation
  docs/disaster-recovery/VELERO_RESTORE_VALIDATION.md
EOF
}

fail() {
  echo "ERROR: $1" >&2
  exit "${2:-1}"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command '$1' not found on PATH."
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--backup)
      BACKUP_NAME="${2:-}"; shift 2 ;;
    -s|--schedule)
      SCHEDULE_NAME="${2:-}"; shift 2 ;;
    -n|--namespace)
      TARGET_NAMESPACE="${2:-}"; shift 2 ;;
    -r|--restore-namespace)
      RESTORE_NAMESPACE="${2:-}"; shift 2 ;;
    -l|--selector)
      LABEL_SELECTOR="${2:-}"; shift 2 ;;
    --restore-pvs)
      RESTORE_PVS=true; shift ;;
    --no-restore-pvs)
      RESTORE_PVS=false; shift ;;
    --wait)
      WAIT_FOR_COMPLETION=true; shift ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    --name)
      RESTORE_NAME="${2:-}"; shift 2 ;;
    --cleanup)
      CLEANUP_AFTER=true; shift ;;
    --verify-trino)
      VERIFY_TRINO=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

require_command "${VELERO_CLI}"
require_command "${KUBECTL_BIN}"
require_command jq
require_command grep

if [[ -n "${BACKUP_NAME}" ]]; then
  echo "Using explicit backup '${BACKUP_NAME}'."
elif [[ -n "${SCHEDULE_NAME}" ]]; then
  USE_SCHEDULE=true
  BACKUP_NAME="$("${VELERO_CLI}" backup get --output json | jq -r --arg schedule "${SCHEDULE_NAME}" '
    .items
    | map(select(.status.phase == "Completed"))
    | map(select(.metadata.labels["velero.io/schedule-name"] == $schedule))
    | sort_by(.status.completionTimestamp)
    | last
    | .metadata.name // empty
  ')"
  [[ -n "${BACKUP_NAME}" ]] || fail "No completed backups found for schedule '${SCHEDULE_NAME}'."
  echo "Latest completed backup for schedule '${SCHEDULE_NAME}': ${BACKUP_NAME}"
else
  echo "No backup provided, defaulting to most recent completed backup."
  BACKUP_NAME="$("${VELERO_CLI}" backup get --output json | jq -r '
    .items
    | map(select(.status.phase == "Completed"))
    | sort_by(.status.completionTimestamp)
    | last
    | .metadata.name // empty
  ')"
  [[ -n "${BACKUP_NAME}" ]] || fail "Unable to determine a completed backup. Supply --backup or --schedule."
  echo "Latest completed backup: ${BACKUP_NAME}"
fi

if [[ -n "${RESTORE_NAMESPACE}" && -z "${TARGET_NAMESPACE}" ]]; then
  fail "--restore-namespace requires --namespace to be specified."
fi

# Default restore namespace to target when not provided (for env automation).
if [[ -n "${TARGET_NAMESPACE}" && -z "${RESTORE_NAMESPACE}" ]]; then
  RESTORE_NAMESPACE="${TARGET_NAMESPACE}"
fi

# Generate a friendly name if none supplied.
if [[ -z "${RESTORE_NAME}" ]]; then
  timestamp="$(date +%Y%m%d-%H%M%S)"
  if [[ -n "${TARGET_NAMESPACE}" ]]; then
    RESTORE_NAME="restore-${TARGET_NAMESPACE}-${timestamp}"
  elif [[ -n "${SCHEDULE_NAME}" ]]; then
    RESTORE_NAME="restore-${SCHEDULE_NAME}-${timestamp}"
  else
    RESTORE_NAME="restore-${timestamp}"
  fi
fi

cmd=("${VELERO_CLI}" restore create "${RESTORE_NAME}")

if [[ "${USE_SCHEDULE}" == true ]]; then
  cmd+=(--from-schedule "${SCHEDULE_NAME}")
else
  cmd+=(--from-backup "${BACKUP_NAME}")
fi

if [[ -n "${TARGET_NAMESPACE}" ]]; then
  cmd+=(--include-namespaces "${TARGET_NAMESPACE}")
fi

if [[ -n "${RESTORE_NAMESPACE}" && "${RESTORE_NAMESPACE}" != "${TARGET_NAMESPACE}" ]]; then
  cmd+=(--namespace-mappings "${TARGET_NAMESPACE}:${RESTORE_NAMESPACE}")
fi

if [[ -n "${LABEL_SELECTOR}" ]]; then
  cmd+=(--selector "${LABEL_SELECTOR}")
fi

if [[ "${RESTORE_PVS}" == true ]]; then
  cmd+=(--restore-volumes=true)
else
  cmd+=(--restore-volumes=false)
fi
cmd+=(--existing-resource-policy update)

if [[ "${WAIT_FOR_COMPLETION}" == true ]]; then
  cmd+=(--wait)
fi

if [[ "${DRY_RUN}" == true ]]; then
  printf 'Dry run:'
  for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  exit 0
fi

echo "Submitting Velero restore '${RESTORE_NAME}'..."
restore_output="$("${cmd[@]}" 2>&1)" || {
  echo "${restore_output}" >&2
  exit 1
}

echo "${restore_output}"

# Attempt to extract the effective restore name from the output in case Velero renamed it.
if [[ "${restore_output}" =~ \"([^\"]+)\" ]]; then
  RESTORE_NAME="${BASH_REMATCH[1]}"
fi

echo "Restore object: ${RESTORE_NAME}"

if [[ "${WAIT_FOR_COMPLETION}" == true ]]; then
  echo "Waiting for restore to reach a terminal phase..."
  phase=""
  while true; do
    phase="$("${VELERO_CLI}" restore get "${RESTORE_NAME}" -o json | jq -r '.status.phase // "Unknown"')"
    echo "  Phase: ${phase}"
    case "${phase}" in
      Completed|PartiallyFailed|Failed)
        break ;;
    esac
    sleep 5
  done
  echo
  echo "Restore details:"
  "${VELERO_CLI}" restore describe "${RESTORE_NAME}" --details || true
  echo
  echo "Restore logs (if available):"
  "${VELERO_CLI}" restore logs "${RESTORE_NAME}" || true

  # Verify namespace status if target was provided.
  verify_namespace="${RESTORE_NAMESPACE:-${TARGET_NAMESPACE}}"
  if [[ -n "${verify_namespace}" ]]; then
    echo
    echo "Pod status in namespace '${verify_namespace}':"
    "${KUBECTL_BIN}" get pods -n "${verify_namespace}" || true
  fi
fi

# Optional Trino verification
if [[ "${VERIFY_TRINO}" == true ]]; then
  ns="${RESTORE_NAMESPACE:-${TARGET_NAMESPACE}}"
  # Default to data-platform if not specified
  if [[ -z "${ns}" ]]; then
    ns="data-platform"
  fi
  echo
  echo "Verifying Trino health in namespace '${ns}'..."
  set +e
  "${KUBECTL_BIN}" -n "${ns}" rollout status deploy/trino-coordinator --timeout=180s
  rc1=$?
  "${KUBECTL_BIN}" -n "${ns}" rollout status deploy/trino-worker --timeout=180s
  rc2=$?
  if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
    echo "Trino deployments not ready" >&2
    exit 1
  fi
  # Check coordinator /v1/info and logs
  "${KUBECTL_BIN}" -n "${ns}" exec deploy/trino-coordinator -- sh -lc 'curl -sf http://localhost:8080/v1/info' >/dev/null 2>&1
  rc3=$?
  coord_logs="$(${KUBECTL_BIN} -n "${ns}" logs deploy/trino-coordinator --tail=400 2>/dev/null)"
  echo "Recent coordinator logs:"
  echo "${coord_logs}" | tail -n 40 || true
  if echo "${coord_logs}" | grep -qE 'Configuration errors:|ApplicationConfigurationException'; then
    echo "Detected configuration errors in Trino coordinator logs" >&2
    exit 1
  fi
  if [[ $rc3 -ne 0 ]]; then
    echo "Failed to query Trino coordinator info endpoint" >&2
    exit 1
  fi
  echo "Trino verification succeeded."
  set -e
fi

if [[ "${CLEANUP_AFTER}" == true ]]; then
  echo
  echo "Cleaning up restore object '${RESTORE_NAME}'..."
  "${VELERO_CLI}" restore delete "${RESTORE_NAME}" --confirm || true
  if [[ -n "${RESTORE_NAMESPACE}" && "${RESTORE_NAMESPACE}" != "${TARGET_NAMESPACE}" ]]; then
    echo "Deleting scratch namespace '${RESTORE_NAMESPACE}'..."
    "${KUBECTL_BIN}" delete namespace "${RESTORE_NAMESPACE}" --ignore-not-found --wait=false || true
  fi
fi

echo "Velero restore validation workflow completed."
