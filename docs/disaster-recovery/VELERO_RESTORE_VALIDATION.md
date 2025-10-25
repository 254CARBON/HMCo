# Velero Restore Validation Guide

This guide captures the validated disaster recovery workflows for the 254Carbon platform. It complements the primary [DR Runbook](DR_RUNBOOK.md) with concrete commands, automation entry points, and success criteria for Velero restores.

---

## Restore Workflows

- **Automation Script**: `scripts/velero-restore-validate.sh`
  - Fetches the latest completed backup (or uses a supplied backup/schedule)
  - Supports namespace-scoped restores with optional namespace mapping
  - Waits for completion, streams logs, and optionally cleans up scratch namespaces
  - Example:
    ```bash
    ./scripts/velero-restore-validate.sh \
      --schedule daily-backup \
      --namespace data-platform \
      --restore-namespace data-platform-dr \
      --wait --cleanup
    ```

- **Restore Templates** (`k8s/storage/`)
  - `velero-restore-full.yaml` – full cluster replay (requires `BACKUP_NAME`)
  - `velero-restore-namespace.yaml` – namespace restore with mapping options
  - `velero-restore-app.yaml` – selective restore via label selector
  - `velero-restore-test.yaml` – rehearsal restore into scratch namespace
  - Usage pattern:
    ```bash
    export BACKUP_NAME="daily-backup-20251021020000"
    export TARGET_NAMESPACE="data-platform"
    export RESTORE_NAMESPACE="data-platform-dr"
    kubectl create -f <(envsubst < k8s/storage/velero-restore-namespace.yaml)
    ```

- **Backup Configuration**: `k8s/storage/velero-backup-config.yaml`
  - Declares the MinIO `BackupStorageLocation`
  - Schedules:
    - `daily-backup` (02:00 UTC, 30-day retention)
    - `hourly-critical-backup` (hourly, 7-day retention)
    - `weekly-full-backup` (Sunday 03:00 UTC, 90-day retention)

---

## Validation Checklist

1. Run hourly/daily backups – confirm `velero backup get` shows latest entries for each schedule.
2. Execute rehearsal restore:
   - `./scripts/velero-restore-validate.sh --schedule daily-backup --namespace data-platform --restore-namespace data-platform-dr --wait --cleanup`
   - Verify Pods in the scratch namespace reach `Running` state.
3. Review restore logs: `velero restore logs <restore-name>`.
4. Confirm data integrity (ConfigMaps, Secrets, PVC mounts) in restored namespace.
5. Clean up scratch namespace (script `--cleanup` flag handles this automatically).

Record results in the DR runbook after each drill.

---

## Targets & Current Performance

| Scenario                    | RTO Target | RTO (Validated) | RPO Target | Current RPO |
|-----------------------------|-----------:|----------------:|-----------:|-------------|
| Namespace restore           | 10 minutes | **90 seconds** ✅ | 1 hour      | **1 hour** ✅ |
| Full cluster rebuild        | 4 hours    | Not yet tested  | 24 hours    | 24 hours (weekly full) |
| Application-level rollback  | 5 minutes  | < 2 minutes ✅   | 1 hour      | 1 hour ✅ |
| Database restore (pg_dump)  | 15 minutes | Manual (scripted) | 15 minutes | 1 hour (Velero) |

---

## Scheduling & Ownership

- **Monthly (First Sunday)**: Run namespace rehearsal restore via automation script and update `DR_RUNBOOK.md`.
- **Quarterly**: Execute full cluster restore using `velero-restore-full.yaml` template and validate cross-namespace services.
- **After Major Changes**: Trigger manual backup (`velero backup create pre-change-<timestamp> --wait`) and capture logs.

Primary owners: Platform Engineering. Escalation: see contacts in `DR_RUNBOOK.md`.

---

## Supporting Artifacts

- `scripts/deploy-velero-backup.sh` – provisions chart + applies backup config
- `scripts/velero-restore-validate.sh` – restore automation
- `k8s/storage/velero-backup-config.yaml` – storage location + schedules
- `k8s/storage/velero-restore-*.yaml` – restore templates
- `docs/disaster-recovery/DR_RUNBOOK.md` – full DR reference

Keep these files in sync whenever procedures or retention targets change.
