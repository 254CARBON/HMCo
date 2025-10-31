# Phase 5: Backup & Disaster Recovery Implementation Guide

## Overview

This guide implements comprehensive backup and disaster recovery capabilities for the 254Carbon platform using Velero. The solution provides automated backups, point-in-time recovery, and disaster recovery procedures.

**Status**: ✅ Ready for Implementation
**Timeline**: 2-3 days
**Dependencies**: MinIO deployed and accessible

## Architecture

```
┌─────────────────────────────────────────────────┐
│            254Carbon Platform                   │
├─────────────────────────────────────────────────┤
│  ├─ Daily Backups (2 AM)                       │
│  ├─ Critical Backups (Every 6 hours)           │
│  ├─ PostgreSQL Backups (3 AM)                  │
│  └─ Manual Backups (On-demand)                 │
├─────────────────────────────────────────────────┤
│            Velero Backup Engine                 │
├─────────────────────────────────────────────────┤
│  ├─ MinIO S3-Compatible Storage                │
│  ├─ Backup Storage Location: 254carbon-backups │
│  └─ Retention Policies                         │
├─────────────────────────────────────────────────┤
│  ├─ Kubernetes Resources                       │
│  ├─ Persistent Volumes                         │
│  ├─ Configuration Data                         │
│  └─ Application State                          │
└─────────────────────────────────────────────────┘
```

## Backup Strategy

### 1. Daily Backups (2 AM)
- **Scope**: All critical namespaces including:
  - `data-platform` - ClickHouse PVs (data + logs), portal services, Trino, Superset
  - `monitoring` - Prometheus, Grafana, Loki metrics and dashboards
  - `registry` - Container registry data
  - `cert-manager` - TLS certificates
  - `cloudflare-tunnel` - Tunnel configuration
  - `vault-prod` - Vault Raft storage and secrets
  - `ml-platform` - MLflow experiment tracking and model registry
  - `kubeflow` - Kubeflow pipelines and workflows
- **Retention**: 30 days
- **Frequency**: Daily at 2:00 AM
- **Volume Backup**: Enabled for all PVs (including ClickHouse data/logs, Vault Raft storage)

### 2. Critical Backups (Hourly)
- **Scope**: Most critical namespaces:
  - `data-platform` - ClickHouse PVs, portal services
  - `monitoring` - Real-time metrics
  - `vault-prod` - Vault storage and secrets
  - `ml-platform` - MLflow tracking
- **Retention**: 7 days
- **Frequency**: Top of every hour
- **Volume Backup**: Enabled for all PVs

### 3. Weekly Full Backups (Sunday 03:00)
- **Scope**: Entire platform (all namespaces except Kubernetes system)
- **Retention**: 90 days
- **Frequency**: Weekly on Sunday at 03:00 UTC

### 4. PostgreSQL Backups (3 AM)
- **Scope**: Database-only backups (via DolphinScheduler workflow)
- **Retention**: 7 days
- **Frequency**: Daily at 3:00 AM

### 5. Manual Backups
- **Scope**: Custom selection
- **Retention**: Configurable
- **Trigger**: Manual execution

## Backup Coverage Documentation

### Complete Resource Coverage

This section documents all resources included in backups per the backup configuration in `k8s/storage/velero-backup-config.yaml`.

#### Data Platform Namespace (`data-platform`)
**Included Resources:**
- ✅ ClickHouse StatefulSet and PersistentVolumeClaims:
  - `clickhouse-data` PVC - Database storage
  - `clickhouse-logs` PVC - Log files
- ✅ Portal services (Deployment, ConfigMaps, Secrets)
- ✅ Trino coordinator and workers (Deployments, ConfigMaps)
- ✅ Superset (Deployment, PostgreSQL PVC, ConfigMaps)
- ✅ DolphinScheduler (StatefulSet, PostgreSQL PVC)
- ✅ MinIO (StatefulSet, PVCs for backup storage)
- ✅ Data Lake components (LakeFS, Iceberg catalogs)
- ✅ All Services, Ingresses, and NetworkPolicies

#### ML Platform Namespace (`ml-platform`)
**Included Resources:**
- ✅ MLflow server (Deployment, PostgreSQL PVC)
- ✅ Experiment tracking data and model registry
- ✅ MLflow artifacts PVC
- ✅ Model serving infrastructure (if deployed)
- ✅ All Services and ConfigMaps

#### Kubeflow Namespace (`kubeflow`)
**Included Resources:**
- ✅ Kubeflow pipelines (Deployments, MySQL PVC)
- ✅ Pipeline definitions and workflow metadata
- ✅ Notebook servers (if persistent)
- ✅ All Services, ConfigMaps, and Secrets

#### Vault Production Namespace (`vault-prod`)
**Included Resources:**
- ✅ Vault StatefulSet
- ✅ Vault Raft storage PVCs (all 3 replicas in HA mode)
- ✅ Vault configuration (ConfigMaps)
- ✅ Vault unseal keys and root token (Kubernetes Secrets)
- ✅ External Secrets Operator deployments
- ✅ ClusterSecretStore definitions

#### Monitoring Namespace (`monitoring`)
**Included Resources:**
- ✅ Prometheus server (StatefulSet, PVC for TSDB)
- ✅ Grafana (Deployment, dashboards PVC)
- ✅ Loki (StatefulSet, logs PVC)
- ✅ AlertManager (StatefulSet, PVC)
- ✅ ServiceMonitor and PrometheusRule CRDs
- ✅ All dashboards and alert configurations

#### Registry Namespace (`registry`)
**Included Resources:**
- ✅ Container registry (Deployment or StatefulSet)
- ✅ Registry storage PVC (all container images)
- ✅ Registry configuration and credentials

#### Certificate Manager Namespace (`cert-manager`)
**Included Resources:**
- ✅ Certificate CRDs and resources
- ✅ Issuer and ClusterIssuer definitions
- ✅ TLS certificates and private keys
- ✅ ACME account keys

#### Cloudflare Tunnel Namespace (`cloudflare-tunnel`)
**Included Resources:**
- ✅ Cloudflare tunnel daemon (Deployment)
- ✅ Tunnel credentials and configuration
- ✅ Ingress routing rules

### Volume Snapshot Settings
- **snapshotVolumes**: `true` - Uses native volume snapshots when available
- **defaultVolumesToFsBackup**: `true` - Falls back to file-system backup (restic/kopia) for volumes without snapshot support
- All PersistentVolumes are backed up including:
  - ClickHouse data and logs volumes
  - Vault Raft storage volumes
  - Database PVCs (Postgres, MySQL for various services)
  - Registry storage volumes
  - Monitoring data volumes (Prometheus TSDB, Grafana, Loki)

### Resources Explicitly Excluded
- Events and ephemeral resources (`events`, `events.events.k8s.io`)
- System namespaces (`kube-system`, `kube-public`, `kube-node-lease`)
- Temporary pods and jobs

### Verification Commands

#### Verify Backup Includes All Expected Namespaces
```bash
# Get the latest backup
BACKUP_NAME=$(velero backup get --selector velero.io/schedule-name=daily-backup \
  --output json | jq -r '.items | map(select(.status.phase=="Completed")) | sort_by(.status.completionTimestamp) | last | .metadata.name')

# Describe backup and verify included namespaces
velero backup describe ${BACKUP_NAME}

# Check included namespaces in backup
velero backup describe ${BACKUP_NAME} --details | grep -A 20 "Namespaces:"
```

#### Verify ClickHouse PVs Are Included
```bash
# Check PVCs in data-platform namespace
kubectl get pvc -n data-platform

# Verify backup includes PVs
velero backup describe ${BACKUP_NAME} --details | grep -A 50 "Persistent Volumes:"

# List volume snapshots
velero backup describe ${BACKUP_NAME} --volume-details
```

#### Verify Vault Storage Is Included
```bash
# Check Vault PVCs
kubectl get pvc -n vault-prod

# Verify in backup
velero backup describe ${BACKUP_NAME} --details | grep -B 5 -A 5 "vault-prod"
```

#### Dry-Run Backup Test
```bash
# Create a test backup to verify configuration without waiting for schedule
velero backup create test-coverage-$(date +%Y%m%d-%H%M%S) \
  --include-namespaces data-platform,ml-platform,vault-prod,monitoring,kubeflow,registry,cert-manager,cloudflare-tunnel \
  --snapshot-volumes \
  --default-volumes-to-fs-backup \
  --ttl 24h

# Monitor the backup
velero backup get
velero backup describe test-coverage-<timestamp> --details
```

## Deployment

### Prerequisites
- ✅ MinIO deployed and accessible
- ✅ Velero namespace created
- ✅ RBAC permissions configured

### Step 1: Deploy Velero
```bash
# Provide MinIO credentials and deploy Velero
export VELERO_S3_ACCESS_KEY="<minio-access-key>"
export VELERO_S3_SECRET_KEY="<minio-secret-key>"
./scripts/deploy-velero-backup.sh
```
The script provisions the `velero-minio-credentials` secret, installs the Helm chart, and applies `k8s/storage/velero-backup-config.yaml` (storage location + schedules).

### Step 2: Verify Deployment
```bash
# Check Velero pods
kubectl get pods -n velero

# Check backup schedules
kubectl get schedules -n velero

# Check storage location
kubectl get backupstoragelocation -n velero
```

### Step 3: Test Backup Creation
```bash
# Kick off an on-demand backup and wait for completion
kubectl -n velero exec deploy/velero -- \
  ./velero backup create smoke-backup-$(date +%Y%m%d-%H%M%S) \
  --include-namespaces data-platform \
  --ttl 24h0m0s \
  --wait
```

## Recovery Procedures

### 1. Full Cluster Recovery
```bash
# Identify the latest weekly full backup
export BACKUP_NAME=$(velero backup get \
  --selector velero.io/schedule-name=weekly-full-backup \
  --output json | jq -r '.items | map(select(.status.phase=="Completed")) | sort_by(.status.completionTimestamp) | last | .metadata.name')

# Create the restore (server generates name via generateName)
kubectl create -f <(envsubst < k8s/storage/velero-restore-full.yaml)
```

### 2. Namespace-Specific Recovery
```bash
export TARGET_NAMESPACE="data-platform"
export RESTORE_NAMESPACE="data-platform"   # or "data-platform-dr" for rehearsal
export BACKUP_NAME=$(velero backup get \
  --selector velero.io/schedule-name=daily-backup \
  --output json | jq -r '.items | map(select(.status.phase=="Completed")) | sort_by(.status.completionTimestamp) | last | .metadata.name')

kubectl create -f <(envsubst < k8s/storage/velero-restore-namespace.yaml)
```

### 3. Application Recovery
```bash
export TARGET_NAMESPACE="data-platform"
export LABEL_KEY="app"
export LABEL_VALUE="datahub-gms"
export BACKUP_NAME=daily-backup-$(date +%Y%m%d020000)

kubectl create -f <(envsubst < k8s/storage/velero-restore-app.yaml)
```

### 4. DR Drill Recovery (Recommended for Testing)
```bash
# Use the dedicated DR drill template for complete recovery testing
export BACKUP_NAME=$(velero backup get \
  --selector velero.io/schedule-name=daily-backup \
  --output json | jq -r '.items | map(select(.status.phase=="Completed")) | sort_by(.status.completionTimestamp) | last | .metadata.name')

kubectl create -f <(envsubst < k8s/storage/velero-restore-dr-drill.yaml)
```

### 5. Automate Restore Validation
```bash
# Restore latest completed backup into a scratch namespace and wait for completion
./scripts/velero-restore-validate.sh \
  --schedule daily-backup \
  --namespace data-platform \
  --restore-namespace data-platform-dr \
  --wait --cleanup
```

## Monitoring & Alerting

### Backup Metrics
- Success/failure rates
- Backup duration
- Storage utilization
- Recovery time

### Alert Rules
```yaml
# Backup failure alerts
- alert: VeleroBackupFailed
  expr: velero_backup_failure_total > 0
  for: 5m
  labels:
    severity: critical

# Backup duration alerts
- alert: VeleroBackupTooSlow
  expr: velero_backup_duration_seconds > 3600
  for: 10m
  labels:
    severity: warning
```

## Disaster Recovery Plan

### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: < 1 hour
- **Recovery Point Objective (RPO)**: < 15 minutes (critical data)

### DR Scenarios
1. **Single Node Failure**: Automatic pod rescheduling
2. **Data Corruption**: Point-in-time recovery
3. **Full Cluster Loss**: Complete restoration from backups

### DR Testing
- Monthly DR drills (see Restore Drill section below)
- Quarterly full recovery tests
- Annual comprehensive DR exercise

### Restore Drill (Non-Production) {#restore-drill}

This section provides step-by-step instructions for conducting disaster recovery drills in a non-production environment.

#### Prerequisites
- Empty Kubernetes cluster available for testing
- Velero CLI installed (`velero` command available)
- `kubectl` configured with cluster access
- `jq` installed for JSON parsing
- MinIO/S3 backup storage accessible from the test cluster

#### Step 1: Install Velero in the Empty Cluster
```bash
# Record start time
echo "Restore drill started: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"

# Set MinIO credentials
export VELERO_S3_ACCESS_KEY="<minio-access-key>"
export VELERO_S3_SECRET_KEY="<minio-secret-key>"

# Deploy Velero (uses existing backup location)
./scripts/deploy-velero-backup.sh

# Verify Velero is ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=velero -n velero --timeout=300s
echo "Velero deployed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 2: Identify the Latest Backup
```bash
# List available backups
velero backup get

# Get the latest successful daily backup
export BACKUP_NAME=$(velero backup get \
  --selector velero.io/schedule-name=daily-backup \
  --output json | jq -r '.items | map(select(.status.phase=="Completed")) | sort_by(.status.completionTimestamp) | last | .metadata.name')

echo "Selected backup: ${BACKUP_NAME}"
echo "Backup identified: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"

# Verify backup details
velero backup describe ${BACKUP_NAME}
```

#### Step 3: Restore Data Platform (ClickHouse, Portal Services)
```bash
# Restore data-platform namespace with all PVs
export TARGET_NAMESPACE="data-platform"
export RESTORE_NAMESPACE="data-platform"

kubectl create -f <(envsubst < k8s/storage/velero-restore-namespace.yaml)

# Monitor restore progress
RESTORE_NAME="restore-${TARGET_NAMESPACE}-$(date +%Y%m%d-%H%M%S)"
velero restore describe ${RESTORE_NAME}

# Wait for restore completion (alternatively add --wait to restore create command)
while true; do
  PHASE=$(velero restore get ${RESTORE_NAME} -o json | jq -r '.status.phase // "Unknown"')
  echo "Restore phase: ${PHASE} - $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  [[ "${PHASE}" == "Completed" || "${PHASE}" == "PartiallyFailed" || "${PHASE}" == "Failed" ]] && break
  sleep 10
done

echo "Data platform restore completed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 4: Restore ML Platform (MLflow)
```bash
# Restore ml-platform namespace
export TARGET_NAMESPACE="ml-platform"
export RESTORE_NAMESPACE="ml-platform"

kubectl create -f <(envsubst < k8s/storage/velero-restore-namespace.yaml)

echo "ML platform restore completed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 5: Restore Vault and Supporting Services
```bash
# Restore vault-prod namespace
export TARGET_NAMESPACE="vault-prod"
export RESTORE_NAMESPACE="vault-prod"
kubectl create -f <(envsubst < k8s/storage/velero-restore-namespace.yaml)

# Restore monitoring namespace
export TARGET_NAMESPACE="monitoring"
export RESTORE_NAMESPACE="monitoring"
kubectl create -f <(envsubst < k8s/storage/velero-restore-namespace.yaml)

echo "Supporting services restore completed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 6: Verify Pod Health
```bash
# Check all pods in restored namespaces
kubectl get pods -n data-platform
kubectl get pods -n ml-platform
kubectl get pods -n vault-prod
kubectl get pods -n monitoring

# Wait for critical pods to be ready
kubectl wait --for=condition=ready pod -l app=clickhouse -n data-platform --timeout=600s
kubectl wait --for=condition=ready pod -l app=mlflow -n ml-platform --timeout=300s

echo "Pod health verified: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 7: Verify ClickHouse Data Integrity
```bash
# Connect to ClickHouse and verify databases
kubectl exec -n data-platform -it deploy/clickhouse -- clickhouse-client --query "SHOW DATABASES"

# Check table counts
kubectl exec -n data-platform -it deploy/clickhouse -- clickhouse-client --query "SELECT database, name, total_rows FROM system.tables WHERE database NOT IN ('system', 'INFORMATION_SCHEMA', 'information_schema')"

# Verify PVs are mounted and contain data
kubectl exec -n data-platform -it deploy/clickhouse -- df -h /var/lib/clickhouse
kubectl exec -n data-platform -it deploy/clickhouse -- du -sh /var/lib/clickhouse/data

echo "ClickHouse data verified: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 8: Verify Portal Reachability
```bash
# Forward portal service port (if not using ingress)
kubectl port-forward -n data-platform svc/portal-services 3000:3000 &
PORTAL_PID=$!

# Wait for service to be ready
sleep 10

# Test portal endpoint
curl -f http://localhost:3000/api/health || echo "Portal health check failed"
curl -f http://localhost:3000/ || echo "Portal not reachable"

# Stop port forward
kill ${PORTAL_PID}

echo "Portal reachability verified: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 9: Verify MLflow Reachability
```bash
# Forward MLflow service port
kubectl port-forward -n ml-platform svc/mlflow 5000:5000 &
MLFLOW_PID=$!

# Wait for service to be ready
sleep 10

# Test MLflow endpoint
curl -f http://localhost:5000/health || echo "MLflow health check failed"
curl -f http://localhost:5000/ || echo "MLflow not reachable"

# Verify experiments are accessible
curl -f http://localhost:5000/api/2.0/mlflow/experiments/list || echo "MLflow API not responding"

# Stop port forward
kill ${MLFLOW_PID}

echo "MLflow reachability verified: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
```

#### Step 10: Document Results
```bash
# Capture final state
echo "=== RESTORE DRILL SUMMARY ==="
echo "Drill completed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""
echo "Backup used: ${BACKUP_NAME}"
velero backup describe ${BACKUP_NAME} --details

echo ""
echo "Restores performed:"
velero restore get

echo ""
echo "Namespace status:"
kubectl get all -n data-platform
kubectl get all -n ml-platform
kubectl get all -n vault-prod

echo ""
echo "PV status:"
kubectl get pv
kubectl get pvc -n data-platform
kubectl get pvc -n ml-platform

echo ""
echo "=== Evidence Captured ==="
echo "All commands above include timestamps for audit trail"
```

#### Automated Restore Drill Script
For convenience, use the provided validation script:
```bash
# Run complete restore drill with verification
./scripts/velero-restore-validate.sh \
  --schedule daily-backup \
  --namespace data-platform \
  --restore-namespace data-platform \
  --restore-pvs \
  --wait

# Verify MLflow separately
./scripts/velero-restore-validate.sh \
  --schedule daily-backup \
  --namespace ml-platform \
  --restore-namespace ml-platform \
  --restore-pvs \
  --wait
```

#### Success Criteria
- ✅ All backups identified and accessible
- ✅ Velero successfully deployed in empty cluster
- ✅ All namespaces restored without errors
- ✅ All pods reach Ready state
- ✅ ClickHouse database accessible with data intact
- ✅ Portal service responds to HTTP requests
- ✅ MLflow API responds and experiments are accessible
- ✅ All persistent volumes restored with correct data
- ✅ Complete timestamp evidence captured

#### Troubleshooting Restore Issues
```bash
# Check restore errors
velero restore describe <restore-name> --details

# View restore logs
velero restore logs <restore-name>

# Check pod logs for startup issues
kubectl logs -n data-platform -l app=clickhouse --tail=100
kubectl logs -n ml-platform -l app=mlflow --tail=100

# Verify PVC binding
kubectl get pvc -n data-platform
kubectl describe pvc clickhouse-data-pvc -n data-platform

# Check storage class compatibility
kubectl get storageclass
```

## Maintenance

### Backup Validation
```bash
# Validate backup integrity
kubectl exec -n velero deployment/velero -- ./velero backup describe <backup-name>

# Rehearse restore into a scratch namespace
export TARGET_NAMESPACE="data-platform"
export RESTORE_NAMESPACE="data-platform-dr"
export BACKUP_NAME=$(velero backup get --selector velero.io/schedule-name=daily-backup \
  --output json | jq -r '.items | map(select(.status.phase=="Completed")) | sort_by(.status.completionTimestamp) | last | .metadata.name')
kubectl create -f <(envsubst < k8s/storage/velero-restore-test.yaml)

# Or run the helper script (auto handles selection/cleanup)
./scripts/velero-restore-validate.sh --schedule daily-backup --namespace data-platform --wait --cleanup
```

### Cleanup
```bash
# Remove old backups
kubectl delete backup <old-backup-name> -n velero

# Clean failed backup attempts
kubectl get backups -n velero | grep Failed | awk '{print $1}' | xargs kubectl delete backup -n velero
```

## Security

### Access Control
- Velero service account with cluster-admin RBAC
- MinIO credentials stored in Kubernetes secrets
- Encrypted backup data at rest

### Audit Logging
- All backup/restore operations logged
- Access attempts monitored
- Failed operations investigated

## Troubleshooting

### Common Issues

**Backup Fails**:
```bash
# Check Velero logs
kubectl logs deployment/velero -n velero

# Check MinIO connectivity
kubectl exec -n data-platform deployment/minio -- mc admin info myminio

# Verify credentials
kubectl get secret minio-backup-credentials -n velero -o yaml
```

**Restore Fails**:
```bash
# Check restore logs
kubectl describe restore <restore-name> -n velero

# Verify namespace exists
kubectl get namespace <target-namespace>

# Check resource conflicts
kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind --ignore-not-found -n <namespace>
```

**Performance Issues**:
```bash
# Monitor backup duration
kubectl get backups -n velero -o wide

# Check resource usage
kubectl top pods -n velero

# Scale Velero if needed
kubectl scale deployment velero -n velero --replicas=2
```

## Best Practices

### 1. Regular Testing
- Test backups monthly
- Validate restore procedures quarterly
- Document recovery times

### 2. Monitoring
- Monitor backup success rates
- Alert on backup failures
- Track storage utilization

### 3. Security
- Rotate MinIO credentials quarterly
- Encrypt sensitive backup data
- Limit Velero access permissions

### 4. Documentation
- Maintain current DR procedures
- Document recovery steps
- Train team on backup procedures

## Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `velero-backup-config.yaml` | Main backup configuration with all namespaces | `k8s/storage/` |
| `velero-restore-full.yaml` | Full cluster restore template | `k8s/storage/` |
| `velero-restore-dr-drill.yaml` | DR drill restore (critical namespaces only) | `k8s/storage/` |
| `velero-restore-namespace.yaml` | Single namespace restore template | `k8s/storage/` |
| `velero-restore-app.yaml` | Application-specific restore template | `k8s/storage/` |
| `velero-restore-test.yaml` | Test restore configuration (no PVs) | `k8s/storage/` |
| `deploy-velero-backup.sh` | Deployment script | `scripts/` |
| `velero-restore-validate.sh` | Automated restore validation script | `scripts/` |
| This guide | Complete procedures | `docs/operations/backup-guide.md` |

## Support

For backup and recovery issues:
1. Check Velero logs: `kubectl logs -n velero`
2. Verify MinIO connectivity
3. Review backup schedules
4. Test restore procedures
5. Escalate to infrastructure team if needed

---
**Status**: ✅ Ready for Production
**Last Updated**: October 20, 2025
**Next Review**: November 20, 2025
