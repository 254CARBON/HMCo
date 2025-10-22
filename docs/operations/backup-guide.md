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
- **Scope**: All namespaces (data-platform, monitoring, registry, vault-prod)
- **Retention**: 30 days
- **Frequency**: Daily at 2:00 AM

### 2. Critical Backups (Every 6 hours)
- **Scope**: Critical namespaces only (data-platform, vault-prod)
- **Retention**: 7 days
- **Frequency**: Every 6 hours

### 3. PostgreSQL Backups (3 AM)
- **Scope**: Database-only backups
- **Retention**: 7 days
- **Frequency**: Daily at 3:00 AM

### 4. Manual Backups
- **Scope**: Custom selection
- **Retention**: Configurable
- **Trigger**: Manual execution

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
# Restore entire cluster
kubectl apply -f k8s/storage/velero-restore-full.yaml
```

### 2. Namespace-Specific Recovery
```bash
# Restore specific namespace
kubectl apply -f k8s/storage/velero-restore-namespace.yaml
```

### 3. Application Recovery
```bash
# Restore specific application
kubectl apply -f k8s/storage/velero-restore-app.yaml
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
- Monthly DR drills
- Quarterly full recovery tests
- Annual comprehensive DR exercise

## Maintenance

### Backup Validation
```bash
# Validate backup integrity
kubectl exec -n velero deployment/velero -- ./velero backup describe <backup-name>

# Test restore without affecting production
kubectl apply -f k8s/storage/velero-restore-test.yaml
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
| `velero-backup-config.yaml` | Main backup configuration | `k8s/storage/` |
| `deploy-velero-backup.sh` | Deployment script | `scripts/` |
| `velero-restore-test.yaml` | Test restore configuration | `k8s/storage/` |
| This guide | Complete procedures | `PHASE5_BACKUP_GUIDE.md` |

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
