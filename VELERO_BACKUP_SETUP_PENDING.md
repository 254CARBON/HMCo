# Velero Backup Setup - Manual Step Required

**Status**: ⏳ Pending manual bucket creation  
**Priority**: HIGH (but not blocking current operations)

## Quick Fix Required

### Option 1: Via MinIO Console (Easiest - 2 minutes)
```bash
# Access MinIO console
open https://minio.254carbon.com

# Login
Username: minioadmin
Password: minioadmin123

# Create bucket
1. Click "Create Bucket"
2. Name: velero-backups
3. Click "Create"
```

### Option 2: Via AWS CLI (If installed)
```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123

aws --endpoint-url http://localhost:9000 s3 mb s3://velero-backups
```

### After bucket is created:
```bash
# Verify Velero sees it
kubectl get backupstoragelocations -n velero

# Should show Phase: Available
```

## What's Already Configured ✅

- Velero deployed (3 pods running)
- MinIO credentials secret created
- BackupStorageLocation configured
- Daily backup schedule created (2 AM UTC)
- Hourly data-platform backup created

## Once Available:

Test backup:
```bash
velero backup create manual-test --wait
velero backup describe manual-test
```

**File**: k8s/backup/velero-minio-backupstoragelocation.yaml (ready to use)

