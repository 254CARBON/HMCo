# Velero Installation Guide for 254Carbon

## Prerequisites

- MinIO running in data-platform namespace
- velero-backups bucket created in MinIO
- Credentials configured

## Installation Steps

### 1. Install Velero CLI

```bash
# Download Velero CLI
wget https://github.com/vmware-tanzu/velero/releases/download/v1.14.1/velero-v1.14.1-linux-amd64.tar.gz
tar -xvf velero-v1.14.1-linux-amd64.tar.gz
sudo mv velero-v1.14.1-linux-amd64/velero /usr/local/bin/
rm -rf velero-v1.14.1-linux-amd64*
```

### 2. Create Credentials File

```bash
cat > /tmp/credentials-velero <<EOF
[default]
aws_access_key_id=$(kubectl get secret minio-secret -n data-platform -o jsonpath='{.data.access-key}' | base64 -d)
aws_secret_access_key=$(kubectl get secret minio-secret -n data-platform -o jsonpath='{.data.secret-key}' | base64 -d)
EOF
```

### 3. Install Velero

```bash
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.10.0 \
  --bucket velero-backups \
  --secret-file /tmp/credentials-velero \
  --use-volume-snapshots=true \
  --backup-location-config region=us-east-1,s3ForcePathStyle="true",s3Url=http://minio-service.data-platform.svc.cluster.local:9000 \
  --snapshot-location-config region=us-east-1 \
  --use-node-agent \
  --namespace velero

# Clean up credentials file
rm /tmp/credentials-velero
```

### 4. Verify Installation

```bash
# Check Velero deployment
kubectl get deployment -n velero

# Check backup location
velero backup-location get

# Check if MinIO is reachable
velero backup-location get -o json | jq '.items[0].status'
```

### 5. Create Backup Schedules

```bash
# Daily backup of data-platform namespace
velero schedule create daily-data-platform \
  --schedule="0 2 * * *" \
  --include-namespaces data-platform \
  --ttl 720h0m0s

# Daily backup of monitoring namespace
velero schedule create daily-monitoring \
  --schedule="0 3 * * *" \
  --include-namespaces monitoring \
  --ttl 720h0m0s

# Weekly full cluster backup
velero schedule create weekly-full \
  --schedule="0 4 * * 0" \
  --ttl 2160h0m0s
```

### 6. Test Backup and Restore

```bash
# Create a test backup
velero backup create test-backup --include-namespaces data-platform

# Check backup status
velero backup describe test-backup

# Test restore (to a different namespace)
velero restore create --from-backup test-backup
```

## Configuration Files Created

- `/home/m/tff/254CARBON/HMCo/k8s/storage/velero-values.yaml` - Helm values (if using Helm)
- `/home/m/tff/254CARBON/HMCo/k8s/storage/velero-setup-job.yaml` - MinIO bucket setup

## Troubleshooting

If Velero pods are not starting:
1. Check MinIO is accessible
2. Verify credentials are correct
3. Ensure velero-backups bucket exists
4. Check Velero logs: `kubectl logs -n velero deployment/velero`


