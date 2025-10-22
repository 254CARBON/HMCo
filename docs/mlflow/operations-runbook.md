# MLFlow Operations Runbook

Operational procedures for managing MLFlow in the 254Carbon data platform.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Common Tasks](#common-tasks)
3. [Backup & Recovery](#backup--recovery)
4. [Scaling & Performance](#scaling--performance)
5. [Maintenance Windows](#maintenance-windows)
6. [Disaster Recovery](#disaster-recovery)

---

## Daily Operations

### Morning Health Check

Run this every morning to verify system health:

```bash
#!/bin/bash
# mlflow-health-check.sh

echo "=== MLFlow Health Check ==="
date

# Check pod status
echo -e "\n1. Pod Status"
kubectl get pods -n data-platform -l app=mlflow

# Check services
echo -e "\n2. Service Status"
kubectl get svc -n data-platform | grep mlflow

# Check ingress
echo -e "\n3. Ingress Status"
kubectl get ingress -n data-platform mlflow-ingress

# Check resource usage
echo -e "\n4. Resource Usage"
kubectl top pods -n data-platform -l app=mlflow

# Check backend services
echo -e "\n5. PostgreSQL Status"
kubectl get pods -n data-platform -l app=postgres | head -1

echo -e "\n6. MinIO Status"
kubectl get pods -n data-platform -l app=minio | head -1

# Check for errors in logs
echo -e "\n7. Recent Errors"
kubectl logs -n data-platform -l app=mlflow --tail=20 | grep -i error || echo "No errors found"

echo -e "\n=== Health Check Complete ==="
```

### Log Rotation

MLFlow logs are managed by Kubernetes. To prevent excessive logging:

```bash
# Check log volume
kubectl exec -n data-platform mlflow-<pod> -- du -sh /var/log

# View active logs
kubectl logs -n data-platform -l app=mlflow --tail=100

# Archive old logs (if needed)
kubectl logs -n data-platform -l app=mlflow > /backup/mlflow-logs-$(date +%Y%m%d).txt
```

---

## Common Tasks

### Create New Experiment

Two methods:

**Method 1: Via API**

```bash
# Create experiment
curl -X POST http://mlflow.data-platform.svc.cluster.local:5000/api/2.0/experiments/create \
  -H "Content-Type: application/json" \
  -d '{"name": "new_experiment"}'

# Verify creation
curl http://mlflow.data-platform.svc.cluster.local:5000/api/2.0/experiments/list | jq .
```

**Method 2: Via Python**

```python
import mlflow
mlflow.set_tracking_uri("http://mlflow.data-platform.svc.cluster.local:5000")
mlflow.create_experiment("new_experiment")
```

### Archive Old Runs

Keep experiments size manageable by archiving old runs:

```python
from mlflow.tracking import MlflowClient
import time

client = MlflowClient("http://mlflow.data-platform.svc.cluster.local:5000")

# Find old runs (older than 90 days)
cutoff_time = int((time.time() - 90*24*60*60) * 1000)

experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"start_time < {cutoff_time}"
    )
    
    # Delete old runs
    for run in runs:
        client.delete_run(run.info.run_id)
        print(f"Deleted run {run.info.run_id}")
```

### Search Experiments

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://mlflow.data-platform.svc.cluster.local:5000")

# List all experiments
experiments = client.search_experiments()
for exp in experiments:
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")

# Search for specific experiment
exp = client.get_experiment_by_name("my_experiment")
if exp:
    print(f"Found: {exp.experiment_id}")
else:
    print("Experiment not found")
```

### Compare Runs

```python
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient("http://mlflow.data-platform.svc.cluster.local:5000")

# Get runs from experiment
runs = client.search_runs(experiment_ids=["0"])

# Create comparison dataframe
data = []
for run in runs:
    data.append({
        "Run ID": run.info.run_id,
        "Status": run.info.status,
        "Accuracy": run.data.metrics.get("accuracy", "N/A"),
        "F1": run.data.metrics.get("f1", "N/A"),
    })

df = pd.DataFrame(data)
print(df.to_string())
```

### Update Run Metadata

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://mlflow.data-platform.svc.cluster.local:5000")

# Get run
run = client.get_run("run_id")

# Update tag
client.set_tag(run.info.run_id, "deployed", "true")

# Update parameter
client.log_param(run.info.run_id, "new_param", "value")
```

---

## Backup & Recovery

### Backup Procedures

#### 1. PostgreSQL Backup

```bash
#!/bin/bash
# backup-mlflow-postgres.sh

BACKUP_DIR="/backups/mlflow"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup MLFlow database
kubectl exec -n data-platform postgres-shared-<pod> -- \
  pg_dump -U mlflow -d mlflow \
  > $BACKUP_DIR/mlflow_db_$TIMESTAMP.sql

# Verify backup
if [ -s $BACKUP_DIR/mlflow_db_$TIMESTAMP.sql ]; then
    echo "PostgreSQL backup successful: $BACKUP_DIR/mlflow_db_$TIMESTAMP.sql"
    # Compress
    gzip $BACKUP_DIR/mlflow_db_$TIMESTAMP.sql
    # Rotate old backups (keep last 7)
    ls -t $BACKUP_DIR/*.sql.gz | tail -n +8 | xargs rm -f
else
    echo "ERROR: PostgreSQL backup failed"
    exit 1
fi
```

#### 2. MinIO Artifact Backup

```bash
#!/bin/bash
# backup-mlflow-artifacts.sh

BACKUP_DIR="/backups/mlflow"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup artifacts from MinIO
# Option A: Using mc CLI
kubectl exec -n data-platform minio-<pod> -- /bin/sh -c \
  "mc mirror local/mlflow-artifacts $BACKUP_DIR/artifacts_$TIMESTAMP --exclude '.minio*'" \
  > /dev/null 2>&1

# Option B: Using S3 sync (if S3 credentials available)
# aws s3 sync s3://mlflow-artifacts $BACKUP_DIR/artifacts_$TIMESTAMP

echo "MinIO artifacts backup to: $BACKUP_DIR/artifacts_$TIMESTAMP"

# Verify
if [ -d "$BACKUP_DIR/artifacts_$TIMESTAMP" ]; then
    # Compress
    tar -czf $BACKUP_DIR/artifacts_$TIMESTAMP.tar.gz \
      -C $BACKUP_DIR artifacts_$TIMESTAMP
    rm -rf $BACKUP_DIR/artifacts_$TIMESTAMP
    echo "Compressed backup: $BACKUP_DIR/artifacts_$TIMESTAMP.tar.gz"
fi
```

#### 3. Configuration Backup

```bash
#!/bin/bash
# backup-mlflow-config.sh

BACKUP_DIR="/backups/mlflow"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/config_$TIMESTAMP

# Backup K8s resources
kubectl get deployment -n data-platform mlflow -o yaml \
  > $BACKUP_DIR/config_$TIMESTAMP/mlflow-deployment.yaml

kubectl get service -n data-platform mlflow -o yaml \
  > $BACKUP_DIR/config_$TIMESTAMP/mlflow-service.yaml

kubectl get configmap -n data-platform mlflow-config -o yaml \
  > $BACKUP_DIR/config_$TIMESTAMP/mlflow-configmap.yaml

kubectl get secret -n data-platform mlflow-backend-secret -o yaml \
  > $BACKUP_DIR/config_$TIMESTAMP/mlflow-secrets.yaml

kubectl get ingress -n data-platform mlflow-ingress -o yaml \
  > $BACKUP_DIR/config_$TIMESTAMP/mlflow-ingress.yaml

tar -czf $BACKUP_DIR/mlflow_config_$TIMESTAMP.tar.gz \
  -C $BACKUP_DIR config_$TIMESTAMP
rm -rf $BACKUP_DIR/config_$TIMESTAMP

echo "Configuration backup: $BACKUP_DIR/mlflow_config_$TIMESTAMP.tar.gz"
```

#### 4. Automated Daily Backup

Create a CronJob for automated backups:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mlflow-backup
  namespace: data-platform
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: default
          containers:
          - name: backup
            image: alpine:3.18
            env:
            - name: BACKUP_DEST
              value: "s3://mlflow-backups"
            command:
            - /bin/sh
            - -c
            - |
              # Install tools
              apk add --no-cache postgresql-client curl
              
              # Backup database
              pg_dump -h postgres-shared-service.data-platform.svc.cluster.local \
                -U mlflow -d mlflow | \
                gzip > /tmp/mlflow_db_$(date +%Y%m%d).sql.gz
              
              # Upload to S3
              curl -X POST -d @/tmp/mlflow_db_$(date +%Y%m%d).sql.gz \
                https://minio.minio.svc.cluster.local:9000/mlflow-backups/
            
            volumeMounts:
            - name: backup
              mountPath: /backup
          
          restartPolicy: OnFailure
          volumes:
          - name: backup
            emptyDir: {}
```

### Recovery Procedures

#### Restore PostgreSQL Database

```bash
#!/bin/bash
# restore-mlflow-postgres.sh

BACKUP_FILE=$1

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Drop existing database
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U datahub -d postgres -c "DROP DATABASE IF EXISTS mlflow;"

# Restore from backup
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | kubectl exec -i -n data-platform postgres-shared-<pod> -- \
      psql -U datahub -d postgres
else
    kubectl exec -i -n data-platform postgres-shared-<pod> -- \
      psql -U datahub -d postgres < $BACKUP_FILE
fi

echo "Database restoration complete"
```

#### Restore MinIO Artifacts

```bash
#!/bin/bash
# restore-mlflow-artifacts.sh

BACKUP_FILE=$1

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Extract and upload to MinIO
TEMP_DIR="/tmp/mlflow_restore"
mkdir -p $TEMP_DIR

tar -xzf $BACKUP_FILE -C $TEMP_DIR

# Upload to MinIO
kubectl exec -it -n data-platform minio-<pod> -- /bin/sh -c \
  "mc mirror $TEMP_DIR/artifacts_* local/mlflow-artifacts"

rm -rf $TEMP_DIR
echo "Artifact restoration complete"
```

---

## Scaling & Performance

### Horizontal Scaling

```bash
# Increase replicas for load distribution
kubectl scale deployment -n data-platform mlflow --replicas=4

# Verify scaling
kubectl get pods -n data-platform -l app=mlflow -o wide

# Monitor resources
kubectl top pods -n data-platform -l app=mlflow
```

### Vertical Scaling (Resource Limits)

```bash
# Increase memory and CPU
kubectl patch deployment -n data-platform mlflow --type json -p \
  '[
    {"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/memory","value":"4Gi"},
    {"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/memory","value":"8Gi"},
    {"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/cpu","value":"1000m"},
    {"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/cpu","value":"2000m"}
  ]'

# Restart deployment to apply changes
kubectl rollout restart deployment -n data-platform mlflow
```

### Database Optimization

```bash
# Analyze and vacuum PostgreSQL
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U mlflow -d mlflow -c "VACUUM ANALYZE;"

# Check table sizes
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U mlflow -d mlflow << 'EOF'
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
EOF

# Create indexes if missing (MLFlow should handle this)
# kubectl exec -it -n data-platform postgres-shared-<pod> -- \
#   psql -U mlflow -d mlflow -f k8s/compute/mlflow/create-indexes.sql
```

---

## Maintenance Windows

### Rolling Update

Perform updates with zero downtime:

```bash
# Update image
kubectl set image deployment/mlflow -n data-platform \
  mlflow=ghcr.io/mlflow/mlflow:v2.11.0 \
  --record

# Monitor rollout
kubectl rollout status deployment/mlflow -n data-platform -w

# Check new version
kubectl exec -n data-platform mlflow-<pod> -- \
  mlflow --version
```

### Configuration Changes

```bash
# Update ConfigMap
kubectl patch configmap -n data-platform mlflow-config --type merge -p \
  '{"data":{"gunicorn_workers":"8"}}'

# Restart pods to apply
kubectl rollout restart deployment -n data-platform mlflow

# Verify changes
kubectl get configmap -n data-platform mlflow-config -o yaml
```

### Certificate Renewal

```bash
# Manually trigger cert renewal
kubectl annotate ingress -n data-platform mlflow-ingress \
  cert-manager.io/issue-temporary-certificate="true" --overwrite

# Check certificate status
kubectl describe ingress -n data-platform mlflow-ingress | grep -A 5 TLS

# View certificate details
kubectl get secret -n data-platform mlflow-tls -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -text -noout
```

---

## Disaster Recovery

### Complete System Recovery

If MLFlow is completely lost:

```bash
#!/bin/bash
# disaster-recovery.sh

echo "MLFlow Disaster Recovery Started"

# Step 1: Recreate PostgreSQL database
echo "Step 1: Recreating database..."
kubectl exec -it -n data-platform postgres-shared-<pod> -- \
  psql -U datahub -d postgres -f - < k8s/compute/mlflow/mlflow-backend-db.sql

# Step 2: Restore database from backup
echo "Step 2: Restoring from backup..."
# (Use restore script above)

# Step 3: Recreate MinIO bucket
echo "Step 3: Recreating MinIO bucket..."
kubectl exec -it -n data-platform minio-<pod> -- /bin/sh -c \
  "mc mb local/mlflow-artifacts && mc version enable local/mlflow-artifacts"

# Step 4: Restore artifacts from backup
echo "Step 4: Restoring artifacts..."
# (Use restore script above)

# Step 5: Redeploy K8s resources
echo "Step 5: Redeploying Kubernetes resources..."
kubectl apply -f k8s/compute/mlflow/mlflow-secrets.yaml
kubectl apply -f k8s/compute/mlflow/mlflow-configmap.yaml
kubectl apply -f k8s/compute/mlflow/mlflow-service.yaml
kubectl apply -f k8s/compute/mlflow/mlflow-deployment.yaml

# Step 6: Verify
echo "Step 6: Verifying deployment..."
kubectl get pods -n data-platform -l app=mlflow
kubectl wait --for=condition=Ready pod -l app=mlflow -n data-platform --timeout=300s

echo "MLFlow Disaster Recovery Complete"
```

### RTO & RPO Targets

- **RTO (Recovery Time Objective)**: < 30 minutes
- **RPO (Recovery Point Objective)**: < 1 hour (daily backups)

---

## Monitoring & Alerting

### Key Metrics to Monitor

1. **Pod Health**
   ```bash
   kubectl get pods -n data-platform -l app=mlflow
   ```

2. **Resource Usage**
   ```bash
   kubectl top pods -n data-platform -l app=mlflow
   ```

3. **API Response Time**
   ```bash
   # Port forward and use load testing tool
   kubectl port-forward -n data-platform svc/mlflow 5000:5000
   ab -n 100 -c 10 http://localhost:5000/health
   ```

4. **Database Size**
   ```bash
   kubectl exec -it -n data-platform postgres-shared-<pod> -- \
     psql -U mlflow -d mlflow -c "SELECT pg_size_pretty(pg_database_size('mlflow'));"
   ```

5. **Artifact Storage Usage**
   ```bash
   kubectl exec -it -n data-platform minio-<pod> -- /bin/sh -c \
     "mc du local/mlflow-artifacts --recursive"
   ```

### Create Alerts

Use Prometheus rules (if configured):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: mlflow-alerts
  namespace: data-platform
spec:
  groups:
  - name: mlflow
    interval: 30s
    rules:
    - alert: MLFlowPodDown
      expr: count(up{job="mlflow"} == 1) < 1
      for: 5m
      annotations:
        summary: "MLFlow pod is down"

    - alert: MLFlowHighMemory
      expr: container_memory_usage_bytes{pod=~"mlflow.*"} > 2e9
      for: 10m
      annotations:
        summary: "MLFlow memory usage >2GB"

    - alert: MLFlowDatabaseError
      expr: increase(mlflow_db_errors_total[5m]) > 0
      annotations:
        summary: "MLFlow database errors detected"
```

---

## Contact & Escalation

### On-Call Support

For issues during business hours:
- Check troubleshooting guide: `docs/mlflow/troubleshooting.md`
- Contact: Data Platform Team

For production emergencies:
- Escalate to DevOps team
- Trigger disaster recovery if needed

### Useful Commands Reference

```bash
# View all resources
kubectl get all -n data-platform -l app=mlflow

# Get detailed status
kubectl describe pod -n data-platform mlflow-<pod>

# View last 100 logs
kubectl logs -n data-platform -l app=mlflow --tail=100

# Port forward for debugging
kubectl port-forward -n data-platform svc/mlflow 5000:5000

# Execute command in pod
kubectl exec -it -n data-platform mlflow-<pod> -- bash

# Get resource definitions
kubectl get deployment -n data-platform mlflow -o yaml > mlflow-deployment-backup.yaml
```

---

**Last Updated**: October 2025
**Next Review**: January 2026
