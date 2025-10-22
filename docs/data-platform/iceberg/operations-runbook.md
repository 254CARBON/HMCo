# Iceberg Operations Runbook

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| Deploy Iceberg | `kubectl apply -f k8s/data-lake/iceberg-rest.yaml` | 2-5 min |
| Check Status | `kubectl get pod -n data-platform -l app=iceberg-rest-catalog` | 10 sec |
| View Logs | `kubectl logs -f deployment/iceberg-rest-catalog -n data-platform` | - |
| Run Tests | See ICEBERG_INTEGRATION_TEST_GUIDE.md | 15-30 min |
| Scale Up | `kubectl scale deployment iceberg-rest-catalog -n data-platform --replicas=3` | 3-5 min |

## Common Operations

### Daily Operations

#### 1. Morning Checks (10-15 minutes)

```bash
#!/bin/bash
# Daily health check script

echo "=== Iceberg REST Catalog Health Check ==="
echo ""

# 1. Check pod status
echo "1. Pod Status:"
kubectl get pod -n data-platform -l app=iceberg-rest-catalog
echo ""

# 2. Check metrics
echo "2. Resource Usage:"
kubectl top pod -n data-platform -l app=iceberg-rest-catalog
echo ""

# 3. Check REST API
echo "3. REST API Health:"
kubectl port-forward -n data-platform svc/iceberg-rest-catalog 8181:8181 &
PID=$!
sleep 2
curl -s http://localhost:8181/v1/config | jq . && echo "✓ API OK" || echo "✗ API FAILED"
kill $PID
echo ""

# 4. Check events for warnings
echo "4. Recent Events:"
kubectl get events -n data-platform --sort-by='.lastTimestamp' | tail -5
```

#### 2. Check Error Logs

```bash
# Check for errors in last hour
kubectl logs --since=1h -n data-platform deployment/iceberg-rest-catalog | grep -i error | tail -20

# Check specific pod
kubectl logs -n data-platform iceberg-rest-catalog-xxx | tail -50
```

### Weekly Operations

#### 1. Backup Configuration

```bash
# Backup PostgreSQL Iceberg database
kubectl exec -it postgres-shared-xxx -- \
  pg_dump -U iceberg_user -d iceberg_rest > backup_iceberg_rest_$(date +%Y%m%d).sql

# Store backup securely
# mv backup_iceberg_rest_*.sql /secure/backup/location/
```

#### 2. Verify Data Integrity

```sql
-- In Trino:
SELECT 
  COUNT(*) as total_tables,
  COUNT(DISTINCT schema_name) as schemas
FROM iceberg_system.tables
WHERE catalog = 'iceberg';

-- Check for corrupted tables
SELECT * FROM iceberg_system.tables 
WHERE last_updated < CURRENT_DATE - INTERVAL '7' DAY;
```

#### 3. Clean Up Old Snapshots

```sql
-- Expire snapshots older than 30 days
CALL iceberg.system.expire_snapshots('database.table', INTERVAL '30' DAY);

-- Remove orphan files
CALL iceberg.system.remove_orphan_files('database.table');
```

### Monthly Operations

#### 1. Credential Rotation

```bash
# Update MinIO admin password
mc admin user change-password minio minioadmin <new-strong-password>

# Update Iceberg database user password
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -c "ALTER USER iceberg_user WITH PASSWORD '<new-password>';"

# Update Kubernetes secrets
kubectl create secret generic iceberg-db-secret \
  -n data-platform \
  --from-literal=password=<new-password> \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new credentials
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform
```

#### 2. Security Audit

```bash
# Verify no exposed credentials
grep -r "password" k8s/ --include="*.yaml" | grep -v "secretKeyRef"

# Check RBAC permissions
kubectl get rolebindings -n data-platform
kubectl get clusterrolebindings | grep data-platform

# Audit logs for suspicious activity
kubectl logs -n data-platform deployment/iceberg-rest-catalog | grep -i "unauthorized\|failed\|denied"
```

#### 3. Performance Tuning

```bash
# Review slow queries
kubectl logs -n data-platform deployment/iceberg-rest-catalog | grep "slow query\|query time"

# Check resource limits
kubectl get deployment iceberg-rest-catalog -n data-platform -o yaml | grep -A 5 resources:

# Review Prometheus metrics
# Navigate to http://localhost:9090 and query:
# - http_request_duration_seconds_bucket
# - http_requests_total
```

## Troubleshooting

### Pod Stuck in CrashLoopBackOff

```bash
# 1. Get detailed error
kubectl describe pod -n data-platform iceberg-rest-catalog-xxx

# 2. Check logs
kubectl logs -n data-platform iceberg-rest-catalog-xxx --previous

# 3. Common causes:
# - Database connectivity: Check PostgreSQL pod status
# - Memory limit: Increase JAVA_OPTS or container memory
# - Configuration error: Verify environment variables
```

### Slow Queries

```bash
# 1. Check resource usage
kubectl top pod -n data-platform iceberg-rest-catalog-xxx

# 2. Check database performance
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -c "SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;"

# 3. Add database indexes if needed
kubectl exec -it postgres-shared-xxx -- \
  psql -U iceberg_user -d iceberg_rest -c "CREATE INDEX IF NOT EXISTS idx_table_name ON iceberg_tables(table_name);"
```

### Connection Refused Errors

```bash
# 1. Verify services are running
kubectl get svc -n data-platform | grep -E "iceberg|postgres|minio"

# 2. Test network connectivity
kubectl exec -it iceberg-rest-catalog-xxx -- \
  curl -v http://postgres-shared-service:5432

# 3. Check DNS resolution
kubectl exec -it iceberg-rest-catalog-xxx -- \
  nslookup postgres-shared-service.data-platform.svc.cluster.local
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# Daily backup script

BACKUP_DIR="/backups/iceberg"
DATE=$(date +%Y%m%d_%H%M%S)

# 1. Backup PostgreSQL database
kubectl exec postgres-shared-xxx -- \
  pg_dump -U iceberg_user -d iceberg_rest > \
  $BACKUP_DIR/iceberg_rest_$DATE.sql.gz

# 2. Backup MinIO bucket
mc mirror --watch minio/iceberg-warehouse/ \
  $BACKUP_DIR/iceberg_warehouse_$DATE/

# 3. Keep only last 30 days
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed at $(date)"
```

### Recovery Procedures

#### Recover PostgreSQL Database

```bash
# 1. Connect to PostgreSQL
kubectl exec -it postgres-shared-xxx -- bash

# 2. Restore from backup
psql -U iceberg_user -d iceberg_rest < /backups/iceberg_rest_YYYYMMDD_HHMMSS.sql.gz
```

#### Recover MinIO Data

```bash
# 1. Restore bucket
mc mirror /backups/iceberg_warehouse_YYYYMMDD_HHMMSS/ \
  minio/iceberg-warehouse/

# 2. Verify data integrity
mc ls minio/iceberg-warehouse/ --recursive | wc -l
```

## Scaling Operations

### Scale Up (High Load)

```bash
# 1. Increase replicas
kubectl scale deployment iceberg-rest-catalog -n data-platform --replicas=3

# 2. Monitor rollout
kubectl rollout status deployment/iceberg-rest-catalog -n data-platform

# 3. Verify all pods are ready
kubectl get pod -n data-platform -l app=iceberg-rest-catalog
```

### Scale Down (Maintenance)

```bash
# 1. Drain connections (wait for in-flight requests)
sleep 60

# 2. Scale down
kubectl scale deployment iceberg-rest-catalog -n data-platform --replicas=1

# 3. Verify
kubectl get pod -n data-platform -l app=iceberg-rest-catalog
```

## Maintenance Windows

### Planned Maintenance

```bash
#!/bin/bash
# Maintenance script

echo "Starting maintenance window..."

# 1. Notify users (if applicable)
# Send message: "Iceberg catalog maintenance in 10 minutes"

# 2. Wait for jobs to complete
sleep 600

# 3. Stop accepting new requests (optional)
# kubectl patch svc iceberg-rest-catalog -n data-platform -p '{"spec":{"selector":{"maintenance":"true"}}}'

# 4. Perform maintenance
# - Apply updates
# - Optimize indexes
# - Clean up old data

# 5. Resume operations
# kubectl patch svc iceberg-rest-catalog -n data-platform -p '{"spec":{"selector":{"maintenance":"false"}}}'

# 6. Verify everything works
kubectl get pod -n data-platform -l app=iceberg-rest-catalog

echo "Maintenance window complete"
```

## Upgrade Procedures

### Upgrade Iceberg REST Catalog

```bash
# 1. Check current version
kubectl get deployment iceberg-rest-catalog -n data-platform -o jsonpath='{.spec.template.spec.containers[0].image}'

# 2. Update image
kubectl set image deployment/iceberg-rest-catalog \
  -n data-platform \
  iceberg-rest-catalog=tabulario/iceberg-rest:0.7.0 \
  --record

# 3. Monitor rollout
kubectl rollout status deployment/iceberg-rest-catalog -n data-platform

# 4. If issues, rollback
kubectl rollout undo deployment/iceberg-rest-catalog -n data-platform
```

## Useful Commands

```bash
# Get all Iceberg resources
kubectl get all -n data-platform -l app=iceberg-rest-catalog

# Get detailed pod info
kubectl describe pod -n data-platform iceberg-rest-catalog-xxx

# Get pod YAML
kubectl get pod iceberg-rest-catalog-xxx -n data-platform -o yaml

# Execute command in pod
kubectl exec -it iceberg-rest-catalog-xxx -n data-platform -- /bin/bash

# Port-forward
kubectl port-forward -n data-platform svc/iceberg-rest-catalog 8181:8181

# View resource limits
kubectl top node
kubectl top pod -n data-platform

# Get events
kubectl get events -n data-platform --sort-by='.lastTimestamp'

# Stream logs
kubectl logs -f deployment/iceberg-rest-catalog -n data-platform

# Apply manifests
kubectl apply -f k8s/data-lake/iceberg-rest.yaml

# Dry-run apply
kubectl apply -f k8s/data-lake/iceberg-rest.yaml --dry-run=client -o yaml
```

## Contact and Escalation

| Issue | Contact | Priority |
|-------|---------|----------|
| Pod crashes | Platform team | Critical |
| High latency | Performance team | High |
| Database issues | DBA team | Critical |
| Network issues | Network team | High |
| Storage issues | Storage team | High |
| Security issues | Security team | Critical |

## References

- [Kubernetes Troubleshooting](https://kubernetes.io/docs/tasks/debug-application-cluster/)
- [Apache Iceberg Operations](https://iceberg.apache.org/docs/latest/maintenance/)
- [PostgreSQL Administration](https://www.postgresql.org/docs/current/admin.html)
- [MinIO Administration](https://docs.min.io/docs/minio-admin-complete-guide.html)

## Appendix: Emergency Contacts

```
On-Call Engineer: [Number/Email]
Platform Lead: [Number/Email]
Infrastructure Team: [Slack Channel]
Emergency: [Escalation Process]
```

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-10-19 | Initial runbook | Platform Team |
| | | |
| | | |
