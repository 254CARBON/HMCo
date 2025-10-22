# Iceberg Integration End-to-End Testing Guide

## Overview

This document provides comprehensive testing procedures to verify the Iceberg REST Catalog integration across all components: Iceberg, Trino, DataHub, and SeaTunnel.

## Prerequisites

Verify all components are deployed:

```bash
# Check all deployments
kubectl get deployments -n data-platform | grep -E "iceberg|trino|datahub|seatunnel"

# Expected output should show:
# - iceberg-rest-catalog running
# - trino-coordinator running
# - trino-worker running
# - datahub-frontend running
# - datahub-gms running (may need to scale from 0)
```

## Phase 1: Iceberg REST Catalog Testing

### 1.1 Health Check

```bash
# Port-forward to Iceberg REST Catalog
kubectl port-forward -n data-platform svc/iceberg-rest-catalog 8181:8181 &

# Test REST API
curl -s http://localhost:8181/v1/config | jq .

# Expected response:
# {
#   "defaults": {},
#   "overrides": {}
# }
```

### 1.2 Create Test Namespace

```bash
curl -X POST http://localhost:8181/v1/namespaces \
  -H "Content-Type: application/json" \
  -d '{"namespace": "test_namespace"}'

# Expected response:
# {
#   "namespace": ["test_namespace"],
#   "properties": {}
# }
```

### 1.3 Verify Namespace

```bash
curl http://localhost:8181/v1/namespaces

# Expected response:
# {
#   "namespaces": [
#     ["test_namespace"]
#   ]
# }
```

## Phase 2: MinIO/S3 Storage Testing

### 2.1 Check Bucket Creation

```bash
# Port-forward to MinIO
kubectl port-forward -n data-platform svc/minio-service 9000:9000 &

# List buckets (using mc or S3 CLI)
aws s3 ls --endpoint-url http://localhost:9000 \
  --access-key minioadmin \
  --secret-key minioadmin123

# Expected output:
# 2025-10-19 12:00:00 datahub-storage
# 2025-10-19 12:00:00 iceberg-warehouse
# 2025-10-19 12:00:00 lakefs-data
# 2025-10-19 12:00:00 seatunnel-output
```

### 2.2 Verify Warehouse Directory

```bash
aws s3 ls s3://iceberg-warehouse/ \
  --endpoint-url http://localhost:9000 \
  --access-key minioadmin \
  --secret-key minioadmin123 \
  --recursive

# Should initially be empty or show test namespace path
```

## Phase 3: Trino Integration Testing

### 3.1 Connect to Trino

```bash
# Port-forward to Trino
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080 &

# Using trino-cli (if available locally)
trino --server http://localhost:8080

# Or use curl to test catalog
curl -s http://localhost:8080/v1/catalogs | jq .

# Should include iceberg catalog
```

### 3.2 Test Iceberg Catalog Connection

```sql
-- Within Trino CLI or tool:

-- List catalogs
SHOW CATALOGS;

-- Expected output:
-- Catalog
-- -------
-- iceberg
-- postgresql-shared
-- postgresql-workflow
-- system
-- (others)

-- List schemas in Iceberg
SHOW SCHEMAS FROM iceberg;

-- Expected output:
-- Schema
-- --------
-- default
-- information_schema
```

### 3.3 Create Test Table

```sql
-- Create test schema if needed
CREATE SCHEMA IF NOT EXISTS iceberg.test_schema;

-- Create test table
CREATE TABLE iceberg.test_schema.test_table (
    id BIGINT,
    name VARCHAR,
    email VARCHAR,
    created_at TIMESTAMP(3) WITH TIME ZONE
)
WITH (
    format = 'PARQUET',
    location = 's3://iceberg-warehouse/test_schema/test_table'
);

-- Verify table creation
SHOW TABLES FROM iceberg.test_schema;

-- Expected output:
-- Table
-- -----------
-- test_table
```

### 3.4 Insert and Query Data

```sql
-- Insert test data
INSERT INTO iceberg.test_schema.test_table
VALUES 
    (1, 'John Doe', 'john@example.com', CURRENT_TIMESTAMP),
    (2, 'Jane Smith', 'jane@example.com', CURRENT_TIMESTAMP),
    (3, 'Bob Johnson', 'bob@example.com', CURRENT_TIMESTAMP);

-- Query data
SELECT * FROM iceberg.test_schema.test_table;

-- Expected output:
-- id | name        | email              | created_at
-- 1  | John Doe    | john@example.com   | 2025-10-19 ...
-- 2  | Jane Smith  | jane@example.com   | 2025-10-19 ...
-- 3  | Bob Johnson | bob@example.com    | 2025-10-19 ...

-- Test aggregation
SELECT COUNT(*) as total_records FROM iceberg.test_schema.test_table;

-- Expected output:
-- total_records
-- 3
```

### 3.5 Verify Parquet Files Created

```bash
# Check S3 for table files
aws s3 ls s3://iceberg-warehouse/test_schema/test_table/ \
  --endpoint-url http://localhost:9000 \
  --access-key minioadmin \
  --secret-key minioadmin123 \
  --recursive

# Expected output:
# metadata files and data files in various directories
```

## Phase 4: PostgreSQL Metadata Verification

### 4.1 Check Database Setup

```bash
# Connect to PostgreSQL
kubectl exec -it -n data-platform postgres-shared-xxx -- \
  psql -U iceberg_user -d iceberg_rest

# List tables
\dt

# Expected output shows Iceberg tables for table metadata
```

### 4.2 Verify Schema

```sql
-- Check iceberg_catalog schema
SELECT schema_name 
FROM information_schema.schemata 
WHERE schema_name = 'iceberg_catalog';

-- Check tables in public schema (if Iceberg creates them)
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE 'iceberg%';
```

## Phase 5: DataHub Integration Testing

### 5.1 Scale Up DataHub GMS

```bash
kubectl scale deployment -n data-platform datahub-gms --replicas=1

# Monitor startup
kubectl logs -f -n data-platform deployment/datahub-gms
```

### 5.2 Verify DataHub Health

```bash
# Port-forward to DataHub
kubectl port-forward -n data-platform svc/datahub-gms 8080:8080 &

# Check health endpoint
curl -s http://localhost:8080/health | jq .

# Expected response indicates service is UP
```

### 5.3 Run Iceberg Ingestion

```bash
# Apply ingestion recipe ConfigMap
kubectl apply -f k8s/datahub/iceberg-ingestion-recipe.yaml

# Run test ingestion job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: datahub-iceberg-test-ingestion
  namespace: data-platform
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: datahub-ingestion
        image: acryldata/datahub-ingestion:latest
        env:
        - name: DATAHUB_GMS_URL
          value: "http://datahub-gms:8080"
        volumeMounts:
        - name: recipe
          mountPath: /recipes
        command: ["/bin/sh"]
        args: ["-c", "datahub ingest -c /recipes/iceberg-recipe.yml --dry-run"]
      volumes:
      - name: recipe
        configMap:
          name: datahub-iceberg-recipe
EOF

# Monitor job
kubectl logs -f job/datahub-iceberg-test-ingestion
```

### 5.4 Verify Metadata in DataHub

```bash
# Port-forward to DataHub Frontend
kubectl port-forward -n data-platform svc/datahub-frontend 9002:9002 &

# Access at http://localhost:9002

# Navigate to:
# Explore → Datasets → Filter by "iceberg"

# Should see discovered Iceberg tables
```

## Phase 6: SeaTunnel Integration Testing

### 6.1 Scale Up SeaTunnel (if needed)

```bash
kubectl scale deployment -n data-platform seatunnel --replicas=1

# Monitor startup
kubectl logs -f deployment/seatunnel
```

### 6.2 Create Kafka Test Topic

```bash
# Create topic in Kafka
kubectl exec -it -n data-platform kafka-0 -- \
  /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic test-events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Verify topic
kubectl exec -it -n data-platform kafka-0 -- \
  /opt/kafka/bin/kafka-topics.sh \
  --list \
  --bootstrap-server localhost:9092 | grep test-events
```

### 6.3 Send Test Events to Kafka

```bash
# Port-forward to Kafka
kubectl port-forward -n data-platform svc/kafka-service 9092:9092 &

# Send test events
python3 << 'EOF'
import json
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send test events
for i in range(10):
    event = {
        "event_id": i,
        "event_type": "user_login" if i % 2 == 0 else "user_logout",
        "event_data": f"User event {i}",
        "event_timestamp": "2025-10-19T12:00:00Z",
        "user_id": i % 5
    }
    producer.send('test-events', event)
    print(f"Sent event {i}")

producer.flush()
print("All events sent!")
EOF
```

### 6.4 Deploy SeaTunnel Job

```bash
# Create Iceberg table for Kafka events
# (using Trino SQL from Phase 3)
CREATE TABLE iceberg.test_schema.kafka_events (
    event_id BIGINT,
    event_type VARCHAR,
    event_data VARCHAR,
    event_timestamp VARCHAR,
    user_id BIGINT
) WITH (format = 'PARQUET');

# Deploy SeaTunnel job ConfigMap
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: seatunnel-test-job
  namespace: data-platform
data:
  kafka-to-iceberg-test.conf: |
    env {
      execution.parallelism = 1
      job.mode = "STREAMING"
    }
    
    source {
      Kafka {
        bootstrap.servers = "kafka-service:9092"
        topic = "test-events"
        consumer.group = "seatunnel-test"
        result_table_name = "kafka_source"
        format = "json"
      }
    }
    
    sink {
      Iceberg {
        catalog_name = "rest"
        catalog_type = "rest"
        warehouse = "s3://iceberg-warehouse/"
        uri = "http://iceberg-rest-catalog:8181"
        database = "test_schema"
        table = "kafka_events"
        s3.endpoint = "http://minio-service:9000"
        s3.access-key-id = "minioadmin"
        s3.secret-access-key = "minioadmin123"
        s3.region = "us-east-1"
      }
    }
EOF

# Run the job
kubectl exec -it -n data-platform seatunnel-xxx -- \
  /opt/seatunnel/bin/seatunnel.sh \
  --config /etc/seatunnel/config/kafka-to-iceberg-test.conf
```

### 6.5 Verify SeaTunnel Data

```sql
-- In Trino:
SELECT * FROM iceberg.test_schema.kafka_events;

-- Should show the events sent from Kafka
```

## Phase 7: End-to-End Workflow Testing

### 7.1 Complete Data Flow Test

1. **Insert data via SeaTunnel** (Kafka → Iceberg)
2. **Query via Trino** (verify data)
3. **Discover via DataHub** (verify metadata)
4. **Monitor via Prometheus** (verify metrics)

```bash
# 1. Kafka → Iceberg (done in Phase 6)

# 2. Query in Trino
SELECT COUNT(*) as event_count FROM iceberg.test_schema.kafka_events;

# 3. Check DataHub metadata
curl -s http://localhost:8080/openapi/v2/entity/urn:li:dataset:urn%3Ali%3Aplatform%3Aiceberg%3Atest_schema.kafka_events | jq .

# 4. Check metrics
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
# Query: iceberg_rest_api_requests_total
```

### 7.2 Data Quality Testing

```sql
-- In Trino:

-- Check for duplicates
SELECT event_id, COUNT(*) as count
FROM iceberg.test_schema.kafka_events
GROUP BY event_id
HAVING COUNT(*) > 1;

-- Expected: No duplicates (or investigate if found)

-- Check for nulls
SELECT 
  COUNT(*) as total_records,
  SUM(CASE WHEN event_id IS NULL THEN 1 ELSE 0 END) as null_event_ids,
  SUM(CASE WHEN event_type IS NULL THEN 1 ELSE 0 END) as null_event_types
FROM iceberg.test_schema.kafka_events;

-- Check data freshness
SELECT 
  MAX(event_timestamp) as latest_event,
  MIN(event_timestamp) as oldest_event
FROM iceberg.test_schema.kafka_events;
```

### 7.3 Performance Testing

```sql
-- In Trino:

-- Test query performance
EXPLAIN ANALYZE
SELECT 
  event_type,
  COUNT(*) as count
FROM iceberg.test_schema.kafka_events
GROUP BY event_type;

-- Test with filtering
EXPLAIN ANALYZE
SELECT *
FROM iceberg.test_schema.kafka_events
WHERE event_type = 'user_login';
```

## Phase 8: Cleanup

```bash
# Clean up test resources
kubectl delete job datahub-iceberg-test-ingestion
kubectl delete job seatunnel-kafka-to-iceberg-test
kubectl delete configmap seatunnel-test-job

# Remove test data from Iceberg (optional)
# In Trino:
DROP TABLE iceberg.test_schema.test_table;
DROP TABLE iceberg.test_schema.kafka_events;
DROP SCHEMA iceberg.test_schema;

# Remove test Kafka topic (optional)
kubectl exec -it -n data-platform kafka-0 -- \
  /opt/kafka/bin/kafka-topics.sh \
  --delete \
  --topic test-events \
  --bootstrap-server localhost:9092
```

## Success Criteria

Integration is successful when:

1. ✅ Iceberg REST Catalog responds to API calls
2. ✅ MinIO buckets are created and accessible
3. ✅ Trino can create and query Iceberg tables
4. ✅ DataHub discovers Iceberg metadata
5. ✅ SeaTunnel can write data to Iceberg
6. ✅ End-to-end data flow works (source → Iceberg → query)
7. ✅ Data quality checks pass
8. ✅ Performance is acceptable

## Troubleshooting

### Iceberg Catalog Unreachable

```bash
# Test connectivity
curl -v http://iceberg-rest-catalog:8181/v1/config
kubectl port-forward svc/iceberg-rest-catalog 8181:8181
```

### Trino Can't Connect to Iceberg

```bash
# Check Trino logs
kubectl logs -f deployment/trino-coordinator | grep -i iceberg

# Verify catalog configuration
kubectl get configmap trino-catalogs -n data-platform -o yaml | grep -A 20 iceberg
```

### SeaTunnel Job Fails

```bash
# Check SeaTunnel logs
kubectl logs -f deployment/seatunnel | tail -100

# Verify MinIO credentials
aws s3 ls --endpoint-url http://minio-service:9000
```

### DataHub Not Discovering Iceberg Tables

```bash
# Check ingestion logs
kubectl logs job/datahub-iceberg-ingestion

# Verify DataHub can reach Iceberg
kubectl exec -it datahub-gms-xxx -- \
  curl http://iceberg-rest-catalog:8181/v1/config
```

## Next Steps

After successful testing:

1. Configure monitoring and alerting
2. Implement security hardening
3. Set up automated ingestion schedules
4. Create operational runbooks
5. Train team on Iceberg best practices
6. Plan data migration strategy
