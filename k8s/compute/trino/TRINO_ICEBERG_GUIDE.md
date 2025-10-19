# Trino Iceberg Integration Guide

## Overview

Trino provides distributed SQL query engine access to Iceberg tables. This guide covers Trino configuration and usage with Iceberg REST Catalog.

## Configuration

### Iceberg Catalog Properties

```properties
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=http://iceberg-rest-catalog:8181
iceberg.rest-catalog.warehouse=s3://iceberg-warehouse/
iceberg.rest-catalog.security=none
```

### S3/MinIO Configuration

```properties
s3.endpoint=http://minio-service:9000
s3.access-key=minioadmin
s3.secret-key=minioadmin123
s3.path-style-access=true
s3.region=us-east-1
```

### Performance Tuning

```properties
# Query optimization
iceberg.pushdown-projection-enabled=true
iceberg.pushdown-filters-enabled=true
iceberg.split-manager-threads=4
iceberg.max-split-size=134217728

# Connection pooling
iceberg.max-connections=20
```

## Usage Examples

### Connect to Trino

```bash
# Port-forward to Trino coordinator
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080 &

# Connect with trino-cli or similar tool
trino --server http://localhost:8080
```

### List Catalogs and Namespaces

```sql
-- Show all catalogs
SHOW CATALOGS;

-- Show Iceberg namespaces (databases)
SHOW SCHEMAS FROM iceberg;

-- Show tables in namespace
SHOW TABLES FROM iceberg.default;
```

### Create Iceberg Table

```sql
-- Create namespace first
CREATE SCHEMA iceberg.my_database;

-- Create table
CREATE TABLE iceberg.my_database.customers (
    customer_id BIGINT,
    name VARCHAR,
    email VARCHAR,
    created_at TIMESTAMP(3) WITH TIME ZONE,
    updated_at TIMESTAMP(3) WITH TIME ZONE
)
WITH (
    format = 'PARQUET',
    location = 's3://iceberg-warehouse/my_database/customers',
    partitioning = ARRAY['year(created_at)']
);
```

### Insert Data

```sql
INSERT INTO iceberg.my_database.customers
VALUES 
    (1, 'John Doe', 'john@example.com', now(), now()),
    (2, 'Jane Smith', 'jane@example.com', now(), now());
```

### Query Data

```sql
-- Simple query
SELECT * FROM iceberg.my_database.customers;

-- Query with filtering
SELECT customer_id, name
FROM iceberg.my_database.customers
WHERE created_at > CURRENT_DATE - INTERVAL '30' DAY;

-- Aggregation
SELECT 
    year(created_at) AS year,
    COUNT(*) AS count
FROM iceberg.my_database.customers
GROUP BY 1
ORDER BY 1 DESC;
```

### Iceberg-Specific Queries

```sql
-- View table metadata
SELECT * FROM iceberg.my_database."customers$metadata";

-- View snapshots
SELECT * FROM iceberg.my_database."customers$snapshots";

-- View partitions
SELECT * FROM iceberg.my_database."customers$partitions";

-- View files
SELECT * FROM iceberg.my_database."customers$files";

-- View manifests
SELECT * FROM iceberg.my_database."customers$manifests";
```

### Time Travel

```sql
-- Query historical data as of specific timestamp
SELECT * FROM iceberg.my_database.customers
FOR VERSION AS OF TIMESTAMP '2025-10-19 10:00:00';

-- Query specific snapshot
SELECT * FROM iceberg.my_database.customers
FOR VERSION AS OF 1234567890;
```

### Table Maintenance

```sql
-- Compact small files
CALL iceberg.system.compact_table('my_database.customers', '1MB');

-- Expire snapshots older than 7 days
CALL iceberg.system.expire_snapshots('my_database.customers', INTERVAL '7' DAY);

-- Remove orphan files
CALL iceberg.system.remove_orphan_files('my_database.customers');
```

## Performance Considerations

### Partitioning Strategy

Use partitioning for large tables:
```sql
CREATE TABLE iceberg.my_database.events (
    event_id BIGINT,
    event_type VARCHAR,
    event_timestamp TIMESTAMP(3),
    user_id BIGINT
)
WITH (
    partitioning = ARRAY['year(event_timestamp)', 'month(event_timestamp)', 'event_type']
);
```

### File Format

Use Parquet for best performance:
```sql
CREATE TABLE ... WITH (format = 'PARQUET', ...);
```

### Projection Pushdown

Enable projection pushdown to improve performance:
```properties
iceberg.pushdown-projection-enabled=true
```

### Filter Pushdown

Enable filter pushdown to reduce data scanned:
```properties
iceberg.pushdown-filters-enabled=true
```

## Troubleshooting

### Connection Issues

Check Trino catalog configuration:
```sql
SELECT * FROM system.metadata.catalogs;
```

### S3/MinIO Access Issues

Test MinIO connectivity:
```bash
curl -v http://minio-service:9000/minio/health/live
```

### Query Slow

1. Check partitioning strategy
2. Enable pushdown optimization
3. Review partition pruning
4. Check table statistics

### Large File Issues

Compact small files:
```sql
CALL iceberg.system.compact_table('schema.table', '128MB');
```

## Best Practices

1. **Use Partitioning**: Partition large tables for faster queries
2. **Regular Maintenance**: Expire old snapshots and remove orphan files
3. **Monitor Performance**: Use EXPLAIN to understand query plans
4. **Test Queries**: Validate queries before running in production
5. **Version Control**: Document table schemas and transformations

## Reference

- [Trino Iceberg Connector](https://trino.io/docs/current/connector/iceberg.html)
- [Apache Iceberg REST API](https://iceberg.apache.org/rest-catalog-spec/)
- [Iceberg System Tables](https://iceberg.apache.org/docs/latest/spark-queries/#system-tables)
