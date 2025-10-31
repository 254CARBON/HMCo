# ClickHouse Data Retention Policies

## Overview

This document describes the data retention policies and TTL (Time To Live) configuration for ClickHouse tables in the 254Carbon data platform.

## Retention Policy Configuration

### Default Retention Policies

The following retention policies are configured in the ClickHouse deployment:

| Policy Name | Retention Period | Description |
|------------|------------------|-------------|
| `raw_data` | 30 days | Raw event data and logs |
| `aggregated_data` | 365 days (1 year) | Pre-aggregated metrics and reports |
| `archive_data` | 2555 days (7 years) | Long-term archived data for compliance |

### TTL Configuration

TTL (Time To Live) is configured at the table level in ClickHouse. Data older than the specified TTL is automatically deleted during merge operations.

## SQL Examples

### 1. Creating a Table with TTL

#### Example 1: Events table with 30-day retention

```sql
CREATE TABLE analytics.events (
    event_id UUID,
    timestamp DateTime,
    user_id UInt64,
    event_type String,
    properties String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id)
TTL timestamp + INTERVAL 30 DAY;
```

**Explanation:**
- Data is automatically deleted 30 days after the `timestamp` value
- Partitioning by month improves TTL performance
- TTL is checked during merge operations

#### Example 2: Metrics table with 1-year retention

```sql
CREATE TABLE analytics.daily_metrics (
    metric_date Date,
    metric_name String,
    metric_value Float64,
    dimensions Map(String, String),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(metric_date)
ORDER BY (metric_date, metric_name)
TTL metric_date + INTERVAL 365 DAY;
```

**Explanation:**
- Aggregated metrics are kept for 1 year
- Date-based partitioning aligns with TTL policy
- Map type stores flexible dimensional data

#### Example 3: Audit logs with 7-year retention (compliance)

```sql
CREATE TABLE security.audit_logs (
    log_id UUID,
    timestamp DateTime64(3),
    user_id String,
    action String,
    resource String,
    ip_address String,
    status String,
    details String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id)
TTL timestamp + INTERVAL 2555 DAY;
```

**Explanation:**
- 7-year retention for compliance requirements (e.g., GDPR, SOX)
- DateTime64 for millisecond precision
- Comprehensive audit trail

### 2. Sample Tables Demonstrating TTLs

#### Sample Table 1: Page Views with TTL

```sql
-- Create the table
CREATE TABLE analytics.page_views (
    view_id UUID,
    timestamp DateTime,
    user_id UInt64,
    page_url String,
    referrer String,
    session_id String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id)
TTL timestamp + INTERVAL 30 DAY;

-- Insert sample data
INSERT INTO analytics.page_views (view_id, timestamp, user_id, page_url, referrer, session_id)
VALUES
    (generateUUIDv4(), now() - INTERVAL 10 DAY, 1001, '/home', 'https://google.com', 'sess_123'),
    (generateUUIDv4(), now() - INTERVAL 5 DAY, 1002, '/products', '/home', 'sess_124'),
    (generateUUIDv4(), now() - INTERVAL 1 DAY, 1003, '/checkout', '/products', 'sess_125');

-- Query to verify TTL settings
SELECT 
    table,
    engine_full,
    partition_key,
    sorting_key,
    ttl_expression
FROM system.tables
WHERE database = 'analytics' AND table = 'page_views';
```

#### Sample Table 2: Aggregated Reports with Different TTL

```sql
-- Create aggregated metrics table
CREATE TABLE analytics.hourly_reports (
    report_hour DateTime,
    metric_name String,
    total_count UInt64,
    avg_value Float64,
    max_value Float64,
    min_value Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(report_hour)
ORDER BY (report_hour, metric_name)
TTL report_hour + INTERVAL 365 DAY;

-- Insert sample aggregated data
INSERT INTO analytics.hourly_reports (report_hour, metric_name, total_count, avg_value, max_value, min_value)
VALUES
    (toStartOfHour(now() - INTERVAL 7 DAY), 'page_views', 15000, 125.5, 500, 10),
    (toStartOfHour(now() - INTERVAL 3 DAY), 'api_calls', 50000, 1250.75, 5000, 50),
    (toStartOfHour(now() - INTERVAL 1 DAY), 'conversions', 500, 42.3, 150, 5);

-- Query retention information
SELECT
    formatReadableSize(sum(bytes_on_disk)) AS data_size,
    count() AS total_rows,
    min(report_hour) AS oldest_data,
    max(report_hour) AS newest_data,
    date_diff('day', oldest_data, newest_data) AS retention_days
FROM analytics.hourly_reports;
```

### 3. Modifying TTL on Existing Tables

#### Add TTL to an existing table

```sql
-- Add TTL to existing table
ALTER TABLE analytics.events 
MODIFY TTL timestamp + INTERVAL 30 DAY;
```

#### Change TTL period

```sql
-- Update TTL to 60 days
ALTER TABLE analytics.events 
MODIFY TTL timestamp + INTERVAL 60 DAY;
```

#### Remove TTL

```sql
-- Remove TTL from table
ALTER TABLE analytics.events 
REMOVE TTL;
```

### 4. Advanced TTL Patterns

#### Example 1: Multi-level TTL (hot/warm/cold storage)

```sql
CREATE TABLE analytics.time_series_data (
    timestamp DateTime,
    sensor_id String,
    value Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, sensor_id)
TTL 
    timestamp + INTERVAL 7 DAY TO DISK 'hot',
    timestamp + INTERVAL 30 DAY TO DISK 'warm',
    timestamp + INTERVAL 90 DAY TO DISK 'cold',
    timestamp + INTERVAL 365 DAY;
```

**Explanation:**
- Data moves to slower storage tiers as it ages
- Hot storage (SSD): 0-7 days
- Warm storage (HDD): 7-30 days
- Cold storage (Archive): 30-90 days
- Deleted: After 365 days

#### Example 2: Column-level TTL

```sql
CREATE TABLE analytics.user_events (
    event_id UUID,
    timestamp DateTime,
    user_id UInt64,
    event_data String TTL timestamp + INTERVAL 30 DAY,  -- PII deleted after 30 days
    event_summary String  -- Summary retained longer
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id)
TTL timestamp + INTERVAL 365 DAY;  -- Entire row deleted after 1 year
```

**Explanation:**
- Sensitive PII data (`event_data`) deleted after 30 days
- Summary data (`event_summary`) retained for full year
- Useful for GDPR compliance

#### Example 3: Conditional TTL with DELETE/TO DISK

```sql
CREATE TABLE analytics.customer_data (
    customer_id UInt64,
    last_activity DateTime,
    data_type String,
    data_value String
) ENGINE = MergeTree()
ORDER BY (customer_id, last_activity)
TTL 
    last_activity + INTERVAL 30 DAY DELETE WHERE data_type = 'temp',
    last_activity + INTERVAL 365 DAY DELETE WHERE data_type = 'cached',
    last_activity + INTERVAL 2555 DAY;  -- Full delete after 7 years
```

## Checking TTL Status

### 1. View TTL Configuration

```sql
-- Show TTL settings for all tables
SELECT 
    database,
    table,
    engine,
    ttl_expression,
    ttl_type
FROM system.tables
WHERE database = 'analytics'
  AND ttl_expression != '';
```

### 2. Monitor TTL Cleanup Progress

```sql
-- Check merge queue for TTL operations
SELECT
    database,
    table,
    elapsed,
    progress,
    is_mutation,
    merge_type
FROM system.merges
WHERE merge_type LIKE '%TTL%';
```

### 3. Verify Data Age Distribution

```sql
-- Analyze data age distribution
SELECT
    toYYYYMM(timestamp) AS month,
    count() AS row_count,
    formatReadableSize(sum(bytes_on_disk)) AS size_on_disk
FROM analytics.events
GROUP BY month
ORDER BY month DESC
LIMIT 12;
```

## TTL Best Practices

### 1. Partition Alignment

Always align TTL with partition key for optimal performance:

```sql
-- Good: TTL aligns with monthly partitioning
CREATE TABLE events (...) 
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 30 DAY;

-- Avoid: Misaligned TTL and partitioning reduces efficiency
```

### 2. Avoid Excessive Granularity

Use reasonable TTL intervals (days, not seconds):

```sql
-- Good
TTL timestamp + INTERVAL 30 DAY

-- Avoid: Too granular
TTL timestamp + INTERVAL 30 * 24 * 3600 SECOND
```

### 3. Monitor TTL Performance

```sql
-- Check TTL merge performance
SELECT
    table,
    count() AS merge_count,
    sum(elapsed) AS total_time,
    avg(elapsed) AS avg_time
FROM system.part_log
WHERE event_type = 'MergeParts'
  AND merge_reason = 'TTLMerge'
  AND event_date >= today() - 7
GROUP BY table;
```

### 4. Test TTL Before Production

```sql
-- Create test table with short TTL
CREATE TABLE test.ttl_test (
    id UInt64,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY id
TTL timestamp + INTERVAL 1 MINUTE;

-- Insert test data
INSERT INTO test.ttl_test VALUES (1, now() - INTERVAL 2 MINUTE);

-- Force merge to trigger TTL
OPTIMIZE TABLE test.ttl_test FINAL;

-- Verify deletion
SELECT count() FROM test.ttl_test;  -- Should be 0
```

## Encrypted Volumes Confirmation

### Volume Encryption Status

To confirm that ClickHouse data is stored on encrypted volumes:

```sql
-- Check storage configuration
SELECT
    name,
    path,
    disk_type,
    total_space,
    free_space
FROM system.disks;

-- Verify encryption settings
SELECT
    name,
    value,
    changed,
    description
FROM system.settings
WHERE name LIKE '%encrypt%' OR name LIKE '%disk%';
```

### Kubernetes Volume Verification

```bash
# Check PersistentVolume encryption status
kubectl get pv -n data-platform -o json | \
  jq '.items[] | select(.spec.claimRef.name | contains("clickhouse")) | {
    name: .metadata.name,
    storageClass: .spec.storageClassName,
    annotations: .metadata.annotations
  }'

# Check PersistentVolumeClaim
kubectl get pvc -n data-platform clickhouse-data -o yaml | \
  grep -E "storageClassName|annotations"

# Verify encrypted storage class
kubectl get storageclass encrypted-local-path -o yaml
```

### Expected Output for Encrypted Volumes

When properly configured, you should see:

```yaml
# Storage class with encryption
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: encrypted-local-path
parameters:
  encrypted: "true"
  # For AWS: encrypted: "true", kmsKeyId: "arn:aws:kms:..."
  # For GCP: diskEncryption: "customer-managed", diskEncryptionKey: "..."
  # For Azure: encrypted: "true", diskEncryptionSetID: "..."
provisioner: rancher.io/local-path
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
```

## Compliance and Governance

### Data Retention Requirements

| Data Type | Retention Period | Regulation | TTL Configuration |
|-----------|-----------------|------------|-------------------|
| Audit Logs | 7 years | SOX, GDPR | `TTL + INTERVAL 2555 DAY` |
| Financial Records | 7 years | SOX, GAAP | `TTL + INTERVAL 2555 DAY` |
| User Activity | 2 years | GDPR | `TTL + INTERVAL 730 DAY` |
| PII Data | 30 days (anonymized) | GDPR | `TTL + INTERVAL 30 DAY` |
| Temporary Data | 7 days | Internal Policy | `TTL + INTERVAL 7 DAY` |

### Audit Trail

All TTL operations are logged in ClickHouse system tables:

```sql
-- Query TTL-related operations from audit log
SELECT
    event_time,
    user,
    query_kind,
    query,
    tables
FROM system.query_log
WHERE query LIKE '%TTL%'
  AND event_time >= now() - INTERVAL 7 DAY
ORDER BY event_time DESC
LIMIT 100;
```

## Troubleshooting

### TTL Not Working

1. **Check TTL Expression**
   ```sql
   SELECT ttl_expression FROM system.tables WHERE table = 'your_table';
   ```

2. **Force TTL Merge**
   ```sql
   OPTIMIZE TABLE your_table FINAL;
   ```

3. **Check Merge Settings**
   ```sql
   SELECT * FROM system.merge_tree_settings WHERE name LIKE '%ttl%';
   ```

### Performance Issues

1. **Monitor Merge Load**
   ```sql
   SELECT * FROM system.merges;
   ```

2. **Adjust Merge Threads**
   ```sql
   SET max_threads = 8;
   OPTIMIZE TABLE your_table;
   ```

## References

- [ClickHouse TTL Documentation](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree#table_engine-mergetree-ttl)
- [Data Retention Best Practices](https://clickhouse.com/docs/en/guides/best-practices/)
- [Storage Configuration](https://clickhouse.com/docs/en/operations/storing-data)

## Support

For questions about retention policies:
1. Review this documentation
2. Check ClickHouse system tables for current configuration
3. Contact the data platform team
4. Escalate to infrastructure team if needed

---

**Last Updated**: October 31, 2025  
**Next Review**: November 30, 2025
