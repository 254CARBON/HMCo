# Query Resource Governance

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Owner**: Platform Operations Team

## Overview

Resource governance prevents runaway queries from impacting platform SLOs. We use tiered resource pools with concurrency caps, memory limits, and timeouts.

## Trino Resource Groups

### Group Hierarchy

```
global (100 concurrent, 80% memory)
├── interactive (30 concurrent, 40% memory, 15min timeout)
│   └── Jupyter, Superset, ad-hoc queries
├── etl (20 concurrent, 50% memory, 2hr timeout)
│   └── DolphinScheduler, Airflow pipelines
├── batch (10 concurrent, 30% memory, 6hr timeout)
│   └── Spark, long-running analytics
└── admin (5 concurrent, 10% memory, no timeout)
    └── Platform administrators
```

### Configuration

Resource groups defined in: `helm/charts/data-platform/charts/trino/resource-groups.json`

Applied via ConfigMap mounted to Trino coordinator.

### Selectors

Queries automatically routed to groups based on:
- **User**: Admin users → admin group
- **Source**: Jupyter/Superset → interactive, Airflow → etl, Spark → batch
- **Client Tags**: Explicit `interactive`, `etl`, or `batch` tags
- **Default**: Interactive group

## ClickHouse Quotas

### Quota Tiers

| Tier | Queries/Hour | Max Memory | Max Time | Users |
|------|--------------|------------|----------|-------|
| **Interactive** | 1000 | 8 GB | 5 min | Analysts, dashboards |
| **ETL** | 500 | 32 GB | 2 hours | Pipelines |
| **Batch** | 100 | 64 GB | 4 hours | Long analytics |
| **Admin** | Unlimited | 128 GB | 24 hours | Admins |

### Settings Profiles

Each tier has:
- `max_threads`: Thread limit per query
- `max_memory_usage`: Memory cap per query
- `max_execution_time`: Query timeout
- `max_rows_to_read`: Scan limit
- `max_bytes_to_read`: Bytes read limit
- `readonly`: Write permission (0/1)

### Configuration

DDL: `clickhouse/ddl/quotas.sql`

Applied on ClickHouse cluster startup.

## Kill Switches

### Trino

Kill queries exceeding limits:
```sql
SELECT query_id, user, state, query
FROM system.runtime.queries
WHERE elapsed_time_millis > 900000;  -- 15 minutes

-- Kill query
CALL system.runtime.kill_query('query_id');
```

### ClickHouse

Kill queries exceeding limits:
```sql
SELECT query_id, user, elapsed, memory_usage, query
FROM system.processes
WHERE elapsed > 300;  -- 5 minutes

-- Kill query
KILL QUERY WHERE query_id = 'query_id';
```

## Synthetic Abuse Test

Automated test runs weekly to validate governance:

```python
# Synthetic abuse test
def abuse_test():
    # Test 1: High concurrency
    queries = [submit_query(f"SELECT count(*) FROM large_table WHERE id = {i}")
               for i in range(100)]
    
    # Verify throttling after 30 concurrent (interactive limit)
    assert len([q for q in queries if q.state == 'QUEUED']) > 0
    
    # Test 2: Memory hog
    query = submit_query("SELECT * FROM large_table ORDER BY random()")
    
    # Verify killed after exceeding memory limit
    assert query.state in ['FAILED', 'KILLED']
    assert 'memory limit exceeded' in query.error_message
    
    # Test 3: Long-running query
    query = submit_query("SELECT sleep(1000)")
    
    # Verify killed after timeout
    assert query.state in ['FAILED', 'KILLED']
    assert 'timeout' in query.error_message

# Run weekly via cron
if __name__ == '__main__':
    abuse_test()
    print("Abuse test passed - governance working correctly")
```

## Monitoring

### Metrics

- `trino_resource_group_queued_queries`: Queued queries per group
- `trino_resource_group_running_queries`: Running queries per group
- `clickhouse_quota_usage`: Quota usage by user/profile
- `query_killed_total`: Total queries killed

### Alerts

- **ResourceGroupSaturated**: Queued queries > 50 for 10 minutes
- **QuotaExceeded**: User exceeded quota
- **QueryKillRateHigh**: > 10 queries killed per hour

## SLO Validation

During abuse test, SLOs must remain unaffected:

- **p95 interactive latency** < 500ms
- **p99 ETL latency** < 5s
- **Dashboard availability** > 99.9%

## User Communication

When user exceeds quota:
1. Query rejected with clear error message
2. Guidance on optimization
3. Instructions to request tier upgrade

Example error:
```
Query failed: Quota exceeded for user john@254carbon.com
Current tier: interactive (1000 queries/hour)
Used: 1000/1000 queries in last hour
Upgrade to ETL tier: contact #data-platform on Slack
```

## Tier Upgrade Process

1. User submits request via Slack (#data-platform)
2. Platform team reviews use case
3. If approved, add user to appropriate role:
   ```sql
   -- Trino
   GRANT interactive_users TO USER 'user@254carbon.com';
   
   -- ClickHouse
   ALTER USER 'user@254carbon.com' SETTINGS PROFILE etl_profile;
   ALTER USER 'user@254carbon.com' QUOTA etl_quota;
   ```

## Best Practices

1. **Use Client Tags**: Tag queries with `interactive`/`etl`/`batch` for correct routing
2. **Optimize Queries**: Add filters, limit scans, use materialized views
3. **Batch Processing**: Use batch tier for long-running analytics
4. **Monitor Usage**: Check quota usage via dashboards
5. **Request Tier Upgrade**: Don't circumvent limits, request appropriate tier
