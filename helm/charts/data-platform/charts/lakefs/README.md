# lakeFS - Data Versioning for Data Lakes

lakeFS brings Git-like version control to data lakes, enabling safe data releases, diffs, and rollbacks.

## Features

- **Branch/Merge/Rollback**: Version your data just like code
- **Data PRs**: Review data changes before merging to production
- **Atomic Operations**: ACID guarantees for data operations
- **Zero-Copy Branching**: Efficient branching without data duplication
- **Hooks**: Pre-commit and pre-merge hooks for data quality checks

## Quick Start

### 1. Enable in values.yaml

```yaml
data-platform:
  lakefs:
    enabled: true
```

### 2. Configure Database

lakeFS requires a PostgreSQL database:

```yaml
lakefs:
  database:
    connectionString: "postgresql://user:pass@postgres:5432/lakefs"
```

### 3. Configure Block Storage

Point lakeFS to your MinIO/S3 storage:

```yaml
lakefs:
  blockstore:
    type: s3
    s3:
      endpoint: "http://minio-service:9000"
      bucket: "lakefs-data"
```

## Usage Workflow

### 1. Create Development Branch

```bash
lakectl branch create \
  lakefs://hmco-curated@dev \
  --source lakefs://hmco-curated@main
```

### 2. Write Data to Dev Branch

```python
# In your ingestion code
spark.write.format("iceberg") \
  .option("path", "lakefs://hmco-curated@dev/trading/trades") \
  .save()
```

### 3. Run Data Quality Checks

```bash
# Great Expectations validation
great_expectations checkpoint run trading_trades_dq \
  --data-source lakefs://hmco-curated@dev/trading/trades
```

### 4. Open Data PR

```bash
lakectl merge \
  --from lakefs://hmco-curated@dev \
  --to lakefs://hmco-curated@prod \
  --dry-run  # Preview changes first
```

### 5. Merge to Production

If quality checks pass:

```bash
lakectl merge \
  --from lakefs://hmco-curated@dev \
  --to lakefs://hmco-curated@prod
```

## Auto-Merge Policy

Configure automatic merging when quality checks pass:

```yaml
lakefs:
  policies:
    autoMerge:
      enabled: true
      sourceBranch: dev
      targetBranch: stage
      requireChecks:
        - great-expectations-validation
```

## Rollback on Error

If bad data reaches production:

```bash
# Revert to previous commit
lakectl revert lakefs://hmco-curated@prod \
  --commit <previous-commit-id>
```

## Integration with Ingestion

Update your UIS ingestion specs to write to lakeFS branches:

```yaml
output:
  path: "lakefs://hmco-curated@dev/iso/lmp_prices/"
  format: "iceberg"
```

## Monitoring

Monitor lakeFS operations in Grafana:
- Branch operations per hour
- Merge success/failure rate
- Storage usage by branch

## References

- [lakeFS Documentation](https://docs.lakefs.io/)
- [lakeFS + Iceberg](https://docs.lakefs.io/integrations/iceberg.html)
