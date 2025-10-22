# Apache Deequ Data Quality Framework

## Overview

Apache Deequ is integrated into the 254Carbon platform for comprehensive data quality management:

- **Constraint Validation**: Completeness, Uniqueness, PatternMatch, Referential Integrity
- **Statistical Profiling**: Column-level statistics and distribution analysis
- **Anomaly Detection**: Automatic detection using IQR or Z-Score methods
- **Quality Metrics Export**: Results to Iceberg, Kafka, PostgreSQL
- **Scheduled Execution**: Daily checks, weekly profiling, 6-hourly anomaly detection
- **Alert Integration**: Prometheus metrics and Kafka topics for alerts

## Architecture

```
Quality Data Sources (Iceberg/Kafka)
         ↓
Deequ CronJobs (Daily/Weekly/Every 6h)
         ↓
Spark Operator (Submits SparkApplications)
         ↓
Quality Check Jobs
    ├── Constraint Validation
    ├── Statistical Profiling
    └── Anomaly Detection
         ↓
Metrics Export
    ├── Iceberg Tables
    │   ├── deequ_quality_checks
    │   ├── deequ_profiles
    │   └── deequ_anomalies
    ├── Kafka Topics
    │   └── data-quality-alerts
    └── PostgreSQL
         ↓
Monitoring & Alerting
    ├── Prometheus Metrics
    ├── Grafana Dashboards
    └── Alert Manager
```

## Components

### 1. Configuration (`deequ-configmap.yaml`)

- **constraints.json**: Quality check definitions for each table
- **profiling-config.yaml**: Statistical profiling settings
- **anomaly-detection.yaml**: Anomaly detection thresholds
- **export-config.yaml**: Metrics export destinations

### 2. RBAC (`deequ-rbac.yaml`)

ServiceAccount: `deequ` with permissions to:
- Create/manage SparkApplications
- Access Iceberg catalogs
- Write to Kafka topics
- Connect to PostgreSQL

### 3. CronJobs (`quality-check-cronjob.yaml`)

Three scheduled jobs:

1. **Daily Checks** (2 AM UTC)
   - Validates constraints on all tables
   - Compares against thresholds
   - Exports results to Iceberg

2. **Weekly Profiling** (Sunday 1 AM UTC)
   - Comprehensive statistical analysis
   - Updates baseline metrics
   - Stores profiles for anomaly detection

3. **Anomaly Detection** (Every 6 hours)
   - Compares current metrics against profiles
   - Detects outliers using IQR/Z-Score
   - Publishes alerts to Kafka

### 4. Metrics Storage (`quality-metrics-schema.sql`)

Four Iceberg tables:

- **deequ_quality_checks**: Individual check results
- **deequ_profiles**: Statistical profiles
- **deequ_anomalies**: Detected anomalies
- **deequ_quality_history**: Historical trend data

## Quick Start

### 1. Define Quality Constraints

Edit `deequ-configmap.yaml`, section `constraints.json`:

```json
{
  "tables": {
    "raw.customers": {
      "checks": [
        {
          "name": "completeness_id",
          "type": "Completeness",
          "column": "id",
          "threshold": 1.0,
          "description": "All customer IDs must be non-null"
        },
        {
          "name": "uniqueness_id",
          "type": "Uniqueness",
          "column": "id",
          "threshold": 1.0
        }
      ]
    }
  }
}
```

### 2. Deploy Deequ Infrastructure

```bash
# Deploy RBAC
kubectl apply -f k8s/compute/deequ/deequ-rbac.yaml

# Deploy Configuration
kubectl apply -f k8s/compute/deequ/deequ-configmap.yaml

# Deploy CronJobs
kubectl apply -f k8s/compute/deequ/quality-check-cronjob.yaml
```

### 3. Initialize Iceberg Tables

```bash
# Execute SQL in Trino/Spark
kubectl exec -it -n data-platform <trino-pod> -- \
  trino --execute-from-file k8s/data-lake/quality-metrics-schema.sql
```

### 4. Monitor Execution

```bash
# Check CronJob status
kubectl get cronjobs -n data-platform -l app=deequ

# View created jobs
kubectl get jobs -n data-platform -l app=deequ

# Check quality check results
kubectl exec -n data-platform <trino-pod> -- \
  trino --execute "SELECT * FROM monitoring.deequ_quality_checks WHERE check_date = CURRENT_DATE"
```

## Quality Check Types

### Completeness

Ensures column has no null values:

```json
{
  "name": "completeness_email",
  "type": "Completeness",
  "column": "email",
  "threshold": 0.95
}
```

### Uniqueness

Ensures all values are unique:

```json
{
  "name": "uniqueness_id",
  "type": "Uniqueness",
  "column": "id",
  "threshold": 1.0
}
```

### PatternMatch

Ensures values match regex pattern:

```json
{
  "name": "pattern_email",
  "type": "PatternMatch",
  "column": "email",
  "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
  "threshold": 0.99
}
```

### Referential Integrity

Ensures foreign key references exist:

```json
{
  "name": "correlation_customer_order",
  "type": "ReferentialIntegrity",
  "column": "customer_id",
  "reference_table": "raw.customers",
  "reference_column": "id",
  "threshold": 0.995
}
```

### Freshness

Ensures data is not older than threshold:

```json
{
  "name": "freshness_data",
  "type": "Freshness",
  "column": "created_at",
  "max_age_days": 10,
  "threshold": 1.0
}
```

## Anomaly Detection

### IQR Method (Default)

Detects values outside 1.5 × IQR from quartiles:

```yaml
# In anomaly-detection.yaml
anomaly_detection:
  method: iqr
  iqr:
    lower_multiplier: 1.5
    upper_multiplier: 1.5
```

### Z-Score Method

Detects values more than 3 standard deviations away:

```yaml
anomaly_detection:
  method: zscore
  zscore:
    threshold: 3.0
```

## Query Quality Results

### All Failed Checks (Last 24 Hours)

```sql
SELECT 
  table_name, 
  check_name, 
  actual_value, 
  threshold_value,
  error_message
FROM monitoring.deequ_quality_checks
WHERE status = 'FAILED' 
  AND check_timestamp >= NOW() - INTERVAL '24' HOUR
ORDER BY check_timestamp DESC;
```

### Quality Score Trend

```sql
SELECT 
  check_date,
  table_name,
  ROUND(AVG(actual_value), 2) as avg_quality_score,
  MIN(actual_value) as min_score,
  MAX(actual_value) as max_score
FROM monitoring.deequ_quality_checks
WHERE check_date >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY check_date, table_name
ORDER BY check_date DESC, table_name;
```

### Detected Anomalies

```sql
SELECT 
  table_name,
  column_name,
  anomaly_type,
  severity,
  current_value,
  baseline_value,
  relative_deviation as deviation_percent,
  anomaly_timestamp
FROM monitoring.deequ_anomalies
WHERE anomaly_date >= CURRENT_DATE - INTERVAL '7' DAY
  AND investigation_status != 'FALSE_POSITIVE'
ORDER BY severity DESC, anomaly_timestamp DESC;
```

## Integration with Other Systems

### MLFlow Tracking

Quality check results are automatically logged to MLFlow:

```python
from services.spark_mlflow_client import SparkMLFlowClient

client = SparkMLFlowClient(experiment_name="data-quality")
run_id = client.start_spark_run("daily-quality-check")

# After quality checks...
client.log_quality_metrics(quality_results, "customers_table")

client.end_run(status="FINISHED")
```

### Kafka Alerts

Quality failures trigger alerts on Kafka topic `data-quality-alerts`:

```json
{
  "table_name": "raw.customers",
  "check_name": "completeness_email",
  "status": "FAILED",
  "severity": "CRITICAL",
  "actual_value": 0.85,
  "threshold": 0.95,
  "timestamp": "2025-10-20T02:00:00Z"
}
```

### Prometheus Metrics

Query in Prometheus:

```
# Quality score by table
deequ_quality_score{table_name="raw.customers"}

# Failed checks count
increase(deequ_check_failures_total[1h])

# Anomaly detection
deequ_anomaly_detected{table_name="raw.events"}
```

## Monitoring & Troubleshooting

### Check Quality Job Status

```bash
# List all quality jobs
kubectl get sparkapplications -n data-platform -l component=quality

# View specific job logs
kubectl logs -n data-platform <job-driver-pod> -f

# Get job details
kubectl describe sparkapplication -n data-platform <job-name>
```

### View CronJob History

```bash
# Check CronJob status
kubectl describe cronjob -n data-platform deequ-daily-checks

# View triggered jobs
kubectl get jobs -n data-platform -l cronjob=deequ-daily-checks

# View job logs
kubectl logs -n data-platform job/<job-name> -f
```

### Grafana Dashboards

Access dashboards at `https://grafana.254carbon.com`:

- **Data Quality Overview**: Quality scores by table
- **Quality Trends**: Historical quality trends
- **Anomaly Detection**: Recent anomalies and their status
- **Check Execution**: Failed checks timeline

## Best Practices

### 1. Threshold Settings

- **Completeness**: Usually 0.95-1.0 (95-100%)
- **Uniqueness**: 1.0 for primary keys, 0.99+ for unique indexes
- **Pattern**: 0.95-0.99 for format validation
- **Referential**: 0.995-1.0 for foreign keys

### 2. Profiling Schedule

- **Daily**: Run checks during off-peak hours (2 AM)
- **Weekly**: Run profiling on Sunday for baseline updates
- **Anomaly**: Every 6 hours for real-time monitoring

### 3. Alert Response

- **Critical (< 70%)**: Immediate investigation required
- **Warning (70-80%)**: Scheduled investigation
- **Info**: Log only, no action needed

### 4. Constraint Maintenance

- Review constraints monthly
- Adjust thresholds based on data trends
- Add new checks as data evolves

## Common Issues

### Quality Checks Not Running

```bash
# Check CronJob is active
kubectl get cronjob -n data-platform deequ-daily-checks

# Check for job failures
kubectl get jobs -n data-platform --sort-by='.status.completionTime'

# View job logs for errors
kubectl logs -n data-platform job/<job-name> -f
```

### Metrics Not Appearing in Iceberg

1. Verify Iceberg tables exist:
```sql
SHOW TABLES IN monitoring;
```

2. Check table write permissions
3. Verify S3/MinIO connectivity
4. Check job logs for export errors

### False Positive Anomalies

1. Mark as `FALSE_POSITIVE` in `deequ_anomalies`
2. Update baselines if data pattern changed
3. Adjust anomaly detection thresholds

## Advanced Configuration

### Custom Quality Rules

Add Scala/Python implementations in Spark job:

```python
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite

check = Check(spark, CheckLevel.Warning, "Custom Checks")

suite = VerificationSuite(spark) \
    .onData(df) \
    .addCheck(check \
        .hasMinLength("name", 2) \
        .hasMaxLength("email", 255) \
        .isNonNegative("age")
    ) \
    .run()
```

### Custom Anomaly Methods

Implement custom detection logic in Python Spark job

### Multi-Table Validation

Check referential integrity across multiple tables:

```python
suite = VerificationSuite(spark) \
    .onData(df_orders) \
    .addCheck(check \
        .uniqueValueRatio("customer_id", df_customers, "id")
    ) \
    .run()
```

## Resources

- **Deequ GitHub**: https://github.com/awslabs/deequ
- **Deequ Examples**: https://github.com/awslabs/deequ/tree/master/examples
- **PySpark Deequ**: https://github.com/Zarfal/deequ

## Support

Issues or questions:

1. Check CronJob status and job logs
2. Verify ConfigMap constraints are valid
3. Check Iceberg tables exist and are writable
4. Review Prometheus/Grafana for metrics
5. Check recent PRs/docs for similar issues
