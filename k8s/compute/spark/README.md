# Apache Spark Integration Guide

## Overview

Apache Spark is natively integrated into the 254Carbon data platform, providing:

- **Distributed ETL**: Kafka → Iceberg, MySQL/PostgreSQL CDC → Iceberg, custom transformations
- **Unified Analytics**: SQL queries on Iceberg tables with full ACID support
- **Data Transformations**: Complex data enrichment and aggregation pipelines
- **Quality Integration**: Deequ-based data quality validation
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **ML Integration**: Automatic tracking to MLFlow experiment server
- **Orchestration**: Scheduled jobs via DolphinScheduler CronJobs

## Architecture

```
DolphinScheduler / Manual Submission
         ↓
    Spark Operator
         ↓
SparkApplications (CRDs)
    ├── Driver Pod
    └── Executor Pods
         ↓
┌─────────────────────────┐
│  Iceberg Data Lake      │
│  (MinIO + REST Catalog) │
└─────────────────────────┘
    ├── Raw schemas
    ├── ODS schemas
    └── DW schemas
         ↓
┌──────────────────────────┐
│  Analytics & Reporting   │
│  (Trino, Doris, etc)     │
└──────────────────────────┘
```

## Components

### 1. Spark Operator (`k8s/compute/spark/spark-operator.yaml`)

Watches for `SparkApplication` CRDs and orchestrates job execution.

**Status**: Running in `data-platform` namespace

```bash
# Check operator status
kubectl get pods -n data-platform -l app=spark-operator
kubectl logs -n data-platform -l app=spark-operator
```

### 2. Spark Configuration (`k8s/compute/spark/spark-configuration.yaml`)

ConfigMaps defining default Spark settings:

- **spark-defaults**: Global Spark configuration
- **spark-mlflow**: MLFlow tracking integration
- **spark-deequ**: Deequ quality framework settings
- **spark-history-conf**: History Server configuration
- **spark-env**: Environment variables

### 3. RBAC (`k8s/compute/spark/spark-rbac.yaml`)

Defines ServiceAccounts and permissions:

- `spark-operator`: Controls operator permissions
- `spark-app`: Used by driver/executor pods
- `spark-runner`: Used by job submissions

### 4. Example Jobs (`k8s/compute/spark/spark-example-jobs.yaml`)

Ready-to-use SparkApplication examples:

- **kafka-to-iceberg-etl**: Stream data from Kafka
- **iceberg-sql-analytics**: SQL analytics on Iceberg
- **data-enrichment-transform**: Data transformation with MLFlow
- **deequ-quality-check**: Validate data quality

### 5. Monitoring (`k8s/monitoring/spark-servicemonitor.yaml`)

Prometheus metrics collection and alerting rules

## Quick Start

### 1. Submit a Spark Job Manually

Create a `SparkApplication` manifest:

```yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: my-spark-job
  namespace: data-platform
spec:
  type: Python
  mode: cluster
  image: apache/spark:3.5.0
  pythonVersion: "3"
  
  mainApplicationFile: s3a://spark-code/my-job.py
  
  arguments:
  - "--input"
  - "s3a://input-data/"
  - "--output"
  - "s3a://output-data/"
  
  driver:
    cores: 2
    memory: "2g"
    serviceAccount: spark-app
  
  executor:
    cores: 2
    instances: 2
    memory: "2g"
```

Submit the job:

```bash
kubectl apply -f my-spark-job.yaml
```

### 2. Monitor Job Execution

```bash
# Get job status
kubectl get sparkapplication -n data-platform my-spark-job

# Watch logs
kubectl logs -n data-platform my-spark-job-driver

# Get full status
kubectl describe sparkapplication -n data-platform my-spark-job
```

### 3. View Job History

Access Spark History Server:

```
https://spark-history.254carbon.com
```

Or query locally:

```bash
kubectl get pod -n data-platform -l app=spark-history-server
```

## Integration Points

### Iceberg Tables

Spark can read/write Iceberg tables with full ACID support:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "rest") \
    .config("spark.sql.catalog.iceberg.uri", "http://iceberg-rest-catalog:8181") \
    .getOrCreate()

# Read from Iceberg
df = spark.sql("SELECT * FROM iceberg.raw.customers LIMIT 100")

# Write to Iceberg
df.writeTo("iceberg.ods.customers_transformed").mode("overwrite").saveAsTable()
```

### Kafka Streaming

Stream data from Kafka into Iceberg:

```python
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka-service:9092") \
    .option("subscribe", "raw-events") \
    .load()

df.writeStream \
    .format("iceberg") \
    .mode("append") \
    .option("checkpointLocation", "s3a://checkpoint/events") \
    .toTable("iceberg.raw.events")
```

### MLFlow Tracking

Track Spark jobs automatically:

```python
from services.spark_mlflow_client import SparkMLFlowClient

client = SparkMLFlowClient(experiment_name="etl-jobs")
run_id = client.start_spark_run("my-etl-job", tags={"source": "kafka"})

# Run Spark transformations...

client.log_metrics({
    "records_processed": df.count(),
    "processing_time_seconds": 123.45
})

client.end_run(status="FINISHED")
```

### Deequ Quality Checks

Validate data quality after transformations:

```python
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite

check = Check(spark, CheckLevel.Warning, "Data Quality Checks")

suite = VerificationSuite(spark) \
    .onData(df) \
    .addCheck(check \
        .hasSize(lambda x: x > 0) \
        .isComplete("customer_id") \
        .isUnique("email")
    ) \
    .run()

# Log results to MLFlow
client.log_quality_metrics(suite.metrics, "customers")
```

## Deployment Scenarios

### ETL Pipeline: Kafka → Iceberg

```bash
# Use kafka-to-iceberg-etl example
kubectl apply -f k8s/compute/spark/spark-example-jobs.yaml
```

### Batch Analytics Job

```bash
# Use iceberg-sql-analytics example
kubectl apply -f k8s/compute/spark/spark-example-jobs.yaml
```

### Scheduled Job via DolphinScheduler

1. Deploy the workflow template:
```bash
kubectl apply -f k8s/dolphinscheduler/spark-job-templates.yaml
```

2. Import into DolphinScheduler UI:
   - Go to `https://dolphin.254carbon.com`
   - Import workflow from `/k8s/dolphinscheduler/spark-job-templates.yaml`
   - Customize parameters
   - Schedule as needed

## Configuration

### Driver/Executor Resources

Adjust resources in SparkApplication spec:

```yaml
driver:
  cores: 4          # CPU cores
  memory: "4g"      # Memory
  coreLimit: "4000m" # Maximum cores
  
executor:
  cores: 2
  instances: 5      # Number of executors
  memory: "2g"
```

### Spark Configuration

Override in `spark-defaults.conf` ConfigMap or via SparkApplication `conf`:

```yaml
conf:
  spark.sql.adaptive.enabled: "true"
  spark.sql.shuffle.partitions: "300"
  spark.sql.broadcastTimeout: "600"
```

### Network & Storage

Configuration is in `spark-configuration.yaml`:

```yaml
# Iceberg Catalog
spark.sql.catalog.iceberg.uri: http://iceberg-rest-catalog:8181
spark.sql.catalog.iceberg.warehouse: s3://iceberg-warehouse/

# MinIO/S3
spark.hadoop.fs.s3a.endpoint: http://minio-service:9000
spark.hadoop.fs.s3a.access.key: minioadmin
spark.hadoop.fs.s3a.secret.key: minioadmin123

# Kafka
spark.kafka.bootstrap.servers: kafka-service:9092
```

## Monitoring & Troubleshooting

### Check Spark Operator Health

```bash
kubectl get pods -n data-platform -l app=spark-operator
kubectl logs -n data-platform -l app=spark-operator -f
```

### Monitor Running Jobs

```bash
# List all jobs
kubectl get sparkapplications -n data-platform

# Get job status
kubectl get sparkapplication -n data-platform kafka-to-iceberg-etl -o wide

# Get detailed status
kubectl describe sparkapplication -n data-platform kafka-to-iceberg-etl
```

### View Driver Logs

```bash
kubectl logs -n data-platform kafka-to-iceberg-etl-driver
```

### View Executor Logs

```bash
kubectl logs -n data-platform kafka-to-iceberg-etl-exec-1
```

### Prometheus Metrics

Query Spark metrics in Prometheus:

```
# Spark Operator uptime
up{job="spark-operator"}

# Spark Application status
spark_app_status{status="running"}

# Job success rate
rate(spark_app_status_successful_total[5m])
```

### Grafana Dashboards

Pre-built dashboards available at `https://grafana.254carbon.com`:

- Spark Cluster Overview
- Job Execution History
- Resource Utilization
- Quality Check Trends

## Best Practices

### 1. Resource Management

- Set appropriate `driverMemory` and `executorMemory`
- Use `coreLimit` to prevent overallocation
- Monitor actual usage in Prometheus

### 2. Data Partitioning

- Partition Iceberg tables by date/month
- Use proper partition keys in Spark transformations
- Aim for 128MB-256MB partition size

### 3. Fault Tolerance

- Enable checkpointing for streaming jobs
- Use retry policies in `SparkApplication` spec
- Store checkpoint data in S3/MinIO

### 4. Quality Assurance

- Run Deequ checks after each transformation
- Log quality metrics to MLFlow
- Set up alerts for quality failures

### 5. Performance Optimization

- Enable adaptive query execution
- Use broadcast joins for small tables
- Tune shuffle partition count based on data size

### 6. Security

- Use ServiceAccount `spark-app` for pod identity
- Secrets stored in `minio-secret` ConfigMap
- Network policies restrict pod communication

## Common Issues

### Job Stuck in PENDING State

```bash
# Check for pod scheduling issues
kubectl describe node
kubectl get events -n data-platform --sort-by='.lastTimestamp'
```

**Solution**: Check resource availability or adjust resource requests

### SparkApplication Crashes

```bash
# Check driver pod logs
kubectl logs -n data-platform <driver-pod-name>

# Check for configuration issues
kubectl get configmap -n data-platform spark-defaults -o yaml
```

### Out of Memory Errors

**Solution**: Increase driver or executor memory, reduce shuffle partition count

### S3/MinIO Connection Errors

**Solution**: Verify MinIO credentials in `minio-secret`, check network policies

## Advanced Topics

### Custom Spark Images

Build custom image with additional libraries:

```dockerfile
FROM apache/spark:3.5.0

RUN pip install pandas numpy scikit-learn

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
```

Then reference in SparkApplication:

```yaml
spec:
  image: my-registry/spark-custom:latest
```

### Kerberos Authentication

Configure in `spark-configuration.yaml`:

```yaml
spark.kerberos.principal: spark@EXAMPLE.COM
spark.kerberos.keytab: /etc/spark/keytab/spark.keytab
```

### Custom Package Dependencies

Add Maven packages:

```yaml
spec:
  sparkConf:
    spark.jars.packages: "org.apache.spark:spark-hadoop-cloud_2.12:3.5.0"
```

## Resources

- **Spark Documentation**: https://spark.apache.org/docs/latest/
- **Spark on Kubernetes**: https://spark.apache.org/docs/latest/running-on-kubernetes.html
- **Iceberg Spark Integration**: https://iceberg.apache.org/docs/latest/spark-configuration/
- **Deequ Documentation**: https://github.com/awslabs/deequ

## Support

For issues or questions:

1. Check logs: `kubectl logs -n data-platform <pod-name>`
2. Review this guide
3. Check Prometheus/Grafana for metrics
4. Review pull request history for similar issues
