# SeaTunnel Iceberg Integration Guide

## Overview

SeaTunnel enables reliable data integration between various sources and Iceberg tables. This guide covers configuring SeaTunnel with Iceberg as a sink.

## Architecture

```
Data Sources          SeaTunnel Engine       Iceberg Platform
    │                      │                      │
    ├─ Kafka ────────┐     │     ┌──────────────┐ │
    ├─ MySQL ────────┼──►  │  ├──┤ Iceberg REST │─┤
    ├─ PostgreSQL ───┤ ▼   │  │  │ Catalog      │ │
    └─ Files ────────┘  Transformers            │ │
                        (optional)          MinIO + DB
                            │
                            ▼
                        Iceberg Sink
```

## Prerequisites

1. **SeaTunnel running** (may need to scale from 0)
2. **Iceberg REST Catalog deployed**
3. **MinIO accessible** with credentials
4. **Kafka or other source** available

## Configuration

### Iceberg Sink Configuration

```json
{
  "plugin_name": "iceberg",
  "result_table_name": "iceberg_sink",
  "schema": {
    "fields": {
      "id": "BIGINT",
      "name": "STRING",
      "email": "STRING",
      "created_at": "TIMESTAMP"
    }
  },
  "sink": {
    "catalog_name": "rest",
    "catalog_type": "rest",
    "warehouse": "s3://iceberg-warehouse/",
    "uri": "http://iceberg-rest-catalog:8181",
    "database": "raw",
    "table": "events",
    
    "s3.access-key-id": "minioadmin",
    "s3.secret-access-key": "minioadmin123",
    "s3.region": "us-east-1",
    "s3.endpoint": "http://minio-service:9000"
  }
}
```

### Iceberg Source Configuration

```json
{
  "plugin_name": "iceberg",
  "result_table_name": "iceberg_source",
  "schema": {
    "fields": {
      "id": "BIGINT",
      "name": "STRING",
      "created_at": "TIMESTAMP"
    }
  },
  "source": {
    "catalog_name": "rest",
    "catalog_type": "rest",
    "warehouse": "s3://iceberg-warehouse/",
    "uri": "http://iceberg-rest-catalog:8181",
    "database": "raw",
    "table": "customers",
    
    "s3.access-key-id": "minioadmin",
    "s3.secret-access-key": "minioadmin123",
    "s3.region": "us-east-1",
    "s3.endpoint": "http://minio-service:9000"
  }
}
```

## Example Jobs

### Job 1: Kafka to Iceberg

Stream real-time events from Kafka into Iceberg:

```conf
env {
  execution.parallelism = 2
  job.mode = "STREAMING"
}

source {
  Kafka {
    bootstrap.servers = "kafka-service:9092"
    topic = "events"
    consumer.group = "seatunnel-iceberg"
    result_table_name = "kafka_events"
    format = "json"
  }
}

sink {
  Iceberg {
    catalog_name = "rest"
    catalog_type = "rest"
    warehouse = "s3://iceberg-warehouse/"
    uri = "http://iceberg-rest-catalog:8181"
    database = "raw"
    table = "events"
    s3.endpoint = "http://minio-service:9000"
    s3.access-key-id = "minioadmin"
    s3.secret-access-key = "minioadmin123"
    partition_by = ["event_type", "year(event_timestamp)"]
  }
}
```

### Job 2: MySQL CDC to Iceberg

Capture changes from MySQL and sync to Iceberg:

```conf
env {
  execution.parallelism = 2
  job.mode = "STREAMING"
}

source {
  MySQL-CDC {
    hostname = "mysql-service"
    port = 3306
    username = "root"
    password = "password"
    database-names = ["db"]
    table-names = ["db.customers"]
    result_table_name = "mysql_cdc"
  }
}

sink {
  Iceberg {
    catalog_name = "rest"
    catalog_type = "rest"
    warehouse = "s3://iceberg-warehouse/"
    uri = "http://iceberg-rest-catalog:8181"
    database = "ods"
    table = "customers_sync"
    partition_by = ["year(updated_at)"]
  }
}
```

### Job 3: Batch ETL (PostgreSQL to Iceberg)

Transform and load data from PostgreSQL to Iceberg:

```conf
env {
  execution.parallelism = 4
  job.mode = "BATCH"
}

source {
  Jdbc {
    driver = "org.postgresql.Driver"
    url = "jdbc:postgresql://postgres:5432/mydb"
    user = "postgres"
    password = "password"
    query = """
      SELECT 
        id, name, email, 
        DATE_TRUNC('day', created_at)::timestamp as day,
        created_at
      FROM customers
      WHERE created_at > '2025-01-01'
    """
    result_table_name = "postgres_source"
  }
}

transform {
  # Add transformation logic if needed
  # sql = """SELECT * FROM postgres_source"""
}

sink {
  Iceberg {
    catalog_name = "rest"
    catalog_type = "rest"
    warehouse = "s3://iceberg-warehouse/"
    uri = "http://iceberg-rest-catalog:8181"
    database = "analytics"
    table = "customers"
    partition_by = ["day"]
    write_format = "parquet"
    compression_codec = "snappy"
  }
}
```

## Deployment

### Deploy SeaTunnel Job

```bash
# Check if SeaTunnel pod is running (scale from 0 if needed)
kubectl scale deployment -n data-platform seatunnel --replicas=1

# Copy job file to SeaTunnel pod
kubectl cp ./jobs/kafka-to-iceberg.conf \
  data-platform/seatunnel-xxx:/opt/seatunnel/jobs/

# Submit job
kubectl exec -it -n data-platform seatunnel-xxx -- \
  /opt/seatunnel/bin/seatunnel.sh \
  --config /opt/seatunnel/jobs/kafka-to-iceberg.conf
```

### Using SeaTunnel UI

1. Access SeaTunnel UI: `http://seatunnel:8080`
2. Create new job
3. Configure source and sink
4. Deploy and monitor

### Using Kubernetes Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: seatunnel-kafka-to-iceberg
  namespace: data-platform
spec:
  template:
    spec:
      containers:
      - name: seatunnel
        image: apache/seatunnel:2.3.3
        command:
        - /opt/seatunnel/bin/seatunnel.sh
        args:
        - --config
        - /jobs/kafka-to-iceberg.conf
        volumeMounts:
        - name: jobs
          mountPath: /jobs
      restartPolicy: Never
      volumes:
      - name: jobs
        configMap:
          name: seatunnel-iceberg-jobs
```

## Monitoring

### Check Job Status

```bash
# View SeaTunnel logs
kubectl logs -f -n data-platform seatunnel-xxx

# Check running jobs
kubectl exec -it -n data-platform seatunnel-xxx -- \
  jps | grep SeaTunnel

# Monitor Iceberg table
kubectl port-forward svc/trino-coordinator 8080:8080 &
# Query: SELECT COUNT(*) FROM iceberg.raw.events;
```

### Performance Tuning

#### Parallelism Configuration

```conf
env {
  execution.parallelism = 4  # Increase for better throughput
}
```

#### Batch Size

```conf
source {
  Kafka {
    fetch.min.bytes = 1MB
    fetch.max.wait.ms = 500
  }
}
```

#### Checkpointing

```conf
env {
  checkpoint.interval = 30000  # 30 seconds
  checkpoint.timeout = 600000  # 10 minutes
  state.backend = "filesystem"
}
```

## Transformations

### Add Timestamp

```conf
transform {
  sql = """
    SELECT 
      *,
      CURRENT_TIMESTAMP as load_timestamp
    FROM kafka_events
  """
}
```

### Filter and Deduplicate

```conf
transform {
  sql = """
    SELECT DISTINCT
      id, name, email, created_at
    FROM mysql_cdc
    WHERE id IS NOT NULL
  """
}
```

## Troubleshooting

### Connection Errors

Check connectivity:
```bash
# Test Iceberg REST Catalog
curl http://iceberg-rest-catalog:8181/v1/config

# Test MinIO
curl http://minio-service:9000/minio/health/live
```

### Schema Mismatch

Ensure schema matches between source and sink:
```conf
schema = {
  fields {
    id = "BIGINT"
    name = "STRING"
    # ... must match source schema
  }
}
```

### Memory Issues

Increase SeaTunnel memory:
```yaml
env:
- name: JAVA_OPTS
  value: "-Xms2g -Xmx4g"
```

### Partition Creation Issues

Verify partition path in MinIO:
```bash
mc ls minio/iceberg-warehouse/raw/events/
```

## Best Practices

1. **Use Partitioning**: Partition Iceberg tables for better query performance
2. **Compression**: Use Snappy or Zstd for compression
3. **Batch Size**: Tune batch sizes for your data volume
4. **Error Handling**: Configure retry policies
5. **Monitoring**: Monitor job status and data quality
6. **Testing**: Test jobs with small datasets first

## References

- [SeaTunnel Iceberg Connector Docs](https://seatunnel.apache.org/docs/2.3.3/connector/iceberg)
- [Apache Iceberg](https://iceberg.apache.org/)
- [SeaTunnel Documentation](https://seatunnel.apache.org/docs/)

## Support

For issues:
1. Check SeaTunnel logs: `kubectl logs -f deployment/seatunnel`
2. Verify Iceberg connectivity: `curl http://iceberg-rest-catalog:8181/v1/config`
3. Test S3 access to MinIO
4. Review SeaTunnel documentation
