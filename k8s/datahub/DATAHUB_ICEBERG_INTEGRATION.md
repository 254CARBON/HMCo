# DataHub Iceberg Integration Guide

## Overview

DataHub serves as a unified metadata catalog for your data platform, including Iceberg tables. This guide covers integrating DataHub with Iceberg REST Catalog.

## Architecture

```
┌──────────────────────┐
│    Iceberg Tables    │
│  (REST Catalog)      │
└──────────┬───────────┘
           │
           │ Metadata Discovery
           ▼
┌──────────────────────────────────┐
│  DataHub Iceberg Ingestion       │
│  ├─ Connectors & Transforms      │
│  └─ Metadata Extraction          │
└──────────┬───────────────────────┘
           │
           │ Publish Events
           ▼
┌──────────────────────────────────┐
│    DataHub GMS                   │
│    (Graph Metadata Service)      │
└──────────┬───────────────────────┘
           │
           │ Store & Index
           ▼
┌──────────────────────────────────┐
│  PostgreSQL + Elasticsearch      │
│  Neo4j (Graph Relations)         │
└──────────────────────────────────┘
```

## Prerequisites

1. **DataHub GMS running** (scales from 0)
2. **Iceberg REST Catalog deployed**
3. **DataHub secret created** (for authentication)
4. **Ingestion framework configured**

## Deployment Steps

### Step 1: Verify Prerequisites

```bash
# Check Iceberg REST Catalog
kubectl get pod -n data-platform -l app=iceberg-rest-catalog

# Check PostgreSQL
kubectl get pod -n data-platform -l app=postgres-shared

# Check Elasticsearch
kubectl get pod -n data-platform -l app=elasticsearch
```

### Step 2: Scale Up DataHub GMS

```bash
# Scale from 0 to 1 replica
kubectl scale deployment -n data-platform datahub-gms --replicas=1

# Monitor startup (should take 1-2 minutes)
kubectl logs -f -n data-platform deployment/datahub-gms
```

### Step 3: Verify DataHub Health

```bash
# Port-forward to DataHub
kubectl port-forward -n data-platform svc/datahub-gms 8080:8080 &

# Check health
curl http://localhost:8080/health

# Expected response
# HTTP/1.1 200 OK
```

## Iceberg Ingestion Recipe

Create a DataHub ingestion recipe for Iceberg. This can be:

1. **Kubernetes Job**: Runs periodically to sync metadata
2. **Python Script**: Direct CLI ingestion
3. **DataHub UI**: Configure via web interface

### Option A: Kubernetes Ingestion Job

```yaml
# File: k8s/datahub/iceberg-ingestion-job.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: datahub-iceberg-ingestion
  namespace: data-platform
spec:
  schedule: "0 * * * *"  # Run hourly
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: default
          containers:
          - name: datahub-ingestion
            image: acryldata/datahub-ingestion:latest
            env:
            - name: DATAHUB_GMS_URL
              value: "http://datahub-gms:8080"
            volumeMounts:
            - name: ingestion-recipe
              mountPath: /recipes
            command: ["/bin/sh"]
            args: ["-c", "datahub ingest -c /recipes/iceberg-recipe.yml"]
          restartPolicy: OnFailure
          volumes:
          - name: ingestion-recipe
            configMap:
              name: datahub-iceberg-recipe
```

### Option B: Direct Python Ingestion

```python
# File: scripts/ingest_iceberg_metadata.py
from datahub.ingestion.run import run
from datahub.ingestion.graph.client import DatahubClientConfig, DataHubGraph

# Recipe configuration
recipe = {
    "source": {
        "type": "iceberg",
        "config": {
            "env": "prod",
            "catalog": {
                "rest_catalog": {
                    "type": "rest",
                    "uri": "http://iceberg-rest-catalog:8181",
                    "s3.access-key-id": "minioadmin",
                    "s3.secret-access-key": "minioadmin123",
                    "s3.region": "us-east-1",
                    "s3.endpoint": "http://minio-service:9000"
                }
            },
            "platform_instance": "hmco_iceberg",
            "profiling": {
                "enabled": True,
                "profile_table_level_only": True
            }
        }
    },
    "sink": {
        "type": "datahub-rest",
        "config": {
            "server": "http://datahub-gms:8080"
        }
    }
}

# Run ingestion
result = run(recipe)
```

## Ingestion Configuration

### Basic Recipe

```yaml
source:
  type: "iceberg"
  config:
    env: "prod"
    catalog:
      rest_catalog:
        type: "rest"
        uri: "http://iceberg-rest-catalog:8181"
        s3.access-key-id: "minioadmin"
        s3.secret-access-key: "minioadmin123"
        s3.region: "us-east-1"
        s3.endpoint: "http://minio-service:9000"
    platform_instance: "hmco_iceberg"
    
sink:
  type: datahub-rest
  config:
    server: "http://datahub-gms:8080"
```

### Advanced Recipe with Profiling

```yaml
source:
  type: "iceberg"
  config:
    env: "prod"
    catalog:
      rest_catalog:
        type: "rest"
        uri: "http://iceberg-rest-catalog:8181"
        s3.access-key-id: "minioadmin"
        s3.secret-access-key: "minioadmin123"
        s3.region: "us-east-1"
        s3.endpoint: "http://minio-service:9000"
    
    platform_instance: "hmco_iceberg"
    
    # Namespace filtering
    namespace_pattern:
      allow: [".*"]
      ignore: ["temp", "scratch"]
    
    # Table filtering
    table_pattern:
      allow: [".*"]
    
    # Data profiling
    profiling:
      enabled: true
      include_field_null_count: true
      include_field_min_value: true
      include_field_max_value: true
      operation_config:
        profile_day_of_week: 0  # Sunday
        profile_date_of_month: 1  # First day
    
    # Ownership extraction from table properties
    user_ownership_property: "owner"
    group_ownership_property: "team"

sink:
  type: datahub-rest
  config:
    server: "http://datahub-gms:8080"
    token: "datahub_ingestion_token"  # Optional authentication

transformers:
  - type: "pattern_add_dataset_ownership"
    config:
      input_pattern: "^(.*)\\.(.*)$"
      output_pattern: "$2"
      regex_flags: MULTILINE
  - type: "add_dataset_tags"
    config:
      tags: ["iceberg", "data_lake"]
```

## Common Ingestion Patterns

### Pattern 1: Hourly Ingestion

```yaml
# Sync all Iceberg metadata every hour
source:
  type: "iceberg"
  config:
    # ... configuration ...

sink:
  type: datahub-rest
  config:
    server: "http://datahub-gms:8080"
```

### Pattern 2: Selective Namespace Ingestion

```yaml
source:
  type: "iceberg"
  config:
    # ... base configuration ...
    namespace_pattern:
      allow: ["production", "analytics"]
      ignore: ["temp", "test"]
```

### Pattern 3: With Data Profiling

```yaml
source:
  type: "iceberg"
  config:
    # ... base configuration ...
    profiling:
      enabled: true
      include_field_null_count: true
      include_field_min_value: true
      include_field_max_value: true
```

## Metadata Discovery Workflow

1. **Connection**: DataHub connects to Iceberg REST Catalog
2. **Enumeration**: Lists all namespaces and tables
3. **Metadata Extraction**: Collects:
   - Table schemas
   - Column types and descriptions
   - Partitioning information
   - Table properties
   - Creation/modification timestamps
4. **Profiling** (optional): Computes:
   - Row counts
   - Column statistics
   - Data quality metrics
5. **Publishing**: Sends metadata to DataHub GMS
6. **Indexing**: Elasticsearch indexes metadata for search

## Verification

### Check Ingestion Status

```bash
# Check ingestion runs
kubectl get jobs -n data-platform -l app=datahub

# Check ingestion logs
kubectl logs -n data-platform -l app=datahub | grep iceberg

# View ingestion history via DataHub UI
# http://datahub-frontend:9002/admin/ingestion
```

### Query Ingested Metadata

Via DataHub UI:
1. Go to Explore → Datasets
2. Filter by "Platform: iceberg"
3. View discovered tables and schemas

Via GraphQL (if available):
```graphql
query {
  search(input: {
    type: DATASET
    query: "platform:iceberg"
    start: 0
    count: 100
  }) {
    searchResults {
      entity {
        entityUrn
        urn
      }
    }
  }
}
```

## Troubleshooting

### Connection Issues

```bash
# Test Iceberg REST Catalog connectivity
kubectl exec -it -n data-platform datahub-gms-xxx -- \
  curl -v http://iceberg-rest-catalog:8181/v1/config
```

### Authentication Errors

Verify credentials in recipe:
```yaml
s3.access-key-id: "minioadmin"
s3.secret-access-key: "minioadmin123"
```

### No Tables Discovered

1. Verify Iceberg namespaces exist
2. Check table permissions
3. Review ingestion logs
4. Verify S3 access

### Ingestion Timeout

Increase timeout in recipe:
```yaml
sink:
  type: datahub-rest
  config:
    server: "http://datahub-gms:8080"
    timeout_sec: 300  # 5 minutes
```

## Best Practices

1. **Regular Ingestion**: Schedule ingestion jobs hourly or daily
2. **Namespace Organization**: Organize tables into logical namespaces
3. **Ownership Tagging**: Use table properties to track ownership
4. **Data Quality**: Enable profiling for important tables
5. **Documentation**: Add descriptions to tables and columns
6. **Access Control**: Implement proper DataHub RBAC policies

## Next Steps

1. Create Iceberg namespaces and sample tables
2. Run first ingestion job
3. Verify metadata appears in DataHub UI
4. Set up ownership and access policies
5. Enable data quality monitoring

## References

- [DataHub Iceberg Integration Docs](https://docs.datahub.com/docs/generated/ingestion/sources/iceberg)
- [DataHub Ingestion Framework](https://datahub.io/docs/metadata-ingestion/)
- [Apache Iceberg REST API](https://iceberg.apache.org/rest-catalog-spec/)
