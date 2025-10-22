# Apache Iceberg Data Lake Integration

## Overview

This documentation describes the complete integration of Apache Iceberg REST Catalog into the HMCo data platform, providing a modern data lakehouse architecture with unified metadata management through DataHub and distributed SQL queries via Trino.

## What is Iceberg?

Apache Iceberg is a table format that brings SQL-table semantics to object storage. It provides:
- **ACID Transactions**: Reliable data updates and deletes
- **Scalable Metadata**: Efficient handling of large numbers of files
- **Schema Evolution**: Safe column additions and updates
- **Time Travel**: Query historical snapshots
- **Hidden Partitioning**: Automatic partition management

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Data Sources                         │
│  (Kafka, MySQL, PostgreSQL, Files, APIs)        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   SeaTunnel     │ (Data Integration & ETL)
         │ (Data Pipelines)│
         └────────┬────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Iceberg REST Catalog        │
    │  ├─ MinIO (S3 Storage)       │
    │  ├─ PostgreSQL (Metadata)    │
    │  └─ Table Management         │
    └──────┬──────────────┬────────┘
           │              │
           ▼              ▼
    ┌────────────┐  ┌───────────┐
    │   Trino    │  │  DataHub  │
    │ (SQL Query │  │(Metadata  │
    │  Engine)   │  │ Catalog)  │
    └────────────┘  └───────────┘
           │              │
           └──────┬───────┘
                  ▼
           ┌────────────────┐
           │ End Users      │
           │ BI Tools       │
           │ Data Scientists│
           └────────────────┘
```

## Components

### 1. Iceberg REST Catalog (Port 8181)
- Unified table metadata management
- REST API for all Iceberg operations
- Support for namespaces and table creation
- Connection to PostgreSQL for metadata persistence
- S3-compatible storage via MinIO

### 2. MinIO Object Storage (Ports 9000/9001)
- S3-compatible storage for Iceberg data files
- Buckets: `iceberg-warehouse`, `seatunnel-output`, `datahub-storage`
- Default credentials: `minioadmin/minioadmin123`

### 3. PostgreSQL (Port 5432)
- Database: `iceberg_rest`
- User: `iceberg_user`
- Stores Iceberg table metadata and catalog state

### 4. Trino Query Engine (Port 8080)
- Distributed SQL query engine
- Iceberg connector for querying tables
- Support for joins across catalogs
- Pushdown optimization for performance

### 5. DataHub Metadata Catalog (Port 8080 GMS / 9002 Frontend)
- Discovers Iceberg table metadata
- Tracks data lineage and ownership
- Provides data governance interface
- Enables metadata search and discovery

### 6. SeaTunnel Data Integration (Ports 5801/8080)
- Data pipelines and ETL jobs
- Kafka to Iceberg streaming
- CDC from MySQL/PostgreSQL
- Batch ETL transformations

## Quick Start

### Prerequisites

```bash
# Kubernetes cluster with data-platform namespace
kubectl get ns data-platform

# All core services running
kubectl get pod -n data-platform | grep -E "postgres|minio|kafka"
```

### Deploy Iceberg

```bash
# 1. Create secrets
kubectl apply -f k8s/secrets/minio-secret.yaml
kubectl apply -f k8s/secrets/datahub-secret.yaml

# 2. Initialize MinIO buckets
kubectl apply -f k8s/data-lake/minio-init-job.yaml
kubectl wait --for=condition=complete job/minio-init-buckets -n data-platform --timeout=5m

# 3. Deploy Iceberg REST Catalog
kubectl apply -f k8s/data-lake/iceberg-rest.yaml

# 4. Configure Trino
kubectl apply -f k8s/compute/trino/trino.yaml

# 5. Configure DataHub
kubectl apply -f k8s/datahub/iceberg-ingestion-recipe.yaml

# 6. Verify deployment
kubectl get pod -n data-platform -l app=iceberg-rest-catalog
```

### Test the Integration

```bash
# Run comprehensive tests
# See: ICEBERG_INTEGRATION_TEST_GUIDE.md

# Quick test:
# 1. Query via Trino: http://localhost:8080
# 2. Create table in Iceberg via Trino
# 3. Ingest metadata to DataHub
# 4. Query data with SeaTunnel
```

## Usage Examples

### Create and Query Iceberg Tables with Trino

```sql
-- Create namespace
CREATE SCHEMA iceberg.analytics;

-- Create table
CREATE TABLE iceberg.analytics.customers (
    id BIGINT,
    name VARCHAR,
    email VARCHAR,
    created_at TIMESTAMP(3) WITH TIME ZONE
)
WITH (
    format = 'PARQUET',
    location = 's3://iceberg-warehouse/analytics/customers'
);

-- Insert data
INSERT INTO iceberg.analytics.customers
VALUES (1, 'John Doe', 'john@example.com', CURRENT_TIMESTAMP);

-- Query data
SELECT * FROM iceberg.analytics.customers;

-- Time travel
SELECT * FROM iceberg.analytics.customers 
FOR VERSION AS OF TIMESTAMP '2025-10-19 10:00:00';
```

### Stream Data with SeaTunnel

```conf
# Kafka to Iceberg
env {
  execution.parallelism = 2
  job.mode = "STREAMING"
}

source {
  Kafka {
    bootstrap.servers = "kafka-service:9092"
    topic = "events"
    result_table_name = "events_stream"
  }
}

sink {
  Iceberg {
    catalog_name = "rest"
    uri = "http://iceberg-rest-catalog:8181"
    database = "raw"
    table = "events"
    warehouse = "s3://iceberg-warehouse/"
  }
}
```

### Discover Metadata with DataHub

```bash
# Ingest Iceberg metadata
kubectl apply -f k8s/datahub/iceberg-ingestion-recipe.yaml
kubectl apply -f k8s/datahub/iceberg-ingestion-recipe.yaml -l type=job

# Access DataHub UI
kubectl port-forward svc/datahub-frontend 9002:9002
# Browse to http://localhost:9002
# Search for Iceberg tables
```

## Documentation

Comprehensive guides for each component:

| Guide | Purpose | Link |
|-------|---------|------|
| Deployment | Setup and configuration | [k8s/data-lake/ICEBERG_DEPLOYMENT.md](k8s/data-lake/ICEBERG_DEPLOYMENT.md) |
| Trino Integration | SQL query engine setup | [k8s/compute/trino/TRINO_ICEBERG_GUIDE.md](k8s/compute/trino/TRINO_ICEBERG_GUIDE.md) |
| DataHub Integration | Metadata catalog setup | [k8s/datahub/DATAHUB_ICEBERG_INTEGRATION.md](k8s/datahub/DATAHUB_ICEBERG_INTEGRATION.md) |
| SeaTunnel Integration | Data pipeline setup | [k8s/seatunnel/SEATUNNEL_ICEBERG_GUIDE.md](k8s/seatunnel/SEATUNNEL_ICEBERG_GUIDE.md) |
| Testing Guide | End-to-end testing procedures | [ICEBERG_INTEGRATION_TEST_GUIDE.md](ICEBERG_INTEGRATION_TEST_GUIDE.md) |
| Security Hardening | Security best practices | [ICEBERG_SECURITY_HARDENING.md](ICEBERG_SECURITY_HARDENING.md) |
| Monitoring & Alerting | Observability setup | [ICEBERG_MONITORING_GUIDE.md](ICEBERG_MONITORING_GUIDE.md) |
| Operations Runbook | Daily operational procedures | [ICEBERG_OPERATIONS_RUNBOOK.md](ICEBERG_OPERATIONS_RUNBOOK.md) |

## Configuration Files

All Kubernetes manifests are located in:
- `k8s/secrets/` - Credentials and secrets
- `k8s/data-lake/` - Iceberg components
- `k8s/compute/trino/` - Trino configuration
- `k8s/datahub/` - DataHub ingestion recipes
- `k8s/seatunnel/jobs/` - Example ETL jobs
- `k8s/monitoring/` - Prometheus and alerting

## Key Features

### ✅ Already Implemented

- [x] Iceberg REST Catalog deployment
- [x] MinIO S3-compatible storage
- [x] PostgreSQL metadata persistence
- [x] Trino SQL query engine integration
- [x] DataHub metadata discovery
- [x] SeaTunnel data pipeline connectors
- [x] Production-ready configurations
- [x] Health checks and monitoring
- [x] Security hardening guidance
- [x] Comprehensive documentation

### 🔄 Recommended Next Steps

1. **Security Hardening**
   - Update default credentials (MinIO, PostgreSQL)
   - Enable TLS/HTTPS for REST APIs
   - Configure RBAC policies
   - Enable audit logging

2. **Data Migration**
   - Plan data migration strategy
   - Create Iceberg tables from existing data
   - Validate data consistency
   - Decommission old storage

3. **Monitoring Setup**
   - Deploy Prometheus monitoring
   - Configure Grafana dashboards
   - Set up alert rules
   - Implement SLI/SLO tracking

4. **Performance Optimization**
   - Tune table partitioning strategies
   - Optimize query performance
   - Monitor resource usage
   - Implement caching strategies

## API Reference

### Iceberg REST API Endpoints

```
GET  /v1/config                    # Configuration info
GET  /v1/namespaces                # List namespaces
POST /v1/namespaces                # Create namespace
GET  /v1/namespaces/{ns}/tables    # List tables
POST /v1/namespaces/{ns}/tables    # Create table
GET  /v1/namespaces/{ns}/tables/{t} # Get table
DELETE /v1/namespaces/{ns}/tables/{t} # Drop table
```

### Trino SQL Examples

```sql
-- List Iceberg catalogs
SHOW CATALOGS;

-- List schemas in Iceberg
SHOW SCHEMAS FROM iceberg;

-- Create table
CREATE TABLE iceberg.schema.table (...) WITH (...);

-- Query table
SELECT * FROM iceberg.schema.table;

-- Time travel
SELECT * FROM iceberg.schema.table 
FOR VERSION AS OF TIMESTAMP '...';
```

## Troubleshooting

### Common Issues

| Issue | Solution | Ref |
|-------|----------|-----|
| Pod won't start | Check logs, verify secrets | Operations Runbook |
| Can't connect to Iceberg | Verify network, check PostgreSQL | Deployment Guide |
| Trino errors | Check catalog config, verify endpoint | Trino Guide |
| DataHub not discovering tables | Run ingestion job, check credentials | DataHub Guide |
| Slow queries | Check partitioning, monitor resources | Monitoring Guide |

See [ICEBERG_OPERATIONS_RUNBOOK.md](ICEBERG_OPERATIONS_RUNBOOK.md) for detailed troubleshooting procedures.

## Performance Benchmarks

Expected performance metrics:

| Metric | Value | Condition |
|--------|-------|-----------|
| Catalog API latency P95 | < 500ms | Normal load |
| Table creation time | 1-5 seconds | Small tables |
| Query latency P95 | 500ms-2s | 1GB+ scans |
| Metadata sync interval | 2 hours | DataHub ingestion |
| Throughput | 1000+ req/sec | Per instance |

## Support and Resources

### Documentation
- [Apache Iceberg Docs](https://iceberg.apache.org/docs/)
- [Iceberg REST Spec](https://iceberg.apache.org/rest-catalog-spec/)
- [Trino Iceberg Connector](https://trino.io/docs/current/connector/iceberg.html)
- [DataHub Iceberg Integration](https://docs.datahub.com/docs/generated/ingestion/sources/iceberg)

### Community
- [Apache Iceberg Slack](https://iceberg.apache.org/slack/)
- [Trino Community](https://trino.io/community.html)
- [DataHub Community](https://datahubproject.io/community)

## License and Attribution

This integration is built on:
- **Apache Iceberg** - Table format (Apache 2.0)
- **Trino** - SQL query engine (Apache 2.0)
- **DataHub** - Metadata platform (Apache 2.0)
- **SeaTunnel** - Data integration (Apache 2.0)

## Version Information

| Component | Version | Status |
|-----------|---------|--------|
| Iceberg REST Catalog | 0.6.0 | Stable |
| Trino | 436 | Stable |
| DataHub | Latest | Stable |
| SeaTunnel | 2.3.12 | Stable |
| MinIO | Latest | Stable |
| PostgreSQL | 15 | Stable |

## Maintenance Schedule

| Task | Frequency | Duration |
|------|-----------|----------|
| Health check | Daily | 15 min |
| Backup | Daily | 30 min |
| Cleanup old snapshots | Weekly | 30 min |
| Credential rotation | Monthly | 1 hour |
| Security audit | Monthly | 2 hours |
| Performance review | Monthly | 1 hour |
| Major version upgrade | Quarterly | 2-4 hours |

## Contact

For issues or questions:
1. Check relevant documentation above
2. Review [ICEBERG_OPERATIONS_RUNBOOK.md](ICEBERG_OPERATIONS_RUNBOOK.md)
3. Contact platform engineering team
4. Escalate to infrastructure team if critical

## Changelog

### Version 1.0 - October 19, 2025
- Initial Iceberg REST Catalog integration
- Trino query engine support
- DataHub metadata discovery
- SeaTunnel data pipeline connectors
- Complete monitoring and alerting setup
- Comprehensive documentation and runbooks

---

**Last Updated**: October 19, 2025  
**Next Review**: October 26, 2025
