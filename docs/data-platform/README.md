# HMCo Data Platform

## Overview

The HMCo data platform is a modern, scalable lakehouse architecture built on Apache Iceberg, Trino, and ClickHouse, designed to handle energy market data at scale.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
│  EIA │ FRED │ NOAA │ Census │ CAISO │ MISO │ SPP │ Others  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  UIS (Universal         │
            │  Ingestion Spec)        │
            └──────────┬─────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  SeaTunnel / Spark Pipelines │
         └──────────┬──────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────┐
    │     MinIO (S3-Compatible Storage)      │
    │  ├─ hmco-raw (30 days)                 │
    │  ├─ hmco-staged (12 months)            │
    │  ├─ hmco-curated (5 years)             │
    │  └─ hmco-ml (1 year)                   │
    └──────────┬────────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Iceberg REST Catalog     │
    │  (Postgres + S3)          │
    └──────────┬───────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  ┌─────────┐   ┌──────────────┐
  │  Trino  │   │  ClickHouse   │
  │ (Query) │   │  (OLAP RT)    │
  └────┬────┘   └──────┬───────┘
       │               │
       └───────┬───────┘
               ▼
    ┌──────────────────────┐
    │  BI Tools / APIs      │
    │  Superset │ Grafana   │
    └──────────────────────┘
```

## Key Components

### Storage Layer

- **MinIO**: S3-compatible object storage with lifecycle policies
  - Raw: 30 days retention
  - Staged: 12 months retention
  - Curated: 5 years retention
  - ML: 1 year retention

### Catalog Layer

- **Iceberg REST Catalog**: Unified metadata management
  - Postgres backend for durability
  - ACID transactions
  - Time travel and schema evolution

### Compute Layer

- **Trino**: Distributed SQL query engine
  - Cross-source federation (Iceberg + ClickHouse)
  - Resource groups for query prioritization
  - 100+ connectors

- **ClickHouse**: High-performance OLAP database
  - 3 shards × 2 replicas (HA)
  - 5-minute real-time LMP data
  - TTL-based retention
  - Materialized views for aggregations

### Orchestration

- **DolphinScheduler**: Workflow management
  - 7 core workflows (EIA, FRED, NOAA, Census, CAISO, MISO, SPP)
  - SLA monitoring and alerting
  - Data quality gates

### Data Quality

- **Great Expectations**: Validation framework
  - Schema validation
  - Range checks
  - Freshness monitoring
  - Fail-closed publish gates

## Data Sources

| Source | Type | Frequency | SLA | Target |
|--------|------|-----------|-----|--------|
| EIA | HTTP API | Daily | 30 min | Iceberg |
| FRED | HTTP API | Daily | 30 min | Iceberg |
| NOAA | HTTP API | Hourly | 20 min | Iceberg |
| Census | HTTP API | Monthly | 24 hours | Iceberg |
| CAISO | OASIS API | 5-min | 10 min | ClickHouse |
| MISO | Web Services | 5-min | 10 min | ClickHouse |
| SPP | Marketplace | 5-min | 10 min | ClickHouse |

## Throughput Targets

- **Ingest Peak**: 100 MB/s sustained (≥ 200k rows/s)
- **Weekly Volume**: 3 TB/week into lake
- **Query Concurrency**: 100 concurrent queries (Trino)
- **RT Latency**: ≤ 10 min lag for ISO 5-min data

## Security

- **MinIO**: Bucket-level policies (etl-writer, bi-reader)
- **ClickHouse**: Role-based access + row-level security per desk/ISO
- **Trino**: Resource groups + query limits
- **Secrets**: HashiCorp Vault via External Secrets Operator
- **Network**: Kubernetes NetworkPolicies for pod-to-pod traffic

## Disaster Recovery

- **RPO**: ≤ 30 minutes
- **RTO**: ≤ 2 hours
- **Strategy**: Warm standby
  - MinIO: Async bucket replication to secondary site
  - ClickHouse: Full + incremental backups to S3
  - Postgres: Daily dumps + WAL archiving
- **Testing**: Quarterly DR drills

## Maintenance

### Automated Jobs

- **Iceberg Maintenance** (Weekly):
  - Expire Snapshots (Sunday 2 AM)
  - Rewrite Manifests (Sunday 3 AM)
  - Compact Data Files (Sunday 4 AM)

- **ClickHouse Backups**:
  - Full: Weekly (Sunday 1 AM)
  - Incremental: Daily (2 AM)

- **Postgres Backups**: Daily (Midnight)

## Documentation

- [Iceberg Integration Guide](iceberg/integration.md)
- [Security & RBAC](security-rbac.md)
- [ClickHouse Retention Policies](clickhouse/retention-policies.md)
- [Backup & DR Guide](../operations/backup-guide.md)

## Quick Start

### Deploy the Platform

```bash
# Deploy Iceberg REST Catalog
helm install iceberg-rest-catalog ./helm/charts/data-platform/charts/iceberg-rest-catalog

# Deploy Trino
helm install trino ./helm/charts/data-platform/charts/trino

# Deploy ClickHouse
helm install clickhouse ./helm/charts/data-platform/charts/clickhouse

# Initialize MinIO buckets
kubectl apply -f helm/charts/data-platform/charts/data-lake/templates/minio-init-job.yaml

# Deploy maintenance jobs
helm install maintenance ./helm/charts/data-platform/charts/maintenance

# Deploy backup infrastructure
helm install backup ./helm/charts/data-platform/charts/backup
```

### Run a Workflow

```bash
# Deploy DolphinScheduler workflows
kubectl apply -f workflows/01-eia-daily.json
kubectl apply -f workflows/04-caiso-rt.json

# Monitor workflow execution
kubectl logs -f deployment/dolphinscheduler-master -n data-platform
```

### Query Data

```bash
# Connect to Trino
trino --server trino-coordinator:8080 --catalog iceberg

# Query EIA data
SELECT region, series, avg(value) as avg_value
FROM hub_curated.eia_daily_fuel
WHERE ts >= CURRENT_DATE - INTERVAL '7' DAY
GROUP BY region, series;

# Query RT LMP data (ClickHouse via Trino)
SELECT iso, node, avg(lmp) as avg_lmp
FROM clickhouse.default.rt_lmp
WHERE ts >= now() - INTERVAL '1' HOUR
GROUP BY iso, node
ORDER BY avg_lmp DESC
LIMIT 10;
```

## Monitoring

- **Prometheus Alerts**: Freshness SLAs, backup status, resource usage
- **Grafana Dashboards**: Query performance, ingestion rates, storage usage
- **Slack Integration**: Critical alerts to #data-platform-critical

## Support

- **Data Engineering**: data-eng@254carbon.com
- **On-call**: +1-555-0123
- **Slack**: #data-platform

---

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Next Review**: November 30, 2025
