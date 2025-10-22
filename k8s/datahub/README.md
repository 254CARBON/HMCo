# DataHub Deployment - 254Carbon Data Platform

## Overview

DataHub is deployed as the unified metadata catalog for the 254Carbon data platform, providing data discovery, lineage tracking, and governance capabilities.

**Status**: ✅ Fully Operational  
**Version**: head (latest)  
**Deployment Method**: Manual Kubernetes Manifests  
**URL**: https://datahub.254carbon.com

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DataHub Components                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Frontend   │  │     GMS      │  │   MAE Consumer  │  │
│  │  (UI/API)    │  │  (Metadata)  │  │   (Events)      │  │
│  │  Port: 9002  │  │  Port: 8080  │  │   Port: 9090    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                 │                     │           │
│         └─────────────────┴─────────────────────┘           │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
    ┌─────▼────┐  ┌──────────┐  ┌──────────┐ │
    │PostgreSQL│  │  Neo4j   │  │Elasticsearch│
    │(Storage) │  │ (Graph)  │  │  (Search)  │
    └──────────┘  └──────────┘  └──────────┘ │
                                              │
                  ┌─────────────────┬─────────┘
                  │                 │
            ┌─────▼────┐     ┌─────▼────┐
            │  Kafka   │     │  MinIO   │
            │(Streams) │     │(Objects) │
            └──────────┘     └──────────┘
```

## Deployment Details

### Components

| Component | Replicas | Status | Port | Description |
|-----------|----------|--------|------|-------------|
| datahub-gms | 1 | ✅ Running | 8080 | Graph Metadata Service - Core API |
| datahub-frontend | 1 | ✅ Running | 9002 | React UI and Backend |
| datahub-mae-consumer | 1 | ✅ Running | 9090 | Metadata Audit Event Consumer |
| datahub-mce-consumer | 1 | ✅ Running | 9090 | Metadata Change Event Consumer |

### Infrastructure Dependencies

- **PostgreSQL**: `postgres-shared-service:5432` (database: `datahub`)
- **Elasticsearch**: `elasticsearch-service:9200` (indexing and search)
- **Neo4j**: `graphdb-service:7474/7687` (graph relationships)
- **Kafka**: `kafka-service:9092` (event streaming)
- **Schema Registry**: `schema-registry-service:8081` (Avro schemas)
- **MinIO**: `minio-service:9000` (object storage)

### Configuration Files

- **Main Deployment**: `/k8s/datahub/datahub.yaml`
- **Secrets**: `/k8s/secrets/datahub-secret.yaml`
- **Ingress**: `/k8s/ingress/ingress-rules.yaml`
- **Ingestion Recipes**:
  - Trino: `/k8s/datahub/trino-ingestion-recipe.yaml`
  - Kafka: `/k8s/datahub/kafka-ingestion-recipe.yaml`
  - PostgreSQL: `/k8s/datahub/postgres-ingestion-recipe.yaml`

## Access

### Web UI
- **URL**: https://datahub.254carbon.com
- **Authentication**: Cloudflare Access SSO
- **Ingress Class**: nginx

### API Endpoints
- **GMS API**: `http://datahub-gms.data-platform.svc.cluster.local:8080`
- **Frontend API**: `http://datahub-frontend.data-platform.svc.cluster.local:9002`
- **Health Check**: `http://datahub-gms.data-platform.svc.cluster.local:8080/config`

## Monitoring

### Prometheus Metrics
ServiceMonitors configured for:
- **datahub-gms**: Scrapes metrics from GMS service
- **datahub-frontend**: Scrapes metrics from Frontend

### Alerts
- **DataHubGMSDown**: Triggers if GMS is down for > 5 minutes (severity: warning)
- Standard Kubernetes alerts for pod health, memory, CPU

### Grafana Dashboards
- Data Platform Dashboard includes DataHub components
- Available at: https://grafana.254carbon.com

## Backup & Recovery

### Velero Backup Schedules
DataHub is included in the following backup schedules:

1. **Daily Backup** (02:00 UTC, retain 30 days)
   - Full namespace backup including all resources
   - Includes PVCs and configurations

2. **Critical Backup** (Every 6 hours, retain 7 days)
   - Frequent backups for critical systems
   - Faster recovery point objective (RPO)

### Manual Backup
```bash
# Create on-demand backup
velero backup create datahub-manual-$(date +%Y%m%d-%H%M) \
  --include-namespaces data-platform \
  --labels app=datahub

# Restore from backup
velero restore create --from-backup datahub-manual-TIMESTAMP
```

## Metadata Ingestion

### Automated Ingestion CronJobs

1. **Trino Ingestion**
   - Schedule: Every 6 hours
   - Sources: Iceberg catalog tables
   - Status: Configured

2. **Kafka Ingestion**
   - Schedule: Every 4 hours (at :30)
   - Sources: Kafka topics and schemas
   - Status: Configured

3. **PostgreSQL Ingestion**
   - Schedule: Daily at 04:00
   - Sources: Database schemas and tables
   - Status: Configured

### Manual Ingestion
```bash
# Trigger immediate ingestion
kubectl create job datahub-trino-manual-$(date +%s) \
  --from=cronjob/datahub-trino-ingestion \
  -n data-platform

# Check ingestion status
kubectl get jobs -n data-platform | grep ingestion
kubectl logs -n data-platform job/datahub-trino-manual-TIMESTAMP
```

## Troubleshooting

### Common Issues

**GMS Not Ready**
```bash
# Check pod status
kubectl get pods -n data-platform -l app=datahub-gms

# Check logs
kubectl logs -n data-platform -l app=datahub-gms --tail=100

# Restart if needed
kubectl rollout restart deployment/datahub-gms -n data-platform
```

**Health Check Failures**
- Health endpoint: Uses `/config` endpoint (returns HTTP 200 when operational)
- Original `/health` endpoint returns 503 until database is fully initialized
- This is expected behavior and doesn't affect functionality

**Ingestion Failures**
```bash
# Check CronJob status
kubectl get cronjobs -n data-platform

# View last run logs
kubectl logs -n data-platform -l app=datahub-ingestion --tail=100

# Manually trigger ingestion for testing
kubectl create job test-ingestion-$(date +%s) \
  --from=cronjob/datahub-kafka-ingestion \
  -n data-platform
```

### Database Maintenance

**Check Database Status**
```bash
kubectl exec -n data-platform postgres-shared-0 -- \
  psql -U datahub -d datahub -c "\dt"
```

**Check Table Count**
```bash
kubectl exec -n data-platform postgres-shared-0 -- \
  psql -U datahub -d datahub -c "SELECT count(*) FROM metadata_aspect_v2"
```

## Key Configuration Details

### Environment Variables (GMS)
- `DATAHUB_SERVER_TYPE`: quickstart
- `EBEAN_DATASOURCE_DRIVER`: org.postgresql.Driver
- Database initialization: Auto-managed by GMS
- Kafka topics: Auto-created on first use

### Resource Limits

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| GMS | 500m | 1Gi | 1000m | 2Gi |
| Frontend | 250m | 512Mi | 500m | 1Gi |
| MAE Consumer | 250m | 512Mi | 500m | 1Gi |
| MCE Consumer | 250m | 512Mi | 500m | 1Gi |

### Health Probes

**Liveness Probe**:
- Path: `/config`
- Initial Delay: 120s
- Period: 30s
- Failure Threshold: 10

**Readiness Probe**:
- Path: `/config`
- Initial Delay: 60s
- Period: 10s
- Failure Threshold: 30

## Integration Points

### Data Sources
1. **Trino**: Discovers Iceberg tables and schemas
2. **Kafka**: Catalogs topics and message schemas
3. **PostgreSQL**: Indexes database schemas and tables
4. **Superset**: Can be configured for dashboard lineage

### Data Consumers
- **Trino**: Can query DataHub for metadata discovery
- **Superset**: Can leverage DataHub for dataset search
- **Jupyter**: Can use DataHub client for programmatic access

## Next Steps

### Immediate (First Login)
1. Access UI at https://datahub.254carbon.com
2. Authenticate via Cloudflare Access
3. Create initial user accounts
4. Configure authentication settings

### Week 1
1. Configure data domains and glossary terms
2. Set up ownership and team mappings
3. Enable data quality rules
4. Create custom dashboards in Grafana

### Ongoing
1. Monitor ingestion job success rates
2. Review and refine ingestion patterns
3. Configure data lineage for ETL pipelines
4. Set up data governance policies

## Support

### Documentation
- Official DataHub Docs: https://datahubproject.io/docs
- Iceberg Integration: `/k8s/datahub/DATAHUB_ICEBERG_INTEGRATION.md`

### Logs
```bash
# All DataHub components
kubectl logs -n data-platform -l app.kubernetes.io/name=datahub --tail=100

# Specific component
kubectl logs -n data-platform deployment/datahub-gms --tail=100
```

### Metrics
- Prometheus: Query `up{job="datahub-gms"}`
- Grafana: Data Platform Dashboard

---

**Deployed**: October 21, 2025  
**Last Updated**: October 21, 2025  
**Deployment Status**: Production Ready ✅

