# Iceberg REST Catalog Deployment Guide

## Overview

This guide documents the deployment and configuration of Apache Iceberg REST Catalog for the HMCo data platform. The Iceberg REST Catalog provides a unified metadata management layer for the data lake.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         Iceberg REST Catalog (Port 8181)             │
├─────────────────────────────────────────────────────┤
│  ├─ PostgreSQL (iceberg_rest database)              │
│  ├─ MinIO (s3://iceberg-warehouse/)                 │
│  └─ Table Metadata Management                       │
├─────────────────────────────────────────────────────┤
│  Integrations:                                       │
│  ├─ Trino (SQL Query Engine)                        │
│  ├─ DataHub (Metadata Catalog)                      │
│  └─ SeaTunnel (Data Integration)                    │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

Before deployment, ensure:

1. **Secrets created** (verify with kubectl):
   ```bash
   kubectl get secret -n data-platform minio-secret
   kubectl get secret -n data-platform datahub-secret
   ```

2. **MinIO running**:
   ```bash
   kubectl get pod -n data-platform -l app=minio
   ```

3. **PostgreSQL ready**:
   ```bash
   kubectl get pod -n data-platform -l app=postgres-shared
   ```

## Deployment Steps

### Step 1: Apply Secrets

```bash
cd /home/m/tff/254CARBON/HMCo
kubectl apply -f k8s/secrets/minio-secret.yaml
kubectl apply -f k8s/secrets/datahub-secret.yaml
```

### Step 2: Update PostgreSQL Configuration

The Iceberg schema is automatically initialized when PostgreSQL starts.

Verify the schema was created:
```bash
kubectl exec -it -n data-platform postgres-shared-xxx -- \
  psql -U iceberg_user -d iceberg_rest -c "\dn iceberg_catalog"
```

Expected output:
```
     List of schemas
      Name      | Owner  
  ──────────────┼────────
   iceberg_catalog | postgres
   public      | postgres
```

### Step 3: Initialize MinIO Buckets

Apply the MinIO initialization job:

```bash
kubectl apply -f k8s/data-lake/minio-init-job.yaml
```

Monitor the job:
```bash
kubectl logs -f -n data-platform job/minio-init-buckets
```

Expected output:
```
MinIO is ready!
Creating bucket: iceberg-warehouse
...
All buckets initialized successfully!
```

### Step 4: Deploy Iceberg REST Catalog

```bash
kubectl apply -f k8s/data-lake/iceberg-rest.yaml
```

### Step 5: Verify Iceberg REST Catalog

Check pod status:
```bash
kubectl get pod -n data-platform -l app=iceberg-rest-catalog
```

Check health endpoint:
```bash
kubectl port-forward -n data-platform svc/iceberg-rest-catalog 8181:8181 &
curl http://localhost:8181/v1/config
```

Expected response:
```json
{
  "defaults": {},
  "overrides": {}
}
```

## Configuration Details

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CATALOG_WAREHOUSE` | `s3://iceberg-warehouse/` | S3 data location |
| `CATALOG_URI` | `jdbc:postgresql://postgres-shared-service:5432/iceberg_rest?currentSchema=iceberg_catalog` | PostgreSQL connection (targets dedicated schema) |
| `CATALOG_JDBC_USER` | `iceberg_user` | Database user |
| `CATALOG_S3_ENDPOINT` | `http://minio-service:9000` | MinIO endpoint |
| `CATALOG_S3_PATH_STYLE_ACCESS` | `true` | MinIO compatibility |

### Image Settings

- `image: tabulario/iceberg-rest:0.6.0`
- `imagePullPolicy: IfNotPresent` (reuses cached image to avoid transient registry issues)

### Resource Requests

```yaml
limits:
  memory: "1.5Gi"
  cpu: "750m"
requests:
  memory: "512Mi"
  cpu: "250m"
```

## Health Checks

### Liveness Probe
- Endpoint: `GET /v1/config`
- Initial delay: 60s
- Period: 30s
- Failure threshold: 3

### Readiness Probe
- Endpoint: `GET /v1/config`
- Initial delay: 30s
- Period: 10s
- Failure threshold: 3

### Startup Probe
- Endpoint: `GET /v1/config`
- Initial delay: 0s
- Period: 5s
- Failure threshold: 12

## Common Operations

### Create Namespace

```bash
curl -X POST http://localhost:8181/v1/namespaces \
  -H "Content-Type: application/json" \
  -d '{"namespace": "my_database"}'
```

### List Namespaces

```bash
curl http://localhost:8181/v1/namespaces
```

### List Tables in Namespace

```bash
curl http://localhost:8181/v1/namespaces/my_database/tables
```

### View MinIO Buckets

```bash
kubectl port-forward -n data-platform svc/minio-service 9000:9000 &
# Access MinIO console at http://localhost:9001
# Username: minioadmin
# Password: minioadmin123
```

## Troubleshooting

### Pod in CrashLoopBackOff

Check logs:
```bash
kubectl logs -n data-platform iceberg-rest-catalog-xxx
```

Common issues:
- **PostgreSQL connection failed**: Verify postgres-shared pod is running
- **MinIO credentials error**: Verify minio-secret is applied
- **Memory errors**: Increase JAVA_OPTS memory limits

### REST API Timeouts

Check catalog service:
```bash
kubectl describe svc iceberg-rest-catalog -n data-platform
kubectl get endpoints iceberg-rest-catalog -n data-platform
```

### Database Connection Issues

Test PostgreSQL connection:
```bash
kubectl exec -it -n data-platform postgres-shared-xxx -- \
  psql -U iceberg_user -d iceberg_rest -c "SELECT 1"
```

## Monitoring

### View Service Endpoints

```bash
kubectl get endpoints -n data-platform iceberg-rest-catalog
```

### Check Resource Usage

```bash
kubectl top pod -n data-platform -l app=iceberg-rest-catalog
```

### View Recent Events

```bash
kubectl get events -n data-platform --sort-by='.lastTimestamp' | tail -10
```

## Next Steps

1. **Configure Trino**: Update Trino catalog to use Iceberg
2. **Setup DataHub Ingestion**: Configure DataHub to discover Iceberg metadata
3. **Create Test Tables**: Create sample Iceberg tables via Trino
4. **Enable SeaTunnel**: Configure SeaTunnel for data ingestion

## References

- [Apache Iceberg REST Catalog Spec](https://iceberg.apache.org/rest-catalog-spec/)
- [Tabulario Iceberg REST](https://github.com/tabular-io/iceberg-rest-image)
- [DataHub Iceberg Integration](https://docs.datahub.com/docs/generated/ingestion/sources/iceberg)
- [Trino Iceberg Connector](https://trino.io/docs/current/connector/iceberg.html)

## Support

For issues or questions:
1. Check the logs: `kubectl logs -f -n data-platform iceberg-rest-catalog-xxx`
2. Verify connectivity: `curl -v http://iceberg-rest-catalog:8181/v1/config`
3. Review configuration: `kubectl get deployment iceberg-rest-catalog -n data-platform -o yaml`
