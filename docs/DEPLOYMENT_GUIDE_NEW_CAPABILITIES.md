# Deployment Guide: 10 Advanced Data Platform Capabilities

This guide provides step-by-step instructions for deploying the new capabilities to your HMCo data platform.

## Prerequisites

- Kubernetes cluster with ArgoCD installed
- Helm 3.x installed
- kubectl configured to access cluster
- PostgreSQL database available (for lakeFS and Marquez)
- Kafka/Redpanda available (for Schema Registry and Debezium)
- Vault installed (for column-level security)

## Deployment Order

Deploy capabilities in this order to handle dependencies:

### Phase 1: Infrastructure (15-30 minutes)

#### 1.1 Deploy lakeFS

```bash
# Create namespace if not exists
kubectl create namespace data-platform

# Create secrets for lakeFS
kubectl create secret generic lakefs-postgres \
  --namespace data-platform \
  --from-literal=connection-string="postgresql://user:pass@postgres:5432/lakefs"

kubectl create secret generic lakefs-auth \
  --namespace data-platform \
  --from-literal=secret-key=$(openssl rand -hex 32)

# Enable in values
cat <<EOF > helm/values/lakefs-values.yaml
lakefs:
  enabled: true
  database:
    connectionString: ""  # Will use secret
  blockstore:
    type: s3
    s3:
      endpoint: "http://minio-service:9000"
      bucket: "lakefs-data"
EOF

# Deploy via ArgoCD or Helm
helm upgrade --install data-platform \
  helm/charts/data-platform \
  --namespace data-platform \
  --values helm/values/lakefs-values.yaml
```

Verify:
```bash
kubectl get pods -n data-platform -l app=lakefs
kubectl logs -n data-platform -l app=lakefs --tail=50
```

#### 1.2 Deploy Schema Registry

```bash
# Create streaming namespace
kubectl create namespace streaming

# Deploy schema registry
helm upgrade --install streaming \
  helm/charts/streaming \
  --namespace streaming \
  --set schema-registry.enabled=true \
  --set schema-registry.kafka.bootstrapServers="kafka-broker:9092"
```

Verify:
```bash
# Check Schema Registry health
curl http://schema-registry.streaming.svc.cluster.local:8081/

# List subjects (should be empty initially)
curl http://schema-registry.streaming.svc.cluster.local:8081/subjects
```

#### 1.3 Deploy Marquez (OpenLineage)

```bash
# Create secret for Marquez database
kubectl create secret generic marquez-postgres \
  --namespace data-platform \
  --from-literal=password=$(openssl rand -base64 32)

# Deploy Marquez
helm upgrade --install data-platform \
  helm/charts/data-platform \
  --namespace data-platform \
  --set marquez.enabled=true
```

Verify:
```bash
# Check Marquez API
curl http://marquez-service.data-platform.svc.cluster.local:5000/api/v1/namespaces

# Access Marquez Web UI
kubectl port-forward -n data-platform svc/marquez-web 3000:3000
# Open http://localhost:3000
```

#### 1.4 Deploy Vault Transform

```bash
# Create vault-transform namespace
kubectl create namespace security

# Deploy Vault Transform
helm upgrade --install vault-transform \
  helm/charts/security/vault-transform \
  --namespace security
```

Configure transformations:
```bash
# Enable Transform secrets engine
kubectl exec -n security vault-0 -- vault secrets enable transform

# Create transformations (example)
kubectl exec -n security vault-0 -- vault write transform/transformation/email-mask \
  type=masking \
  template="user-****@****.com" \
  masking_character="*"
```

### Phase 2: Data Movement (20-40 minutes)

#### 2.1 Deploy Debezium CDC

```bash
# Create secrets for source databases
kubectl create secret generic debezium-postgres \
  --namespace streaming \
  --from-literal=username=debezium \
  --from-literal=password=$(openssl rand -base64 32)

# Deploy Debezium
helm upgrade --install streaming \
  helm/charts/streaming \
  --namespace streaming \
  --set debezium.enabled=true
```

Configure connectors:
```bash
# Create PostgreSQL CDC connector
curl -X POST http://debezium.streaming.svc.cluster.local:8083/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "postgres-trading-cdc",
    "config": {
      "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
      "database.hostname": "postgres",
      "database.port": "5432",
      "database.user": "debezium",
      "database.dbname": "trading",
      "database.server.name": "hmco-trading-db",
      "table.include.list": "public.trades,public.positions"
    }
  }'
```

Verify:
```bash
# Check connector status
curl http://debezium.streaming.svc.cluster.local:8083/connectors/postgres-trading-cdc/status

# Check Kafka topics
kubectl exec -n streaming kafka-0 -- kafka-topics.sh --list --bootstrap-server localhost:9092 | grep cdc
```

#### 2.2 Update UIS to 1.2

```bash
# Update ingestion specs to use schema registry
cat <<EOF > sdk/uis/examples/lmp-with-schema.yaml
name: lmp-prices-v2
provider:
  name: iso-api
  type: rest_api
schemaRef: "hmco.energy.lmp_prices-v1"
compatMode: "BACKWARD"
output:
  path: "lakefs://hmco-curated@dev/energy/lmp_prices/"
  format: "iceberg"
EOF

# Register schema in Schema Registry
curl -X POST http://schema-registry:8081/subjects/hmco.energy.lmp_prices-v1/versions \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d '{
    "schema": "{\"type\":\"record\",\"name\":\"LMPPrice\",...}"
  }'
```

### Phase 3: Analytics (15-30 minutes)

#### 3.1 Deploy dbt

```bash
# Install dbt in runner pod or CI/CD
pip install dbt-core dbt-trino dbt-clickhouse

# Configure profiles
export CLICKHOUSE_PASSWORD=$(kubectl get secret -n data-platform clickhouse-password -o jsonpath='{.data.password}' | base64 -d)

# Run dbt
cd analytics/dbt
dbt deps
dbt run --target prod
dbt test
```

Schedule dbt runs:
```yaml
# Create CronJob for dbt
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dbt-daily-run
  namespace: data-platform
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dbt
            image: ghcr.io/dbt-labs/dbt-core:latest
            command: ["dbt", "run", "--profiles-dir", "/app"]
            volumeMounts:
            - name: dbt-project
              mountPath: /app
          volumes:
          - name: dbt-project
            configMap:
              name: dbt-project
```

### Phase 4: Services (10-20 minutes)

#### 4.1 Deploy Data Sharing Service

```bash
# Deploy data-sharing service
kubectl create namespace data-sharing

kubectl create deployment data-sharing \
  --namespace data-sharing \
  --image=python:3.11 \
  -- uvicorn services.data_sharing.app.main:app --host 0.0.0.0 --port 8000

kubectl expose deployment data-sharing \
  --namespace data-sharing \
  --port=8000 \
  --type=ClusterIP
```

Create ingress:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: data-sharing
  namespace: data-sharing
spec:
  rules:
  - host: data-sharing.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: data-sharing
            port:
              number: 8000
```

#### 4.2 Deploy ClickHouse MV Optimizer

```bash
# Deploy as CronJob
kubectl create configmap ch-mv-policy \
  --namespace data-platform \
  --from-file=policy.yaml=services/ch-mv-optimizer/config/policy.yaml

kubectl create cronjob ch-mv-optimizer \
  --namespace data-platform \
  --image=python:3.11 \
  --schedule="*/30 * * * *" \
  -- python services/ch-mv-optimizer/app/optimizer.py
```

#### 4.3 Deploy Cost Attribution Service

```bash
# Deploy cost-attribution
kubectl create deployment cost-attribution \
  --namespace data-platform \
  --image=python:3.11 \
  -- python services/cost-attribution/app/collector.py

# Create ConfigMap for cost config
kubectl create configmap cost-config \
  --namespace data-platform \
  --from-literal=trino_query_cost=0.001 \
  --from-literal=trino_tb_scanned_cost=5.00
```

Import Grafana dashboard:
```bash
# Import cost dashboard
kubectl create configmap grafana-cost-dashboard \
  --namespace monitoring \
  --from-file=dashboard.json=services/cost-attribution/dashboards/cost-dashboard.json
```

### Phase 5: Autoscaling (5-10 minutes)

#### 5.1 Install KEDA (if not already installed)

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace
```

#### 5.2 Enable Trino Autoscaling

```bash
# Update Trino values to enable KEDA
helm upgrade --install data-platform \
  helm/charts/data-platform \
  --namespace data-platform \
  --set trino.keda.enabled=true \
  --set trino.keda.minReplicas=2 \
  --set trino.keda.maxReplicas=10
```

Verify:
```bash
kubectl get scaledobjects -n data-platform
kubectl describe scaledobject trino-worker-scaler -n data-platform
```

## Post-Deployment Configuration

### Configure Column-Level Security

#### For ClickHouse:
```sql
-- Connect to ClickHouse
clickhouse-client

-- Create row policy
CREATE ROW POLICY trading_desk_filter ON trading.trades
FOR SELECT USING desk = currentUser()
TO trader;

-- Create column mask
-- (Note: ClickHouse doesn't have native column masking, use views)
CREATE VIEW trading.trades_masked AS
SELECT
  trade_id,
  desk,
  regexp_replace(trader_email, '(.{3}).*@', '$1***@') AS trader_email,
  amount
FROM trading.trades;
```

#### For Trino:
```sql
-- Configure row filters in catalog properties
-- Edit iceberg.properties:
iceberg.security-manager=io.trino.plugin.iceberg.security.IcebergSecurityManager
iceberg.row-filters=file:///etc/trino/row-filters.json

-- Create row-filters.json:
{
  "filters": [
    {
      "table": "trading.trades",
      "expression": "trader_id = '${USER}' OR '${USER}' IN (SELECT user FROM admin_users)"
    }
  ]
}
```

### Configure lakeFS Branches

```bash
# Install lakectl CLI
curl -L https://github.com/treeverse/lakeFS/releases/latest/download/lakectl-linux-amd64 \
  -o /usr/local/bin/lakectl
chmod +x /usr/local/bin/lakectl

# Configure lakectl
lakectl config

# Create initial branches
lakectl branch create \
  lakefs://hmco-curated/dev \
  --source lakefs://hmco-curated/main

lakectl branch create \
  lakefs://hmco-curated/stage \
  --source lakefs://hmco-curated/main

lakectl branch create \
  lakefs://hmco-curated/prod \
  --source lakefs://hmco-curated/main
```

### Register Partners for Data Sharing

```bash
# Register a partner
curl -X POST http://data-sharing:8000/partners/ \
  -H "Content-Type: application/json" \
  -d '{
    "partner_id": "acme-corp",
    "partner_name": "ACME Corporation",
    "email": "data@acme.com",
    "organization": "ACME Trading"
  }'

# Grant dataset access
curl -X POST http://data-sharing:8000/entitlements/ \
  -H "Content-Type: application/json" \
  -d '{
    "partner_id": "acme-corp",
    "dataset_type": "lmp",
    "dataset_name": "lmp_prices",
    "scope": "read_only",
    "expires_at": "2025-12-31T23:59:59Z"
  }'

# Issue access token
curl -X POST http://data-sharing:8000/tokens/ \
  -d "partner_id=acme-corp" \
  -d "datasets[]=lmp_prices" \
  -d "duration_hours=24"
```

## Monitoring

### Check Service Health

```bash
# lakeFS
kubectl get pods -n data-platform -l app=lakefs
curl http://lakefs.data-platform.svc.cluster.local:8000/healthcheck

# Schema Registry
curl http://schema-registry.streaming.svc.cluster.local:8081/

# Marquez
curl http://marquez.data-platform.svc.cluster.local:5000/api/v1/health

# Debezium
curl http://debezium.streaming.svc.cluster.local:8083/connectors

# Data Sharing
curl http://data-sharing.data-sharing.svc.cluster.local:8000/health
```

### View Metrics in Grafana

Access dashboards:
- Cost Attribution: http://grafana.254carbon.com/d/cost-attribution
- lakeFS Metrics: http://grafana.254carbon.com/d/lakefs
- CDC Lag: http://grafana.254carbon.com/d/debezium-lag
- dbt Runs: http://grafana.254carbon.com/d/dbt-runs

## Troubleshooting

### lakeFS Issues

```bash
# Check logs
kubectl logs -n data-platform -l app=lakefs --tail=100

# Common issues:
# - Database connection: verify PostgreSQL connectivity
# - S3/MinIO access: check credentials and endpoint
# - Memory: increase resources if OOM errors
```

### Schema Registry Issues

```bash
# Check connectivity to Kafka
kubectl exec -n streaming schema-registry-0 -- \
  curl -v kafka-broker:9092

# Check schema registry logs
kubectl logs -n streaming -l app=schema-registry --tail=100

# List registered schemas
curl http://schema-registry:8081/subjects
```

### Debezium Issues

```bash
# Check connector status
curl http://debezium:8083/connectors/postgres-trading-cdc/status | jq

# Check connector logs
kubectl logs -n streaming -l app=debezium --tail=100

# Common issues:
# - WAL not enabled: ALTER SYSTEM SET wal_level = 'logical';
# - Permissions: GRANT SELECT ON ALL TABLES IN SCHEMA public TO debezium;
# - Network: verify source database connectivity
```

### dbt Issues

```bash
# Test connection
dbt debug --profiles-dir analytics/dbt

# Run with verbose logging
dbt run --target dev --debug

# Common issues:
# - Trino connection: verify host and port
# - Permissions: ensure dbt user has CREATE TABLE rights
# - Dependencies: run `dbt deps` to install packages
```

## Rollback Procedures

### Rollback lakeFS Deployment

```bash
helm rollback data-platform <revision> --namespace data-platform
```

### Rollback Schema Changes

```bash
# Schema Registry maintains history
curl -X DELETE http://schema-registry:8081/subjects/hmco.energy.lmp_prices-v2
```

### Rollback Data in lakeFS

```bash
# Revert branch to previous commit
lakectl revert lakefs://hmco-curated@prod --commit <previous-commit>
```

## Security Checklist

- [ ] All secrets created and configured
- [ ] Network policies applied to restrict inter-service communication
- [ ] Ingress configured with TLS certificates
- [ ] Row and column policies configured
- [ ] Audit logging enabled and forwarded to SIEM
- [ ] Token expiry configured for data sharing
- [ ] Vault Transform transformations tested
- [ ] RBAC configured for all services

## Performance Tuning

### lakeFS
- Increase `replicaCount` for high availability
- Scale database for better performance
- Enable caching for frequently accessed branches

### Schema Registry
- Set `replicaCount` to 3+ for production
- Increase heap size for large schemas
- Configure topic replication factor to 3

### Debezium
- Tune `max.batch.size` and `max.queue.size`
- Increase parallelism for large tables
- Monitor CDC lag and adjust resources

### dbt
- Use incremental materialization for large tables
- Run models in parallel with `--threads`
- Create indexes on frequently joined columns

### ClickHouse MV Optimizer
- Adjust `min_query_count` threshold
- Tune storage limits based on cluster capacity
- Review and adjust MV patterns

## Next Steps

1. **Training**: Train teams on new capabilities
2. **Migration**: Migrate existing pipelines to use lakeFS branches
3. **Monitoring**: Set up alerts for all services
4. **Documentation**: Update runbooks with service-specific procedures
5. **Optimization**: Monitor usage and tune configurations
6. **Expansion**: Extend capabilities based on feedback

## Support

For issues or questions:
- Internal Wiki: https://wiki.254carbon.com/data-platform
- Slack: #data-platform-support
- Email: data-platform@254carbon.com
