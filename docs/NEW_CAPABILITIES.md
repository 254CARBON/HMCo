# Advanced Data Platform Capabilities

This document describes 10 new capabilities that enhance the HMCo data platform for scale, safety, and partner distribution.

## Overview

These capabilities unlock:
- **Reversibility & Safety**: lakeFS + OpenLineage → no more blind changes
- **Stability Under Change**: Schema contracts + CDC done right
- **Speed Without Toil**: Adaptive MVs + autoscaling
- **Monetization-Ready**: Partner sharing & chargeback
- **Compliance**: Column-level controls with auditable lineage

---

## 1. Data Versioning with lakeFS

**Location**: `helm/charts/data-platform/charts/lakefs/`

**Purpose**: Git-like version control for data lakes with branch/merge/rollback

### Features
- Branch/merge/rollback for data
- Data PRs with quality gates
- Atomic operations with ACID guarantees
- Zero-copy branching

### Quick Start
```bash
# Enable in data-platform values
lakefs:
  enabled: true
  database:
    connectionString: "postgresql://..."
```

### Workflow
```bash
# Create dev branch
lakectl branch create lakefs://hmco-curated@dev

# Write to dev
spark.write.path("lakefs://hmco-curated@dev/...")

# Merge to prod after DQ passes
lakectl merge --from dev --to prod
```

**Documentation**: See `helm/charts/data-platform/charts/lakefs/README.md`

---

## 2. Schema Registry

**Location**: `helm/charts/streaming/schema-registry/`

**Purpose**: Enforce schema contracts for streaming and batch data

### Features
- Avro/Protobuf/JSON schema management
- Compatibility checking (BACKWARD, FORWARD, FULL)
- CI integration to block breaking changes
- Integration with Kafka/Redpanda

### UIS Integration
```yaml
# sdk/uis/schema/uis-1.2.json adds:
schemaRef: "hmco.trading.orders-v1"
compatMode: "BACKWARD"
```

### CI Check
```yaml
# .github/workflows/schema-compatibility.yml
# Validates schema compatibility on PR
```

**Documentation**: Schema registry is industry-standard Confluent implementation

---

## 3. OpenLineage with Marquez

**Location**: `helm/charts/data-platform/charts/marquez/`

**Purpose**: End-to-end data lineage tracking

### Features
- OpenLineage standard compliance
- Lineage UI showing sources → transforms → outputs
- Integration with Spark, Flink, Trino, DolphinScheduler
- Track code commit → data output

### Instrumentation
```python
# Spark jobs automatically emit lineage
spark.conf.set("spark.extraListeners", 
               "io.openlineage.spark.agent.OpenLineageSparkListener")
```

### UI Access
- Lineage graph: https://lineage.254carbon.com
- Query: "Where did this number come from?"
- Answer: Full DAG from raw source to final table

**Documentation**: [OpenLineage Docs](https://openlineage.io/)

---

## 4. CDC with Debezium

**Location**: `helm/charts/streaming/debezium/`

**Purpose**: Near-real-time change data capture from databases

### Features
- PostgreSQL and MySQL connectors
- Emit changes to Kafka topics
- Automatic Iceberg upserts
- ClickHouse materialization

### Configuration
```yaml
connectors:
  postgres:
    table.include.list: "public.trades,public.positions"
    topic.prefix: "cdc.trading"
outputTargets:
  iceberg:
    writeMode: "upsert"
    primaryKeys: ["trade_id"]
```

### Result
Insert/update/delete in source DB → reflected in curated tables within minutes

**Documentation**: [Debezium Docs](https://debezium.io/)

---

## 5. Analytics Modeling with dbt

**Location**: `analytics/dbt/`

**Purpose**: Declarative analytics models with tests and documentation

### Features
- Models for LMP, weather, outages
- Dual targets: Trino (Iceberg) + ClickHouse
- Automated tests and documentation
- CI integration

### Example Model
```sql
-- analytics/dbt/models/marts/lmp/lmp_hourly_summary.sql
{{ config(materialized='table') }}

SELECT
  location_id,
  date_trunc('hour', interval_start_time) as hour,
  avg(lmp_price) as avg_lmp
FROM {{ ref('stg_lmp_data') }}
GROUP BY 1, 2
```

### Usage
```bash
cd analytics/dbt
dbt run --select lmp_hourly_summary
dbt test
```

**Documentation**: See `analytics/dbt/README.md`

---

## 6. Partner Data Sharing

**Location**: `services/data-sharing/`

**Purpose**: Secure, token-based data sharing with external partners

### Features
- Partner registration and entitlements
- Time-scoped access tokens
- Row-level and column-level filtering
- Complete audit logging

### API
```python
# Register partner
POST /partners/
{
  "partner_id": "acme-corp",
  "organization": "ACME Trading"
}

# Grant entitlement
POST /entitlements/
{
  "partner_id": "acme-corp",
  "dataset_name": "lmp_prices",
  "scope": "read_only",
  "expires_at": "2024-12-31"
}

# Issue token
POST /tokens/
{
  "partner_id": "acme-corp",
  "datasets": ["lmp_prices"],
  "duration_hours": 24
}
```

### Access Pattern
```python
# Partner queries via Trino with token
trino --header "Authorization: Bearer ${TOKEN}"
SELECT * FROM iceberg.curated.lmp_prices
```

**Documentation**: Built with FastAPI, see `services/data-sharing/app/main.py`

---

## 7. Adaptive Materialization for ClickHouse

**Location**: `services/ch-mv-optimizer/`

**Purpose**: Automatically create and manage materialized views based on query patterns

### Features
- Analyzes query_log for hot queries
- Auto-creates MVs for common patterns
- Enforces storage guardrails
- Drops unused MVs

### Policy
```yaml
# services/ch-mv-optimizer/config/policy.yaml
optimizer:
  min_query_count: 100
  min_avg_query_time_ms: 500
  target_p95_ms: 200

guardrails:
  max_storage_percent: 20
  max_mvs_per_table: 5
  unused_retention_days: 7
```

### Patterns
- Hourly aggregates
- Daily rollups
- Top-K queries

**Result**: p95 latency < 200ms on hot queries

---

## 8. Column-Level Security

**Location**: `helm/charts/security/vault-transform/`

**Purpose**: Tokenization and masking for PII and sensitive data

### Features
- Vault Transform secrets engine
- Masking patterns (email, SSN, credit card)
- Reversible tokenization
- ClickHouse and Trino policies

### Configuration
```yaml
transformations:
  - name: email-mask
    type: masking
    template: "user-****@****.com"

clickhouse:
  column_masks:
    - table: traders
      column: email
      mask_type: email-mask
      apply_to_roles: [viewer]
```

### ClickHouse Policies
```sql
-- Row-level security
CREATE ROW POLICY trading_desk_filter ON trading.trades
FOR SELECT USING desk = currentUser()
TO trader;

-- Column masking
CREATE COLUMN POLICY email_mask ON trading.traders
MODIFY email TO 'MASKED' FOR viewer;
```

**Documentation**: [Vault Transform](https://www.vaultproject.io/docs/secrets/transform)

---

## 9. Workload Autoscaling

**Location**: `helm/charts/data-platform/charts/trino/templates/keda-scaledobject.yaml`

**Purpose**: Auto-scale query engines based on workload

### Features
- KEDA-based autoscaling for Trino workers
- Scale on queue depth and CPU
- Resource groups (interactive, ETL, adhoc)
- Spark on-demand with spot instances

### Configuration
```yaml
keda:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  queueThreshold: 10
  cpuThreshold: 70
```

### Resource Groups
```json
{
  "subGroups": [
    {"name": "interactive", "softMemoryLimit": "40%"},
    {"name": "etl", "softMemoryLimit": "50%"},
    {"name": "adhoc", "softMemoryLimit": "10%"}
  ]
}
```

**Result**: Auto scale-out during peak hours, scale-to-zero overnight

---

## 10. Usage Analytics & Chargeback

**Location**: `services/cost-attribution/`

**Purpose**: Track and attribute costs by user, dataset, and team

### Features
- Collect metrics from Trino, ClickHouse, MinIO, Spark
- Cost calculation per query/TB-scanned/GB-stored
- Aggregation by user, dataset, team
- Grafana dashboards

### Metrics
```python
# Prometheus metrics
data_platform_cost_by_user{user="analyst1", service="trino"}
data_platform_cost_by_dataset{dataset="trading.trades"}
data_platform_queries_total{service="clickhouse", user="etl_bot"}
data_platform_bytes_scanned_total{dataset="lmp_prices"}
```

### Dashboard
- Total platform cost
- Top 10 users by cost
- Top 10 expensive datasets
- Cost per TB scanned
- Cost per query

### Usage
```python
# Generate daily report
python services/cost-attribution/app/collector.py
```

**Dashboard**: `services/cost-attribution/dashboards/cost-dashboard.json`

---

## Integration Summary

### CI/CD Workflows
- **schema-compatibility.yml**: Validates schema changes on PR
- **dbt-test.yml**: Runs dbt compile and test on PR

### Monitoring
All services expose Prometheus metrics:
- lakeFS: branch operations, merge success rate
- Schema Registry: schema registrations, compatibility checks
- Marquez: lineage graph depth, event throughput
- Debezium: CDC lag, connector health
- dbt: model execution time, test pass rate
- Data Sharing: token issuance, access logs
- MV Optimizer: MVs created/dropped, storage usage
- Cost Attribution: total cost, cost by dimension
- Autoscaling: replica count, queue depth

### Portal Integration
Add links to portal UI:
- lakeFS: Data version browser
- Marquez: Lineage explorer
- Data Sharing: Partner management
- Cost Attribution: Chargeback reports

---

## Deployment Order

1. **Infrastructure**
   - lakeFS (requires PostgreSQL)
   - Schema Registry (requires Kafka)
   - Marquez (requires PostgreSQL)
   - Vault Transform

2. **Ingestion**
   - Update UIS to 1.2 with schema registry
   - Configure Debezium connectors
   - Point ingestion to lakeFS branches

3. **Analytics**
   - Deploy dbt models
   - Configure MV optimizer
   - Set up resource groups

4. **Access & Cost**
   - Configure data sharing service
   - Set up column-level security
   - Deploy cost attribution
   - Enable autoscaling

---

## Testing Strategy

Each capability includes tests:
- **Unit tests**: Service logic
- **Integration tests**: Cross-component workflows
- **E2E tests**: Full data pipelines
- **Performance tests**: Scalability validation

Run tests:
```bash
pytest tests/integration/test_lakefs_workflow.py
pytest tests/integration/test_schema_registry.py
pytest tests/integration/test_cdc_pipeline.py
pytest tests/integration/test_dbt_models.py
```

---

## Security Considerations

1. **Secrets Management**: All credentials via Vault
2. **Network Policies**: Restrict inter-service communication
3. **Audit Logging**: All access logged to central SIEM
4. **Least Privilege**: Row/column policies enforced
5. **Token Expiry**: All access tokens time-limited

---

## References

- [lakeFS Documentation](https://docs.lakefs.io/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/)
- [OpenLineage](https://openlineage.io/)
- [Debezium](https://debezium.io/)
- [dbt Documentation](https://docs.getdbt.com/)
- [Vault Transform](https://www.vaultproject.io/docs/secrets/transform)
- [KEDA Autoscaling](https://keda.sh/)
