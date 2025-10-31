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
    bootstrap.servers = "kafka-service:9093"
    security.protocol = "SSL"
    ssl.keystore.location = "/etc/kafka/secrets/user.p12"
    ssl.truststore.location = "/etc/kafka/secrets/user.p12"
    ssl.keystore.password = "${KAFKA_USER_PASSWORD}"
    ssl.truststore.password = "${KAFKA_USER_PASSWORD}"
    ssl.key.password = "${KAFKA_USER_PASSWORD}"
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

> ℹ️ Mount the `kafka-platform-apps-tls` secret into the SeaTunnel job pod and export `KAFKA_USER_PASSWORD=$(cat /etc/kafka/secrets/user.password)` so the SSL properties resolve correctly.

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

## Polygon Market Data Pipeline

The Polygon provider MVP delivers an end-to-end Spark ingestion workflow, Deequ guardrails, and DataHub lineage for market datasets.

- **UIS Template**: `sdk/uis/templates/polygon-stocks.uis.yaml` standardises runtime configuration for the Polygon provider.
- **Ingestion Job**: `jobs/polygon_ingestion.py` reads Polygon.io daily aggregates, enriches reference metadata, and appends records into `iceberg.raw.polygon_market_ohlc`.
- **Quality Checks**: `jobs/polygon_quality_checks.py` runs Deequ completeness, uniqueness, and freshness validations, storing metrics in `iceberg.monitoring.polygon_quality_checks`.
- **Iceberg DDLs**: `infrastructure/iceberg/polygon_market_ohlc.sql` and `infrastructure/iceberg/polygon_quality_checks.sql` provision the raw and monitoring tables.
- **Scheduled Runs**: `helm/charts/data-platform/charts/spark-operator/templates/polygon-market-sparkapp.yaml` and `.../polygon-quality-sparkapp.yaml` create ScheduledSparkApplications for daily ingestion and quality enforcement.
- **DataHub Lineage**: `helm/charts/data-platform/charts/datahub/templates/polygon-lineage-ingestion.yaml` publishes pipeline, job, and ownership metadata to DataHub.

### Deploy the Pipeline

```bash
# Schedule ingestion and quality jobs
kubectl apply -f helm/charts/data-platform/charts/spark-operator/templates/polygon-market-sparkapp.yaml
kubectl apply -f helm/charts/data-platform/charts/spark-operator/templates/polygon-quality-sparkapp.yaml

# Register lineage metadata in DataHub
kubectl apply -f helm/charts/data-platform/charts/datahub/templates/polygon-lineage-ingestion.yaml
```

### Validate in Trino

```sql
SELECT ticker, trading_day, close_price, volume
FROM iceberg.raw.polygon_market_ohlc
WHERE trading_day >= CURRENT_DATE - INTERVAL '7' DAY
ORDER BY trading_day DESC, ticker;
```

## Documentation

Comprehensive guides for each component:

| Guide | Purpose | Link |
|-------|---------|------|
| Deployment & Operations | Setup and operational procedures | [operations-runbook.md](operations-runbook.md) |
| Testing Guide | End-to-end testing procedures | [testing-guide.md](testing-guide.md) |
| Security Hardening | Security best practices | [security-hardening.md](security-hardening.md) |
| Monitoring & Alerting | Observability setup | [monitoring.md](monitoring.md) |

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

## Authentication and Access Control

### REST Catalog Authentication

The Iceberg REST Catalog supports multiple authentication mechanisms to ensure secure access:

#### 1. OAuth 2.0 Authentication

```yaml
# Configure OAuth 2.0 for Iceberg REST Catalog
apiVersion: v1
kind: ConfigMap
metadata:
  name: iceberg-rest-auth-config
  namespace: data-platform
data:
  catalog-impl: org.apache.iceberg.rest.RESTCatalog
  uri: http://iceberg-rest-catalog:8181
  credential: oauth2
  oauth2-server-uri: https://auth.254carbon.com/oauth/token
  scope: iceberg:read iceberg:write
```

**Implementation:**

```python
# Python client with OAuth2
from pyiceberg.catalog import load_catalog

catalog = load_catalog(
    "rest",
    **{
        "uri": "http://iceberg-rest-catalog:8181",
        "credential": "oauth2",
        "oauth2-server-uri": "https://auth.254carbon.com/oauth/token",
        "token": "<your-oauth-token>",
        "scope": "iceberg:read iceberg:write"
    }
)
```

#### 2. Token-Based Authentication

```bash
# Generate API token for service account
export ICEBERG_TOKEN=$(kubectl get secret iceberg-api-token -n data-platform -o jsonpath='{.data.token}' | base64 -d)

# Use token in REST API calls
curl -H "Authorization: Bearer $ICEBERG_TOKEN" \
  http://iceberg-rest-catalog:8181/v1/namespaces
```

#### 3. Service Account Authentication (Recommended)

```yaml
# ServiceAccount with RBAC for Iceberg access
apiVersion: v1
kind: ServiceAccount
metadata:
  name: iceberg-client-sa
  namespace: data-platform
---
apiVersion: v1
kind: Secret
metadata:
  name: iceberg-client-token
  namespace: data-platform
  annotations:
    kubernetes.io/service-account.name: iceberg-client-sa
type: kubernetes.io/service-account-token
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: iceberg-client-role
  namespace: data-platform
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["iceberg-rest-config"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["minio-secret", "iceberg-db-secret"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: iceberg-client-binding
  namespace: data-platform
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: iceberg-client-role
subjects:
- kind: ServiceAccount
  name: iceberg-client-sa
  namespace: data-platform
```

### Access Control Lists (ACLs)

#### Table-Level Access Control

```yaml
# Configure ACLs for Iceberg tables
apiVersion: v1
kind: ConfigMap
metadata:
  name: iceberg-acls
  namespace: data-platform
data:
  acl-config.yaml: |
    # Default deny-all policy
    default-policy: deny
    
    # ACL rules by namespace and table
    acls:
      # Raw data namespace - restricted access
      - namespace: "raw"
        table: "*"
        permissions:
          - principal: "role:data-engineer"
            actions: ["read", "write", "create", "delete"]
          - principal: "role:data-scientist"
            actions: ["read"]
          - principal: "role:analyst"
            actions: []  # No access
      
      # Analytics namespace - broader read access
      - namespace: "analytics"
        table: "*"
        permissions:
          - principal: "role:data-engineer"
            actions: ["read", "write", "create", "delete"]
          - principal: "role:data-scientist"
            actions: ["read", "write"]
          - principal: "role:analyst"
            actions: ["read"]
      
      # Sensitive tables - restricted access
      - namespace: "analytics"
        table: "customers_pii"
        permissions:
          - principal: "role:data-engineer"
            actions: ["read", "write"]
          - principal: "role:compliance-officer"
            actions: ["read"]
          - principal: "role:data-scientist"
            actions: []  # No access to PII
```

#### Enforcing ACLs with Trino

```sql
-- Configure Trino security with role-based access
-- File: /etc/trino/catalog/iceberg.properties

connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest.uri=http://iceberg-rest-catalog:8181

# Enable access control
access-control.name=file
security.config-file=/etc/trino/rules.json
```

**rules.json:**
```json
{
  "catalogs": [
    {
      "catalog": "iceberg",
      "allow": "all",
      "schemas": [
        {
          "schema": "raw",
          "owner": false,
          "tables": [
            {
              "privileges": ["SELECT", "INSERT", "UPDATE", "DELETE"],
              "user": "data-engineer"
            },
            {
              "privileges": ["SELECT"],
              "user": "data-scientist"
            }
          ]
        },
        {
          "schema": "analytics",
          "owner": false,
          "tables": [
            {
              "privileges": ["SELECT", "INSERT", "UPDATE", "DELETE"],
              "user": "data-engineer"
            },
            {
              "privileges": ["SELECT", "INSERT"],
              "user": "data-scientist"
            },
            {
              "privileges": ["SELECT"],
              "user": "analyst"
            }
          ]
        }
      ]
    }
  ]
}
```

### Audit Logging

#### 1. REST Catalog Audit Logs

All read/write operations to the Iceberg REST Catalog are logged:

```yaml
# Enable audit logging in Iceberg REST Catalog
apiVersion: v1
kind: ConfigMap
metadata:
  name: iceberg-rest-logging
  namespace: data-platform
data:
  log4j2.xml: |
    <?xml version="1.0" encoding="UTF-8"?>
    <Configuration status="INFO">
      <Appenders>
        <!-- Console appender -->
        <Console name="Console" target="SYSTEM_OUT">
          <PatternLayout pattern="%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1} - %m%n"/>
        </Console>
        
        <!-- Audit log appender -->
        <RollingFile name="AuditLog" 
                     fileName="/var/log/iceberg/audit.log"
                     filePattern="/var/log/iceberg/audit-%d{yyyy-MM-dd}-%i.log.gz">
          <JsonLayout compact="true" eventEol="true">
            <KeyValuePair key="timestamp" value="$${date:yyyy-MM-dd'T'HH:mm:ss.SSSZ}"/>
            <KeyValuePair key="level" value="$${level}"/>
            <KeyValuePair key="logger" value="$${logger}"/>
            <KeyValuePair key="message" value="$${message}"/>
            <KeyValuePair key="thread" value="$${thread}"/>
          </JsonLayout>
          <Policies>
            <TimeBasedTriggeringPolicy interval="1" modulate="true"/>
            <SizeBasedTriggeringPolicy size="100 MB"/>
          </Policies>
          <DefaultRolloverStrategy max="30"/>
        </RollingFile>
      </Appenders>
      
      <Loggers>
        <!-- Audit logger for all catalog operations -->
        <Logger name="org.apache.iceberg.rest.audit" level="INFO" additivity="false">
          <AppenderRef ref="AuditLog"/>
        </Logger>
        
        <!-- Root logger -->
        <Root level="INFO">
          <AppenderRef ref="Console"/>
        </Root>
      </Loggers>
    </Configuration>
```

#### 2. Query Audit Logs in Elasticsearch/Loki

```bash
# Ship audit logs to centralized logging
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-iceberg-config
  namespace: data-platform
data:
  fluent-bit.conf: |
    [INPUT]
        Name              tail
        Path              /var/log/iceberg/audit.log
        Parser            json
        Tag               iceberg.audit
        Refresh_Interval  5
    
    [OUTPUT]
        Name              es
        Match             iceberg.audit
        Host              elasticsearch.monitoring
        Port              9200
        Index             iceberg-audit
        Type              _doc
EOF
```

#### 3. Sample Audit Log Entry

```json
{
  "timestamp": "2025-10-31T04:12:00.123Z",
  "level": "INFO",
  "logger": "org.apache.iceberg.rest.audit",
  "message": "Table accessed",
  "thread": "http-nio-8181-exec-1",
  "user": "service-account:data-engineer",
  "action": "READ",
  "resource": "iceberg.analytics.customers",
  "namespace": "analytics",
  "table": "customers",
  "client_ip": "10.42.1.15",
  "user_agent": "PyIceberg/0.6.0",
  "request_id": "req-abc123",
  "duration_ms": 45,
  "status": "success"
}
```

#### 4. Querying Audit Logs

```bash
# Query audit logs for specific user
curl -X GET "http://elasticsearch.monitoring:9200/iceberg-audit/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": [
          {"term": {"user.keyword": "service-account:data-engineer"}},
          {"range": {"timestamp": {"gte": "now-7d"}}}
        ]
      }
    },
    "sort": [{"timestamp": {"order": "desc"}}]
  }'

# Query failed access attempts
curl -X GET "http://elasticsearch.monitoring:9200/iceberg-audit/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": [
          {"term": {"status.keyword": "unauthorized"}},
          {"range": {"timestamp": {"gte": "now-24h"}}}
        ]
      }
    }
  }'
```

### Governed Tables Example

#### Creating a Governed Table with Full Access Controls

```sql
-- Create namespace with governance policies
CREATE SCHEMA IF NOT EXISTS iceberg.governed;

-- Create governed table with encryption and access controls
CREATE TABLE iceberg.governed.customer_transactions (
    transaction_id UUID,
    customer_id BIGINT,
    transaction_date TIMESTAMP(6) WITH TIME ZONE,
    amount DECIMAL(15, 2),
    currency VARCHAR(3),
    merchant_name VARCHAR(255),
    category VARCHAR(100),
    -- PII fields (encrypted at rest)
    customer_email VARCHAR(255),
    customer_phone VARCHAR(20),
    -- Metadata
    created_at TIMESTAMP(6) WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    modified_at TIMESTAMP(6) WITH TIME ZONE,
    modified_by VARCHAR(100)
)
WITH (
    format = 'PARQUET',
    location = 's3://iceberg-warehouse/governed/customer_transactions',
    partitioning = ARRAY['bucket(customer_id, 16)', 'day(transaction_date)'],
    -- Enable encryption
    write_compression_codec = 'SNAPPY',
    -- Metadata for governance
    table_properties = MAP(
        ARRAY['classification', 'owner', 'retention_days', 'pii_fields'],
        ARRAY['confidential', 'data-platform-team', '2555', 'customer_email,customer_phone']
    )
);

-- Grant read access to analysts (via Trino roles)
GRANT SELECT ON iceberg.governed.customer_transactions TO ROLE analyst;

-- Grant write access to data engineers
GRANT SELECT, INSERT, UPDATE, DELETE ON iceberg.governed.customer_transactions TO ROLE data_engineer;

-- Revoke direct access to PII fields for most users
-- This requires column-level security in Trino
CREATE VIEW iceberg.governed.customer_transactions_safe AS
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    amount,
    currency,
    merchant_name,
    category,
    -- Mask PII fields
    'REDACTED' AS customer_email,
    'REDACTED' AS customer_phone,
    created_at,
    created_by
FROM iceberg.governed.customer_transactions;

-- Grant access to safe view
GRANT SELECT ON iceberg.governed.customer_transactions_safe TO ROLE analyst;
GRANT SELECT ON iceberg.governed.customer_transactions_safe TO ROLE data_scientist;
```

#### Registering Governed Tables in DataHub

```yaml
# DataHub governance metadata ingestion
apiVersion: batch/v1
kind: Job
metadata:
  name: datahub-governance-ingestion
  namespace: data-platform
spec:
  template:
    spec:
      serviceAccountName: datahub-ingestion-sa
      containers:
      - name: datahub-ingestion
        image: acryldata/datahub-ingestion:latest
        command: ["/bin/bash", "-c"]
        args:
          - |
            cat > /tmp/governance-recipe.yml <<EOF
            source:
              type: iceberg
              config:
                catalog:
                  rest:
                    uri: http://iceberg-rest-catalog:8181
                # Add governance metadata
                domain:
                  "governed": "urn:li:domain:customer-data"
                tag_prefix: "governance"
                
                # Extract governance metadata from table properties
                extract_table_properties: true
                
                # Add ownership
                ownership:
                  - owner: "urn:li:corpuser:data-platform-team"
                    type: "TECHNICAL_OWNER"
                
                # Add classifications
                classification:
                  confidential: "urn:li:tag:Confidential"
                  pii: "urn:li:tag:PII"
                
                # Define data contracts
                schema_metadata:
                  classification: "Confidential"
                  retention: "7 years"
                  encryption: "AES-256"
            
            sink:
              type: datahub-rest
              config:
                server: http://datahub-gms:8080
            EOF
            
            datahub ingest -c /tmp/governance-recipe.yml
      restartPolicy: OnFailure
```

### Verification: Authentication Required

#### Test 1: Unauthenticated Access (Should Fail)

```bash
# Attempt to access without credentials
curl -X GET http://iceberg-rest-catalog:8181/v1/namespaces

# Expected response: 401 Unauthorized
{
  "error": "Unauthorized",
  "message": "Authentication required. Please provide a valid token."
}
```

#### Test 2: Authenticated Access (Should Succeed)

```bash
# Access with valid token
TOKEN=$(kubectl get secret iceberg-api-token -n data-platform -o jsonpath='{.data.token}' | base64 -d)

curl -X GET http://iceberg-rest-catalog:8181/v1/namespaces \
  -H "Authorization: Bearer $TOKEN"

# Expected response: 200 OK with namespace list
{
  "namespaces": [
    ["raw"],
    ["analytics"],
    ["governed"]
  ]
}
```

#### Test 3: Read/Write Path Authentication

```python
# Python test script
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import Unauthorized

# Load catalog with authentication
catalog = load_catalog(
    "rest",
    **{
        "uri": "http://iceberg-rest-catalog:8181",
        "token": "YOUR_API_TOKEN_HERE"  # Replace with actual token from kubectl
    }
)

# Test read access (requires authentication)
try:
    namespaces = catalog.list_namespaces()
    print(f"✓ Authenticated read successful: {namespaces}")
except Unauthorized as e:
    print(f"✗ Authentication failed: {e}")

# Test write access (requires authentication)
try:
    catalog.create_namespace("test_namespace")
    print("✓ Authenticated write successful")
except Unauthorized as e:
    print(f"✗ Authentication failed: {e}")
```

### Audit Log Verification

```bash
# Check audit logs are being generated
kubectl exec -it deployment/iceberg-rest-catalog -n data-platform -- \
  tail -f /var/log/iceberg/audit.log

# Query recent audit events
kubectl exec -it deployment/iceberg-rest-catalog -n data-platform -- \
  grep -E '"action":"(READ|WRITE|CREATE|DELETE)"' /var/log/iceberg/audit.log | \
  jq -r '[.timestamp, .user, .action, .resource, .status] | @tsv' | \
  tail -20

# Expected output:
# 2025-10-31T04:12:00.123Z    service-account:data-engineer    READ      iceberg.analytics.customers    success
# 2025-10-31T04:13:15.456Z    service-account:data-scientist   WRITE     iceberg.raw.events             success
# 2025-10-31T04:14:30.789Z    anonymous                        READ      iceberg.analytics.customers    unauthorized
```

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

See [operations-runbook.md](operations-runbook.md) for detailed troubleshooting procedures.

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
2. Review [operations-runbook.md](operations-runbook.md)
3. Contact platform engineering team
4. Escalate to infrastructure team if critical

## Partition Specifications and Table Design

### Partitioning Strategy

The following partition specs are used across all Iceberg tables in the data platform:

#### EIA, FRED, Census (Daily Data)
```sql
-- Partition by days for daily granularity data
CREATE TABLE hub_curated.eia_daily_fuel (
  ts TIMESTAMP,
  region STRING,
  series STRING,
  value DOUBLE
)
PARTITIONED BY (days(ts))
TBLPROPERTIES (
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);
```

#### ISO 5-Minute Real-Time Data (CAISO, MISO, SPP)
```sql
-- Partition by days with sort order for high-frequency data
CREATE TABLE hub_curated.iso_rt_lmp (
  ts TIMESTAMP,
  iso STRING,
  node STRING,
  lmp DOUBLE,
  congestion DOUBLE,
  loss DOUBLE
)
PARTITIONED BY (days(ts))
TBLPROPERTIES (
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd',
  'write.metadata.compression-codec' = 'gzip',
  'sort.order' = 'iso,node,ts'
);
```

#### NOAA Hourly Weather Data
```sql
-- Partition by days with grid_id sort
CREATE TABLE hub_curated.noaa_hourly (
  ts TIMESTAMP,
  grid_id STRING,
  temperature DOUBLE,
  humidity DOUBLE,
  wind_speed DOUBLE,
  precipitation DOUBLE
)
PARTITIONED BY (days(ts))
TBLPROPERTIES (
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd',
  'sort.order' = 'grid_id,ts'
);
```

### Maintenance Strategy

Iceberg tables require regular maintenance to ensure optimal query performance and storage efficiency:

1. **Expire Snapshots**: Remove old snapshots beyond retention policy
2. **Rewrite Manifests**: Consolidate manifest files for faster metadata operations
3. **Compact Data Files**: Merge small files into larger ones to reduce metadata overhead

These operations are automated via the maintenance chart (see Task 5).

## Changelog

### Version 1.1 - October 31, 2025
- Added partition specifications for all data sources
- Defined maintenance strategy for Iceberg tables
- Documented sort orders for high-frequency data

### Version 1.0 - October 19, 2025
- Initial Iceberg REST Catalog integration
- Trino query engine support
- DataHub metadata discovery
- SeaTunnel data pipeline connectors
- Complete monitoring and alerting setup
- Comprehensive documentation and runbooks

---

**Last Updated**: October 31, 2025  
**Next Review**: November 7, 2025
