# Data Platform Security and RBAC

## Overview

This document describes the security architecture and role-based access control (RBAC) for the HMCo data platform.

## Security Layers

### 1. MinIO (Object Storage)

#### Bucket Policies

**ETL Writer Policy**
- **Permissions**: PutObject, GetObject on `hmco-raw/*` and `hmco-staged/*`
- **Users**: `etl_user`, SeaTunnel service accounts
- **Purpose**: Allow data ingestion to raw and staged buckets only

**BI Reader Policy**
- **Permissions**: GetObject on `hmco-curated/*`
- **Users**: `bi_reader`, Trino, Superset
- **Purpose**: Read-only access to curated data for analytics

#### Policy Implementation

```bash
# Create ETL writer user
mc admin user add minio etl_user <password>
mc admin policy attach minio etl-writer --user etl_user

# Create BI reader user
mc admin user add minio bi_reader <password>
mc admin policy attach minio bi-reader --user bi_reader
```

### 2. ClickHouse (OLAP Database)

#### Roles

**etl_writer**
- **Grants**: INSERT, SELECT on all databases
- **Quota**: 10,000 queries/hour, 1B rows read
- **Purpose**: Data loading from ingestion pipelines

**bi_reader**
- **Grants**: SELECT on all databases
- **Quota**: 1,000 queries/hour, 100M rows read
- **Purpose**: Read-only access for BI tools and dashboards

**trino**
- **Grants**: SELECT on all databases
- **Quota**: Same as bi_reader
- **Purpose**: Federated queries from Trino

#### Row-Level Security

Per-desk row policies on `rt_lmp` table:

```sql
-- CAISO desk: see only CAISO data
CREATE ROW POLICY desk_caiso ON default.rt_lmp
FOR SELECT USING iso = 'CAISO'
TO desk_caiso_user;

-- MISO desk: see only MISO data
CREATE ROW POLICY desk_miso ON default.rt_lmp
FOR SELECT USING iso = 'MISO'
TO desk_miso_user;

-- SPP desk: see only SPP data
CREATE ROW POLICY desk_spp ON default.rt_lmp
FOR SELECT USING iso = 'SPP'
TO desk_spp_user;
```

#### Quotas

```xml
<bi_quota>
  <interval>
    <duration>3600</duration>
    <queries>1000</queries>
    <result_rows>10000000</result_rows>
    <execution_time>7200</execution_time>
  </interval>
</bi_quota>
```

### 3. Trino (Query Engine)

#### Catalog-Level Access Control

- **iceberg catalog**: Read access to all authenticated users
- **clickhouse catalog**: Read access to all authenticated users
- **system catalog**: Admin only

#### Resource Groups

**bi-adhoc**
- **Concurrency**: 20 queries
- **Memory**: 30% of cluster
- **Users**: Interactive BI users

**bi-dashboard**
- **Concurrency**: 30 queries
- **Memory**: 40% of cluster
- **Priority**: 10x (higher than adhoc)
- **Users**: Dashboard services (Superset, Metabase)

**etl**
- **Concurrency**: 50 queries
- **Memory**: 50% of cluster
- **Priority**: 100x (highest)
- **Users**: ETL jobs and pipelines

#### Query Limits

```json
{
  "maxQueryExecutionTime": "30m",
  "maxQueryMemory": "50GB",
  "maxQueryMemoryPerNode": "8GB"
}
```

### 4. Iceberg REST Catalog

#### Authentication

- **Type**: OAuth 2.0 / OIDC
- **Provider**: Keycloak or Auth0
- **Tokens**: JWT with 1-hour expiry

#### Authorization

- **Namespace-level**: Read/write permissions per namespace
- **Table-level**: Fine-grained access control per table

Example configuration:

```yaml
catalog:
  authorization:
    type: ranger
    policies:
      - namespace: hub_curated
        table: eia_daily_fuel
        permissions:
          - principal: bi_reader
            actions: [SELECT]
          - principal: etl_writer
            actions: [SELECT, INSERT, UPDATE, DELETE]
```

## User Management

### Creating Users

#### MinIO

```bash
# Create user
mc admin user add minio <username> <password>

# Attach policy
mc admin policy attach minio <policy-name> --user <username>

# List users
mc admin user list minio
```

#### ClickHouse

```sql
-- Create user
CREATE USER etl_user IDENTIFIED BY 'password';

-- Grant role
GRANT etl_writer TO etl_user;

-- Set quota
ALTER USER etl_user SETTINGS max_rows_to_read = 1000000000;
```

#### Trino

Users authenticated via OAuth2/OIDC. Resource group assignment based on user attributes.

## Audit Logging

### MinIO Audit Events

```bash
# Enable audit logging
mc admin config set minio audit_webhook:primary endpoint="http://audit-service:8080/events"

# View audit logs
mc admin trace minio --verbose
```

### ClickHouse Query Log

```sql
-- Query log table
SELECT *
FROM system.query_log
WHERE user = 'bi_reader'
  AND type = 'QueryFinish'
  AND query_start_time > now() - INTERVAL 1 DAY
ORDER BY query_start_time DESC;
```

### Trino Query History

```sql
-- Query history
SELECT *
FROM system.runtime.queries
WHERE user = 'bi_reader@254carbon.com'
  AND created > CURRENT_TIMESTAMP - INTERVAL '1' DAY;
```

## Network Security

### Network Policies

```yaml
# Deny all ingress by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: data-platform-default-deny
  namespace: data-platform
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  
---
# Allow Trino to access ClickHouse
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-trino-to-clickhouse
  namespace: data-platform
spec:
  podSelector:
    matchLabels:
      app: clickhouse
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: trino
    ports:
    - protocol: TCP
      port: 8123
    - protocol: TCP
      port: 9000
```

## Secrets Management

All secrets managed via External Secrets Operator with HashiCorp Vault backend:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: clickhouse-passwords
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: clickhouse-passwords
  data:
  - secretKey: etl_password
    remoteRef:
      key: clickhouse/users/etl
      property: password
  - secretKey: bi_password
    remoteRef:
      key: clickhouse/users/bi
      property: password
```

## Security Checklist

- [ ] All service-to-service communication uses TLS
- [ ] Secrets stored in Vault, not in Git
- [ ] Network policies restrict pod-to-pod traffic
- [ ] RBAC configured for all data stores
- [ ] Row-level security enabled for sensitive tables
- [ ] Resource quotas prevent runaway queries
- [ ] Audit logging enabled for all components
- [ ] Regular security scans (Trivy, Snyk)
- [ ] Quarterly access reviews
- [ ] Incident response plan documented

## Incident Response

### Suspected Privilege Escalation

1. **Immediately revoke access**
   ```bash
   mc admin user disable minio <username>
   clickhouse-client --query "ALTER USER <username> DISABLE"
   ```

2. **Review audit logs**
   ```bash
   mc admin trace minio --last 1h > /tmp/audit.log
   clickhouse-client --query "SELECT * FROM system.query_log WHERE user = '<username>'"
   ```

3. **Notify security team**: security@254carbon.com

4. **Post-incident review**: Document findings and update access controls

## Contact

- **Security Team**: security@254carbon.com
- **Data Engineering**: data-eng@254carbon.com
- **On-call**: +1-555-0123

---

**Last Updated**: October 31, 2025  
**Next Review**: November 30, 2025
