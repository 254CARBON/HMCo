# Iceberg Security Hardening Guide

## Overview

This document provides security best practices and hardening procedures for the Iceberg REST Catalog integration.

## 1. Credentials and Secrets Management

### 1.1 MinIO Credentials

**Current State (Development):**
- Username: `minioadmin`
- Password: `minioadmin123`

**Production Changes Required:**

```bash
# 1. Update MinIO credentials via UI or CLI
# Access MinIO console: http://minio-service:9001
# Change admin user password immediately

# 2. Create service accounts for each application
# SeaTunnel account
mc admin user add minio seatunnel <strong-password>
mc admin policy attach minio readwrite --user seatunnel --bucket iceberg-warehouse

# DataHub account
mc admin user add minio datahub <strong-password>
mc admin policy attach minio readwrite --user datahub

# Iceberg account
mc admin user add minio iceberg <strong-password>
mc admin policy attach minio readwrite --user iceberg

# 3. Update Kubernetes secrets
kubectl create secret generic minio-secret-prod \
  -n data-platform \
  --from-literal=access-key=seatunnel \
  --from-literal=secret-key=<strong-password> \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Update deployments to use new secret
# Edit: k8s/data-lake/iceberg-rest.yaml
# Change secretKeyRef.name from minio-secret to minio-secret-prod
```

### 1.2 PostgreSQL Credentials

**Current State:**
```
iceberg_user: "iceberg_password"
```

**Production Changes:**

```bash
# 1. Generate strong password
NEW_PASSWORD=$(openssl rand -base64 32)
echo "Generated password: $NEW_PASSWORD"

# 2. Update PostgreSQL
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -c "ALTER USER iceberg_user WITH PASSWORD '$NEW_PASSWORD';"

# 3. Update Kubernetes secret
kubectl create secret generic iceberg-db-secret \
  -n data-platform \
  --from-literal=password=$NEW_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Update Iceberg REST Catalog deployment
# Edit: k8s/data-lake/iceberg-rest.yaml
# Change CATALOG_JDBC_PASSWORD to use secretKeyRef
```

### 1.3 DataHub Secret

**Update:**

```bash
# Generate secure DataHub secret
DATAHUB_SECRET=$(openssl rand -base64 32)
echo "Generated secret: $DATAHUB_SECRET"

# Update secret
kubectl create secret generic datahub-secret \
  -n data-platform \
  --from-literal=DATAHUB_SECRET=$DATAHUB_SECRET \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 2. Network Security

### 2.1 Network Policies

Create network policies to restrict traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: iceberg-network-policy
  namespace: data-platform
spec:
  podSelector:
    matchLabels:
      app: iceberg-rest-catalog
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: trino-coordinator
    - podSelector:
        matchLabels:
          app: datahub-gms
    - podSelector:
        matchLabels:
          app: seatunnel
    ports:
    - protocol: TCP
      port: 8181
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres-shared
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: minio
    ports:
    - protocol: TCP
      port: 9000
```

### 2.2 TLS/HTTPS Configuration

**Enable TLS for Iceberg REST Catalog:**

```yaml
# File: k8s/data-lake/iceberg-rest-tls.yaml
apiVersion: v1
kind: Secret
metadata:
  name: iceberg-rest-tls
  namespace: data-platform
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
---
# Update Iceberg deployment to use TLS
# Add to deployment spec:
env:
- name: CATALOG_REST_TLS_ENABLED
  value: "true"
- name: CATALOG_REST_TLS_CERT_PATH
  value: "/etc/tls/certs/tls.crt"
- name: CATALOG_REST_TLS_KEY_PATH
  value: "/etc/tls/certs/tls.key"
volumeMounts:
- name: tls-certs
  mountPath: /etc/tls/certs
  readOnly: true
volumes:
- name: tls-certs
  secret:
    secretName: iceberg-rest-tls
```

### 2.3 Service Mesh Integration (Optional)

Consider using Istio for advanced network policies:

```bash
# Install Istio (optional for production)
istioctl install --set profile=production -y

# Enable sidecar injection for data-platform namespace
kubectl label namespace data-platform istio-injection=enabled

# Create VirtualService for Iceberg
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: iceberg-rest-catalog
  namespace: data-platform
spec:
  hosts:
  - iceberg-rest-catalog
  http:
  - match:
    - uri:
        prefix: "/v1"
    route:
    - destination:
        host: iceberg-rest-catalog
        port:
          number: 8181
EOF
```

## 3. Access Control and RBAC

### 3.1 Kubernetes RBAC

Create fine-grained RBAC for services:

```yaml
---
# ServiceAccount for Iceberg REST Catalog
apiVersion: v1
kind: ServiceAccount
metadata:
  name: iceberg-rest-sa
  namespace: data-platform
---
# Role for Iceberg REST
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: iceberg-rest-role
  namespace: data-platform
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["minio-secret", "iceberg-db-secret"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["iceberg-rest-catalog-docs"]
  verbs: ["get"]
---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: iceberg-rest-binding
  namespace: data-platform
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: iceberg-rest-role
subjects:
- kind: ServiceAccount
  name: iceberg-rest-sa
  namespace: data-platform
```

### 3.2 Iceberg REST Catalog Access Control

Configure table-level and namespace-level access controls:

```yaml
# ConfigMap for Iceberg ACL rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: iceberg-rest-acl-config
  namespace: data-platform
data:
  acl-rules.yaml: |
    # Default policy: deny all unless explicitly allowed
    default_policy: deny
    
    # Service account permissions
    service_accounts:
      - name: "trino-coordinator"
        permissions:
          - namespace: "*"
            actions: ["read", "write", "create", "delete"]
      
      - name: "seatunnel-job"
        permissions:
          - namespace: "raw"
            actions: ["read", "write"]
          - namespace: "analytics"
            actions: ["read"]
      
      - name: "datahub-gms"
        permissions:
          - namespace: "*"
            actions: ["read", "describe"]
    
    # User role-based permissions
    roles:
      - name: "data-engineer"
        permissions:
          - namespace: "raw"
            table: "*"
            actions: ["read", "write", "create", "delete", "alter"]
          - namespace: "analytics"
            table: "*"
            actions: ["read", "write", "create", "delete", "alter"]
          - namespace: "governed"
            table: "*"
            actions: ["read", "write"]
      
      - name: "data-scientist"
        permissions:
          - namespace: "raw"
            table: "*"
            actions: ["read"]
          - namespace: "analytics"
            table: "*"
            actions: ["read", "write"]
          - namespace: "governed"
            table: "*"
            actions: ["read"]
      
      - name: "analyst"
        permissions:
          - namespace: "analytics"
            table: "*"
            actions: ["read"]
          - namespace: "governed"
            table: "*_safe"  # Only safe views
            actions: ["read"]
      
      - name: "compliance-officer"
        permissions:
          - namespace: "*"
            table: "*"
            actions: ["read", "describe"]
    
    # Table-specific overrides
    table_overrides:
      - namespace: "governed"
        table: "customer_transactions"
        permissions:
          - role: "data-engineer"
            actions: ["read", "write"]
          - role: "compliance-officer"
            actions: ["read"]
          - role: "data-scientist"
            actions: []  # Explicitly deny
      
      - namespace: "analytics"
        table: "aggregated_metrics"
        permissions:
          - role: "*"  # All roles
            actions: ["read"]
```

### 3.3 DataHub Access Control

Configure DataHub RBAC for Iceberg metadata with detailed policies:

```yaml
# DataHub Policy Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: datahub-policies
  namespace: data-platform
data:
  policies.yaml: |
    # Default authorization policy
    default_policy: "DENY"
    
    # Platform-level policies
    platform_policies:
      - name: "Platform Admins Full Access"
        type: "PLATFORM"
        actors:
          users: ["admin@254carbon.com"]
          groups: ["platform-admins"]
        privileges: ["MANAGE_POLICIES", "MANAGE_USERS", "MANAGE_DOMAINS"]
      
      - name: "All Users View Platform"
        type: "PLATFORM"
        actors:
          allUsers: true
        privileges: ["VIEW_ANALYTICS"]
    
    # Metadata policies for Iceberg
    metadata_policies:
      - name: "Data Engineers - Iceberg Full Access"
        type: "METADATA"
        resources:
          type: "dataset"
          filter:
            criteria:
              - field: "platform"
                values: ["iceberg"]
        actors:
          groups: ["data-engineers"]
        privileges: [
          "VIEW_DATASET_USAGE",
          "VIEW_DATASET_PROFILE",
          "EDIT_DATASET_METADATA",
          "EDIT_DATASET_DOCUMENTATION",
          "EDIT_DATASET_TAGS",
          "EDIT_DATASET_OWNERS",
          "EDIT_DATASET_DEPRECATION",
          "EDIT_LINEAGE"
        ]
      
      - name: "Data Scientists - Iceberg Read Access"
        type: "METADATA"
        resources:
          type: "dataset"
          filter:
            criteria:
              - field: "platform"
                values: ["iceberg"]
              - field: "tags"
                values: ["analytics", "ml-ready"]
        actors:
          groups: ["data-scientists"]
        privileges: [
          "VIEW_DATASET_USAGE",
          "VIEW_DATASET_PROFILE",
          "VIEW_DATASET_DOCUMENTATION"
        ]
      
      - name: "Analysts - Iceberg Safe Tables Only"
        type: "METADATA"
        resources:
          type: "dataset"
          filter:
            criteria:
              - field: "platform"
                values: ["iceberg"]
              - field: "tags"
                values: ["analyst-safe"]
        actors:
          groups: ["analysts"]
        privileges: [
          "VIEW_DATASET_USAGE",
          "VIEW_DATASET_PROFILE"
        ]
      
      - name: "Compliance - Iceberg Audit Access"
        type: "METADATA"
        resources:
          type: "dataset"
          filter:
            criteria:
              - field: "platform"
                values: ["iceberg"]
        actors:
          groups: ["compliance-officers"]
        privileges: [
          "VIEW_DATASET_USAGE",
          "VIEW_DATASET_PROFILE",
          "VIEW_DATASET_DOCUMENTATION",
          "VIEW_DATASET_OWNERS",
          "VIEW_LINEAGE"
        ]
    
    # Domain-based policies
    domain_policies:
      - name: "Customer Data Domain - Restricted"
        type: "METADATA"
        resources:
          type: "dataset"
          filter:
            criteria:
              - field: "domain"
                values: ["urn:li:domain:customer-data"]
        actors:
          groups: ["customer-data-stewards", "compliance-officers"]
        privileges: [
          "VIEW_DATASET_USAGE",
          "EDIT_DATASET_METADATA",
          "EDIT_DATASET_TAGS"
        ]
```

### 3.4 Trino Access Control with Iceberg

Configure Trino role-based access for querying Iceberg tables:

```properties
# File: /etc/trino/access-control/iceberg-rules.json
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
              "privileges": ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"],
              "group": "data-engineers"
            },
            {
              "privileges": ["SELECT"],
              "group": "data-scientists"
            }
          ]
        },
        {
          "schema": "analytics",
          "owner": false,
          "tables": [
            {
              "privileges": ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"],
              "group": "data-engineers"
            },
            {
              "privileges": ["SELECT", "INSERT", "UPDATE"],
              "group": "data-scientists"
            },
            {
              "privileges": ["SELECT"],
              "group": "analysts"
            }
          ]
        },
        {
          "schema": "governed",
          "owner": false,
          "tables": [
            {
              "table": "customer_transactions",
              "privileges": ["SELECT", "INSERT", "UPDATE"],
              "group": "data-engineers"
            },
            {
              "table": "customer_transactions",
              "privileges": ["SELECT"],
              "group": "compliance-officers"
            },
            {
              "table": "customer_transactions_safe",
              "privileges": ["SELECT"],
              "group": "data-scientists"
            },
            {
              "table": "customer_transactions_safe",
              "privileges": ["SELECT"],
              "group": "analysts"
            }
          ]
        }
      ]
    }
  ],
  "sessionProperties": [
    {
      "property": ".*",
      "allow": true
    }
  ]
}
```

### 3.5 Column-Level Security

Implement column-level access control for sensitive data:

```sql
-- Create secure view with column masking
CREATE VIEW iceberg.governed.customer_transactions_masked AS
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    amount,
    currency,
    merchant_name,
    category,
    -- Mask PII based on user role
    CASE 
        WHEN current_user IN (SELECT user FROM system.roles WHERE role = 'data-engineer')
        THEN customer_email
        ELSE CONCAT(SUBSTR(customer_email, 1, 2), '***@***.com')
    END AS customer_email,
    CASE 
        WHEN current_user IN (SELECT user FROM system.roles WHERE role = 'data-engineer')
        THEN customer_phone
        ELSE CONCAT(SUBSTR(customer_phone, 1, 3), '-***-****')
    END AS customer_phone,
    created_at,
    created_by
FROM iceberg.governed.customer_transactions;

-- Grant access to masked view
GRANT SELECT ON iceberg.governed.customer_transactions_masked TO ROLE data_scientist;
GRANT SELECT ON iceberg.governed.customer_transactions_masked TO ROLE analyst;
```

## 4. Data Encryption

### 4.1 Encryption at Rest

**MinIO Server-Side Encryption:**

```bash
# Enable MinIO KMS (optional)
# Set environment variables:
MINIO_SSE_MASTER_KEY=$(openssl rand -base64 32)

# Or use AWS KMS/Vault for production
```

**PostgreSQL Encryption:**

```bash
# Enable pgcrypto extension
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -d iceberg_rest -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

# Use encrypted columns for sensitive data
CREATE TABLE encrypted_metadata (
  id UUID PRIMARY KEY,
  data BYTEA,
  encrypted_data TEXT
);
```

### 4.2 Encryption in Transit

- **Enable TLS** for all service-to-service communication
- **Use HTTPS** for REST API calls
- **Enable SSL** for PostgreSQL connections

```yaml
# Update Iceberg REST to use SSL for PostgreSQL and scope to dedicated schema
CATALOG_URI: "jdbc:postgresql://postgres-shared-service:5432/iceberg_rest?currentSchema=iceberg_catalog&ssl=true&sslmode=require"
```

## 5. Audit Logging

### 5.1 Enable Audit Logs

```yaml
# Kubernetes audit logging (if not already enabled)
# Edit kubelet config to enable audit logging

# Application logging
# Ensure all components log to centralized system:
# - Elasticsearch
# - Loki
# - CloudWatch

# Iceberg REST Catalog logging
env:
- name: LOGGING_LEVEL
  value: "INFO"
- name: AUDIT_LOG_ENABLED
  value: "true"
```

### 5.2 Comprehensive Audit Logging Configuration

```yaml
# ConfigMap for structured audit logging
apiVersion: v1
kind: ConfigMap
metadata:
  name: iceberg-rest-audit-config
  namespace: data-platform
data:
  log4j2.xml: |
    <?xml version="1.0" encoding="UTF-8"?>
    <Configuration status="INFO">
      <Appenders>
        <!-- Audit log appender with JSON format -->
        <RollingFile name="AuditLog" 
                     fileName="/var/log/iceberg/audit.log"
                     filePattern="/var/log/iceberg/audit-%d{yyyy-MM-dd}-%i.log.gz">
          <JsonLayout compact="true" eventEol="true">
            <KeyValuePair key="timestamp" value="$${date:yyyy-MM-dd'T'HH:mm:ss.SSSZ}"/>
            <KeyValuePair key="level" value="$${level}"/>
            <KeyValuePair key="service" value="iceberg-rest-catalog"/>
            <KeyValuePair key="event_type" value="audit"/>
            <KeyValuePair key="message" value="$${message}"/>
          </JsonLayout>
          <Policies>
            <TimeBasedTriggeringPolicy interval="1" modulate="true"/>
            <SizeBasedTriggeringPolicy size="100 MB"/>
          </Policies>
          <DefaultRolloverStrategy max="90"/>  <!-- 90-day retention -->
        </RollingFile>
        
        <!-- Security events appender -->
        <RollingFile name="SecurityLog" 
                     fileName="/var/log/iceberg/security.log"
                     filePattern="/var/log/iceberg/security-%d{yyyy-MM-dd}-%i.log.gz">
          <JsonLayout compact="true" eventEol="true"/>
          <Policies>
            <TimeBasedTriggeringPolicy interval="1" modulate="true"/>
            <SizeBasedTriggeringPolicy size="50 MB"/>
          </Policies>
          <DefaultRolloverStrategy max="365"/>  <!-- 1-year retention for security -->
        </RollingFile>
        
        <!-- Access log appender -->
        <RollingFile name="AccessLog" 
                     fileName="/var/log/iceberg/access.log"
                     filePattern="/var/log/iceberg/access-%d{yyyy-MM-dd-HH}-%i.log.gz">
          <PatternLayout pattern="%d{ISO8601} %X{remote_addr} %X{user} %X{method} %X{path} %X{status} %X{duration_ms}ms%n"/>
          <Policies>
            <TimeBasedTriggeringPolicy interval="1" modulate="true"/>
            <SizeBasedTriggeringPolicy size="200 MB"/>
          </Policies>
          <DefaultRolloverStrategy max="30"/>
        </RollingFile>
      </Appenders>
      
      <Loggers>
        <!-- Audit logger for catalog operations -->
        <Logger name="org.apache.iceberg.rest.audit" level="INFO" additivity="false">
          <AppenderRef ref="AuditLog"/>
        </Logger>
        
        <!-- Security events logger -->
        <Logger name="org.apache.iceberg.rest.security" level="INFO" additivity="false">
          <AppenderRef ref="SecurityLog"/>
        </Logger>
        
        <!-- Access logger for all HTTP requests -->
        <Logger name="org.apache.iceberg.rest.access" level="INFO" additivity="false">
          <AppenderRef ref="AccessLog"/>
        </Logger>
        
        <!-- Root logger -->
        <Root level="INFO">
          <AppenderRef ref="Console"/>
        </Root>
      </Loggers>
    </Configuration>
```

### 5.3 Audit Log Shipping to Elasticsearch

```yaml
# Fluent Bit DaemonSet for log shipping
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit-iceberg
  namespace: data-platform
spec:
  selector:
    matchLabels:
      app: fluent-bit-iceberg
  template:
    metadata:
      labels:
        app: fluent-bit-iceberg
    spec:
      serviceAccountName: fluent-bit
      containers:
      - name: fluent-bit
        image: fluent/fluent-bit:2.1.10  # Pin to specific version for production
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: config
          mountPath: /fluent-bit/etc/
        resources:
          limits:
            memory: 200Mi
          requests:
            cpu: 100m
            memory: 100Mi
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: config
        configMap:
          name: fluent-bit-iceberg-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-iceberg-config
  namespace: data-platform
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Daemon        off
        Log_Level     info
        Parsers_File  parsers.conf
    
    [INPUT]
        Name              tail
        Path              /var/log/iceberg/audit.log
        Parser            json
        Tag               iceberg.audit
        DB                /var/log/flb_audit.db
        Mem_Buf_Limit     5MB
        Skip_Long_Lines   On
        Refresh_Interval  10
    
    [INPUT]
        Name              tail
        Path              /var/log/iceberg/security.log
        Parser            json
        Tag               iceberg.security
        DB                /var/log/flb_security.db
        Mem_Buf_Limit     5MB
    
    [INPUT]
        Name              tail
        Path              /var/log/iceberg/access.log
        Parser            apache
        Tag               iceberg.access
        DB                /var/log/flb_access.db
        Mem_Buf_Limit     5MB
    
    [FILTER]
        Name              kubernetes
        Match             iceberg.*
        Kube_URL          https://kubernetes.default.svc:443
        Kube_CA_File      /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File   /var/run/secrets/kubernetes.io/serviceaccount/token
    
    [FILTER]
        Name              modify
        Match             iceberg.*
        Add               cluster_name 254carbon-prod
        Add               environment production
    
    [OUTPUT]
        Name              es
        Match             iceberg.audit
        Host              elasticsearch.monitoring.svc.cluster.local
        Port              9200
        Index             iceberg-audit
        Type              _doc
        Logstash_Format   On
        Logstash_Prefix   iceberg-audit
        Retry_Limit       5
    
    [OUTPUT]
        Name              es
        Match             iceberg.security
        Host              elasticsearch.monitoring.svc.cluster.local
        Port              9200
        Index             iceberg-security
        Type              _doc
        Logstash_Format   On
        Logstash_Prefix   iceberg-security
        Retry_Limit       5
    
    [OUTPUT]
        Name              es
        Match             iceberg.access
        Host              elasticsearch.monitoring.svc.cluster.local
        Port              9200
        Index             iceberg-access
        Type              _doc
        Logstash_Format   On
        Logstash_Prefix   iceberg-access
        Retry_Limit       5
  
  parsers.conf: |
    [PARSER]
        Name        json
        Format      json
        Time_Key    timestamp
        Time_Format %Y-%m-%dT%H:%M:%S.%L%z
    
    [PARSER]
        Name        apache
        Format      regex
        Regex       ^(?<remote_addr>[^ ]*) (?<user>[^ ]*) (?<method>[^ ]*) (?<path>[^ ]*) (?<status>[^ ]*) (?<duration_ms>[^ ]*)ms$
        Time_Key    timestamp
        Time_Format %d/%b/%Y:%H:%M:%S %z
```

### 5.4 Monitor for Security Events

**Security Event Types to Monitor:**

1. **Authentication Failures**
   ```json
   {
     "event_type": "authentication_failure",
     "timestamp": "2025-10-31T04:15:00.123Z",
     "user": "anonymous",
     "client_ip": "10.42.1.99",
     "endpoint": "/v1/namespaces",
     "reason": "invalid_token"
   }
   ```

2. **Authorization Failures**
   ```json
   {
     "event_type": "authorization_failure",
     "timestamp": "2025-10-31T04:16:00.456Z",
     "user": "service-account:analyst",
     "action": "DELETE",
     "resource": "iceberg.raw.events",
     "reason": "insufficient_permissions"
   }
   ```

3. **Sensitive Data Access**
   ```json
   {
     "event_type": "sensitive_data_access",
     "timestamp": "2025-10-31T04:17:00.789Z",
     "user": "service-account:data-engineer",
     "action": "SELECT",
     "resource": "iceberg.governed.customer_transactions",
     "classification": "confidential",
     "pii_fields": ["customer_email", "customer_phone"]
   }
   ```

4. **Schema Modifications**
   ```json
   {
     "event_type": "schema_modification",
     "timestamp": "2025-10-31T04:18:00.012Z",
     "user": "service-account:data-engineer",
     "action": "ALTER",
     "resource": "iceberg.analytics.metrics",
     "changes": ["added_column: new_metric"]
   }
   ```

**Elasticsearch Queries for Security Monitoring:**

```bash
# Query failed authentication attempts (last 24 hours)
curl -X GET "http://elasticsearch.monitoring:9200/iceberg-security-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": [
          {"term": {"event_type.keyword": "authentication_failure"}},
          {"range": {"timestamp": {"gte": "now-24h"}}}
        ]
      }
    },
    "aggs": {
      "by_client_ip": {
        "terms": {"field": "client_ip.keyword", "size": 10}
      }
    }
  }'

# Query unauthorized access attempts
curl -X GET "http://elasticsearch.monitoring:9200/iceberg-security-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": [
          {"term": {"event_type.keyword": "authorization_failure"}},
          {"range": {"timestamp": {"gte": "now-7d"}}}
        ]
      }
    },
    "aggs": {
      "by_user": {
        "terms": {"field": "user.keyword", "size": 20}
      },
      "by_resource": {
        "terms": {"field": "resource.keyword", "size": 20}
      }
    }
  }'

# Query sensitive data access patterns
curl -X GET "http://elasticsearch.monitoring:9200/iceberg-audit-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": [
          {"term": {"classification.keyword": "confidential"}},
          {"exists": {"field": "pii_fields"}},
          {"range": {"timestamp": {"gte": "now-30d"}}}
        ]
      }
    },
    "aggs": {
      "access_by_user": {
        "terms": {"field": "user.keyword", "size": 50}
      },
      "access_by_table": {
        "terms": {"field": "resource.keyword", "size": 50}
      }
    }
  }'

# Query unusual access patterns (high-frequency access)
curl -X GET "http://elasticsearch.monitoring:9200/iceberg-access-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "range": {"timestamp": {"gte": "now-1h"}}
    },
    "aggs": {
      "requests_per_user": {
        "terms": {
          "field": "user.keyword",
          "size": 20,
          "order": {"_count": "desc"}
        }
      }
    }
  }'
```

### 5.5 Alerting Rules for Security Events

```yaml
# Prometheus AlertManager rules for security events
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-iceberg-security-rules
  namespace: monitoring
data:
  iceberg-security.rules: |
    groups:
    - name: iceberg_security
      interval: 30s
      rules:
      # Alert on authentication failures
      - alert: IcebergHighAuthenticationFailureRate
        expr: |
          sum(rate(iceberg_authentication_failures_total[5m])) > 10
        for: 5m
        labels:
          severity: warning
          component: iceberg-rest-catalog
        annotations:
          summary: "High authentication failure rate detected"
          description: "More than 10 authentication failures per second in the last 5 minutes"
      
      # Alert on unauthorized access attempts
      - alert: IcebergUnauthorizedAccessAttempts
        expr: |
          sum(increase(iceberg_authorization_failures_total[15m])) > 50
        for: 5m
        labels:
          severity: critical
          component: iceberg-rest-catalog
        annotations:
          summary: "Multiple unauthorized access attempts detected"
          description: "More than 50 authorization failures in the last 15 minutes"
      
      # Alert on sensitive data access spikes
      - alert: IcebergSensitiveDataAccessSpike
        expr: |
          sum(rate(iceberg_sensitive_data_access_total[10m])) > 
          sum(avg_over_time(iceberg_sensitive_data_access_total[1h])) * 3
        for: 10m
        labels:
          severity: warning
          component: iceberg-rest-catalog
        annotations:
          summary: "Unusual spike in sensitive data access"
          description: "Sensitive data access rate is 3x higher than the 1-hour average"
      
      # Alert on schema modifications
      - alert: IcebergUnauthorizedSchemaModification
        expr: |
          sum(increase(iceberg_schema_modifications_total{authorized="false"}[5m])) > 0
        for: 1m
        labels:
          severity: critical
          component: iceberg-rest-catalog
        annotations:
          summary: "Unauthorized schema modification attempted"
          description: "One or more unauthorized schema modification attempts detected"
```

### 5.6 Audit Log Retention and Compliance

```yaml
# Elasticsearch Index Lifecycle Management (ILM) policy
apiVersion: v1
kind: ConfigMap
metadata:
  name: elasticsearch-ilm-iceberg-audit
  namespace: monitoring
data:
  policy.json: |
    {
      "policy": {
        "phases": {
          "hot": {
            "min_age": "0ms",
            "actions": {
              "rollover": {
                "max_age": "1d",
                "max_size": "50gb"
              },
              "set_priority": {
                "priority": 100
              }
            }
          },
          "warm": {
            "min_age": "7d",
            "actions": {
              "allocate": {
                "number_of_replicas": 1
              },
              "shrink": {
                "number_of_shards": 1
              },
              "forcemerge": {
                "max_num_segments": 1
              },
              "set_priority": {
                "priority": 50
              }
            }
          },
          "cold": {
            "min_age": "30d",
            "actions": {
              "allocate": {
                "number_of_replicas": 0
              },
              "set_priority": {
                "priority": 0
              }
            }
          },
          "delete": {
            "min_age": "2555d",  # 7 years (2555 days) for compliance (SOX, GDPR)
            "actions": {
              "delete": {}
            }
          }
        }
      }
    }
```

**Compliance Requirements:**
- **Audit Logs**: Retained for 7 years (2555 days) for compliance (SOX, GDPR)
- **Security Logs**: Retained for 1 year (365 days)
- **Access Logs**: Retained for 30 days
- **Encryption**: All logs encrypted at rest and in transit
- **Integrity**: Log tampering detection via checksums

## 6. Secret Management

### 6.1 Use Vault for Secret Storage

**Integrate with Vault (if deployed):**

```bash
# Store Iceberg credentials in Vault
vault kv put secret/iceberg/minio \
  access-key=minioadmin \
  secret-key=minioadmin123

vault kv put secret/iceberg/database \
  username=iceberg_user \
  password=iceberg_password

# Update deployments to fetch from Vault
# Use external-secrets operator for automatic sync
```

### 6.2 Rotate Credentials Regularly

```bash
#!/bin/bash
# Script: rotate-iceberg-credentials.sh

# Rotate MinIO credentials
MC_HOST=minio-service:9000
NEW_PASS=$(openssl rand -base64 32)

# Update MinIO
mc admin user change-password $MC_HOST minioadmin $NEW_PASS

# Update Kubernetes secret
kubectl create secret generic minio-secret \
  -n data-platform \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=$NEW_PASS \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new credentials
kubectl rollout restart deployment/iceberg-rest-catalog -n data-platform

echo "Credentials rotated successfully"
```

## 7. Compliance and Governance

### 7.1 Data Classification

```yaml
# Tag data in DataHub with classification
datasets:
  - urn: "urn:li:dataset:urn:li:platform:iceberg:raw.customers"
    tags:
      - "pii"
      - "confidential"
      - "retention:7years"
```

### 7.2 Data Retention Policies

```sql
-- Set Iceberg retention policies
-- Expire old snapshots
CALL iceberg.system.expire_snapshots('raw.customers', INTERVAL '90' DAY);

-- Remove orphan files
CALL iceberg.system.remove_orphan_files('raw.customers');
```

### 7.3 Data Privacy

- **PII Protection**: Mask sensitive columns
- **Data Anonymization**: Apply anonymization rules
- **Access Logging**: Log all data access
- **Compliance Scanning**: Regular data classification audits

## 8. Security Checklist

### Pre-Production

- [ ] Change all default credentials
- [ ] Enable TLS/HTTPS
- [ ] Configure network policies
- [ ] Enable audit logging
- [ ] Set up secret management
- [ ] Configure RBAC
- [ ] Enable data encryption at rest
- [ ] Enable encryption in transit
- [ ] Configure backup encryption
- [ ] Document security procedures

### Ongoing

- [ ] Rotate credentials monthly
- [ ] Review audit logs weekly
- [ ] Patch security vulnerabilities
- [ ] Monitor for unauthorized access
- [ ] Test disaster recovery procedures
- [ ] Conduct security audits quarterly
- [ ] Update security policies

## 9. Security Tools

### Recommended Tools

1. **Secret Management**: Vault, Sealed Secrets
2. **Network Security**: Istio, Calico
3. **RBAC**: Kubernetes native, DataHub policies
4. **Monitoring**: Prometheus, Loki, Elasticsearch
5. **Scanning**: Trivy, Falco, OPA/Gatekeeper

### Implementation

```bash
# Install security tools
helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system
helm install falco falcosecurity/falco -n falco --create-namespace
helm install kyverno kyverno/kyverno -n kyverno --create-namespace
```

## References

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Apache Iceberg Security](https://iceberg.apache.org/docs/nightly/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)

## Support

For security concerns or incidents:
1. Contact security team
2. Review audit logs
3. Implement remediation
4. Document incident
5. Update security policies
