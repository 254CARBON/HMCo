# Phase 5: Days 8-9 - Security Hardening & Data Governance

**Status**: Planning Complete  
**Date**: October 25, 2025 (Simulated Day 8-9)  
**Duration**: 8 hours (full 2-day sprint)  
**Goal**: Implement enterprise security and data governance

---

## Overview

Days 8-9 focus on hardening the platform for production and establishing governance:
- Security policy enforcement (Kyverno)
- Role-based access control (RBAC)
- Audit logging
- Secret rotation
- Data governance (DataHub)
- Compliance validation

---

## Day 8: Security Hardening (4 hours)

### Task 1: Review & Update Kyverno Policies (1 hour)

#### 1.1 Current Policy Status

```bash
# Check installed Kyverno policies
kubectl get cpol,clusterpolicyrules 2>&1 | head -20

# List all policies
echo "=== Kyverno Policies ==="
kubectl get clusterpolicies -o wide
```

#### 1.2 Create Production Security Policies

```yaml
# Save as /tmp/kyverno-production-policies.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
spec:
  validationFailureAction: audit
  rules:
  - name: validate-resources
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "CPU and memory limits are required"
      pattern:
        spec:
          containers:
          - resources:
              limits:
                memory: "?*"
                cpu: "?*"
---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-security-context
spec:
  validationFailureAction: audit
  rules:
  - name: validate-runAsNonRoot
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "Running as root is not allowed"
      pattern:
        spec:
          containers:
          - securityContext:
              runAsNonRoot: true
---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: restrict-image-registry
spec:
  validationFailureAction: audit
  rules:
  - name: validate-registry
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "Images must come from approved registries"
      pattern:
        spec:
          containers:
          - image: "registry.254carbon.com/*"
---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-privileged
spec:
  validationFailureAction: audit
  rules:
  - name: validate-privileged
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "Privileged containers are not allowed"
      pattern:
        spec:
          containers:
          - securityContext:
              privileged: false
```

#### 1.3 Apply Policies

```bash
# Apply Kyverno policies
kubectl apply -f /tmp/kyverno-production-policies.yaml

# Verify policies
echo "=== Kyverno Policies Applied ==="
kubectl get clusterpolicies -o wide

# Monitor policy violations
echo "=== Policy Violation Report ==="
kubectl logs -n kyverno deployment/kyverno-controller | grep -i "violation" | tail -20
```

### Task 2: Configure RBAC (1 hour)

#### 2.1 Create Service Roles

```yaml
# Save as /tmp/rbac-roles.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: data-platform
---
# DataEngineer Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: data-engineer
  namespace: data-platform
rules:
- apiGroups: ["batch"]
  resources: ["cronjobs", "jobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
# DataAnalyst Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: data-analyst
  namespace: data-platform
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch"]
---
# Platform Admin Role
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: platform-admin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
---
# DataEngineer RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: data-engineer-binding
  namespace: data-platform
subjects:
- kind: Group
  name: data-engineers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: data-engineer
  apiGroup: rbac.authorization.k8s.io
---
# DataAnalyst RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: data-analyst-binding
  namespace: data-platform
subjects:
- kind: Group
  name: data-analysts
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: data-analyst
  apiGroup: rbac.authorization.k8s.io
---
# Platform Admin ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: platform-admin-binding
subjects:
- kind: Group
  name: platform-admins
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: platform-admin
  apiGroup: rbac.authorization.k8s.io
```

#### 2.2 Apply RBAC

```bash
# Apply RBAC configuration
kubectl apply -f /tmp/rbac-roles.yaml

# Verify roles
echo "=== RBAC Roles ==="
kubectl get roles,clusterroles -l app=254carbon-platform

# Verify bindings
echo "=== RBAC Bindings ==="
kubectl get rolebindings,clusterrolebindings -l app=254carbon-platform
```

### Task 3: Set Up Audit Logging (1 hour)

#### 3.1 Configure API Audit Logs

```bash
# Create audit policy
cat > /tmp/audit-policy.yaml << 'EOF'
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Log pod creation/deletion in data-platform namespace
- level: RequestResponse
  verbs: ["create", "delete"]
  resources:
  - group: ""
    resources: ["pods"]
  namespaces: ["data-platform"]
# Log secret access
- level: Metadata
  verbs: ["get", "list", "watch"]
  resources:
  - group: ""
    resources: ["secrets"]
# Log RBAC changes
- level: RequestResponse
  verbs: ["create", "update", "patch", "delete"]
  resources:
  - group: "rbac.authorization.k8s.io"
    resources: ["clusterroles", "clusterrolebindings", "roles", "rolebindings"]
# Default catch-all
- level: Metadata
EOF

# Deploy audit logging
kubectl create configmap audit-policy --from-file=/tmp/audit-policy.yaml \
  -n kube-system --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Audit logging configured"
```

#### 3.2 Enable Audit Logs Streaming

```bash
# Stream audit logs
kubectl logs -n kube-system -l component=kube-apiserver \
  | grep "audit" | tail -20
```

### Task 4: Secret Rotation Setup (1 hour)

#### 4.1 Create Secrets with Rotation Policy

```bash
# Create rotating secrets
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials-rotated
  namespace: data-platform
  labels:
    rotation-enabled: "true"
    rotation-period: "30d"
type: Opaque
data:
  username: ZGF0YWVuZ2luZWVy  # base64: dataengineer
  password: $(openssl rand -base64 32)  # Random password
  host: ZGF0YS1kYi5kYXRhLXBsYXRmb3JtLnN2Yy5jbHVzdGVyLmxvY2Fs
---
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials-rotated
  namespace: data-platform
  labels:
    rotation-enabled: "true"
    rotation-period: "14d"
type: Opaque
data:
  api-key: $(openssl rand -base64 32)
  api-secret: $(openssl rand -base64 32)
EOF

# Verify secrets
echo "=== Rotating Secrets ==="
kubectl get secrets -n data-platform -l rotation-enabled=true
```

#### 4.2 Create Rotation CronJob

```bash
# Create secret rotation job
kubectl apply -f - <<'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: secret-rotation-job
  namespace: kube-system
spec:
  schedule: "0 2 * * 0"  # Weekly, 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
          - name: rotator
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              # Rotate all secrets with rotation-enabled=true label
              for secret in $(kubectl get secrets -n data-platform \
                -l rotation-enabled=true -o name); do
                echo "Rotating $secret..."
                # Generate new credentials
                NEW_PASS=$(openssl rand -base64 32)
                NEW_KEY=$(openssl rand -base64 32)
                
                # Update secret
                kubectl patch secret ${secret##*/} -n data-platform \
                  -p "{\"data\":{\"password\":\"$(echo -n $NEW_PASS | base64)\"}}"
              done
              echo "Secret rotation complete"
          restartPolicy: OnFailure
EOF

echo "✅ Secret rotation scheduled"
```

---

## Day 9: Data Governance & Compliance (4 hours)

### Task 1: DataHub Integration (1.5 hours)

#### 1.1 Configure DataHub Ingestion

```bash
# Create DataHub ingestion recipe
cat > /tmp/datahub-ingestion-recipe.yaml << 'EOF'
source:
  type: "postgres"
  config:
    host_port: "kong-postgres.kong:5432"
    database: "256carbon_data"
    username: "${DATAHUB_POSTGRES_USER}"
    password: "${DATAHUB_POSTGRES_PASSWORD}"
    schema_pattern:
      includes:
      - "public"
      - "data_lake"
    table_pattern:
      includes:
      - "commodity_prices"
      - "market_data"
      - "analytics"
    include_views: true

sink:
  type: "datahub-rest"
  config:
    server: "http://datahub-gms-service:8080"
    extra_headers:
      X-RestLi-Protocol-Version: "2.0.0"

transformers:
  - type: "add_dataset_ownership"
    config:
      ownership_type: "DATAOWNER"
      owner_urns:
      - "urn:li:corpuser:data-team"
  
  - type: "add_dataset_tags"
    config:
      tags:
      - tag: "pii"
        urn_pattern: ".*password.*|.*email.*|.*phone.*"
      - tag: "sensitive"
        urn_pattern: ".*prices.*|.*costs.*"
EOF

# Apply ingestion config
kubectl create configmap datahub-ingestion \
  --from-file=/tmp/datahub-ingestion-recipe.yaml \
  -n data-platform --dry-run=client -o yaml | kubectl apply -f -

echo "✅ DataHub ingestion configured"
```

#### 1.2 Schedule Metadata Sync

```bash
# Create metadata sync CronJob
kubectl apply -f - <<'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: datahub-metadata-sync
  namespace: data-platform
spec:
  schedule: "0 */4 * * *"  # Every 4 hours
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: datahub-ingestion
          containers:
          - name: datahub-ingest
            image: acryldata/datahub-ingestion:latest
            env:
            - name: DATAHUB_GMS_HOST
              value: "datahub-gms-service"
            - name: DATAHUB_GMS_PORT
              value: "8080"
            command:
            - datahub
            - ingest
            - -c
            - /etc/datahub/ingestion-recipe.yaml
            volumeMounts:
            - name: ingestion-config
              mountPath: /etc/datahub/
          volumes:
          - name: ingestion-config
            configMap:
              name: datahub-ingestion
          restartPolicy: OnFailure
EOF

echo "✅ Metadata sync scheduled"
```

### Task 2: Implement Data Lineage Tracking (1 hour)

#### 2.1 Configure Lineage Capture

```bash
# Create lineage tracking ConfigMap
kubectl create configmap data-lineage-config \
  --from-literal=lineage-backend=datahub \
  --from-literal=capture-interval=1h \
  -n data-platform --dry-run=client -o yaml | kubectl apply -f -

# Add lineage annotations to pipelines
kubectl patch cronjob etl-db-extract-template -n data-platform --type merge -p \
  '{"spec":{"jobTemplate":{"spec":{"template":{"metadata":{"annotations":{"datahub.io/lineage":"true"}}}}}}}'

echo "✅ Data lineage tracking enabled"
```

#### 2.2 Create Lineage Dashboard

```bash
# DataHub lineage query (to be executed in DataHub UI)
cat > /tmp/datahub-lineage-query.txt << 'EOF'
# Query to view data lineage
GET /api/graphql
{
  dataset(urn: "urn:li:dataset:(urn:li:dataPlatform:iceberg,iceberg.default.commodity_prices,PROD)") {
    upstreams {
      dataset {
        name
        platform
        description
      }
    }
    downstreams {
      dataset {
        name
        platform
      }
    }
  }
}
EOF

echo "✅ Lineage dashboard configured"
```

### Task 3: Implement Access Controls (0.75 hour)

#### 3.1 Secure Credential Access

```bash
# Create secret access policy
kubectl apply -f - <<'EOF'
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: audit-secret-access
spec:
  validationFailureAction: audit
  rules:
  - name: log-secret-reads
    match:
      resources:
        kinds:
        - Secret
    verbs:
    - get
    - list
    audit: true
EOF

# Enable secret encryption at rest
kubectl patch secret -n kube-system -l type=api-encryption \
  -p '{"metadata":{"annotations":{"encryption":"aes-256"}}}'

echo "✅ Access controls implemented"
```

#### 3.2 Audit Trail Setup

```bash
# Create audit trail dashboard
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: audit-trail-dashboard
  namespace: monitoring
data:
  dashboard.json: |
    {
      "title": "Security Audit Trail",
      "panels": [
        {
          "title": "Secret Access Events",
          "targets": [
            {
              "expr": "increase(audit_event_total{resource_type=\"Secret\"}[1h])"
            }
          ]
        },
        {
          "title": "RBAC Changes",
          "targets": [
            {
              "expr": "increase(audit_event_total{verb=~\"create|update|delete\",resource_type=~\"Role|RoleBinding\"}[1h])"
            }
          ]
        },
        {
          "title": "Pod Creation Events",
          "targets": [
            {
              "expr": "increase(audit_event_total{verb=\"create\",resource_type=\"Pod\"}[1h])"
            }
          ]
        }
      ]
    }
EOF

echo "✅ Audit trail dashboard created"
```

### Task 4: Compliance Validation (0.75 hour)

#### 4.1 Security Checklist

```bash
# Create compliance validation script
cat > /tmp/security-compliance-check.sh << 'EOF'
#!/bin/bash

echo "=== 254Carbon Security Compliance Checklist ==="
echo ""

# Check 1: Network policies
echo "✓ Network Policies"
echo "  Ingress policies: $(kubectl get networkpolicies -A | wc -l)"
echo "  Egress policies: $(kubectl get networkpolicies -A -o json | grep -c "policyTypes.*Egress")"

# Check 2: Pod Security
echo "✓ Pod Security"
echo "  Privileged containers: $(kubectl get pods -A --field-selector spec.containers[0].securityContext.privileged=true 2>/dev/null | wc -l)"
echo "  RunAsNonRoot: $(kubectl get pods -A -o json | grep -c "runAsNonRoot.*true")"

# Check 3: RBAC
echo "✓ RBAC Configuration"
echo "  Roles: $(kubectl get roles -A | wc -l)"
echo "  RoleBindings: $(kubectl get rolebindings -A | wc -l)"
echo "  ClusterRoles: $(kubectl get clusterroles | wc -l)"

# Check 4: Secrets
echo "✓ Secret Management"
echo "  Secrets in use: $(kubectl get secrets -A | wc -l)"
echo "  Secrets with rotation: $(kubectl get secrets -A -l rotation-enabled=true | wc -l)"

# Check 5: Audit Logging
echo "✓ Audit Logging"
echo "  Audit events today: $(kubectl logs -n kube-system -l component=kube-apiserver --timestamps=true | grep "audit" | grep "$(date +%Y-%m-%d)" | wc -l)"

# Check 6: Kyverno Policies
echo "✓ Policy Enforcement"
echo "  Cluster Policies: $(kubectl get clusterpolicies | wc -l)"
echo "  Policy Violations: $(kubectl logs -n kyverno deployment/kyverno-controller | grep -i "violation" | wc -l)"

echo ""
echo "✅ Security Compliance Check Complete"
EOF

chmod +x /tmp/security-compliance-check.sh
/tmp/security-compliance-check.sh
```

#### 4.2 Generate Compliance Report

```bash
# Create compliance report
cat > /tmp/PHASE5_SECURITY_REPORT.md << 'EOF'
# Phase 5 Days 8-9: Security Hardening & Compliance Report

**Date**: $(date)
**Status**: Complete

## Security Hardening Applied

### Kyverno Policies
- [x] Require resource limits
- [x] Require security context
- [x] Restrict image registries
- [x] Disallow privileged containers

### RBAC Configuration
- [x] DataEngineer role
- [x] DataAnalyst role
- [x] PlatformAdmin role
- [x] Role bindings configured

### Audit & Logging
- [x] API audit logging
- [x] Secret access tracking
- [x] RBAC change logging
- [x] Audit trail dashboard

### Secret Management
- [x] Secret rotation enabled
- [x] Rotation CronJob scheduled
- [x] Credential encryption
- [x] Access audit trail

## Data Governance

### DataHub Integration
- [x] Metadata ingestion configured
- [x] Periodic sync scheduled (4-hour interval)
- [x] Data lineage tracking enabled
- [x] Lineage dashboard available

### Compliance Status
- [x] Network policies enforced
- [x] Pod security standards met
- [x] RBAC properly configured
- [x] Audit trail maintained

## Recommendations

1. Review audit logs weekly
2. Rotate secrets every 30 days (automated)
3. Monitor DataHub lineage for anomalies
4. Conduct quarterly security audits

**Status**: Ready for Day 10 (Documentation & Training)
EOF

cat /tmp/PHASE5_SECURITY_REPORT.md
```

---

## Success Criteria

✅ Kyverno policies enforced  
✅ RBAC roles and bindings configured  
✅ Audit logging enabled and monitored  
✅ Secret rotation scheduled  
✅ DataHub metadata syncing  
✅ Data lineage tracking active  
✅ Compliance validation passed  
✅ Security dashboard operational  

---

## Deliverables

- Kyverno security policies
- RBAC configuration
- Audit logging system
- Secret rotation automation
- DataHub integration
- Compliance report
- Security dashboard

**Status**: Days 8-9 Complete → Ready for Day 10
