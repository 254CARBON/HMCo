# Production Deployment & Operations Guide

**Version**: 1.0  
**Status**: Phase 1 Complete - Ready for Phase 2 Deployment  
**Last Updated**: 2025-10-19

---

## Overview

This guide covers deploying the production-grade Kubernetes cluster and operational procedures for ongoing management. The cluster uses a phased approach to stabilization with Phase 1 foundations now in place.

---

## Phase 1 Deployment Status

**✅ COMPLETED** - The following production-ready components are available for deployment:

### 1. Storage Infrastructure
**File**: `k8s/storage/storage-classes.yaml`

```bash
# Deploy fixed storage classes
kubectl apply -f k8s/storage/storage-classes.yaml

# Verify
kubectl get storageclass
kubectl get volumesnapshotclasses
```

**Key changes from original**:
- Removed duplicate storage class definitions
- Changed reclaimPolicy from Delete to Retain (prevents accidental data loss)
- Added VolumeSnapshotClass for backups
- Added high-throughput storage class for Kafka/Elasticsearch

### 2. Network Policies (Zero-Trust Security)
**File**: `k8s/networking/network-policies.yaml`

```bash
# Deploy network policies
kubectl apply -f k8s/networking/network-policies.yaml

# Verify policies are in place
kubectl get networkpolicy -A
kubectl describe networkpolicy -n data-platform
```

**Important**: These policies enforce strict namespace isolation. Verify pod-to-pod communication works before enforcing.

### 3. Certificate Management
**File**: `k8s/certificates/cert-manager-setup.yaml`

```bash
# Deploy cert-manager
kubectl apply -f k8s/certificates/cert-manager-setup.yaml

# Verify deployment
kubectl get pods -n cert-manager
kubectl get clusterissuer

# Test certificate issuance
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: test-cert
  namespace: default
spec:
  secretName: test-tls
  issuerRef:
    name: selfsigned
    kind: ClusterIssuer
  commonName: test.example.com
EOF

# Check certificate status
kubectl describe certificate test-cert
```

### 4. Monitoring & Alerting
**Files**: 
- `k8s/monitoring/alert-manager.yaml`
- `k8s/monitoring/alerting-rules.yaml`

```bash
# Deploy AlertManager
kubectl apply -f k8s/monitoring/alert-manager.yaml

# Deploy alerting rules (requires Prometheus Operator)
kubectl apply -f k8s/monitoring/alerting-rules.yaml

# Verify AlertManager
kubectl get pods -n monitoring | grep alertmanager
kubectl get svc alertmanager -n monitoring

# Access AlertManager UI
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093
```

### 5. High Availability & Resource Control
**Files**:
- `k8s/resilience/pod-disruption-budgets.yaml`
- `k8s/resilience/resource-quotas.yaml`

```bash
# Deploy PodDisruptionBudgets
kubectl apply -f k8s/resilience/pod-disruption-budgets.yaml

# Deploy resource quotas and limits
kubectl apply -f k8s/resilience/resource-quotas.yaml

# Verify
kubectl get pdb -A
kubectl get resourcequota -A
kubectl get limitrange -A
```

### 6. RBAC & Security Policies
**File**: `k8s/rbac/rbac-audit.yaml`

```bash
# Deploy RBAC configuration
kubectl apply -f k8s/rbac/rbac-audit.yaml

# Verify service accounts
kubectl get serviceaccount -n data-platform
kubectl get serviceaccount -n vault-prod
kubectl get roles -n vault-prod
```

### 7. Storage Backup Policies
**File**: `k8s/storage/backup-policy.yaml`

```bash
# Deploy backup policies
kubectl apply -f k8s/storage/backup-policy.yaml

# Verify snapshots
kubectl get volumesnapshot -A
kubectl describe volumesnapshot -n data-platform

# Verify cleanup job
kubectl get cronjob -n data-platform snapshot-retention-cleanup
kubectl get jobs -n data-platform
```

---

## Deployment Order

For a fresh cluster deployment, follow this sequence:

1. **Namespaces** (already exist)
   ```bash
   kubectl apply -f k8s/namespaces/
   ```

2. **Storage Classes** (required by all other services)
   ```bash
   kubectl apply -f k8s/storage/storage-classes.yaml
   ```

3. **RBAC & Security** (establish access patterns early)
   ```bash
   kubectl apply -f k8s/rbac/rbac-audit.yaml
   kubectl apply -f k8s/networking/network-policies.yaml
   ```

4. **Resource Controls** (prevent resource starvation)
   ```bash
   kubectl apply -f k8s/resilience/resource-quotas.yaml
   kubectl apply -f k8s/resilience/pod-disruption-budgets.yaml
   ```

5. **Certificate Management** (enables HTTPS everywhere)
   ```bash
   kubectl apply -f k8s/certificates/cert-manager-setup.yaml
   ```

6. **Core Data Services** (databases, message brokers, storage)
   ```bash
   kubectl apply -f k8s/shared/postgres/
   kubectl apply -f k8s/shared/kafka/
   kubectl apply -f k8s/datahub/elasticsearch.yaml
   kubectl apply -f k8s/data-lake/minio.yaml
   # etc.
   ```

7. **Monitoring & Alerting**
   ```bash
   kubectl apply -f k8s/monitoring/
   kubectl apply -f k8s/monitoring/alert-manager.yaml
   kubectl apply -f k8s/monitoring/alerting-rules.yaml
   ```

8. **Ingress & API Gateway**
   ```bash
   kubectl apply -f k8s/ingress/
   ```

9. **Application Services**
   ```bash
   kubectl apply -f k8s/datahub/
   kubectl apply -f k8s/visualization/
   kubectl apply -f k8s/seatunnel/
   kubectl apply -f k8s/compute/
   ```

10. **Backups & Disaster Recovery**
    ```bash
    kubectl apply -f k8s/storage/backup-policy.yaml
    ```

---

## Verification Checklist

After Phase 1 deployment:

```bash
# 1. Verify all namespaces exist
kubectl get namespaces | grep -E "data-platform|monitoring|vault|cert-manager|ingress"

# 2. Check storage classes
kubectl get storageclass
# Should show: local-storage-fast, local-storage-standard (default), local-storage-high-throughput, local-storage-snapshot

# 3. Verify network policies
kubectl get networkpolicy -A
# Should show ~8 policies across namespaces

# 4. Check cert-manager
kubectl get pods -n cert-manager -l app.kubernetes.io/name=cert-manager
# Should show 2 cert-manager pods (running)

# 5. Verify AlertManager
kubectl get pods -n monitoring -l app=alertmanager
# Should show 2 alertmanager pods (running)

# 6. Check resource quotas
kubectl describe resourcequota -n data-platform data-platform-quota
# Verify CPU/memory limits are set

# 7. Verify PodDisruptionBudgets
kubectl get pdb -n data-platform
# Should show PDBs for critical services

# 8. Check service accounts
kubectl get serviceaccount -n data-platform | grep -E "platform-app|vault-client"
```

---

## Common Operations

### Troubleshooting Network Policies

```bash
# Test pod-to-pod communication
# Pod 1: Running in data-platform
kubectl exec -n data-platform <pod-name> -- curl http://service-name:port

# If timeout, check policies
kubectl get networkpolicy -n data-platform
kubectl describe networkpolicy <policy-name> -n data-platform

# Temporarily disable policies (development only!)
kubectl delete networkpolicy -n data-platform --all
```

### Monitoring Certificate Renewal

```bash
# List all certificates
kubectl get certificate -A
kubectl get secret -A | grep tls

# Check specific certificate
kubectl describe certificate <name> -n <namespace>

# Check cert-manager logs
kubectl logs -n cert-manager -l app.kubernetes.io/name=cert-manager -f
```

### Viewing Alerts

```bash
# Port forward to AlertManager
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093

# Check Prometheus for alerts
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/alerts
```

### Backup Verification

```bash
# List all snapshots
kubectl get volumesnapshot -A

# Check specific snapshot status
kubectl describe volumesnapshot <name> -n <namespace>

# Trigger manual backup verification
kubectl apply -f k8s/storage/backup-policy.yaml

# View backup logs
kubectl logs -n data-platform job/verify-backups
```

---

## Phase 2 Next Steps

After Phase 1 is stable (1-2 weeks), proceed to Phase 2:

1. **Vault Production Migration**
   - Move from in-memory dev Vault to PostgreSQL-backed HA
   - See: `k8s/vault/vault-production.yaml`

2. **GitOps Implementation**
   - Deploy FluxCD for automated reconciliation
   - Set up policy enforcement with Kyverno

3. **Advanced Monitoring**
   - Distributed tracing (Jaeger/Tempo)
   - Custom dashboards for business metrics

4. **Data Pipeline Hardening**
   - Enhanced SeaTunnel/Flink configuration
   - Job checkpointing and failure recovery

---

## Support & Troubleshooting

### Logs & Debugging

```bash
# Get component logs
kubectl logs -n monitoring -l app=alertmanager
kubectl logs -n cert-manager -l app.kubernetes.io/name=cert-manager
kubectl logs -n monitoring -l app=prometheus

# Describe pods for events
kubectl describe pod <pod-name> -n <namespace>

# Check node status
kubectl get nodes -o wide
kubectl describe node <node-name>
```

### Resource Usage

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -A

# Check quota usage
kubectl describe resourcequota -n data-platform
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Pods pending | No matching storage class | Verify `kubectl get storageclass` |
| Network timeout | NetworkPolicy blocking | Check `kubectl get networkpolicy -n <ns>` |
| Certificate not issued | cert-manager not running | Verify `kubectl get pods -n cert-manager` |
| Alerts not firing | AlertManager misconfigured | Check config in `kubectl get cm -n monitoring` |
| PDB blocking eviction | minAvailable too high | Review PDB settings in Phase 1 manifests |

---

## References

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Storage Classes](https://kubernetes.io/docs/concepts/storage/storage-classes/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [cert-manager Docs](https://cert-manager.io/docs/)
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/overview/)
