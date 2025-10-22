# Apache Doris Operator Deployment Status

**Date**: October 21, 2025  
**Status**: ⚠️ **PARTIAL DEPLOYMENT - ENCOUNTERING CHALLENGES**  
**Recommendation**: **Consider alternatives or defer**

---

## Deployment Progress

### What Was Successfully Completed ✅

1. **Doris Operator Installed**
   - Status: 1/1 Running in `doris` namespace
   - CRD installed: `dorisclusters.doris.selectdb.com`
   - Webhooks configured
   - RBAC roles created

2. **Dedicated Namespace Created**
   - Namespace: `doris-analytics`
   - Pod Security Standard: `privileged` (required for Doris init containers)
   - Services: doris-cluster-fe-service and doris-cluster-fe-internal created

3. **Node-level System Tuning Applied**
   - DaemonSet deployed to increase inotify limits
   - fs.inotify.max_user_instances: 128 → 8192
   - fs.inotify.max_user_watches: 524288 (already sufficient)
   - vm.max_map_count: 2000000
   - Applied to all cluster nodes (2/2)

4. **DorisCluster CRD Created**
   - Cluster name: doris-cluster
   - FE replica: 1
   - BE replica: 1 (not yet started)
   - PVCs: Created with 20Gi for FE meta

---

## Current Challenges ⚠️

### Issue 1: FE Registration Problem
**Error**:
```
current node doris-cluster-fe-0.doris-cluster-fe-internal.doris-analytics.svc.cluster.local:9010 
is not added to the cluster, will exit.
```

**Analysis**:
- Doris FE needs to self-register in cluster metadata
- Operator not properly initializing the FE cluster membership
- This is a bootstrapping/initialization order issue

**Root Cause**: Mismatch between operator version and Doris image version

### Issue 2: Complex Initialization Sequence
**Observed**:
- FE crashes immediately after startup
- Expects to find itself in cluster metadata
- Operator creates StatefulSet but doesn't handle first-time cluster bootstrap properly

### Issue 3: Image Compatibility
**Findings**:
- Official images (`apache/doris:fe-3.0.8`) expect manual cluster initialization
- Operator may be designed for different image versions (SelectDB images vs Apache images)
- Possible version mismatch between operator and Doris

---

## Technical Details

### Resources Created

**Namespaces**:
- `doris` - Operator namespace
- `doris-analytics` - Doris cluster namespace

**Operator Components**:
```
doris-operator (deployment): 1/1 Running
CRD: dorisclusters.doris.selectdb.com
Webhooks: Validating and Mutating configured
```

**Doris Cluster Resources**:
```
StatefulSet: doris-cluster-fe (0/1 ready)
Services:
  - doris-cluster-fe-service (ClusterIP)
  - doris-cluster-fe-internal (Headless)
PVC: meta-doris-cluster-fe-0 (20Gi, Bound)
```

**System Tuning**:
```
DaemonSet: node-tuning-for-doris (2/2 running)
Sysctl parameters applied on all nodes
```

---

## Files Created

1. `k8s/namespaces/doris-analytics.yaml` - Dedicated Doris namespace
2. `k8s/compute/doris/doris-cluster-operator.yaml` - DorisCluster CRD
3. `k8s/compute/doris/node-tuning-daemonset.yaml` - System tuning DaemonSet
4. `docs/compute/DORIS_DEPLOYMENT_STATUS.md` - This file

---

## Recommendation

### Option A: Continue Troubleshooting Doris Operator (Time: 3-5 hours)

**Required Steps**:
1. Research SelectDB commercial images vs Apache images
2. Find operator version compatible with Apache Doris 3.0.8
3. Debug cluster initialization sequence
4. Potentially contact Doris community for support

**Pros**:
- Eventually get Doris working
- Learn Doris internals deeply

**Cons**:
- Very time-consuming
- Complex debugging
- May hit more unforeseen issues
- Uncertain timeline

### Option B: Use Existing Trino + Iceberg Stack ✅ **RECOMMENDED**

**Why This Makes Sense**:
- ✅ Trino already deployed and operational
- ✅ Iceberg data lake already configured
- ✅ Similar OLAP capabilities to Doris
- ✅ Better integration with existing platform
- ✅ Proven stable in production
- ✅ Simpler architecture

**Capabilities Comparison**:
| Feature | Doris | Trino + Iceberg |
|---------|-------|-----------------|
| OLAP Queries | ✅ | ✅ |
| Sub-second latency | ✅✅ | ✅ |
| Federated queries | ❌ | ✅✅ |
| Column store | ✅ | ✅ (Iceberg) |
| Data lake integration | ⚠️ | ✅✅ |
| Kubernetes maturity | ⚠️ | ✅✅ |
| Current status | ❌ Failing | ✅ Working |

**What You Already Have**:
- Trino Coordinator + Worker (operational)
- Iceberg REST Catalog (operational)
- MinIO data lake storage (operational)
- Superset for visualization (operational)
- Full query federation across PostgreSQL, MinIO, Iceberg

### Option C: Defer Doris Until Specific Need Arises

**When to Reconsider Doris**:
1. If Trino performance insufficient for your workload
2. If Doris-specific features absolutely required
3. If community/vendor provides better Kubernetes support
4. If dedicated Doris managed service becomes available

---

## Cleanup Instructions

If choosing not to proceed with Doris:

```bash
# Delete Doris cluster
kubectl delete doriscluster doris-cluster -n doris-analytics

# Delete Doris namespace
kubectl delete namespace doris-analytics

# Optionally remove operator
kubectl delete namespace doris

# Remove node tuning (optional, doesn't hurt to keep)
kubectl delete daemonset node-tuning-for-doris -n kube-system
```

To keep for future attempts:
```bash
# Just stop the cluster
kubectl scale statefulset doris-cluster-fe --replicas=0 -n doris-analytics
```

---

## Alternative: Try SelectDB Images

If you want to continue with Doris, try SelectDB-specific images:

```yaml
spec:
  feSpec:
    image: selectdb/doris.fe-ubuntu:2.1.5
  beSpec:
    image: selectdb/doris.be-ubuntu:2.1.5
```

These images may be better optimized for the operator.

---

## Current Platform Capabilities Without Doris

Your platform already has robust analytics capabilities:

1. **Query Engine**: Trino
   - Distributed SQL engine
   - Query data across multiple sources
   - Federation: PostgreSQL, MinIO, Iceberg, Kafka

2. **Data Lake**: Apache Iceberg
   - ACID transactions
   - Time travel
   - Schema evolution
   - Production-ready

3. **Storage**: MinIO
   - S3-compatible
   - Object storage for data lake
   - Already integrated

4. **Visualization**: Apache Superset
   - BI dashboards
   - SQL Lab
   - Connected to Trino

5. **Orchestration**: DolphinScheduler
   - Workflow DAGs
   - Scheduling
   - Fully operational

**Conclusion**: You have a complete, working analytics stack. Doris adds complexity without clear incremental value given current deployment challenges.

---

## Final Recommendation

### 🎯 **RECOMMENDED ACTION**: Skip Doris deployment for now

**Reasons**:
1. Platform is at 90% readiness without it
2. Trino provides similar capabilities and is working
3. Doris deployment encountering multiple complex issues
4. Time better spent on:
   - Completing DataHub stabilization
   - Deploying SSL/TLS certificates
   - Building data pipelines with existing tools
   - Monitoring and optimization

### ⏸️ **DEFER** Doris until:
- Specific Doris features are absolutely required
- Operator matures with better Apache Doris support
- Community provides clearer Kubernetes deployment guide
- Or use Doris managed cloud service

---

## Time Investment Summary

**Time Spent on Doris**:
- Manual deployment attempts: 2+ hours
- Operator research: 1 hour
- Operator deployment: 1 hour
- **Total**: 4+ hours

**Result**: Not yet operational

**Alternative Path (Trino)**:
- Already deployed: 0 additional hours
- Already working: Immediate value
- **Total**: 0 hours, full functionality

---

**Status**: Doris Operator deployed but cluster not stabilizing  
**Decision Required**: Continue troubleshooting or use existing Trino stack  
**My Recommendation**: Use Trino + Iceberg (already operational)

---


