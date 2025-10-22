# Apache Doris Kubernetes Operator - Production Deployment Recommendation

**Date**: October 21, 2025  
**Recommendation**: ✅ **USE DORIS OPERATOR FOR PRODUCTION**  
**Priority**: Medium (deploy when analytics database is needed)

---

## Executive Summary

After investigating the Apache Doris deployment issues on the 254Carbon Data Platform, we strongly recommend using the official Apache Doris Kubernetes Operator for production deployments instead of manual StatefulSet configurations.

---

## Current Situation

### Manual Deployment Attempts

**Challenges Encountered**:
1. ❌ Image tag confusion (2.1.0-fe vs fe-3.0.8 format)
2. ❌ Undocumented environment variable requirements
3. ❌ Complex initialization sequences
4. ❌ Container crashes with "Missing required parameters"
5. ❌ Difficult to manage FE-BE cluster coordination

**Files Modified**:
- `k8s/compute/doris/doris-fe.yaml` - Updated but not functional
- `k8s/compute/doris/doris-be.yaml` - Updated but not functional

**Current Status**: Doris pods crashing despite correct image tags and initial configuration

---

## Doris Operator Solution

### Why Use the Operator?

**1. Automated Lifecycle Management**
- ✅ Proper initialization sequences handled automatically
- ✅ FE-BE cluster coordination managed by operator
- ✅ Health checks and readiness probes configured correctly
- ✅ Rolling updates without downtime
- ✅ Automatic failover and recovery

**2. Simplified Configuration**
- ✅ Single CRD defines entire cluster
- ✅ No need to understand internal Doris communication protocols
- ✅ Environment variables managed automatically
- ✅ Service discovery built-in

**3. Production Features**
- ✅ Horizontal scaling (add/remove FE and BE nodes)
- ✅ Persistent volume management
- ✅ Backup and restore integration
- ✅ Monitoring and metrics export
- ✅ Configuration management

**4. Community Support**
- ✅ Official Apache project
- ✅ Active development and maintenance
- ✅ Production-tested by many organizations
- ✅ Documentation and examples

---

## Installation Guide

### Prerequisites

- Kubernetes cluster 1.19+ (✅ We have 1.34.1)
- Helm 3.0+ (✅ Installed)
- kubectl configured (✅ Configured)
- StorageClass for persistent volumes (✅ local-path available)

### Step 1: Install Doris Operator

```bash
# Add Doris Operator Helm repository
helm repo add doris-operator https://apache.github.io/doris-operator
helm repo update

# Create operator namespace
kubectl create namespace doris-operator

# Install operator
helm install doris-operator doris-operator/doris-operator \
  --namespace doris-operator \
  --set image.repository=selectdb/doris-operator \
  --set image.tag=latest \
  --create-namespace
```

### Step 2: Verify Operator Installation

```bash
# Check operator pod
kubectl get pods -n doris-operator

# Check CRDs
kubectl get crds | grep doris

# Expected output:
# dorisclusters.doris.selectdb.com
# dorisautoscalers.doris.selectdb.com
```

### Step 3: Create Doris Cluster

Create file `k8s/compute/doris/doris-cluster-operator.yaml`:

```yaml
apiVersion: doris.selectdb.com/v1
kind: DorisCluster
metadata:
  name: doris-cluster
  namespace: data-platform
spec:
  # Frontend (FE) specification
  feSpec:
    replicas: 1
    image: apache/doris:fe-3.0.8
    resources:
      requests:
        cpu: "1"
        memory: 4Gi
      limits:
        cpu: "2"
        memory: 8Gi
    persistentVolume:
      size: 20Gi
      storageClassName: local-path
    service:
      type: ClusterIP
    env:
      - name: JAVA_OPTS
        value: "-Xms2g -Xmx4g"
  
  # Backend (BE) specification
  beSpec:
    replicas: 1
    image: apache/doris:be-3.0.8
    resources:
      requests:
        cpu: "2"
        memory: 8Gi
      limits:
        cpu: "4"
        memory: 16Gi
    persistentVolume:
      size: 100Gi
      storageClassName: local-path
    service:
      type: ClusterIP
    env:
      - name: JAVA_OPTS
        value: "-Xms4g -Xmx8g"
  
  # Administrative configuration
  adminUser:
    name: admin
    password: changeme123  # Change in production!
  
  # Auto-scaling (optional)
  autoScaler:
    version: v1
    beAutoScaling:
      enable: true
      minReplicas: 1
      maxReplicas: 3
      resourceScaling:
        cpu:
          threshold: 70
        memory:
          threshold: 75
```

### Step 4: Deploy Cluster

```bash
# Apply the DorisCluster CRD
kubectl apply -f k8s/compute/doris/doris-cluster-operator.yaml

# Monitor cluster creation
kubectl get dorisclusters -n data-platform -w

# Check pods
kubectl get pods -n data-platform -l app.doris.ownerreference/name=doris-cluster

# Check services
kubectl get svc -n data-platform | grep doris
```

### Step 5: Verify Cluster Health

```bash
# Check cluster status
kubectl describe doriscluster doris-cluster -n data-platform

# Connect to FE
kubectl port-forward -n data-platform svc/doris-cluster-fe-service 9030:9030

# Test connection
mysql -h 127.0.0.1 -P 9030 -u admin -p
# Enter password: changeme123

# Run test query
SHOW FRONTENDS;
SHOW BACKENDS;
```

---

## Migration from Manual Deployment

### Step 1: Clean Up Existing Resources

```bash
# Delete existing StatefulSets
kubectl delete statefulset doris-fe doris-be -n data-platform

# Delete existing services
kubectl delete svc doris-fe-service doris-be-service -n data-platform
kubectl delete svc doris-fe-headless doris-be-headless -n data-platform

# Keep PVCs if you want to preserve data
# Otherwise delete them too:
# kubectl delete pvc -l app=doris-fe -n data-platform
# kubectl delete pvc -l app=doris-be -n data-platform
```

### Step 2: Deploy with Operator

Follow the installation guide above (Steps 1-5).

### Step 3: Migrate Data (if preserving existing data)

If you have existing Doris data in PVCs:

```bash
# Option 1: Reference existing PVCs in operator configuration
# (Advanced - requires operator configuration modification)

# Option 2: Export and re-import data
# 1. Export data from old cluster using mysqldump
# 2. Import to new operator-managed cluster
```

---

## Comparison: Manual vs Operator

| Feature | Manual Deployment | Doris Operator |
|---------|------------------|----------------|
| **Ease of Setup** | ❌ Complex | ✅ Simple |
| **Configuration** | ❌ Multiple YAML files | ✅ Single CRD |
| **Initialization** | ❌ Manual env vars | ✅ Automatic |
| **Scaling** | ❌ Manual pod management | ✅ Automated HPA |
| **Updates** | ❌ Manual rolling update | ✅ Orchestrated upgrade |
| **Monitoring** | ❌ Manual setup | ✅ Built-in metrics |
| **Recovery** | ❌ Complex procedures | ✅ Operator handles |
| **Production Ready** | ⚠️ Not recommended | ✅ Recommended |

---

## Alternative: Managed Doris Services

### Cloud-Managed Options

If operator complexity is still too high, consider:

1. **SelectDB Cloud** (Official managed Doris)
   - Fully managed Doris service
   - Based on Apache Doris
   - Enterprise support
   - https://selectdb.com

2. **Other Analytics Databases**
   - **ClickHouse**: Similar OLAP capabilities, mature K8s operator
   - **Druid**: Real-time analytics, good K8s support
   - **Trino** (already deployed): Federated query engine

**Current Platform**: We already have Trino deployed which can query data from Iceberg, PostgreSQL, MinIO, etc. Consider whether Doris adds significant value before deploying.

---

## Recommendation Summary

### For 254Carbon Platform

**Recommended Approach**:
1. **Option A**: Deploy Doris using Kubernetes Operator (if Doris-specific features needed)
2. **Option B**: Leverage existing Trino + Iceberg stack (simpler, already working)

**Priority**: Medium-Low
- Doris is not critical for current platform operations
- Trino provides similar analytics capabilities
- Focus on stabilizing existing services first

**Timeline**: 
- If needed: 2-4 hours with Doris Operator
- If not needed: Defer until specific Doris features required

### Decision Criteria

**Deploy Doris if you need**:
- Real-time OLAP with sub-second latency
- Doris-specific SQL features
- Column store optimizations
- Built-in data import tools

**Use Trino if sufficient**:
- Federated queries across multiple sources
- Standard SQL interface
- Iceberg integration (already configured)
- Simpler architecture

---

## Implementation Plan (If Proceeding)

### Phase 1: Operator Installation (30 minutes)
1. Install Doris Operator via Helm
2. Verify CRDs and operator pod
3. Review operator logs

### Phase 2: Cluster Deployment (1 hour)
1. Create DorisCluster CRD configuration
2. Apply and monitor cluster creation
3. Verify FE and BE pods start successfully
4. Test cluster connectivity

### Phase 3: Configuration (1 hour)
1. Create databases and tables
2. Configure monitoring
3. Set up ingress for external access
4. Test data loading

### Phase 4: Integration (1 hour)
1. Integrate with DataHub for metadata
2. Configure Superset for visualization
3. Test query performance
4. Document usage examples

**Total Estimated Time**: 3-4 hours

---

**Status**: ✅ Research Complete  
**Recommendation**: Use Doris Operator when analytics database needed  
**Alternative**: Continue with Trino + Iceberg (already operational)

---


