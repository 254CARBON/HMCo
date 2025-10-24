# Phase 3: Advanced Features - Implementation Plan

**Date**: October 24, 2025 04:15 UTC  
**Starting Platform Health**: 83% (108/130 pods)  
**Phase 2 Status**: ✅ 95% Complete  
**Estimated Duration**: 3-4 hours  
**Target**: Deploy advanced analytics and ML capabilities

---

## 🎯 Phase 3 Objectives

### Primary Goals
1. **Deploy DataHub with full prerequisites** - Enable data catalog and governance
2. **Deploy Doris OLAP** - Add real-time analytics capabilities  
3. **Complete ML Platform** - Enable end-to-end ML workflows
4. **Achieve 90%+ platform health** - Reach production-grade stability

### Success Metrics
| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Platform Health | 83% | 90%+ | High |
| Running Pods | 108 | 120+ | High |
| Advanced Services | 0/3 | 3/3 | Critical |
| ML Capabilities | Partial | Complete | High |
| Data Catalog | None | Operational | Medium |

---

## 📋 Phase 3 Components

### 1. DataHub Prerequisites & Deployment (90-120 min)

#### 1.1 Elasticsearch Cluster (30 min)
**Purpose**: Search and indexing backend for DataHub

**Deployment**:
```bash
# Using Bitnami Helm chart
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install elasticsearch bitnami/elasticsearch \
  --namespace data-platform \
  --set master.replicaCount=1 \
  --set data.replicaCount=2 \
  --set coordinating.replicaCount=1 \
  --set master.persistence.size=20Gi \
  --set data.persistence.size=50Gi
```

**Expected Resources**:
- Master node: 1 pod
- Data nodes: 2 pods
- Coordinating node: 1 pod
- **Total**: 4 pods, ~70Gi storage

#### 1.2 Kafka Cluster (30 min)
**Purpose**: Message streaming for DataHub metadata events

**Deployment**:
```bash
# Using Bitnami Kafka
helm install kafka bitnami/kafka \
  --namespace data-platform \
  --set replicaCount=3 \
  --set persistence.size=20Gi \
  --set zookeeper.enabled=false \
  --set kraft.enabled=true
```

**Expected Resources**:
- Kafka brokers: 3 pods
- **Total**: 3 pods, 60Gi storage

#### 1.3 Neo4j Graph Database (20 min)
**Purpose**: Graph database for DataHub lineage

**Deployment**:
```bash
# Using Neo4j Helm chart
helm repo add neo4j https://helm.neo4j.com/neo4j
helm install neo4j neo4j/neo4j \
  --namespace data-platform \
  --set neo4j.password=datahub_neo4j_password \
  --set core.standalone=true \
  --set volumes.data.mode=defaultStorageClass \
  --set volumes.data.size=30Gi
```

**Expected Resources**:
- Neo4j core: 1 pod
- **Total**: 1 pod, 30Gi storage

#### 1.4 Enable DataHub Services (10 min)
**Action**: Update data-platform Helm values

```yaml
# helm/charts/data-platform/values.yaml
datahub:
  enabled: true  # Change from false
  elasticsearch:
    host: elasticsearch-master.data-platform.svc.cluster.local
    port: 9200
  kafka:
    bootstrap:
      servers: kafka.data-platform.svc.cluster.local:9092
  neo4j:
    uri: bolt://neo4j.data-platform.svc.cluster.local:7687
    password: datahub_neo4j_password
```

**Expected DataHub Pods**:
- datahub-gms: 1 pod
- datahub-frontend: 1 pod
- datahub-mae-consumer: 1 pod
- datahub-mce-consumer: 1 pod
- **Total**: 4 pods

**Total Phase 1 Resources**: 12 new pods, 160Gi storage

---

### 2. Doris OLAP Deployment (45-60 min)

#### 2.1 Create Doris Namespace (2 min)
```bash
kubectl create namespace doris-analytics
kubectl label namespace doris-analytics istio-injection=enabled
```

#### 2.2 Install Doris Operator (15 min)
```bash
# Using official Doris Operator
kubectl apply -f https://raw.githubusercontent.com/apache/doris-operator/main/config/operator/operator.yaml -n doris-analytics

# Wait for operator
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=doris-operator -n doris-analytics --timeout=300s
```

**Expected Resources**:
- Doris operator: 1 pod

#### 2.3 Deploy DorisCluster CRD (25 min)
**Create DorisCluster manifest**:

```yaml
# config/doris-cluster.yaml
apiVersion: doris.selectdb.com/v1
kind: DorisCluster
metadata:
  name: doris-cluster
  namespace: doris-analytics
spec:
  feSpec:
    replicas: 1
    image: apache/doris:fe-3.0.8
    limits:
      cpu: 2
      memory: 4Gi
    requests:
      cpu: 1
      memory: 2Gi
    persistentVolume:
      storageClassName: local-storage-standard
      requests:
        storage: 30Gi
  beSpec:
    replicas: 2
    image: apache/doris:be-3.0.8
    limits:
      cpu: 4
      memory: 8Gi
    requests:
      cpu: 2
      memory: 4Gi
    persistentVolume:
      storageClassName: local-storage-standard
      requests:
        storage: 100Gi
```

```bash
kubectl apply -f config/doris-cluster.yaml
```

**Expected Resources**:
- FE (Frontend): 1 pod, 30Gi
- BE (Backend): 2 pods, 200Gi
- **Total**: 3 pods, 230Gi storage

#### 2.4 Configure Superset Connection (10 min)
```bash
# Connect Superset to Doris
kubectl exec -n data-platform deploy/superset-web -- superset fab create-db \
  --database_name "Doris OLAP" \
  --sqlalchemy_uri "mysql://admin:doris_password@doris-fe-service.doris-analytics:9030/information_schema"
```

**Total Phase 2 Resources**: 4 new pods, 230Gi storage

---

### 3. Complete ML Platform (60-75 min)

#### 3.1 Deploy Ray Cluster (20 min)
**Sync Ray ArgoCD Applications**:

```bash
# Sync Ray infrastructure
kubectl annotate application kuberay-crds -n argocd argocd.argoproj.io/refresh=hard --overwrite
kubectl annotate application kuberay-operator-helm -n argocd argocd.argoproj.io/refresh=hard --overwrite
kubectl annotate application ray-serve -n argocd argocd.argoproj.io/refresh=hard --overwrite

# Wait for operator
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=kuberay-operator -n ray-system --timeout=300s
```

**Create Ray Cluster**:
```yaml
# config/ray-cluster.yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: ml-cluster
  namespace: ml-platform
spec:
  rayVersion: '2.9.0'
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0
          resources:
            limits:
              cpu: 2
              memory: 4Gi
            requests:
              cpu: 1
              memory: 2Gi
  workerGroupSpecs:
  - replicas: 2
    minReplicas: 1
    maxReplicas: 4
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0
          resources:
            limits:
              cpu: 4
              memory: 8Gi
            requests:
              cpu: 2
              memory: 4Gi
```

```bash
kubectl apply -f config/ray-cluster.yaml
```

**Expected Resources**:
- Ray head: 1 pod
- Ray workers: 2 pods
- KubeRay operator: 1 pod
- **Total**: 4 pods

#### 3.2 Deploy MLflow (20 min)
**Update ML Platform values**:

```yaml
# helm/charts/ml-platform/charts/mlflow/values.yaml
mlflow:
  enabled: true
  replicaCount: 1
  backendStore:
    postgres:
      host: postgresql.kong.svc.cluster.local
      port: 5432
      database: mlflow
      username: mlflow
      password: mlflow_password
  artifactStore:
    s3:
      enabled: true
      bucket: mlflow-artifacts
      endpoint: http://minio-service.data-platform.svc.cluster.local:9000
      accessKey: minio
      secretKey: minio_secret
```

**Sync ArgoCD**:
```bash
kubectl annotate application ml-platform -n argocd argocd.argoproj.io/refresh=hard --overwrite
```

**Expected Resources**:
- MLflow server: 1 pod
- **Total**: 1 pod

#### 3.3 Deploy Kubeflow Pipelines (25 min)
**Enable in ML Platform**:

```yaml
# helm/charts/ml-platform/charts/kubeflow/values.yaml
kubeflow:
  pipelines:
    enabled: true
  katib:
    enabled: true
```

**Sync ArgoCD**:
```bash
kubectl annotate application ml-platform -n argocd argocd.argoproj.io/refresh=hard --overwrite
```

**Expected Resources**:
- KF Pipelines API: 1 pod
- KF Pipelines UI: 1 pod
- KF Persistence: 1 pod
- Katib: 3 pods
- **Total**: 6 pods

**Total Phase 3 Resources**: 11 new pods

---

## 🔄 Implementation Sequence

### **Step 1: Prerequisites Preparation** (10 min)
```bash
# Create namespaces
kubectl create namespace doris-analytics
kubectl create namespace ml-platform --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repos
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add neo4j https://helm.neo4j.com/neo4j
helm repo update

# Verify storage availability
kubectl get sc
```

### **Step 2: Deploy Elasticsearch** (30 min)
```bash
# Install Elasticsearch
helm install elasticsearch bitnami/elasticsearch \
  --namespace data-platform \
  --set master.replicaCount=1 \
  --set data.replicaCount=2 \
  --set coordinating.replicaCount=1 \
  --set master.persistence.size=20Gi \
  --set data.persistence.size=50Gi \
  --wait

# Verify
kubectl get pods -n data-platform -l app.kubernetes.io/name=elasticsearch
```

### **Step 3: Deploy Kafka** (30 min)
```bash
# Install Kafka with KRaft
helm install kafka bitnami/kafka \
  --namespace data-platform \
  --set replicaCount=3 \
  --set persistence.size=20Gi \
  --set kraft.enabled=true \
  --set zookeeper.enabled=false \
  --wait

# Verify
kubectl get pods -n data-platform -l app.kubernetes.io/name=kafka
```

### **Step 4: Deploy Neo4j** (20 min)
```bash
# Install Neo4j
helm install neo4j neo4j/neo4j \
  --namespace data-platform \
  --set neo4j.password=datahub_neo4j_password \
  --set core.standalone=true \
  --set volumes.data.size=30Gi \
  --wait

# Verify
kubectl get pods -n data-platform -l app.kubernetes.io/name=neo4j
```

### **Step 5: Enable DataHub** (20 min)
```bash
# Update Helm values
# Edit helm/charts/data-platform/values.yaml
# Set datahub.enabled=true

# Commit and push
git add helm/charts/data-platform/values.yaml
git commit -m "feat: enable DataHub with prerequisites"
git push

# Sync ArgoCD
kubectl annotate application data-platform -n argocd argocd.argoproj.io/refresh=hard --overwrite

# Wait for DataHub
kubectl wait --for=condition=ready pod -l app=datahub-gms -n data-platform --timeout=600s
```

### **Step 6: Deploy Doris Operator** (45 min)
```bash
# Install operator
kubectl apply -f https://raw.githubusercontent.com/apache/doris-operator/main/config/operator/operator.yaml -n doris-analytics

# Wait
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=doris-operator -n doris-analytics --timeout=300s

# Deploy cluster
kubectl apply -f config/doris-cluster.yaml

# Wait for FE
kubectl wait --for=condition=ready pod -l app=doris-fe -n doris-analytics --timeout=600s
```

### **Step 7: Deploy ML Platform** (60 min)
```bash
# Sync Ray
kubectl annotate application kuberay-operator-helm -n argocd argocd.argoproj.io/refresh=hard --overwrite
kubectl annotate application ray-serve -n argocd argocd.argoproj.io/refresh=hard --overwrite

# Deploy Ray cluster
kubectl apply -f config/ray-cluster.yaml

# Enable MLflow & Kubeflow
# Edit helm/charts/ml-platform/values.yaml

# Commit and sync
git add helm/charts/ml-platform
git commit -m "feat: enable complete ML platform with Ray, MLflow, Kubeflow"
git push

kubectl annotate application ml-platform -n argocd argocd.argoproj.io/refresh=hard --overwrite
```

---

## 📊 Expected Final State

### Platform Health
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Total Pods | 130 | 157 | +27 |
| Running Pods | 108 | 145+ | +37 |
| Platform Health | 83% | 92%+ | +9% |
| Storage Used | ~300Gi | ~690Gi | +390Gi |

### New Services Operational
✅ **DataHub** - Data catalog and governance  
✅ **Elasticsearch** - Search and indexing (4 pods)  
✅ **Kafka** - Event streaming (3 pods)  
✅ **Neo4j** - Graph database (1 pod)  
✅ **Doris** - OLAP analytics (3 pods)  
✅ **Ray** - Distributed computing (4 pods)  
✅ **MLflow** - ML experiment tracking (1 pod)  
✅ **Kubeflow** - ML pipelines (6 pods)  

### Service URLs
- DataHub: https://datahub.254carbon.com
- Doris FE: doris-fe-service.doris-analytics:9030
- Ray Dashboard: https://ray.254carbon.com
- MLflow: https://mlflow.254carbon.com
- Kubeflow: https://kubeflow.254carbon.com

---

## 🛠️ Rollback Plan

### If Elasticsearch fails:
```bash
helm uninstall elasticsearch -n data-platform
kubectl delete pvc -n data-platform -l app.kubernetes.io/name=elasticsearch
```

### If Kafka fails:
```bash
helm uninstall kafka -n data-platform
kubectl delete pvc -n data-platform -l app.kubernetes.io/name=kafka
```

### If DataHub fails:
```bash
# Disable in values.yaml
datahub.enabled: false
# Sync ArgoCD
```

### If Doris fails:
```bash
kubectl delete -f config/doris-cluster.yaml
kubectl delete namespace doris-analytics
```

### If ML Platform fails:
```bash
kubectl annotate application ml-platform -n argocd argocd.argoproj.io/sync-options=Prune=false --overwrite
# Disable in values
# Sync ArgoCD
```

---

## ⚠️ Risk Assessment

### High Risk Items
1. **Storage capacity** - Requires 390Gi additional storage
   - Mitigation: Verify available storage before deployment
   - Alternative: Reduce replica counts or storage sizes

2. **Resource constraints** - 27 additional pods
   - Mitigation: Monitor node resources during deployment
   - Alternative: Deploy in phases with pauses

3. **Network complexity** - Multiple inter-service dependencies
   - Mitigation: Test connectivity after each component
   - Alternative: Use debug pods for troubleshooting

### Medium Risk Items
1. **Elasticsearch startup time** - Can take 5-10 minutes
2. **Kafka initialization** - Requires proper configuration
3. **Neo4j authentication** - Password must be consistent

---

## 📝 Pre-Deployment Checklist

- [ ] Verify platform health ≥80% before starting
- [ ] Check available storage capacity (need 400Gi+)
- [ ] Verify all Phase 2 critical services running
- [ ] Backup current platform state
- [ ] Ensure Git repository is clean
- [ ] Verify ArgoCD applications healthy
- [ ] Check node resources (CPU, memory available)

---

## 🎯 Post-Deployment Validation

### Component Health Checks
```bash
# Elasticsearch
kubectl exec -n data-platform elasticsearch-master-0 -- curl -s http://localhost:9200/_cluster/health

# Kafka
kubectl exec -n data-platform kafka-0 -- kafka-topics.sh --list --bootstrap-server localhost:9092

# Neo4j
kubectl exec -n data-platform neo4j-0 -- cypher-shell -u neo4j -p datahub_neo4j_password "RETURN 1"

# DataHub
curl -I https://datahub.254carbon.com

# Doris
kubectl exec -n doris-analytics doris-fe-0 -- mysql -h 127.0.0.1 -P 9030 -u root -e "SHOW FRONTENDS"

# Ray
kubectl get raycluster -n ml-platform

# MLflow
curl -I https://mlflow.254carbon.com

# Kubeflow
curl -I https://kubeflow.254carbon.com
```

### Integration Tests
1. **DataHub**: Create test dataset and verify lineage
2. **Doris**: Create table and run simple query
3. **Ray**: Submit test job
4. **MLflow**: Log test experiment
5. **Kubeflow**: Create test pipeline

---

## 🚀 Success Criteria

### Must Have (Critical)
- [ ] All 27 new pods running
- [ ] DataHub UI accessible
- [ ] Doris FE responding to queries
- [ ] Ray cluster operational
- [ ] MLflow tracking server responding
- [ ] Platform health ≥90%

### Should Have (Important)
- [ ] Kubeflow pipelines functional
- [ ] DataHub ingesting metadata
- [ ] Doris connected to Superset
- [ ] Ray workers auto-scaling
- [ ] MLflow connected to MinIO

### Nice to Have (Optional)
- [ ] DataHub lineage visualization working
- [ ] Doris performance optimized
- [ ] Ray dashboard accessible externally
- [ ] Kubeflow notebooks deployed
- [ ] Complete ML workflow tested

---

**Plan Status**: READY FOR EXECUTION  
**Estimated Time**: 3-4 hours  
**Risk Level**: MEDIUM  
**Recommended Approach**: Phased deployment with validation between phases

**Next Action**: Execute Step 1 (Prerequisites Preparation)

