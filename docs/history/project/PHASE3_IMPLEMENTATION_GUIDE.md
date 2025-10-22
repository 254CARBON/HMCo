# Phase 3: High Availability & Resilience - Implementation Guide

**Status**: Ready for Implementation  
**Duration**: 2-3 days  
**Objective**: Configure multi-node cluster with proper anti-affinity, resource management, and service resilience

---

## Overview

Phase 3 transforms the single-node cluster into a highly available, resilient multi-node platform:

1. **Multi-Node Configuration** - Expand from 1 to 3+ nodes
2. **Pod Anti-Affinity** - Distribute pods across nodes
3. **Resource Management** - Set requests/limits and quotas
4. **Service High Availability** - HA for databases and critical services
5. **Horizontal Pod Autoscaling** - Dynamic scaling based on demand

---

## Task 1: Current Single-Node Assessment

### Current State
```
Nodes: 1 (dev-cluster-control-plane)
Pods: 66 total
Distribution: All on single node
Status: No HA, single point of failure
```

### Cluster Information
```bash
kubectl get nodes -o wide
kubectl top nodes
kubectl get pods -A --field-selector=spec.nodeName=dev-cluster-control-plane
```

### Key Metrics
- Total CPU: ~4 cores (varies by node type)
- Total Memory: ~8-16GB (varies by node type)
- Current Utilization: ~30-35%
- Pod Density: 66 pods per node

---

## Task 2: Pod Anti-Affinity Configuration

### Objective
Ensure pods are distributed across multiple nodes to survive node failures.

### Implementation

**Step 2.1: Apply pod anti-affinity to critical services**

```yaml
# Example: DataHub anti-affinity
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datahub-gms
  namespace: data-platform
spec:
  replicas: 3
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - datahub-gms
              topologyKey: kubernetes.io/hostname
```

**Step 2.2: Apply to all critical services**

Critical services needing anti-affinity:
- PostgreSQL (if multi-replica)
- Elasticsearch
- Kafka
- MinIO
- Vault
- All API services (Trino, Doris, Superset, etc.)

**Step 2.3: Pod Disruption Budgets**

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: datahub-pdb
  namespace: data-platform
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: datahub-gms
```

### Verification
```bash
# Check anti-affinity rules
kubectl get pods -A -o wide | grep <service-name>

# Verify pod distribution across nodes
kubectl get pods -A -o wide | awk '{print $8}' | sort | uniq -c
```

---

## Task 3: Resource Management

### Objective
Properly allocate resources to ensure stability and efficient scheduling.

### Implementation

**Step 3.1: Set resource requests and limits**

```yaml
resources:
  requests:
    cpu: "250m"        # Minimum guaranteed
    memory: "512Mi"    # Minimum guaranteed
  limits:
    cpu: "500m"        # Maximum allowed
    memory: "1Gi"      # Maximum allowed
```

**Step 3.2: Resource Quotas per namespace**

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: data-platform-quota
  namespace: data-platform
spec:
  hard:
    requests.cpu: "8"          # Max 8 CPUs total
    requests.memory: "16Gi"    # Max 16GB total
    limits.cpu: "16"           # Max 16 CPU limits
    limits.memory: "32Gi"      # Max 32GB limits
    pods: "100"                # Max 100 pods
```

**Step 3.3: Horizontal Pod Autoscaling (HPA)**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trino-autoscaler
  namespace: data-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trino
  minReplicas: 2
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Task 4: Service High Availability

### Objective
Ensure critical data services have proper replication and failover capabilities.

### 4.1 PostgreSQL High Availability

**Current State**: Single-instance PostgreSQL
**Target State**: Streaming replication with failover

**Implementation**:
1. Deploy 3 PostgreSQL replicas using StatefulSet
2. Configure streaming replication
3. Set up automatic failover with patroni or pg_auto_failover
4. Configure persistent volumes for each replica

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-primary
  namespace: data-platform
spec:
  selector:
    app: postgres
    role: primary
  ports:
  - port: 5432
    name: postgres
  clusterIP: None

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: data-platform
spec:
  serviceName: postgres
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
        role: replica
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - postgres
              topologyKey: kubernetes.io/hostname
      containers:
      - name: postgres
        image: postgres:15.5
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "1000m"
            memory: "2Gi"
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 50Gi
```

### 4.2 Elasticsearch High Availability

**Current State**: Single node
**Target State**: 3-node cluster with proper sharding

```yaml
# Similar to PostgreSQL, deploy as StatefulSet with:
# - 3 replicas spread across nodes
# - Shared storage for data
# - Discovery configuration
# - Shard replication factor
```

### 4.3 MinIO High Availability

**Current State**: Single server
**Target State**: Distributed mode with erasure coding

```bash
# MinIO distributed requires:
# - Minimum 4 drives (can be on same or different nodes)
# - Erasure coding for data protection
# - Load balancing across servers
```

### 4.4 Vault High Availability

**Current State**: Single instance (scaling 0)
**Target State**: 3-node integrated storage cluster

```yaml
# Already configured in Phase 2
# Requires:
# - Scaled to 3 replicas
# - Shared PostgreSQL backend
# - Load balancer (via service)
```

---

## Task 5: Monitoring High Availability

### Objective
Ensure HA configuration is observable and alerts on failures.

### Implementation

**Step 5.1: Monitor node health**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: node-health
  namespace: monitoring
spec:
  groups:
  - name: node.rules
    interval: 30s
    rules:
    - alert: NodeNotReady
      expr: kube_node_status_condition{condition="Ready",status="true"} == 0
      for: 5m
      annotations:
        summary: "Node {{ $labels.node }} is not ready"
```

**Step 5.2: Monitor pod distribution**

```yaml
- alert: PodNotDistributed
  expr: count(kube_pod_info) by (node_name) > 30
  for: 5m
  annotations:
    summary: "Node {{ $labels.node_name }} has {{ $value }} pods (should be distributed)"
```

**Step 5.3: Monitor PodDisruptionBudget violations**

```yaml
- alert: PDBViolation
  expr: kube_poddisruptionbudget_disruptions_allowed < 1
  annotations:
    summary: "PDB {{ $labels.poddisruptionbudget }} cannot tolerate disruption"
```

---

## Task 6: Cluster Expansion (Multi-Node)

### Prerequisites
- Infrastructure to support additional nodes (local or cloud)
- Network connectivity between nodes
- Shared storage backend (optional but recommended)

### Implementation

**Option A: Add Worker Nodes to Existing Cluster**

```bash
# 1. Provision new VMs/instances (2-3 more nodes)
# 2. Install Kubernetes runtime (containerd/docker)
# 3. Join to cluster:
kubeadm join <control-plane-ip>:6443 \
  --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash>

# 4. Label nodes for workload segregation:
kubectl label node node-2 node-type=worker
kubectl label node node-3 node-type=worker
```

**Option B: If Using Kind/Docker for Testing**

```bash
# Kind clusters can only have 1 control plane in docker
# For HA testing, use kubeadm or managed Kubernetes
```

### Verification
```bash
kubectl get nodes -o wide
kubectl top nodes
kubectl describe nodes
```

---

## Task 7: Load Balancing

### Objective
Ensure services are accessible even if nodes fail.

### Implementation

**Step 7.1: Service type configuration**

```yaml
# Use ClusterIP with ingress (current)
# Or NodePort for direct access:
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: data-platform
spec:
  type: NodePort
  selector:
    app: datahub-gms
  ports:
  - port: 8080
    targetPort: 8080
    nodePort: 30001
```

**Step 7.2: Ingress load balancing**

```yaml
# Already configured with NGINX ingress
# Ensure replicas are spread across nodes:
kubectl get deployment -n ingress-nginx -o wide
```

**Step 7.3: Service mesh (optional)**

For advanced load balancing and traffic management:
- Deploy Istio or Linkerd
- Configure DestinationRules for load balancing
- Implement circuit breakers and retries

---

## Completion Checklist - Phase 3

- [ ] **Pod Anti-Affinity**
  - [ ] Anti-affinity rules applied to critical services
  - [ ] Pod distribution verified across nodes
  - [ ] PodDisruptionBudgets configured

- [ ] **Resource Management**
  - [ ] Resource requests/limits set for all pods
  - [ ] ResourceQuotas applied to namespaces
  - [ ] HPA rules configured for dynamic services

- [ ] **Service HA**
  - [ ] PostgreSQL replication configured
  - [ ] Elasticsearch cluster formed
  - [ ] MinIO distributed mode verified
  - [ ] Vault scaled to 3 replicas

- [ ] **Monitoring**
  - [ ] Node health alerts configured
  - [ ] Pod distribution monitored
  - [ ] PDB alerts active

- [ ] **Multi-Node (if applicable)**
  - [ ] Additional nodes provisioned
  - [ ] Nodes joined to cluster
  - [ ] Workload distribution verified

---

## Success Metrics - Phase 3

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Node Count | 3+ | `kubectl get nodes` |
| Pod Distribution | Spread across nodes | `kubectl get pods -A -o wide` |
| Resource Utilization | <70% per node | `kubectl top nodes` |
| Service Replicas | 2+ per service | `kubectl get deploy -A` |
| HPA Active | Services scale dynamically | Check CPU/memory metrics |
| Failover Tested | <5min recovery | Simulate node failure |

---

## Troubleshooting

### Pods stuck in Pending
```bash
# Check resource availability
kubectl describe node <node-name>

# Check pod requirements
kubectl describe pod <pod-name> -n <namespace>

# Relax affinity rules if needed
```

### Anti-affinity causing pod eviction
```bash
# Reduce replicas or adjust topologyKey
kubectl edit deployment <name> -n <namespace>
```

### HPA not scaling
```bash
# Check metrics server
kubectl get deployment metrics-server -n kube-system

# Verify HPA status
kubectl describe hpa <name> -n <namespace>
```

---

## Next Phase: Phase 4 - Monitoring & Observability

After HA is configured:
1. Enhanced Prometheus configuration
2. Comprehensive Grafana dashboards
3. AlertManager with notification channels
4. Distributed tracing (optional)

---

**Phase 3 Status**: Ready to Begin  
**Estimated Duration**: 2-3 days  
**Next Review**: After Task 1 (Assessment) completion
