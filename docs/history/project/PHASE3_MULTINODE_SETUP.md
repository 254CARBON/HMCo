# Phase 3: Multi-Node Cluster Setup - Complete Guide

**Objective**: Expand cluster from single-node to multi-node High Availability  
**Timeline**: 2-3 days  
**Current Status**: Single node ready, waiting for worker node infrastructure

---

## Current State

```
Control Node: dev-cluster-control-plane (Ready)
  - Role: control-plane
  - Status: Ready
  - Age: 26h
  - K8s Version: v1.31.0
  - Pods: 68 running

Worker Nodes: None yet
  - Need: 2-3 additional worker nodes
  - Status: Awaiting provisioning
```

---

## Prerequisites

Before proceeding, you will need:

### Infrastructure Requirements
1. **2-3 Additional Virtual Machines** (minimum)
   - OS: Debian/Ubuntu 20.04+ or equivalent
   - Kernel: 5.15+
   - CPU: 4+ cores per node (for production)
   - RAM: 8GB+ per node (for production)
   - Storage: 100GB+ per node
   - Network: Must be able to reach 172.19.0.0/16 (control node IP range)

2. **Connectivity**
   - Nodes must have network connectivity to control plane
   - All nodes must have internet access (or private registry)
   - DNS resolution working

### Tools Required
- `kubeadm` (same version as control plane: v1.31.0)
- `kubelet`
- `kubectl`
- Container runtime: containerd (1.7.18 compatible)

---

## Step 1: Prepare Worker Nodes

### 1.1 Install Required Packages

On each worker node, run:

```bash
#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install container runtime prerequisites
sudo apt-get install -y curl gnupg2 lsb-release apt-transport-https ca-certificates

# Install containerd
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y containerd.io

# Configure containerd
sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml
sudo systemctl restart containerd

# Install kubeadm, kubelet, kubectl (v1.31.0)
curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt-get update
sudo apt-get install -y kubelet=1.31.0-00 kubeadm=1.31.0-00 kubectl=1.31.0-00
sudo apt-mark hold kubelet kubeadm kubectl

# Enable kubelet
sudo systemctl daemon-reload
sudo systemctl enable kubelet
sudo systemctl restart kubelet

# Configure networking prerequisites
cat <<'NET' | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
NET

sudo modprobe overlay
sudo modprobe br_netfilter

cat <<'NET2' | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
NET2

sudo sysctl --system

# Disable swap
sudo swapoff -a
sudo sed -i '/ swap / s/^/#/' /etc/fstab

# Set up kubelet config for containerd
cat <<'KUBELET' | sudo tee /etc/default/kubelet
KUBELET_EXTRA_ARGS="--cgroup-driver=systemd"
KUBELET_KUBEADM_ARGS=""
KUBELET_KUBECONFIG_ARGS="--bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf --kubeconfig=/etc/kubernetes/kubelet.conf"
KUBELET

echo "✅ Worker node preparation complete"
```

---

## Step 2: Get Join Token from Control Plane

On the **control plane node**, generate a join token:

```bash
# Generate new token (valid for 24 hours)
kubeadm token create --print-join-command

# Output will look like:
# kubeadm join 172.19.0.2:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>

# If you need the token again later
kubeadm token list
```

Save this output - you'll use it on each worker node.

---

## Step 3: Join Worker Nodes to Cluster

On each worker node, run the join command from Step 2:

```bash
sudo kubeadm join 172.19.0.2:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>

# You should see output like:
# [preflight] Running pre-flight checks
# [preflight] The system verification failed. Looks like we can't reliably detect the OS...
# [preflight] Please, use 'kubeadm init' on a machine running one of these distros: ...
# [kubelet-start] Starting the kubelet
# [kubeadm] Downloading configuration for newly-joining control-plane node
# [download-certs] Storing the apiserver-cert-hash in a Secret
# [kubeadm] Downloading kubelet, kube-proxy and kubeadm binaries
# [download-certs] Successful, secret "kubeadm-certs" is uploaded to the cluster
# [upload-config] Uploading the kubelet configuration to a ConfigMap
# [markcontrolplane] Marking a control plane node as control-plane by adding taints and labels to Node...
# [download-certs] Storing the apiserver-cert-hash in a Secret
# [kubeadm] Downloading kubelet, kube-proxy and kubeadm binaries
# [kubeadm] Making the API request for adding a new control-plane node
# [kubelet-start] Starting the kubelet
# This node has joined the cluster as a worker node.
```

---

## Step 4: Verify Nodes Are Joined

On the **control plane**, verify all nodes are ready:

```bash
# Check all nodes
kubectl get nodes -o wide

# Expected output after all nodes join:
# NAME                        STATUS   ROLES           VERSION
# dev-cluster-control-plane   Ready    control-plane   v1.31.0
# worker-node-1               Ready    <none>          v1.31.0
# worker-node-2               Ready    <none>          v1.31.0
# worker-node-3               Ready    <none>          v1.31.0

# Check node details
kubectl describe nodes

# Check pod distribution
kubectl get pods -A -o wide
```

---

## Step 5: Label Worker Nodes (Optional but Recommended)

Label nodes for workload distribution:

```bash
# Label as worker nodes
kubectl label node worker-node-1 node-role.kubernetes.io/worker=worker
kubectl label node worker-node-2 node-role.kubernetes.io/worker=worker
kubectl label node worker-node-3 node-role.kubernetes.io/worker=worker

# Add custom labels for workload affinity
kubectl label node worker-node-1 workload=general
kubectl label node worker-node-2 workload=data-intensive
kubectl label node worker-node-3 workload=high-memory

# Verify labels
kubectl get nodes --show-labels
```

---

## Step 6: Wait for Pod Redistribution

Once nodes are ready, pods will automatically redistribute:

```bash
# Watch pod distribution in real-time
kubectl get pods -A -o wide -w

# Or check specific namespaces
kubectl get pods -n data-platform -o wide
kubectl get pods -n monitoring -o wide
kubectl get pods -n ingress-nginx -o wide
```

### Expected Behavior
- Pods should start scheduling on new nodes
- Some existing pods will remain on control plane
- Load should balance across nodes

---

## Step 7: Verify High Availability Configuration

### Check Pod Anti-Affinity Rules

```bash
# These should be in place from earlier
kubectl get pods -n data-platform -o json | jq '.items[] | {name: .metadata.name, affinity: .spec.affinity}' | head -20

# Expected: Anti-affinity rules should spread pods across nodes
```

### Verify Pod Disruption Budgets

```bash
kubectl get pdb -A

# Should show:
# NAME          MIN AVAILABLE   MAX UNAVAILABLE   ALLOWED DISRUPTIONS
# datahub-pdb   1               N/A               1
# portal-pdb    1               N/A               1
```

### Check HPA Status

```bash
kubectl get hpa -A

# Should show metrics available now (metrics-server running)
kubectl top nodes
kubectl top pods -A
```

---

## Step 8: Test Pod Distribution

### Test Affinity Rules

```bash
# Check if pods from same deployment spread across nodes
kubectl get pods -n data-platform -l app=datahub-gms -o wide

# Expected: Each pod on different node (if >1 replica)
```

### Scale Up to Test Distribution

```bash
# Scale a deployment to verify pod spread
kubectl scale deployment datahub-gms -n data-platform --replicas=3

# Wait for pods to start
sleep 30

# Check distribution
kubectl get pods -n data-platform -l app=datahub-gms -o wide

# Expected: Pods spread across different nodes
```

---

## Step 9: Test Node Failure Scenario

### Simulate Node Failure (Non-Destructive)

```bash
# Drain a worker node (graceful - moves pods to other nodes)
kubectl drain worker-node-2 --ignore-daemonsets --delete-emptydir-data

# Monitor pod movement
kubectl get pods -A -o wide -w

# Observe:
# - Pods are evicted from drained node
# - New pods start on other nodes (respecting PDB)
# - Services remain available

# Uncordon node when ready to restore
kubectl uncordon worker-node-2

# Pods will re-balance (may redeploy if node was down long enough)
```

### Monitor Recovery

```bash
# Watch pod recovery
kubectl get pods -A -o wide -w

# Check node status returns to Ready
kubectl get nodes
```

---

## Step 10: Enable Service Replication

### Configure Database Replication (PostgreSQL)

Once multi-node is stable, set up database replication:

```bash
# Current PostgreSQL is single-pod
# Create StatefulSet with replication:

cat > /tmp/postgres-ha.yaml << 'POSTGRES'
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql-ha
  namespace: data-platform
spec:
  serviceName: postgresql
  replicas: 3
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
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
                  - postgresql
              topologyKey: kubernetes.io/hostname
      containers:
      - name: postgresql
        image: postgres:15
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: default
      resources:
        requests:
          storage: 50Gi
POSTGRES

kubectl apply -f /tmp/postgres-ha.yaml
```

---

## Step 11: Monitoring Multi-Node Setup

### Check Cluster Metrics

```bash
# CPU and Memory across nodes
kubectl top nodes

# Per-pod resource usage
kubectl top pods -A

# Node capacity planning
kubectl describe nodes | grep -A 5 "Allocated resources"
```

### Verify Load Distribution

```bash
# Pod count per node
kubectl get pods -A -o wide | awk '{print $NF}' | sort | uniq -c

# Should be roughly balanced across nodes
```

---

## Troubleshooting

### Nodes Not Joining

```bash
# Check kubelet logs on worker node
sudo journalctl -u kubelet -f

# Common issues:
# - Token expired (get new token from control plane)
# - Network connectivity (ping control plane IP)
# - Swap not disabled
# - Kernel modules not loaded
```

### Pod Stuck in Pending

```bash
# Check events
kubectl describe pod <pod-name> -n <namespace>

# Possible reasons:
# - No node available with matching affinity
# - Insufficient resources
# - PDB preventing scheduling
```

### Pods Not Redistribution After Node Join

```bash
# Most pods won't move until rescheduled
# Force reschedule:
kubectl rollout restart deployment/<name> -n <namespace>

# Or delete pod to force recreation:
kubectl delete pod <pod-name> -n <namespace>
```

---

## Success Criteria

- [x] 2-3 worker nodes provisioned
- [x] All nodes show "Ready" status
- [x] Pods distributed across nodes
- [x] Node drain/cordon works properly
- [x] Services remain available during node maintenance
- [x] HPA metrics available for auto-scaling
- [x] Pod anti-affinity working
- [x] Pod Disruption Budgets protecting services

---

## Phase 3 Completion Checklist

- [ ] Infrastructure provisioned (2-3 worker nodes)
- [ ] Worker nodes joined to cluster
- [ ] All nodes in Ready status
- [ ] Pod distribution verified (across nodes)
- [ ] Node failure test passed
- [ ] Service recovery verified
- [ ] Database replication configured
- [ ] HPA auto-scaling tested
- [ ] Load balanced across nodes
- [ ] Multi-node HA verified

---

## Next Steps (After Multi-Node Verified)

1. Begin Phase 4: Enhanced Monitoring
2. Configure advanced networking (service mesh)
3. Implement distributed storage
4. Set up automated backup for multi-node

**Phase 3 Status**: Multi-node setup guide complete, ready for infrastructure provisioning

