# Kind to Bare Metal Kubernetes Migration Plan

**Migration Type**: Kind (Single-Node) → Multi-Node Bare Metal Kubernetes  
**Nodes**: 
- Control Plane + Worker: cpu1 (192.168.1.228) - Current machine
- Worker: 192.168.1.220 - Second node
**Date**: October 20, 2025  
**Current Cluster**: dev-cluster (Kind v1.31.0)  
**Target Cluster**: Production bare-metal (kubeadm v1.34.1)

---

## Migration Strategy

### Approach: Fresh Cluster + Workload Migration

**Why Not In-Place**: 
- Kind uses containerized nodes (not suitable for production)
- Different networking stack (kind vs real nodes)
- Different storage (local-path vs real persistent storage)
- Fresh cluster ensures clean state

**Migration Path**:
1. **Export** all working configurations from Kind
2. **Build** new bare-metal cluster with 2 nodes
3. **Deploy** Cloudflare infrastructure first
4. **Migrate** workloads systematically
5. **Verify** and cutover DNS
6. **Decommission** Kind cluster

---

## Phase 1: Pre-Migration Preparation

### 1.1 Export Current Configurations

```bash
# Create export directory
mkdir -p cluster-export/{cloudflare,ingress,apps,secrets,configs}

# Export Cloudflare components
kubectl get deployment,service,configmap,secret -n cloudflare-tunnel -o yaml > cluster-export/cloudflare/cloudflare-tunnel.yaml

# Export cert-manager (Helm values)
helm get values cert-manager -n cert-manager > cluster-export/cloudflare/cert-manager-values.yaml

# Export all ingress
kubectl get ingress -A -o yaml > cluster-export/ingress/all-ingress.yaml

# Export working deployments
kubectl get deployment -n data-platform -o yaml > cluster-export/apps/data-platform-deployments.yaml
kubectl get statefulset -n data-platform -o yaml > cluster-export/apps/data-platform-statefulsets.yaml

# Export services
kubectl get svc -A -o yaml > cluster-export/apps/all-services.yaml

# Export configmaps (non-sensitive)
kubectl get configmap -n data-platform -o yaml > cluster-export/configs/data-platform-configs.yaml

# List what's working
kubectl get pods -A | grep "1/1.*Running\|2/2.*Running" > cluster-export/working-pods.txt
```

### 1.2 Document Current State

```bash
# Save cluster info
kubectl cluster-info > cluster-export/cluster-info.txt
kubectl get nodes -o wide > cluster-export/nodes.txt
kubectl get pods -A -o wide > cluster-export/all-pods.txt
kubectl get pvc -A > cluster-export/pvcs.txt
kubectl get sc > cluster-export/storage-classes.txt

# Save network policies
kubectl get networkpolicy -A -o yaml > cluster-export/network-policies.yaml

# Save RBAC
kubectl get clusterrole,clusterrolebinding -o yaml > cluster-export/rbac.yaml
```

### 1.3 Backup Critical Data

**Services with persistent data**:
- PostgreSQL (datahub, superset databases)
- MinIO (object storage)
- Harbor (registry data)
- Grafana dashboards
- Vault data

```bash
# PostgreSQL backups
kubectl exec -n data-platform postgres-shared-0 -- pg_dumpall -U postgres > cluster-export/postgres-backup.sql

# MinIO data (if critical)
kubectl port-forward -n data-platform minio-0 9000:9000
mc alias set minio-local http://localhost:9000 <access-key> <secret-key>
mc mirror minio-local cluster-export/minio-backup/

# Harbor database
kubectl exec -n registry harbor-database-0 -- pg_dump -U postgres registry > cluster-export/harbor-db-backup.sql
```

---

## Phase 2: Prepare Second Node

### 2.1 SSH Setup

```bash
# Generate SSH key on cpu1 (if not exists)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy to second node
ssh-copy-id m@192.168.1.220

# Test connection
ssh m@192.168.1.220 "hostname && uname -a"
```

### 2.2 Install Prerequisites on Both Nodes

**On BOTH cpu1 (192.168.1.228) AND 192.168.1.220**:

```bash
# Disable swap
sudo swapoff -a
sudo sed -i '/ swap / s/^/#/' /etc/fstab

# Load kernel modules
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

# Sysctl params
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo sysctl --system

# Install containerd
sudo apt-get update
sudo apt-get install -y containerd

# Configure containerd
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl restart containerd
sudo systemctl enable containerd

# Install kubeadm, kubelet, kubectl (if not present)
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gpg
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.34/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.34/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

---

## Phase 3: Initialize New Cluster

### 3.1 Stop Kind Cluster (Don't Delete Yet)

```bash
# Stop Kind but keep data for reference
kind export kubeconfig --name dev-cluster > cluster-export/dev-cluster-kubeconfig.yaml
kind get kubeconfig --name dev-cluster > ~/.kube/kind-dev-cluster.config

# Stop Kind (optional - can run in parallel during setup)
# kind delete cluster --name dev-cluster  # DON'T RUN YET
```

### 3.2 Initialize Control Plane on cpu1 (192.168.1.228)

```bash
# Initialize cluster with specific pod network
sudo kubeadm init \
  --pod-network-cidr=10.244.0.0/16 \
  --service-cidr=10.96.0.0/12 \
  --apiserver-advertise-address=192.168.1.228 \
  --control-plane-endpoint=192.168.1.228:6443 \
  --upload-certs

# Save the output! You'll need the join command for the worker node

# Setup kubectl for non-root user
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Verify control plane
kubectl get nodes
kubectl get pods -n kube-system
```

### 3.3 Install CNI (Flannel or Calico)

**Option A: Flannel** (Simpler)
```bash
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

**Option B: Calico** (More features, better for production)
```bash
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/custom-resources.yaml
```

Wait for all kube-system pods to be Running.

### 3.4 Join Worker Node (192.168.1.220)

**On cpu1**: Get the join command
```bash
# If you lost the join command from kubeadm init:
kubeadm token create --print-join-command
```

**On 192.168.1.220**: Run the join command
```bash
sudo kubeadm join 192.168.1.228:6443 \
  --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash>
```

**On cpu1**: Verify node joined
```bash
kubectl get nodes
# Should show 2 nodes: cpu1 (control-plane+worker) and worker node

# Label worker node
kubectl label node <worker-hostname> node-role.kubernetes.io/worker=worker
```

---

## Phase 4: Deploy Core Infrastructure

### 4.1 Storage Class (Critical First)

```bash
# Deploy Longhorn for distributed storage (recommended)
kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.7.2/deploy/longhorn.yaml

# Or use local-path-provisioner (simpler but not distributed)
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.30/deploy/local-path-storage.yaml

# Set as default
kubectl patch storageclass longhorn -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

### 4.2 NGINX Ingress Controller

```bash
# Deploy NGINX Ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.1/deploy/static/provider/baremetal/deploy.yaml

# Verify
kubectl get pods -n ingress-nginx

# Label namespace (learned from previous issue!)
kubectl label namespace ingress-nginx name=ingress-nginx

# Get NodePort
kubectl get svc -n ingress-nginx ingress-nginx-controller
# Note the NodePort for HTTP (e.g., 30080) and HTTPS (e.g., 30443)
```

### 4.3 cert-manager (via Helm)

```bash
# Add Helm repo
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install cert-manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Create ClusterIssuers
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ops@254carbon.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
    - http01:
        ingress:
          class: nginx
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: selfsigned
spec:
  selfSigned: {}
EOF
```

---

## Phase 5: Deploy Cloudflare Infrastructure

### 5.1 Cloudflare Tunnel

```bash
# Create namespace
kubectl create namespace cloudflare-tunnel
kubectl label namespace cloudflare-tunnel app.kubernetes.io/name=cloudflare-tunnel

# Create tunnel credentials secret
kubectl create secret generic cloudflare-tunnel-credentials \
  -n cloudflare-tunnel \
  --from-literal=credentials.json='<CREDENTIALS_JSON>'

# Deploy cloudflared (using existing config)
kubectl apply -f k8s/cloudflare/cloudflared-deployment.yaml

# Verify
kubectl get pods -n cloudflare-tunnel
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel
```

**Note**: Tunnel credentials are already configured (Tunnel ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5)

### 5.2 Verify Cloudflare Connectivity

```bash
# Check tunnel connections
kubectl logs -n cloudflare-tunnel -f | grep "Registered tunnel connection"

# DNS should already be configured (14 records pointing to tunnel)
# No changes needed - tunnel will reconnect automatically
```

---

## Phase 6: Migrate Workloads

### 6.1 Deploy Namespaces and Network Policies

```bash
# Create namespaces
kubectl create namespace data-platform
kubectl create namespace monitoring
kubectl create namespace vault-prod
kubectl create namespace registry

# Deploy network policies
kubectl apply -f k8s/networking/

# Fix ingress-nginx namespace label (critical!)
kubectl label namespace ingress-nginx name=ingress-nginx
```

### 6.2 Deploy Foundational Services

**Order matters**:

```bash
# 1. PostgreSQL
kubectl apply -f k8s/shared/postgres/postgres-shared.yaml

# 2. Zookeeper
kubectl apply -f k8s/shared/zookeeper/zookeeper.yaml

# 3. MinIO
kubectl apply -f k8s/shared/minio/minio.yaml

# 4. Redis
kubectl apply -f k8s/shared/redis/redis.yaml

# Wait for all to be Running before proceeding
kubectl get pods -n data-platform -w
```

### 6.3 Deploy Portal

```bash
# Build and load portal images (on each node)
cd /home/m/tff/254CARBON/HMCo/portal
docker build -t 254carbon-portal:latest .

cd /home/m/tff/254CARBON/HMCo/services/portal-services
docker build -t 254carbon/portal-services:latest .

# For bare-metal, push to Harbor registry instead of loading locally
# Or use a shared registry both nodes can access

# Deploy portal
kubectl apply -f k8s/portal/portal-services.yaml
kubectl apply -f k8s/ingress/portal-deployment.yaml

# Apply ingress
kubectl apply -f k8s/ingress/
```

### 6.4 Deploy Other Services

```bash
# DataHub
kubectl apply -f k8s/datahub/datahub.yaml

# Harbor
kubectl apply -f k8s/registry/

# Visualization
kubectl apply -f k8s/visualization/superset.yaml

# Compute
kubectl apply -f k8s/compute/trino/trino.yaml
kubectl apply -f k8s/compute/doris/

# Monitoring
kubectl apply -f k8s/monitoring/grafana.yaml

# Vault
kubectl apply -f k8s/vault/vault-production.yaml
```

---

## Phase 7: Storage Migration

### 7.1 PostgreSQL Data

```bash
# Restore PostgreSQL backup on new cluster
kubectl cp cluster-export/postgres-backup.sql data-platform/postgres-shared-0:/tmp/
kubectl exec -n data-platform postgres-shared-0 -- psql -U postgres -f /tmp/postgres-backup.sql
```

### 7.2 MinIO Data (If Needed)

```bash
# Use mc (MinIO client) to sync data
mc mirror cluster-export/minio-backup/ new-minio/
```

---

## Phase 8: Testing & Validation

### 8.1 Verify Core Infrastructure

```bash
# Nodes healthy
kubectl get nodes

# All kube-system pods running
kubectl get pods -n kube-system

# CNI operational
kubectl get pods -n kube-flannel  # or -n calico-system

# Storage provisioner
kubectl get sc
kubectl get pv
```

### 8.2 Test Cloudflare Connectivity

```bash
# Tunnel connected
kubectl get pods -n cloudflare-tunnel
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel | grep "Registered"

# Services accessible
curl -I https://portal.254carbon.com
```

### 8.3 Verify Services

```bash
# Check all pods
kubectl get pods -A

# Test portal
https://portal.254carbon.com

# Test other services
for svc in harbor minio grafana; do
  curl -I https://$svc.254carbon.com
done
```

---

## Phase 9: Cutover & Cleanup

### 9.1 Final Verification

- [ ] All critical services running on new cluster
- [ ] Cloudflare Tunnel connected (check dashboard)
- [ ] DNS resolving correctly (no changes needed)
- [ ] Portal accessible and functional
- [ ] SSL certificates issued
- [ ] All ingress working

### 9.2 Switch Context

```bash
# Save kind config
cp ~/.kube/config ~/.kube/config.kind.backup

# New cluster is now default
kubectl config use-context kubernetes-admin@kubernetes

# Verify
kubectl get nodes
# Should show 2 nodes (cpu1 and 192.168.1.220)
```

### 9.3 Decommission Kind

```bash
# Only after everything works on new cluster!
kind delete cluster --name dev-cluster

# Clean up Docker
docker system prune -a --volumes
```

---

## Key Differences: Kind vs Bare Metal

| Aspect | Kind (Current) | Bare Metal (Target) |
|--------|----------------|---------------------|
| Nodes | 1 (containerized) | 2 (real hardware) |
| Storage | local-path (single node) | Longhorn (distributed) |
| Networking | kind | Flannel/Calico |
| HA | No (single node) | Yes (2+ nodes) |
| PVC Binding | Limited | Full support |
| Production Ready | No | Yes |
| Resource Limits | Container limits | Real hardware |
| IP Addresses | Bridge network | Real network (192.168.1.x) |

---

## Migration Checklist

### Pre-Migration
- [ ] SSH access to 192.168.1.220
- [ ] Both nodes have prerequisites installed
- [ ] Exported all Kind configurations
- [ ] Backed up critical data
- [ ] Documented current state

### During Migration
- [ ] Control plane initialized on cpu1
- [ ] Worker joined from 192.168.1.220
- [ ] CNI deployed and operational
- [ ] Storage class configured
- [ ] NGINX Ingress deployed
- [ ] cert-manager installed via Helm
- [ ] Cloudflare Tunnel deployed
- [ ] Network policies applied
- [ ] Workloads migrated

### Post-Migration
- [ ] All services running
- [ ] Cloudflare Tunnel connected
- [ ] Portal functional
- [ ] SSL certificates issued
- [ ] Data restored
- [ ] Kind cluster deleted

---

## Critical Configuration Items

### Must Preserve From Kind

1. **Cloudflare Tunnel Credentials**
   - Tunnel ID: 291bc289-e3c3-4446-a9ad-8e327660ecd5
   - Keep same credentials (no DNS changes needed)

2. **Network Policy Labels**
   - `namespace/ingress-nginx`: `name=ingress-nginx`
   - This was critical for portal to work!

3. **Ingress Configurations**
   - All annotations for timeouts
   - Cloudflare Access references
   - Route configurations (/api, /api/services, /)

4. **Portal Images**
   - 254carbon-portal:latest
   - 254carbon/portal-services:latest
   - Need to be available on new cluster (push to Harbor or load on each node)

---

## Estimated Timeline

- **Phase 1** (Export): 30 minutes
- **Phase 2** (Node prep): 45 minutes (both nodes)
- **Phase 3** (Initialize cluster): 30 minutes
- **Phase 4** (Core infrastructure): 45 minutes
- **Phase 5** (Cloudflare): 15 minutes
- **Phase 6** (Workloads): 60 minutes
- **Phase 7** (Data migration): 30 minutes
- **Phase 8** (Testing): 30 minutes
- **Phase 9** (Cutover): 15 minutes

**Total**: ~5-6 hours for complete migration

---

## Rollback Plan

If issues arise:

1. **Keep Kind running** during new cluster setup
2. **Don't delete Kind** until new cluster verified
3. **DNS doesn't change** (tunnel reconnects to new cluster)
4. **Cloudflare tunnel** can be paused/resumed in dashboard
5. **Switch kubectl context** back to Kind if needed

---

## Next Immediate Steps

### Step 1: Decide Migration Approach

**Option A: Full migration now** (5-6 hours, production-ready result)
**Option B: Parallel setup** (Keep Kind, build new cluster alongside)
**Option C: Incremental** (Move services one by one)

### Step 2: Prepare Second Node

```bash
# Test SSH access
ssh m@192.168.1.220 "hostname"

# If fails, set up SSH keys
ssh-keygen -t ed25519
ssh-copy-id m@192.168.1.220
```

### Step 3: Begin Migration

Once you confirm approach, I'll execute the migration systematically.

---

**Would you like me to proceed with the full migration? Or would you prefer a different approach?**

I can execute all phases automatically, or we can do it step-by-step with verification at each stage.
