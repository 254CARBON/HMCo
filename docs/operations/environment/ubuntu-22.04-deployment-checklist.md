# Ubuntu 22.04 Kubernetes Bare Metal Deployment Checklist

**Date**: October 20, 2025  
**OS**: Ubuntu 22.04 LTS  
**Kubernetes Version**: 1.28+  
**Container Runtime**: containerd  

---

## Pre-Deployment Checklist

### Infrastructure Preparation

- [ ] **3-5 bare metal servers provisioned**
  - Ubuntu 22.04 LTS clean install
  - 4+ CPU cores per node
  - 8GB+ RAM per node (16GB recommended)
  - 100GB+ storage per node

- [ ] **Networking Configured**
  - [ ] Static IP addresses assigned to all nodes
  - [ ] Hostname set on each server
  - [ ] Network connectivity between all nodes verified
  - [ ] Internet access available (or proxy configured)
  - [ ] DNS resolution working

- [ ] **SSH Access Ready**
  - [ ] SSH key-based access configured
  - [ ] Root or sudo access available
  - [ ] SSH known_hosts updated

- [ ] **Time Synchronization Verified**
  ```bash
  # Ubuntu 22.04 uses systemd-timesyncd
  timedatectl status
  ```

- [ ] **Kind Cluster Backup**
  - [ ] Backup script tested: `./scripts/09-backup-from-kind.sh`
  - [ ] Backup location verified
  - [ ] Data integrity checked

---

## Phase-by-Phase Deployment

### ✅ Phase 1: Server Preparation (All Nodes)

**Duration**: 5-10 minutes per node

**On each node**, execute:
```bash
ssh root@<node-ip>

# Download and run preparation script
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/01-prepare-servers.sh)" _ k8s-node-1
```

**What gets configured**:
- Ubuntu 22.04 system updates
- Hostname resolution
- Swap disabled
- Kernel modules (overlay, br_netfilter)
- Sysctl settings for Kubernetes
- systemd-timesyncd (Ubuntu 22.04 default)
- UFW firewall with K8s rules
- System limits

**Verification**:
```bash
# On each node, verify:
uname -r                    # Should be 5.15+
sysctl net.ipv4.ip_forward  # Should be 1
swapoff -a
grep swap /etc/fstab        # Should show all commented
```

**Deployment Checklist**:
- [ ] Node 1 prepared
- [ ] Node 2 prepared
- [ ] Node 3 prepared
- [ ] (Node 4 prepared if 4-5 node cluster)
- [ ] (Node 5 prepared if 5 node cluster)
- [ ] All nodes verified

---

### ✅ Phase 2a: Container Runtime (All Nodes)

**Duration**: 3-5 minutes per node

**On each node**, execute:
```bash
ssh root@<node-ip>
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/02-install-container-runtime.sh)"
```

**What gets installed**:
- Docker repository (for containerd)
- containerd runtime
- systemd cgroup driver configuration

**Verification**:
```bash
# On each node:
systemctl status containerd
containerd --version
```

**Deployment Checklist**:
- [ ] Node 1 containerd installed
- [ ] Node 2 containerd installed
- [ ] Node 3 containerd installed
- [ ] (Node 4 if applicable)
- [ ] (Node 5 if applicable)
- [ ] All nodes verified

---

### ✅ Phase 2b: Kubernetes Components (All Nodes)

**Duration**: 5-10 minutes per node

**On each node**, execute:
```bash
ssh root@<node-ip>
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/03-install-kubernetes.sh)"
```

**What gets installed**:
- Kubernetes repository
- kubeadm 1.28+
- kubelet 1.28+
- kubectl 1.28+
- Version pinning (hold)

**Verification**:
```bash
# On each node:
kubeadm version
kubelet --version
kubectl version --client
```

**Deployment Checklist**:
- [ ] Node 1 K8s components installed
- [ ] Node 2 K8s components installed
- [ ] Node 3 K8s components installed
- [ ] (Node 4 if applicable)
- [ ] (Node 5 if applicable)
- [ ] All versions verified (1.28.x)

---

### ✅ Phase 2c: Control Plane Initialization

**Duration**: 5-10 minutes

**On control plane node only**, execute:
```bash
ssh root@<control-plane-ip>
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/04-init-control-plane.sh)"
```

**What happens**:
- kubeadm init with pod/service CIDR
- Flannel CNI installation
- kubectl configuration
- Join command generation

**Save the output - contains join command for workers!**

**Verification**:
```bash
# On control plane:
kubectl get nodes
kubectl get pods -n kube-system
kubectl get pods -n kube-flannel

# All should show Ready/Running status
```

**Deployment Checklist**:
- [ ] Control plane initialized
- [ ] Flannel CNI pods running
- [ ] Join command saved
- [ ] kubectl access verified

---

### ✅ Phase 2d: Join Worker Nodes

**Duration**: 3-5 minutes per worker node

**On each worker node**, execute:
```bash
ssh root@<worker-ip>

# Use the join command from Phase 2c
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/05-join-worker-nodes.sh)" _ "kubeadm join <control-plane-ip>:6443 --token ... --discovery-token-ca-cert-hash sha256:..."
```

**Verification** (on control plane):
```bash
kubectl get nodes -o wide

# All nodes should show Ready status
```

**Deployment Checklist**:
- [ ] Worker Node 1 joined
- [ ] Worker Node 2 joined
- [ ] Worker Node 3 joined (if 5 node) or (if 4 node)
- [ ] (Worker Node 4 joined if 5 node cluster)
- [ ] All nodes showing Ready status

---

### ✅ Phase 3: Storage Infrastructure

**Duration**: 5-10 minutes

**Prepare storage on each worker node**:
```bash
# On each worker node:
ssh root@<worker-ip>

sudo mkdir -p /mnt/openebs/local
sudo chmod 755 /mnt/openebs/local

# If using dedicated disk (e.g., /dev/sdb1):
# sudo mkfs.ext4 /dev/sdb1
# sudo mount /dev/sdb1 /mnt/openebs/local
# sudo bash -c 'echo "/dev/sdb1 /mnt/openebs/local ext4 defaults 0 2" >> /etc/fstab'
```

**Deploy storage** (on control plane):
```bash
ssh root@<control-plane-ip>
cd /home/m/tff/254CARBON/HMCo

./scripts/06-deploy-storage.sh "/home/m/tff/254CARBON/HMCo"
```

**Verification**:
```bash
kubectl get storageclass
kubectl get pv
kubectl get pods -n openebs
```

**Deployment Checklist**:
- [ ] Storage directories created on all workers
- [ ] OpenEBS operator deployed
- [ ] Storage classes available
- [ ] Local PVs created

---

### ✅ Phase 4: Platform Services Deployment

**Duration**: 10-15 minutes

**Deploy all services** (on control plane):
```bash
ssh root@<control-plane-ip>
cd /home/m/tff/254CARBON/HMCo

./scripts/07-deploy-platform.sh "/home/m/tff/254CARBON/HMCo"
```

**Services deployed in order**:
1. Namespaces
2. RBAC and networking
3. Data platform (Zookeeper, Kafka, MinIO, LakeFS, Iceberg)
4. Compute (Trino, Spark)
5. Supporting (Vault, Monitoring, Superset, DataHub, DolphinScheduler)
6. Cloudflare tunnel and ingress

**Monitor deployment**:
```bash
# Watch pods come up
kubectl get pods --all-namespaces -w

# Check for errors
kubectl get pods --all-namespaces | grep -v Running
```

**Deployment Checklist**:
- [ ] All namespaces created
- [ ] Core infrastructure pods running
- [ ] Data platform services running
- [ ] Compute services running
- [ ] Supporting services running
- [ ] No pods in CrashLoopBackOff or Pending

---

### ✅ Phase 5: Data Migration

**Duration**: 10-20 minutes

**Backup from Kind cluster** (on control machine):
```bash
cd /home/m/tff/254CARBON/HMCo

# Switch to Kind context
kubectl config use-context kind-dev-cluster

# Backup
./scripts/09-backup-from-kind.sh "./backups/kind-backup-$(date +%Y%m%d)"
```

**Restore to bare metal** (on control machine):
```bash
# Switch to bare metal context
kubectl config use-context <bare-metal-context>

cd /home/m/tff/254CARBON/HMCo

# Restore
./scripts/10-restore-to-bare-metal.sh "./backups/kind-backup-YYYYMMDD"
```

**Deployment Checklist**:
- [ ] Kind cluster backup created
- [ ] Backup verified
- [ ] Namespaces restored
- [ ] Configurations restored
- [ ] PVCs restored
- [ ] Data verified in applications

---

### ✅ Phase 6: Comprehensive Validation

**Duration**: 5-10 minutes

**Run validation script** (on control plane):
```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/08-validate-deployment.sh
```

**Manual verification**:
```bash
# Cluster health
kubectl get nodes -o wide
kubectl get pods --all-namespaces

# Storage
kubectl get pvc --all-namespaces
kubectl get pv

# Services
kubectl get svc --all-namespaces

# Cloudflare tunnel
kubectl get pods -n cloudflare-tunnel
kubectl logs -n cloudflare-tunnel <pod-name>
```

**Deployment Checklist**:
- [ ] All nodes Ready
- [ ] All pods Running/Completed
- [ ] Storage provisioned and bound
- [ ] Services accessible
- [ ] DNS resolution working
- [ ] Cloudflare tunnel connected
- [ ] No errors in validation output

---

### ✅ Phase 7: Production Cutover

**Timing**: Schedule after Phase 6 validation success

**Pre-cutover verification**:
```bash
# Final health check
./scripts/08-validate-deployment.sh

# Test critical services
kubectl run -it --rm test --image=busybox --restart=Never \
  -- nslookup kafka.data-platform.svc.cluster.local

# Verify Cloudflare access
curl -v https://254carbon.com
```

**Cutover steps**:
1. [ ] Kind cluster services stopped (optional, keep for reference)
2. [ ] DNS/Cloudflare updated (if applicable)
3. [ ] Monitoring activated
4. [ ] Alert channels configured
5. [ ] Team briefed on rollback procedures

**Production Checklist**:
- [ ] All services operational
- [ ] Data accessible and accurate
- [ ] SSO authentication working
- [ ] External access functional
- [ ] Monitoring active
- [ ] Alerts configured
- [ ] Team trained on operations

---

## Post-Deployment (Day 1-7)

- [ ] Monitor cluster for 24-48 hours
- [ ] Verify data pipelines are running
- [ ] Check monitoring dashboards
- [ ] Review and adjust resource limits
- [ ] Configure automated backups
- [ ] Document access procedures
- [ ] Train operations team

---

## Troubleshooting Quick Links

| Issue | Check |
|-------|-------|
| Nodes not Ready | `kubectl describe node <name>`, `journalctl -u kubelet` |
| Pods Pending | `kubectl describe pod`, check storage/resources |
| ImagePullBackOff | Check pull secrets, image availability |
| Service not accessible | Check NetworkPolicy, DNS resolution |
| Storage issues | `kubectl get pvc`, check `/mnt/openebs/local` |

---

## Rollback Plan (If Needed)

If critical issues arise during deployment:

```bash
# Immediate rollback (within 1 hour)
kubectl config use-context kind-dev-cluster
kubectl get pods --all-namespaces

# If Kind cluster is stable, traffic can be redirected back
# Check DNS/Cloudflare configuration
```

---

## Success Criteria

After complete deployment, verify:

✅ Cluster Status:
```bash
kubectl get nodes              # All Ready
kubectl get pods --all-namespaces  # All Running/Completed
```

✅ Storage Status:
```bash
kubectl get pvc --all-namespaces   # All Bound
kubectl get pv                     # All Available
```

✅ Service Status:
```bash
kubectl get svc --all-namespaces   # All have ClusterIP/external
curl https://254carbon.com         # Cloudflare tunnel working
```

✅ Application Status:
- Data accessible in applications
- Monitoring dashboards functional
- SSO authentication working
- Data pipelines executing

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| Phase 1: Prepare | 15-30 min | 15-30 min |
| Phase 2a: Runtime | 10-15 min | 25-45 min |
| Phase 2b: K8s | 15-30 min | 40-75 min |
| Phase 2c: Control | 5-10 min | 45-85 min |
| Phase 2d: Workers | 10-15 min | 55-100 min |
| Phase 3: Storage | 5-10 min | 60-110 min |
| Phase 4: Services | 10-15 min | 70-125 min |
| Phase 5: Data | 10-20 min | 80-145 min |
| Phase 6: Validate | 5-10 min | 85-155 min |
| **Total** | **1.5-2.5 hrs** | **85-155 min** |

---

## Ubuntu 22.04 Specific Notes

✅ **systemd-timesyncd** is default NTP client (replaces chrony)  
✅ **UFW** firewall available by default  
✅ **containerd** available from docker.com repos  
✅ **Kubernetes 1.28+** fully compatible  
✅ **Python 3.10+** included for any scripting needs  

---

**Status**: Ready for deployment  
**Last Updated**: October 20, 2025  
**Tested On**: Ubuntu 22.04 LTS
