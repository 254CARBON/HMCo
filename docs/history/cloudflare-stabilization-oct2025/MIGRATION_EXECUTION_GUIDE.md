# Kubernetes Migration - Execution Guide

**Status**: Ready to Execute  
**Migration**: Kind → 2-Node Bare Metal Cluster  
**Nodes**: cpu1 (192.168.1.228) + 192.168.1.220  
**Date**: October 20, 2025

---

## ✅ Phase 1: Pre-Migration - COMPLETE

### Configurations Exported
- ✅ Cloudflare Tunnel config
- ✅ cert-manager Helm values
- ✅ All ingress configurations
- ✅ Network policies
- ✅ ClusterIssuers
- ✅ Working pods list (36 healthy pods documented)

### Data Backed Up
- ✅ PostgreSQL: 106KB backup saved
- ✅ Harbor database: Backed up
- ✅ All exports in `/home/m/tff/254CARBON/HMCo/cluster-export/`

---

## ⚠️ Phase 2: SSH Setup - REQUIRES MANUAL ACTION

### Current Situation
- ✅ SSH key generated on cpu1: `/home/m/.ssh/id_ed25519`
- ❌ Cannot auto-copy to 192.168.1.220 (needs password or existing auth)

### **ACTION REQUIRED**: Choose One Option

#### Option A: Manual SSH Key Copy
```bash
# Run this command and enter the password for m@192.168.1.220:
ssh-copy-id m@192.168.1.220

# Then test:
ssh m@192.168.1.220 "hostname"
```

#### Option B: Manually Add Key
```bash
# 1. Copy this public key:
cat ~/.ssh/id_ed25519.pub
# Output: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG1g8qAoK3UbJt/zk3FxBxyR+Q4x9Q28OA6JZ6Q/RCp1 m@cpu1

# 2. On 192.168.1.220, run:
mkdir -p ~/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG1g8qAoK3UbJt/zk3FxBxyR+Q4x9Q28OA6JZ6Q/RCp1 m@cpu1" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# 3. Test from cpu1:
ssh m@192.168.1.220 "hostname"
```

#### Option C: Use Existing Access
If you already have access to 192.168.1.220, just confirm:
```bash
ssh m@192.168.1.220 "hostname && ip addr show | grep 192.168.1.220"
```

---

## 🚀 Phase 3: Automated Migration (After SSH Setup)

Once SSH is working, I'll execute the following automatically:

### 3.1 Prepare Both Nodes
```bash
# Will run on BOTH nodes via SSH:
- Disable swap
- Load kernel modules (overlay, br_netfilter)
- Configure sysctl for Kubernetes
- Install/configure containerd
- Install kubeadm, kubelet, kubectl (v1.34.1)
```

### 3.2 Initialize New Cluster
```bash
# On cpu1 (192.168.1.228):
sudo kubeadm init \
  --pod-network-cidr=10.244.0.0/16 \
  --apiserver-advertise-address=192.168.1.228 \
  --control-plane-endpoint=192.168.1.228:6443
```

### 3.3 Join Worker Node
```bash
# On 192.168.1.220:
sudo kubeadm join 192.168.1.228:6443 --token <token> --discovery-token-ca-cert-hash <hash>
```

### 3.4 Deploy Infrastructure
```bash
# In order:
1. Flannel CNI
2. local-path-provisioner (storage)
3. NGINX Ingress
4. cert-manager (Helm)
5. Cloudflare Tunnel
6. Network policies
```

### 3.5 Migrate Workloads
```bash
# Deploy in dependency order:
1. PostgreSQL, Zookeeper, MinIO, Redis
2. Portal + Portal Services
3. Harbor
4. DataHub
5. Other services
```

### 3.6 Restore Data
```bash
# Restore PostgreSQL backups
# Verify data integrity
```

### 3.7 Verify & Cutover
```bash
# Test all services
# Verify Cloudflare Tunnel connected
# Switch kubectl context
# Delete Kind cluster
```

---

## 📊 Expected Results

### After Migration
```
Nodes: 2 (cpu1 as control-plane+worker, 192.168.1.220 as worker)
Pods: Distributed across both nodes
Storage: PVCs can bind properly (Grafana, Vault will work!)
HA: True high availability
Production Ready: Yes
```

### Services That Will Improve
- ✅ Grafana: Will run (currently pending PVC)
- ✅ Vault: Will run (currently pending PVC)
- ✅ Doris: Will run (currently pending PVC)
- ✅ Better resource distribution
- ✅ Actual HA for critical services

---

## 🎯 Next Steps

### Immediate (Required)
**Set up SSH access to 192.168.1.220** using one of the options above.

Once SSH is working, tell me and I'll:
1. Automatically execute all remaining phases
2. Migrate the entire cluster
3. Verify everything is working
4. Complete the cutover

---

## ⏱️ Timeline (After SSH Setup)

- Node preparation: ~30 min
- Cluster initialization: ~15 min
- Infrastructure deployment: ~45 min
- Workload migration: ~60 min
- Data restoration: ~15 min
- Verification: ~30 min

**Total**: ~3 hours automated execution

---

## 🛡️ Safety

- ✅ Kind cluster keeps running during migration
- ✅ All data backed up
- ✅ Cloudflare tunnel can reconnect (no DNS changes)
- ✅ Can rollback by switching kubectl context
- ✅ No destructive operations until verified

---

**Please set up SSH access to 192.168.1.220 using one of the options above, then let me know and I'll execute the complete automated migration!**
