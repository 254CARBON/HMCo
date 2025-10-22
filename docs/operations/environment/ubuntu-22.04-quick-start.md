# Ubuntu 22.04 Bare Metal Kubernetes - Quick Start (Copy & Paste)

**Ready to deploy?** Copy these commands exactly as shown.

---

## Prerequisites (Check These First)

```bash
# On each bare metal server:
cat /etc/os-release | grep PRETTY_NAME    # Should show Ubuntu 22.04
uname -r                                  # Should be 5.15+
ip addr show                              # Verify IP addresses
ping 8.8.8.8                              # Verify internet access
```

---

## Phase 0: From Your Control Machine

### Backup Kind Cluster

```bash
cd /home/m/tff/254CARBON/HMCo

# Switch to Kind
kubectl config use-context kind-dev-cluster

# Backup
./scripts/09-backup-from-kind.sh "./backups/kind-ubuntu22-$(date +%Y%m%d-%H%M%S)"

# Verify backup
ls -lah ./backups/kind-ubuntu22-*/
```

---

## Phase 1: On Each Bare Metal Node

### Node Setup (SSH to each node)

```bash
# 1. SSH to first node
ssh root@192.168.1.100

# 2. Run this command on that node:
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/01-prepare-servers.sh)" _ "k8s-node1"

# 3. Wait for completion, should see:
# "Server preparation complete!"

# 4. Exit and move to next node
exit

# 5. Repeat for node 2:
ssh root@192.168.1.101
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/01-prepare-servers.sh)" _ "k8s-node2"
exit

# 6. And node 3 (if 3-node cluster):
ssh root@192.168.1.102
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/01-prepare-servers.sh)" _ "k8s-node3"
exit
```

**Each node takes ~5-10 minutes**

---

## Phase 2a: Container Runtime (Each Node)

```bash
# SSH to each node and run:

# Node 1
ssh root@192.168.1.100
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/02-install-container-runtime.sh)"
exit

# Node 2
ssh root@192.168.1.101
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/02-install-container-runtime.sh)"
exit

# Node 3
ssh root@192.168.1.102
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/02-install-container-runtime.sh)"
exit
```

**Each node takes ~3-5 minutes**

---

## Phase 2b: Kubernetes Installation (Each Node)

```bash
# SSH to each node and run:

# Node 1
ssh root@192.168.1.100
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/03-install-kubernetes.sh)"
exit

# Node 2
ssh root@192.168.1.101
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/03-install-kubernetes.sh)"
exit

# Node 3
ssh root@192.168.1.102
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/03-install-kubernetes.sh)"
exit
```

**Each node takes ~5-10 minutes**

---

## Phase 2c: Initialize Control Plane

```bash
# SSH to control plane ONLY (node 1):
ssh root@192.168.1.100

# Run init script:
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/04-init-control-plane.sh)"

# **IMPORTANT**: Save the "kubeadm join" command output!
# It looks like:
# kubeadm join 192.168.1.100:6443 --token abc123 --discovery-token-ca-cert-hash sha256:...

# Keep this terminal open to copy the join command

# Don't exit yet!
```

**Takes ~5-10 minutes**

---

## Phase 2d: Join Worker Nodes

### Get Join Command

From the control plane terminal (still open from Phase 2c), you'll see:
```
kubeadm join 192.168.1.100:6443 --token abc123def456 --discovery-token-ca-cert-hash sha256:abc123def456...
```

Copy the entire join command.

### Execute Join on Each Worker

```bash
# SSH to worker node 1:
ssh root@192.168.1.101

# Paste and run the join command:
kubeadm join 192.168.1.100:6443 --token abc123def456 --discovery-token-ca-cert-hash sha256:abc123def456...

# Wait for completion
exit

# SSH to worker node 2:
ssh root@192.168.1.102

# Paste same join command:
kubeadm join 192.168.1.100:6443 --token abc123def456 --discovery-token-ca-cert-hash sha256:abc123def456...

# Wait for completion
exit
```

**Each worker takes ~3-5 minutes**

### Verify Cluster (on control plane)

```bash
ssh root@192.168.1.100

# Check all nodes are Ready:
kubectl get nodes

# Output should show:
# NAME       STATUS   ROLES           AGE     VERSION
# k8s-node1  Ready    control-plane   5m      v1.28.x
# k8s-node2  Ready    <none>          2m      v1.28.x
# k8s-node3  Ready    <none>          1m      v1.28.x
```

**All nodes must show "Ready" before proceeding!**

---

## Phase 3: Storage Setup

### Prepare Storage Directories (Each Worker Node)

```bash
# SSH to each worker node:

# Node 2
ssh root@192.168.1.101
sudo mkdir -p /mnt/openebs/local
sudo chmod 755 /mnt/openebs/local
exit

# Node 3
ssh root@192.168.1.102
sudo mkdir -p /mnt/openebs/local
sudo chmod 755 /mnt/openebs/local
exit
```

### Deploy Storage (Control Plane)

```bash
ssh root@192.168.1.100

cd /home/m/tff/254CARBON/HMCo

./scripts/06-deploy-storage.sh "/home/m/tff/254CARBON/HMCo"

# Verify:
kubectl get storageclass
kubectl get pods -n openebs
```

**Takes ~5-10 minutes**

---

## Phase 4: Deploy Services

```bash
ssh root@192.168.1.100

cd /home/m/tff/254CARBON/HMCo

./scripts/07-deploy-platform.sh "/home/m/tff/254CARBON/HMCo"

# Monitor progress:
kubectl get pods --all-namespaces -w

# Press Ctrl+C to stop watching
```

**Takes ~10-15 minutes**

---

## Phase 5: Restore Data

### From Control Machine

```bash
cd /home/m/tff/254CARBON/HMCo

# Switch to bare metal cluster
kubectl config use-context <your-bare-metal-context>

# Or if not configured, SSH to control plane:
ssh root@192.168.1.100

cd /home/m/tff/254CARBON/HMCo

# Run restore:
./scripts/10-restore-to-bare-metal.sh "./backups/kind-ubuntu22-YYYYMMDD-HHMMSS"

# Replace YYYYMMDD-HHMMSS with your actual backup directory
```

**Takes ~10-20 minutes**

---

## Phase 6: Validate Everything

```bash
ssh root@192.168.1.100

cd /home/m/tff/254CARBON/HMCo

./scripts/08-validate-deployment.sh

# Review output for any errors
```

**Takes ~5-10 minutes**

### Manual Verification

```bash
ssh root@192.168.1.100

# Check all pods
kubectl get pods --all-namespaces

# Should show all pods Running/Completed
# No pods should show Pending or CrashLoopBackOff

# Check storage
kubectl get pvc --all-namespaces

# All should show Bound

# Check services
kubectl get svc --all-namespaces

# Should show many services with IPs
```

---

## Success Indicators

After Phase 6, you should see:

```bash
# Check 1: Nodes Ready
$ kubectl get nodes
NAME       STATUS   ROLES           AGE    VERSION
k8s-node1  Ready    control-plane   20m    v1.28.x
k8s-node2  Ready    <none>          15m    v1.28.x
k8s-node3  Ready    <none>          14m    v1.28.x

# Check 2: Pods Running
$ kubectl get pods --all-namespaces | tail -20
# Should show Running/Completed, no Pending

# Check 3: Storage Available
$ kubectl get pvc --all-namespaces
# All should show Bound

# Check 4: Services Running
$ kubectl get svc --all-namespaces | wc -l
# Should show 20+ services
```

---

## If Something Goes Wrong

### Check Logs

```bash
ssh root@192.168.1.100

# Pod logs:
kubectl logs -n data-platform <pod-name>

# Pod events:
kubectl describe pod -n data-platform <pod-name>

# Node issues:
kubectl describe node <node-name>

# Cluster events:
kubectl get events --all-namespaces --sort-by='.lastTimestamp'
```

### Node Issues

```bash
# SSH to problematic node:
ssh root@<node-ip>

# Check kubelet:
systemctl status kubelet

# View kubelet logs:
journalctl -u kubelet -n 50

# Restart kubelet:
systemctl restart kubelet
```

### Reset a Node (Last Resort)

```bash
# On the node to reset:
ssh root@<node-ip>

# Reset kubelet
kubeadm reset -f

# Clean up:
rm -rf /etc/kubernetes/
rm -rf /var/lib/kubelet/

# Rejoin cluster with kubeadm join command
```

---

## Timeline

| Phase | Duration |
|-------|----------|
| Phase 1: Prep (3 nodes) | 15-30 min |
| Phase 2a: Runtime (3 nodes) | 10-15 min |
| Phase 2b: K8s (3 nodes) | 15-30 min |
| Phase 2c: Init Control | 5-10 min |
| Phase 2d: Join Workers | 10-15 min |
| Phase 3: Storage | 5-10 min |
| Phase 4: Services | 10-15 min |
| Phase 5: Data | 10-20 min |
| Phase 6: Validate | 5-10 min |
| **TOTAL** | **1.5-2.5 hours** |

---

## What's Next?

After successful validation:

1. **Test Services**
   ```bash
   # Check Cloudflare tunnel
   curl -v https://254carbon.com
   
   # Check specific service
   kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080
   # Visit http://localhost:8080/ui/
   ```

2. **Configure Monitoring**
   ```bash
   ssh root@192.168.1.100
   kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
   # Visit http://localhost:9090
   ```

3. **Monitor for 24 Hours**
   - Watch dashboards
   - Check logs
   - Verify data integrity

4. **Decommission Kind Cluster** (when confident)
   ```bash
   kind delete cluster --name dev-cluster
   ```

---

## Emergency Rollback

If critical issues arise:

```bash
# Switch back to Kind cluster
kubectl config use-context kind-dev-cluster

# Verify Kind is still working
kubectl get pods --all-namespaces

# If Kind is stable, all traffic can be redirected back
```

---

## Copy-Paste Checklist

- [ ] Ubuntu 22.04 verified on all nodes
- [ ] Phase 1: Prep scripts run on all nodes
- [ ] Phase 2a: Runtime scripts run on all nodes
- [ ] Phase 2b: K8s scripts run on all nodes
- [ ] Phase 2c: Control plane initialized
- [ ] Phase 2d: Join command executed on workers
- [ ] Phase 3: Storage directories created and deployed
- [ ] Phase 4: Platform services deployed
- [ ] Phase 5: Data backed up and restored
- [ ] Phase 6: Validation successful
- [ ] All pods running, all PVCs bound
- [ ] Services accessible
- [ ] Data verified

---

**Total Time**: Approximately 2-2.5 hours for complete deployment

**For detailed information**: See `UBUNTU_22_04_DEPLOYMENT_CHECKLIST.md`

**For troubleshooting**: See `docs/operations/deployment/full-migration-runbook.md`
