# Bare Metal Kubernetes Migration - Quick Reference

**TL;DR**: Step-by-step commands to migrate from Kind to bare metal production cluster

---

## Pre-Migration (On Control Machine)

### 1. Backup Kind Cluster Data

```bash
cd /home/m/tff/254CARBON/HMCo

# Switch to Kind cluster
kubectl config use-context kind-dev-cluster

# Create backup
./scripts/09-backup-from-kind.sh "./backups/kind-cluster-$(date +%Y%m%d)"

# Verify backup
ls -la ./backups/
```

---

## Bare Metal Deployment

### 2. Prepare Each Server (Run as root on each node)

```bash
# SSH to each server
ssh root@<node-ip>

# Run preparation script
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/01-prepare-servers.sh)" _ k8s-node-$(hostname -I | awk '{print $1}' | tr '.' '-')
```

Expected time: 5-10 minutes per server

### 3. Install Container Runtime (All Nodes)

```bash
ssh root@<node-ip>

bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/02-install-container-runtime.sh)"
```

Expected time: 3-5 minutes per server

### 4. Install Kubernetes (All Nodes)

```bash
ssh root@<node-ip>

bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/03-install-kubernetes.sh)"
```

Expected time: 5-10 minutes per server

### 5. Initialize Control Plane (Control Plane Node Only)

```bash
ssh root@<control-plane-ip>

bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/04-init-control-plane.sh)"
```

**Important**: Save the `kubeadm join` command printed at the end

Expected time: 5-10 minutes

### 6. Join Worker Nodes (Each Worker Node)

```bash
ssh root@<worker-node-ip>

# Use the join command from step 5
bash -c "$(curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/05-join-worker-nodes.sh)" _ "kubeadm join <control-plane-ip>:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>"
```

Expected time: 3-5 minutes per worker node

### 7. Verify Cluster (Control Plane)

```bash
ssh root@<control-plane-ip>

# Check all nodes are ready
kubectl get nodes

# Should show: All nodes "Ready" status
```

### 8. Setup Local Storage (All Worker Nodes)

```bash
# On each worker node, create storage directory
ssh root@<worker-node-ip>

sudo mkdir -p /mnt/openebs/local
sudo chmod 755 /mnt/openebs/local

# If using dedicated disk:
# sudo mkfs.ext4 /dev/sdb1
# sudo mount /dev/sdb1 /mnt/openebs/local
```

### 9. Deploy Storage Infrastructure (Control Plane)

```bash
ssh root@<control-plane-ip>

cd /home/m/tff/254CARBON/HMCo
./scripts/06-deploy-storage.sh "/home/m/tff/254CARBON/HMCo"
```

Expected time: 5-10 minutes

### 10. Deploy Platform Services (Control Plane)

```bash
ssh root@<control-plane-ip>

cd /home/m/tff/254CARBON/HMCo
./scripts/07-deploy-platform.sh "/home/m/tff/254CARBON/HMCo"
```

Expected time: 10-15 minutes

### 11. Restore Data (Control Plane)

```bash
# From control machine, switch context
kubectl config use-context <bare-metal-cluster-context>

cd /home/m/tff/254CARBON/HMCo

# Restore from backup
./scripts/10-restore-to-bare-metal.sh "./backups/kind-cluster-YYYYMMDD"
```

Expected time: 10-20 minutes (depends on data volume)

---

## Post-Migration Validation

### 12. Validate Deployment

```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/08-validate-deployment.sh
```

Checks:
- ✓ All nodes Ready
- ✓ All pods Running
- ✓ Services accessible
- ✓ Storage provisioned
- ✓ Network connectivity

---

## Quick Commands Reference

### Cluster Status
```bash
# All nodes
kubectl get nodes -o wide

# All pods
kubectl get pods --all-namespaces

# All services
kubectl get svc --all-namespaces

# Storage status
kubectl get pvc --all-namespaces
kubectl get pv
```

### Service Verification
```bash
# Check specific pod logs
kubectl logs -n data-platform <pod-name>

# Describe pod for events
kubectl describe pod -n data-platform <pod-name>

# Get into pod for debugging
kubectl exec -it -n data-platform <pod-name> -- /bin/bash

# Port forward for testing
kubectl port-forward -n data-platform svc/service-name 8080:8080
```

### Resource Management
```bash
# Check resource usage
kubectl top nodes
kubectl top pods --all-namespaces

# View resource requests/limits
kubectl describe nodes
kubectl describe pod -n data-platform <pod-name>
```

### Troubleshooting
```bash
# Get cluster events
kubectl get events --all-namespaces --sort-by='.lastTimestamp'

# Check node status
kubectl describe node <node-name>

# Kubelet logs (on node)
journalctl -u kubelet -n 50

# Container runtime logs (on node)
journalctl -u containerd -n 50
```

---

## Rollback to Kind (If Needed)

```bash
# Switch back to Kind context
kubectl config use-context kind-dev-cluster

# Verify Kind cluster is running
kubectl get nodes

# If Kind cluster was stopped, restart
kind create cluster --name dev-cluster --image kindest/node:v1.31.0
```

---

## Troubleshooting Quick Fixes

### Nodes Not Ready
```bash
ssh root@<node-ip>
systemctl restart kubelet
systemctl status kubelet
```

### Pod Stuck in Pending
```bash
# Check events
kubectl describe pod -n <namespace> <pod-name>

# Usually due to:
# - No available storage
# - Insufficient resources
# - Missing pull secrets
```

### ImagePullBackOff
```bash
# Verify image availability
kubectl describe pod -n <namespace> <pod-name>

# Check registry credentials
kubectl get secrets -n <namespace>

# Create pull secret if needed
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=<username> \
  --docker-password=<password>
```

---

## Next Steps After Migration

1. **Automate Backups**
   ```bash
   kubectl apply -f k8s/resilience/backup-policy.yaml
   ```

2. **Setup Monitoring**
   ```bash
   kubectl apply -f k8s/monitoring/prometheus-rules.yaml
   ```

3. **Configure Alerting**
   - Review alert thresholds
   - Setup notification channels
   - Test alert flow

4. **Decommission Kind** (when confident in new cluster)
   ```bash
   kind delete cluster --name dev-cluster
   ```

---

## Timing Summary

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Prepare servers (3 nodes) | 15-30 min |
| 2 | Install container runtime | 10-15 min |
| 3 | Install Kubernetes | 15-30 min |
| 4 | Init control plane | 5-10 min |
| 5 | Join worker nodes | 10-15 min |
| 6 | Deploy storage | 5-10 min |
| 7 | Deploy services | 10-15 min |
| 8 | Restore data | 10-20 min |
| 9 | Validate | 5-10 min |
| **Total** | **Full migration** | **1.5 - 2.5 hours** |

---

## Emergency Contacts & Resources

- **Kubernetes Docs**: https://kubernetes.io/docs/
- **Troubleshooting**: https://kubernetes.io/docs/tasks/debug/
- **OpenEBS**: https://openebs.io/docs/
- **Local Issues**: Check `/var/log/syslog` and `journalctl`

---

**For detailed information, see**: `full-migration-runbook.md`
