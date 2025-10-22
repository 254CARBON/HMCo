# 🚀 START DEPLOYMENT HERE

**Status**: ✅ FULLY AUTOMATED - READY TO LAUNCH  
**Script**: `./scripts/00-deploy-all.sh`  
**Estimated Time**: 1.5 - 2.5 Hours  

---

## ONE-LINE DEPLOYMENT

Copy-paste this with your node IPs:

```bash
cd /home/m/tff/254CARBON/HMCo && ./scripts/00-deploy-all.sh -c 192.168.1.100 -w 192.168.1.101,192.168.1.102
```

**That's it!** The script will handle everything:
- Backup Kind cluster ✓
- Prepare all servers ✓
- Install Kubernetes ✓
- Deploy services ✓
- Restore data ✓
- Validate everything ✓

---

## BEFORE YOU RUN

### ✅ Checklist

- [ ] 3-5 Ubuntu 22.04 servers ready
- [ ] Static IPs assigned (e.g., 192.168.1.100, 192.168.1.101, 192.168.1.102)
- [ ] SSH root access working to all nodes
- [ ] Control plane IP documented
- [ ] Worker IPs documented
- [ ] Kind cluster running (will be backed up automatically)
- [ ] Network connectivity verified (ping between nodes)

### ⚡ Quick Infrastructure Setup

```bash
# On each server, verify Ubuntu 22.04:
cat /etc/os-release | grep PRETTY_NAME    # Should show Ubuntu 22.04

# Verify network connectivity:
ping 192.168.1.100  # From another node
ping 192.168.1.101  # etc.

# Verify SSH access:
ssh root@192.168.1.100 "echo 'SSH works!'"
```

---

## DEPLOYMENT OPTIONS

### 🎯 Option 1: 3-Node Cluster (Recommended)

```bash
./scripts/00-deploy-all.sh \
  -c 192.168.1.100 \
  -w 192.168.1.101,192.168.1.102
```

### 🎯 Option 2: 5-Node Cluster

```bash
./scripts/00-deploy-all.sh \
  -c 192.168.1.100 \
  -w 192.168.1.101,192.168.1.102,192.168.1.103,192.168.1.104
```

### 🎯 Option 3: Custom Backup Directory

```bash
./scripts/00-deploy-all.sh \
  -c 192.168.1.100 \
  -w 192.168.1.101,192.168.1.102 \
  -b /path/to/backup
```

### 🎯 Option 4: Skip Kind Backup (if already done)

```bash
./scripts/00-deploy-all.sh \
  -c 192.168.1.100 \
  -w 192.168.1.101,192.168.1.102 \
  -s
```

---

## STEP-BY-STEP

### Step 1: Navigate to Project

```bash
cd /home/m/tff/254CARBON/HMCo
```

### Step 2: Update Your IPs

```bash
# Replace these with YOUR node IPs:
CONTROL_PLANE="192.168.1.100"
WORKERS="192.168.1.101,192.168.1.102"
```

### Step 3: Run Deployment

```bash
./scripts/00-deploy-all.sh -c ${CONTROL_PLANE} -w ${WORKERS}
```

### Step 4: Watch Progress

The script will display:
- ✓ Real-time progress in terminal
- ✓ Color-coded status (green=success, red=error)
- ✓ Estimated time for each phase
- ✓ Final summary report

### Step 5: Validate Success

When complete, you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║ Deployment Summary                                           ║
╚══════════════════════════════════════════════════════════════╝

Status: ✅ SUCCESS

Duration: 2h 15m 30s

All phases completed successfully!

Next Steps:
1. Monitor the cluster for 24-48 hours
2. Configure automated backups
3. Setup monitoring and alerting
4. Train operations team

Access Control Plane:
  ssh root@192.168.1.100
```

---

## WHAT GETS DEPLOYED

### Services
- ✅ Zookeeper & Kafka (messaging)
- ✅ MinIO (object storage)
- ✅ LakeFS (data versioning)
- ✅ Iceberg REST Catalog (metadata)
- ✅ Trino (SQL queries)
- ✅ Spark (data processing)
- ✅ Vault (secrets management)
- ✅ Prometheus & Grafana (monitoring)
- ✅ Superset (visualization)
- ✅ DataHub (data catalog)
- ✅ DolphinScheduler (workflow)
- ✅ Cloudflare Tunnel (external access)

### Infrastructure
- ✅ Kubernetes cluster (3-5 nodes)
- ✅ Flannel networking (CNI)
- ✅ OpenEBS storage (persistent volumes)
- ✅ RBAC & security policies
- ✅ Ingress rules
- ✅ Monitoring & logging

---

## DURING DEPLOYMENT

### Monitor in Another Terminal

```bash
# Watch cluster coming up:
ssh root@192.168.1.100
kubectl get nodes -w

# In another shell:
ssh root@192.168.1.100
kubectl get pods --all-namespaces -w

# View logs:
tail -f .deployment-logs/deployment-*.log
```

### Expected Timeline

```
Phase 1 (Prepare):        15-30 min
Phase 2a (Runtime):       10-15 min
Phase 2b (K8s):          15-30 min
Phase 2c (Control):        5-10 min
Phase 2d (Workers):       10-15 min
Phase 3 (Storage):         5-10 min
Phase 4 (Services):       10-15 min
Phase 5 (Data):           10-20 min
Phase 6 (Validate):        5-10 min
                          ──────────
                 TOTAL: 1.5-2.5 hours
```

---

## IF SOMETHING GOES WRONG

### Deployment Fails

```bash
# 1. Check logs
tail -f .deployment-logs/deployment-*.log

# 2. Find failed phase in log
# 3. SSH to problematic node and investigate
ssh root@FAILED_NODE_IP

# 4. Check services
systemctl status kubelet
systemctl status containerd

# 5. Fix issue
# 6. Re-run deployment (skips completed phases)
./scripts/00-deploy-all.sh -c 192.168.1.100 -w 192.168.1.101,192.168.1.102
```

### Need to Rollback

```bash
# Revert to Kind cluster (all data backed up)
kubectl config use-context kind-dev-cluster
kubectl get nodes

# If Kind is stable, redirect traffic back
# Keep bare metal cluster for debugging
```

### SSH Connection Issues

```bash
# Verify node is online
ping 192.168.1.100

# Test SSH
ssh -v root@192.168.1.100

# Fix SSH key permissions
chmod 600 ~/.ssh/id_rsa
chmod 700 ~/.ssh
```

---

## AFTER DEPLOYMENT

### 1. Verify Everything Works

```bash
ssh root@192.168.1.100

# Check cluster
kubectl get nodes
kubectl get pods --all-namespaces

# Check storage
kubectl get pvc --all-namespaces

# Check services
kubectl get svc --all-namespaces
```

### 2. Test Applications

```bash
# Cloudflare access
curl -v https://254carbon.com

# Monitoring dashboard
ssh -L 9090:localhost:9090 root@192.168.1.100
# Visit http://localhost:9090

# Data access
# Test in your applications
```

### 3. Monitor for 24-48 Hours

- Watch pod status
- Check logs for errors
- Verify data integrity
- Test failover scenarios

### 4. Configure Backups

```bash
kubectl apply -f k8s/resilience/backup-policy.yaml
```

### 5. Setup Alerts

```bash
kubectl apply -f k8s/monitoring/prometheus-rules.yaml
```

---

## DEPLOYMENT SCRIPT FEATURES

✅ **Fully Automated** - All phases run automatically  
✅ **Parallel Execution** - Optimized for speed  
✅ **Error Handling** - Stops on errors, clear messages  
✅ **Logging** - Complete logs saved to `.deployment-logs/`  
✅ **Validation** - Checks after each phase  
✅ **Idempotent** - Safe to re-run if needed  
✅ **Colored Output** - Easy to read status  
✅ **Summary Report** - Complete deployment summary  
✅ **Backup Integration** - Automatic Kind cluster backup  
✅ **Data Restoration** - Automatic data restore  

---

## QUICK REFERENCE

| Need | Command |
|------|---------|
| **Deploy** | `./scripts/00-deploy-all.sh -c 192.168.1.100 -w 192.168.1.101,192.168.1.102` |
| **Skip backup** | Add `-s` flag |
| **Custom backup dir** | Add `-b /path` |
| **Help** | `./scripts/00-deploy-all.sh -h` |
| **View logs** | `tail -f .deployment-logs/deployment-*.log` |
| **Check nodes** | `ssh root@192.168.1.100 kubectl get nodes` |
| **Verify pods** | `ssh root@192.168.1.100 kubectl get pods --all-namespaces` |

---

## CONFIGURATION FILE

Edit `DEPLOYMENT_CONFIG.sh` for persistent settings:

```bash
CONTROL_PLANE_IP="192.168.1.100"
WORKER_IPS="192.168.1.101,192.168.1.102"
AUTO_VALIDATE="true"
AUTO_ROLLBACK="false"
```

Then use:
```bash
source DEPLOYMENT_CONFIG.sh
./scripts/00-deploy-all.sh -c ${CONTROL_PLANE_IP} -w ${WORKER_IPS}
```

---

## SUPPORT & RESOURCES

📖 **Documentation**:
- `AUTOMATED_DEPLOYMENT.md` - Full guide
- `full-migration-runbook.md` - Detailed procedures
- `UBUNTU_22_04_QUICK_REFERENCE.md` - Troubleshooting

📝 **Logs**: `.deployment-logs/deployment-*.log`

🔗 **External**:
- Kubernetes: https://kubernetes.io/docs/
- Ubuntu: https://ubuntu.com/

---

## 🎯 READY?

```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/00-deploy-all.sh -c 192.168.1.100 -w 192.168.1.101,192.168.1.102
```

**That's it! Deployment will complete in 1.5 - 2.5 hours.**

---

## Timeline

| Task | Time |
|------|------|
| Read this guide | 5 min |
| Prepare IPs | 2 min |
| Run deployment | 1.5-2.5 hrs |
| Monitor result | 1-2 min |
| **Total** | **~2-3 hrs** |

---

**PROJECT STATUS**: ✅ **READY FOR FULL AUTOMATED DEPLOYMENT**

**Start now:**
```bash
./scripts/00-deploy-all.sh -c YOUR_CONTROL_IP -w WORKER_IPS
```
