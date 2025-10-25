# 254Carbon Deployment Guide - Kind to Bare Metal Kubernetes Migration

**Last Updated**: October 20, 2025  
**Status**: Implementation Complete  
**Audience**: DevOps Engineers, Operations Team

---

## Overview

This guide provides everything needed to migrate the 254Carbon platform from a development Kind (Kubernetes in Docker) cluster to a production-ready bare metal Kubernetes deployment.

### What You Get

✓ Step-by-step installation scripts  
✓ Automated deployment and validation  
✓ Data backup and restore procedures  
✓ Comprehensive troubleshooting guides  
✓ Post-migration operational procedures  

### Estimated Duration

**Total Time**: 1.5 - 2.5 hours for complete migration  
- Infrastructure prep: 30-45 min
- Kubernetes setup: 30-40 min  
- Storage/Services: 20-30 min
- Data migration: 10-20 min
- Validation: 5-10 min

---

## Before You Start

### Prerequisites Checklist

- [ ] 3-5 bare metal servers prepared with Ubuntu 20.04+ or Debian 11+
- [ ] All servers have static IP addresses and networking configured
- [ ] SSH access to all servers with sudo/root capability
- [ ] Existing Kind cluster running (for data backup)
- [ ] `kubectl` installed on control machine
- [ ] Git access to deployment repository
- [ ] Backup storage space (~50GB minimum recommended)

### Architecture

```
254Carbon Bare Metal Cluster
├─ Control Plane (k8s-node1)
│  ├─ etcd
│  ├─ API Server
│  ├─ Scheduler
│  └─ Controller Manager
│
├─ Worker Node (k8s-node2)
│  ├─ kubelet
│  ├─ kube-proxy
│  ├─ Container Runtime (containerd)
│  └─ OpenEBS LocalPV Storage
│
└─ Worker Node (k8s-node3)
   ├─ kubelet
   ├─ kube-proxy
   ├─ Container Runtime (containerd)
   └─ OpenEBS LocalPV Storage

CNI: Flannel
Storage: OpenEBS (local storage)
External Access: Cloudflare Tunnel
```

---

## Quick Start (TL;DR)

For experienced Kubernetes operators who want just the essentials:

**[👉 quick-reference.md](quick-reference.md)**

Contains quick copy-paste commands for each phase.

---

## Complete Deployment Steps

### Step 1: Backup Kind Cluster

**Location**: [full-migration-runbook.md - Phase 4.1](full-migration-runbook.md#41-backup-from-kind-cluster)

```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/09-backup-from-kind.sh "./backups"
```

**Output**: Complete backup in `./backups/kind-migration-YYYYMMDD-HHMMSS/`

---

### Steps 2-9: Bare Metal Deployment

Detailed procedures in: **[full-migration-runbook.md](full-migration-runbook.md)**

| Step | Phase | Script | Duration | Notes |
|------|-------|--------|----------|-------|
| 2 | Prepare Servers | `01-prepare-servers.sh` | 5-10 min | Run on each node |
| 3 | Container Runtime | `02-install-container-runtime.sh` | 3-5 min | Run on each node |
| 4 | Kubernetes | `03-install-kubernetes.sh` | 5-10 min | Run on each node |
| 5 | Control Plane | `04-init-control-plane.sh` | 5-10 min | Control plane only |
| 6 | Worker Nodes | `05-join-worker-nodes.sh` | 3-5 min | Each worker node |
| 7 | Storage | `06-deploy-storage.sh` | 5-10 min | Control plane |
| 8 | Platform Services | `07-deploy-platform.sh` | 10-15 min | Control plane |
| 9 | Data Restore | `10-restore-to-bare-metal.sh` | 10-20 min | Control plane |

---

## Key Documentation

### For Migration Planning
- **[full-migration-runbook.md](full-migration-runbook.md)** - Full 9-phase runbook with procedures, verification steps, and troubleshooting
- **[production-migration-plan.md](production-migration-plan.md)** - Approved migration plan with scope and dependencies

### For Quick Reference
- **[quick-reference.md](quick-reference.md)** - Command-line cheat sheet
- **[quick-reference.md#troubleshooting-quick-fixes](quick-reference.md#troubleshooting-quick-fixes)** - Common problems and solutions

### For Existing Deployments
- **[../../readiness/production-readiness.md](../../readiness/production-readiness.md)** - Production hardening roadmap
- **[../../troubleshooting/](../../troubleshooting/)** - Fix common issues

### For Day-2 Operations
- **[../../../scripts/](../../../scripts/)** - Automated operational procedures
- **[../../../k8s/monitoring/](../../../k8s/monitoring/)** - Monitoring and alerting setup
- **[../../../k8s/backup/](../../../k8s/backup/)** - Backup and disaster recovery

---

## Migration Scripts

All scripts are in `scripts/` directory:

```
01-prepare-servers.sh          ← Prepare bare metal servers
02-install-container-runtime.sh ← Install containerd
03-install-kubernetes.sh        ← Install K8s components
04-init-control-plane.sh        ← Initialize control plane
05-join-worker-nodes.sh         ← Join worker nodes
06-deploy-storage.sh            ← Deploy OpenEBS
07-deploy-platform.sh           ← Deploy platform services
08-validate-deployment.sh       ← Validate everything works
09-backup-from-kind.sh          ← Backup data from Kind
10-restore-to-bare-metal.sh     ← Restore data to bare metal
```

### Script Usage Examples

```bash
# Backup Kind cluster
./scripts/09-backup-from-kind.sh "./backups/kind-backup-$(date +%Y%m%d)"

# Deploy platform to new cluster
./scripts/07-deploy-platform.sh "/home/m/tff/254CARBON/HMCo"

# Validate new deployment
./scripts/08-validate-deployment.sh

# Restore data
./scripts/10-restore-to-bare-metal.sh "./backups/kind-backup-20251020"
```

---

## Network Configuration

### Service Network Layout

```
Pod Network CIDR:        10.244.0.0/16
Service CIDR:            10.96.0.0/12
Kubernetes DNS:          10.96.0.10:53
API Server:              <control-plane-ip>:6443
Flannel VXLAN:           UDP 8472
```

### Firewall Rules (UFW)

```bash
# Automatically configured by 01-prepare-servers.sh
# If needed to add manually:

# Kubernetes API
sudo ufw allow 6443/tcp

# Kubelet API
sudo ufw allow 10250/tcp

# kube-proxy
sudo ufw allow 10256/tcp

# Flannel VXLAN
sudo ufw allow 8472/udp

# etcd
sudo ufw allow 2379:2380/tcp

# NodePort services
sudo ufw allow 30000:32767/tcp
```

---

## Storage Architecture

### Local Storage Setup

```bash
# On each worker node, create storage directory
sudo mkdir -p /mnt/openebs/local
sudo chmod 755 /mnt/openebs/local

# If using dedicated disks:
sudo mkfs.ext4 /dev/sdb1
sudo mount /dev/sdb1 /mnt/openebs/local
```

### StorageClass Configuration

Default storage class: `local-storage-standard`

```yaml
storageClassName: local-storage-standard
volumeBindingMode: WaitForFirstConsumer  # Wait until pod scheduled
allowVolumeExpansion: true
reclaimPolicy: Delete
```

### Storage Resources

- **Per Node**: 50Gi initial allocation (adjust as needed)
- **Total Capacity**: 150Gi+ for 3-node cluster
- **Reserved**: 20% buffer for system processes

---

## Verification Procedures

### Quick Health Check

```bash
# After deployment, verify cluster is ready
kubectl get nodes               # All should be Ready
kubectl get pods --all-namespaces  # Check for failures
kubectl get pvc --all-namespaces   # Storage provisioned
```

### Detailed Validation

```bash
# Run automated validation
./scripts/08-validate-deployment.sh

# Check logs
kubectl logs -n data-platform <pod-name>

# Describe issues
kubectl describe pod -n data-platform <pod-name>

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never \
  -- nslookup kafka.data-platform
```

### Service Tests

```bash
# MinIO health
kubectl run -it --rm test --image=curlimages/curl --restart=Never \
  -n data-platform -- curl http://minio:9000/minio/health/live

# Kafka connectivity
kubectl run -it --rm test --image=busybox --restart=Never \
  -n data-platform -- nc -zv kafka:9092

# Trino UI
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080 &
# Visit http://localhost:8080/ui/
```

---

## Troubleshooting Guide

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Nodes not Ready | kubelet service stopped | `systemctl restart kubelet` on node |
| Pods Pending | No storage available | Check `/mnt/openebs/local` on nodes |
| ImagePullBackOff | Private registry access | Check image pull secrets |
| CrashLoopBackOff | Pod startup error | `kubectl logs` to view error |
| Service not accessible | NetworkPolicy blocking | Check `kubectl get networkpolicy` |

### Getting Help

```bash
# Check cluster events
kubectl get events --all-namespaces --sort-by='.lastTimestamp'

# Check node status
kubectl describe node <node-name>

# View kubelet logs on node
ssh root@<node-ip>
journalctl -u kubelet -n 100

# View containerd logs
journalctl -u containerd -n 100
```

---

## Rollback Plan

If critical issues occur during migration:

### Immediate Rollback (within 1 hour)

```bash
# Switch DNS/Cloudflare back to Kind cluster
# All traffic automatically redirects to Kind

# Monitor Kind cluster logs
kubectl config use-context kind-dev-cluster
kubectl get pods --all-namespaces
```

### Full Rollback (if needed)

1. Keep Kind cluster running and verified
2. Disable bare metal cluster deployment
3. Point DNS/routing back to Kind
4. Investigate issues offline
5. Re-attempt deployment when ready

---

## Post-Migration Tasks

### Immediate (Day 1)

- [ ] Verify all services operational
- [ ] Test SSO/Cloudflare tunnel access
- [ ] Confirm data availability
- [ ] Document cluster access info
- [ ] Brief operations team

### Short Term (Week 1)

- [ ] Configure automated backups
- [ ] Setup log aggregation
- [ ] Implement monitoring alerts
- [ ] Document operational procedures
- [ ] Test disaster recovery

### Long Term (Month 1+)

- [ ] Performance tuning
- [ ] Capacity planning
- [ ] Network optimization
- [ ] Security hardening
- [ ] Multi-region HA consideration

### When Ready: Decommission Kind

```bash
# After 2-3 weeks of stable operation
kind delete cluster --name dev-cluster

# Remove from kubeconfig
kubectl config delete-context kind-dev-cluster
```

---

## Support & Resources

### Documentation
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [kubeadm Troubleshooting](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/troubleshooting-kubeadm/)
- [OpenEBS Docs](https://openebs.io/docs/)
- [Flannel Docs](https://coreos.com/flannel/)

### Community
- Kubernetes Slack: [Kubernetes Community](https://kubernetes.slack.com)
- Stack Overflow: Tag `kubernetes`
- GitHub Issues: [254Carbon/HMCo](https://github.com/254Carbon/HMCo/issues)

### Internal
- Previous deployment docs: `docs/` directory
- Runbooks: `k8s/` directory
- Scripts: `scripts/` directory

---

## Version History

| Date | Version | Status | Notes |
|------|---------|--------|-------|
| 2025-10-20 | 1.0 | Production | Initial migration implementation complete |

---

## Next Steps

1. **Read the full runbook**: [full-migration-runbook.md](full-migration-runbook.md)
2. **Prepare infrastructure**: Arrange bare metal servers
3. **Test backups**: Run backup scripts on existing Kind cluster
4. **Start migration**: Follow quick reference or full runbook
5. **Validate deployment**: Run validation scripts
6. **Monitor closely**: First 48 hours are critical

---

**Need help?** Check [quick-reference.md#troubleshooting-quick-fixes](quick-reference.md#troubleshooting-quick-fixes) or review full runbook troubleshooting section.
