# 254Carbon Bare Metal Migration - Complete Resource Index

**Status**: ✅ Implementation Complete  
**Last Updated**: October 20, 2025  
**Project**: Kubernetes cluster migration from Kind to production bare metal

---

## 📋 Start Here

### New to this migration?

1. **[deployment-guide.md](deployment-guide.md)** ← Start with this overview
   - What you'll get
   - Architecture overview
   - Quick start navigation
   - Support resources

2. **[quick-reference.md](quick-reference.md)** ← Commands cheat sheet
   - Copy-paste ready commands
   - Step-by-step procedure
   - Quick troubleshooting
   - Timing summary

3. **[full-migration-runbook.md](full-migration-runbook.md)** ← Detailed runbook
   - 70+ page comprehensive guide
   - Each phase explained in detail
   - Verification procedures
   - Full troubleshooting guide

---

## 📚 Documentation Map

### Executive Level
| Document | Purpose | Audience |
|----------|---------|----------|
| [deployment-guide.md](deployment-guide.md) | Navigation & overview | Managers, leads |
| [K8S_IMPLEMENTATION_SUMMARY.md](K8S_IMPLEMENTATION_SUMMARY.md) | What was built | Project stakeholders |
| [k8s-bare.plan.md](k8s-bare.plan.md) | Approved migration plan | Decision makers |

### Operational Level
| Document | Purpose | Audience |
|----------|---------|----------|
| [quick-reference.md](quick-reference.md) | Commands & cheat sheet | DevOps engineers |
| [full-migration-runbook.md](full-migration-runbook.md) | Complete procedures | Operations team |

### Technical Details
| Document | Purpose | Audience |
|----------|---------|----------|
| [k8s/storage/local-storage-provisioner.yaml](k8s/storage/local-storage-provisioner.yaml) | Storage config | Infrastructure team |
| [scripts/](scripts/) | Automated procedures | Automation engineers |

---

## 🔧 Automation Scripts

### Setup Sequence

```
Start here ↓

Step 1: Backup Kind Cluster
└─ 09-backup-from-kind.sh
   └─ Backup directory created

Step 2: Prepare Each Bare Metal Server
└─ 01-prepare-servers.sh (run on each node)

Step 3: Install Container Runtime
└─ 02-install-container-runtime.sh (run on each node)

Step 4: Install Kubernetes Components
└─ 03-install-kubernetes.sh (run on each node)

Step 5: Initialize Control Plane
└─ 04-init-control-plane.sh (control plane only)

Step 6: Join Worker Nodes
└─ 05-join-worker-nodes.sh (each worker node)

Step 7: Deploy Storage Infrastructure
└─ 06-deploy-storage.sh

Step 8: Deploy Platform Services
└─ 07-deploy-platform.sh

Step 9: Restore Data
└─ 10-restore-to-bare-metal.sh

Step 10: Validate Everything
└─ 08-validate-deployment.sh

✓ Migration complete!
```

### Script Reference

| # | Script | Purpose | Duration | Target |
|---|--------|---------|----------|--------|
| 1 | `01-prepare-servers.sh` | Kernel, networking, firewall setup | 5-10 min | Each node |
| 2 | `02-install-container-runtime.sh` | Install containerd | 3-5 min | Each node |
| 3 | `03-install-kubernetes.sh` | Install K8s components | 5-10 min | Each node |
| 4 | `04-init-control-plane.sh` | Initialize control plane | 5-10 min | Control plane |
| 5 | `05-join-worker-nodes.sh` | Join workers to cluster | 3-5 min | Each worker |
| 6 | `06-deploy-storage.sh` | Deploy OpenEBS | 5-10 min | Control plane |
| 7 | `07-deploy-platform.sh` | Deploy all services | 10-15 min | Control plane |
| 8 | `08-validate-deployment.sh` | Validate cluster | 5-10 min | Control plane |
| 9 | `09-backup-from-kind.sh` | Backup Kind data | 10-20 min | Control machine |
| 10 | `10-restore-to-bare-metal.sh` | Restore to new cluster | 10-20 min | Control plane |

**Location**: `/home/m/tff/254CARBON/HMCo/scripts/`

---

## 📖 Phase-by-Phase Navigation

### Phase 1: Preparation
**Duration**: 30-45 minutes

**Read**: [full-migration-runbook.md - Phase 1](full-migration-runbook.md#phase-1-infrastructure-preparation)

**Execute**:
```bash
# On each server
./scripts/01-prepare-servers.sh
```

---

### Phase 2: Kubernetes Installation
**Duration**: 30-40 minutes

**Read**: [full-migration-runbook.md - Phase 2](full-migration-runbook.md#phase-2-kubernetes-installation)

**Execute** (in order):
```bash
# All nodes
./scripts/02-install-container-runtime.sh
./scripts/03-install-kubernetes.sh

# Control plane
./scripts/04-init-control-plane.sh

# Save join command, then each worker:
./scripts/05-join-worker-nodes.sh <join-command>
```

---

### Phase 3: Storage
**Duration**: 5-10 minutes

**Read**: [full-migration-runbook.md - Phase 3](full-migration-runbook.md#phase-3-storage-infrastructure)

**Execute**:
```bash
./scripts/06-deploy-storage.sh
```

---

### Phase 4: Service Migration
**Duration**: 10-15 minutes

**Read**: [full-migration-runbook.md - Phase 4](full-migration-runbook.md#phase-4-service-migration)

**Execute**:
```bash
./scripts/07-deploy-platform.sh
```

---

### Phase 5: Data Migration
**Duration**: 10-20 minutes

**Read**: [full-migration-runbook.md - Phase 5](full-migration-runbook.md#phase-5-data-migration)

**Backup from Kind**:
```bash
./scripts/09-backup-from-kind.sh "./backups"
```

**Restore to Bare Metal**:
```bash
./scripts/10-restore-to-bare-metal.sh "./backups/kind-migration-YYYYMMDD-HHMMSS"
```

---

### Phase 6: Validation
**Duration**: 5-10 minutes

**Read**: [full-migration-runbook.md - Phase 6](full-migration-runbook.md#phase-6-validation)

**Execute**:
```bash
./scripts/08-validate-deployment.sh
```

---

### Phase 7: Cutover
**Duration**: Variable (part of deployment window)

**Read**: [full-migration-runbook.md - Phase 7](full-migration-runbook.md#phase-7-production-cutover)

**Includes**:
- Pre-cutover checklist
- DNS/routing updates
- Monitoring setup
- Rollback procedures

---

## ❓ Troubleshooting Guide

### Quick Fixes
**Location**: [quick-reference.md#troubleshooting-quick-fixes](quick-reference.md#troubleshooting-quick-fixes)

Common issues with quick solutions:
- Nodes not ready
- Pods stuck in pending
- ImagePullBackOff errors

### Detailed Troubleshooting
**Location**: [full-migration-runbook.md#troubleshooting](full-migration-runbook.md#troubleshooting)

In-depth troubleshooting for:
- Network connectivity
- Storage issues
- Pod startup failures
- Service access problems

### Command Reference
**Location**: [quick-reference.md#quick-commands-reference](quick-reference.md#quick-commands-reference)

Essential kubectl commands:
```bash
# Cluster status
kubectl get nodes -o wide
kubectl get pods --all-namespaces
kubectl get pvc --all-namespaces

# Debugging
kubectl describe pod -n <namespace> <pod-name>
kubectl logs -n <namespace> <pod-name>
kubectl events --all-namespaces
```

---

## 🎯 Common Use Cases

### "I want to start the migration now"
1. Read: [quick-reference.md](quick-reference.md)
2. Execute scripts in order
3. For issues: Check troubleshooting section

### "I need to understand the full process"
1. Read: [deployment-guide.md](deployment-guide.md)
2. Then: [full-migration-runbook.md](full-migration-runbook.md)
3. Reference: [quick-reference.md](quick-reference.md) as needed

### "Something is not working"
1. Check: [quick-reference.md#troubleshooting-quick-fixes](quick-reference.md#troubleshooting-quick-fixes)
2. If not found: Check [full-migration-runbook.md#troubleshooting](full-migration-runbook.md#troubleshooting)
3. Still stuck: Review pod logs and events

### "I need to verify if deployment is successful"
1. Run: `./scripts/08-validate-deployment.sh`
2. Check results against success criteria
3. Review any failures in detail

### "I need to roll back the migration"
1. Read: [full-migration-runbook.md - Rollback Plan](full-migration-runbook.md#73-rollback-plan)
2. Switch kubectl context back to Kind
3. Monitor until stable

### "I want to back up the Kind cluster"
1. Run: `./scripts/09-backup-from-kind.sh "./backups"`
2. Verify backup contents
3. Store securely

---

## 📊 Implementation Checklist

### Pre-Migration
- [ ] Read deployment-guide.md
- [ ] Review quick-reference.md
- [ ] Procure 3-5 bare metal servers
- [ ] Configure networking and static IPs
- [ ] Run backup of Kind cluster
- [ ] Document current configuration

### During Migration
- [ ] Run Phase 1-2 setup scripts
- [ ] Verify Kubernetes cluster
- [ ] Deploy storage (Phase 3)
- [ ] Deploy services (Phase 4)
- [ ] Migrate data (Phase 5)
- [ ] Run validation (Phase 6)

### Post-Migration
- [ ] Monitor deployment for 24 hours
- [ ] Verify all services operational
- [ ] Configure backups and monitoring
- [ ] Document procedures
- [ ] Train operations team
- [ ] Plan Kind cluster decommission

---

## 🔗 Related Documentation

### Existing Project Docs
- [../../readiness/production-readiness.md](../../../readiness/production-readiness.md) - Production hardening
- [IMMEDIATE_REMEDIATION.md](../IMMEDIATE_REMEDIATION.md) - Fix current issues
- [docs/cloudflare/](../docs/cloudflare/) - Tunnel setup
- [docs/sso/](../docs/sso/) - Authentication setup

### External Resources
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [kubeadm Guide](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/)
- [OpenEBS Documentation](https://openebs.io/docs/)
- [Flannel Networking](https://coreos.com/flannel/)

---

## 📈 Timeline Summary

| Phase | Task | Duration | Parallelizable |
|-------|------|----------|-----------------|
| 1 | Prepare servers | 5-10 min/node | Yes (3-5 nodes) |
| 2a | Container runtime | 3-5 min/node | Yes |
| 2b | Kubernetes install | 5-10 min/node | Yes |
| 2c | Init control plane | 5-10 min | No |
| 2d | Join workers | 3-5 min/node | Partial |
| 3 | Deploy storage | 5-10 min | No |
| 4 | Deploy services | 10-15 min | No |
| 5 | Migrate data | 10-20 min | No |
| 6 | Validate | 5-10 min | No |
| **Total** | **Complete** | **1.5-2.5 hrs** | **Optimized** |

---

## 💾 File Structure

```
/home/m/tff/254CARBON/HMCo/
│
├── 📄 deployment-guide.md               ← Navigation guide
├── 📄 quick-reference.md     ← Quick commands
├── 📄 full-migration-runbook.md       ← Full runbook
├── 📄 K8S_IMPLEMENTATION_SUMMARY.md     ← What was built
├── 📄 index.md     ← This file
│
├── 📁 scripts/
│   ├── 01-prepare-servers.sh
│   ├── 02-install-container-runtime.sh
│   ├── 03-install-kubernetes.sh
│   ├── 04-init-control-plane.sh
│   ├── 05-join-worker-nodes.sh
│   ├── 06-deploy-storage.sh
│   ├── 07-deploy-platform.sh
│   ├── 08-validate-deployment.sh
│   ├── 09-backup-from-kind.sh
│   └── 10-restore-to-bare-metal.sh
│
├── 📁 k8s/
│   ├── 📁 storage/
│   │   └── local-storage-provisioner.yaml
│   └── [other existing manifests]
│
└── [other project files]
```

---

## ✅ Validation Criteria

After deployment, verify:

- [ ] `kubectl get nodes` shows all nodes Ready
- [ ] `kubectl get pods --all-namespaces` shows Running pods
- [ ] `kubectl get pvc --all-namespaces` shows Bound PVCs
- [ ] `./scripts/08-validate-deployment.sh` succeeds
- [ ] Services accessible via endpoints
- [ ] Data restored and accessible
- [ ] Cloudflare tunnel connected
- [ ] Monitoring dashboards functional

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Read overview
cat deployment-guide.md | head -50

# 2. Check prerequisites
echo "Do I have 3-5 bare metal servers?"
echo "Are they networked and SSH-accessible?"

# 3. Backup existing
./scripts/09-backup-from-kind.sh "./backups"

# 4. Start migration (follow quick-reference.md)
./scripts/01-prepare-servers.sh  # on each node
# ... continue with other scripts

# 5. Validate
./scripts/08-validate-deployment.sh
```

---

## 📞 Support

**Documentation**:
- Level 1: [quick-reference.md](quick-reference.md)
- Level 2: [full-migration-runbook.md](full-migration-runbook.md)
- Level 3: Script comments and inline documentation

**Online Resources**:
- Kubernetes: https://kubernetes.io/docs/
- OpenEBS: https://openebs.io/docs/
- Flannel: https://coreos.com/flannel/docs/

---

## 📝 Version Info

| Item | Value |
|------|-------|
| **Implementation Date** | October 20, 2025 |
| **Status** | ✅ Complete |
| **Kubernetes Version** | 1.28+ |
| **Container Runtime** | containerd |
| **CNI Plugin** | Flannel |
| **Storage Provider** | OpenEBS (local) |
| **Target Deployment** | Bare metal (3-5 nodes) |

---

**Ready to migrate?** Start with [deployment-guide.md](deployment-guide.md) or jump to [quick-reference.md](quick-reference.md) if you're experienced.
