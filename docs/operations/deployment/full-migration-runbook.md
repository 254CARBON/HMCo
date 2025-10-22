# Kubernetes Bare Metal Migration Runbook

**Last Updated**: October 20, 2025  
**Status**: Implementation Ready  
**Project**: 254Carbon Data Platform  
**Scope**: Migrate from Kind (Kubernetes in Docker) to production bare metal cluster

---

## Executive Summary

This document provides step-by-step procedures to migrate the 254Carbon platform from a local Kind cluster to a production-grade Kubernetes deployment on bare metal servers. The migration preserves all services, configurations, and data while establishing a scalable, resilient foundation for production operations.

### Key Objectives
- Migrate from Kind to kubeadm-based Kubernetes
- Deploy on 3-5 bare metal nodes
- Maintain all existing services and data
- Zero-downtime service preservation strategy
- Enable horizontal scaling for future growth

### Timeline Estimate
- **Phase 1-2**: 2-3 hours (infrastructure + Kubernetes setup)
- **Phase 3**: 1 hour (storage deployment)
- **Phase 4**: 2-3 hours (service migration)
- **Phase 5**: 1-2 hours (data migration)
- **Phase 6**: 1 hour (validation)
- **Total**: 7-10 hours

---

## Prerequisites

### Infrastructure Requirements

#### Hardware (per node)
- **CPU**: 4+ cores (Intel/AMD x86-64)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 100GB+ per node
- **Network**: Gigabit Ethernet, static IPs recommended

#### Operating System
- Ubuntu 20.04 LTS / 22.04 LTS (recommended)
- Debian 11+ (alternative)
- Kernel 5.15+ (check with `uname -r`)

#### Network Setup
- All nodes must have network connectivity
- Static IP addresses preferred
- DNS resolution configured (or `/etc/hosts` entries)
- Firewall rules for Kubernetes ports (see Phase 1)

### Tools and Access
- SSH access to all servers (root or sudo capable)
- `kubectl` installed on control machine
- Git access to repository (for deployment files)
- Existing Kind cluster running (for data backup)

---

## Phase 1: Infrastructure Preparation

### Objective
Prepare bare metal servers with prerequisites for Kubernetes installation.

### 1.1 Server Initial Setup

On **each bare metal server**, execute:

```bash
# Connect to server
ssh root@<server-ip>

# Run preparation script
curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/01-prepare-servers.sh | bash -s -- "k8s-node-1"
```

**What this script does:**
- Updates system packages
- Sets hostname
- Disables swap
- Enables kernel modules (overlay, br_netfilter)
- Configures sysctl for Kubernetes
- Installs NTP
- Configures UFW firewall with Kubernetes ports
- Sets system limits for container workloads

**Verification:**
```bash
# Verify settings applied
uname -r                          # Check kernel version (5.15+)
sysctl net.ipv4.ip_forward        # Should be 1
sudo cat /etc/fstab | grep swap   # Should show commented swap
```

### 1.2 Network Configuration

Verify network connectivity between all nodes:

```bash
# From each node, test connectivity to others
ping <other-node-ip>

# Verify DNS resolution
nslookup google.com
```

Document node configuration:
```
Node 1 (Control Plane):
  - Hostname: k8s-control-1
  - IP: 192.168.1.100
  - Role: Control Plane

Node 2 (Worker):
  - Hostname: k8s-worker-1
  - IP: 192.168.1.101
  - Role: Worker

Node 3 (Worker):
  - Hostname: k8s-worker-2
  - IP: 192.168.1.102
  - Role: Worker

[Add additional nodes as needed]
```

---

## Phase 2: Kubernetes Installation

### Objective
Install Kubernetes components and initialize the cluster.

### 2.1 Install Container Runtime (All Nodes)

On **each node**, run:

```bash
ssh root@<node-ip>

curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/02-install-container-runtime.sh | bash
```

**Verification:**
```bash
systemctl status containerd
containerd --version
```

### 2.2 Install Kubernetes Components (All Nodes)

On **each node**, run:

```bash
ssh root@<node-ip>

curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/03-install-kubernetes.sh | bash
```

**Verification:**
```bash
kubeadm version
kubelet --version
kubectl version --client
```

### 2.3 Initialize Control Plane

On **control plane node only**, run:

```bash
ssh root@<control-plane-ip>

curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/04-init-control-plane.sh | bash
```

**Expected Output:**
The script will display a `kubeadm join` command similar to:

```
kubeadm join 192.168.1.100:6443 --token abc123.def456 \
  --discovery-token-ca-cert-hash sha256:abc123def456...
```

**Important**: Save this command for joining worker nodes.

**Verification:**
```bash
# On control plane
kubectl get nodes
kubectl get pods -n kube-system
kubectl get pods -n kube-flannel
```

All system pods should show as `Running` or `Completed`.

### 2.4 Join Worker Nodes

On **each worker node**, run the saved join command:

```bash
ssh root@<worker-node-ip>

# Run the command from control plane output
curl -fsSL https://raw.githubusercontent.com/254Carbon/HMCo/main/scripts/05-join-worker-nodes.sh | bash -s -- "kubeadm join 192.168.1.100:6443 --token abc123.def456 --discovery-token-ca-cert-hash sha256:abc123def456..."
```

**Verification (on control plane):**
```bash
kubectl get nodes
# Should show all nodes in Ready status
```

---

## Phase 3: Storage Infrastructure

### Objective
Deploy OpenEBS for persistent storage across cluster.

### 3.1 Prepare Local Storage (Each Node)

On **each worker node**, prepare local storage directories:

```bash
ssh root@<node-ip>

# Create local storage directory
sudo mkdir -p /mnt/openebs/local
sudo chmod 755 /mnt/openebs/local

# If using dedicated disk, mount it first
# Example for /dev/sdb1:
# sudo mkfs.ext4 /dev/sdb1
# sudo mount /dev/sdb1 /mnt/openebs/local
# sudo bash -c 'echo "/dev/sdb1 /mnt/openebs/local ext4 defaults 0 2" >> /etc/fstab'
```

### 3.2 Deploy Storage (From Control Plane)

On **control plane node**:

```bash
cd /home/m/tff/254CARBON/HMCo

# Run storage deployment script
./scripts/06-deploy-storage.sh "/home/m/tff/254CARBON/HMCo"
```

**Verification:**
```bash
kubectl get storageclass
kubectl get pv
kubectl get pods -n openebs -o wide
```

---

## Phase 4: Service Migration

### Objective
Deploy all 254Carbon platform services to bare metal cluster.

### 4.1 Backup from Kind Cluster

On **control machine**, switch to Kind context:

```bash
# Check current context
kubectl config get-contexts

# Switch to Kind if needed
kubectl config use-context kind-dev-cluster

# Run backup
cd /home/m/tff/254CARBON/HMCo
./scripts/09-backup-from-kind.sh "./backups"
```

This creates a backup containing:
- Namespace definitions
- All resources (deployments, statefulsets, etc.)
- Secrets and ConfigMaps
- PersistentVolume definitions
- PVC data (if available)

**Backup location**: `./backups/kind-migration-YYYYMMDD-HHMMSS/`

### 4.2 Deploy Services to Bare Metal

On **control machine**, switch to bare metal context:

```bash
# List available contexts
kubectl config get-contexts

# Switch to bare metal cluster
kubectl config use-context <bare-metal-context>

# Verify connectivity
kubectl cluster-info

# Deploy platform
cd /home/m/tff/254CARBON/HMCo
./scripts/07-deploy-platform.sh "/home/m/tff/254CARBON/HMCo"
```

This deploys services in order:
1. Namespaces
2. RBAC and networking
3. Storage infrastructure
4. Data platform services (Zookeeper, Kafka, MinIO, LakeFS, Iceberg)
5. Compute services (Trino, Spark)
6. Supporting services (Monitoring, Vault, Superset)
7. Cloudflare tunnel and ingress

---

## Phase 5: Data Migration

### Objective
Restore backed-up data to new cluster.

### 5.1 Restore Data

On **control machine**, switch to bare metal context:

```bash
# Ensure on bare metal cluster
kubectl config use-context <bare-metal-context>

# Restore from backup
cd /home/m/tff/254CARBON/HMCo
./scripts/10-restore-to-bare-metal.sh "./backups/kind-migration-YYYYMMDD-HHMMSS"
```

This restores:
- Namespace definitions
- RBAC roles and bindings
- Storage class definitions
- PersistentVolumes
- ConfigMaps and Secrets
- All namespaced resources
- PVC data (optional)

---

## Phase 6: Validation

### Objective
Verify all services are running correctly on new cluster.

### 6.1 Run Validation Script

```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/08-validate-deployment.sh
```

This script checks:
- Cluster health and node status
- Pod status in all namespaces
- Service connectivity
- DNS resolution
- Storage provisioning
- Ingress configuration
- Resource usage

### 6.2 Manual Verification

Check critical services:

```bash
# Check all pods
kubectl get pods --all-namespaces

# Check services
kubectl get svc --all-namespaces

# Check storage
kubectl get pvc --all-namespaces

# Test specific service
kubectl run -it --rm debug --image=busybox --restart=Never \
  -- nslookup kafka.data-platform.svc.cluster.local
```

### 6.3 Service-Specific Tests

```bash
# Test MinIO
kubectl run -it --rm test-minio --image=curlimages/curl --restart=Never \
  -n data-platform \
  -- curl -s http://minio:9000/minio/health/live

# Test Trino
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080 &
curl http://localhost:8080/ui/

# Test Vault
kubectl port-forward -n vault-prod svc/vault 8200:8200 &
curl http://localhost:8200/v1/sys/health
```

---

## Phase 7: Production Cutover

### Objective
Switch production traffic to new cluster with monitoring and rollback plan.

### 7.1 Pre-Cutover Checklist

- [ ] All nodes are Ready: `kubectl get nodes`
- [ ] All pods are Running: `kubectl get pods --all-namespaces | grep -v Running`
- [ ] Storage working: `kubectl get pvc --all-namespaces`
- [ ] Services accessible: Test critical endpoints
- [ ] Backups verified: Test restore procedure
- [ ] DNS configured: Update Cloudflare if needed
- [ ] Monitoring active: Check Prometheus/Grafana
- [ ] Alerting configured: Verify alert thresholds

### 7.2 Cutover Procedure

**Step 1: Stop services on Kind cluster** (optional, if keeping for reference)

```bash
kubectl config use-context kind-dev-cluster

# Scale down deployments
kubectl scale deployment --all -n data-platform --replicas=0
```

**Step 2: Update DNS/Load Balancer** (if applicable)

Update DNS records to point to bare metal cluster IPs or update Cloudflare tunnel configuration.

**Step 3: Monitor new cluster**

```bash
# Watch pod status
kubectl get pods --all-namespaces -w

# Monitor resource usage
kubectl top nodes
kubectl top pods --all-namespaces

# Check logs for errors
kubectl logs -f -n data-platform <pod-name>
```

**Step 4: Validate functionality**

- Test data ingestion pipelines
- Verify data is accessible
- Check monitoring dashboards
- Confirm SSO/Cloudflare tunnel access

### 7.3 Rollback Plan

If critical issues occur:

**Immediate Rollback (within 1 hour):**
```bash
# Switch DNS back to Kind cluster
# Or revert Cloudflare tunnel config

# Keep monitoring to ensure services recover
```

**Full Rollback (if needed):**
1. Restore Kind cluster from backups
2. Disable bare metal cluster
3. Point DNS back to Kind

---

## Post-Migration Tasks

### 8.1 Operational Setup

```bash
# Configure automated backups
kubectl apply -f k8s/resilience/backup-policy.yaml

# Set up log aggregation (ELK/Loki)
kubectl apply -f k8s/monitoring/logging/

# Configure alerting rules
kubectl apply -f k8s/monitoring/prometheus-rules.yaml
```

### 8.2 Documentation Update

Update your deployment documentation:
- [x] Record new cluster control plane IP
- [x] Update kubeconfig files
- [x] Document node roles and responsibilities
- [x] Update runbooks with new procedures
- [x] Train operations team

### 8.3 Performance Tuning

```bash
# Review resource usage
kubectl top nodes
kubectl top pods --all-namespaces

# Adjust resource requests/limits if needed
kubectl set resources deployment <name> -n <namespace> --requests=cpu=100m,memory=100Mi
```

### 8.4 Decommission Kind Cluster (when ready)

```bash
# After confirming bare metal is stable (2-3 weeks)
kind delete cluster --name dev-cluster

# Remove Kind from kubeconfig
kubectl config delete-context kind-dev-cluster
kubectl config delete-cluster kind-dev-cluster
```

---

## Troubleshooting

### Common Issues

#### Nodes Not Ready
```bash
# Check kubelet service
systemctl status kubelet

# View kubelet logs
journalctl -u kubelet -n 100

# Check node conditions
kubectl describe node <node-name>
```

#### Pods Not Starting
```bash
# Check pod events
kubectl describe pod <pod-name> -n <namespace>

# View pod logs
kubectl logs <pod-name> -n <namespace>

# Check resource availability
kubectl top nodes
kubectl describe nodes
```

#### Storage Issues
```bash
# Check PVC status
kubectl get pvc --all-namespaces

# View PV details
kubectl describe pv <pv-name>

# Check OpenEBS
kubectl get pods -n openebs
kubectl logs -n openebs <pod-name>
```

#### Network Connectivity
```bash
# Test DNS
kubectl run -it --rm debug --image=busybox --restart=Never \
  -- nslookup kubernetes.default

# Test service connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never \
  -- curl http://service-name:port

# Check NetworkPolicy
kubectl get networkpolicy --all-namespaces
```

### Support Resources

- Kubernetes Docs: https://kubernetes.io/docs/
- OpenEBS: https://openebs.io/docs/
- Troubleshooting: `kubectl get events --all-namespaces`
- Node logs: `journalctl -xeu kubelet` on each node

---

## Appendix: Configuration Reference

### Kubernetes Network Configuration
- Pod Network CIDR: `10.244.0.0/16`
- Service CIDR: `10.96.0.0/12`
- CNI Plugin: Flannel

### Storage Configuration
- Storage Provider: OpenEBS
- Default Storage Class: `local-storage-standard`
- Local Storage Path: `/mnt/openebs/local`

### Service Port Mappings
```
API Server: 6443
Kubelet API: 10250
kube-proxy: 10256
Flannel VXLAN: 8472/UDP
etcd: 2379-2380
NodePort Range: 30000-32767
```

### Required Disk Space per Node
```
OS + Kubelet: 20GB
Kubernetes system: 10GB
Application workloads: 30GB
Buffer: 10GB
---
Total: 70GB minimum (100GB recommended)
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-20 | 1.0 | Initial migration runbook created |

---

**Document Maintainer**: 254Carbon DevOps Team  
**Last Reviewed**: October 20, 2025  
**Next Review**: November 20, 2025
