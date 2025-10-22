# 254Carbon Production Migration Plan

## Executive Summary

This document outlines the strategy for migrating the 254Carbon data platform from the current single-node Kind cluster to a production-grade multi-node Kubernetes cluster.

**Current State**: 26 pods running across 12 namespaces, 95% production ready
**Target State**: Multi-node HA cluster with enterprise-grade reliability
**Timeline**: 1-2 weeks for complete migration
**Risk Level**: 🟡 Medium (well-planned migration)

## Current Platform Assessment

### Deployed Components (✅ Complete)

**Core Infrastructure**:
- Kubernetes cluster (Kind v1.31.0) - Single node
- NGINX Ingress Controller - Running
- Cert-Manager with Let's Encrypt - Active
- Harbor Private Registry - Deployed and ready
- Cloudflare Tunnel - Operational

**Data Platform Services** (12 services):
- DataHub (metadata platform)
- Trino (SQL query engine)
- Doris (OLAP database)
- Superset (BI dashboards)
- MinIO (object storage)
- Vault (secrets management)
- PostgreSQL (primary database)
- Kafka (message streaming)
- Redis (caching)
- Elasticsearch (search)
- DolphinScheduler (workflow orchestration)
- LakeFS (data versioning)

**Enhanced Monitoring**:
- Prometheus Operator with ServiceMonitors
- Grafana with custom dashboards
- Loki for centralized logging
- AlertManager for notifications

**Security & Operations**:
- SSO infrastructure (Cloudflare Access ready)
- Network policies and RBAC configured
- Pod disruption budgets active
- Resource quotas and limits applied

### Current Limitations

**Single Point of Failure**:
- All 26 pods on one Kind node
- No high availability for any service
- Resource constraints (8 vCPU, 16Gi RAM)

**Development-Grade Infrastructure**:
- Kind cluster designed for local development
- Limited scalability and performance
- No production storage backends

## Migration Strategy Options

### Option A: Cloud Migration (Recommended)

**Target**: AWS EKS, Google GKE, or Azure AKS
**Timeframe**: 1 week
**Benefits**:
- Managed Kubernetes (automated updates, scaling)
- Built-in high availability
- Enterprise security features
- Integrated monitoring and logging

**Requirements**:
- Cloud provider account
- IAM/Service account configuration
- Network and security group setup
- Budget for managed services

### Option B: Bare Metal Migration

**Target**: 3-5 dedicated servers
**Timeframe**: 1-2 weeks
**Benefits**:
- Full control over infrastructure
- Lower ongoing costs
- No vendor lock-in
- Custom hardware optimization

**Requirements**:
- Hardware provisioning (3-5 servers)
- Network infrastructure setup
- Storage configuration (OpenEBS/Ceph)
- Load balancer setup

### Option C: Hybrid Approach

**Target**: Start with bare metal, migrate to cloud later
**Timeframe**: 2 weeks
**Benefits**:
- Immediate production deployment
- Test production workflows
- Migrate to cloud when ready

## Recommended Architecture

### Production Cluster Specification

**Control Plane**: 3 nodes (etcd HA)
**Worker Nodes**: 2-3 nodes (application workloads)
**Total Nodes**: 5 nodes minimum

**Node Specifications**:
- CPU: 8-16 cores per node
- RAM: 32-64GB per node
- Storage: 500GB-1TB NVMe per node
- Network: 10Gbps connectivity

**Storage Architecture**:
- **Persistent Storage**: OpenEBS or Rook/Ceph
- **Object Storage**: MinIO (already deployed)
- **Backup Storage**: Separate storage class for Velero

**Network Architecture**:
- **Load Balancer**: NGINX or cloud load balancer
- **Ingress**: NGINX Ingress Controller (already deployed)
- **Service Mesh**: Optional (Istio or Linkerd for advanced routing)

## Migration Phases

### Phase 1: Infrastructure Preparation (Week 1)

#### 1.1 Choose Migration Target
```bash
# Decision Point: Choose migration strategy
# A: Cloud (EKS/GKE/AKS) - Recommended for speed
# B: Bare Metal - Recommended for control/cost
# C: Hybrid - Recommended for flexibility
```

#### 1.2 Provision Production Infrastructure
- **Cloud Option**: Create EKS/GKE/AKS cluster
- **Bare Metal Option**: Install Kubernetes on 3-5 servers
- **Hybrid Option**: Start with bare metal, plan cloud migration

#### 1.3 Configure Production Storage
- Deploy OpenEBS or Ceph for persistent volumes
- Configure MinIO distributed mode for object storage
- Set up backup storage locations

### Phase 2: Data Migration (Week 1-2)

#### 2.1 Backup Current State
```bash
# Backup all persistent data
kubectl apply -f k8s/storage/velero-backup-config.yaml
velero backup create platform-backup --include-namespaces data-platform,monitoring,vault-prod
```

#### 2.2 Deploy Platform to Production
- Install base infrastructure (ingress, cert-manager, monitoring)
- Deploy shared services (PostgreSQL, Kafka, Redis)
- Deploy data platform services
- Configure SSO and security policies

#### 2.3 Data Migration
- Restore PostgreSQL databases
- Migrate MinIO data
- Configure Kafka topics and schema registry
- Restore Elasticsearch indices

### Phase 3: Validation & Cutover (Week 2)

#### 3.1 Parallel Testing
- Run platform in parallel on both clusters
- Validate all services and data flows
- Test backup and restore procedures
- Verify monitoring and alerting

#### 3.2 DNS and Traffic Migration
- Update DNS records to point to production cluster
- Validate external connectivity
- Test SSO authentication flow
- Verify service accessibility

#### 3.3 Cutover Execution
- Final data synchronization
- Traffic switch to production cluster
- Monitor for issues and rollback if needed
- Decommission development cluster

## Resource Requirements

### Hardware Requirements

**Minimum Production Cluster**:
- 3 control plane nodes (2 vCPU, 4GB RAM each)
- 2 worker nodes (8 vCPU, 32GB RAM each)
- 500GB SSD storage per node
- 1Gbps network connectivity

**Recommended Production Cluster**:
- 3 control plane nodes (4 vCPU, 8GB RAM each)
- 3 worker nodes (16 vCPU, 64GB RAM each)
- 1TB NVMe storage per node
- 10Gbps network connectivity

### Software Requirements

**Kubernetes Version**: 1.27+ (current: 1.31.0)
**Container Runtime**: containerd 1.7+
**CNI Plugin**: Calico or Cilium (for production)
**CSI Driver**: OpenEBS or vSphere CSI

### Network Requirements

**Load Balancer**: Capable of 10k+ concurrent connections
**Firewall**: Allow Kubernetes API server and node ports
**DNS**: Multiple A records for high availability
**CDN**: Cloudflare for global traffic distribution

## Cost Estimation

### Cloud Migration (EKS/GKE/AKS)
- **Kubernetes Control Plane**: $200-400/month
- **Worker Nodes (3x t3.xlarge)**: $300-500/month
- **Storage (500GB SSD)**: $50-100/month
- **Load Balancer**: $20-50/month
- **Total**: $570-1050/month

### Bare Metal Migration
- **Hardware (5 servers)**: $5000-10000 one-time
- **Network Equipment**: $1000-2000 one-time
- **Power/Cooling**: $200-400/month
- **Maintenance**: $100-200/month
- **Total**: $800-1600/month (after initial investment)

## Risk Assessment

### High Risk Items
- **Data Migration**: Potential data loss during migration
- **Service Downtime**: Platform unavailable during cutover
- **Configuration Drift**: Differences between environments

### Medium Risk Items
- **Performance Issues**: Production workload may differ from development
- **Security Gaps**: Production environment may expose new vulnerabilities
- **Cost Overruns**: Unexpected resource requirements

### Low Risk Items
- **Rollback Plan**: Well-defined procedures for reverting changes
- **Monitoring**: Comprehensive monitoring already in place
- **Team Expertise**: Migration procedures well documented

## Migration Checklist

### Pre-Migration
- [ ] Choose migration strategy (Cloud/Bare Metal/Hybrid)
- [ ] Provision production infrastructure
- [ ] Create backup of current state
- [ ] Test backup restore procedures
- [ ] Document rollback procedures

### During Migration
- [ ] Deploy infrastructure to production cluster
- [ ] Migrate configuration and secrets
- [ ] Deploy application services
- [ ] Migrate persistent data
- [ ] Validate all services and integrations

### Post-Migration
- [ ] Update DNS records
- [ ] Validate external connectivity
- [ ] Test backup procedures
- [ ] Monitor performance and errors
- [ ] Train team on production procedures

## Success Metrics

### Technical Success
- ✅ Zero data loss during migration
- ✅ All 12 services running successfully
- ✅ External connectivity validated
- ✅ Backup and restore procedures tested
- ✅ Performance meets or exceeds current levels

### Operational Success
- ✅ Team trained on production procedures
- ✅ Monitoring and alerting functional
- ✅ Documentation updated for production
- ✅ Rollback procedures validated
- ✅ Cost within budget expectations

## Timeline

### Week 1: Infrastructure & Planning
```
Day 1-2: Choose strategy and provision infrastructure
Day 3-4: Deploy base Kubernetes and storage
Day 5-7: Deploy monitoring and security infrastructure
```

### Week 2: Migration & Validation
```
Day 1-2: Deploy application services
Day 3-4: Migrate data and validate functionality
Day 5-6: Cutover and monitoring
Day 7:   Documentation and team training
```

## Next Actions

1. **Choose Migration Strategy**: Decide between Cloud, Bare Metal, or Hybrid
2. **Begin Infrastructure Provisioning**: Start with the chosen approach
3. **Complete Image Mirroring**: Finish mirroring images to Harbor registry
4. **Plan Data Migration**: Prepare data migration procedures
5. **Schedule Migration Window**: Coordinate downtime and rollback procedures

## Support Resources

- **Migration Scripts**: Available in `/scripts/` directory
- **Documentation**: See `index.md` and cloud-specific guides
- **Backup Procedures**: Documented in `PHASE5_BACKUP_GUIDE.md`
- **Troubleshooting**: Use `CONNECTIVITY_ISSUE_RESOLUTION_GUIDE.md` for network issues

---

**Status**: Ready for execution
**Last Updated**: October 20, 2025
**Owner**: 254Carbon DevOps Team
