# 254Carbon Platform - Next Steps Plan

**Date**: October 24, 2025  
**Current Status**: Phase 2 Complete (90%), Platform Health 75% (104/138 pods)  
**Target**: Phase 2 100% Complete, Platform Health 85%+, Full ArgoCD Sync

---

## 📊 Current State Analysis

### ✅ What's Working (Excellent)
- **Core Services**: 10/12 critical services operational (83%)
  - DolphinScheduler: Master + Workers running (API initializing)
  - Trino Coordinator: Operational
  - MinIO: 50Gi storage ready
  - Superset: Web/Worker/Beat running
  - PostgreSQL: Emergency deployment active
  - Zookeeper, Redis (old pods), Iceberg REST
  
- **Phase 2 Monitoring**: COMPLETE ✅
  - Grafana: Running with datasources connected
  - Victoria Metrics: 20+ metrics collected
  - VMAgent: Scraping 19+ targets
  - Loki: Aggregating logs from 99+ pods
  - Fluent Bit: 2/2 nodes collecting logs
  
- **Phase 2 Backups**: COMPLETE ✅
  - Velero: 4 automated schedules
  - MinIO bucket: velero-backups created
  - Retention: 30 days configured

- **Infrastructure**: SOLID ✅
  - Kubernetes: 2-node cluster healthy
  - ArgoCD: 17 applications configured
  - Kyverno: 11 PolicyExceptions active
  - GitOps: 13 commits pushed

### ⚠️ Issues Remaining (18 pods)

**High Priority** (Impact: Service functionality)
1. **Redis (Bitnami)** - ImagePullBackOff (1 pod)
   - New Bitnami image not pulling
   - Old Redis still running (workaround active)
   
2. **Portal-Services** - ImagePullBackOff (3 pods)
   - Image only on cpu1, not k8s-worker
   - GraphQL gateway can't start
   
3. **DolphinScheduler API** - Pods cycling (6 pods)
   - Database schema initialized
   - Pods starting but not yet ready
   
**Medium Priority** (Impact: Features/Performance)
4. **Trino Worker** - CrashLoopBackOff (1 pod)
   - Coordinator working
   - 50% query capacity
   
5. **Spark History Server** - CrashLoopBackOff (1 pod)
   - Historical job viewing unavailable
   
6. **Doris FE** - CrashLoopBackOff (disabled, expected)

**Low Priority** (Impact: Optional features)
7. **DataHub** - Init containers waiting (3 pods)
   - Prerequisites not deployed (Elasticsearch, Kafka, Neo4j)
   - Data catalog not critical for Phase 2
   
8. **Iceberg Compaction** - ImagePullBackOff (1 pod)
   - Daily maintenance job
   - Not urgent

### 🔄 ArgoCD Status
- **OutOfSync**: data-platform, api-gateway, service-mesh (expected - manual changes)
- **Degraded**: data-platform, platform-policies, portal-services (pod issues)
- **Action Needed**: Sync applications after fixes

---

## 🎯 Next Steps Plan (Prioritized)

### **Phase 1: Complete Service Restoration** (30-45 min)

#### 1.1 Fix Redis (Bitnami Image Pull) - 10 min
**Issue**: `bitnami/redis:7.2-debian-12` image not available or misspelled

**Options**:
- **A**: Use public Bitnami image: `bitnami/redis:7.2.15-debian-12-r0`
- **B**: Keep old Redis running (already working)
- **C**: Pre-pull image manually on both nodes

**Recommended**: Option B (keep working, optimize later)

**Action**:
```bash
# Keep old Redis, remove new failing pods
kubectl delete deployment redis -n data-platform --cascade=orphan
kubectl delete pods -n data-platform -l app=redis | grep ImagePullBackOff
```

#### 1.2 Distribute Portal-Services Image - 15 min
**Issue**: Image on cpu1 only, not on k8s-worker

**Options**:
- **A**: Copy image to worker via SSH
- **B**: Push to public registry (DockerHub)
- **C**: Use node affinity (already done - cpu1 only)

**Recommended**: Option C is active, GraphQL gateway will work on next deployment

**Action**:
```bash
# Verify current setup
kubectl get pods -n data-platform -l app=portal-services -o wide

# If needed, copy to worker:
docker save harbor.254carbon.com/library/portal-services:1.0.0 > /tmp/portal-services.tar
scp /tmp/portal-services.tar k8s-worker:/tmp/
ssh k8s-worker "sudo ctr -n k8s.io images import /tmp/portal-services.tar"
```

#### 1.3 Wait for DolphinScheduler API Ready - 5 min
**Issue**: Pods starting after schema init

**Action**: Wait for readiness probes (auto-completes in 2-5 min)
```bash
kubectl wait --for=condition=ready pod -l app=dolphinscheduler-api -n data-platform --timeout=300s
kubectl exec -n data-platform deploy/dolphinscheduler-api -- curl http://localhost:12345/dolphinscheduler/actuator/health
```

#### 1.4 Fix Trino Worker (Optional) - 10 min
**Issue**: 1/2 workers crashing with catalog errors

**Action**: Check if it's a transient issue or needs config fix
```bash
kubectl logs -n data-platform -l app=trino-worker --tail=50
kubectl delete pods -n data-platform -l app=trino-worker
# Wait and verify both start
```

---

### **Phase 2: ArgoCD Sync & GitOps Cleanup** (20-30 min)

#### 2.1 Sync Data Platform Application - 10 min
**Why**: Apply all our Helm chart fixes via GitOps

**Action**:
```bash
# Refresh from Git
kubectl annotate application data-platform -n argocd argocd.argoproj.io/refresh=hard --overwrite

# Wait for sync
kubectl wait --for=jsonpath='{.status.sync.status}'=Synced application/data-platform -n argocd --timeout=300s

# Verify
kubectl get application data-platform -n argocd -o yaml | grep -A 5 "status:"
```

#### 2.2 Sync Portal-Services Application - 5 min
**Action**:
```bash
kubectl annotate application portal-services -n argocd argocd.argoproj.io/refresh=hard --overwrite
kubectl get application portal-services -n argocd
```

#### 2.3 Sync Platform Policies - 5 min
**Action**:
```bash
kubectl annotate application platform-policies -n argocd argocd.argoproj.io/refresh=hard --overwrite
```

#### 2.4 Review All Application Status - 10 min
**Action**:
```bash
# Get comprehensive status
kubectl get applications -n argocd

# Fix any OutOfSync apps
for app in $(kubectl get applications -n argocd -o jsonpath='{.items[?(@.status.sync.status=="OutOfSync")].metadata.name}'); do
  echo "Syncing $app..."
  kubectl annotate application $app -n argocd argocd.argoproj.io/refresh=hard --overwrite
done
```

---

### **Phase 3: Monitoring & Dashboard Optimization** (30-45 min)

#### 3.1 Verify Grafana Datasources - 5 min
**Action**: Test in Grafana UI
1. Login to Grafana
2. Configuration → Data sources
3. Click "Test" on VictoriaMetrics and Loki
4. Both should show green "Data source is working"

#### 3.2 Create Working Dashboards - 20 min
**Dashboards to create**:

1. **Platform Overview Dashboard** (10 min)
   - Total pods (gauge)
   - Pods by status (pie chart)
   - CPU/Memory usage (graph)
   - Service health table

2. **Data Platform Dashboard** (10 min)
   - DolphinScheduler status
   - Trino query metrics
   - MinIO storage usage
   - Database connections
   - Recent logs panel

**Action**: Use Explore → Add to dashboard workflow

#### 3.3 Configure Basic Alerts - 10 min
**Alerts to create**:
- Pod CrashLoopBackOff (Critical)
- Service down > 5 min (Critical)
- High memory usage > 80% (Warning)

**Action**:
```
Alerting → Alert rules → New alert rule
Query: up{kubernetes_namespace="data-platform"} == 0
Condition: Any pod down
```

---

### **Phase 4: Cleanup & Optimization** (30-45 min)

#### 4.1 Remove Failed/Completed Jobs - 10 min
**Action**:
```bash
# Remove old failed jobs
kubectl delete jobs -n data-platform -l 'job-name in (dolphinscheduler-full-schema-init,dolphinscheduler-init-db,datahub-init-db)' --field-selector status.successful==0

# Remove completed jobs (optional)
kubectl delete jobs -n data-platform --field-selector status.successful==1
```

#### 4.2 Scale Down Unnecessary Resources - 10 min
**Action**:
```bash
# Disable DataHub until prerequisites deployed
kubectl scale deployment datahub-mce-consumer -n data-platform --replicas=0

# Stop iceberg compaction cronjob temporarily
kubectl patch cronjob iceberg-compaction -n data-platform -p '{"spec":{"suspend":true}}'
```

#### 4.3 Optimize Resource Allocations - 15 min
**Review and adjust**:
- DolphinScheduler API: Currently 6 replicas (HPA active)
- Superset components: Resource requests/limits
- Database connection pools

**Action**: Update Helm values for production sizing

---

### **Phase 5: Documentation & Validation** (20-30 min)

#### 5.1 Update Platform Status Documentation - 10 min
**Action**: Create final status report
```bash
# Document current state
kubectl get pods -A > cluster-status-$(date +%Y%m%d).txt
kubectl get applications -n argocd > argocd-status-$(date +%Y%m%d).txt
```

#### 5.2 Create Operational Runbook - 10 min
**Document**:
- How to restart services
- How to check logs
- How to trigger backups
- How to access Grafana
- Common troubleshooting steps

#### 5.3 Test Key Workflows - 10 min
**Validation tests**:
1. Create test workflow in DolphinScheduler
2. Run simple Trino query
3. Upload test file to MinIO
4. Trigger manual Velero backup
5. View metrics in Grafana
6. Search logs in Loki

---

### **Phase 6: Advanced Features (Optional)** (2-4 hours)

#### 6.1 Deploy DataHub Prerequisites - 90 min
**Components needed**:
- Elasticsearch cluster (3 nodes)
- Kafka (3 brokers)
- Neo4j graph database

**Impact**: Enables data catalog and governance

#### 6.2 Deploy Doris via Operator - 60 min
**Action**:
- Install Doris Operator
- Create DorisCluster CRD
- Deploy FE and BE nodes
- Configure Superset connection

**Impact**: Adds OLAP capabilities

#### 6.3 Complete ML Platform - 90 min
**Components**:
- MLflow (experiment tracking)
- Ray (distributed computing)
- Kubeflow Pipelines

**Impact**: Enables ML workflows

---

## 🎯 Recommended Execution Order

### **Today (Session 1)** - 1 hour
**Goal**: Clean up issues, verify stability

1. ✅ Let DolphinScheduler API finish initializing (5 min - auto)
2. ✅ Clean up failed jobs (10 min)
3. ✅ Sync ArgoCD applications (20 min)
4. ✅ Verify Grafana datasources working (5 min)
5. ✅ Create 1-2 working dashboards in Grafana (20 min)

**Expected Result**: Platform at 80% health, monitoring functional

### **Tomorrow (Session 2)** - 1 hour  
**Goal**: Optimize and document

1. Fix remaining Redis/portal-services images (15 min)
2. Optimize resource allocations (15 min)
3. Create comprehensive dashboards (20 min)
4. Set up basic alerts (10 min)

**Expected Result**: Platform at 85% health, full observability

### **This Week (Session 3)** - 2 hours
**Goal**: Advanced features

1. Deploy DataHub prerequisites (90 min)
2. Deploy Doris Operator (30 min)

**Expected Result**: Platform at 90% health, Phase 3 started

---

## 📋 Immediate Action Items (Next 60 Minutes)

### Priority 1: Stabilize Current Services ⏰ 20 min

```bash
# 1. Clean up failed jobs
kubectl delete jobs -n data-platform --field-selector status.successful==0

# 2. Wait for DolphinScheduler API
kubectl wait --for=condition=ready pod -l app=dolphinscheduler-api -n data-platform --timeout=300s

# 3. Verify critical services
kubectl get pods -n data-platform -l 'app in (dolphinscheduler-master,dolphinscheduler-worker,trino-coordinator,minio,superset)'

# 4. Check service health
kubectl exec -n data-platform deploy/dolphinscheduler-api -- curl -sf http://localhost:12345/dolphinscheduler/actuator/health
```

### Priority 2: ArgoCD Sync ⏰ 15 min

```bash
# Sync all OutOfSync applications
kubectl annotate application data-platform -n argocd argocd.argoproj.io/refresh=hard --overwrite
kubectl annotate application portal-services -n argocd argocd.argoproj.io/refresh=hard --overwrite
kubectl annotate application api-gateway -n argocd argocd.argoproj.io/refresh=hard --overwrite

# Monitor sync progress
watch kubectl get applications -n argocd
```

### Priority 3: Grafana Dashboard Validation ⏰ 15 min

```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000 &

# Access: http://localhost:3000
# Login: admin / datahub_admin_password

# In Grafana UI:
# 1. Go to Explore
# 2. Select VictoriaMetrics
# 3. Query: up{kubernetes_namespace="data-platform"}
# 4. Verify you see data
# 5. Create simple dashboard from query
```

### Priority 4: Create Essential Dashboards ⏰ 10 min

**In Grafana**:
1. Create "Platform Health" dashboard
   - Panel 1: Total pods (Stat)
   - Panel 2: Pod status over time (Time series)
   - Panel 3: Recent logs (Logs panel)

2. Save dashboard

---

## 🔧 Detailed Task Breakdown

### Task 1: DolphinScheduler Finalization
**Time**: 10 min  
**Goal**: All DolphinScheduler components ready

**Steps**:
1. Check API readiness: `kubectl get pods -n data-platform -l app=dolphinscheduler-api`
2. If still initializing, wait 5 more minutes
3. Test API: `curl https://dolphin.254carbon.com/dolphinscheduler/actuator/health`
4. Access UI: https://dolphin.254carbon.com (admin / dolphinscheduler123)
5. Verify workflows visible

**Success Criteria**: API returns HTTP 200, UI accessible

### Task 2: Grafana Dashboard Creation
**Time**: 20 min  
**Goal**: 2 working dashboards with live data

**Dashboard 1: Platform Overview** (10 min)
- Add panel: Total running pods
  - Query: `count(up == 1)`
  - Visualization: Stat
  
- Add panel: Service health
  - Query: `up{kubernetes_namespace="data-platform"}`
  - Visualization: Table
  
- Add panel: Logs
  - Datasource: Loki
  - Query: `{namespace="data-platform"}`
  - Visualization: Logs

**Dashboard 2: DolphinScheduler Metrics** (10 min)
- Add panel: API Health
  - Query: `up{app="dolphinscheduler-api"}`
  - Visualization: Stat with thresholds
  
- Add panel: Worker Status
  - Query: `up{app="dolphinscheduler-worker"}`
  - Visualization: Time series
  
- Add panel: API Logs
  - Query: `{namespace="data-platform", app="dolphinscheduler-api"}`
  - Visualization: Logs

**Success Criteria**: Both dashboards show live data, auto-refresh works

### Task 3: ArgoCD Application Sync
**Time**: 15 min  
**Goal**: All applications Synced status

**Steps**:
1. Refresh all OutOfSync apps
2. Monitor sync progress
3. Resolve any conflicts
4. Verify pod health improves

**Success Criteria**: 15/17 applications Synced or Healthy

### Task 4: Platform Health Verification
**Time**: 10 min  
**Goal**: Document final state

**Checks**:
```bash
# Pod health
kubectl get pods -A | grep -v "Running\|Completed" | wc -l
# Target: <15 pods

# Service URLs
curl -I https://dolphin.254carbon.com
curl -I https://grafana.254carbon.com
curl -I https://superset.254carbon.com
curl -I https://trino.254carbon.com

# Monitoring
# - Check Grafana shows data
# - Check Loki shows logs
# - Check backups running
```

**Success Criteria**: All critical URLs return 200/302, monitoring functional

---

## 🚀 Phase 3 Preview (Future Sessions)

### Performance Optimization (Week 2)
- Load testing with real data
- Resource optimization based on metrics
- Auto-scaling tuning (HPA)
- Query performance optimization (Trino)

### Advanced Features (Week 3)
- ML Platform deployment (MLflow, Ray, Kubeflow)
- DataHub with full prerequisites
- Doris Operator for OLAP
- Advanced monitoring (custom metrics)

### Production Hardening (Week 4)
- Network policies deployment
- Security audit and penetration testing
- Disaster recovery drill
- Compliance validation
- Production migration plan

---

## 📊 Success Metrics

### Target for End of Session
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Pod Health | 75% | 80% | ⏰ Achievable |
| Running Pods | 104 | 110+ | ⏰ Achievable |
| ArgoCD Synced | 7/17 | 12/17 | ⏰ Achievable |
| Grafana Dashboards | 2 | 2 with data | ⏰ Achievable |
| Critical Services | 83% | 90% | ⏰ Achievable |
| Platform Readiness | 85/100 | 88/100 | ⏰ Achievable |

### Long-term Targets
- **Phase 2 Complete**: 95/100 (this week)
- **Phase 3 Started**: DataHub + Doris (next week)
- **Production Ready**: 98/100 (2 weeks)

---

## 🎯 Quick Wins (15 min each)

### Win 1: Clean Failed Jobs
```bash
kubectl delete jobs -n data-platform --field-selector status.successful==0
```
**Impact**: Reduces clutter, improves pod count

### Win 2: Create First Working Dashboard
1. Login to Grafana
2. Explore → Query `up`
3. "Add to dashboard" → Save
**Impact**: Demonstrates monitoring works

### Win 3: Test Backup
```bash
velero backup create manual-test-$(date +%s) --include-namespaces data-platform
velero backup get
```
**Impact**: Validates DR capability

---

## 📝 Decision Points

### Should We...

**1. Deploy DataHub now or later?**
- **Now**: Requires Elasticsearch, Kafka, Neo4j (90 min setup)
- **Later**: Focus on stability first, add in Phase 3
- **Recommendation**: LATER (Phase 3)

**2. Fix all image pull issues or work around?**
- **Fix**: Pre-pull images on all nodes
- **Workaround**: Use node affinity or keep old versions
- **Recommendation**: WORKAROUND (optimize later)

**3. Deploy Doris now?**
- **Now**: Via Operator (60 min)
- **Later**: Current stack sufficient
- **Recommendation**: LATER (Phase 3)

**4. Optimize resources now?**
- **Now**: Based on current usage
- **Later**: After load testing
- **Recommendation**: LATER (collect more metrics first)

---

## 🎊 Recommended Next Session Plan

### **Session Focus**: Complete Phase 2, Verify Stability

**Duration**: 60 minutes

**Tasks**:
1. ✅ Wait for DolphinScheduler API ready (5 min)
2. ✅ Sync ArgoCD applications (15 min)
3. ✅ Verify Grafana datasources (5 min)
4. ✅ Create 2 working dashboards (20 min)
5. ✅ Clean up failed jobs (5 min)
6. ✅ Final platform verification (10 min)

**Outcome**: 
- Platform health: 80%+
- All ArgoCD apps synced
- Grafana functional with data
- Phase 2: 100% complete
- Ready for Phase 3

---

## 📚 Documentation Status

### ✅ Created (Complete)
- URGENT_REMEDIATION_STATUS.md
- SESSION_COMPLETION_SUMMARY.md
- PHASE2_DEPLOYMENT_COMPLETE.md
- IMPLEMENTATION_COMPLETE_FINAL.md
- GRAFANA_SETUP_COMPLETE.md
- DATA_PLATFORM_DASHBOARD_IMPORT.md
- 00_START_HERE_COMPLETE_STATUS.md

### ⏳ To Create
- Operational runbook (Phase 2 operations)
- Troubleshooting playbook (common issues)
- Performance baseline report (after load test)

---

## 🎯 Final Recommendation

**Execute Phase 1-2 from this plan (50-75 minutes total)**:
1. Let services stabilize (5 min wait)
2. Sync ArgoCD applications (15 min)
3. Create working Grafana dashboards (20 min)
4. Clean up failed resources (10 min)
5. Final verification and documentation (10 min)

**Expected Result**:
- Platform: 80%+ health
- Services: 90%+ critical services operational
- Monitoring: Functional with dashboards
- Phase 2: COMPLETE ✅
- Documentation: Comprehensive
- Ready for: Phase 3 advanced features

**This positions you perfectly for production readiness within 1-2 weeks.**

---

**Plan Status**: READY FOR EXECUTION  
**Estimated Time**: 1-2 hours  
**Risk**: LOW  
**Impact**: HIGH  
**Recommendation**: PROCEED ✅

