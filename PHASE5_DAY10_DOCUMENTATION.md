# Phase 5: Day 10 - Operational Documentation & Team Training

**Status**: Ready for Team Handoff  
**Date**: October 25, 2025 (Simulated Day 10)  
**Duration**: 2-3 hours  
**Goal**: Complete operational documentation and enable team

---

## Overview

Day 10 delivers comprehensive documentation enabling teams to operate and develop on the 254Carbon platform independently.

---

## SECTION 1: OPERATIONAL RUNBOOKS

### 1.1 Daily Operations Checklist

**Every Morning (5 minutes)**

```bash
#!/bin/bash
# Daily health check

echo "=== 254Carbon Daily Health Check ==="
echo "$(date)"

# Check platform health
echo ""
echo "1. Platform Health:"
kubectl get nodes
kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory

# Check critical services
echo ""
echo "2. Critical Services:"
kubectl get deployment -n data-platform -o wide | grep -E "dolphinscheduler|trino"
kubectl get statefulset -n kafka -o wide
kubectl get pods -n data-platform | grep -E "CrashLoop|Pending"

# Check resource usage
echo ""
echo "3. Resource Utilization:"
kubectl top nodes
kubectl top pods -A --sort-by=memory | head -10

# Check PVC usage
echo ""
echo "4. Storage:"
kubectl get pvc -A | grep -v "STATUS"

# Alerts summary
echo ""
echo "5. Recent Alerts:"
kubectl logs -n monitoring prometheus-0 --tail=20 | grep -i "alert" | tail -5
```

### 1.2 Troubleshooting Guide

#### Issue: Pod in CrashLoopBackOff

**Symptoms**: Pod repeatedly crashes  
**Resolution**:

```bash
# 1. Check logs
POD_NAME="your-pod-name"
kubectl logs -n data-platform $POD_NAME --tail=50
kubectl logs -n data-platform $POD_NAME --previous  # Previous attempt

# 2. Check events
kubectl describe pod -n data-platform $POD_NAME | grep -A 20 "Events:"

# 3. Common fixes
# Insufficient memory: Increase resources
# DB connection: Verify database connectivity
# Config error: Check ConfigMap/Secret

# 4. Force restart
kubectl rollout restart deployment/dolphinscheduler-api -n data-platform
kubectl rollout status deployment/dolphinscheduler-api -n data-platform
```

#### Issue: Kafka High Latency

**Symptoms**: Message ingestion is slow  
**Resolution**:

```bash
# 1. Check broker health
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# 2. Check topic replication
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe --topic your-topic

# 3. Scale brokers if needed
kubectl scale statefulset datahub-kafka-kafka-pool -n kafka --replicas=4

# 4. Monitor producer metrics
kubectl logs -n data-platform etl-db-extract-template-xxxxx | grep "latency"
```

#### Issue: Trino Slow Queries

**Symptoms**: SQL queries take >10 seconds  
**Resolution**:

```bash
# 1. Access Trino UI
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080

# 2. Check running queries
# Open http://localhost:8080/ui/query.html

# 3. Analyze query plan
TRINO_POD=$(kubectl get pods -n data-platform -l app=trino-coordinator -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n data-platform $TRINO_POD -- \
  trino --execute "EXPLAIN ANALYZE SELECT ..."

# 4. Create indexes (if supported)
# Add partitioning to large tables
# Optimize join order

# 5. Scale workers if needed
kubectl scale deployment trino-worker -n data-platform --replicas=4
```

#### Issue: PostgreSQL Connection Pool Exhausted

**Symptoms**: "Too many connections" errors  
**Resolution**:

```bash
# 1. Check current connections
kubectl exec -n kong kong-postgres-0 -it -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# 2. Kill long-running queries
kubectl exec -n kong kong-postgres-0 -it -- \
  psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state='idle';"

# 3. Increase max connections
kubectl set env statefulset/kong-postgres \
  -n kong \
  POSTGRES_MAX_CONNECTIONS=300

# 4. Verify HikariCP settings in DolphinScheduler
kubectl get deployment dolphinscheduler-api -n data-platform -o yaml | grep HIKARI

# 5. Monitor pool usage
kubectl logs -n data-platform deployment/dolphinscheduler-api | grep -i "connection"
```

### 1.3 Emergency Response Playbooks

#### Emergency: Platform Down

**When**: Multiple critical services failing  
**Action Plan** (30 minutes to restore):

```bash
# Step 1: Assess (5 min)
kubectl get nodes
kubectl get pods -A --sort-by=.status.startTime | tail -20

# Step 2: Check logs (5 min)
kubectl logs -n kube-system -l component=kubelet --tail=50
kubectl describe nodes | grep -A 5 "Conditions"

# Step 3: Restart critical services (10 min)
kubectl rollout restart deployment/dolphinscheduler-api -n data-platform
kubectl rollout restart statefulset/datahub-kafka-kafka-pool -n kafka
kubectl rollout restart statefulset/kong-postgres -n kong

# Step 4: Verify recovery (10 min)
kubectl get pods -A | grep -E "CrashLoop|Pending"
kubectl logs -n data-platform deployment/dolphinscheduler-api --tail=10
```

#### Emergency: Data Loss Risk

**When**: Database unreachable  
**Action Plan** (Immediate):

```bash
# Step 1: Verify backup status
kubectl get backup -n velero

# Step 2: Check last backup time
kubectl describe backup -n velero $(kubectl get backup -n velero -o jsonpath='{.items[0].metadata.name}')

# Step 3: Restore if needed
kubectl create restore --from-backup my-backup-name -n velero

# Step 4: Verify restored data
kubectl exec -n kong kong-postgres-0 -it -- psql -U postgres -l
```

---

## SECTION 2: ARCHITECTURE DOCUMENTATION

### 2.1 Platform Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    254CARBON PLATFORM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      EXTERNAL DATA SOURCES                       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │ DB APIs  │ │S3/Cloud  │ │ REST API │ │ Files    │           │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │   │
│  │       └─────────────┴─────────────┴─────────────┘               │   │
│  │                         ↓                                         │   │
│  │                  [Network Policies]                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                           ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │          INGESTION LAYER (DolphinScheduler + Connectors)         │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │ DolphinScheduler (6/6 API, 1/2 Workers, 1/1 Master)      │ │  │
│  │  │ - Workflow orchestration                                  │ │  │
│  │  │ - ETL job execution                                       │ │  │
│  │  │ - Data quality checks                                     │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                           ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │           STREAM LAYER (Kafka 3-Broker Cluster)                 │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │ Kafka (3 brokers, KRaft mode)                             │ │  │
│  │  │ - Event streaming (commodities, market data)              │ │  │
│  │  │ - 7,153 rec/sec baseline throughput                       │ │  │
│  │  │ - 3x replication for HA                                   │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                           ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │            STORAGE LAYER (Iceberg + PostgreSQL)                 │  │
│  │  ┌────────────────────┐         ┌──────────────────┐           │  │
│  │  │ Trino + Iceberg    │         │ PostgreSQL       │           │  │
│  │  │ - SQL analytics    │         │ - Operational DB │           │  │
│  │  │ - Data lake        │         │ - Metadata store │           │  │
│  │  │ - Partitioned data │         │ - 50 conn pool   │           │  │
│  │  └────────────────────┘         └──────────────────┘           │  │
│  │  ┌────────────────────┐         ┌──────────────────┐           │  │
│  │  │ MinIO (S3)         │         │ Doris            │           │  │
│  │  │ - Data warehouse   │         │ - OLAP analytics │           │  │
│  │  │ - File storage     │         │ (optional)       │           │  │
│  │  └────────────────────┘         └──────────────────┘           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                           ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │         ANALYTICS LAYER (Trino + Superset + Ray)                │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │ Superset: BI dashboards                                 │  │  │
│  │  │ Trino: SQL query engine                                 │  │  │
│  │  │ Ray: Distributed ML (3 nodes)                           │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                           ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │      OBSERVABILITY LAYER (Grafana + VictoriaMetrics + Loki)     │  │
│  │  ├─ Grafana: Dashboards & alerts                               │  │
│  │  ├─ VictoriaMetrics: Metrics storage                           │  │
│  │  ├─ Loki: Log aggregation                                      │  │
│  │  └─ Prometheus: Metrics collection                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │            GOVERNANCE LAYER (DataHub + Kyverno)                 │  │
│  │  ├─ DataHub: Metadata catalog & lineage                         │  │
│  │  ├─ Kyverno: Policy enforcement                                 │  │
│  │  └─ RBAC: Access control                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Data Flow:
  External Source → Ingestion → Stream → Storage → Analytics → Dashboard → Alerts
```

### 2.2 Data Flow Examples

#### Example 1: Daily Commodity Price Pipeline

```
Time: 06:00 AM
1. DolphinScheduler triggers commodity-price-pipeline
2. API Connector fetches prices from commodity API
3. Data Quality Checker validates: nulls, ranges, duplicates
4. Messages published to Kafka topic: "commodities"
5. Kafka consumers read from 3 partitions in parallel
6. Trino loads data into Iceberg table (partitioned by date)
7. Superset refresh queries for dashboards
8. Grafana shows pipeline execution metrics
9. Alerts if:
   - API timeout
   - Data quality < 80%
   - Kafka lag > 5min
   - Trino query > 10sec
```

#### Example 2: Real-Time Analytics

```
Continuous Flow:
1. Producer sends market events to Kafka (100msg/sec baseline)
2. Multiple consumers:
   a) Trino reads → Real-time analytics table
   b) Ray processes → ML predictions
   c) Superset displays → Live dashboard
3. Grafana monitors:
   - Kafka throughput
   - Trino latency
   - Ray job status
4. Alerts trigger if performance degrades
```

---

## SECTION 3: DEVELOPER GUIDES

### 3.1 Creating Your First ETL Pipeline

**Step 1: Access DolphinScheduler**

```bash
# Open in browser
https://dolphin.254carbon.com

# Login with platform credentials
```

**Step 2: Create Workflow**

```bash
1. Click "Project Center"
2. Create new project: "my-etl-project"
3. Create workflow: "daily-data-load"
```

**Step 3: Add Tasks**

```bash
# Task 1: Extract from database
Type: Shell
Command:
#!/bin/bash
python3 << 'EOF'
import psycopg2
from kafka import KafkaProducer

# Extract data
conn = psycopg2.connect("dbname=source user=etl")
cur = conn.cursor()
cur.execute("SELECT * FROM products WHERE modified_date > NOW() - INTERVAL '1 day'")
data = cur.fetchall()

# Produce to Kafka
producer = KafkaProducer(bootstrap_servers=['kafka:9092'])
for row in data:
    producer.send('products', str(row).encode())
producer.flush()
EOF

# Task 2: Quality check
Type: Shell  
Command: /path/to/quality_checker.py

# Task 3: Load to warehouse
Type: Shell
Command: /path/to/load_to_iceberg.py
```

**Step 4: Set Schedule**

```bash
Cron: 0 2 * * * (2 AM daily)
Start: 2025-10-26
End: 2026-12-31
```

**Step 5: Run & Monitor**

```bash
1. Click "Run"
2. Monitor in "Execution History"
3. Check logs for errors
4. View metrics in Grafana
```

### 3.2 Querying the Data Lake

**Connect with Trino**

```bash
# Via Web UI
https://trino.254carbon.com

# Via CLI
kubectl exec -n data-platform trino-coordinator-xxx -it -- trino
```

**Sample Queries**

```sql
-- List all tables
SHOW TABLES FROM iceberg.default;

-- Query commodity prices
SELECT 
  commodity,
  DATE(timestamp) as date,
  COUNT(*) as record_count,
  AVG(price) as avg_price,
  MIN(price) as min_price,
  MAX(price) as max_price
FROM iceberg.default.commodity_prices
WHERE DATE(timestamp) >= DATE('2025-10-01')
GROUP BY commodity, DATE(timestamp)
ORDER BY date DESC, commodity;

-- Create materialized view
CREATE TABLE iceberg.default.daily_summary AS
SELECT 
  commodity,
  DATE(timestamp) as date,
  AVG(price) as avg_price
FROM iceberg.default.commodity_prices
GROUP BY commodity, DATE(timestamp);
```

### 3.3 Creating Superset Dashboards

**Step 1: Add Data Source**

```bash
1. Settings → Data Sources
2. Add Trino connection:
   - Engine: Trino
   - Host: trino-coordinator.data-platform
   - Port: 8080
   - Database: iceberg
```

**Step 2: Create Dataset**

```bash
1. Datasets → Create new
2. Select table: iceberg.default.commodity_prices
3. Configure columns and filters
```

**Step 3: Create Dashboard**

```bash
1. Dashboards → Create new
2. Add charts:
   - Line chart: Commodity prices over time
   - Bar chart: Average price by commodity
   - Table: Latest prices
3. Set refresh interval: 1 minute
```

---

## SECTION 4: TEAM TRAINING MATERIALS

### 4.1 Platform Walkthrough (30 min)

**What You Have**:
- Production-grade data platform
- 7,153 msg/sec streaming capacity
- SQL query engine with analytics
- ML compute cluster
- Automated workflows
- Enterprise monitoring

**How to Use It**:
1. Create workflows (DolphinScheduler)
2. Stream events (Kafka)
3. Query data (Trino)
4. Build dashboards (Superset)
5. Run ML jobs (Ray)
6. Monitor everything (Grafana)

**First Week Tasks**:
- [ ] Login to all web UIs
- [ ] Create sample workflow
- [ ] Run sample query
- [ ] Build test dashboard
- [ ] Set up Slack alerts

### 4.2 Access Control & Permissions

**Your Role Determines Access**

```
DataEngineer:
  ✓ Create/edit workflows
  ✓ Deploy data connectors
  ✓ View logs
  ✓ Manage secrets (limited)
  ✗ Modify platform config
  ✗ Delete resources

DataAnalyst:
  ✓ Query data (read-only)
  ✓ Create dashboards
  ✓ View logs
  ✗ Create workflows
  ✗ Modify data
  ✗ Access secrets

PlatformAdmin:
  ✓ All access
  ✓ Modify configurations
  ✓ Manage users
  ✓ Scale resources
```

### 4.3 Common Operations

**Create a Data Pipeline**

```bash
1. DolphinScheduler: Design workflow
2. Use templates: api-connector, db-extract-load
3. Configure credentials: Use platform secrets
4. Set schedule: Cron expression
5. Monitor: Grafana dashboards
6. Alert: Slack notifications
```

**Troubleshoot Issues**

```bash
# Check pod status
kubectl get pods -n data-platform

# View logs
kubectl logs -n data-platform <pod-name>

# Get events
kubectl describe pod -n data-platform <pod-name>

# Restart service
kubectl rollout restart deployment/<service> -n data-platform
```

---

## SECTION 5: SUPPORT ESCALATION

### Level 1: Self-Service

- Check documentation
- Review logs
- Test connectivity
- Restart pod

### Level 2: Team Support

- Slack: #data-platform
- Runbook: /docs/troubleshooting
- Pair programming session

### Level 3: Escalation

- Platform team lead
- Infrastructure team
- Emergency procedures

---

## Success Criteria Checklist

✅ Runbooks documented  
✅ Architecture explained  
✅ Developer guides created  
✅ Team trained  
✅ Support process defined  
✅ Dashboard created  
✅ Alerts configured  

---

## Additional Resources

- **Repository**: /home/m/tff/254CARBON/HMCo
- **Documentation**: docs/
- **Examples**: examples/
- **API Reference**: api-docs/
- **Video Tutorials**: (internal wiki)

---

**Day 10 Complete**: Platform ready for independent team operation

**Status**: ✅ PHASE 5 DAYS 6-10 COMPLETE - PRODUCTION-READY PLATFORM
