# 254Carbon Platform - Quick Start Guide

**Created**: October 24, 2025 05:05 UTC  
**Platform Health**: 77.5% (114/147 pods)  
**Status**: READY FOR USE ✅

---

## 🚀 **5-Minute Quick Start**

### **What's Available RIGHT NOW**

✅ **Workflow Orchestration** - DolphinScheduler (13 pods operational)  
✅ **SQL Analytics** - Trino (coordinator + worker)  
✅ **Business Intelligence** - Superset (3 pods)  
✅ **Monitoring** - Grafana with dashboards  
✅ **Event Streaming** - Kafka (3 brokers)  
✅ **Distributed Computing** - Ray cluster (3 nodes)  
✅ **Object Storage** - MinIO (50Gi)  
✅ **Data Catalog** - DataHub (frontend ready)  

---

## 📖 **Service Access Guide**

### **1. Grafana - Platform Monitoring** 🔥 **START HERE**

**Access**: https://grafana.254carbon.com  
**Login**: `admin` / `grafana123`

**What to do**:
```bash
# Option A: Via browser
open https://grafana.254carbon.com

# Option B: Port-forward
kubectl port-forward -n monitoring svc/grafana 3000:3000 &
open http://localhost:3000
```

**Once in Grafana**:
1. **Explore** → Select "VictoriaMetrics" datasource
2. **Query**: `up{kubernetes_namespace="data-platform"}`
3. **See live metrics** from all data platform services
4. **Dashboards** → Browse existing dashboards:
   - "Data Platform - Live Metrics & Logs"
   - "Data Platform Overview"

**Try This Query**:
```promql
# See all running services
up == 1

# CPU usage
container_cpu_usage_seconds_total

# Memory usage
container_memory_working_set_bytes
```

---

### **2. DolphinScheduler - Workflow Orchestration** 🔥 **HIGHLY RECOMMENDED**

**Access**: https://dolphin.254carbon.com  
**Login**: `admin` / `dolphinscheduler123`

**What to do**:
```bash
open https://dolphin.254carbon.com
```

**Create Your First Workflow**:

1. **Login** with admin credentials
2. **Project Management** → Create new project: `test-project`
3. **Workflow Definition** → Create workflow → Drag-and-drop editor
4. **Add Task** → Shell task:
   ```bash
   echo "Hello from DolphinScheduler!"
   date
   hostname
   ```
5. **Save** → **Run** → **Monitor execution**

**Available Workers**: 6 workers ready to execute tasks

**Advanced Example** - Spark Task:
```python
# Submit Spark job via DolphinScheduler
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DolphinScheduler Test") \
    .getOrCreate()

df = spark.range(1000)
print(f"Generated {df.count()} rows")
spark.stop()
```

---

### **3. Trino - SQL Analytics** 🔥 **POWERFUL**

**Access**: https://trino.254carbon.com

**What to do**:
```bash
# Option A: Via browser
open https://trino.254carbon.com

# Option B: Port-forward and use CLI
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080 &

# Install Trino CLI (if not installed)
# wget https://repo1.maven.org/maven2/io/trino/trino-cli/436/trino-cli-436-executable.jar
# chmod +x trino-cli-436-executable.jar
# ./trino-cli-436-executable.jar --server http://localhost:8080
```

**Example Queries**:
```sql
-- List available catalogs
SHOW CATALOGS;

-- Explore Iceberg catalog
SHOW SCHEMAS IN iceberg;

-- Create a test table
CREATE TABLE iceberg.test.sample_data (
  id INT,
  name VARCHAR,
  created_at TIMESTAMP
);

-- Insert test data
INSERT INTO iceberg.test.sample_data VALUES 
  (1, 'Test Record 1', CURRENT_TIMESTAMP),
  (2, 'Test Record 2', CURRENT_TIMESTAMP);

-- Query the data
SELECT * FROM iceberg.test.sample_data;

-- Check table properties
SHOW CREATE TABLE iceberg.test.sample_data;
```

**Data Sources Available**:
- Iceberg catalog (MinIO-backed)
- System catalog
- Additional catalogs can be configured

---

### **4. Superset - Business Intelligence** 📊

**Access**: https://superset.254carbon.com

**What to do**:
```bash
open https://superset.254carbon.com
```

**Create Your First Dashboard**:

1. **Login** (credentials configured during deployment)
2. **Data** → **Databases** → Add Trino connection:
   ```
   Name: Trino Analytics
   SQLAlchemy URI: trino://trino-coordinator.data-platform:8080/iceberg
   ```
3. **SQL Lab** → Run query:
   ```sql
   SELECT * FROM iceberg.test.sample_data;
   ```
4. **Create Chart** → Choose visualization
5. **Add to Dashboard** → Create new dashboard

**Chart Types Available**:
- Time series
- Bar charts
- Pie charts
- Tables
- Maps
- And many more...

---

### **5. Ray - Distributed Computing** ⚡ **ADVANCED**

**Ray Cluster**: 1 head + 2 workers (10 CPU cores)

**Access Dashboard**:
```bash
kubectl port-forward -n ml-platform svc/ml-cluster-head-svc 8265:8265 &
open http://localhost:8265
```

**Submit a Distributed Job**:

```python
# Method 1: From local machine (requires ray installed)
import ray

# Connect to cluster
ray.init("ray://localhost:10001")  # After port-forward

# Define distributed function
@ray.remote
def monte_carlo_pi(num_samples):
    import random
    inside = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

# Run distributed computation
num_trials = 100
samples_per_trial = 1000000
futures = [monte_carlo_pi.remote(samples_per_trial) for _ in range(num_trials)]
results = ray.get(futures)
pi_estimate = 4 * sum(results) / (num_trials * samples_per_trial)
print(f"Pi estimate: {pi_estimate}")

ray.shutdown()
```

**Method 2: Submit via kubectl**:
```bash
# Create job pod
kubectl run ray-job -n ml-platform --rm -it --restart=Never \
  --image=rayproject/ray:2.9.0 -- python3 -c "
import ray
ray.init('ray://ml-cluster-head-svc:10001')
print('Connected to Ray cluster')
print(f'Available resources: {ray.available_resources()}')
ray.shutdown()
"
```

**What You'll See in Dashboard**:
- Active workers
- Resource usage (CPU, memory)
- Running tasks
- Task timeline
- Cluster metrics

---

### **6. Kafka - Event Streaming** 📡

**Kafka Cluster**: 3 brokers (KRaft mode)  
**Bootstrap Server**: `datahub-kafka-kafka-bootstrap.kafka.svc.cluster.local:9092`

**Create a Topic**:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: events-stream
  namespace: kafka
  labels:
    strimzi.io/cluster: datahub-kafka
spec:
  partitions: 6
  replicas: 3
  config:
    retention.ms: 604800000
    segment.bytes: 1073741824
EOF
```

**Produce Messages**:
```bash
kubectl run kafka-producer -n kafka --rm -it --restart=Never \
  --image=quay.io/strimzi/kafka:latest-kafka-4.0.0 -- \
  bin/kafka-console-producer.sh \
  --bootstrap-server datahub-kafka-kafka-bootstrap:9092 \
  --topic events-stream

# Type messages, press Enter after each
# Ctrl+C when done
```

**Consume Messages**:
```bash
kubectl run kafka-consumer -n kafka --rm -it --restart=Never \
  --image=quay.io/strimzi/kafka:latest-kafka-4.0.0 -- \
  bin/kafka-console-consumer.sh \
  --bootstrap-server datahub-kafka-kafka-bootstrap:9092 \
  --topic events-stream \
  --from-beginning
```

**List Topics**:
```bash
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

---

### **7. MinIO - Object Storage** 💾

**Service**: `minio-service.data-platform:9000`  
**Storage**: 50Gi available

**Access via CLI**:
```bash
# Get credentials
MINIO_ACCESS=$(kubectl get secret minio-secret -n data-platform -o jsonpath='{.data.access-key}' | base64 -d)
MINIO_SECRET=$(kubectl get secret minio-secret -n data-platform -o jsonpath='{.data.secret-key}' | base64 -d)

# Port-forward
kubectl port-forward -n data-platform svc/minio-service 9000:9000 &

# Use mc (MinIO client) or AWS CLI
mc alias set myminio http://localhost:9000 $MINIO_ACCESS $MINIO_SECRET

# List buckets
mc ls myminio/

# Upload file
echo "test data" > test.txt
mc cp test.txt myminio/data-lake/test.txt

# List objects
mc ls myminio/data-lake/
```

**Buckets Available**:
- `data-lake` - Primary data storage
- `iceberg-warehouse` - Iceberg tables
- `velero-backups` - Backup storage

---

### **8. DataHub - Data Catalog** 📚

**Access**: https://datahub.254carbon.com  
**Status**: Frontend ready, GMS initializing

**What to do** (when GMS ready):
```bash
# Check GMS status
kubectl get pod -l app=datahub-gms -n data-platform

# When ready, access UI
open https://datahub.254carbon.com
```

**Metadata Ingestion** (programmatic):
```python
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter

emitter = DatahubRestEmitter("http://datahub-gms.data-platform:8080")

# Create dataset metadata
dataset_urn = make_dataset_urn("trino", "iceberg.test.sample_data")
# ... add metadata and emit
```

**Browse Metadata**:
- Datasets
- Schemas
- Lineage graphs
- Data quality metrics
- Usage statistics

---

## 🔥 **RECOMMENDED WORKFLOWS TO TRY**

### **Workflow 1: End-to-End Data Pipeline** (15 min)

**Goal**: Ingest → Process → Query → Visualize

```bash
# Step 1: Upload data to MinIO
kubectl exec -n data-platform minio-0 -- \
  sh -c 'echo "id,name,value\n1,test,100\n2,demo,200" > /tmp/data.csv'

# Step 2: Create Iceberg table in Trino
# Via Trino UI: https://trino.254carbon.com
CREATE TABLE iceberg.test.sales AS
SELECT * FROM (VALUES 
  (1, 'Product A', 100.50),
  (2, 'Product B', 200.75)
) AS t(id, product, amount);

# Step 3: Query in Trino
SELECT product, SUM(amount) as total 
FROM iceberg.test.sales 
GROUP BY product;

# Step 4: Visualize in Superset
# Connect Superset to Trino
# Create chart from the query
# Add to dashboard
```

---

### **Workflow 2: DolphinScheduler ETL Pipeline** (20 min)

**Goal**: Automated data workflow

**In DolphinScheduler UI**:

1. **Create Project**: `commodity-analytics`

2. **Create Workflow**: `daily-etl-pipeline`

3. **Add Tasks**:
   
   **Task 1 - Extract** (Shell):
   ```bash
   # Simulate data extraction
   curl -o /tmp/commodity_data.json \
     https://api.example.com/commodities
   ```
   
   **Task 2 - Transform** (Python):
   ```python
   import pandas as pd
   
   # Read data
   df = pd.read_json('/tmp/commodity_data.json')
   
   # Transform
   df['processed_date'] = pd.Timestamp.now()
   
   # Save
   df.to_parquet('/tmp/transformed_data.parquet')
   ```
   
   **Task 3 - Load** (SQL via Trino):
   ```sql
   -- Via Trino connection
   COPY iceberg.commodities.daily_data
   FROM '/tmp/transformed_data.parquet'
   WITH (format = 'PARQUET');
   ```

4. **Set Schedule**: Daily at 2 AM
5. **Run Manually** to test
6. **Monitor** in dashboard

---

### **Workflow 3: Ray Distributed Processing** (10 min)

**Goal**: Parallel data processing

**Create job script**:
```python
# save as ray_job.py
import ray
import time

ray.init("ray://ml-cluster-head-svc.ml-platform.svc.cluster.local:10001")

@ray.remote
def process_batch(batch_id, data_size):
    """Simulate heavy computation"""
    result = sum(range(data_size))
    return f"Batch {batch_id}: {result}"

# Submit 10 parallel tasks
futures = [process_batch.remote(i, 1000000) for i in range(10)]

# Get results
results = ray.get(futures)
for r in results:
    print(r)

# Check cluster resources
print(f"\nCluster resources: {ray.cluster_resources()}")

ray.shutdown()
```

**Run the job**:
```bash
# Option 1: From pod
kubectl run ray-client -n ml-platform --rm -it --restart=Never \
  --image=rayproject/ray:2.9.0 -- python3 /path/to/ray_job.py

# Option 2: Submit job file
kubectl cp ray_job.py ml-platform/ml-cluster-head-hkgr5:/tmp/
kubectl exec -n ml-platform ml-cluster-head-hkgr5 -- python3 /tmp/ray_job.py
```

**Monitor in Ray Dashboard**:
```bash
kubectl port-forward -n ml-platform svc/ml-cluster-head-svc 8265:8265 &
open http://localhost:8265
```

---

### **Workflow 4: Kafka Event Pipeline** (15 min)

**Goal**: Stream events through Kafka

**Step 1 - Create Topic**:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: commodity-prices
  namespace: kafka
  labels:
    strimzi.io/cluster: datahub-kafka
spec:
  partitions: 6
  replicas: 3
  config:
    retention.ms: 604800000
EOF
```

**Step 2 - Produce Events**:
```bash
# Start producer
kubectl run kafka-producer -n kafka --rm -it --restart=Never \
  --image=quay.io/strimzi/kafka:latest-kafka-4.0.0 -- \
  bin/kafka-console-producer.sh \
  --bootstrap-server datahub-kafka-kafka-bootstrap:9092 \
  --topic commodity-prices

# Send messages (JSON format):
{"commodity":"gold","price":1850.50,"timestamp":"2025-10-24T05:00:00Z"}
{"commodity":"silver","price":23.45,"timestamp":"2025-10-24T05:00:00Z"}
{"commodity":"copper","price":3.78,"timestamp":"2025-10-24T05:00:00Z"}
```

**Step 3 - Consume Events** (in another terminal):
```bash
kubectl run kafka-consumer -n kafka --rm -it --restart=Never \
  --image=quay.io/strimzi/kafka:latest-kafka-4.0.0 -- \
  bin/kafka-console-consumer.sh \
  --bootstrap-server datahub-kafka-kafka-bootstrap:9092 \
  --topic commodity-prices \
  --from-beginning
```

**Step 4 - Process with Spark** (via DolphinScheduler):
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Kafka Stream Processor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "datahub-kafka-kafka-bootstrap.kafka:9092") \
    .option("subscribe", "commodity-prices") \
    .load()

# Process and write to Iceberg
query = df.selectExpr("CAST(value AS STRING)") \
    .writeStream \
    .format("iceberg") \
    .outputMode("append") \
    .option("path", "iceberg.commodities.streaming_prices") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start()

query.awaitTermination()
```

---

### **Workflow 5: ML Experiment with Ray** (20 min)

**Goal**: Distributed ML training

**Create training script**:
```python
# ml_training.py
import ray
from ray import train
from ray.train import ScalingConfig
import numpy as np

ray.init("ray://ml-cluster-head-svc.ml-platform.svc.cluster.local:10001")

def train_model(config):
    """Simple training function"""
    for epoch in range(10):
        # Simulate training
        loss = np.random.random() / (epoch + 1)
        
        # Report metrics (will show in Ray dashboard)
        train.report({"loss": loss, "epoch": epoch})
        
    return {"final_loss": loss}

# Create distributed training
trainer = ray.train.Trainer(
    backend="torch",
    num_workers=2,  # Use both Ray workers
    use_gpu=False,
    scaling_config=ScalingConfig(num_workers=2)
)

# Run training
results = trainer.fit(train_func=train_model)
print(f"Training complete! Final loss: {results['final_loss']}")

ray.shutdown()
```

**Run the training**:
```bash
kubectl exec -n ml-platform ml-cluster-head-hkgr5 -- python3 /path/to/ml_training.py
```

**Monitor**:
- Ray Dashboard: http://localhost:8265
- See task distribution across workers
- View resource utilization
- Check execution timeline

---

## 🔍 **Platform Monitoring**

### **Check Overall Health**:
```bash
# Platform health
kubectl get pods -A --no-headers | \
  awk '{if ($4=="Running") running++; total++} 
       END {printf "Health: %.1f%% (%d/%d)\n", running/total*100, running, total}'

# Services by namespace
kubectl get pods -A --no-headers | \
  awk '{print $1}' | sort | uniq -c | sort -rn | head -10

# Non-running pods
kubectl get pods -A | grep -v "Running\|Completed"
```

### **Check Specific Service**:
```bash
# DolphinScheduler
kubectl get pods -n data-platform -l 'app in (dolphinscheduler-api,dolphinscheduler-master,dolphinscheduler-worker)'

# Kafka
kubectl get pods -n kafka

# Ray
kubectl get pods -n ml-platform
kubectl get raycluster -n ml-platform

# DataHub
kubectl get pods -n data-platform | grep datahub
```

### **View Logs**:
```bash
# Real-time logs
kubectl logs -f -l app=dolphinscheduler-api -n data-platform --tail=50

# All logs from a service
kubectl logs -l app=trino-coordinator -n data-platform --tail=100

# Logs from specific pod
kubectl logs <pod-name> -n data-platform
```

---

## 🎯 **Recommended Learning Path**

### **Day 1 (Today - 30 min)**:
1. ✅ **Grafana**: Explore dashboards and metrics (5 min)
2. ✅ **DolphinScheduler**: Create simple shell workflow (10 min)
3. ✅ **Trino**: Run SQL queries, create test table (10 min)
4. ✅ **Ray Dashboard**: View cluster resources (5 min)

### **Day 2 (Tomorrow - 60 min)**:
1. **Superset**: Create dashboard from Trino data (20 min)
2. **Kafka**: Create topic, produce/consume messages (20 min)
3. **DolphinScheduler**: Create multi-step ETL workflow (20 min)

### **Day 3 (This Week - 90 min)**:
1. **Ray**: Distributed computation job (30 min)
2. **End-to-End Pipeline**: Kafka → Spark → Iceberg → Trino → Superset (60 min)

### **Week 2: Advanced Features**
1. ML training with Ray
2. DataHub metadata ingestion
3. Complex workflows in DolphinScheduler
4. Real-time analytics with Kafka Streams

---

## 📊 **Sample Use Cases**

### **Use Case 1: Commodity Price Analytics**

**Data Flow**:
```
Price API → MinIO → Trino → Iceberg → Superset Dashboard
```

**Implementation**:
1. DolphinScheduler workflow fetches prices hourly
2. Stores JSON in MinIO
3. Trino queries create aggregated tables
4. Superset visualizes trends

### **Use Case 2: Real-Time Trading Signals**

**Data Flow**:
```
Market Data → Kafka → Spark Streaming → Iceberg → Real-time Dashboard
```

**Implementation**:
1. Stream market data to Kafka topic
2. Spark Streaming processes in real-time
3. Writes results to Iceberg tables
4. Superset shows live metrics

### **Use Case 3: ML Price Prediction**

**Data Flow**:
```
Historical Data → Ray → Feature Engineering → Training → MLflow → Model Registry
```

**Implementation**:
1. Load historical prices from Iceberg
2. Distributed feature engineering with Ray
3. Train model across Ray workers
4. Log experiments to MLflow
5. Deploy via Ray Serve

---

## 🛠️ **Troubleshooting**

### **Service Not Accessible**:
```bash
# Check pod status
kubectl get pods -n data-platform -l app=<service-name>

# Check ingress
kubectl get ingress -n data-platform

# Check service
kubectl get svc -n data-platform

# Check logs
kubectl logs -l app=<service-name> -n data-platform --tail=50
```

### **Workflow Failed in DolphinScheduler**:
```bash
# Check worker logs
kubectl logs -l app=dolphinscheduler-worker -n data-platform --tail=100

# Check master logs
kubectl logs -l app=dolphinscheduler-master -n data-platform --tail=100

# Restart workers if needed
kubectl rollout restart deployment dolphinscheduler-worker -n data-platform
```

### **Ray Job Stuck**:
```bash
# Check Ray dashboard
kubectl port-forward -n ml-platform svc/ml-cluster-head-svc 8265:8265

# Check worker logs
kubectl logs -l ray.io/node-type=worker -n ml-platform

# Check cluster status
kubectl get raycluster -n ml-platform
```

### **Kafka Issues**:
```bash
# Check broker status
kubectl get pods -n kafka

# Check topic status
kubectl get kafkatopic -n kafka

# Describe topic
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe --topic <topic-name>
```

---

## 📚 **Additional Resources**

### **Grafana Queries**:
```promql
# Pod CPU usage
sum(rate(container_cpu_usage_seconds_total{namespace="data-platform"}[5m])) by (pod)

# Memory usage
sum(container_memory_working_set_bytes{namespace="data-platform"}) by (pod)

# Service availability
up{kubernetes_namespace="data-platform"}

# Pod restart count
kube_pod_container_status_restarts_total{namespace="data-platform"}
```

### **Loki Queries** (in Grafana):
```logql
# All data-platform logs
{namespace="data-platform"}

# DolphinScheduler API logs
{namespace="data-platform", app="dolphinscheduler-api"}

# Error logs only
{namespace="data-platform"} |= "ERROR"

# Last 100 lines
{namespace="data-platform"} | tail 100
```

---

## ⚡ **Quick Commands Reference**

```bash
# Port-forward services
kubectl port-forward -n monitoring svc/grafana 3000:3000 &
kubectl port-forward -n ml-platform svc/ml-cluster-head-svc 8265:8265 &
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080 &
kubectl port-forward -n data-platform svc/minio-service 9000:9000 &

# Check platform health
kubectl get pods -A | grep -v "Running\|Completed" | wc -l

# Restart a service
kubectl rollout restart deployment <name> -n data-platform

# Scale a service
kubectl scale deployment <name> -n data-platform --replicas=<count>

# Get logs
kubectl logs -f -l app=<service> -n data-platform --tail=100

# Execute in pod
kubectl exec -it <pod-name> -n data-platform -- bash
```

---

## 🎯 **Next Steps After Playing**

Once you've explored the platform:

1. **Gather Requirements**: What real data do you want to process?
2. **Design Pipeline**: Map out your actual use case
3. **Implement Workflow**: Build in DolphinScheduler
4. **Monitor & Optimize**: Use Grafana to tune performance
5. **Scale as Needed**: Add workers, increase resources

---

## 🎊 **You're Ready!**

The platform is operational and waiting for you. Start with Grafana and DolphinScheduler - they're the easiest entry points and will give you a feel for the entire system.

**Have fun exploring your advanced analytics platform!** 🚀

---

**Quick Links**:
- Grafana: https://grafana.254carbon.com (admin/grafana123)
- DolphinScheduler: https://dolphin.254carbon.com (admin/dolphinscheduler123)
- Superset: https://superset.254carbon.com
- Trino: https://trino.254carbon.com
- DataHub: https://datahub.254carbon.com (when GMS ready)

**Need Help?** Check logs with `kubectl logs` or view in Grafana Loki.
