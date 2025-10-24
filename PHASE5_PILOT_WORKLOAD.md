# Phase 5: Pilot Workload - Commodity Price Ingestion Pipeline

**Status**: Implementation Starting  
**Date**: October 25, 2025  
**Target**: Deploy real production ETL workflow  
**Duration**: ~3-4 hours

---

## 🎯 Objective

Deploy the **first production workload** - an automated commodity price ingestion pipeline that demonstrates the platform's end-to-end capabilities:

```
External API → Kafka → Trino Data Lake → Superset Dashboard → Grafana Monitoring
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMMODITY PRICE PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. EXTRACT (6am daily)                                          │
│     └─> DolphinScheduler CronJob                                │
│         └─> Fetch from commodity API                            │
│         └─> Validate data quality                               │
│                                                                   │
│  2. STREAM (Real-time)                                           │
│     └─> Publish to Kafka topic (commodities)                    │
│     └─> 3-broker cluster with replication                       │
│                                                                   │
│  3. TRANSFORM (Continuous)                                       │
│     └─> Kafka consumer in Trino                                 │
│     └─> Store in Iceberg tables                                 │
│     └─> Partitioned by date/commodity                           │
│                                                                   │
│  4. ANALYZE (On-demand)                                          │
│     └─> Trino SQL queries                                       │
│     └─> Aggregate prices by commodity/timeframe                 │
│                                                                   │
│  5. VISUALIZE (Real-time)                                        │
│     └─> Superset dashboards                                     │
│     └─> Price trends, alerts, metrics                           │
│                                                                   │
│  6. MONITOR (Continuous)                                         │
│     └─> Grafana pipeline health                                 │
│     └─> Kafka throughput, data quality metrics                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Create Kafka Topic for Commodity Data

```bash
# Connect to Kafka
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --create \
  --topic commodities \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config compression.type=snappy

# Verify topic created
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --list | grep commodities
```

---

## Step 2: Create Iceberg Table for Storage

```bash
# Create commodity price table in Trino/Iceberg
kubectl exec -n data-platform trino-coordinator-xxx -it -- \
  trino --execute "
    CREATE TABLE IF NOT EXISTS iceberg.default.commodity_prices (
      id VARCHAR,
      commodity VARCHAR,
      price DOUBLE,
      unit VARCHAR,
      timestamp TIMESTAMP,
      source VARCHAR,
      data_quality_score DOUBLE,
      extracted_at TIMESTAMP
    )
    WITH (
      format = 'PARQUET',
      partitioning = ARRAY['DATE(timestamp)'],
      bucketing_version = 2,
      bucketed_by = ARRAY['commodity'],
      bucket_count = 10
    );
  "

# Verify table creation
kubectl exec -n data-platform trino-coordinator-xxx -it -- \
  trino --execute "
    SHOW TABLES FROM iceberg.default LIKE 'commodity%';
  "
```

---

## Step 3: Create DolphinScheduler Workflow

Create a workflow definition file:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: commodity-pipeline-workflow
  namespace: data-platform
data:
  workflow.json: |
    {
      "name": "Commodity Price Ingestion Pipeline",
      "description": "Daily commodity price extraction from API and load to data lake",
      "tasks": [
        {
          "id": "extract_commodities",
          "name": "Extract from Commodity API",
          "type": "SHELL",
          "command": "python3",
          "script": "#!/usr/bin/env python3\nimport requests\nimport json\nfrom datetime import datetime\n\nAPI_ENDPOINT = 'https://api.example.com/commodities'\nKAFKA_BOOTSTRAP = 'datahub-kafka-kafka-bootstrap.kafka:9092'\n\ntry:\n    # Fetch from commodity API\n    response = requests.get(API_ENDPOINT, timeout=30)\n    response.raise_for_status()\n    \n    data = response.json()\n    print(f'Fetched {len(data)} commodities')\n    \n    # Publish to Kafka\n    from kafka import KafkaProducer\n    producer = KafkaProducer(\n        bootstrap_servers=[KAFKA_BOOTSTRAP],\n        value_serializer=lambda v: json.dumps(v).encode('utf-8')\n    )\n    \n    for item in data:\n        item['extracted_at'] = datetime.utcnow().isoformat()\n        item['timestamp'] = datetime.fromisoformat(item.get('timestamp', datetime.utcnow().isoformat()))\n        producer.send('commodities', value=item)\n    \n    producer.flush()\n    print(f'Successfully published to Kafka')\n    \nexcept Exception as e:\n    print(f'Error: {str(e)}')\n    exit(1)",
          "timeout": 300,
          "retryTimes": 3,
          "retryInterval": 60
        },
        {
          "id": "validate_quality",
          "name": "Validate Data Quality",
          "type": "SHELL",
          "command": "python3",
          "script": "#!/usr/bin/env python3\nimport json\nfrom kafka import KafkaConsumer\n\nKAFKA_BOOTSTRAP = 'datahub-kafka-kafka-bootstrap.kafka:9092'\n\ntry:\n    consumer = KafkaConsumer(\n        'commodities',\n        bootstrap_servers=[KAFKA_BOOTSTRAP],\n        value_deserializer=lambda m: json.loads(m.decode('utf-8')),\n        group_id='dq-checker',\n        auto_offset_reset='latest',\n        consumer_timeout_ms=5000\n    )\n    \n    quality_score = 100\n    record_count = 0\n    \n    for message in consumer:\n        record_count += 1\n        data = message.value\n        \n        # Check required fields\n        required_fields = ['commodity', 'price', 'timestamp']\n        for field in required_fields:\n            if field not in data or data[field] is None:\n                quality_score -= 10\n        \n        # Check price validity\n        if data.get('price', 0) <= 0:\n            quality_score -= 5\n    \n    quality_score = max(0, quality_score)\n    print(f'Data Quality Score: {quality_score}% ({record_count} records)')\n    \n    if quality_score < 80:\n        print('Warning: Data quality below threshold')\n        exit(1)\n        \nexcept Exception as e:\n    print(f'Quality check failed: {str(e)}')\n    exit(1)",
          "timeout": 300,
          "retryTimes": 2,
          "retryInterval": 60
        },
        {
          "id": "load_to_iceberg",
          "name": "Load to Iceberg Data Lake",
          "type": "SHELL",
          "command": "python3",
          "script": "#!/usr/bin/env python3\nimport json\nfrom kafka import KafkaConsumer\nimport trino.dbapi\n\nKAFKA_BOOTSTRAP = 'datahub-kafka-kafka-bootstrap.kafka:9092'\nTRINO_HOST = 'trino-coordinator.data-platform'\nTRINO_PORT = 8080\n\ntry:\n    # Connect to Trino\n    conn = trino.dbapi.connect(\n        host=TRINO_HOST,\n        port=TRINO_PORT,\n        catalog='iceberg',\n        schema='default'\n    )\n    cursor = conn.cursor()\n    \n    # Consume from Kafka and insert to Iceberg\n    consumer = KafkaConsumer(\n        'commodities',\n        bootstrap_servers=[KAFKA_BOOTSTRAP],\n        value_deserializer=lambda m: json.loads(m.decode('utf-8')),\n        group_id='iceberg-loader',\n        auto_offset_reset='latest',\n        consumer_timeout_ms=10000\n    )\n    \n    insert_count = 0\n    for message in consumer:\n        data = message.value\n        \n        # Insert into Iceberg table\n        cursor.execute(\n            '''INSERT INTO commodity_prices \n               (id, commodity, price, unit, timestamp, source, data_quality_score, extracted_at)\n               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',\n            (\n                data.get('id'),\n                data.get('commodity'),\n                float(data.get('price', 0)),\n                data.get('unit', 'USD'),\n                data.get('timestamp'),\n                data.get('source', 'api'),\n                100.0,\n                data.get('extracted_at')\n            )\n        )\n        insert_count += 1\n    \n    conn.commit()\n    print(f'Successfully loaded {insert_count} records to Iceberg')\n    \nexcept Exception as e:\n    print(f'Error loading to Iceberg: {str(e)}')\n    exit(1)",
          "timeout": 300,
          "retryTimes": 1,
          "retryInterval": 60
        }
      ],
      "schedule": {\n        "startTime\": \"2025-10-26 06:00:00\",\n        \"endTime\": \"2026-12-31 23:59:59\",\n        "crontab": "0 6 * * *"\n      }
    }
```

---

## Step 4: Deploy Workflow via DolphinScheduler API

```bash
# Create the workflow in DolphinScheduler
WORKFLOW_JSON=$(cat <<'EOF'
{
  "name": "Commodity Price Pipeline",
  "description": "Daily commodity price ingestion",
  "processDefinitionJson": {
    "globalParams": [],
    "tasks": [
      {
        "name": "extract_commodities",
        "type": "SHELL",
        "runFlag": "NORMAL",
        "taskPriority": "MEDIUM",
        "workerGroup": "default",
        "failRetryTimes": 3,
        "failRetryInterval": 1,
        "timeoutFlag": "OPEN",
        "timeoutNotifyStrategy": "WARN",
        "timeout": 3600,
        "delayTime": 0,
        "taskParams": {
          "rawScript": "echo 'Extracting commodity data...'",
          "localParams": [],
          "resourceList": [],
          "dependence": {},
          "conditionResult": {"successNode": [""], "failedNode": [""]},
          "switchResult": {}
        },
        "displayName": "Extract Commodities"
      }
    ],
    "tenantId": 1,
    "timeout": 0
  }
}
EOF
)

# Execute via DolphinScheduler API
kubectl exec -n data-platform dolphinscheduler-api-xxx -it -- \
  curl -X POST http://localhost:8080/dolphinscheduler/api/v1/projects/1/processDefinition \
  -H "Content-Type: application/json" \
  -d "$WORKFLOW_JSON"
```

---

## Step 5: Create Superset Dashboard

```bash
# Access Superset and create dashboard
# 1. Open: https://superset.254carbon.com
# 2. Create datasource pointing to Trino iceberg.default.commodity_prices
# 3. Create charts:
#    - Price trend line chart (commodity vs time)
#    - Bar chart (prices by commodity)
#    - Latest prices table
# 4. Combine into dashboard "Commodity Price Monitor"
```

---

## Step 6: Set Up Grafana Monitoring

Create Grafana dashboard for pipeline health:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-commodity-dashboard
  namespace: monitoring
data:
  commodity-dashboard.json: |
    {
      "dashboard": {
        "title": "Commodity Price Pipeline",
        "panels": [
          {
            "title": "Kafka Messages Per Second",
            "targets": [
              {
                "expr": "rate(kafka_server_network_received_bytes_total{topic=\"commodities\"}[5m])"
              }
            ]
          },
          {
            "title": "Pipeline Execution Time",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, dolphinscheduler_task_execute_time_bucket)"
              }
            ]
          },
          {
            "title": "Data Quality Score",
            "targets": [
              {
                "expr": "commodity_data_quality_score"
              }
            ]
          },
          {
            "title": "Iceberg Records Inserted",
            "targets": [
              {
                "expr": "increase(iceberg_insert_records_total[1h])"
              }
            ]
          }
        ]
      }
    }
```

---

## Step 7: Set Up Alerts

```bash
# Create alert rule for pipeline failures
kubectl apply -f - <<'EOF'
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: commodity-pipeline-alerts
  namespace: monitoring
spec:
  groups:
  - name: commodity-pipeline
    interval: 30s
    rules:
    - alert: CommodityPipelineFailure
      expr: dolphinscheduler_task_failed_total > 0
      for: 5m
      annotations:
        summary: "Commodity pipeline failed"
        description: "The commodity price ingestion pipeline has failed. Check logs immediately."
    
    - alert: KafkaHighLatency
      expr: kafka_producer_latency_avg > 1000
      for: 5m
      annotations:
        summary: "High Kafka latency"
        description: "Kafka producer latency is above 1 second. Check broker health."
    
    - alert: DataQualityDegraded
      expr: commodity_data_quality_score < 80
      for: 10m
      annotations:
        summary: "Data quality below threshold"
        description: "Commodity data quality score is below 80%. Review extraction logic."
EOF
```

---

## Validation & Testing

### Test 1: Verify Kafka Topic

```bash
# Check messages flowing through Kafka
kubectl exec -n kafka datahub-kafka-kafka-pool-0 -- \
  bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic commodities \
  --from-beginning \
  --max-messages 5
```

### Test 2: Query Iceberg Table

```bash
# Check data in Trino
kubectl exec -n data-platform trino-coordinator-xxx -it -- \
  trino --execute "
    SELECT 
      commodity,
      COUNT(*) as record_count,
      AVG(price) as avg_price,
      MIN(price) as min_price,
      MAX(price) as max_price
    FROM iceberg.default.commodity_prices
    GROUP BY commodity
    ORDER BY commodity;
  "
```

### Test 3: Check Dashboard

```bash
# Verify Superset dashboard displays data
# Open: https://superset.254carbon.com/dashboard/commodity-prices
```

### Test 4: Monitor Pipeline

```bash
# Check Grafana dashboard
# Open: https://grafana.254carbon.com/d/commodity-pipeline
```

---

## Success Criteria

✅ Kafka topic receiving data  
✅ Data persisted in Iceberg tables  
✅ Trino queries returning results  
✅ Superset dashboard showing visualizations  
✅ Grafana monitoring pipeline health  
✅ Alerts firing on failures  
✅ End-to-end latency < 5 minutes  

---

## Phase 5 Outcome

**A production commodity price pipeline deployed, validated, and monitored - proving the platform's end-to-end capability for real workloads.**

Next: Iterate with additional pipelines and performance optimization.

**Status**: Ready to deploy ✅
