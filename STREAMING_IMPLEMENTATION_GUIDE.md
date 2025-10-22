# Real-time Streaming Platform Implementation Guide

**Platform**: 254Carbon Real-time Analytics  
**Date**: October 22, 2025  
**Status**: ✅ Implementation Complete

---

## Executive Summary

Successfully implemented a comprehensive real-time streaming platform featuring:

- **Kafka Connect**: 3-worker distributed cluster for data ingestion/egress
- **Apache Flink**: Stream processing with exactly-once semantics
- **Apache Doris**: Real-time OLAP with sub-second query performance
- **Monitoring**: Comprehensive metrics, alerts, and dashboards
- **Use Cases**: Real-time commodity price monitoring and alerting

## Architecture

```
External APIs → Kafka Connect → Kafka (3 brokers) → Flink → Doris/Iceberg
                     ↓               ↓                  ↓         ↓
                 Connectors      Schema Reg        Processing  Real-time
                                                   & Alerts   Analytics
```

## Components Deployed

### 1. Kafka Infrastructure (Scaled to 3 Brokers)

**File**: `k8s/shared/kafka/kafka-production.yaml`

- 3 Kafka brokers for high availability
- Replication factor: 3
- Min in-sync replicas: 2
- Storage: 50Gi per broker
- Resources: 4-6Gi memory, 1-2 CPU per broker

**Verification**:
```bash
kubectl get pods -n data-platform -l app=kafka
kubectl exec -n data-platform kafka-0 -- kafka-broker-api-versions --bootstrap-server kafka-service:9092
```

### 2. Apache Flink Operator

**Files**:
- `k8s/streaming/flink/flink-crds.yaml` - Custom Resource Definitions
- `k8s/streaming/flink/flink-operator.yaml` - Operator deployment
- `k8s/streaming/flink/flink-rbac.yaml` - RBAC for Flink apps

**Features**:
- Manages FlinkDeployment CRDs
- Automatic job lifecycle management
- High availability with 2 JobManager replicas
- Metrics exposed for Prometheus

**Verification**:
```bash
kubectl get pods -n flink-operator
kubectl get crd | grep flink
```

### 3. Kafka Connect Cluster

**File**: `k8s/streaming/kafka-connect/kafka-connect.yaml`

- 3 distributed workers
- Avro converter with Schema Registry
- JMX metrics enabled
- Plugin path: `/usr/share/confluent-hub-components`

**Resources**: 2-4Gi memory, 0.5-2 CPU per worker

**API Access**:
```bash
kubectl port-forward -n data-platform svc/kafka-connect-service 8083:8083
curl http://localhost:8083/connectors
```

### 4. Kafka Connect Connectors

**Source Connectors** (`k8s/streaming/connectors/http-source-connector.yaml`):
- HTTP Source for API polling (EIA, FRED, Weather)
- PostgreSQL CDC (Debezium)
- MySQL CDC

**Sink Connectors** (`k8s/streaming/connectors/sink-connectors.yaml`):
- Iceberg Sink (long-term storage)
- Elasticsearch Sink (real-time search)
- S3/MinIO Sink (archival)
- JDBC Sink (operational databases)

### 5. Apache Doris Cluster

**Frontend (FE)** - `k8s/streaming/doris/doris-fe.yaml`:
- 3 replicas for HA
- Query coordination and metadata management
- Resources: 4-8Gi memory, 2-4 CPU

**Backend (BE)** - `k8s/streaming/doris/doris-be.yaml`:
- 3 initial replicas (scalable to 10)
- Data storage and computation
- Resources: 8-16Gi memory, 2-8 CPU
- Storage: 100Gi per node

**Schema** - `k8s/streaming/doris/doris-init.yaml`:
- Database: `commodity_realtime`
- Tables:
  - `energy_prices_rt` - Real-time prices
  - `price_1min_agg` - 1-minute OHLC bars
  - `market_alerts` - Anomaly alerts
  - `weather_events_rt` - Weather data
  - `trading_signals` - Trading recommendations

**Access**:
```bash
# MySQL protocol
kubectl port-forward -n data-platform svc/doris-fe-service 9030:9030
mysql -h 127.0.0.1 -P 9030 -uroot

# Web UI
kubectl port-forward -n data-platform svc/doris-fe-service 8030:8030
# Open http://localhost:8030
```

### 6. Flink Streaming Applications

#### Data Enricher (`flink-applications/data-enricher.yaml`)
- Enriches raw price data with metadata from Iceberg
- Joins streaming data with dimension tables
- Outputs to `commodity-prices-enriched` topic

#### Price Aggregator (`flink-applications/price-aggregator.yaml`)
- Creates 1-minute OHLC price bars
- Calculates volume-weighted averages
- Writes directly to Doris via Doris connector

#### Anomaly Detector (`flink-applications/anomaly-detector.yaml`)
- Detects price changes >5% in 1 minute
- Generates alerts with severity levels
- Publishes to `market-alerts` topic and Doris

### 7. Monitoring Infrastructure

**Service Monitors** (`monitoring/streaming-servicemonitors.yaml`):
- Kafka Connect metrics
- Flink JobManager/TaskManager metrics
- Doris FE/BE metrics

**Alert Rules** (`monitoring/streaming-alerts.yaml`):
- Connector failures
- Flink job failures/restarts
- High backpressure
- Checkpoint failures
- Doris node failures
- High query latency
- Data quality issues

**Grafana Dashboards** (`monitoring/grafana-dashboards.yaml`):
- Kafka Connect monitoring
- Apache Flink monitoring
- Apache Doris monitoring
- Streaming infrastructure overview

### 8. Real-time Commodity Monitoring Use Case

**File**: `k8s/streaming/use-cases/realtime-commodity-monitoring.yaml`

**Features**:
- Routine Load from Kafka to Doris (continuous ingestion)
- Real-time trading signal queries
- Superset dashboard queries for visualization
- Sub-second latency for price updates

**Queries**:
- Latest prices (5-minute window)
- Price spreads and changes
- Active alerts
- Volume-weighted average prices (VWAP)
- Top movers

## Deployment

### Automated Deployment

```bash
# Deploy entire streaming platform
./scripts/deploy-streaming-platform.sh
```

This script:
1. Scales Kafka to 3 brokers
2. Deploys Flink Operator
3. Deploys Kafka Connect cluster
4. Deploys Apache Doris (FE + BE)
5. Configures connectors
6. Deploys Flink applications
7. Sets up monitoring
8. Configures use cases

### Manual Deployment

```bash
# 1. Scale Kafka
kubectl apply -f k8s/shared/kafka/kafka-production.yaml

# 2. Deploy Flink
kubectl apply -f k8s/streaming/flink/flink-crds.yaml
kubectl apply -f k8s/streaming/flink/flink-operator.yaml
kubectl apply -f k8s/streaming/flink/flink-rbac.yaml

# 3. Deploy Kafka Connect
kubectl apply -f k8s/streaming/kafka-connect/kafka-connect.yaml

# 4. Deploy Doris
kubectl apply -f k8s/streaming/doris/doris-fe.yaml
kubectl apply -f k8s/streaming/doris/doris-be.yaml
kubectl apply -f k8s/streaming/doris/doris-init.yaml

# 5. Deploy connectors
kubectl apply -f k8s/streaming/connectors/

# 6. Deploy Flink apps
kubectl apply -f k8s/streaming/flink/flink-applications/

# 7. Deploy monitoring
kubectl apply -f k8s/streaming/monitoring/

# 8. Deploy use cases
kubectl apply -f k8s/streaming/use-cases/
```

### Register Connectors

```bash
./scripts/register-connectors.sh
```

## Verification

### Check All Components

```bash
# Kafka brokers
kubectl get pods -n data-platform -l app=kafka

# Kafka Connect
kubectl get pods -n data-platform -l app=kafka-connect
kubectl port-forward -n data-platform svc/kafka-connect-service 8083:8083
curl http://localhost:8083/connectors

# Flink Operator
kubectl get pods -n flink-operator

# Flink Deployments
kubectl get flinkdeployment -n data-platform

# Doris
kubectl get pods -n data-platform -l app=doris
kubectl exec -n data-platform doris-fe-0 -- mysql -uroot -e "SHOW DATABASES;"

# Monitoring
kubectl get servicemonitor -n monitoring | grep streaming
kubectl get prometheusrule -n monitoring | grep streaming
```

### Test Data Flow

```bash
# 1. Send test data to Kafka
kubectl exec -n data-platform kafka-0 -- kafka-console-producer \
  --bootstrap-server kafka-service:9092 \
  --topic commodity-prices-raw <<EOF
{"commodity_code":"CL","price":75.50,"currency":"USD","unit":"barrel","price_timestamp":"$(date -u +%Y-%m-%dT%H:%M:%S)"}
EOF

# 2. Check Flink job is processing
kubectl get flinkdeployment -n data-platform

# 3. Query Doris for data
kubectl exec -n data-platform doris-fe-0 -- mysql -uroot -e \
  "SELECT * FROM commodity_realtime.energy_prices_rt ORDER BY price_timestamp DESC LIMIT 10;"

# 4. Check alerts
kubectl exec -n data-platform doris-fe-0 -- mysql -uroot -e \
  "SELECT * FROM commodity_realtime.market_alerts WHERE alert_timestamp > NOW() - INTERVAL 1 HOUR;"
```

## Performance Metrics

### Expected Performance

- **Ingestion Latency**: < 100ms (API to Kafka)
- **Processing Latency**: < 1 second (Kafka to Doris)
- **Query Latency**: < 100ms (P95 in Doris)
- **Throughput**: 100K messages/second
- **Availability**: 99.9% uptime

### Monitoring Queries

```promql
# Kafka Connect throughput
rate(kafka_connect_source_task_source_record_poll_total[5m])

# Flink records per second
rate(flink_taskmanager_job_task_numRecordsInPerSecond[5m])

# Doris query latency (P95)
histogram_quantile(0.95, rate(doris_fe_query_latency_ms_bucket[5m]))

# End-to-end latency
histogram_quantile(0.95, rate(stream_processing_latency_seconds_bucket[5m]))
```

## Troubleshooting

### Kafka Connect Issues

```bash
# Check worker logs
kubectl logs -n data-platform -l app=kafka-connect --tail=100

# Check connector status
curl http://localhost:8083/connectors/<connector-name>/status

# Restart connector
curl -X POST http://localhost:8083/connectors/<connector-name>/restart
```

### Flink Job Issues

```bash
# Check job status
kubectl get flinkdeployment -n data-platform

# Check job logs
kubectl logs -n data-platform <flink-jobmanager-pod>

# Access Flink UI
kubectl port-forward -n data-platform svc/<flink-jobmanager> 8081:8081
```

### Doris Issues

```bash
# Check FE logs
kubectl logs -n data-platform doris-fe-0

# Check BE logs
kubectl logs -n data-platform doris-be-0

# Check Routine Load status
kubectl exec -n data-platform doris-fe-0 -- mysql -uroot -e "SHOW ROUTINE LOAD;"

# Pause/Resume Routine Load
kubectl exec -n data-platform doris-fe-0 -- mysql -uroot -e \
  "PAUSE ROUTINE LOAD FOR commodity_realtime.load_energy_prices_rt;"
```

## Next Steps

1. **Add More Data Sources**: Configure additional HTTP source connectors for more commodity APIs
2. **Optimize Doris**: Fine-tune materialized views and indexes based on query patterns
3. **Scale Resources**: Adjust replicas and resources based on actual load
4. **Implement More Flink Jobs**: Add jobs for correlation analysis, forecasting, etc.
5. **Create Superset Dashboards**: Build real-time dashboards connected to Doris
6. **Enable Security**: Add authentication and encryption for production use

## Resources

- **Kafka Connect Documentation**: https://docs.confluent.io/platform/current/connect/
- **Apache Flink Documentation**: https://nightlies.apache.org/flink/flink-docs-stable/
- **Apache Doris Documentation**: https://doris.apache.org/docs/get-starting/
- **Platform Architecture**: See `k8s/streaming/README.md`

## Support

For issues or questions:
1. Check logs: `kubectl logs -n <namespace> <pod-name>`
2. Check metrics: Grafana dashboards in monitoring namespace
3. Review alerts: Prometheus AlertManager
4. Consult documentation: Component-specific READMEs

---

**Implementation Complete**: October 22, 2025  
**Platform Version**: v2.0.0 (with Real-time Streaming)


