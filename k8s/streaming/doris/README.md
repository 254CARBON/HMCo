# Apache Doris Real-time OLAP

## Overview

Apache Doris provides sub-second query performance for real-time analytics on streaming commodity data.

## Features

- **Real-time Ingestion**: Routine Load from Kafka, Stream Load from Flink
- **Materialized Views**: Pre-aggregated price bars (OHLC)
- **Dynamic Partitioning**: Automatic time-based partitioning
- **High Availability**: 3 FE nodes, 3 BE nodes with replication
- **MPP Architecture**: Massively parallel query execution

## Architecture

```
Kafka → Routine Load → Doris BE (Storage) ← Query ← Superset
Flink → Stream Load → Doris FE (Coordinator)
```

## Deployment

```bash
# 1. Deploy FE nodes
kubectl apply -f doris-fe.yaml

# 2. Deploy BE nodes
kubectl apply -f doris-be.yaml

# 3. Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app=doris -n data-platform --timeout=10m

# 4. Initialize schema
kubectl apply -f doris-init.yaml

# 5. Verify deployment
kubectl logs -n data-platform job/doris-init
```

## Accessing Doris

```bash
# MySQL protocol (query interface)
kubectl port-forward -n data-platform svc/doris-fe-service 9030:9030

# Connect with MySQL client
mysql -h 127.0.0.1 -P 9030 -uroot

# Web UI
kubectl port-forward -n data-platform svc/doris-fe-service 8030:8030
# Open http://localhost:8030
```

## Schema

### Database: `commodity_realtime`

Tables:
- `energy_prices_rt`: Real-time commodity prices
- `price_1min_agg`: 1-minute OHLC aggregations
- `market_alerts`: Real-time anomaly alerts
- `weather_events_rt`: Streaming weather data
- `trading_signals`: Real-time trading signals

## Data Loading

### From Kafka (Routine Load)

```sql
CREATE ROUTINE LOAD commodity_realtime.load_energy_prices ON energy_prices_rt
COLUMNS(price_id, commodity_code, price, currency, unit, price_timestamp, ingestion_timestamp)
PROPERTIES
(
    "desired_concurrent_number"="3",
    "max_batch_interval" = "20",
    "max_batch_rows" = "300000",
    "max_batch_size" = "209715200",
    "strict_mode" = "false"
)
FROM KAFKA
(
    "kafka_broker_list" = "kafka-service:9092",
    "kafka_topic" = "commodity-prices",
    "property.group.id" = "doris_commodity_prices",
    "property.kafka_default_offsets" = "OFFSET_BEGINNING"
);
```

### From Flink (Stream Load)

See `../flink/flink-applications/doris-sink.yaml`

## Performance Tuning

- Tablet size: 200-300MB optimal
- Replication factor: 3 for critical tables
- Compaction: Automated with 4 tasks per disk
- Query cache: 4GB per BE node

## Monitoring

Metrics available at:
- FE: `http://doris-fe-service:8030/metrics`
- BE: `http://doris-be-service:8040/metrics`


