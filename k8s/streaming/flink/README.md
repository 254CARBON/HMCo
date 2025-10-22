# Apache Flink Stream Processing

## Overview

Apache Flink provides real-time stream processing for the 254Carbon platform with:
- Event-time processing with watermarks
- Exactly-once state consistency
- Scalable stateful computations
- SQL interface for streaming queries

## Architecture

```
Kafka Topics → Flink Jobs → Doris/Iceberg/Elasticsearch
                  ↓
            State Backend
         (RocksDB on MinIO)
```

## Components

### Flink Operator
- Manages Flink cluster lifecycle
- Handles job deployments and upgrades
- Monitors job health

### Flink Applications
- Price aggregation (windowing)
- Anomaly detection
- Data enrichment
- Quality validation

## Deployment

```bash
# 1. Install CRDs
kubectl apply -f flink-crds.yaml

# 2. Deploy operator
kubectl apply -f flink-operator.yaml

# 3. Create RBAC for Flink apps
kubectl apply -f flink-rbac.yaml

# 4. Verify operator is running
kubectl get pods -n flink-operator
kubectl logs -n flink-operator -l app=flink-operator
```

## Creating Flink Applications

See `flink-applications/` directory for example streaming jobs.

## Monitoring

- Flink UI: Port-forward `kubectl port-forward -n data-platform svc/flink-jobmanager 8081:8081`
- Metrics: Available at `:9999/metrics` for Prometheus scraping


