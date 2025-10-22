# Feast Feature Store

**Platform**: 254Carbon Advanced Analytics Platform  
**Component**: Real-time Feature Store  
**Technology**: Feast 0.35.0  
**Status**: Implementation Phase 1

---

## Overview

Feast is an open-source feature store that provides:

- **Online Store**: Low-latency feature serving from Redis
- **Offline Store**: Historical features for training from Iceberg/Doris
- **Feature Registry**: Centralized feature definitions in PostgreSQL
- **Materialization**: Automated sync from offline to online store
- **Versioning**: Feature definition versioning and lineage

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Offline Store (Iceberg/Doris)                          │
│  Historical Features for Training                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Materialize (Hourly/Daily)
                     ↓
┌─────────────────────────────────────────────────────────┐
│  Feast Server (2 replicas)                              │
│  Feature Registry (PostgreSQL)                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Write Features
                     ↓
┌─────────────────────────────────────────────────────────┐
│  Online Store (Redis)                                   │
│  Low-latency Feature Serving                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Fetch Features (< 10ms)
                     ↓
┌─────────────────────────────────────────────────────────┐
│  ML Serving (Ray Serve, Seldon)                         │
│  Real-time Predictions                                  │
└─────────────────────────────────────────────────────────┘
```

## Features

### Commodity Price Features
- `current_price`: Latest commodity price
- `price_change_1h`: 1-hour price change %
- `price_change_24h`: 24-hour price change %
- `volume_24h`: 24-hour trading volume
- `volatility_7d`: 7-day price volatility
- `ma_7d`: 7-day moving average
- `ma_30d`: 30-day moving average
- `rsi_14d`: 14-day RSI indicator

### Economic Features
- `gdp_growth`: GDP growth rate
- `inflation_rate`: CPI inflation rate
- `interest_rate`: Central bank interest rate
- `unemployment_rate`: Unemployment percentage
- `industrial_production`: Industrial production index

### Weather Features
- `temperature_avg`: Average temperature
- `precipitation`: Precipitation level
- `wind_speed`: Wind speed
- `storm_index`: Storm severity index

## Deployment

### Prerequisites

```bash
# Ensure PostgreSQL is running
kubectl get pods -n data-platform -l app=postgres

# Ensure Redis is running
kubectl get pods -n data-platform -l app=redis
```

### Deploy Feast

```bash
# 1. Deploy Feast server
kubectl apply -f k8s/ml-platform/feast-feature-store/feast-deployment.yaml

# 2. Initialize Feast (create DB, apply features)
kubectl apply -f k8s/ml-platform/feast-feature-store/feast-init-job.yaml

# 3. Wait for initialization
kubectl wait --for=condition=complete job/feast-init -n data-platform --timeout=300s

# 4. Deploy materialization jobs
kubectl apply -f k8s/ml-platform/feast-feature-store/feast-materialization-job.yaml

# 5. Verify deployment
kubectl get pods -n data-platform -l app=feast-server
```

### Verify Installation

```bash
# Check Feast server status
kubectl get pods -n data-platform -l app=feast-server

# Check init job logs
kubectl logs -n data-platform job/feast-init

# Port-forward to Feast server
kubectl port-forward -n data-platform svc/feast-server 8080:8080

# Test health endpoint
curl http://localhost:8080/health
```

## Usage

### Python SDK

```python
from feast import FeatureStore
import pandas as pd

# Initialize Feast client
store = FeatureStore(
    repo_path="feast_repo",
    config={
        "project": "254carbon",
        "registry": "postgresql://postgres:postgres@postgres-shared-service:5432/feast",
        "online_store": {
            "type": "redis",
            "connection_string": "redis-service:6379"
        }
    }
)

# Fetch online features for real-time serving
entity_rows = pd.DataFrame({
    "commodity_code": ["crude_oil_wti", "natural_gas"],
})

features = store.get_online_features(
    features=[
        "commodity_price_features:current_price",
        "commodity_price_features:price_change_24h",
        "commodity_price_features:volatility_7d",
    ],
    entity_rows=entity_rows
).to_dict()

print(features)
```

### gRPC API

```python
import grpc
from feast.protos.serving.serving_pb2 import GetOnlineFeaturesRequest
from feast.protos.serving.serving_pb2_grpc import ServingServiceStub

# Connect to Feast server
channel = grpc.insecure_channel("feast-server.data-platform.svc.cluster.local:6566")
stub = ServingServiceStub(channel)

# Request features
request = GetOnlineFeaturesRequest(
    features=[
        "commodity_price_features:current_price",
        "commodity_price_features:volatility_7d",
    ],
    entity_rows=[
        {"commodity_code": "crude_oil_wti"}
    ]
)

response = stub.GetOnlineFeatures(request)
```

### REST API

```bash
# Get features via HTTP
curl -X POST http://feast-server.data-platform.svc.cluster.local:8080/get-online-features \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      "commodity_price_features:current_price",
      "commodity_price_features:price_change_24h"
    ],
    "entities": {
      "commodity_code": ["crude_oil_wti"]
    }
  }'
```

## Materialization

### Manual Materialization

```bash
# Run materialization manually
kubectl exec -n data-platform -it deploy/feast-server -- \
  feast materialize-incremental $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)
```

### Scheduled Materialization

Two CronJobs are deployed:

1. **Hourly** (`feast-materialize-hourly`): Materializes last 2 hours
   - Schedule: Every hour at :05
   - For: Real-time features (prices, weather)

2. **Daily** (`feast-materialize-daily`): Materializes last 7 days
   - Schedule: Daily at 2 AM
   - For: Slower-changing features (economic indicators)

### Check Materialization Status

```bash
# Check hourly job
kubectl get cronjob feast-materialize-hourly -n data-platform

# Check last runs
kubectl get jobs -n data-platform -l app=feast,component=materialization

# View logs
kubectl logs -n data-platform -l app=feast,component=materialization-job --tail=100
```

## Integration with Ray Serve

Ray Serve automatically uses Feast for feature fetching:

```python
# In Ray Serve deployment
from feast import FeatureStore

store = FeatureStore(repo_path="/feast/feature_repo")

# Fetch features during prediction
features = store.get_online_features(
    features=["commodity_price_features"],
    entity_rows={"commodity_code": [entity_id]}
).to_dict()

prediction = model.predict(features)
```

## Monitoring

### Metrics

Feast exposes Prometheus metrics:

```bash
# Access metrics
kubectl port-forward -n data-platform svc/feast-server 8080:8080
curl http://localhost:8080/metrics
```

Key metrics:
- `feast_online_request_latency`: Feature fetch latency
- `feast_feature_request_count`: Total feature requests
- `feast_feature_not_found_count`: Missing features
- `feast_materialization_duration`: Materialization time

### Health Checks

```bash
# Check server health
curl http://feast-server.data-platform.svc.cluster.local:8080/health

# Check Redis connectivity
kubectl exec -n data-platform -it deploy/feast-server -- redis-cli -h redis-service ping

# Check PostgreSQL connectivity
kubectl exec -n data-platform -it deploy/feast-server -- \
  psql -h postgres-shared-service -U postgres -d feast -c "SELECT COUNT(*) FROM feature_views"
```

## Feature Engineering Pipeline

Features are computed in Flink (Phase 1.2) and written to:
1. **Iceberg** (offline store) for training
2. **Redis** (online store) via Feast materialization for serving

```
Kafka → Flink Feature Engineering → Iceberg (offline)
                                  → Feast Materialization → Redis (online)
```

## Best Practices

1. **Feature Naming**: Use descriptive names with entity prefix
2. **TTL Configuration**: Set appropriate TTL based on feature freshness needs
3. **Materialization Schedule**: Align with data update frequency
4. **Monitoring**: Track feature request latency and missing features
5. **Versioning**: Use feature views for versioning
6. **Documentation**: Document feature definitions and transformations

## Troubleshooting

### Features Not Found

```bash
# List all feature views
kubectl exec -n data-platform -it deploy/feast-server -- \
  feast feature-views list

# Check Redis keys
kubectl exec -n data-platform -it deploy/redis -- \
  redis-cli KEYS "feast:*"
```

### Materialization Failures

```bash
# Check job logs
kubectl logs -n data-platform job/feast-materialize-hourly-<timestamp>

# Manual materialize with verbose output
kubectl exec -n data-platform -it deploy/feast-server -- \
  feast materialize-incremental $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --verbose
```

### High Latency

```bash
# Check Redis performance
kubectl exec -n data-platform -it deploy/redis -- redis-cli --latency

# Check network latency
kubectl exec -n data-platform -it deploy/feast-server -- \
  ping redis-service
```

## Next Steps

- [ ] Integrate with Flink for real-time feature computation (Phase 1.2)
- [ ] Add more feature views for different commodities
- [ ] Implement feature monitoring and drift detection
- [ ] Set up feature lineage tracking with DataHub
- [ ] Enable feature versioning and A/B testing

## Resources

- **Feast Docs**: https://docs.feast.dev/
- **Feast Python SDK**: https://rtd.feast.dev/en/latest/
- **Feature Engineering Best Practices**: https://docs.feast.dev/getting-started/concepts



