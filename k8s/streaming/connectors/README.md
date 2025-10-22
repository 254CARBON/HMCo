# Real-time Data Connectors

**Platform**: 254Carbon Advanced Analytics Platform  
**Component**: Real-time Data Pipelines  
**Technology**: Debezium CDC, WebSocket Gateway  
**Status**: Implementation Phase 1.2

---

## Overview

Real-time data connectors provide:

- **CDC (Change Data Capture)**: Stream database changes to Kafka in real-time
- **WebSocket Gateway**: Real-time data streaming to clients via WebSocket
- **Unified API**: Kong-managed streaming endpoints
- **Event Processing**: Flink jobs for real-time transformations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL Database                                        │
│  (Logical Replication)                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ CDC Stream
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Debezium PostgreSQL Connector                              │
│  (Kafka Connect)                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Events
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Apache Kafka                                               │
│  Topics: cdc.commodity_prices, cdc.economic_indicators      │
└──────┬──────────────────────────────────────────┬───────────┘
       │                                           │
       │ Consume                                   │ Consume
       ↓                                           ↓
┌──────────────────┐                    ┌──────────────────────┐
│  Flink Jobs      │                    │  WebSocket Gateway   │
│  (Processing)    │                    │  (Streaming API)     │
└──────────────────┘                    └──────────────────────┘
                                                   │
                                                   │ WebSocket
                                                   ↓
                                        ┌──────────────────────┐
                                        │  Web Clients         │
                                        └──────────────────────┘
```

## Components

### 1. Debezium PostgreSQL CDC Connector

Captures changes from PostgreSQL and streams to Kafka.

**Features**:
- Logical replication-based CDC
- Initial snapshot + continuous streaming
- Schema evolution support
- Transaction guarantees

**Topics Created**:
- `cdc.commodity_prices`: Price updates
- `cdc.economic_indicators`: Economic data changes
- `cdc.ml_models`: ML model deployments
- `cdc.feature_values`: Feature store updates

### 2. WebSocket Gateway

Provides real-time streaming API for clients.

**Features**:
- WebSocket connections (10,000 concurrent)
- JWT authentication
- Topic subscription model
- Auto-reconnection support
- Metrics and monitoring

### 3. Kafka Connect

Distributed connector framework.

**Features**:
- 3 workers for HA
- REST API for management
- Plugin system
- Monitoring via JMX

## Deployment

### Prerequisites

```bash
# Ensure Kafka Connect is deployed
kubectl get pods -n data-platform -l app=kafka-connect

# Ensure PostgreSQL has logical replication enabled
# Check postgresql.conf: wal_level = logical
```

### Deploy CDC Connector

```bash
# 1. Set up PostgreSQL for CDC
kubectl apply -f k8s/streaming/connectors/debezium-postgres-connector.yaml

# 2. Wait for setup job
kubectl wait --for=condition=complete job/debezium-postgres-setup -n data-platform --timeout=300s

# 3. Deploy connector
kubectl apply -f k8s/streaming/connectors/debezium-connector-deployment.yaml

# 4. Verify connector is running
kubectl exec -n data-platform kafka-0 -- \
  curl -s http://kafka-connect-service:8083/connectors/postgres-cdc-connector/status | jq .
```

### Deploy WebSocket Gateway

```bash
# Deploy WebSocket gateway
kubectl apply -f k8s/streaming/connectors/websocket-gateway.yaml

# Verify deployment
kubectl get pods -n data-platform -l app=websocket-gateway

# Check logs
kubectl logs -n data-platform -l app=websocket-gateway --tail=50
```

### Verify Installation

```bash
# Check Debezium connector status
curl http://kafka-connect-service.data-platform.svc.cluster.local:8083/connectors | jq .

# List CDC topics
kubectl exec -n data-platform kafka-0 -- \
  kafka-topics --bootstrap-server kafka-service:9092 --list | grep cdc

# Test WebSocket endpoint
kubectl port-forward -n data-platform svc/websocket-gateway 8080:8080

# In another terminal, test WebSocket
wscat -c "ws://localhost:8080/ws/stream?token=test" \
  -x '{"action":"subscribe","topics":["realtime_prices"]}'
```

## Usage

### Monitor Database Changes

```bash
# Consume CDC events from Kafka
kubectl exec -n data-platform kafka-0 -- \
  kafka-console-consumer \
    --bootstrap-server kafka-service:9092 \
    --topic cdc.commodity_prices \
    --from-beginning
```

### WebSocket Client Example

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://websocket-gateway.data-platform.svc.cluster.local:8080/ws/stream?token=your-jwt-token');

ws.on('open', () => {
  // Subscribe to real-time price updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    topics: ['realtime_prices', 'realtime_events']
  }));
});

ws.on('message', (data) => {
  const event = JSON.parse(data);
  console.log('Received:', event);
  
  // Handle different event types
  if (event.topic === 'realtime.commodity.prices') {
    updatePriceChart(event.value);
  }
});

ws.on('close', () => {
  console.log('Connection closed, reconnecting...');
  setTimeout(() => connectWebSocket(), 5000);
});
```

### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def stream_prices():
    uri = "ws://websocket-gateway.data-platform.svc.cluster.local:8080/ws/stream?token=your-jwt"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe
        await websocket.send(json.dumps({
            "action": "subscribe",
            "topics": ["realtime_prices"]
        }))
        
        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Price update: {data}")

asyncio.run(stream_prices())
```

### Insert Test Data

```bash
# Trigger CDC by inserting data
kubectl exec -n data-platform -it deploy/postgres-shared -- psql -U postgres -d datahub <<EOF
INSERT INTO commodity_prices (commodity_code, price, currency, source)
VALUES ('crude_oil_wti', 75.50, 'USD', 'test');

UPDATE commodity_prices 
SET price = 76.00 
WHERE commodity_code = 'crude_oil_wti';
EOF

# Check Kafka for CDC events
kubectl exec -n data-platform kafka-0 -- \
  kafka-console-consumer \
    --bootstrap-server kafka-service:9092 \
    --topic cdc.commodity_prices \
    --max-messages 2
```

## Kong Integration

Add WebSocket endpoint to Kong:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongService
metadata:
  name: websocket-stream
  namespace: kong
spec:
  host: websocket-gateway.data-platform.svc.cluster.local
  port: 8080
  protocol: http
---
apiVersion: configuration.konghq.com/v1
kind: KongRoute
metadata:
  name: websocket-stream-route
  namespace: kong
spec:
  service: websocket-stream
  paths:
  - /stream/ws
  protocols:
  - http
  - https
  - ws
  - wss
```

External access:
```
wss://api.254carbon.com/stream/ws
```

## Monitoring

### Debezium Metrics

```bash
# Check connector status
curl http://kafka-connect-service.data-platform.svc.cluster.local:8083/connectors/postgres-cdc-connector/status | jq .

# View connector metrics
curl http://kafka-connect-service.data-platform.svc.cluster.local:8083/metrics | grep debezium
```

### WebSocket Metrics

```bash
# Check active connections
curl http://websocket-gateway.data-platform.svc.cluster.local:8080/metrics

# View logs
kubectl logs -n data-platform -l app=websocket-gateway --tail=100 -f
```

### Kafka Lag Monitoring

```bash
# Check consumer lag
kubectl exec -n data-platform kafka-0 -- \
  kafka-consumer-groups \
    --bootstrap-server kafka-service:9092 \
    --describe \
    --group ws-client-group
```

## Performance Tuning

### Debezium Configuration

```json
{
  "max.queue.size": "16384",
  "max.batch.size": "4096",
  "poll.interval.ms": "100",
  "snapshot.fetch.size": "10240"
}
```

### WebSocket Configuration

```yaml
websocket:
  max_connections: 10000
  ping_interval: 30
  message_buffer: 1000
  compression: true
```

## Troubleshooting

### Connector Not Starting

```bash
# Check Kafka Connect logs
kubectl logs -n data-platform -l app=kafka-connect --tail=100

# Verify PostgreSQL replication slot
kubectl exec -n data-platform -it deploy/postgres-shared -- \
  psql -U postgres -d datahub -c "SELECT * FROM pg_replication_slots;"
```

### WebSocket Connection Issues

```bash
# Check gateway logs
kubectl logs -n data-platform -l app=websocket-gateway --tail=50

# Test connectivity
kubectl exec -n data-platform -it kafka-0 -- \
  nc -zv websocket-gateway 8080

# Verify JWT secret
kubectl get secret jwt-secret -n data-platform -o yaml
```

### Missing CDC Events

```bash
# Check if publication exists
kubectl exec -n data-platform -it deploy/postgres-shared -- \
  psql -U postgres -d datahub -c "SELECT * FROM pg_publication;"

# Check table configuration
kubectl exec -n data-platform -it deploy/postgres-shared -- \
  psql -U postgres -d datahub -c "\d commodity_prices"

# Restart connector
curl -X POST http://kafka-connect-service:8083/connectors/postgres-cdc-connector/restart
```

## Best Practices

1. **CDC Design**: Only capture necessary tables to reduce load
2. **Replication Slots**: Monitor slot lag to prevent WAL buildup
3. **WebSocket**: Implement client-side reconnection logic
4. **Authentication**: Always use JWT in production
5. **Rate Limiting**: Implement rate limits for WebSocket subscriptions
6. **Monitoring**: Track CDC lag and WebSocket connection count

## Next Steps

- [ ] Implement Flink feature engineering jobs (Phase 1.1 continued)
- [ ] Add more CDC connectors (MySQL, MongoDB)
- [ ] Implement WebSocket message filtering
- [ ] Add Server-Sent Events (SSE) support
- [ ] Create real-time dashboard examples
- [ ] Implement WebSocket authentication with Kong

## Resources

- **Debezium Docs**: https://debezium.io/documentation/
- **Kafka Connect**: https://kafka.apache.org/documentation/#connect
- **WebSocket API**: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket



