# 254Carbon Event Producer Library

**Version**: 1.0.0  
**Languages**: Python, Node.js  
**Purpose**: Simplified event production for all platform services

---

## Overview

The Event Producer Library provides a simplified interface for all 254Carbon services to produce events to Kafka topics. It handles:

- ✅ Automatic event ID generation (UUID)
- ✅ Timestamp management
- ✅ Event schema validation
- ✅ Delivery tracking and callbacks
- ✅ Error handling and retries
- ✅ Metrics collection

## Installation

### Python

```bash
pip install -r requirements.txt
```

Or add to your `requirements.txt`:
```
confluent-kafka[avro]==2.3.0
```

### Node.js

```bash
npm install @254carbon/event-producer
# or
yarn add @254carbon/event-producer
```

Or add to your `package.json`:
```json
{
  "dependencies": {
    "kafkajs": "^2.2.4",
    "uuid": "^9.0.1"
  }
}
```

## Quick Start

### Python Example

```python
from event_producer import get_event_producer

# Initialize producer
producer = get_event_producer("my-service")

# Produce data ingestion event
producer.produce_data_ingestion_event(
    dataset_name="commodity_prices",
    record_count=10000,
    size_bytes=1024000,
    location="s3://bucket/data/commodity_prices.parquet",
    status="SUCCESS"
)

# Produce service health event
producer.produce_service_health_event(
    service_name="my-service",
    namespace="data-platform",
    health_status="HEALTHY",
    latency_ms=25,
    error_rate=0.001
)

# Close producer
producer.close()
```

### Node.js Example

```javascript
const { getEventProducer } = require('@254carbon/event-producer');

(async () => {
  // Initialize producer
  const producer = getEventProducer('my-service');
  await producer.connect();

  // Produce data ingestion event
  await producer.produceDataIngestionEvent({
    datasetName: 'commodity_prices',
    recordCount: 10000,
    sizeBytes: 1024000,
    location: 's3://bucket/data/commodity_prices.parquet',
    status: 'SUCCESS'
  });

  // Produce service health event
  await producer.produceServiceHealthEvent({
    serviceName: 'my-service',
    namespace: 'data-platform',
    healthStatus: 'HEALTHY',
    latencyMs: 25,
    errorRate: 0.001
  });

  // Disconnect
  await producer.disconnect();
})();
```

## Event Types

### Data Events

#### Data Ingestion Event
Records when data is ingested into the platform.

```python
producer.produce_data_ingestion_event(
    dataset_name="dataset_name",
    record_count=1000,
    size_bytes=50000,
    location="s3://bucket/path",
    status="SUCCESS",  # SUCCESS, FAILURE, PARTIAL
    format="parquet",
    metadata={"source": "external_api"}
)
```

#### Data Quality Event
Records data quality check results.

```python
producer.produce_data_quality_event(
    dataset_name="dataset_name",
    check_type="completeness",
    check_name="null_check",
    result="PASS",  # PASS, FAIL, WARNING
    score=0.99,
    failed_records=10,
    total_records=10000,
    message="Quality check passed"
)
```

### System Events

#### Service Health Event
Records service health status changes.

```python
producer.produce_service_health_event(
    service_name="my-service",
    namespace="data-platform",
    health_status="HEALTHY",  # HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN
    latency_ms=25,
    error_rate=0.001,
    message="All systems operational"
)
```

### Audit Events

#### API Call Event
Records all API calls for audit purposes.

```python
producer.produce_api_call_event(
    service="my-service",
    endpoint="/api/data",
    method="GET",
    status_code=200,
    latency_ms=50,
    request_size=256,
    response_size=2048,
    user_id="user-123",
    ip_address="10.0.1.100"
)
```

## Configuration

### Environment Variables

```bash
# Kafka configuration
export KAFKA_BOOTSTRAP_SERVERS="kafka-service.data-platform.svc.cluster.local:9092"
export KAFKA_SCHEMA_REGISTRY_URL="http://schema-registry.data-platform.svc.cluster.local:8081"

# Service identification
export SERVICE_NAME="my-service"
export SERVICE_NAMESPACE="data-platform"
```

### Advanced Configuration

```python
from event_producer import EventProducer

producer = EventProducer(
    bootstrap_servers="kafka-service:9092",
    schema_registry_url="http://schema-registry:8081",
    source_service="my-service"
)
```

## Integration Examples

### Flask API

```python
from flask import Flask, request
from event_producer import get_event_producer
import time

app = Flask(__name__)
producer = get_event_producer("api-service")

@app.route('/api/data', methods=['GET'])
def get_data():
    start = time.time()
    
    # Your API logic here
    result = {"status": "success"}
    
    # Record API call
    producer.produce_api_call_event(
        service="api-service",
        endpoint="/api/data",
        method=request.method,
        status_code=200,
        latency_ms=int((time.time() - start) * 1000),
        request_size=len(request.data),
        response_size=len(str(result)),
        user_id=request.headers.get('X-User-ID'),
        ip_address=request.remote_addr
    )
    
    return result

@app.teardown_appcontext
def shutdown_producer(exception=None):
    producer.close()
```

### Express.js API

```javascript
const express = require('express');
const { getEventProducer } = require('@254carbon/event-producer');

const app = express();
const producer = getEventProducer('api-service');

(async () => {
  await producer.connect();

  app.get('/api/data', async (req, res) => {
    const start = Date.now();
    
    // Your API logic here
    const result = { status: 'success' };
    
    // Record API call
    await producer.produceAPICallEvent({
      service: 'api-service',
      endpoint: '/api/data',
      method: req.method,
      statusCode: 200,
      latencyMs: Date.now() - start,
      requestSize: req.headers['content-length'] || 0,
      responseSize: JSON.stringify(result).length,
      userId: req.headers['x-user-id'],
      ipAddress: req.ip
    });
    
    res.json(result);
  });

  process.on('SIGTERM', async () => {
    await producer.disconnect();
    process.exit(0);
  });

  app.listen(3000);
})();
```

### Background Job

```python
from event_producer import get_event_producer
import time

def process_data(dataset_name, data_location):
    producer = get_event_producer("background-worker")
    
    try:
        # Process data
        record_count = process(data_location)
        
        # Record successful ingestion
        producer.produce_data_ingestion_event(
            dataset_name=dataset_name,
            record_count=record_count,
            size_bytes=get_file_size(data_location),
            location=data_location,
            status="SUCCESS"
        )
        
    except Exception as e:
        # Record failure
        producer.produce_data_ingestion_event(
            dataset_name=dataset_name,
            record_count=0,
            size_bytes=0,
            location=data_location,
            status="FAILURE",
            metadata={"error": str(e)}
        )
        raise
    
    finally:
        producer.close()
```

## Monitoring

### Metrics

The library tracks:
- `delivered`: Number of successfully delivered events
- `errors`: Number of failed deliveries

```python
stats = producer.get_stats()
print(f"Delivered: {stats['delivered']}, Errors: {stats['errors']}")
```

### Grafana Dashboard

A dedicated Grafana dashboard is available for monitoring event production:

**Dashboard**: Event-Driven Architecture  
**URL**: https://grafana.254carbon.com/d/events

**Metrics shown:**
- Event production rate by topic
- Consumer lag
- Failed event deliveries
- Event type distribution

## Best Practices

### 1. Use Context Managers (Python)

```python
with get_event_producer("my-service") as producer:
    producer.produce_data_ingestion_event(...)
# Producer automatically closed
```

### 2. Reuse Producer Instances

```python
# BAD: Creating new producer for each event
for item in items:
    producer = get_event_producer("service")
    producer.produce_data_ingestion_event(...)
    producer.close()

# GOOD: Reuse single producer
producer = get_event_producer("service")
for item in items:
    producer.produce_data_ingestion_event(...)
producer.close()
```

### 3. Handle Errors Gracefully

```python
try:
    producer.produce_data_ingestion_event(...)
except Exception as e:
    logger.error(f"Failed to produce event: {e}")
    # Continue with main logic
```

### 4. Include Correlation IDs

```python
event = producer._create_base_event(
    EventType.DATA_INGESTION,
    correlationId=request_id,
    causationId=parent_event_id
)
```

### 5. Flush Before Shutdown

```python
producer.flush(timeout=10.0)
producer.close()
```

## Troubleshooting

### Connection Refused

**Error**: `Failed to connect to Kafka`

**Solution**:
```bash
# Verify Kafka is running
kubectl get pods -n data-platform -l app=kafka

# Test connectivity
kubectl exec -it <your-pod> -- telnet kafka-service 9092
```

### Events Not Appearing

**Error**: Events produced but not visible in Kafka

**Solution**:
```bash
# Check topic exists
kubectl exec kafka-0 -- kafka-topics --list

# Check consumer lag
kubectl exec kafka-0 -- kafka-consumer-groups --describe --group <group>

# Verify flush was called
producer.flush()
```

### High Error Rate

**Error**: Many failed deliveries

**Solution**:
- Check Kafka broker status
- Verify topic configuration
- Check network policies allow traffic
- Review Kafka logs

## Testing

### Unit Tests

```python
import unittest
from unittest.mock import Mock, patch
from event_producer import EventProducer

class TestEventProducer(unittest.TestCase):
    @patch('event_producer.Producer')
    def test_produce_event(self, mock_producer):
        producer = EventProducer(source_service="test")
        producer.produce_data_ingestion_event(
            dataset_name="test",
            record_count=100,
            size_bytes=1000,
            location="s3://test",
            status="SUCCESS"
        )
        
        # Verify produce was called
        mock_producer.return_value.produce.assert_called_once()
```

### Integration Tests

```bash
# Start Kafka locally
docker-compose up -d kafka

# Run integration tests
python tests/integration_test.py
```

## Support

For issues or questions:
1. Check Kafka connectivity: `kubectl get pods -l app=kafka`
2. Review logs: `kubectl logs <your-pod>`
3. Check event dashboard in Grafana
4. Review event schemas in Schema Registry

## Version History

### 1.0.0 (October 2025)
- Initial release
- Python and Node.js implementations
- 12 Kafka topics
- 8 event types
- Full documentation

## License

MIT License - 254Carbon Platform Team



