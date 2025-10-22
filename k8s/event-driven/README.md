# Event-Driven Architecture

**Platform**: 254Carbon Service Integration  
**Technology**: Apache Kafka + Schema Registry + Flink  
**Status**: Implementation in Progress  
**Updated**: October 22, 2025

---

## Overview

Transform the 254Carbon platform into an event-driven architecture enabling:

- **Asynchronous Communication**: Decoupled services via event streams
- **Event Sourcing**: Complete audit trail of all changes
- **CQRS**: Separate read/write models for performance
- **Stream Processing**: Real-time analytics and transformations
- **Event Replay**: Time-travel and debugging capabilities

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   Event Producers                          │
│  DataHub | Trino | Superset | DolphinScheduler | ...     │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────┐
│              Apache Kafka (3 Brokers)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ data-events  │  │system-events │  │ audit-events │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────┬──────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                          │
        ▼                          ▼
┌──────────────────┐    ┌──────────────────┐
│  Flink Stream    │    │  Event Consumers │
│   Processing     │    │   (Services)     │
└──────────────────┘    └──────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│     Materialized Views & Analytics       │
│  Iceberg | Doris | Elasticsearch        │
└──────────────────────────────────────────┘
```

## Event Categories

### 1. Data Events
- **data-ingestion**: New data ingested into platform
- **data-quality**: Data quality check results
- **data-transformation**: Data transformation completions
- **data-lineage**: Lineage graph updates

### 2. System Events
- **service-health**: Service health status changes
- **deployment-events**: Deployments and rollbacks
- **config-changes**: Configuration updates
- **security-events**: Authentication/authorization events

### 3. Audit Events
- **user-actions**: All user actions across platform
- **api-calls**: API invocations
- **data-access**: Data access patterns
- **admin-operations**: Administrative operations

## Event Schema Design

### Base Event Schema

All events inherit from this base schema:

```json
{
  "type": "record",
  "name": "BaseEvent",
  "namespace": "com.carbon254.events",
  "fields": [
    {"name": "eventId", "type": "string"},
    {"name": "eventType", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {"name": "source", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "correlationId", "type": ["null", "string"]},
    {"name": "causationId", "type": ["null", "string"]}
  ]
}
```

### Data Ingestion Event

```json
{
  "type": "record",
  "name": "DataIngestionEvent",
  "namespace": "com.carbon254.events.data",
  "fields": [
    {"name": "eventId", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {"name": "source", "type": "string"},
    {"name": "datasetName", "type": "string"},
    {"name": "recordCount", "type": "long"},
    {"name": "sizeBytes", "type": "long"},
    {"name": "format", "type": "string"},
    {"name": "location", "type": "string"},
    {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["SUCCESS", "FAILURE", "PARTIAL"]}},
    {"name": "metadata", "type": {"type": "map", "values": "string"}}
  ]
}
```

### Service Health Event

```json
{
  "type": "record",
  "name": "ServiceHealthEvent",
  "namespace": "com.carbon254.events.system",
  "fields": [
    {"name": "eventId", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {"name": "serviceName", "type": "string"},
    {"name": "namespace", "type": "string"},
    {"name": "healthStatus", "type": {"type": "enum", "name": "HealthStatus", "symbols": ["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"]}},
    {"name": "latencyMs", "type": "long"},
    {"name": "errorRate", "type": "double"},
    {"name": "message", "type": ["null", "string"]}
  ]
}
```

## Kafka Topics

### Topic Configuration

```yaml
# Production configuration for all topics
replication.factor: 3
min.insync.replicas: 2
retention.ms: 604800000  # 7 days
compression.type: lz4
cleanup.policy: delete
```

### Topic List

| Topic Name | Partitions | Retention | Purpose |
|------------|------------|-----------|---------|
| data-ingestion | 12 | 7 days | Data ingestion events |
| data-quality | 6 | 30 days | Quality check results |
| data-lineage | 3 | 90 days | Lineage updates |
| system-health | 3 | 7 days | Service health |
| deployment-events | 3 | 90 days | Deployments |
| config-changes | 3 | 365 days | Config updates |
| security-events | 6 | 365 days | Security events |
| audit-user-actions | 12 | 365 days | User actions |
| audit-api-calls | 12 | 30 days | API calls |
| audit-data-access | 12 | 90 days | Data access |

## Producer Implementation

### Python Producer Example

```python
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import uuid
import time

class EventProducer:
    def __init__(self, bootstrap_servers, schema_registry_url):
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'carbon254-producer',
            'compression.type': 'lz4',
            'linger.ms': 10,
            'batch.size': 16384
        })
        
        self.schema_registry = SchemaRegistryClient({
            'url': schema_registry_url
        })
    
    def produce_data_ingestion_event(self, dataset_name, record_count, 
                                     size_bytes, location, status):
        event = {
            'eventId': str(uuid.uuid4()),
            'timestamp': int(time.time() * 1000),
            'source': 'data-ingestion-service',
            'datasetName': dataset_name,
            'recordCount': record_count,
            'sizeBytes': size_bytes,
            'format': 'parquet',
            'location': location,
            'status': status,
            'metadata': {}
        }
        
        self.producer.produce(
            topic='data-ingestion',
            key=dataset_name,
            value=event,
            on_delivery=self._delivery_callback
        )
        self.producer.flush()
    
    def _delivery_callback(self, err, msg):
        if err:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')
```

### Node.js Producer Example

```javascript
const { Kafka } = require('kafkajs');

class EventProducer {
  constructor(brokers, clientId) {
    this.kafka = new Kafka({
      clientId,
      brokers,
      compression: 'lz4',
    });
    
    this.producer = this.kafka.producer({
      maxInFlightRequests: 5,
      idempotent: true,
    });
  }
  
  async connect() {
    await this.producer.connect();
  }
  
  async produceServiceHealthEvent(serviceName, namespace, healthStatus, latencyMs, errorRate) {
    const event = {
      eventId: generateUUID(),
      timestamp: Date.now(),
      serviceName,
      namespace,
      healthStatus,
      latencyMs,
      errorRate,
      message: null
    };
    
    await this.producer.send({
      topic: 'system-health',
      messages: [
        {
          key: serviceName,
          value: JSON.stringify(event),
        },
      ],
    });
  }
}
```

## Consumer Implementation

### Python Consumer Example

```python
from confluent_kafka import Consumer, KafkaError

class EventConsumer:
    def __init__(self, bootstrap_servers, group_id, topics):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000
        })
        
        self.consumer.subscribe(topics)
    
    def consume_events(self, handler):
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f'Error: {msg.error()}')
                        break
                
                event = json.loads(msg.value().decode('utf-8'))
                handler(event)
                
        finally:
            self.consumer.close()
```

## Stream Processing with Flink

### Real-time Aggregation

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema

def process_data_ingestion_events():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(3)
    
    # Consume from data-ingestion topic
    kafka_consumer = FlinkKafkaConsumer(
        topics='data-ingestion',
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': 'kafka-service:9092',
            'group.id': 'flink-processor'
        }
    )
    
    # Process events
    stream = env.add_source(kafka_consumer) \
        .map(lambda x: json.loads(x)) \
        .key_by(lambda x: x['datasetName']) \
        .window(TumblingProcessingTimeWindows.of(Time.minutes(5))) \
        .aggregate(AggregateFunction())
    
    # Write results
    kafka_producer = FlinkKafkaProducer(
        topic='data-ingestion-aggregated',
        serialization_schema=SimpleStringSchema(),
        producer_config={
            'bootstrap.servers': 'kafka-service:9092'
        }
    )
    
    stream.add_sink(kafka_producer)
    env.execute('Data Ingestion Aggregation')
```

## CQRS Implementation

### Command Side (Write Model)

```python
class CommandHandler:
    def __init__(self, event_producer):
        self.producer = event_producer
    
    async def handle_create_dataset(self, command):
        # Validate command
        if not self._validate_create_dataset(command):
            raise ValidationError("Invalid dataset creation command")
        
        # Execute command
        dataset = await self._create_dataset(command)
        
        # Produce event
        event = DatasetCreatedEvent(
            event_id=uuid.uuid4(),
            timestamp=time.time(),
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            created_by=command.user_id
        )
        
        await self.producer.produce('data-events', event)
        
        return dataset.id
```

### Query Side (Read Model)

```python
class QueryHandler:
    def __init__(self, read_store):
        self.read_store = read_store
    
    async def handle_get_dataset(self, query):
        # Read from optimized read model
        return await self.read_store.get_dataset(query.dataset_id)
    
    async def handle_search_datasets(self, query):
        # Search in Elasticsearch
        return await self.read_store.search_datasets(
            query.search_term,
            query.filters
        )
```

### Projection Builder

```python
class ProjectionBuilder:
    def __init__(self, event_consumer, read_store):
        self.consumer = event_consumer
        self.read_store = read_store
    
    async def build_projections(self):
        async for event in self.consumer.consume('data-events'):
            if event['eventType'] == 'DatasetCreated':
                await self._handle_dataset_created(event)
            elif event['eventType'] == 'DatasetUpdated':
                await self._handle_dataset_updated(event)
            elif event['eventType'] == 'DatasetDeleted':
                await self._handle_dataset_deleted(event)
    
    async def _handle_dataset_created(self, event):
        await self.read_store.insert_dataset({
            'id': event['dataset_id'],
            'name': event['dataset_name'],
            'created_at': event['timestamp'],
            'created_by': event['created_by']
        })
```

## Monitoring

### Kafka Metrics

```yaml
kafka_broker_messages_in_per_sec
kafka_broker_bytes_in_per_sec
kafka_broker_bytes_out_per_sec
kafka_broker_request_rate
kafka_broker_total_time_mean
kafka_consumer_lag
```

### Prometheus Alerts

```yaml
- alert: KafkaConsumerLag
  expr: kafka_consumer_lag > 10000
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High consumer lag on {{ $labels.topic }}"
```

## Best Practices

1. **Event Versioning**: Include version in all events
2. **Idempotency**: Design consumers to handle duplicate events
3. **Schema Evolution**: Use Schema Registry for compatibility
4. **Partitioning**: Partition by entity ID for ordering guarantees
5. **Error Handling**: Implement dead-letter queues
6. **Monitoring**: Track lag, throughput, and errors
7. **Replay**: Design for event replay capability
8. **Documentation**: Document all event types

## Resources

- **Kafka Documentation**: https://kafka.apache.org/documentation/
- **Schema Registry**: https://docs.confluent.io/platform/current/schema-registry/
- **Flink**: https://flink.apache.org/



