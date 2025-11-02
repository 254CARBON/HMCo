# UIS to Flink Compiler (Streaming)

This module provides compilation from Unified Ingestion Spec (UIS) specifications to Apache Flink job configurations for real-time streaming processing.

## Overview

The Flink compiler generates configurations for executing real-time streaming data processing pipelines using Apache Flink. It supports various streaming sources (WebSockets, Kafka, CDC) and sinks (Kafka, Iceberg, ClickHouse) with advanced state management and exactly-once processing guarantees.

## Features

- **Real-Time Streaming**: Optimized for continuous data processing with low latency
- **Multiple Sources**: WebSockets, Kafka, CDC, and webhook sources
- **Multiple Sinks**: Kafka topics, Iceberg tables, ClickHouse databases
- **State Management**: RocksDB state backend with checkpointing and savepoints
- **Exactly-Once Processing**: End-to-end exactly-once guarantees
- **Event Time Processing**: Watermarking and late data handling
- **Kubernetes Integration**: Native Kubernetes deployment support
- **Monitoring & Observability**: Built-in metrics and alerting

## Supported Provider Types

- `websocket`: WebSocket connections for real-time data
- `webhook`: HTTP webhook receivers for event-driven data
- `kafka`: Kafka streaming topics
- `database`: Change Data Capture (CDC) from databases

## Supported Sink Types

- `kafka`: Kafka topics with transactional writes
- `iceberg`: Apache Iceberg tables with streaming writes
- `clickhouse`: ClickHouse analytical database

## Usage

### Basic Compilation

```python
from uis.spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, IngestionMode, AuthType
from uis.compilers.flink import FlinkCompiler

# Create streaming UIS specification
spec = UnifiedIngestionSpec(
    version="1.1",
    name="realtime-analytics",
    provider=ProviderConfig(
        name="websocket_provider",
        display_name="WebSocket Analytics Provider",
        provider_type=ProviderType.WEBSOCKET,
        base_url="wss://api.marketdata.com",
        config={
            "taskmanager_memory": "8g",
            "task_slots": "4",
            "kafka_bootstrap_servers": "kafka:9092"
        },
        tenant_id="trading-platform",
        owner="trading-team@company.com",
        mode=IngestionMode.WEBSOCKET,
        parallelism=4,
        endpoints=[
            EndpointConfig(
                name="market_ticks",
                path="/ws/ticks",
                query_params={"protocol": "wss"},
                headers={"Authorization": "Bearer {{token}}"},
                field_mapping={
                    "timestamp": "event_time",
                    "symbol": "ticker",
                    "price": "last_price",
                    "volume": "trade_volume"
                },
                rate_limit_per_second=1000
            )
        ],
        sinks=[{
            "type": SinkType.KAFKA,
            "kafka_topic": "processed-ticks",
            "kafka_key_field": "symbol",
            "config": {
                "bootstrap_servers": "kafka:9092",
                "transactional": True
            }
        }]
    ),
    created_by="data-engineer"
)

# Compile to Flink configuration
compiler = FlinkCompiler()
config = compiler.compile(spec)

# Generate Flink SQL
flink_sql = compiler.compile_to_flink_sql(spec)

# Generate Flink job arguments
flink_args = compiler.compile_to_flink_args(spec)
```

### Flink Job Execution

Execute the generated configurations using the Flink job runner:

```bash
# Compile UIS spec to Flink config
python -c "
from uis.spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, IngestionMode, AuthType
from uis.compilers.flink import FlinkCompiler
import json

# Create and compile spec (as above)
compiler = FlinkCompiler()
config = compiler.compile(spec)

# Save config for execution
with open('streaming-config.json', 'w') as f:
    json.dump(config, f, indent=2)
"

# Execute Flink job
flink run \\
  --jobmanager flink-jobmanager:9081 \\
  --parallelism 4 \\
  --detached \\
  --job-name uis-realtime-analytics \\
  streaming-job.py
```

## Generated Configuration Structure

```json
{
  "job_type": "streaming",
  "flink_config": {
    "taskmanager.memory.process.size": "8g",
    "parallelism.default": "4",
    "state.backend": "rocksdb",
    "state.checkpointing.mode": "EXACTLY_ONCE",
    "execution.checkpointing.interval": "60000",
    "kafka.bootstrap.servers": "kafka:9092"
  },
  "sources": [
    {
      "type": "websocket",
      "name": "websocket_market_ticks",
      "url": "wss://api.marketdata.com/ws/ticks",
      "protocol": "wss",
      "headers": {"Authorization": "Bearer {{token}}"},
      "format": "json",
      "reconnect_config": {
        "max_attempts": 10,
        "initial_delay_ms": 1000,
        "max_delay_ms": 30000,
        "backoff_multiplier": 2.0
      }
    }
  ],
  "transforms": [
    {
      "type": "field_mapping",
      "name": "map_market_ticks",
      "mapping": {
        "timestamp": "event_time",
        "symbol": "ticker"
      }
    },
    {
      "type": "watermark",
      "name": "add_watermark",
      "timestamp_column": "event_time",
      "max_out_of_orderness_ms": 5000
    }
  ],
  "sinks": [
    {
      "type": "kafka",
      "name": "kafka_processed-ticks",
      "bootstrap_servers": "kafka:9092",
      "topic": "processed-ticks",
      "key_field": "symbol",
      "format": "json",
      "transactional": true
    }
  ],
  "state_management": {
    "backend": "rocksdb",
    "checkpoint_config": {
      "interval_ms": 60000,
      "timeout_ms": 300000,
      "mode": "EXACTLY_ONCE"
    }
  },
  "monitoring": {
    "metrics_enabled": true,
    "metrics_prefix": "uis.realtime-analytics"
  }
}
```

## Flink SQL Generation

The compiler can generate Flink SQL for table-based operations:

```sql
-- Generated Flink SQL
CREATE TABLE websocket_market_ticks (
    event_time TIMESTAMP(3),
    data STRING,
    metadata ROW<timestamp TIMESTAMP(3), source STRING>
) WITH (
    'connector' = 'websocket',
    'url' = 'wss://api.marketdata.com/ws/ticks',
    'format' = 'json'
);

CREATE TABLE kafka_processed_ticks (
    event_time TIMESTAMP(3),
    data STRING,
    metadata ROW<timestamp TIMESTAMP(3), source STRING>
) WITH (
    'connector' = 'kafka',
    'topic' = 'processed-ticks',
    'properties.bootstrap.servers' = 'kafka:9092',
    'format' = 'json'
);

INSERT INTO kafka_processed_ticks
SELECT
    event_time,
    data,
    ROW(event_time, 'flink') as metadata
FROM websocket_market_ticks;
```

## State Management

Advanced state management with exactly-once guarantees:

```json
{
  "state_management": {
    "backend": "rocksdb",
    "checkpoint_config": {
      "interval_ms": 60000,
      "timeout_ms": 300000,
      "mode": "EXACTLY_ONCE",
      "num_retained": 10,
      "externalized_retention": "RETAIN_ON_CANCELLATION"
    },
    "savepoint_config": {
      "dir": "s3://flink-savepoints/realtime-analytics/",
      "retention": "RETAIN_ON_CANCELLATION"
    },
    "state_ttl": {
      "enabled": true,
      "time_ms": 86400000,
      "cleanup_ms": 3600000
    }
  }
}
```

## WebSocket Integration

Real-time WebSocket data processing with reconnection logic:

```python
# WebSocket source configuration
source = {
    "type": "websocket",
    "name": "market_data_ws",
    "url": "wss://api.marketdata.com/ws/ticks",
    "protocol": "wss",
    "headers": {"Authorization": "Bearer {{token}}"},
    "reconnect_config": {
        "max_attempts": 10,
        "initial_delay_ms": 1000,
        "max_delay_ms": 30000,
        "backoff_multiplier": 2.0
    },
    "heartbeat": {
        "interval_ms": 30000,
        "timeout_ms": 10000
    }
}
```

## CDC Integration

Change Data Capture with Debezium for database changes:

```python
# CDC source configuration
source = {
    "type": "cdc",
    "name": "postgres_changes",
    "connector": "debezium",
    "database": "postgres",
    "hostname": "postgres",
    "database_name": "analytics",
    "table_names": ["users", "orders"],
    "debezium_config": {
        "snapshot.mode": "initial",
        "include.schema.changes": "false"
    }
}
```

## Monitoring and Observability

Built-in monitoring with Prometheus metrics:

```json
{
  "monitoring": {
    "metrics_enabled": true,
    "metrics_prefix": "uis.realtime-analytics",
    "prometheus": {
      "enabled": true,
      "port": 9999,
      "path": "/metrics"
    },
    "custom_metrics": [
      "source_throughput",
      "sink_throughput",
      "processing_latency_ms",
      "checkpoint_duration_ms",
      "state_size_bytes",
      "error_rate"
    ],
    "alerts": {
      "processing_delay_threshold_ms": 300000,
      "error_rate_threshold": 0.01,
      "checkpoint_failure_threshold": 3
    }
  }
}
```

## Development

### Running Tests

```bash
cd sdk/uis/compilers/flink
python test_compiler.py
```

### Adding New Source Types

1. Add source type to `_build_sources_config` method
2. Implement specific source builder (e.g., `_build_websocket_source`)
3. Add validation in `validate_config`
4. Update tests and documentation

### Adding New Transform Types

1. Add transform type to `_build_transforms_config` method
2. Implement transform logic in `apply_transforms` method
3. Add validation and error handling
4. Update tests and documentation

## Integration with Kubernetes

Deploy Flink jobs to Kubernetes:

```bash
# Submit job to Flink cluster
flink run \\
  --target kubernetes-session \\
  --parallelism 4 \\
  --job-name uis-realtime-analytics \\
  --job-config streaming-config.json \\
  flink-job.py
```

## Performance Tuning

### Memory Configuration
```python
# For high-throughput scenarios
config = {
    "taskmanager_memory": "16g",
    "task_slots": "8",
    "parallelism": "8",
    "state.checkpointing.interval": "30000"  # 30 seconds
}
```

### Network Tuning
```python
# For low-latency requirements
config = {
    "taskmanager.network.memory.fraction": "0.2",
    "taskmanager.network.memory.max": "2gb",
    "pipeline.operator-chaining": "true"
}
```

## Troubleshooting

### Common Issues

1. **Checkpoint Failures**: Check S3 connectivity and permissions
2. **Out of Memory**: Increase task manager memory or reduce parallelism
3. **WebSocket Reconnection**: Verify network connectivity and endpoint availability
4. **Schema Evolution**: Update schema contracts in UIS specification

### Debugging

Enable debug logging:
```python
flink_config = {
    "flink.debug.mode": "true",
    "logging.level": "DEBUG",
    "metrics.reporter.prometheus.enabled": "true"
}
```

## Integration with DolphinScheduler

1. **Create Streaming Task**: Add Flink streaming task in DolphinScheduler
2. **Configure Resources**: Upload job configuration as task resource
3. **Set Parameters**: Configure connection parameters and secrets
4. **Monitor**: Track execution through DolphinScheduler UI

## License

This compiler is part of the HMCo data platform SDK.

