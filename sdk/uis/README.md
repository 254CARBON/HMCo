# Unified Ingestion Spec (UIS) 1.1 SDK

The Unified Ingestion Spec (UIS) is a declarative configuration format for defining data ingestion pipelines. This SDK provides parsing, validation, and compilation capabilities for UIS specifications.

## Features

- **Declarative Configuration**: Define data sources, transformations, and sinks in YAML/JSON
- **Type Safety**: Full Pydantic model validation
- **Multiple Ingestion Modes**: Batch, micro-batch, streaming, websocket, webhook support
- **Provider Agnostic**: Support for REST APIs, GraphQL, WebSockets, files, and more
- **Quality Gates**: Schema validation and SLO enforcement
- **Multi-Language Transforms**: WASM, Spark, Flink, Python, and SQL transforms

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Parse a UIS File

```python
from uis.parser import UISParser

parser = UISParser()
spec = parser.parse_file('examples/polygon-api.yaml')

print(f"Spec: {spec.name}")
print(f"Provider: {spec.provider.display_name}")
print(f"Endpoints: {len(spec.provider.endpoints)}")
```

### Validate a Specification

```python
from uis.validator import UISValidator

validator = UISValidator()
errors = validator.validate_completeness(spec)

if errors:
    print("Validation errors:", errors)
else:
    is_ready, issues = validator.is_production_ready(spec)
    print(f"Production ready: {is_ready}")
```

### Create a Spec Programmatically

```python
from uis.spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig
from uis.spec import ProviderType, SinkType, IngestionMode

spec = UnifiedIngestionSpec(
    version="1.1",
    name="my-ingestion-pipeline",
    description="Ingest data from external API",
    provider=ProviderConfig(
        name="external_api",
        display_name="External Data API",
        provider_type=ProviderType.REST_API,
        base_url="https://api.example.com",
        tenant_id="my-tenant",
        owner="data-team@company.com",
        endpoints=[
            EndpointConfig(
                name="users",
                path="/users",
                method="GET",
                pagination="cursor",
                response_path="$.data"
            )
        ],
        sinks=[{
            "type": SinkType.ICEBERG,
            "table_name": "raw.users",
            "config": {"warehouse": "prod"}
        }]
    ),
    created_by="data-engineer"
)
```

## UIS 1.1 Specification

### Provider Types

- `rest_api`: REST API endpoints
- `graphql`: GraphQL APIs
- `websocket`: WebSocket connections
- `webhook`: Webhook receivers
- `file_ftp`: File and FTP sources
- `database`: Database connections
- `kafka`: Kafka topics
- `s3`: S3-compatible object storage

### Ingestion Modes

- `batch`: Periodic batch processing
- `micro_batch`: Small batch processing (minutes)
- `streaming`: Real-time streaming
- `websocket`: Live WebSocket data
- `webhook`: Event-driven webhooks
- `graphql_subscription`: GraphQL subscriptions

### Sink Types

- `iceberg`: Apache Iceberg tables
- `clickhouse`: ClickHouse database
- `kafka`: Kafka topics
- `postgres`: PostgreSQL database
- `elasticsearch`: Elasticsearch indices

## Examples

See the `examples/` directory for complete UIS specifications:

- `polygon-api.yaml`: Stock market data ingestion
- `weather-api.yaml`: Weather data pipeline
- `streaming-sensor.yaml`: IoT sensor streaming

## Validation

The SDK provides comprehensive validation:

### Completeness Validation
- Required fields for production deployment
- Authentication configuration
- Sink configuration completeness

### Performance Validation
- Rate limiting consistency
- Parallelism recommendations
- Transform complexity analysis

### Security Validation
- Credential handling
- HTTPS enforcement
- Secret detection

## Development

### Running Tests

```bash
python test_parser.py
```

### Adding New Provider Types

1. Add the provider type to the `ProviderType` enum
2. Update the validation logic in `validator.py`
3. Add compilation logic in the compilers directory

## License

This SDK is part of the HMCo data platform and follows the same licensing terms.
