# UIS to SeaTunnel Compiler

This module provides compilation from Unified Ingestion Spec (UIS) specifications to SeaTunnel job configurations.

## Overview

SeaTunnel is a distributed, high-performance data integration platform that supports real-time streaming and batch data processing. This compiler translates UIS declarative specifications into SeaTunnel's JSON-based job configuration format.

## Features

- **REST API Sources**: HTTP endpoints with pagination, authentication, and rate limiting
- **File Sources**: CSV, JSON, XML files from various sources
- **Multiple Sinks**: Iceberg, ClickHouse, Kafka output support
- **Data Transforms**: Field mapping, schema validation, and custom transformations
- **Configuration Validation**: Comprehensive validation of generated configurations

## Supported Provider Types

- `rest_api`: REST API endpoints with JSON responses
- `file_ftp`: File-based data sources (CSV, JSON, XML)

## Supported Sink Types

- `iceberg`: Apache Iceberg tables with catalog integration
- `clickhouse`: ClickHouse database tables
- `kafka`: Kafka topics with JSON serialization

## Usage

### Basic Compilation

```python
from uis.spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, AuthType
from uis.compilers.seatunnel import SeaTunnelCompiler

# Create UIS specification
spec = UnifiedIngestionSpec(
    version="1.1",
    name="my-ingestion",
    provider=ProviderConfig(
        name="api_provider",
        display_name="API Provider",
        provider_type=ProviderType.REST_API,
        base_url="https://api.example.com",
        tenant_id="my-tenant",
        owner="team@example.com",
        endpoints=[
            EndpointConfig(
                name="users",
                path="/users",
                auth=AuthType.API_KEY,
                auth_config={"header_name": "Authorization"},
                pagination="cursor",
                response_path="$.data"
            )
        ],
        sinks=[{
            "type": SinkType.ICEBERG,
            "table_name": "users",
            "config": {"warehouse": "s3://warehouse/"}
        }]
    ),
    created_by="data-engineer"
)

# Compile to SeaTunnel configuration
compiler = SeaTunnelCompiler()
config = compiler.compile(spec)

# Save as JSON
json_config = compiler.compile_to_json(spec)
with open('seatunnel-job.json', 'w') as f:
    f.write(json_config)
```

### Configuration Validation

```python
# Validate generated configuration
errors = compiler.validate_config(config)
if errors:
    print("Configuration errors:", errors)
else:
    print("Configuration is valid")
```

## SeaTunnel Job Structure

Generated SeaTunnel configurations follow this structure:

```json
{
  "env": {
    "parallelism": 1,
    "checkpoint.interval": 10000,
    "checkpoint.timeout": 60000,
    "checkpoint.storage": "hdfs://namenode:9000/seatunnel/checkpoint"
  },
  "source": [
    {
      "plugin_name": "Http",
      "result_table_name": "source_data",
      "url": "https://api.example.com/data",
      "method": "GET",
      "headers": {"Authorization": "Bearer {{api_key}}"},
      "format": "json",
      "json_field": "$.results"
    }
  ],
  "transform": [
    {
      "plugin_name": "FieldMapper",
      "source_table_name": "source_data",
      "result_table_name": "mapped_data",
      "field_mapper": {"old_field": "new_field"}
    }
  ],
  "sink": [
    {
      "plugin_name": "Iceberg",
      "source_table_name": "mapped_data",
      "result_table_name": "iceberg_output",
      "catalog_name": "hive_prod",
      "table": "my_table",
      "warehouse": "s3://warehouse/"
    }
  ]
}
```

## Authentication

The compiler supports various authentication methods:

### API Key Authentication
```python
EndpointConfig(
    name="endpoint",
    path="/data",
    auth=AuthType.API_KEY,
    auth_config={
        "header_name": "X-API-Key",
        "header_format": "{{api_key}}"
    }
)
```

### Basic Authentication
```python
EndpointConfig(
    name="endpoint",
    path="/data",
    auth=AuthType.BASIC,
    auth_config={"username": "user", "password": "pass"}
)
```

## Pagination Support

### Cursor-based Pagination
```python
EndpointConfig(
    name="endpoint",
    path="/data",
    pagination="cursor",
    pagination_config={
        "cursor_param": "cursor",
        "page_size": 100,
        "max_pages": 1000
    }
)
```

### Offset-based Pagination
```python
EndpointConfig(
    name="endpoint",
    path="/data",
    pagination="offset",
    pagination_config={
        "offset_param": "offset",
        "limit_param": "limit",
        "page_size": 100
    }
)
```

## Rate Limiting

```python
EndpointConfig(
    name="endpoint",
    path="/data",
    rate_limit_per_second=10,
    rate_limit_group="api_group"
)
```

## Field Mapping

```python
EndpointConfig(
    name="endpoint",
    path="/data",
    field_mapping={
        "api_field_name": "database_column_name",
        "created_at": "created_timestamp",
        "user_id": "customer_id"
    }
)
```

## Development

### Running Tests

```bash
cd sdk/uis/compilers/seatunnel
python test_compiler.py
```

### Adding New Source Types

1. Add the source type to the `_build_source_config` method
2. Implement the specific source builder (e.g., `_build_rest_source`)
3. Add validation in `validate_config`
4. Update tests and documentation

### Adding New Sink Types

1. Add the sink type to the `_build_sink_config` method
2. Implement the specific sink builder (e.g., `_build_iceberg_sink`)
3. Add validation in `validate_config`
4. Update tests and documentation

## Integration with DolphinScheduler

Generated SeaTunnel configurations can be executed as DolphinScheduler tasks:

1. Save the compiled JSON configuration
2. Create a DolphinScheduler task with the SeaTunnel job type
3. Configure task parameters and scheduling
4. Monitor execution through the DolphinScheduler UI

## Troubleshooting

### Common Issues

1. **Missing Authentication**: Ensure `credentials_ref` is configured in the provider
2. **Pagination Errors**: Verify pagination configuration matches API documentation
3. **Schema Mismatch**: Check field mappings and response paths
4. **Rate Limiting**: Adjust `rate_limit_per_second` based on API limits

### Validation Errors

The compiler provides detailed validation of generated configurations. Common validation errors include:

- Missing `plugin_name` in source/sink configurations
- Invalid pagination parameters
- Missing required fields for specific sink types
- Inconsistent rate limiting configurations

## License

This compiler is part of the HMCo data platform SDK.

