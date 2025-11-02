# OpenAPI to UIS Importer

This tool automatically generates Unified Ingestion Spec (UIS) configurations from OpenAPI 3.0+ specifications. It parses API documentation and creates complete UIS specs with endpoints, field mappings, authentication, and sink configurations.

## Features

- **Automatic Endpoint Discovery**: Extract all API endpoints from OpenAPI specs
- **Smart Field Mapping**: Convert API response fields to database-friendly column names
- **Authentication Detection**: Automatically detect and configure API authentication
- **Pagination Support**: Detect and configure pagination parameters
- **Schema Validation**: Generate JSON Schema contracts from API responses
- **Prebuilt Templates**: Ready-to-use configurations for popular APIs
- **Multiple Output Formats**: Generate YAML or JSON UIS specifications

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Import from OpenAPI File

```bash
# Basic import
python importer.py examples/polygon-sample.yaml -o polygon-uis.yaml

# With Polygon template
python importer.py examples/polygon-sample.yaml -t polygon -o polygon-uis.yaml

# With ClickHouse sink
python importer.py examples/polygon-sample.yaml --clickhouse -o polygon-uis.yaml
```

### Import from OpenAPI URL

```bash
# Import from live API documentation
python importer.py https://api.example.com/openapi.yaml -o api-uis.yaml
```

### Programmatic Usage

```python
from uis.tools.openapi_import import OpenAPIImporter

# Import from file
importer = OpenAPIImporter()
result = importer.import_from_file('examples/polygon-sample.yaml')

if result['success']:
    uis_spec = result['uis_spec']

    # Export to YAML
    yaml_content = importer.export_to_yaml(result, 'output.yaml')

    # Export to JSON
    json_content = importer.export_to_json(result, 'output.json')

    print(f"Generated {result['metadata']['endpoints_count']} endpoints")
else:
    print(f"Import failed: {result['errors']}")
```

## Supported OpenAPI Features

### ✅ **Fully Supported**
- OpenAPI 3.0.0 - 3.1.0 specifications
- REST API endpoints (GET, POST, PUT, PATCH, DELETE)
- Request/response schemas
- Query parameters and path parameters
- Authentication schemes (API Key, Bearer, OAuth2)
- Pagination patterns (cursor, offset, page-based)

### ⚠️ **Partially Supported**
- File upload/download endpoints
- Complex authentication flows
- Webhook definitions

### 🚫 **Not Supported**
- OpenAPI 2.0 (Swagger) specifications
- GraphQL APIs
- Non-HTTP protocols

## Prebuilt Templates

The tool includes prebuilt templates for popular APIs:

### Polygon Stock API
```bash
python importer.py examples/polygon-sample.yaml -t polygon
```
**Features:**
- Real-time and historical stock data
- Iceberg and ClickHouse sinks
- 5-minute freshness target
- Market data tagging

### EIA Energy API
```bash
python importer.py examples/eia-sample.yaml -t eia
```
**Features:**
- Government energy data
- 6-hour batch processing
- Energy-specific tagging

### Alpha Vantage
```bash
python importer.py https://www.alphavantage.co/openapi.yaml -t alpha_vantage
```
**Features:**
- Stock, forex, and crypto data
- Rate limiting (5 requests/minute)
- Financial data tagging

### Weather APIs
```bash
python importer.py https://api.weather.com/openapi.yaml -t weather
```
**Features:**
- 30-minute freshness target
- ClickHouse for fast queries
- Weather-specific tagging

## Command Line Options

### Basic Options
```bash
python importer.py <openapi_file_or_url> [options]

Options:
  -o, --output PATH        Output UIS file path
  -f, --format {yaml,json} Output format (default: yaml)
  -t, --template TEMPLATE  Use prebuilt template (polygon, eia, alpha_vantage, weather)
```

### Configuration Options
```bash
  --provider-type TYPE     Override provider type (rest_api, websocket)
  --tenant TENANT         Tenant ID (default: default)
  --owner OWNER           Data owner (default: api-importer)
  --table TABLE           Default table name
  --namespace NAMESPACE   Default namespace
  --clickhouse            Enable ClickHouse sink
  --provider-tags TAG [...]  Override provider metadata tags
```

### Filtering Options
```bash
  --methods METHOD [...]      Filter by HTTP methods (GET, POST, etc.)
  --tags TAG [...]            Filter by API tags (alias: --filter-tags)
  --include-deprecated        Include deprecated endpoints
```

## Generated UIS Structure

The importer generates complete UIS specifications:

```yaml
version: "1.1"
name: "polygon-stock-api"
description: "Generated from Polygon Stock API v3.0.0"
provider:
  name: "polygon-api"
  display_name: "Polygon Stock API"
  provider_type: "rest_api"
  base_url: "https://api.polygon.io"
  credentials_ref: "vault://polygon-api/credentials"
  tenant_id: "trading-platform"
  owner: "trading-team@company.com"
  endpoints:
    - name: "get_v3_reference_tickers"
      path: "/v3/reference/tickers"
      method: "GET"
      auth: "api_key"
      query_params:
        market: "stocks"
        limit: "100"
      pagination: "cursor"
      response_path: "$.results"
      field_mapping:
        ticker: "ticker"
        name: "company_name"
        market: "market_type"
  sinks:
    - type: "iceberg"
      table_name: "market_data.polygon_tickers"
      config:
        catalog: "hive_prod"
        namespace: "market_data"
        warehouse: "s3://prod-warehouse/"
  schema_contract:
    type: "object"
    properties:
      ticker:
        type: "string"
      name:
        type: "string"
      market:
        type: "string"
  slos:
    freshness_target_minutes: 5
    accuracy_threshold: 0.99
    block_on_schema_drift: true
```

## Authentication Configuration

The importer automatically detects authentication from OpenAPI security schemes:

### API Key Authentication
```yaml
auth: "api_key"
auth_config:
  header_name: "X-API-Key"
  location: "header"
```

### Bearer Token
```yaml
auth: "bearer"
auth_config:
  header_name: "Authorization"
  token_format: "Bearer {{token}}"
```

### OAuth2
```yaml
auth: "oauth2"
auth_config:
  flow: "client_credentials"
  token_url: "https://api.example.com/oauth/token"
  scopes: ["read", "write"]
```

## Field Mapping

The importer automatically converts API field names to database-friendly names:

| API Field | Database Field | Notes |
|-----------|----------------|--------|
| `user_id` | `user_id` | Already snake_case |
| `fullName` | `full_name` | camelCase → snake_case |
| `createdAt` | `created_at` | camelCase → snake_case |
| `email_address` | `email_address` | Already snake_case |
| `API_KEY` | `api_key` | UPPER_CASE → snake_case |

### Nested Objects
```yaml
# API Response: {"user": {"id": 1, "profile": {"name": "John"}}}
# Generated mapping:
field_mapping:
  user_id: "user_id"
  user_profile_name: "user_profile_name"
```

## Pagination Support

The importer detects common pagination patterns:

### Cursor-based Pagination
```yaml
pagination: "cursor"
pagination_config:
  cursor_param: "cursor"
  page_size: 100
  max_pages: 1000
```

### Offset-based Pagination
```yaml
pagination: "offset"
pagination_config:
  offset_param: "offset"
  limit_param: "limit"
  page_size: 100
```

## Schema Validation

The importer generates JSON Schema contracts from OpenAPI response schemas:

```yaml
schema_contract:
  type: "object"
  properties:
    ticker:
      type: "string"
      minLength: 1
      maxLength: 10
    name:
      type: "string"
      minLength: 1
    market:
      type: "string"
      enum: ["stocks", "crypto", "forex"]
  required: ["ticker", "name"]
```

## Rate Limiting

The importer extracts rate limiting information from API descriptions:

```yaml
# From: "Rate limit: 100 requests per minute"
rate_limits:
  requests_per_minute: 100
```

## Development

### Running Tests

```bash
cd sdk/uis/tools/openapi_import
python test_importer.py
```

### Adding New Templates

1. Add template to `get_prebuilt_templates()` method
2. Define template-specific options (auth, sinks, schedules)
3. Add test cases for the new template
4. Update documentation

### Extending Parser

1. Add new OpenAPI feature to `extract_*` methods
2. Update validation in `validate_spec()`
3. Add tests for new features

## Integration Examples

### With Polygon API
```bash
# Generate UIS for Polygon Stock API
python importer.py examples/polygon-sample.yaml -t polygon -o polygon-uis.yaml

# The generated UIS includes:
# - API key authentication
# - Cursor-based pagination
# - Field mapping for stock data
# - Iceberg and ClickHouse sinks
# - 5-minute freshness SLO
```

### With EIA API
```bash
# Generate UIS for EIA Energy API
python importer.py https://api.eia.gov/openapi.yaml -t eia -o eia-uis.yaml

# The generated UIS includes:
# - Batch processing schedule (6 hours)
# - Energy data tagging
# - Government data compliance
```

## Troubleshooting

### Common Issues

1. **Missing Authentication**: Ensure OpenAPI spec includes security schemes
2. **Invalid Pagination**: Check pagination parameter names in API docs
3. **Schema Mismatch**: Verify response schema matches actual API responses
4. **Rate Limits**: Adjust rate limiting based on API documentation

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

importer = OpenAPIImporter()
result = importer.import_from_file('api.yaml')
```

## Advanced Usage

### Custom Field Mappings
```python
# Override default field mappings
options = {
    "field_mappings": {
        "api_field": "custom_db_field",
        "another_field": "renamed_field"
    }
}
```

### Custom Validation Rules
```python
# Add custom validation rules
options = {
    "validation_rules": {
        "required_fields": ["id", "name", "email"],
        "field_constraints": {
            "email": {"format": "email"},
            "age": {"minimum": 0, "maximum": 150}
        }
    }
}
```

## License

This tool is part of the HMCo data platform SDK.

