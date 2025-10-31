# HMCo Analytics - dbt Models

Declarative analytics modeling for LMP, weather, and outages data using dbt.

## Overview

This dbt project transforms raw data into analytics-ready datasets for:
- Locational Marginal Pricing (LMP) analysis
- Weather-driven trading strategies
- Outage impact analysis

## Project Structure

```
analytics/dbt/
├── dbt_project.yml       # Project configuration
├── profiles.yml          # Connection profiles (Trino + ClickHouse)
├── models/
│   ├── staging/          # Raw → Clean transformations
│   │   └── stg_lmp_data.sql
│   └── marts/            # Business logic models
│       ├── lmp/          # LMP analytics
│       │   └── lmp_hourly_summary.sql
│       ├── weather/      # Weather features
│       │   └── weather_lmp_join.sql
│       └── outages/      # Outage analysis
├── tests/                # Custom data tests
├── macros/               # Reusable SQL macros
└── snapshots/            # Type-2 SCD snapshots
```

## Quick Start

### 1. Install dbt

```bash
pip install dbt-core dbt-trino dbt-clickhouse
```

### 2. Configure Profiles

Set your target environment:

```bash
export DBT_TARGET=dev  # or prod
export CLICKHOUSE_PASSWORD=<password>
```

### 3. Run Models

```bash
cd analytics/dbt

# Install dependencies
dbt deps

# Run all models
dbt run

# Run specific model
dbt run --select lmp_hourly_summary

# Run tests
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

## Model Descriptions

### Staging Models

#### `stg_lmp_data`
- Cleans and standardizes raw LMP data from various ISOs
- Filters out null prices and invalid timestamps
- Tags: `staging`, `lmp`

### Mart Models

#### `lmp_hourly_summary`
- Hourly aggregated LMP by location and ISO
- Includes min, max, avg, stddev of prices
- Materialized as table for fast queries
- Tags: `marts`, `lmp`

#### `weather_lmp_join`
- Enriches LMP data with weather features
- Enables weather-driven trading strategies
- Tags: `marts`, `weather`, `lmp`

## Data Quality Tests

All models include data quality tests:

```yaml
models:
  - name: lmp_hourly_summary
    columns:
      - name: location_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_locations')
              field: location_id
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - location_id
            - hour
```

## Deployment Targets

### Trino (Iceberg)
- Large-scale analytical queries
- Full SQL support
- Data lake integration

```yaml
trino_prod:
  type: trino
  host: trino-coordinator
  database: iceberg
  schema: analytics
```

### ClickHouse
- Sub-second aggregate queries
- Real-time dashboards
- High-concurrency workloads

```yaml
clickhouse_prod:
  type: clickhouse
  host: clickhouse-service
  database: analytics
```

## CI/CD Integration

dbt tests run automatically on PR:

```yaml
# .github/workflows/dbt-test.yml
- dbt deps
- dbt parse
- dbt compile
- dbt test
```

## Bridge to OpenMetadata

Models are automatically registered in OpenMetadata:

```python
# Post-run hook
dbt run
python scripts/sync_to_openmetadata.py
```

## Best Practices

1. **Incremental Models**: Use incremental materialization for large tables
2. **Documentation**: Document all models in `schema.yml`
3. **Tests**: Add tests for all critical columns
4. **Naming**: Use prefixes: `stg_` (staging), `fct_` (fact), `dim_` (dimension)
5. **Macros**: Extract repeated logic into macros

## Example Query

```sql
-- Query the hourly LMP mart
SELECT
  location_name,
  hour,
  avg_lmp,
  temperature
FROM {{ ref('weather_lmp_join') }}
WHERE hour >= '2024-01-01'
AND avg_lmp > 100
ORDER BY avg_lmp DESC
LIMIT 10
```

## Monitoring

Track dbt run metrics:
- Model execution time
- Test pass/fail rates
- Freshness checks

## References

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt-trino adapter](https://github.com/starburstdata/dbt-trino)
- [dbt-clickhouse adapter](https://github.com/ClickHouse/dbt-clickhouse)
