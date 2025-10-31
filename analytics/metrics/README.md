# Semantic Metrics Layer

This directory contains the **governed KPI metrics** for HMCo's data platform. These metrics are the single source of truth for key performance indicators used across all BI tools, notebooks, and analytics applications.

## Purpose

**Stop metric drift.** Ensure that "P&L driver X" means the same thing everywhere.

## Canonical Metrics

### 1. LMP Spread
**Definition:** Difference between locational marginal price and hub reference price  
**Owner:** trading-analytics  
**SLA:** 60 minutes  
**Usage:** All LMP spread analyses must reference `{{ ref('lmp_spread') }}`

### 2. Nodal Congestion Factor
**Definition:** Congestion component as a factor of total LMP  
**Owner:** trading-analytics  
**SLA:** 60 minutes  
**Usage:** All congestion analyses must reference `{{ ref('nodal_congestion_factor') }}`

### 3. Degree Day Delta
**Definition:** Variance from historical average degree days (HDD/CDD)  
**Owner:** weather-analytics  
**SLA:** 360 minutes  
**Usage:** All weather normalization must reference `{{ ref('degree_day_delta') }}`

### 4. Load Factor
**Definition:** Ratio of average load to peak load over time period  
**Owner:** demand-analytics  
**SLA:** 60 minutes  
**Usage:** All load utilization analyses must reference `{{ ref('load_factor') }}`

### 5. Data Freshness %
**Definition:** Percentage of datasets delivered within SLA  
**Owner:** data-platform  
**SLA:** 15 minutes  
**Usage:** All data quality dashboards must reference `{{ ref('data_freshness_pct') }}`

## Architecture

```
analytics/metrics/
├── dbt_project.yml          # Metric definitions and semantic layer config
├── models/
│   └── core_kpis/           # Canonical KPI models
│       ├── lmp_spread.sql
│       ├── nodal_congestion_factor.sql
│       ├── degree_day_delta.sql
│       ├── load_factor.sql
│       ├── data_freshness_pct.sql
│       └── schema.yml       # Column tests and documentation
└── tests/
    └── test_metric_consistency.sql  # Cross-metric validation

```

## Lineage Integration

All metrics are registered in **OpenMetadata** with full lineage tracking:
- Source tables (Trino/ClickHouse)
- dbt models
- Downstream BI dashboards
- Notebook usage

## Breaking Change Protection

**CI will block PRs that change metric logic without explicit approval.**

Tests in `tests/test_metric_consistency.sql` validate that:
1. Metric SQL produces consistent results
2. Schema changes don't break downstream consumers
3. Value ranges remain reasonable

## Usage in BI Tools

### Superset
```sql
SELECT * FROM metrics.lmp_spread
WHERE metric_timestamp >= CURRENT_DATE - INTERVAL '7' DAY
```

### Jupyter Notebooks
```python
from hmco.metrics import get_metric

df = get_metric('lmp_spread', start_date='2025-01-01')
```

### Trino Queries
```sql
SELECT iso, AVG(lmp_spread) 
FROM iceberg.metrics.lmp_spread
WHERE metric_timestamp >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
GROUP BY iso
```

## Development Workflow

1. **Add New Metric**: Create SQL file in `models/core_kpis/`
2. **Document**: Add to `schema.yml` with tests
3. **Register Metric**: Update `dbt_project.yml` metrics section
4. **Add Tests**: Create validation tests in `tests/`
5. **CI Check**: Breaking change tests run automatically
6. **Register in OpenMetadata**: Lineage hook runs post-deploy

## Deployment

Metrics are materialized as **tables** in the `metrics` schema:
- **Trino**: `iceberg.metrics.<metric_name>`
- **ClickHouse**: `metrics.<metric_name>` (replicated from Trino)

Refresh schedule:
- **Real-time metrics** (LMP, congestion): Every 5 minutes
- **Batch metrics** (degree days, load factor): Hourly
- **Platform metrics** (freshness): Every 15 minutes

## Governance

**Metric changes require:**
1. PR approval from metric owner (see `schema.yml`)
2. Passing CI tests (consistency + breaking change detection)
3. OpenMetadata lineage validation
4. Downstream impact analysis

**Contact:** data-platform@254carbon.com
