# Curve & Factor Library

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Owner**: Trading Analytics Team

## Overview

The Curve & Factor Library provides reusable forward curves and risk factors for trading analytics, risk management, and backtesting. All curves are snapshotted at EOD (End of Day) with Iceberg snapshot IDs for perfect reproducibility.

## Architecture

```
┌────────────────────────────────────────┐
│  Market Data Sources                   │
│  ├─ ISO DA/RT LMP                     │
│  ├─ NYMEX/ICE Futures                 │
│  ├─ Bilateral Trades                  │
│  └─ OTC Quotes                        │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│  EOD Curve Builder (DolphinScheduler)  │
│  ├─ Compute forward curves            │
│  ├─ Extract risk factors              │
│  └─ Snapshot to Iceberg               │
└────────────────┬───────────────────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
┌──────▼──────┐    ┌──────▼───────┐
│  Iceberg    │    │  ClickHouse  │
│  curves_eod │    │  curves_     │
│  (history)  │    │  latest      │
│             │    │  (fast)      │
└─────────────┘    └──────────────┘
```

## Curve Types

### 1. Forward Power Curves

Standardized power delivery periods (buckets).

**Buckets**:
- **5x16**: Monday-Friday, HE 7-22 (On-Peak)
- **7x8**: Daily, HE 23-6 (Off-Peak)
- **2x16**: Saturday-Sunday, HE 7-22 (Weekend Peak)
- **7x24**: Around-the-clock
- **HLH**: High Load Hours (typically HE 7-22, Mon-Fri)
- **LLH**: Low Load Hours (typically HE 23-6, all days + weekends)

**Terms**:
- Daily: Next 7 days
- Monthly: Next 12 months (Prompt month + 11)
- Quarterly: Next 8 quarters
- Seasonal: Summer (Apr-Sep), Winter (Oct-Mar)
- Calendar Years: Next 3 years

### 2. Natural Gas Curves

**Henry Hub** (NYMEX) and **Basis Spreads** to regional hubs.

**Terms**:
- Daily: Next 30 days
- Monthly: Next 24 months
- Seasonal: Summer/Winter strips
- Calendar Years: Next 5 years

### 3. Volatility Surfaces

Implied volatility by strike and expiry.

### 4. Correlation Matrices

Correlations between hubs, commodities, and factors.

## Data Schema

### EOD Curves (Iceberg)

**Table**: `curves_eod`

| Field | Type | Description |
|-------|------|-------------|
| `curve_date` | date | EOD date of the curve |
| `curve_id` | string | Unique curve identifier |
| `commodity` | string | POWER, GAS, LNG, etc. |
| `region` | string | ISO/hub (CAISO_SP15, HENRY_HUB) |
| `bucket` | string | 5x16, 7x8, 2x16, 7x24 |
| `delivery_start` | date | Delivery period start |
| `delivery_end` | date | Delivery period end |
| `term` | string | DAILY, MONTHLY, QUARTERLY, SEASONAL |
| `price` | double | Forward price (USD/MWh or USD/MMBtu) |
| `bid` | double | Bid price |
| `ask` | double | Ask price |
| `volume_mw` | double | Traded volume (if applicable) |
| `source` | string | ICE, NYMEX, BILATERAL, MODEL |
| `snapshot_id` | long | Iceberg snapshot ID |
| `created_at` | timestamp | Timestamp of curve build |

**Partition**: By `curve_date` and `commodity`  
**Sort**: (curve_date, curve_id, delivery_start)

### Latest Curves (ClickHouse)

**Table**: `curves_latest`

Materialized latest EOD curves for fast dashboard queries.

```sql
CREATE TABLE curves_latest (
  curve_date Date,
  curve_id String,
  commodity LowCardinality(String),
  region LowCardinality(String),
  bucket LowCardinality(String),
  delivery_start Date,
  delivery_end Date,
  term LowCardinality(String),
  price Float64 CODEC(T64, ZSTD),
  bid Float64 CODEC(T64, ZSTD),
  ask Float64 CODEC(T64, ZSTD),
  volume_mw Float64 CODEC(T64, ZSTD),
  source LowCardinality(String),
  snapshot_id Int64,
  created_at DateTime
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY curve_date
ORDER BY (curve_id, delivery_start);
```

### Risk Factors (Iceberg)

**Table**: `factors_eod`

Principal components, basis spreads, spark spreads, etc.

| Field | Type | Description |
|-------|------|-------------|
| `factor_date` | date | Date of the factor |
| `factor_id` | string | Factor identifier (PC1_POWER_WEST, BASIS_SP15_NP15) |
| `factor_type` | string | PCA, BASIS, SPREAD, CORRELATION |
| `value` | double | Factor value |
| `unit` | string | Unit (USD/MWh, correlation, etc.) |
| `metadata` | map<string,string> | Additional metadata |
| `snapshot_id` | long | Iceberg snapshot ID |

**Partition**: By `factor_date` and `factor_type`

## Curve Computation Specs

### Power Forward Curve (CAISO SP15)

**Inputs**:
- ISO DA LMP (historical, last 30 days)
- ICE power futures (if available)
- OTC broker quotes
- Fundamental model (load forecast, generation stack)

**Methodology**:
1. **Prompt Month**: DA LMP average + forward premium
2. **Month 2-3**: ICE futures if liquid, else load-weighted historical average + seasonal adjustment
3. **Month 4-12**: Seasonal average + long-term trend
4. **Cal Years**: Historical cal year average + inflation adjustment

**Standardization**:
- All prices in USD/MWh
- Buckets: 5x16, 7x8, 2x16 computed separately
- 7x24 = weighted average of 5x16 (16h * 5d) + 7x8 (8h * 7d) + 2x16 (16h * 2d)

### Natural Gas Forward Curve (Henry Hub)

**Inputs**:
- NYMEX NG futures (front 24 months)
- EIA storage reports
- Weather forecasts (for near-term)

**Methodology**:
1. **Prompt Month**: NYMEX front month
2. **Month 2-24**: NYMEX strip if available
3. **Beyond 24 months**: Long-term mean reversion model

### Basis Curves

**Methodology**:
- Basis = Regional Hub Price - Henry Hub Price
- Historical average basis by month
- Adjusted for pipeline capacity constraints

## Snapshot Management

### Recording Snapshots

```python
# Pseudo-code for EOD curve builder
curves_df = build_curves(curve_date)
table = iceberg.load_table("curves_eod")
table.append(curves_df)
snapshot_id = table.current_snapshot().snapshot_id

# Record snapshot metadata
snapshot_metadata = {
    "curve_date": curve_date,
    "snapshot_id": snapshot_id,
    "build_timestamp": datetime.utcnow(),
    "status": "SUCCESS"
}
write_to_metadata_table(snapshot_metadata)
```

### Reconstructing Historical Curves

```python
# Load curve from specific snapshot
table = iceberg.load_table("curves_eod")
historical_curves = table.snapshot_as_of_snapshot_id(snapshot_id).to_pandas()
```

### Latest Curves (Fast Path)

```python
# Query latest curves from ClickHouse
query = """
SELECT * FROM curves_latest
WHERE curve_date = (SELECT max(curve_date) FROM curves_latest)
"""
latest_curves = clickhouse_client.query(query).to_df()
```

## Standard Buckets

### Definition

| Bucket | Days | Hours | Total Hours/Week |
|--------|------|-------|------------------|
| 5x16 | Mon-Fri | HE 7-22 | 80 |
| 7x8 | All | HE 23-6 | 56 |
| 2x16 | Sat-Sun | HE 7-22 | 32 |
| 7x24 | All | All | 168 |

### Weighting Formula

```python
price_7x24 = (price_5x16 * 80 + price_7x8 * 56 + price_2x16 * 32) / 168
```

## Query Examples

### Get Latest Prompt Month 5x16 Price for CAISO SP15

```sql
SELECT price
FROM curves_latest
WHERE curve_id = 'CAISO_SP15_5x16'
  AND term = 'MONTHLY'
  AND delivery_start = toStartOfMonth(addMonths(today(), 1))
  AND curve_date = (SELECT max(curve_date) FROM curves_latest);
```

### Historical Curve Comparison (This Year vs Last Year)

```sql
WITH this_year AS (
  SELECT delivery_start, avg(price) AS price_2025
  FROM curves_eod
  WHERE curve_id = 'CAISO_SP15_5x16'
    AND term = 'MONTHLY'
    AND curve_date BETWEEN '2025-01-01' AND '2025-12-31'
  GROUP BY delivery_start
),
last_year AS (
  SELECT delivery_start, avg(price) AS price_2024
  FROM curves_eod
  WHERE curve_id = 'CAISO_SP15_5x16'
    AND term = 'MONTHLY'
    AND curve_date BETWEEN '2024-01-01' AND '2024-12-31'
  GROUP BY delivery_start
)
SELECT
  this_year.delivery_start,
  price_2025,
  price_2024,
  price_2025 - price_2024 AS price_change
FROM this_year
JOIN last_year USING (delivery_start);
```

### Extract Risk Factors for PCA

```sql
SELECT factor_date, value
FROM factors_eod
WHERE factor_id = 'PC1_POWER_WEST'
  AND factor_date >= today() - INTERVAL 1 YEAR
ORDER BY factor_date;
```

## Workflow (DolphinScheduler)

**DAG**: `10-curves-eod.json`

### Tasks

1. **collect_market_data** (00:00 UTC)
   - Fetch ISO DA LMP (yesterday)
   - Fetch NYMEX/ICE settlements
   - Fetch OTC quotes

2. **compute_curves** (00:15 UTC)
   - Run curve computation (Spark job)
   - Validate curves (no negative prices, monotonicity checks)

3. **snapshot_iceberg** (00:30 UTC)
   - Append to Iceberg `curves_eod`
   - Record snapshot ID

4. **publish_clickhouse** (00:45 UTC)
   - Write to ClickHouse `curves_latest`
   - Update materialized views

5. **compute_factors** (01:00 UTC)
   - PCA on price matrix
   - Correlation matrices
   - Basis spreads
   - Append to `factors_eod`

6. **validate_and_alert** (01:15 UTC)
   - Check for anomalies
   - Alert if curve moves > 3 sigma

## Data Quality

1. **No Negative Prices**: Curves must be >= 0
2. **Monotonicity**: Near-term should not be lower than far-term for storage commodities
3. **Consistency**: 7x24 = weighted average of sub-buckets
4. **Completeness**: All standard buckets present
5. **Freshness**: Curves published by 02:00 UTC daily

## Performance SLAs

- **Curve Build Time**: < 15 minutes
- **Snapshot Write**: < 5 minutes
- **ClickHouse Publish**: < 2 minutes
- **Total EOD Process**: < 2 hours (by 02:00 UTC)
- **Query Latency**: < 500ms for latest curves

## Future Enhancements

- **Intraday Curves**: Update curves every 4 hours
- **Volatility Surfaces**: Implied vol by strike/expiry
- **Multi-Commodity**: Add coal, emissions, renewables
- **ML Curves**: ML-based curve construction
- **Real-Time Marks**: Streaming updates from trading desk
