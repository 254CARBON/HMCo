# Canonical ISO Data Model

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Owner**: Data Platform Team

## Overview

This document defines the canonical schema for Independent System Operator (ISO) market data across multiple markets (CAISO, MISO, SPP, with PJM, NYISO, and ERCOT to follow). The goal is to provide a unified data model that allows analytics to work across all ISOs without forking logic for each market.

## Design Principles

1. **Uniformity**: Same query works across all ISOs by filtering on `iso` field
2. **Backward Compatibility**: Existing queries continue to work
3. **Extensibility**: New ISOs can be added without schema changes
4. **Normalization**: All units standardized (MW for power, USD/MWh for prices)
5. **Time Consistency**: All timestamps in UTC

## Core Entities

### Real-Time LMP (Locational Marginal Pricing)

Unified schema for 5-minute real-time LMP data across all ISOs.

**Table**: `iso_rt_lmp`

| Field | Type | Unit | Description | Required |
|-------|------|------|-------------|----------|
| `ts` | DateTime | UTC | Timestamp of the pricing interval | Yes |
| `iso` | LowCardinality(String) | - | ISO market identifier (CAISO, MISO, SPP, etc.) | Yes |
| `node` | String | - | Pricing node/location identifier | Yes |
| `hub` | Nullable(String) | - | Hub identifier if applicable | No |
| `zone` | Nullable(String) | - | Load zone identifier | No |
| `lmp` | Float64 | USD/MWh | Total Locational Marginal Price | Yes |
| `energy_component` | Float64 | USD/MWh | Energy component of LMP | Yes |
| `congestion_component` | Float64 | USD/MWh | Congestion component of LMP | Yes |
| `loss_component` | Float64 | USD/MWh | Loss component of LMP | Yes |
| `market_run` | LowCardinality(String) | - | Market run type (RTM, RTPD, etc.) | No |
| `interval_minutes` | UInt8 | minutes | Interval duration (typically 5) | Yes |

**Primary Key**: `(iso, node, ts)`  
**Partition Key**: `toDate(ts)`  
**Sort Order**: `(iso, node, ts)`

### Day-Ahead LMP

Unified schema for day-ahead market prices.

**Table**: `iso_da_lmp`

| Field | Type | Unit | Description | Required |
|-------|------|------|-------------|----------|
| `ts` | DateTime | UTC | Timestamp of the pricing interval | Yes |
| `trade_date` | Date | UTC | Trading day | Yes |
| `iso` | LowCardinality(String) | - | ISO market identifier | Yes |
| `node` | String | - | Pricing node/location identifier | Yes |
| `hub` | Nullable(String) | - | Hub identifier if applicable | No |
| `zone` | Nullable(String) | - | Load zone identifier | No |
| `lmp` | Float64 | USD/MWh | Total Locational Marginal Price | Yes |
| `energy_component` | Float64 | USD/MWh | Energy component of LMP | Yes |
| `congestion_component` | Float64 | USD/MWh | Congestion component of LMP | Yes |
| `loss_component` | Float64 | USD/MWh | Loss component of LMP | Yes |
| `hour_ending` | UInt8 | hour | Hour of the trading day (1-24) | Yes |

**Primary Key**: `(iso, node, trade_date, hour_ending)`  
**Partition Key**: `trade_date`  
**Sort Order**: `(iso, node, trade_date, hour_ending)`

### Day-Ahead Awards

Unified schema for day-ahead market awards (cleared generation/load).

**Table**: `iso_da_award`

| Field | Type | Unit | Description | Required |
|-------|------|------|-------------|----------|
| `trade_date` | Date | UTC | Trading day | Yes |
| `hour_ending` | UInt8 | hour | Hour of the trading day (1-24) | Yes |
| `iso` | LowCardinality(String) | - | ISO market identifier | Yes |
| `resource_id` | String | - | Resource/unit identifier | Yes |
| `resource_name` | Nullable(String) | - | Resource name | No |
| `node` | String | - | Pricing node/location | Yes |
| `zone` | Nullable(String) | - | Load zone | No |
| `awarded_mw` | Float64 | MW | Awarded/cleared energy | Yes |
| `self_schedule_mw` | Float64 | MW | Self-scheduled energy | No |
| `price` | Float64 | USD/MWh | Clearing price | Yes |
| `resource_type` | LowCardinality(String) | - | Type (GENERATOR, LOAD, etc.) | No |

**Primary Key**: `(iso, resource_id, trade_date, hour_ending)`  
**Partition Key**: `trade_date`  
**Sort Order**: `(iso, trade_date, resource_id, hour_ending)`

## Node & Hub Mapping Tables

### Node Synonyms

Maps ISO-specific node names to canonical identifiers.

**Table**: `iso_node_mapping`

| Field | Type | Description |
|-------|------|-------------|
| `iso` | LowCardinality(String) | ISO market identifier |
| `native_node_id` | String | Original node ID from ISO |
| `canonical_node_id` | String | Standardized node identifier |
| `node_name` | String | Human-readable node name |
| `node_type` | LowCardinality(String) | Type (GEN, LOAD, HUB, etc.) |
| `latitude` | Float64 | Geographic latitude |
| `longitude` | Float64 | Geographic longitude |
| `zone` | Nullable(String) | Associated load zone |
| `hub` | Nullable(String) | Associated hub |
| `voltage_kv` | Nullable(UInt16) | Voltage level |
| `active` | UInt8 | 1 if currently active |

**Primary Key**: `(iso, native_node_id)`

### Hub Mapping

Standardized hub identifiers across ISOs.

**Table**: `iso_hub_mapping`

| Field | Type | Description |
|-------|------|-------------|
| `iso` | LowCardinality(String) | ISO market identifier |
| `native_hub_id` | String | Original hub ID from ISO |
| `canonical_hub_id` | String | Standardized hub identifier |
| `hub_name` | String | Human-readable hub name |
| `zone` | Nullable(String) | Associated load zone |
| `active` | UInt8 | 1 if currently active |

**Primary Key**: `(iso, native_hub_id)`

## ISO-Specific Mappings

### CAISO

- **Market Runs**: RTM (Real-Time Market), RTPD (Real-Time Pre-Dispatch), DAM (Day-Ahead)
- **Major Hubs**: SP15 (Southern California), NP15 (Northern California)
- **Interval**: 5 minutes for RT, hourly for DA

### MISO

- **Market Runs**: RTD (Real-Time Dispatch), RTEx (Real-Time Extended), DAM
- **Major Hubs**: Minnesota Hub, Indiana Hub, Illinois Hub, Michigan Hub
- **Interval**: 5 minutes for RT, hourly for DA

### SPP

- **Market Runs**: RTBM (Real-Time Balancing Market), SCED (Security Constrained Economic Dispatch), DAM
- **Major Hubs**: North Hub, South Hub
- **Interval**: 5 minutes for RT, hourly for DA

## Materialized Views

### Hourly Rollup

**View**: `iso_rt_lmp_hourly`

```sql
CREATE MATERIALIZED VIEW iso_rt_lmp_hourly AS
SELECT
  toStartOfHour(ts) AS hour,
  iso,
  node,
  hub,
  zone,
  avg(lmp) AS avg_lmp,
  min(lmp) AS min_lmp,
  max(lmp) AS max_lmp,
  avg(energy_component) AS avg_energy,
  avg(congestion_component) AS avg_congestion,
  avg(loss_component) AS avg_loss,
  count() AS sample_count
FROM iso_rt_lmp
GROUP BY hour, iso, node, hub, zone;
```

### Daily Hub Summary

**View**: `iso_da_hub_summary`

```sql
CREATE MATERIALIZED VIEW iso_da_hub_summary AS
SELECT
  trade_date,
  iso,
  hub,
  avg(lmp) AS avg_lmp,
  min(lmp) AS min_lmp,
  max(lmp) AS max_lmp,
  sum(lmp * (hour_ending <= 16 AND hour_ending > 6 ? 1 : 0)) / 10 AS on_peak_avg,
  sum(lmp * (hour_ending > 16 OR hour_ending <= 6 ? 1 : 0)) / 14 AS off_peak_avg,
  count() AS hour_count
FROM iso_da_lmp
WHERE hub IS NOT NULL
GROUP BY trade_date, iso, hub;
```

## Usage Examples

### Query RT LMP for CAISO SP15 Hub

```sql
SELECT 
  ts,
  node,
  lmp,
  congestion_component,
  loss_component
FROM iso_rt_lmp
WHERE iso = 'CAISO'
  AND hub = 'SP15'
  AND ts >= now() - INTERVAL 1 HOUR
ORDER BY ts DESC
LIMIT 100;
```

### Compare Average Prices Across ISOs

```sql
SELECT 
  iso,
  toStartOfDay(ts) AS day,
  avg(lmp) AS avg_lmp,
  avg(congestion_component) AS avg_congestion
FROM iso_rt_lmp
WHERE ts >= today() - INTERVAL 7 DAY
GROUP BY iso, day
ORDER BY iso, day;
```

### Find Congested Nodes (Congestion > $10/MWh)

```sql
SELECT 
  iso,
  node,
  avg(congestion_component) AS avg_congestion,
  count() AS occurrence_count
FROM iso_rt_lmp
WHERE ts >= now() - INTERVAL 1 DAY
  AND congestion_component > 10
GROUP BY iso, node
HAVING occurrence_count > 10
ORDER BY avg_congestion DESC;
```

## Data Quality Rules

1. **Completeness**: All required fields must be non-NULL
2. **Timeliness**: RT data arrives within 10 minutes of interval end
3. **Reasonableness**: 
   - LMP between -$1000 and $10000/MWh
   - Components sum to within $0.01 of LMP
4. **Consistency**: Node IDs validated against mapping tables
5. **Uniqueness**: No duplicate (iso, node, ts) tuples

## Migration Path

For existing tables:

1. Create new canonical tables with proper schema
2. Backfill historical data with transformation ETL
3. Update pipelines to write to new tables
4. Create views with old names pointing to new tables
5. Deprecate old tables after 90 days

## Future Extensions

- **ERCOT**: 15-minute SCED intervals, SPP market
- **PJM**: 5-minute dispatch, capacity market data
- **NYISO**: Zonal vs nodal pricing distinction
- **Ancillary Services**: Regulation, spinning reserve prices
- **Forward Markets**: Monthly, seasonal strips
