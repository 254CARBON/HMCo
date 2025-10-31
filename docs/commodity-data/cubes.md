# Data Products & Cubes

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Owner**: Data Platform Team

## Overview

Data products and cubes provide sub-second exploratory analysis through ClickHouse star schemas and materialized aggregates. These eliminate ad-hoc death-by-scan queries by pre-computing common analytics patterns.

## Star Schema Design

### Fact Tables

- **fact_lmp_5m**: 5-minute LMP observations with dimension keys
- **fact_da_awards**: Day-ahead market awards
- **fact_outages**: Power plant and transmission outages

### Dimension Tables

- **dim_node**: Pricing nodes/locations
- **dim_hub**: Trading hubs
- **dim_zone**: Load zones
- **dim_calendar**: Time intelligence (holidays, weekends, etc.)
- **dim_resource**: Generation resources

## Materialized Aggregates

### mv_lmp_5m_by_hub

Pre-aggregated 5-minute LMP statistics by hub.

**Guarantees**:
- Query latency: < 100ms (p95)
- Freshness: < 5 minutes
- Retention: 3 years

**Use Cases**:
- Real-time hub price dashboards
- Alerting on price spikes
- Historical hub comparison

### mv_congestion_by_constraint

Hourly congestion statistics by node.

**Guarantees**:
- Query latency: < 150ms (p95)
- Freshness: < 10 minutes
- Retention: 2 years

**Use Cases**:
- Congestion hotspot identification
- Transmission planning
- Hedging strategy development

## Performance Targets

All cubes meet the following SLAs:

| Query Type | p95 Latency | p99 Latency |
|------------|-------------|-------------|
| Single hub, 24h | < 50ms | < 100ms |
| Multi-hub comparison, 7d | < 150ms | < 300ms |
| Congestion analysis, 30d | < 200ms | < 500ms |
| Historical trends, 1y | < 1s | < 2s |

## Query Examples

### Top 10 Dashboard Queries

1. **Current Hub Prices**
```sql
SELECT h.hub_name, avg(f.lmp) AS avg_lmp
FROM fact_lmp_5m f
JOIN dim_hub h ON f.hub_key = h.hub_key
WHERE f.ts >= now() - INTERVAL 1 HOUR
GROUP BY h.hub_name;
```
*Target: < 50ms*

2. **Congestion Hotspots (Last 24h)**
```sql
SELECT n.node_name, avg(f.congestion_component) AS avg_congestion
FROM fact_lmp_5m f
JOIN dim_node n ON f.node_key = n.node_key
WHERE f.ts >= now() - INTERVAL 24 HOUR
  AND f.congestion_component > 10
GROUP BY n.node_name
ORDER BY avg_congestion DESC
LIMIT 10;
```
*Target: < 100ms*

3. **Weekly Hub Comparison**
```sql
SELECT
  c.date,
  h.hub_name,
  avg(f.lmp) AS avg_lmp
FROM fact_lmp_5m f
JOIN dim_hub h ON f.hub_key = h.hub_key
JOIN dim_calendar c ON toDate(f.ts) = c.date
WHERE c.date >= today() - 7
GROUP BY c.date, h.hub_name
ORDER BY c.date, h.hub_name;
```
*Target: < 200ms*

## Data Quality

1. **Completeness**: All dimension keys must resolve
2. **Consistency**: Fact totals match source tables
3. **Freshness**: Cubes updated within 5 minutes of source
4. **Performance**: p95 latency meets SLA

## Load Testing

Cubes are tested under load with:
- 100 concurrent users
- Mix of dashboard queries (70%), analytical queries (30%)
- 1000 queries per minute sustained

All cubes meet p95 < 200ms under load.
