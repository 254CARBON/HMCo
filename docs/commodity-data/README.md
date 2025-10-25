# Commodity Data Platform Documentation

**Platform**: 254Carbon Commodity Data Analytics  
**Version**: 1.0.0  
**Updated**: October 21, 2025

---

## Overview

The 254Carbon platform is optimized for processing commodity, financial, and alternative data sources. It provides:

- **Automated data ingestion** from market APIs (crude oil, natural gas, electricity, LNG)
- **Economic indicator tracking** (FRED, World Bank, IMF)
- **Weather impact analysis** (NOAA forecasts)
- **GPU-accelerated analytics** (RAPIDS with 196GB GPU)
- **Data quality validation** (Apache Deequ)
- **Real-time dashboards** (Superset & Grafana)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  External Data Sources                                      │
│  ├─ FRED (Economic Indicators)                             │
│  ├─ EIA (Energy Prices)                                    │
│  ├─ NOAA (Weather Data)                                    │
│  ├─ World Bank (Commodity Indices)                         │
│  └─ Custom APIs/Files                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  SeaTunnel Connectors (Data Ingestion Layer)               │
│  ├─ HTTP/REST API connectors                              │
│  ├─ S3/MinIO file readers                                 │
│  ├─ Schema validation                                     │
│  └─ Retry logic                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Apache Iceberg Data Lake (MinIO Storage)                  │
│  ├─ commodity_data.energy_prices                          │
│  ├─ commodity_data.economic_indicators                    │
│  ├─ commodity_data.weather_forecasts                      │
│  └─ commodity_data.alternative_data                       │
└────────────────────┬────────────────────────────────────────┘
                     │
       ┌─────────────┴─────────────┐
       │                           │
┌──────▼──────┐            ┌──────▼───────┐
│   Deequ     │            │    Trino     │
│  Validation │            │  Query Engine│
│             │            │              │
│ - Completeness         │ - SQL Interface  │
│ - Validity             │ - Aggregations   │
│ - Freshness            │ - JOINs          │
└──────┬──────┘            └──────┬───────┘
       │                           │
       └─────────────┬─────────────┘
                     │
       ┌─────────────┴─────────────────┐
       │                               │
┌──────▼──────┐              ┌────────▼─────────┐
│  Superset   │              │  RAPIDS GPU      │
│  Dashboards │              │  Analytics       │
│             │              │                  │
│ - Market views           │ - Time series      │
│ - Quality metrics        │ - Anomaly detection│
│ - Alerts                 │ - Forecasting      │
└─────────────┘              └──────────────────┘
```

---

## Data Sources

### Configured Connectors

#### 1. Market Data (REST APIs)
**Purpose**: Real-time/daily commodity prices  
**Frequency**: Daily at 2 AM UTC  
**Sources**: ICE, CME, NYMEX  
**Data**: Crude oil (WTI, Brent), Natural gas, LNG, Electricity  

**Configuration**: `k8s/seatunnel/commodity-data-connectors.yaml`

#### 2. Economic Indicators (FRED API)
**Purpose**: Macroeconomic factors affecting commodities  
**Frequency**: Daily at 3 AM UTC  
**Sources**: Federal Reserve, World Bank, BLS  
**Indicators**:
- `DCOILWTICO`: WTI Crude Oil Spot Price
- `DHHNGSP`: Henry Hub Natural Gas Spot Price
- `DPROPANEMBTX`: Propane Prices
- `GASREGW`: Regular Gasoline Prices
- `ELECPRICE`: Electricity Price Index

**Configuration**: See `economic-data-connector.conf` in ConfigMap

#### 3. Weather Data (NOAA API)
**Purpose**: Weather impact on commodity production/transport  
**Frequency**: Every 4 hours  
**Coverage**: Houston, Chicago, New York, Gulf Coast  
**Data**: Temperature, wind speed, precipitation, forecasts  

**Configuration**: See `weather-data-connector.conf`

#### 4. Alternative Data (S3/MinIO)
**Purpose**: Custom data sources, proprietary feeds  
**Frequency**: Weekly (Sundays 4 AM UTC)  
**Format**: Parquet, CSV  
**Location**: `s3://commodity-data/alternative/`

---

## Workflows

### 1. Daily Market Data Ingestion
**Schedule**: Daily at 2 AM UTC  
**Duration**: ~15-30 minutes  
**Steps**:
1. Ingest crude oil prices
2. Ingest natural gas prices
3. Ingest electricity prices
4. Ingest LNG prices
5. Validate data completeness
6. Send success notification

**Polygon Provider MVP Enhancements**:
- Spark job `jobs/polygon_ingestion.py` persists Polygon.io OHLC data to `iceberg.raw.polygon_market_ohlc`.
- Deequ checks `jobs/polygon_quality_checks.py` append results to `iceberg.monitoring.polygon_quality_checks`.
- DataHub lineage and ownership are synced by `helm/charts/data-platform/charts/datahub/templates/polygon-lineage-ingestion.yaml`.

**Import**: See `COMMODITY_QUICKSTART.md`

### 2. Daily Economic Indicators
**Schedule**: Daily at 3 AM UTC  
**Duration**: ~10-20 minutes  
**Steps**:
1. Collect FRED indicators (5 series)
2. Collect World Bank commodity prices
3. Aggregate and compute summary statistics

### 3. Hourly Weather Data
**Schedule**: Every 4 hours  
**Duration**: ~5 minutes  
**Steps**:
1. Collect US weather forecasts (3 regions)
2. Analyze weather impact on commodities

### 4. Weekly Alternative Data
**Schedule**: Sundays at 4 AM UTC  
**Duration**: Variable (depends on data volume)  
**Steps**:
1. Scan MinIO for new files
2. Ingest and validate
3. Integrate with main datasets

### 5. Daily Data Quality Checks
**Schedule**: Daily at 6 AM UTC  
**Duration**: ~5-10 minutes  
**Steps**:
1. Check data freshness
2. Check data completeness  
3. Identify invalid records
4. Generate quality report

---

## Data Quality Framework

### Apache Deequ Validation

**What it checks:**
- **Completeness**: No missing required fields
- **Validity**: Values within expected ranges
- **Uniqueness**: No duplicate records
- **Consistency**: Cross-table data alignment
- **Freshness**: Data age within acceptable limits

**Validation Schedule**: Daily at 6 AM UTC via CronJob

**View Results**:
```bash
kubectl logs -n data-platform job/daily-data-quality-validation-<timestamp>
```

### Quality Metrics

Available in Prometheus:
- `commodity_data_completeness_percent{table="..."}`
- `commodity_data_null_value_rate{table="...", field="..."}`
- `commodity_data_quality_score{table="..."}`
- `commodity_data_duplicate_count{table="..."}`
- `commodity_data_last_ingestion_timestamp{table="..."}`

**View in Grafana**: "Commodity Data Quality" dashboard

---

## GPU Processing

### RAPIDS Environment

**Hardware**: 196GB GPU (8x Nvidia K80 with CUDA 11.4)  
**Allocation**: 2 GPUs per RAPIDS pod (expandable to 8)  
**Memory**: 64GB RAM per pod  
**CPU**: 16 cores limit

**Installed Libraries**:
- cuDF: GPU DataFrames
- cuML: Machine Learning
- cuGraph: Graph analytics  
- Dask-CUDA: Distributed computing
- PyIceberg: Iceberg integration
- Trino client: SQL access

### Access RAPIDS Jupyter

```bash
# Option 1: Via Cloudflare (after DNS configured)
https://rapids.254carbon.com

# Option 2: Port-forward
kubectl port-forward -n data-platform svc/rapids-service 8888:8888
# Access: http://localhost:8888
```

### Example Notebooks

Located in `/rapids/scripts/`:
- `time_series_analysis.py`: Volatility calculation, trend analysis
- `data_validation.py`: GPU-accelerated quality checks

### Running GPU Jobs

```python
# Example: Load and analyze commodity data
import cudf
from pyiceberg.catalog import load_catalog

# Load data into GPU memory
catalog = load_catalog('iceberg_catalog', uri='http://iceberg-rest-catalog:8181')
table = catalog.load_table('commodity_data.energy_prices')
gpu_df = cudf.from_pandas(table.scan().to_pandas())

# Fast operations on GPU
volatility = gpu_df.groupby('commodity')['price'].std()
print(volatility)
```

---

## Dashboards

### Superset Dashboards (5)

1. **Commodity Price Monitoring**
   - Current prices for all commodities
   - 30-day price trends
   - Price distribution by commodity
   - Daily price changes
   - Volatility indicators

2. **Data Pipeline Health**
   - Ingestion status
   - Data freshness by source
   - Records ingested per day
   - Quality scores
   - Failed ingestion attempts

3. **Economic Indicators**
   - FRED indicators trends
   - Correlation matrices
   - Latest indicator values
   - Data coverage by source

4. **Weather Impact Analysis**
   - Temperature trends by region
   - Precipitation probability
   - Weather alerts map
   - Wind speed warnings

5. **Data Quality Metrics**
   - Overall quality score
   - Null value rates
   - Anomalies detected
   - Duplicate records
   - Completeness by field

### Grafana Dashboards (4)

1. **Commodity Market Overview** (refresh: 5m)
2. **Commodity Pipeline Monitoring** (refresh: 30s)
3. **Commodity Data Quality** (refresh: 5m)
4. **GPU Processing Performance** (refresh: 30s)

---

## API Reference

### SeaTunnel REST API

```bash
# Execute connector
POST http://seatunnel-service:8080/execute
Content-Type: application/json

{
  "connector": "market-data",
  "commodity": "crude_oil",
  "date": "2025-10-21"
}

# List connectors
GET http://seatunnel-service:8080/connectors

# Check status
GET http://seatunnel-service:8080/status/{job_id}
```

### Trino SQL API

```bash
# Port-forward
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080

# Query via HTTP
curl -X POST http://localhost:8080/v1/statement \
  -H "X-Trino-User: admin" \
  -H "X-Trino-Catalog: iceberg_catalog" \
  -H "X-Trino-Schema: commodity_data" \
  -d "SELECT * FROM energy_prices LIMIT 10"
```

---

## Data Schema

### energy_prices

| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR | Commodity ticker/symbol |
| commodity | VARCHAR | Commodity type (crude_oil, natural_gas, etc.) |
| price | DOUBLE | Price value |
| volume | BIGINT | Trading volume |
| price_date | DATE | Trading date |
| location | VARCHAR | Geographic location/exchange |
| units | VARCHAR | Price units (USD/bbl, USD/MMBtu) |
| source | VARCHAR | Data source |
| ingestion_time | TIMESTAMP | When data was ingested |

### economic_indicators

| Column | Type | Description |
|--------|------|-------------|
| indicator_code | VARCHAR | FRED/WB indicator code |
| observation_date | DATE | Observation date |
| value | DOUBLE | Indicator value |
| source | VARCHAR | Data source (FRED, WORLD_BANK) |
| units | VARCHAR | Value units |
| ingestion_time | TIMESTAMP | When data was ingested |

### weather_forecasts

| Column | Type | Description |
|--------|------|-------------|
| location | VARCHAR | Geographic location |
| temperature | DOUBLE | Temperature (Celsius) |
| wind_speed | DOUBLE | Wind speed (km/h) |
| precipitation_probability | INTEGER | Precipitation probability (%) |
| forecast_time | TIMESTAMP | Forecast timestamp |
| ingestion_date | DATE | Ingestion date |
| ingestion_time | TIMESTAMP | When data was ingested |

---

## Best Practices

### Data Ingestion

1. **Use scheduled workflows** instead of manual runs
2. **Enable retry logic** for API failures (configured in workflows)
3. **Monitor API rate limits** to avoid throttling
4. **Validate data immediately** after ingestion
5. **Archive historical data** after 2 years

### Query Optimization

1. **Use date partitions** in WHERE clauses
2. **Limit result sets** for exploratory queries
3. **Create views** for common queries
4. **Use EXPLAIN** to optimize complex queries
5. **Leverage GPU** for large aggregations

### Data Quality

1. **Run daily validation** (automated via CronJob)
2. **Monitor quality scores** in dashboards
3. **Set up alerts** for quality degradation
4. **Document data issues** for stakeholders
5. **Implement data contracts** with upstream sources

---

## Troubleshooting

### No Data After Workflow Run

1. Check workflow logs in DolphinScheduler UI
2. Verify API keys are correct: `kubectl get secret seatunnel-api-keys -n data-platform -o yaml`
3. Test connector manually: `kubectl exec -n data-platform deploy/seatunnel-engine -- sh`
4. Check Iceberg catalog: `trino> SHOW TABLES IN commodity_data;`

### Slow Query Performance

1. Check Trino resource allocation
2. Verify data is partitioned by date
3. Use `EXPLAIN` to analyze query plan
4. Consider using RAPIDS for large aggregations
5. Create materialized views for common queries

### Data Quality Issues

1. Check Deequ validation job logs
2. Review quality metrics in Superset
3. Investigate upstream data source
4. Update validation rules if needed
5. Enable data quality alerts

---

## Performance Benchmarks

**Expected Performance** (with 788GB RAM, 88 cores, 196GB GPU):

| Operation | Expected Performance | Notes |
|-----------|---------------------|-------|
| Daily ingestion | 10,000+ records/min | Per connector |
| Query latency (P95) | < 5 seconds | Simple aggregations |
| Complex analytics | < 30 seconds | Multi-table JOINs |
| GPU processing | 100,000+ rows/sec | cuDF operations |
| Data quality validation | < 5 minutes | Full dataset |
| Dashboard refresh | < 10 seconds | All charts |

---

## Scaling Guidelines

### When to Scale Up

**Add CPU/Memory:**
- Query latency > 10 seconds
- DolphinScheduler workers at >80% CPU
- Trino workers at >80% memory

**Add GPU Resources:**
- RAPIDS jobs queuing
- GPU utilization > 90%
- Need more parallel analysis jobs

**Add Storage:**
- Iceberg tables > 1TB
- MinIO usage > 80%

### How to Scale

```bash
# Increase worker replicas
kubectl scale deployment dolphinscheduler-worker -n data-platform --replicas=8

# Add Trino workers
kubectl scale deployment trino-worker -n data-platform --replicas=4

# Allocate more GPUs to RAPIDS
kubectl set resources deployment/rapids-commodity-processor -n data-platform \
  --limits=nvidia.com/gpu=4

# Increase resource limits
kubectl set resources deployment/trino-coordinator -n data-platform \
  --requests=cpu=8000m,memory=16Gi \
  --limits=cpu=16000m,memory=32Gi
```

---

## Security

### API Key Management

**Storage**: Kubernetes secrets (base64 encoded)  
**Rotation**: Manual (quarterly recommended)  
**Access**: Limited to SeaTunnel pods only

**Update keys**:
```bash
kubectl edit secret seatunnel-api-keys -n data-platform
```

### Data Encryption

**At Rest**: MinIO encryption (if enabled)  
**In Transit**: TLS for all external API calls  
**In Cluster**: Unencrypted (trusted network)

### Access Control

**DolphinScheduler**: Username/password (change from default!)  
**Superset**: Username/password + RBAC  
**Grafana**: SSO via Cloudflare Access  
**Trino**: No authentication (internal only)  
**RAPIDS**: No authentication (configure via Cloudflare)

---

## Maintenance

### Daily Tasks
- Monitor workflow execution
- Check data quality dashboard
- Review alerts in Grafana
- Verify API rate limits

### Weekly Tasks
- Review data quality report
- Check for failed workflows
- Update commodity list if needed
- Archive old validation logs

### Monthly Tasks
- Update API keys if expiring
- Review resource utilization
- Optimize slow queries
- Update data retention policies

---

## Support

### Log Locations

```bash
# DolphinScheduler workflows
kubectl logs -n data-platform -l app=dolphinscheduler-master

# Data ingestion
kubectl logs -n data-platform -l app=seatunnel

# Data quality
kubectl logs -n data-platform -l app=data-quality-exporter

# GPU processing
kubectl logs -n data-platform -l app=rapids
```

### Metrics Endpoints

- **Prometheus**: http://localhost:9090 (port-forward from monitoring/prometheus)
- **Grafana**: https://grafana.254carbon.com
- **DolphinScheduler**: https://dolphinscheduler.254carbon.com
- **Superset**: https://superset.254carbon.com

### Quick Links

- **Quick Start**: [/COMMODITY_QUICKSTART.md](/COMMODITY_QUICKSTART.md)
- **Full Deployment**: [/COMMODITY_PLATFORM_DEPLOYMENT.md](/COMMODITY_PLATFORM_DEPLOYMENT.md)
- **Main README**: [/README.md](/README.md)

---

**Questions?** Check the comprehensive deployment guide in `COMMODITY_PLATFORM_DEPLOYMENT.md`
