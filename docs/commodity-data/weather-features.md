# Weather Feature Factory

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Owner**: Data Platform Team

## Overview

The Weather Feature Factory transforms raw NOAA weather data (HRRR, GFS models) into tradeable signals and features useful for power and natural gas trading. It uses H3 hexagonal spatial indexing to aggregate weather observations by load zones and corridors.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Raw Weather Sources                        │
│  ├─ HRRR (High-Resolution Rapid Refresh)   │
│  ├─ GFS (Global Forecast System)           │
│  └─ NOAA Observations                      │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  Ingestion (Hourly)                        │
│  └─ Iceberg: weather_raw                   │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  PostGIS + H3 Spatial Processing           │
│  ├─ H3 hexagonal binning (res 7)          │
│  ├─ Zone-to-hex mapping                   │
│  └─ Corridor aggregation                  │
└────────────────┬────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
┌──────▼──────┐    ┌──────▼────────┐
│  Iceberg    │    │  ClickHouse   │
│  features_  │    │  features_    │
│  weather    │    │  weather_     │
│  (offline)  │    │  online       │
└─────────────┘    └───────────────┘
```

## H3 Spatial Index

We use H3 (Hexagonal Hierarchical Spatial Index) at multiple resolutions:

| Resolution | Avg Edge Length | Use Case |
|------------|-----------------|----------|
| 7 | ~5.7 km | Load zone aggregation |
| 8 | ~2.2 km | Sub-zone features |
| 9 | ~0.9 km | Weather station coverage |

### Why H3?

- **Uniform Coverage**: Hexagons provide better spatial uniformity than squares
- **Hierarchical**: Natural aggregation from fine to coarse resolution
- **Efficient Neighbors**: Each hex has exactly 6 neighbors (except pentagons)
- **Industry Standard**: Used by Uber, Meta, AWS for geospatial analytics

## Data Tables

### Raw Weather (Iceberg)

**Table**: `weather_raw`

| Field | Type | Description |
|-------|------|-------------|
| `ts` | timestamp | Observation/forecast timestamp (UTC) |
| `source` | string | Source system (HRRR, GFS, NOAA) |
| `latitude` | double | Latitude |
| `longitude` | double | Longitude |
| `temperature_f` | double | Temperature (Fahrenheit) |
| `temperature_c` | double | Temperature (Celsius) |
| `dew_point_f` | double | Dew point (Fahrenheit) |
| `humidity_pct` | double | Relative humidity (%) |
| `wind_speed_mph` | double | Wind speed (mph) |
| `wind_direction_deg` | int | Wind direction (degrees) |
| `pressure_mb` | double | Atmospheric pressure (mb) |
| `cloud_cover_pct` | int | Cloud cover (%) |
| `precipitation_in` | double | Precipitation (inches) |
| `solar_radiation_wm2` | double | Solar radiation (W/m²) |
| `forecast_hour` | int | Forecast hour (0 = observation) |

**Partition**: By date and source  
**Retention**: 2 years raw, 10 years aggregated

### Feature Weather (Iceberg)

**Table**: `features_weather`

Enriched weather data with H3 index and derived features.

| Field | Type | Description |
|-------|------|-------------|
| `ts` | timestamp | Timestamp (UTC) |
| `h3_index_res7` | string | H3 hex index at resolution 7 |
| `h3_index_res8` | string | H3 hex index at resolution 8 |
| `zone` | string | Load zone (CAISO_SP15, MISO_IND, etc.) |
| `latitude` | double | Hex center latitude |
| `longitude` | double | Hex center longitude |
| `temperature_f` | double | Temperature (°F) |
| `cdd` | double | Cooling degree days (base 65°F) |
| `hdd` | double | Heating degree days (base 65°F) |
| `wind_speed_mph` | double | Wind speed |
| `wind_power_density` | double | Wind power density (W/m²) |
| `cloud_cover_pct` | int | Cloud cover |
| `solar_capacity_factor` | double | Estimated solar capacity factor |
| `humidity_pct` | double | Relative humidity |
| `precipitation_in` | double | Precipitation |
| `feels_like_f` | double | Heat index / wind chill |
| `forecast_hour` | int | Forecast hour |
| `rolling_24h_avg_temp` | double | 24-hour rolling avg temperature |
| `temp_change_24h` | double | Temperature change vs 24h ago |
| `source` | string | Source system |

**Partition**: By date and zone  
**Sort**: (zone, h3_index_res7, ts)

### Online Weather Features (ClickHouse)

**Table**: `features_weather_online`

Real-time queryable weather features for dashboards and APIs.

```sql
CREATE TABLE features_weather_online (
  ts DateTime CODEC(Delta, ZSTD),
  zone LowCardinality(String),
  h3_index String,
  temperature_f Float64,
  cdd Float64,
  hdd Float64,
  wind_speed_mph Float64,
  solar_capacity_factor Float64,
  forecast_hour UInt8,
  rolling_24h_avg_temp Float64,
  temp_change_24h Float64
)
ENGINE = ReplicatedMergeTree()
PARTITION BY toDate(ts)
ORDER BY (zone, h3_index, ts)
TTL toDate(ts) + INTERVAL 30 DAY DELETE;
```

## Feature Definitions

### Cooling Degree Days (CDD)

Measure of cooling demand. Base temperature 65°F.

```python
CDD = max(0, temperature_f - 65)
```

**Use Case**: Predict electricity demand for air conditioning in summer.

### Heating Degree Days (HDD)

Measure of heating demand. Base temperature 65°F.

```python
HDD = max(0, 65 - temperature_f)
```

**Use Case**: Predict natural gas demand for heating in winter.

### Wind Power Density

Estimated available wind power per unit area.

```python
wind_power_density = 0.5 * air_density * wind_speed^3
```

where `air_density ≈ 1.225 kg/m³` at sea level.

**Use Case**: Forecast wind generation, impact on renewable penetration.

### Solar Capacity Factor

Estimated fraction of nameplate solar capacity available.

```python
solar_capacity_factor = (1 - cloud_cover_pct / 100) * 
                        max(0, cos(solar_zenith_angle))
```

**Use Case**: Forecast solar generation.

### Feels Like Temperature

Apparent temperature accounting for wind chill or heat index.

```python
if temperature_f <= 50 and wind_speed_mph > 3:
    # Wind chill
    feels_like = 35.74 + 0.6215*T - 35.75*V^0.16 + 0.4275*T*V^0.16
else if temperature_f >= 80 and humidity_pct > 40:
    # Heat index
    feels_like = -42.379 + 2.04901523*T + 10.14333127*RH - ...
else:
    feels_like = temperature_f
```

**Use Case**: Demand modeling for extreme weather.

## Zone-to-H3 Mapping

**PostGIS Table**: `iso_zone_h3_mapping`

Maps ISO load zones to H3 hexagons.

```sql
CREATE TABLE iso_zone_h3_mapping (
  zone VARCHAR(50),
  h3_index_res7 VARCHAR(20),
  h3_index_res8 VARCHAR(20),
  geom GEOMETRY(POLYGON, 4326),
  PRIMARY KEY (zone, h3_index_res7)
);

-- Index for spatial queries
CREATE INDEX idx_zone_h3_geom ON iso_zone_h3_mapping USING GIST (geom);
```

**Seed Data**: Load zone polygons for CAISO, MISO, SPP.

## Feature Jobs

### HRRR/GFS Ingestion Job

**Schedule**: Hourly at :00  
**Source**: NOAA S3 buckets  
**Target**: Iceberg `weather_raw`  
**Tool**: Apache Spark with H3-Spark

```python
# Pseudo-code
raw_weather = spark.read.grib(noaa_hrrr_path)
raw_weather = raw_weather.withColumn("h3_res7", h3_point_to_cell(lat, lon, 7))
raw_weather.write.format("iceberg").mode("append").save("weather_raw")
```

### Feature Derivation Job

**Schedule**: Hourly at :15  
**Source**: Iceberg `weather_raw`  
**Target**: Iceberg `features_weather`, ClickHouse `features_weather_online`

```python
# Pseudo-code
weather = spark.read.format("iceberg").load("weather_raw")
weather = weather.filter(hour == current_hour)

# Join with zone mapping
weather = weather.join(zone_mapping, on="h3_res7")

# Compute derived features
weather = weather.withColumn("cdd", compute_cdd(temperature_f))
weather = weather.withColumn("hdd", compute_hdd(temperature_f))
weather = weather.withColumn("wind_power_density", compute_wind_power(wind_speed))

# Rolling aggregations
window_24h = Window.partitionBy("zone", "h3_res7").orderBy("ts").rangeBetween(-24*3600, 0)
weather = weather.withColumn("rolling_24h_avg_temp", avg("temperature_f").over(window_24h))

# Write to both Iceberg and ClickHouse
weather.write.format("iceberg").mode("append").save("features_weather")
weather.write.format("clickhouse").mode("append").save("features_weather_online")
```

## Query Examples

### CDD Delta vs Yesterday for CAISO SP15 (Next 24h)

```sql
SELECT
  toStartOfHour(ts) AS hour,
  avg(cdd) AS avg_cdd,
  lag(avg_cdd) OVER (ORDER BY hour) AS prev_day_cdd,
  avg_cdd - prev_day_cdd AS cdd_delta
FROM features_weather_online
WHERE zone = 'CAISO_SP15'
  AND ts >= now()
  AND ts < now() + INTERVAL 24 HOUR
  AND forecast_hour > 0
GROUP BY hour
ORDER BY hour;
```

**Expected Latency**: < 3s (meets DoD requirement)

### Average Temperature by Zone (Current)

```sql
SELECT
  zone,
  avg(temperature_f) AS avg_temp_f,
  avg(cdd) AS avg_cdd,
  avg(hdd) AS avg_hdd
FROM features_weather_online
WHERE ts >= now() - INTERVAL 1 HOUR
  AND forecast_hour = 0
GROUP BY zone
ORDER BY zone;
```

### Wind Power Forecast for Next 6 Hours

```sql
SELECT
  zone,
  toStartOfHour(ts) AS forecast_hour,
  avg(wind_power_density) AS avg_wind_power,
  avg(wind_speed_mph) AS avg_wind_speed
FROM features_weather_online
WHERE ts >= now()
  AND ts < now() + INTERVAL 6 HOUR
  AND forecast_hour BETWEEN 1 AND 6
GROUP BY zone, forecast_hour
ORDER BY zone, forecast_hour;
```

## Data Quality Checks

1. **Completeness**: All hexes in zone covered
2. **Timeliness**: Features available within 15 min of raw data
3. **Reasonableness**:
   - Temperature: -50°F to 120°F
   - Wind speed: 0 to 150 mph
   - CDD/HDD: >= 0
4. **Consistency**: Hex aggregations match zone totals
5. **Freshness**: Latest observation < 2 hours old

## Performance Targets

- **Ingestion Latency**: Raw weather → Iceberg < 10 min
- **Feature Latency**: Raw → Features < 15 min
- **Query Latency**: CDD delta query < 3s (p95)
- **Throughput**: 10M observations/hour

## Future Enhancements

- **Ensemble Forecasts**: Multi-model blending (HRRR, GFS, ECMWF)
- **Downscaling**: Statistical downscaling for local effects
- **Demand Models**: ML models for load forecasting
- **Corridor Features**: Transmission corridor-specific weather
- **Climate Indices**: ENSO, NAO, PDO for seasonal forecasts
