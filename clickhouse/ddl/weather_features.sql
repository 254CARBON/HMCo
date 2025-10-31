-- Weather features with H3 spatial indexing and derived metrics
-- Online feature table for real-time queries and dashboards

CREATE TABLE IF NOT EXISTS features_weather_online ON CLUSTER '{cluster}' (
  ts DateTime CODEC(Delta, ZSTD),
  zone LowCardinality(String),
  h3_index String,
  h3_res7 String,
  h3_res8 String,
  latitude Float64,
  longitude Float64,
  temperature_f Float64 CODEC(T64, ZSTD),
  temperature_c Float64 CODEC(T64, ZSTD),
  cdd Float64 CODEC(T64, ZSTD),
  hdd Float64 CODEC(T64, ZSTD),
  wind_speed_mph Float64 CODEC(T64, ZSTD),
  wind_power_density Float64 CODEC(T64, ZSTD),
  solar_capacity_factor Float64 CODEC(T64, ZSTD),
  cloud_cover_pct UInt8,
  humidity_pct Float64,
  precipitation_in Float64 CODEC(T64, ZSTD),
  feels_like_f Float64 CODEC(T64, ZSTD),
  forecast_hour UInt8,
  rolling_24h_avg_temp Float64 CODEC(T64, ZSTD),
  temp_change_24h Float64 CODEC(T64, ZSTD),
  source LowCardinality(String)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/features_weather_online','{replica}')
PARTITION BY toDate(ts)
ORDER BY (zone, h3_index, ts)
TTL toDate(ts) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Materialized view for hourly zone aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS features_weather_hourly_zone
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/features_weather_hourly_zone','{replica}')
PARTITION BY toYYYYMM(hour)
ORDER BY (zone, hour)
AS
SELECT
  toStartOfHour(ts) AS hour,
  zone,
  avgState(temperature_f) AS avg_temp_state,
  minState(temperature_f) AS min_temp_state,
  maxState(temperature_f) AS max_temp_state,
  avgState(cdd) AS avg_cdd_state,
  avgState(hdd) AS avg_hdd_state,
  avgState(wind_speed_mph) AS avg_wind_state,
  avgState(wind_power_density) AS avg_wind_power_state,
  avgState(solar_capacity_factor) AS avg_solar_cf_state,
  countState() AS sample_count_state
FROM features_weather_online
WHERE forecast_hour = 0
GROUP BY hour, zone;

-- View for querying hourly zone aggregates
CREATE VIEW IF NOT EXISTS v_features_weather_hourly_zone AS
SELECT
  hour,
  zone,
  avgMerge(avg_temp_state) AS avg_temp_f,
  minMerge(min_temp_state) AS min_temp_f,
  maxMerge(max_temp_state) AS max_temp_f,
  avgMerge(avg_cdd_state) AS avg_cdd,
  avgMerge(avg_hdd_state) AS avg_hdd,
  avgMerge(avg_wind_state) AS avg_wind_speed,
  avgMerge(avg_wind_power_state) AS avg_wind_power,
  avgMerge(avg_solar_cf_state) AS avg_solar_cf,
  countMerge(sample_count_state) AS sample_count
FROM features_weather_hourly_zone
GROUP BY hour, zone;

-- Forecast comparison table (forecast vs actual)
CREATE TABLE IF NOT EXISTS weather_forecast_accuracy ON CLUSTER '{cluster}' (
  forecast_ts DateTime,
  valid_ts DateTime,
  forecast_hour UInt8,
  zone LowCardinality(String),
  h3_index String,
  temperature_forecast Float64,
  temperature_actual Float64,
  temperature_error Float64,
  wind_forecast Float64,
  wind_actual Float64,
  wind_error Float64,
  source LowCardinality(String)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/weather_forecast_accuracy','{replica}')
PARTITION BY toDate(valid_ts)
ORDER BY (zone, valid_ts, forecast_hour)
TTL toDate(valid_ts) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;
