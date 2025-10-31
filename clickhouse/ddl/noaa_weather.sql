-- NOAA weather data tables
-- Hourly weather observations and forecasts

CREATE TABLE IF NOT EXISTS noaa_hourly ON CLUSTER '{cluster}' (
  ts DateTime,
  grid_id String,
  temperature Float64 CODEC(T64, ZSTD),
  humidity Float64 CODEC(T64, ZSTD),
  wind_speed Float64 CODEC(T64, ZSTD),
  precipitation Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/noaa_hourly','{replica}')
PARTITION BY toDate(ts)
ORDER BY (grid_id, ts)
-- NOAA data retained indefinitely (no TTL)
SETTINGS index_granularity=8192;
