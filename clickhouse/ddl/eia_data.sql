-- EIA (Energy Information Administration) data tables
-- Daily fuel and generation data

CREATE TABLE IF NOT EXISTS eia_daily_fuel ON CLUSTER '{cluster}' (
  ts Date,
  region LowCardinality(String),
  series LowCardinality(String),
  value Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/eia_daily_fuel','{replica}')
PARTITION BY toYYYYMM(ts)
ORDER BY (region, series, ts)
TTL ts + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity=8192;
