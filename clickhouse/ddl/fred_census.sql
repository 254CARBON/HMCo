-- FRED (Federal Reserve Economic Data) and Census data tables

-- FRED daily economic indicators
CREATE TABLE IF NOT EXISTS fred_daily ON CLUSTER '{cluster}' (
  ts Date,
  series_id LowCardinality(String),
  value Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/fred_daily','{replica}')
PARTITION BY toYYYYMM(ts)
ORDER BY (series_id, ts)
TTL ts + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity=8192;

-- US Census static data
CREATE TABLE IF NOT EXISTS census_static ON CLUSTER '{cluster}' (
  id String,
  geography LowCardinality(String),
  measure LowCardinality(String),
  value Float64,
  year UInt16,
  updated_at DateTime DEFAULT now()
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/census_static','{replica}')
PARTITION BY year
ORDER BY (geography, measure, id)
-- Census data retained indefinitely (no TTL)
SETTINGS index_granularity=8192;
