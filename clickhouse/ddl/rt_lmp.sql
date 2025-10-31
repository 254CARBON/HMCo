-- Real-time LMP (Locational Marginal Pricing) tables for ISO markets
-- Supports CAISO, MISO, SPP 5-minute data with 3-year TTL

-- Main real-time LMP table (5-minute granularity)
CREATE TABLE IF NOT EXISTS rt_lmp ON CLUSTER '{cluster}' (
  ts DateTime CODEC(Delta, ZSTD),
  iso LowCardinality(String),
  node String,
  lmp Float64 CODEC(T64, ZSTD),
  congestion Float64 CODEC(T64, ZSTD),
  loss Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/rt_lmp','{replica}')
PARTITION BY toDate(ts)
ORDER BY (iso, node, ts)
TTL toDate(ts) + INTERVAL 3 YEAR DELETE
SETTINGS index_granularity=8192;

-- Aggregated 5-minute rollup table (longer retention)
CREATE TABLE IF NOT EXISTS rt_lmp_5m ON CLUSTER '{cluster}' (
  ts DateTime,
  iso LowCardinality(String),
  node String,
  lmp Float64 CODEC(T64, ZSTD),
  congestion Float64 CODEC(T64, ZSTD),
  loss Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/rt_lmp_5m','{replica}')
PARTITION BY toDate(ts)
ORDER BY (iso, node, ts)
TTL toDate(ts) + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity=8192;

-- Materialized view to populate 5-minute rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS rt_lmp_mv TO rt_lmp_5m AS
SELECT 
  iso, 
  node, 
  toStartOfFiveMinutes(ts) AS ts, 
  avg(lmp) AS lmp,
  avg(congestion) AS congestion, 
  avg(loss) AS loss
FROM rt_lmp 
GROUP BY iso, node, ts;

-- Day-ahead (DA) prices table
CREATE TABLE IF NOT EXISTS da_prices ON CLUSTER '{cluster}' (
  ts DateTime,
  iso LowCardinality(String),
  node String,
  price Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/da_prices','{replica}')
PARTITION BY toDate(ts)
ORDER BY (iso, node, ts)
TTL toDate(ts) + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity=8192;
