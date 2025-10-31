-- Canonical Day-Ahead Award table (cleared generation/load across all ISOs)

CREATE TABLE IF NOT EXISTS iso_da_award ON CLUSTER '{cluster}' (
  trade_date Date,
  hour_ending UInt8,
  iso LowCardinality(String),
  resource_id String,
  resource_name Nullable(String),
  node String,
  zone Nullable(String),
  awarded_mw Float64 CODEC(T64, ZSTD),
  self_schedule_mw Float64 DEFAULT 0 CODEC(T64, ZSTD),
  price Float64 CODEC(T64, ZSTD),
  resource_type LowCardinality(String)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/iso_da_award','{replica}')
PARTITION BY trade_date
ORDER BY (iso, trade_date, resource_id, hour_ending)
TTL trade_date + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity=8192;

-- Materialized view for daily resource summary
CREATE MATERIALIZED VIEW IF NOT EXISTS iso_da_resource_summary
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/iso_da_resource_summary','{replica}')
PARTITION BY toYYYYMM(trade_date)
ORDER BY (iso, resource_id, trade_date)
AS
SELECT
  trade_date,
  iso,
  resource_id,
  resource_name,
  zone,
  resource_type,
  sumState(awarded_mw) AS total_awarded_mw_state,
  avgState(price) AS avg_price_state,
  minState(price) AS min_price_state,
  maxState(price) AS max_price_state,
  countState() AS hour_count_state
FROM iso_da_award
GROUP BY trade_date, iso, resource_id, resource_name, zone, resource_type;

-- View for querying daily resource summary
CREATE VIEW IF NOT EXISTS v_iso_da_resource_summary AS
SELECT
  trade_date,
  iso,
  resource_id,
  resource_name,
  zone,
  resource_type,
  sumMerge(total_awarded_mw_state) AS total_awarded_mw,
  avgMerge(avg_price_state) AS avg_price,
  minMerge(min_price_state) AS min_price,
  maxMerge(max_price_state) AS max_price,
  countMerge(hour_count_state) AS hour_count
FROM iso_da_resource_summary
GROUP BY trade_date, iso, resource_id, resource_name, zone, resource_type;
