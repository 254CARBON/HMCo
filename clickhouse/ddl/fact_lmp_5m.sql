-- Star schema for LMP data products
-- Fact table: 5-minute LMP with dimension foreign keys

CREATE TABLE IF NOT EXISTS fact_lmp_5m ON CLUSTER '{cluster}' (
  ts DateTime CODEC(Delta, ZSTD),
  node_key UInt64,
  hub_key Nullable(UInt32),
  zone_key Nullable(UInt32),
  calendar_key UInt32,
  lmp Float64 CODEC(T64, ZSTD),
  energy_component Float64 CODEC(T64, ZSTD),
  congestion_component Float64 CODEC(T64, ZSTD),
  loss_component Float64 CODEC(T64, ZSTD)
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/fact_lmp_5m','{replica}')
PARTITION BY toDate(ts)
ORDER BY (node_key, ts)
TTL toDate(ts) + INTERVAL 3 YEAR DELETE
SETTINGS index_granularity=8192;

-- Dimension: Nodes
CREATE TABLE IF NOT EXISTS dim_node ON CLUSTER '{cluster}' (
  node_key UInt64,
  iso LowCardinality(String),
  native_node_id String,
  canonical_node_id String,
  node_name String,
  node_type LowCardinality(String),
  latitude Nullable(Float64),
  longitude Nullable(Float64),
  active UInt8,
  effective_date Date,
  expiry_date Nullable(Date)
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/dim_node','{replica}')
ORDER BY node_key
SETTINGS index_granularity=8192;

-- Dimension: Hubs
CREATE TABLE IF NOT EXISTS dim_hub ON CLUSTER '{cluster}' (
  hub_key UInt32,
  iso LowCardinality(String),
  native_hub_id String,
  canonical_hub_id String,
  hub_name String,
  active UInt8
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/dim_hub','{replica}')
ORDER BY hub_key
SETTINGS index_granularity=8192;

-- Dimension: Calendar (for time intelligence)
CREATE TABLE IF NOT EXISTS dim_calendar ON CLUSTER '{cluster}' (
  calendar_key UInt32,
  date Date,
  year UInt16,
  quarter UInt8,
  month UInt8,
  day UInt8,
  day_of_week UInt8,
  day_of_year UInt16,
  week_of_year UInt8,
  is_weekend UInt8,
  is_holiday UInt8,
  holiday_name Nullable(String),
  month_name String,
  quarter_name String
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/dim_calendar','{replica}')
ORDER BY calendar_key
SETTINGS index_granularity=8192;

-- Materialized view: 5-minute rollup by hub
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_lmp_5m_by_hub
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/mv_lmp_5m_by_hub','{replica}')
PARTITION BY toYYYYMM(ts)
ORDER BY (hub_key, ts)
AS
SELECT
  toStartOfFiveMinutes(ts) AS ts,
  hub_key,
  avgState(lmp) AS avg_lmp_state,
  minState(lmp) AS min_lmp_state,
  maxState(lmp) AS max_lmp_state,
  avgState(congestion_component) AS avg_congestion_state,
  countState() AS sample_count_state
FROM fact_lmp_5m
WHERE hub_key IS NOT NULL
GROUP BY ts, hub_key;

-- Materialized view: Congestion by constraint
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_congestion_by_constraint
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/mv_congestion_by_constraint','{replica}')
PARTITION BY toYYYYMM(ts)
ORDER BY (node_key, ts)
AS
SELECT
  toStartOfHour(ts) AS ts,
  node_key,
  avgState(congestion_component) AS avg_congestion_state,
  maxState(congestion_component) AS max_congestion_state,
  countState(if(congestion_component > 10, 1, 0)) AS high_congestion_count_state
FROM fact_lmp_5m
GROUP BY ts, node_key;
