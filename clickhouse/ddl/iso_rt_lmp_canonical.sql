-- Canonical Real-Time LMP table (unified across CAISO, MISO, SPP)
-- Replaces individual ISO-specific tables with a single normalized schema

CREATE TABLE IF NOT EXISTS iso_rt_lmp ON CLUSTER '{cluster}' (
  ts DateTime CODEC(Delta, ZSTD),
  iso LowCardinality(String),
  node String,
  hub Nullable(String),
  zone Nullable(String),
  lmp Float64 CODEC(T64, ZSTD),
  energy_component Float64 CODEC(T64, ZSTD),
  congestion_component Float64 CODEC(T64, ZSTD),
  loss_component Float64 CODEC(T64, ZSTD),
  market_run LowCardinality(String),
  interval_minutes UInt8 DEFAULT 5
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/iso_rt_lmp','{replica}')
PARTITION BY toDate(ts)
ORDER BY (iso, node, ts)
TTL toDate(ts) + INTERVAL 3 YEAR DELETE
SETTINGS index_granularity=8192;

-- Materialized view for hourly rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS iso_rt_lmp_hourly
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/iso_rt_lmp_hourly','{replica}')
PARTITION BY toYYYYMM(hour)
ORDER BY (iso, node, hour)
AS
SELECT
  toStartOfHour(ts) AS hour,
  iso,
  node,
  hub,
  zone,
  avgState(lmp) AS avg_lmp_state,
  minState(lmp) AS min_lmp_state,
  maxState(lmp) AS max_lmp_state,
  avgState(energy_component) AS avg_energy_state,
  avgState(congestion_component) AS avg_congestion_state,
  avgState(loss_component) AS avg_loss_state,
  countState() AS sample_count_state
FROM iso_rt_lmp
GROUP BY hour, iso, node, hub, zone;

-- View for querying hourly rollup
CREATE VIEW IF NOT EXISTS v_iso_rt_lmp_hourly AS
SELECT
  hour,
  iso,
  node,
  hub,
  zone,
  avgMerge(avg_lmp_state) AS avg_lmp,
  minMerge(min_lmp_state) AS min_lmp,
  maxMerge(max_lmp_state) AS max_lmp,
  avgMerge(avg_energy_state) AS avg_energy,
  avgMerge(avg_congestion_state) AS avg_congestion,
  avgMerge(avg_loss_state) AS avg_loss,
  countMerge(sample_count_state) AS sample_count
FROM iso_rt_lmp_hourly
GROUP BY hour, iso, node, hub, zone;
