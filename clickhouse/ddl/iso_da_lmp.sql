-- Canonical Day-Ahead LMP table (unified across all ISOs)

CREATE TABLE IF NOT EXISTS iso_da_lmp ON CLUSTER '{cluster}' (
  ts DateTime CODEC(Delta, ZSTD),
  trade_date Date,
  iso LowCardinality(String),
  node String,
  hub Nullable(String),
  zone Nullable(String),
  lmp Float64 CODEC(T64, ZSTD),
  energy_component Float64 CODEC(T64, ZSTD),
  congestion_component Float64 CODEC(T64, ZSTD),
  loss_component Float64 CODEC(T64, ZSTD),
  hour_ending UInt8
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/iso_da_lmp','{replica}')
PARTITION BY trade_date
ORDER BY (iso, node, trade_date, hour_ending)
TTL trade_date + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity=8192;

-- Materialized view for daily hub summary
CREATE MATERIALIZED VIEW IF NOT EXISTS iso_da_hub_summary
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/iso_da_hub_summary','{replica}')
PARTITION BY toYYYYMM(trade_date)
ORDER BY (iso, hub, trade_date)
AS
SELECT
  trade_date,
  iso,
  hub,
  avgState(lmp) AS avg_lmp_state,
  minState(lmp) AS min_lmp_state,
  maxState(lmp) AS max_lmp_state,
  sumState(if(hour_ending <= 16 AND hour_ending > 6, lmp, 0)) / 10 AS on_peak_avg_state,
  sumState(if(hour_ending > 16 OR hour_ending <= 6, lmp, 0)) / 14 AS off_peak_avg_state,
  countState() AS hour_count_state
FROM iso_da_lmp
WHERE hub IS NOT NULL
GROUP BY trade_date, iso, hub;

-- View for querying daily hub summary
CREATE VIEW IF NOT EXISTS v_iso_da_hub_summary AS
SELECT
  trade_date,
  iso,
  hub,
  avgMerge(avg_lmp_state) AS avg_lmp,
  minMerge(min_lmp_state) AS min_lmp,
  maxMerge(max_lmp_state) AS max_lmp,
  sumMerge(on_peak_avg_state) AS on_peak_avg,
  sumMerge(off_peak_avg_state) AS off_peak_avg,
  countMerge(hour_count_state) AS hour_count
FROM iso_da_hub_summary
GROUP BY trade_date, iso, hub;
