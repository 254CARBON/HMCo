-- Forward curves and risk factors for trading analytics
-- Latest EOD curves for fast dashboard queries

CREATE TABLE IF NOT EXISTS curves_latest ON CLUSTER '{cluster}' (
  curve_date Date,
  curve_id String,
  commodity LowCardinality(String),
  region LowCardinality(String),
  bucket LowCardinality(String),
  delivery_start Date,
  delivery_end Date,
  term LowCardinality(String),
  price Float64 CODEC(T64, ZSTD),
  bid Nullable(Float64) CODEC(T64, ZSTD),
  ask Nullable(Float64) CODEC(T64, ZSTD),
  volume_mw Nullable(Float64) CODEC(T64, ZSTD),
  source LowCardinality(String),
  snapshot_id Int64,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/curves_latest','{replica}', created_at)
PARTITION BY curve_date
ORDER BY (curve_id, delivery_start)
TTL curve_date + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity=8192;

-- Risk factors table (PCA, basis, spreads, correlations)
CREATE TABLE IF NOT EXISTS factors_latest ON CLUSTER '{cluster}' (
  factor_date Date,
  factor_id String,
  factor_type LowCardinality(String),
  value Float64 CODEC(T64, ZSTD),
  unit LowCardinality(String),
  metadata Map(String, String),
  snapshot_id Int64,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/factors_latest','{replica}', created_at)
PARTITION BY factor_date
ORDER BY (factor_id, factor_date)
TTL factor_date + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity=8192;

-- Snapshot metadata tracking
CREATE TABLE IF NOT EXISTS curve_snapshot_metadata ON CLUSTER '{cluster}' (
  snapshot_id Int64,
  curve_date Date,
  build_timestamp DateTime,
  status LowCardinality(String),
  row_count UInt64,
  validation_passed UInt8,
  error_message Nullable(String)
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/curve_snapshot_metadata','{replica}', build_timestamp)
ORDER BY (curve_date, snapshot_id)
SETTINGS index_granularity=8192;

-- Materialized view for curve changes (day-over-day)
CREATE MATERIALIZED VIEW IF NOT EXISTS curves_dod_changes
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/curves_dod_changes','{replica}')
PARTITION BY toYYYYMM(curve_date)
ORDER BY (curve_id, delivery_start, curve_date)
AS
SELECT
  curve_date,
  curve_id,
  delivery_start,
  anyState(price) AS price_today_state,
  anyState(lag(price) OVER (PARTITION BY curve_id, delivery_start ORDER BY curve_date)) AS price_yesterday_state
FROM curves_latest
GROUP BY curve_date, curve_id, delivery_start;

-- View for querying curve changes
CREATE VIEW IF NOT EXISTS v_curves_dod_changes AS
SELECT
  curve_date,
  curve_id,
  delivery_start,
  anyMerge(price_today_state) AS price_today,
  anyMerge(price_yesterday_state) AS price_yesterday,
  price_today - price_yesterday AS price_change,
  (price_today - price_yesterday) / price_yesterday * 100 AS price_change_pct
FROM curves_dod_changes
GROUP BY curve_date, curve_id, delivery_start;

-- Standard buckets reference table
CREATE TABLE IF NOT EXISTS curve_buckets ON CLUSTER '{cluster}' (
  bucket_id LowCardinality(String),
  bucket_name String,
  days String,
  hours String,
  total_hours_per_week UInt16,
  description String
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/curve_buckets','{replica}')
ORDER BY bucket_id
SETTINGS index_granularity=8192;

-- Seed bucket definitions
INSERT INTO curve_buckets (bucket_id, bucket_name, days, hours, total_hours_per_week, description) VALUES
  ('5x16', 'On-Peak Weekday', 'Mon-Fri', 'HE 7-22', 80, 'Monday-Friday, Hours Ending 7-22'),
  ('7x8', 'Off-Peak All Week', 'All Days', 'HE 23-6', 56, 'All days, Hours Ending 23-6'),
  ('2x16', 'Weekend Peak', 'Sat-Sun', 'HE 7-22', 32, 'Saturday-Sunday, Hours Ending 7-22'),
  ('7x24', 'Around the Clock', 'All Days', 'All Hours', 168, 'All days, all hours'),
  ('HLH', 'High Load Hours', 'Mon-Fri', 'HE 7-22', 80, 'High Load Hours (Mon-Fri peak)'),
  ('LLH', 'Low Load Hours', 'All Days', 'HE 23-6 + Weekends', 88, 'Low Load Hours (off-peak + weekends)');
