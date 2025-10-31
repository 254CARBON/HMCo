-- Unit Commitment Probability Surfaces: start/stop/ramp predictions
-- Hazard models with weather, outages, and fuel spreads

CREATE TABLE IF NOT EXISTS unit_commit_probabilities ON CLUSTER '{cluster}' (
  timestamp DateTime,
  iso LowCardinality(String),
  unit_id String COMMENT 'Generator unit identifier',
  prob_start Float32 COMMENT 'P(start) for this hour',
  prob_stop Float32 COMMENT 'P(stop) for this hour',
  prob_online Float32 COMMENT 'P(unit is online)',
  expected_ramp_up Float32 COMMENT 'Expected ramp up capacity (MW)',
  expected_ramp_down Float32 COMMENT 'Expected ramp down capacity (MW)',
  weather_factor Float32 COMMENT 'Weather impact factor',
  fuel_spread Float32 COMMENT 'Fuel spread ($/MMBtu)',
  model_version String,
  features String COMMENT 'JSON of input features',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/unit_commit_probabilities','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, unit_id, timestamp)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS unit_status_history ON CLUSTER '{cluster}' (
  timestamp DateTime,
  iso LowCardinality(String),
  unit_id String,
  status LowCardinality(String) COMMENT 'ONLINE, OFFLINE, STARTING, STOPPING',
  output_mw Float32 COMMENT 'Current output (MW)',
  capacity_mw Float32 COMMENT 'Max capacity (MW)',
  fuel_type LowCardinality(String),
  source LowCardinality(String) COMMENT 'awards, telemetry, etc.',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/unit_status_history','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, unit_id, timestamp)
TTL toDate(timestamp) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS unit_commit_performance ON CLUSTER '{cluster}' (
  evaluation_period DateTime,
  iso LowCardinality(String),
  unit_id String,
  metric_name LowCardinality(String) COMMENT 'brier_score, roc_auc, calibration_error',
  metric_value Float32,
  sample_size UInt32,
  model_version String,
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/unit_commit_performance','{replica}', computed_at)
PARTITION BY (iso, toYYYYMM(evaluation_period))
ORDER BY (iso, unit_id, metric_name, evaluation_period)
SETTINGS index_granularity=8192;
