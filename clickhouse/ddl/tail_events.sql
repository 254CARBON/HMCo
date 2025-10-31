-- Extreme-Tail Spike Engine: EVT + generative oversampling
-- Calibrated exceedance probabilities and synthetic tail scenarios

CREATE TABLE IF NOT EXISTS tail_event_thresholds ON CLUSTER '{cluster}' (
  iso LowCardinality(String),
  node_id String,
  variable_name LowCardinality(String) COMMENT 'LMP, spread, load, etc.',
  threshold Float64 COMMENT 'EVT threshold (u)',
  exceedance_rate Float64 COMMENT 'Rate above threshold',
  shape_param Float64 COMMENT 'GPD shape parameter (ξ)',
  scale_param Float64 COMMENT 'GPD scale parameter (σ)',
  model_type LowCardinality(String) COMMENT 'POT, GEV, etc.',
  calibration_period_start DateTime,
  calibration_period_end DateTime,
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/tail_event_thresholds','{replica}', computed_at)
PARTITION BY iso
ORDER BY (iso, node_id, variable_name)
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS tail_exceedance_probs ON CLUSTER '{cluster}' (
  timestamp DateTime,
  iso LowCardinality(String),
  node_id String,
  variable_name LowCardinality(String),
  level Float64 COMMENT 'Exceedance level',
  probability Float64 COMMENT 'P(X > level)',
  confidence_lower Float64,
  confidence_upper Float64,
  model_version String,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/tail_exceedance_probs','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, node_id, variable_name, timestamp, level)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS tail_synthetic_scenarios ON CLUSTER '{cluster}' (
  scenario_id String,
  iso LowCardinality(String),
  node_id String,
  variable_name LowCardinality(String),
  scenario_value Float64 COMMENT 'Synthetic tail value',
  return_period_years Float32 COMMENT 'Estimated return period',
  generation_method LowCardinality(String) COMMENT 'GAN, GPD, bootstrap, etc.',
  metadata String COMMENT 'JSON metadata',
  generated_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/tail_synthetic_scenarios','{replica}', generated_at)
PARTITION BY iso
ORDER BY (iso, node_id, variable_name, scenario_id)
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS tail_spike_alerts ON CLUSTER '{cluster}' (
  alert_id String,
  timestamp DateTime,
  iso LowCardinality(String),
  node_id String,
  variable_name LowCardinality(String),
  spike_probability Float64 COMMENT 'P(spike in next hour)',
  expected_magnitude Float64,
  confidence_level Float64,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/tail_spike_alerts','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, timestamp, alert_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;
