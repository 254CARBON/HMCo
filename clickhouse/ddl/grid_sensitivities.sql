-- Grid Sensitivities: PTDF/LODF Storage
-- Near-real-time network sensitivities for nodal price prediction

-- PTDF (Power Transfer Distribution Factor) estimates
CREATE TABLE IF NOT EXISTS grid_ptdf_estimates ON CLUSTER '{cluster}' (
  iso LowCardinality(String) COMMENT 'ISO/RTO identifier',
  hub LowCardinality(String) COMMENT 'Hub/zone identifier',
  node_id String COMMENT 'Node identifier',
  line_id String COMMENT 'Transmission line identifier',
  sensitivity_factor Float64 COMMENT 'PTDF coefficient: MW flow change -> $/MWh LMP change',
  confidence_lower Float64 COMMENT 'Lower confidence bound',
  confidence_upper Float64 COMMENT 'Upper confidence bound',
  std_error Float64 COMMENT 'Standard error of estimate',
  model_version String COMMENT 'Model version identifier',
  computed_at DateTime COMMENT 'When sensitivity was computed',
  valid_from DateTime COMMENT 'Start of validity period',
  valid_to DateTime COMMENT 'End of validity period',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_ptdf_estimates','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(valid_from))
ORDER BY (iso, hub, node_id, line_id, valid_from)
TTL toDate(valid_to) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

-- LODF (Line Outage Distribution Factor) estimates
CREATE TABLE IF NOT EXISTS grid_lodf_estimates ON CLUSTER '{cluster}' (
  iso LowCardinality(String) COMMENT 'ISO/RTO identifier',
  hub LowCardinality(String) COMMENT 'Hub/zone identifier',
  outage_line_id String COMMENT 'Line ID that has outage',
  impacted_line_id String COMMENT 'Line ID that is impacted by outage',
  lodf_factor Float64 COMMENT 'LODF coefficient: fraction of outage flow redistributed',
  confidence_lower Float64 COMMENT 'Lower confidence bound',
  confidence_upper Float64 COMMENT 'Upper confidence bound',
  std_error Float64 COMMENT 'Standard error of estimate',
  model_version String COMMENT 'Model version identifier',
  computed_at DateTime COMMENT 'When LODF was computed',
  valid_from DateTime COMMENT 'Start of validity period',
  valid_to DateTime COMMENT 'End of validity period',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_lodf_estimates','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(valid_from))
ORDER BY (iso, outage_line_id, impacted_line_id, valid_from)
TTL toDate(valid_to) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

-- Network topology signals (co-movement, flow proxies)
CREATE TABLE IF NOT EXISTS grid_topology_signals ON CLUSTER '{cluster}' (
  timestamp DateTime COMMENT 'Signal timestamp',
  iso LowCardinality(String) COMMENT 'ISO/RTO identifier',
  signal_name String COMMENT 'Signal identifier',
  signal_value Float64 COMMENT 'Signal value',
  signal_type LowCardinality(String) COMMENT 'Type: comovement, flow_proxy, centrality, pca',
  metadata String COMMENT 'JSON metadata for signal',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_topology_signals','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, signal_name, timestamp)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Predicted LMP deltas from PTDF/LODF
CREATE TABLE IF NOT EXISTS grid_lmp_delta_predictions ON CLUSTER '{cluster}' (
  prediction_id String COMMENT 'Unique prediction identifier',
  timestamp DateTime COMMENT 'Prediction timestamp',
  iso LowCardinality(String) COMMENT 'ISO/RTO identifier',
  node_id String COMMENT 'Node identifier',
  predicted_delta Float64 COMMENT 'Predicted LMP change ($/MWh)',
  confidence_lower Float64 COMMENT 'Lower confidence bound',
  confidence_upper Float64 COMMENT 'Upper confidence bound',
  confidence_level Float64 COMMENT 'Confidence level (e.g., 0.95)',
  model_type LowCardinality(String) COMMENT 'Model type: PTDF or LODF',
  model_version String COMMENT 'Model version',
  scenario_id String COMMENT 'Scenario identifier for what-if analysis',
  flow_changes String COMMENT 'JSON of line flow changes that triggered prediction',
  actual_delta Nullable(Float64) COMMENT 'Actual LMP change (for validation)',
  prediction_error Nullable(Float64) COMMENT 'Prediction error when actual is known',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_lmp_delta_predictions','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, node_id, timestamp, prediction_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Model performance metrics
CREATE TABLE IF NOT EXISTS grid_sensitivity_model_metrics ON CLUSTER '{cluster}' (
  model_version String COMMENT 'Model version identifier',
  model_type LowCardinality(String) COMMENT 'PTDF or LODF',
  iso LowCardinality(String) COMMENT 'ISO/RTO identifier',
  metric_name LowCardinality(String) COMMENT 'Metric name',
  metric_value Float64 COMMENT 'Metric value',
  evaluation_period_start DateTime COMMENT 'Start of evaluation period',
  evaluation_period_end DateTime COMMENT 'End of evaluation period',
  sample_size UInt32 COMMENT 'Number of samples in evaluation',
  metadata String COMMENT 'JSON metadata',
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_sensitivity_model_metrics','{replica}', computed_at)
PARTITION BY (iso, toYYYYMM(evaluation_period_start))
ORDER BY (iso, model_type, model_version, metric_name, evaluation_period_start)
SETTINGS index_granularity=8192;

-- Materialized view: Latest PTDF sensitivities per node-line pair
CREATE MATERIALIZED VIEW IF NOT EXISTS grid_ptdf_latest
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_ptdf_latest','{replica}', computed_at)
PARTITION BY iso
ORDER BY (iso, hub, node_id, line_id)
AS
SELECT
  iso,
  hub,
  node_id,
  line_id,
  sensitivity_factor,
  confidence_lower,
  confidence_upper,
  std_error,
  model_version,
  computed_at
FROM grid_ptdf_estimates
QUALIFY ROW_NUMBER() OVER (PARTITION BY iso, hub, node_id, line_id ORDER BY computed_at DESC) = 1;

-- Materialized view: Latest LODF factors
CREATE MATERIALIZED VIEW IF NOT EXISTS grid_lodf_latest
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/grid_lodf_latest','{replica}', computed_at)
PARTITION BY iso
ORDER BY (iso, outage_line_id, impacted_line_id)
AS
SELECT
  iso,
  hub,
  outage_line_id,
  impacted_line_id,
  lodf_factor,
  confidence_lower,
  confidence_upper,
  std_error,
  model_version,
  computed_at
FROM grid_lodf_estimates
QUALIFY ROW_NUMBER() OVER (PARTITION BY iso, outage_line_id, impacted_line_id ORDER BY computed_at DESC) = 1;

-- Materialized view: Prediction accuracy aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS grid_prediction_accuracy_hourly
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/grid_prediction_accuracy_hourly','{replica}')
PARTITION BY (iso, toYYYYMM(hour))
ORDER BY (iso, node_id, model_type, hour)
AS
SELECT
  iso,
  node_id,
  model_type,
  toStartOfHour(timestamp) as hour,
  avgState(prediction_error) as avg_error,
  stddevPopState(prediction_error) as std_error,
  countState() as prediction_count,
  avgState(abs(prediction_error)) as mae,
  avgState(prediction_error * prediction_error) as mse_state
FROM grid_lmp_delta_predictions
WHERE actual_delta IS NOT NULL AND prediction_error IS NOT NULL
GROUP BY iso, node_id, model_type, hour;
