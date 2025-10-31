-- DA↔RT Imbalance Risk: Schedule risk pricing and hedge optimization
-- Expected imbalance cost curves with P10/P50/P90 quantiles per hour/node

-- Imbalance cost forecasts (quantile regression outputs)
CREATE TABLE IF NOT EXISTS imbalance_cost_forecasts ON CLUSTER '{cluster}' (
  forecast_id String COMMENT 'Unique forecast identifier',
  timestamp DateTime COMMENT 'Forecast timestamp (hour ahead)',
  iso LowCardinality(String) COMMENT 'ISO/RTO identifier',
  node_id String COMMENT 'Node or hub identifier',
  quantile Float32 COMMENT 'Quantile level (0.1, 0.5, 0.9, etc.)',
  imbalance_cost Float64 COMMENT 'Expected imbalance cost ($/MWh)',
  schedule_mw Float64 COMMENT 'Scheduled MW (DA position)',
  model_version String COMMENT 'Model version',
  features String COMMENT 'JSON of input features',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/imbalance_cost_forecasts','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, node_id, timestamp, quantile, forecast_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Historical imbalance realizations
CREATE TABLE IF NOT EXISTS imbalance_realizations ON CLUSTER '{cluster}' (
  timestamp DateTime COMMENT 'Realization timestamp',
  iso LowCardinality(String),
  node_id String,
  da_schedule Float64 COMMENT 'DA scheduled position (MW)',
  rt_delivery Float64 COMMENT 'RT delivered position (MW)',
  imbalance Float64 COMMENT 'Imbalance: RT - DA (MW)',
  da_price Float64 COMMENT 'DA LMP ($/MWh)',
  rt_price Float64 COMMENT 'RT LMP ($/MWh)',
  imbalance_cost Float64 COMMENT 'Actual imbalance cost ($)',
  imbalance_cost_per_mw Float64 COMMENT 'Unit imbalance cost ($/MWh)',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/imbalance_realizations','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, node_id, timestamp)
TTL toDate(timestamp) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

-- Risk premia by hub/node
CREATE TABLE IF NOT EXISTS imbalance_risk_premia ON CLUSTER '{cluster}' (
  timestamp DateTime COMMENT 'Period timestamp',
  iso LowCardinality(String),
  node_id String,
  period LowCardinality(String) COMMENT 'Aggregation period: hourly, daily, monthly',
  mean_imbalance_cost Float64 COMMENT 'Mean imbalance cost ($/MWh)',
  p10_imbalance_cost Float64 COMMENT '10th percentile',
  p50_imbalance_cost Float64 COMMENT 'Median',
  p90_imbalance_cost Float64 COMMENT '90th percentile',
  volatility Float64 COMMENT 'Standard deviation',
  risk_premium Float64 COMMENT 'Risk premium above median ($/MWh)',
  sample_size UInt32 COMMENT 'Number of observations',
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/imbalance_risk_premia','{replica}', computed_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, node_id, period, timestamp)
SETTINGS index_granularity=8192;

-- Hedge performance tracking
CREATE TABLE IF NOT EXISTS imbalance_hedge_performance ON CLUSTER '{cluster}' (
  hedge_id String COMMENT 'Hedge strategy identifier',
  timestamp DateTime COMMENT 'Performance timestamp',
  iso LowCardinality(String),
  node_id String,
  baseline_cost Float64 COMMENT 'Imbalance cost without hedge ($)',
  hedged_cost Float64 COMMENT 'Imbalance cost with hedge ($)',
  hedge_benefit Float64 COMMENT 'Cost reduction ($)',
  hedge_benefit_pct Float64 COMMENT 'Cost reduction (%)',
  hedge_instruments String COMMENT 'JSON of hedge instruments used',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/imbalance_hedge_performance','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, hedge_id, node_id, timestamp)
SETTINGS index_granularity=8192;

-- Materialized view: Latest forecasts per node/hour
CREATE MATERIALIZED VIEW IF NOT EXISTS imbalance_forecast_latest
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/imbalance_forecast_latest','{replica}', created_at)
PARTITION BY iso
ORDER BY (iso, node_id, timestamp, quantile)
AS
SELECT
  iso,
  node_id,
  timestamp,
  quantile,
  imbalance_cost,
  schedule_mw,
  model_version,
  created_at
FROM imbalance_cost_forecasts
QUALIFY ROW_NUMBER() OVER (PARTITION BY iso, node_id, timestamp, quantile ORDER BY created_at DESC) = 1;
