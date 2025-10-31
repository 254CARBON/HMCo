-- Alpha Attribution & Decision Shapley: driver→P&L decomposition
-- Per-trade/strategy attribution with Shapley values over decisions

CREATE TABLE IF NOT EXISTS alpha_decisions ON CLUSTER '{cluster}' (
  decision_id String,
  timestamp DateTime,
  strategy_id String,
  decision_type LowCardinality(String) COMMENT 'ENTER, EXIT, ADJUST, HEDGE',
  instrument String,
  quantity Float64,
  price Float64,
  rationale String COMMENT 'JSON of decision drivers and weights',
  feature_snapshot String COMMENT 'JSON of all features at decision time',
  model_outputs String COMMENT 'JSON of model predictions/signals',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/alpha_decisions','{replica}', created_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (strategy_id, timestamp, decision_id)
TTL toDate(timestamp) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS alpha_pnl ON CLUSTER '{cluster}' (
  pnl_id String,
  timestamp DateTime,
  strategy_id String,
  trade_id Nullable(String),
  decision_id Nullable(String),
  instrument String,
  realized_pnl Float64 COMMENT 'Realized P&L ($)',
  unrealized_pnl Float64 COMMENT 'Unrealized P&L ($)',
  total_pnl Float64 COMMENT 'Total P&L ($)',
  quantity Float64,
  entry_price Float64,
  exit_price Nullable(Float64),
  holding_period_hours Nullable(Float32),
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/alpha_pnl','{replica}', created_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (strategy_id, timestamp, pnl_id)
TTL toDate(timestamp) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS alpha_shapley_values ON CLUSTER '{cluster}' (
  attribution_id String,
  decision_id String,
  strategy_id String,
  feature_name String COMMENT 'Feature or signal name',
  shapley_value Float64 COMMENT 'Shapley value contribution to P&L',
  shapley_value_pct Float32 COMMENT 'Percentage contribution',
  feature_value Float64 COMMENT 'Actual feature value',
  counterfactual_pnl Float64 COMMENT 'Expected P&L without this feature',
  computation_method LowCardinality(String) COMMENT 'exact, sampling, kernel_shap',
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/alpha_shapley_values','{replica}', computed_at)
PARTITION BY strategy_id
ORDER BY (strategy_id, decision_id, feature_name)
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS alpha_factor_attribution ON CLUSTER '{cluster}' (
  evaluation_period DateTime,
  strategy_id String,
  factor_name String COMMENT 'Broad factor: momentum, value, carry, vol, etc.',
  factor_pnl Float64 COMMENT 'P&L attributed to this factor ($)',
  factor_pnl_pct Float32 COMMENT 'Percentage of total P&L',
  sharpe_contribution Float32 COMMENT 'Contribution to Sharpe ratio',
  top_features Array(String) COMMENT 'Top contributing features within factor',
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/alpha_factor_attribution','{replica}', computed_at)
PARTITION BY toYYYYMM(evaluation_period)
ORDER BY (strategy_id, evaluation_period, factor_name)
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS alpha_signal_performance ON CLUSTER '{cluster}' (
  evaluation_period DateTime,
  strategy_id String,
  signal_name String,
  signal_category LowCardinality(String) COMMENT 'PRICE, FLOW, WEATHER, FUNDAMENTAL, etc.',
  avg_shapley Float64 COMMENT 'Average Shapley value',
  total_pnl Float64 COMMENT 'Total P&L from decisions using this signal',
  decision_count UInt32 COMMENT 'Number of decisions influenced by signal',
  win_rate Float32 COMMENT 'Percentage of profitable decisions',
  sharpe_ratio Float32,
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/alpha_signal_performance','{replica}', computed_at)
PARTITION BY toYYYYMM(evaluation_period)
ORDER BY (strategy_id, signal_category, signal_name, evaluation_period)
SETTINGS index_granularity=8192;

CREATE MATERIALIZED VIEW IF NOT EXISTS alpha_attribution_summary
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/alpha_attribution_summary','{replica}')
PARTITION BY strategy_id
ORDER BY (strategy_id, feature_name)
AS
SELECT
  strategy_id,
  feature_name,
  avgState(shapley_value) as avg_shapley,
  sumState(shapley_value) as total_contribution,
  countState() as decision_count
FROM alpha_shapley_values
GROUP BY strategy_id, feature_name;
