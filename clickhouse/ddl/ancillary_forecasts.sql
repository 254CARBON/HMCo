-- Ancillary Services: Joint energy + AS forecasting and co-optimization
-- Multi-task model outputs for regulation, spinning, and non-spinning reserves

-- Ancillary services price forecasts
CREATE TABLE IF NOT EXISTS ancillary_price_forecasts ON CLUSTER '{cluster}' (
  forecast_id String COMMENT 'Unique forecast identifier',
  timestamp DateTime COMMENT 'Forecast timestamp',
  iso LowCardinality(String),
  product_type LowCardinality(String) COMMENT 'REG_UP, REG_DOWN, SPIN, NON_SPIN, etc.',
  market_type LowCardinality(String) COMMENT 'DA or RT',
  price_forecast Float64 COMMENT 'Forecasted clearing price ($/MW)',
  confidence_lower Float64 COMMENT 'Lower CI bound',
  confidence_upper Float64 COMMENT 'Upper CI bound',
  model_version String,
  features String COMMENT 'JSON of input features',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ancillary_price_forecasts','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, product_type, market_type, timestamp, forecast_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Ancillary services demand forecasts
CREATE TABLE IF NOT EXISTS ancillary_demand_forecasts ON CLUSTER '{cluster}' (
  forecast_id String,
  timestamp DateTime,
  iso LowCardinality(String),
  product_type LowCardinality(String),
  demand_forecast Float64 COMMENT 'Forecasted AS requirement (MW)',
  confidence_lower Float64,
  confidence_upper Float64,
  model_version String,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ancillary_demand_forecasts','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, product_type, timestamp, forecast_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Co-optimization results: optimal energy + AS split
CREATE TABLE IF NOT EXISTS ancillary_coopt_allocations ON CLUSTER '{cluster}' (
  allocation_id String,
  timestamp DateTime,
  iso LowCardinality(String),
  portfolio_id String COMMENT 'Portfolio or resource identifier',
  energy_mw Float64 COMMENT 'Allocated to energy market (MW)',
  reg_up_mw Float64 COMMENT 'Allocated to regulation up (MW)',
  reg_down_mw Float64 COMMENT 'Allocated to regulation down (MW)',
  spin_mw Float64 COMMENT 'Allocated to spinning reserves (MW)',
  non_spin_mw Float64 COMMENT 'Allocated to non-spinning reserves (MW)',
  expected_revenue Float64 COMMENT 'Expected total revenue ($)',
  energy_revenue Float64 COMMENT 'Expected energy revenue ($)',
  as_revenue Float64 COMMENT 'Expected AS revenue ($)',
  optimization_objective LowCardinality(String) COMMENT 'max_revenue, min_risk, etc.',
  constraints String COMMENT 'JSON of constraints applied',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ancillary_coopt_allocations','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, portfolio_id, timestamp, allocation_id)
SETTINGS index_granularity=8192;

-- Historical AS clearing prices and volumes
CREATE TABLE IF NOT EXISTS ancillary_clearing_history ON CLUSTER '{cluster}' (
  timestamp DateTime,
  iso LowCardinality(String),
  product_type LowCardinality(String),
  market_type LowCardinality(String),
  clearing_price Float64 COMMENT 'Actual clearing price ($/MW)',
  cleared_volume Float64 COMMENT 'Total cleared volume (MW)',
  demand Float64 COMMENT 'AS requirement (MW)',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ancillary_clearing_history','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, product_type, market_type, timestamp)
TTL toDate(timestamp) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity=8192;

-- Portfolio P&L with AS positions
CREATE TABLE IF NOT EXISTS ancillary_portfolio_pnl ON CLUSTER '{cluster}' (
  timestamp DateTime,
  iso LowCardinality(String),
  portfolio_id String,
  energy_pnl Float64 COMMENT 'Energy market P&L ($)',
  reg_pnl Float64 COMMENT 'Regulation P&L ($)',
  spin_pnl Float64 COMMENT 'Spinning reserve P&L ($)',
  non_spin_pnl Float64 COMMENT 'Non-spinning reserve P&L ($)',
  total_pnl Float64 COMMENT 'Total P&L ($)',
  energy_mw Float64 COMMENT 'Energy position (MW)',
  as_mw Float64 COMMENT 'Total AS position (MW)',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ancillary_portfolio_pnl','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, portfolio_id, timestamp)
SETTINGS index_granularity=8192;

-- Materialized view: Latest AS price forecasts
CREATE MATERIALIZED VIEW IF NOT EXISTS ancillary_price_latest
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ancillary_price_latest','{replica}', created_at)
PARTITION BY iso
ORDER BY (iso, product_type, market_type, timestamp)
AS
SELECT
  iso,
  product_type,
  market_type,
  timestamp,
  price_forecast,
  confidence_lower,
  confidence_upper,
  model_version,
  created_at
FROM ancillary_price_forecasts
QUALIFY ROW_NUMBER() OVER (PARTITION BY iso, product_type, market_type, timestamp ORDER BY created_at DESC) = 1;
