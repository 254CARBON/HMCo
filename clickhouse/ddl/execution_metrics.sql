-- Execution & Market Impact: Liquidity-aware routing and scheduling
-- Slippage tracking across ICE/CME/EEX venues

CREATE TABLE IF NOT EXISTS execution_orders ON CLUSTER '{cluster}' (
  order_id String,
  parent_order_id Nullable(String) COMMENT 'Parent order if this is a child slice',
  timestamp DateTime,
  venue LowCardinality(String) COMMENT 'ICE, CME, EEX, etc.',
  instrument String COMMENT 'Contract identifier',
  instrument_type LowCardinality(String) COMMENT 'FUTURE, OPTION, etc.',
  side LowCardinality(String) COMMENT 'BUY, SELL',
  order_type LowCardinality(String) COMMENT 'LIMIT, MARKET, TWAP, POV, etc.',
  quantity Float64 COMMENT 'Order quantity',
  limit_price Nullable(Float64),
  arrival_price Float64 COMMENT 'Price at order arrival',
  avg_fill_price Nullable(Float64),
  filled_quantity Float64,
  order_status LowCardinality(String) COMMENT 'PENDING, FILLED, PARTIAL, CANCELLED',
  execution_algo LowCardinality(String) COMMENT 'POV, TWAP, IS, etc.',
  urgency LowCardinality(String) COMMENT 'LOW, MEDIUM, HIGH',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/execution_orders','{replica}', created_at)
PARTITION BY (venue, toYYYYMM(timestamp))
ORDER BY (venue, instrument, timestamp, order_id)
TTL toDate(timestamp) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS execution_fills ON CLUSTER '{cluster}' (
  fill_id String,
  order_id String,
  timestamp DateTime,
  venue LowCardinality(String),
  instrument String,
  quantity Float64,
  price Float64,
  fees Float64,
  liquidity_flag LowCardinality(String) COMMENT 'MAKER, TAKER',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/execution_fills','{replica}', ingestion_timestamp)
PARTITION BY (venue, toYYYYMM(timestamp))
ORDER BY (venue, instrument, timestamp, fill_id)
TTL toDate(timestamp) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS execution_slippage ON CLUSTER '{cluster}' (
  order_id String,
  timestamp DateTime,
  venue LowCardinality(String),
  instrument String,
  arrival_price Float64,
  avg_fill_price Float64,
  vwap Float64 COMMENT 'Volume-weighted average price during period',
  slippage_bps Float32 COMMENT 'Slippage in basis points',
  slippage_dollars Float64 COMMENT 'Slippage in dollars',
  market_impact_bps Float32 COMMENT 'Estimated market impact',
  timing_cost_bps Float32 COMMENT 'Timing/opportunity cost',
  market_volume Float64 COMMENT 'Total market volume during execution',
  participation_rate Float32 COMMENT 'Our volume / market volume',
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/execution_slippage','{replica}', computed_at)
PARTITION BY (venue, toYYYYMM(timestamp))
ORDER BY (venue, instrument, timestamp, order_id)
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS execution_venue_liquidity ON CLUSTER '{cluster}' (
  timestamp DateTime,
  venue LowCardinality(String),
  instrument String,
  bid_price Float64,
  ask_price Float64,
  bid_size Float64,
  ask_size Float64,
  spread_bps Float32,
  depth_5_levels Float64 COMMENT 'Total depth in top 5 levels',
  trade_count_5m UInt32 COMMENT 'Trades in last 5 minutes',
  volume_5m Float64 COMMENT 'Volume in last 5 minutes',
  volatility_5m Float32 COMMENT 'Price volatility in last 5 minutes',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/execution_venue_liquidity','{replica}', ingestion_timestamp)
PARTITION BY (venue, toYYYYMM(timestamp))
ORDER BY (venue, instrument, timestamp)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS execution_algo_performance ON CLUSTER '{cluster}' (
  evaluation_period DateTime,
  venue LowCardinality(String),
  execution_algo LowCardinality(String),
  metric_name LowCardinality(String) COMMENT 'avg_slippage, fill_rate, etc.',
  metric_value Float64,
  order_count UInt32,
  total_quantity Float64,
  computed_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/execution_algo_performance','{replica}', computed_at)
PARTITION BY toYYYYMM(evaluation_period)
ORDER BY (venue, execution_algo, metric_name, evaluation_period)
SETTINGS index_granularity=8192;

CREATE MATERIALIZED VIEW IF NOT EXISTS execution_slippage_daily
ENGINE = ReplicatedAggregatingMergeTree('/clickhouse/tables/{shard}/execution_slippage_daily','{replica}')
PARTITION BY toYYYYMM(day)
ORDER BY (venue, instrument, day)
AS
SELECT
  venue,
  instrument,
  toDate(timestamp) as day,
  avgState(slippage_bps) as avg_slippage_bps,
  avgState(market_impact_bps) as avg_impact_bps,
  sumState(slippage_dollars) as total_slippage_dollars,
  countState() as order_count
FROM execution_slippage
GROUP BY venue, instrument, day;
