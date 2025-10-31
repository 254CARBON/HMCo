-- Real-time LMP forecasts with quantile predictions
-- Stores physics-aware nowcast outputs with diagnostics

CREATE TABLE IF NOT EXISTS rt_forecasts (
    timestamp DateTime64(3),
    node_id String,
    iso LowCardinality(String),
    run_id String,
    run_timestamp DateTime64(3),
    
    -- Quantile predictions
    p10 Float32,  -- 10th percentile
    p50 Float32,  -- Median prediction
    p90 Float32,  -- 90th percentile
    
    -- Decomposition (if available)
    energy_p50 Nullable(Float32),
    congestion_p50 Nullable(Float32),
    loss_p50 Nullable(Float32),
    
    -- Diagnostics
    model_version LowCardinality(String) DEFAULT '0.1.0',
    inference_time_ms Float32,
    calibration_factor Nullable(Float32),
    
    -- Metadata
    created_at DateTime64(3) DEFAULT now64()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (iso, node_id, timestamp, run_timestamp)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Materialized view for latest forecasts per node
CREATE MATERIALIZED VIEW IF NOT EXISTS rt_forecasts_latest
ENGINE = ReplacingMergeTree(run_timestamp)
PARTITION BY iso
ORDER BY (node_id, timestamp)
AS SELECT
    timestamp,
    node_id,
    iso,
    run_id,
    run_timestamp,
    p10,
    p50,
    p90,
    energy_p50,
    congestion_p50,
    loss_p50,
    model_version,
    inference_time_ms
FROM rt_forecasts;

-- Aggregated metrics for monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS rt_forecasts_metrics
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(run_timestamp)
ORDER BY (iso, toStartOfHour(run_timestamp), model_version)
AS SELECT
    iso,
    toStartOfHour(run_timestamp) as hour,
    model_version,
    
    -- Forecast statistics
    count() as num_forecasts,
    countDistinct(node_id) as num_nodes,
    countDistinct(run_id) as num_runs,
    
    -- Value ranges
    avgState(p50) as avg_p50,
    minState(p50) as min_p50,
    maxState(p50) as max_p50,
    
    -- Spread metrics (uncertainty)
    avgState(p90 - p10) as avg_spread,
    maxState(p90 - p10) as max_spread,
    
    -- Performance
    avgState(inference_time_ms) as avg_inference_time_ms,
    quantileState(0.95)(inference_time_ms) as p95_inference_time_ms,
    maxState(inference_time_ms) as max_inference_time_ms
FROM rt_forecasts
GROUP BY iso, hour, model_version;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_rt_forecasts_run_id ON rt_forecasts (run_id) TYPE bloom_filter GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_rt_forecasts_node ON rt_forecasts (node_id) TYPE bloom_filter GRANULARITY 4;

-- Comments
ALTER TABLE rt_forecasts COMMENT 'Real-time LMP forecasts with quantile predictions from physics-aware graph transformers. Contains P10/P50/P90 predictions with sub-500ms inference time target.';
