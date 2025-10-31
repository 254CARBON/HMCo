-- Probabilistic curve scenarios with conformal calibration
-- Stores 100-1000 scenarios per hub/tenor for VaR/ES

CREATE TABLE IF NOT EXISTS curve_scenarios (
    timestamp DateTime64(3),
    hub_id String,
    iso LowCardinality(String),
    tenor LowCardinality(String),  -- 'DA', 'RT', 'week', 'month'
    scenario_id UInt16,  -- 0-999
    
    -- Scenario values
    lmp_value Float32,
    volume_mw Nullable(Float32),
    
    -- Quantile assignment
    quantile_bin UInt8,  -- 0-99 for percentile bins
    
    -- Calibration
    conformal_adj Float32 DEFAULT 0.0,
    calibration_window_days UInt16 DEFAULT 30,
    
    -- Hierarchical consistency
    parent_hub_id Nullable(String),
    consistency_weight Float32 DEFAULT 1.0,
    
    -- Metadata
    model_version LowCardinality(String) DEFAULT '0.1.0',
    run_id String,
    run_timestamp DateTime64(3),
    created_at DateTime64(3) DEFAULT now64()
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), tenor)
ORDER BY (hub_id, timestamp, scenario_id)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Scenario statistics for quick VaR/ES calculation
CREATE MATERIALIZED VIEW IF NOT EXISTS curve_scenarios_stats
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (hub_id, tenor, timestamp)
AS SELECT
    timestamp,
    hub_id,
    iso,
    tenor,
    run_timestamp,
    
    -- Distribution statistics
    avgState(lmp_value) as mean_lmp,
    stddevPopState(lmp_value) as std_lmp,
    minState(lmp_value) as min_lmp,
    maxState(lmp_value) as max_lmp,
    
    -- Quantiles for VaR
    quantileState(0.05)(lmp_value) as p05,
    quantileState(0.10)(lmp_value) as p10,
    quantileState(0.50)(lmp_value) as p50,
    quantileState(0.90)(lmp_value) as p90,
    quantileState(0.95)(lmp_value) as p95,
    
    -- Expected shortfall (ES) at 95%
    avgIfState(lmp_value, lmp_value >= quantile(0.95)(lmp_value)) as es_95,
    
    count() as num_scenarios
FROM curve_scenarios
GROUP BY timestamp, hub_id, iso, tenor, run_timestamp;

-- Index for scenario sampling
CREATE INDEX IF NOT EXISTS idx_scenarios_run ON curve_scenarios (run_id) TYPE bloom_filter GRANULARITY 4;

ALTER TABLE curve_scenarios COMMENT 'Probabilistic price curve scenarios with conformal calibration for risk analysis';
