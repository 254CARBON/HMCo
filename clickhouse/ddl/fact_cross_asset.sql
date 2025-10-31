-- Cross-commodity signal features for gas, power, emissions, FX
-- Enables spark spread analysis and cross-market arbitrage

CREATE TABLE IF NOT EXISTS fact_cross_asset (
    timestamp DateTime64(3),
    iso LowCardinality(String),
    hub_id String,
    
    -- Power metrics
    power_lmp Float32,
    power_load_mw Float32,
    power_gen_mw Float32,
    
    -- Gas metrics
    gas_price_mmbtu Float32,
    gas_storage_bcf Nullable(Float32),
    gas_flow_mmcfd Nullable(Float32),
    
    -- Derived cross-commodity signals
    spark_spread Float32,  -- power_lmp - (gas_price * heat_rate)
    implied_heat_rate Float32,
    dark_spread Nullable(Float32),  -- for coal
    
    -- Emissions
    carbon_price_ton Nullable(Float32),
    carbon_intensity Nullable(Float32),
    carbon_adjusted_spread Nullable(Float32),
    
    -- FX (for international LNG)
    fx_rate Nullable(Float32),
    lng_netback_price Nullable(Float32),
    lng_arrival_pressure Nullable(Float32),
    
    -- Metadata
    data_quality LowCardinality(String) DEFAULT 'good',
    created_at DateTime64(3) DEFAULT now64()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (iso, hub_id, timestamp)
TTL timestamp + INTERVAL 365 DAY
SETTINGS index_granularity = 8192;

-- Aggregated hourly cross-asset signals
CREATE MATERIALIZED VIEW IF NOT EXISTS fact_cross_asset_hourly
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (iso, hub_id, toStartOfHour(timestamp))
AS SELECT
    toStartOfHour(timestamp) as hour,
    iso,
    hub_id,
    
    avgState(spark_spread) as avg_spark_spread,
    minState(spark_spread) as min_spark_spread,
    maxState(spark_spread) as max_spark_spread,
    
    avgState(implied_heat_rate) as avg_heat_rate,
    avgState(gas_price_mmbtu) as avg_gas_price,
    avgState(power_lmp) as avg_power_lmp,
    
    avgState(carbon_adjusted_spread) as avg_carbon_spread
FROM fact_cross_asset
GROUP BY hour, iso, hub_id;

ALTER TABLE fact_cross_asset COMMENT 'Cross-commodity analytics for gas-power arbitrage, spark spreads, and LNG netbacks';
