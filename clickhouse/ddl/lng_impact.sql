-- LNG→Power Coupling: AIS/regas/pipeline to hub price impact
-- Impact curves for expected send-out on regional hubs & nodal prices

CREATE TABLE IF NOT EXISTS lng_vessel_tracking ON CLUSTER '{cluster}' (
  timestamp DateTime,
  vessel_imo String COMMENT 'IMO vessel identifier',
  vessel_name String,
  position_lat Float64,
  position_lon Float64,
  destination_port String,
  cargo_capacity_bcm Float32 COMMENT 'Cargo capacity (billion cubic meters)',
  eta DateTime COMMENT 'Estimated time of arrival',
  status LowCardinality(String) COMMENT 'EN_ROUTE, BERTHED, LOADING, UNLOADING',
  source LowCardinality(String) COMMENT 'AIS, port_ops, etc.',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/lng_vessel_tracking','{replica}', ingestion_timestamp)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (vessel_imo, timestamp)
TTL toDate(timestamp) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS lng_regas_capacity ON CLUSTER '{cluster}' (
  timestamp DateTime,
  terminal_id String,
  terminal_name String,
  region LowCardinality(String),
  send_out_capacity Float32 COMMENT 'Send-out capacity (MMcf/day)',
  utilization Float32 COMMENT 'Current utilization (0-1)',
  storage_level Float32 COMMENT 'Storage level (bcm)',
  maintenance_status LowCardinality(String),
  weather_factor Float32 COMMENT 'Weather impact on operations',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/lng_regas_capacity','{replica}', ingestion_timestamp)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (terminal_id, timestamp)
TTL toDate(timestamp) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS lng_power_impact ON CLUSTER '{cluster}' (
  timestamp DateTime,
  region LowCardinality(String),
  hub_id String COMMENT 'Gas or power hub',
  expected_sendout Float32 COMMENT 'Expected LNG send-out (MMcf/day)',
  hub_price_impact Float64 COMMENT 'Expected price impact ($/MMBtu or $/MWh)',
  impact_confidence_lower Float64,
  impact_confidence_upper Float64,
  causal_method LowCardinality(String) COMMENT 'IV, causal_forest, etc.',
  vessels_arriving UInt8 COMMENT 'Number of vessels in arrival window',
  model_version String,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/lng_power_impact','{replica}', created_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (region, hub_id, timestamp)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS lng_spread_forecasts ON CLUSTER '{cluster}' (
  forecast_id String,
  timestamp DateTime,
  from_hub String COMMENT 'Source gas hub',
  to_hub String COMMENT 'Destination power hub',
  spread_forecast Float64 COMMENT 'Expected spread ($/MWh or $/MMBtu)',
  lng_factor Float32 COMMENT 'Contribution from LNG arrivals',
  model_version String,
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/lng_spread_forecasts','{replica}', created_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (from_hub, to_hub, timestamp, forecast_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;
