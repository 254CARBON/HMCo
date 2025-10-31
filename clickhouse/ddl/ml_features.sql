-- Feature Store: Online features (ClickHouse) for low-latency serving
-- Offline features stored in Iceberg for training sets

-- Online feature values table
CREATE TABLE IF NOT EXISTS ml_feature_values_online ON CLUSTER '{cluster}' (
  entity_id String,
  feature_name String,
  feature_value Float64,
  feature_timestamp DateTime,
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ml_feature_values_online','{replica}', ingestion_timestamp)
PARTITION BY toDate(feature_timestamp)
ORDER BY (entity_id, feature_name, feature_timestamp)
TTL toDate(feature_timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;

-- Feature registry metadata (for discovery)
CREATE TABLE IF NOT EXISTS ml_feature_registry ON CLUSTER '{cluster}' (
  feature_name String,
  feature_group String,
  feature_type LowCardinality(String),
  description String,
  owner String,
  created_at DateTime,
  updated_at DateTime DEFAULT now(),
  entity_type LowCardinality(String),
  value_type LowCardinality(String),
  tags Array(String),
  online_enabled UInt8 DEFAULT 1,
  offline_table String
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ml_feature_registry','{replica}', updated_at)
ORDER BY (feature_group, feature_name)
SETTINGS index_granularity=8192;

-- Materialized view for latest feature values per entity
CREATE MATERIALIZED VIEW IF NOT EXISTS ml_features_latest
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/ml_features_latest','{replica}', feature_timestamp)
PARTITION BY entity_id
ORDER BY (entity_id, feature_name)
AS
SELECT
  entity_id,
  feature_name,
  feature_value,
  feature_timestamp
FROM ml_feature_values_online
QUALIFY ROW_NUMBER() OVER (PARTITION BY entity_id, feature_name ORDER BY feature_timestamp DESC) = 1;
