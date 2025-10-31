-- ISO Node Mapping tables for canonical node/hub identifiers

-- Node mapping table
CREATE TABLE IF NOT EXISTS iso_node_mapping ON CLUSTER '{cluster}' (
  iso LowCardinality(String),
  native_node_id String,
  canonical_node_id String,
  node_name String,
  node_type LowCardinality(String),
  latitude Nullable(Float64),
  longitude Nullable(Float64),
  zone Nullable(String),
  hub Nullable(String),
  voltage_kv Nullable(UInt16),
  active UInt8 DEFAULT 1,
  updated_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/iso_node_mapping','{replica}', updated_at)
ORDER BY (iso, native_node_id)
SETTINGS index_granularity=8192;

-- Hub mapping table
CREATE TABLE IF NOT EXISTS iso_hub_mapping ON CLUSTER '{cluster}' (
  iso LowCardinality(String),
  native_hub_id String,
  canonical_hub_id String,
  hub_name String,
  zone Nullable(String),
  active UInt8 DEFAULT 1,
  updated_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/iso_hub_mapping','{replica}', updated_at)
ORDER BY (iso, native_hub_id)
SETTINGS index_granularity=8192;

-- Zone mapping table
CREATE TABLE IF NOT EXISTS iso_zone_mapping ON CLUSTER '{cluster}' (
  iso LowCardinality(String),
  native_zone_id String,
  canonical_zone_id String,
  zone_name String,
  active UInt8 DEFAULT 1,
  updated_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/iso_zone_mapping','{replica}', updated_at)
ORDER BY (iso, native_zone_id)
SETTINGS index_granularity=8192;

-- Seed data for CAISO
INSERT INTO iso_hub_mapping (iso, native_hub_id, canonical_hub_id, hub_name, active) VALUES
  ('CAISO', 'SP15', 'CAISO_SP15', 'Southern California SP15', 1),
  ('CAISO', 'NP15', 'CAISO_NP15', 'Northern California NP15', 1),
  ('CAISO', 'ZP26', 'CAISO_ZP26', 'Zone 26', 1);

-- Seed data for MISO
INSERT INTO iso_hub_mapping (iso, native_hub_id, canonical_hub_id, hub_name, active) VALUES
  ('MISO', 'MINN.HUB', 'MISO_MINN', 'Minnesota Hub', 1),
  ('MISO', 'IND.HUB', 'MISO_IND', 'Indiana Hub', 1),
  ('MISO', 'ILL.HUB', 'MISO_ILL', 'Illinois Hub', 1),
  ('MISO', 'MICH.HUB', 'MISO_MICH', 'Michigan Hub', 1);

-- Seed data for SPP
INSERT INTO iso_hub_mapping (iso, native_hub_id, canonical_hub_id, hub_name, active) VALUES
  ('SPP', 'NORTH.HUB', 'SPP_NORTH', 'North Hub', 1),
  ('SPP', 'SOUTH.HUB', 'SPP_SOUTH', 'South Hub', 1);
