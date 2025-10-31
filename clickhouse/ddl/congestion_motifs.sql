-- Topology Motif Miner: Recurrent congestion patterns
-- Library of motifs with entry/exit signals and typical duration

CREATE TABLE IF NOT EXISTS congestion_motifs ON CLUSTER '{cluster}' (
  motif_id String COMMENT 'Unique motif identifier',
  iso LowCardinality(String),
  motif_name String COMMENT 'Human-readable motif name',
  pattern_signature String COMMENT 'SAX/Matrix Profile signature',
  constrained_lines Array(String) COMMENT 'Lines involved in pattern',
  affected_nodes Array(String) COMMENT 'Nodes with price impact',
  avg_duration_hours Float32 COMMENT 'Average duration in hours',
  occurrence_count UInt32 COMMENT 'Historical occurrence count',
  avg_lmp_spread Float64 COMMENT 'Average LMP spread during motif ($/MWh)',
  precursor_signals String COMMENT 'JSON of leading indicators',
  discovery_timestamp DateTime COMMENT 'When motif was discovered',
  last_updated DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/congestion_motifs','{replica}', last_updated)
PARTITION BY iso
ORDER BY (iso, motif_id)
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS congestion_motif_occurrences ON CLUSTER '{cluster}' (
  occurrence_id String,
  motif_id String,
  iso LowCardinality(String),
  start_time DateTime,
  end_time DateTime,
  duration_hours Float32,
  max_lmp_spread Float64,
  avg_lmp_spread Float64,
  captured_correctly UInt8 COMMENT '1 if motif detected in real-time',
  ingestion_timestamp DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/congestion_motif_occurrences','{replica}', ingestion_timestamp)
PARTITION BY (iso, toYYYYMM(start_time))
ORDER BY (iso, motif_id, start_time)
TTL toDate(start_time) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity=8192;

CREATE TABLE IF NOT EXISTS congestion_motif_alerts ON CLUSTER '{cluster}' (
  alert_id String,
  timestamp DateTime,
  iso LowCardinality(String),
  motif_id String,
  confidence_score Float32 COMMENT 'Detection confidence 0-1',
  expected_duration_hours Float32,
  expected_spread Float64,
  recommended_action String COMMENT 'Trading recommendation',
  created_at DateTime DEFAULT now()
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/congestion_motif_alerts','{replica}', created_at)
PARTITION BY (iso, toYYYYMM(timestamp))
ORDER BY (iso, timestamp, alert_id)
TTL toDate(timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity=8192;
