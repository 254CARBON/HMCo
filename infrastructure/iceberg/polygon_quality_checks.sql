-- Monitoring table for Polygon Deequ quality checks

CREATE TABLE IF NOT EXISTS iceberg.monitoring.polygon_quality_checks (
  table_name STRING,
  check_name STRING,
  check_level STRING,
  check_status STRING,
  constraint_message STRING,
  actual_value STRING,
  expected_value STRING,
  check_timestamp TIMESTAMP,
  trading_day DATE,
  record_count BIGINT,
  freshness_hours DOUBLE,
  metrics STRING
)
USING iceberg
PARTITIONED BY (table_name, trading_day)
TBLPROPERTIES (
  'format-version' = '2',
  'write.parquet.compression-codec' = 'zstd'
);
