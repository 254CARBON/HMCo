-- Iceberg Table for Deequ Quality Check Results
-- This table stores the results of all data quality checks
CREATE TABLE IF NOT EXISTS iceberg.monitoring.deequ_quality_checks (
  check_id STRING,
  check_timestamp TIMESTAMP,
  check_date DATE,
  table_name STRING,
  table_schema STRING,
  check_name STRING,
  check_type STRING,
  check_column STRING,
  threshold_value DECIMAL(5, 2),
  actual_value DECIMAL(5, 2),
  status STRING,
  passed BOOLEAN,
  record_count BIGINT,
  failure_count BIGINT,
  matched_count BIGINT,
  error_message STRING,
  metadata MAP<STRING, STRING>,
  execution_time_ms BIGINT,
  created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (table_name, check_date)
TBLPROPERTIES (
  'write.parquet.compression-codec' = 'snappy',
  'format-version' = '2'
);

-- Iceberg Table for Statistical Profiles
-- This table stores column-level statistical profiles for anomaly detection
CREATE TABLE IF NOT EXISTS iceberg.monitoring.deequ_profiles (
  profile_id STRING,
  profile_timestamp TIMESTAMP,
  profile_date DATE,
  table_name STRING,
  table_schema STRING,
  column_name STRING,
  data_type STRING,
  -- Basic statistics
  record_count BIGINT,
  distinct_count BIGINT,
  completeness DECIMAL(5, 2),
  distinctness DECIMAL(5, 2),
  -- Numeric statistics
  min_value DECIMAL(38, 10),
  max_value DECIMAL(38, 10),
  mean_value DECIMAL(38, 10),
  stddev_value DECIMAL(38, 10),
  median_value DECIMAL(38, 10),
  -- Percentiles for anomaly detection
  percentile_25 DECIMAL(38, 10),
  percentile_75 DECIMAL(38, 10),
  percentile_95 DECIMAL(38, 10),
  -- String statistics
  max_length INT,
  min_length INT,
  avg_length DECIMAL(10, 2),
  -- Value distribution
  top_values ARRAY<STRING>,
  top_value_counts ARRAY<BIGINT>,
  null_count BIGINT,
  null_percentage DECIMAL(5, 2),
  -- Temporal information
  min_date DATE,
  max_date DATE,
  avg_date_gap_days DOUBLE,
  -- Quality metrics
  entropy DOUBLE,
  skewness DOUBLE,
  kurtosis DOUBLE,
  created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (table_name, profile_date)
TBLPROPERTIES (
  'write.parquet.compression-codec' = 'snappy',
  'format-version' = '2'
);

-- Iceberg Table for Detected Anomalies
-- This table stores anomalies detected via statistical analysis
CREATE TABLE IF NOT EXISTS iceberg.monitoring.deequ_anomalies (
  anomaly_id STRING,
  anomaly_timestamp TIMESTAMP,
  anomaly_date DATE,
  table_name STRING,
  table_schema STRING,
  column_name STRING,
  anomaly_type STRING,
  severity STRING,
  -- Baseline values
  baseline_value DOUBLE,
  baseline_timestamp TIMESTAMP,
  -- Current values
  current_value DOUBLE,
  current_timestamp TIMESTAMP,
  -- Deviation metrics
  absolute_deviation DOUBLE,
  relative_deviation DECIMAL(5, 2),
  stddev_count DECIMAL(5, 2),
  iqr_multiplier DECIMAL(5, 2),
  -- Detection details
  detection_method STRING,
  confidence_score DECIMAL(3, 2),
  supporting_records INT,
  is_confirmed BOOLEAN,
  investigation_status STRING,
  notes STRING,
  created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (table_name, anomaly_date)
TBLPROPERTIES (
  'write.parquet.compression-codec' = 'snappy',
  'format-version' = '2'
);

-- Iceberg Table for Quality Check History
-- This table tracks historical quality trends for dashboard/analytics
CREATE TABLE IF NOT EXISTS iceberg.monitoring.deequ_quality_history (
  history_date DATE,
  table_name STRING,
  check_name STRING,
  check_type STRING,
  passed_count INT,
  warning_count INT,
  failed_count INT,
  total_count INT,
  pass_rate DECIMAL(5, 2),
  avg_score DECIMAL(5, 2),
  min_score DECIMAL(5, 2),
  max_score DECIMAL(5, 2),
  trend VARCHAR(50),
  created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (history_date, table_name)
TBLPROPERTIES (
  'write.parquet.compression-codec' = 'snappy',
  'format-version' = '2'
);

-- View: Quality Check Summary by Table
CREATE VIEW IF NOT EXISTS iceberg.monitoring.deequ_quality_summary AS
SELECT
  table_name,
  check_date,
  COUNT(*) as total_checks,
  SUM(CASE WHEN status = 'PASSED' THEN 1 ELSE 0 END) as passed_checks,
  SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as warning_checks,
  SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_checks,
  ROUND(SUM(CASE WHEN status = 'PASSED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pass_rate,
  ROUND(AVG(CAST(actual_value as DECIMAL(5, 2))), 2) as avg_quality_score,
  MAX(check_timestamp) as last_check_time
FROM iceberg.monitoring.deequ_quality_checks
GROUP BY table_name, check_date
ORDER BY check_date DESC, table_name;

-- View: Recent Anomalies (last 7 days)
CREATE VIEW IF NOT EXISTS iceberg.monitoring.deequ_recent_anomalies AS
SELECT
  table_name,
  column_name,
  anomaly_type,
  severity,
  COUNT(*) as anomaly_count,
  MAX(anomaly_timestamp) as latest_anomaly,
  SUM(CASE WHEN is_confirmed THEN 1 ELSE 0 END) as confirmed_count,
  SUM(CASE WHEN investigation_status = 'RESOLVED' THEN 1 ELSE 0 END) as resolved_count
FROM iceberg.monitoring.deequ_anomalies
WHERE anomaly_date >= CURRENT_DATE - INTERVAL '7' DAY
  AND investigation_status != 'FALSE_POSITIVE'
GROUP BY table_name, column_name, anomaly_type, severity
ORDER BY severity DESC, latest_anomaly DESC;
