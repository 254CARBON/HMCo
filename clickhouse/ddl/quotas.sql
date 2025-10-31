-- ClickHouse quotas and resource limits
-- Tiered resource governance to prevent runaway queries

-- Interactive user quota (dashboards, Jupyter)
CREATE QUOTA IF NOT EXISTS interactive_quota
  FOR INTERVAL 1 hour MAX queries = 1000, 
                        MAX query_selects = 5000,
                        MAX result_rows = 10000000,
                        MAX result_bytes = 10000000000,  -- 10 GB
                        MAX execution_time = 3600  -- 1 hour
  FOR INTERVAL 1 day MAX queries = 10000,
                       MAX query_selects = 50000,
                       MAX result_rows = 100000000,
                       MAX result_bytes = 100000000000,  -- 100 GB
                       MAX execution_time = 86400  -- 24 hours
  TO interactive_users;

-- ETL user quota (pipelines, scheduled jobs)
CREATE QUOTA IF NOT EXISTS etl_quota
  FOR INTERVAL 1 hour MAX queries = 500,
                        MAX query_selects = 2000,
                        MAX result_rows = 100000000,
                        MAX result_bytes = 50000000000,  -- 50 GB
                        MAX execution_time = 7200  -- 2 hours
  FOR INTERVAL 1 day MAX queries = 5000,
                       MAX query_selects = 20000,
                       MAX result_rows = 1000000000,
                       MAX result_bytes = 500000000000,  -- 500 GB
                       MAX execution_time = 86400  -- 24 hours
  TO etl_users;

-- Batch user quota (long-running analytics)
CREATE QUOTA IF NOT EXISTS batch_quota
  FOR INTERVAL 1 hour MAX queries = 100,
                        MAX query_selects = 500,
                        MAX result_rows = 1000000000,
                        MAX result_bytes = 100000000000,  -- 100 GB
                        MAX execution_time = 14400  -- 4 hours
  FOR INTERVAL 1 day MAX queries = 1000,
                       MAX query_selects = 5000,
                       MAX result_rows = 10000000000,
                       MAX result_bytes = 1000000000000,  -- 1 TB
                       MAX execution_time = 86400  -- 24 hours
  TO batch_users;

-- Admin quota (unrestricted but monitored)
CREATE QUOTA IF NOT EXISTS admin_quota
  FOR INTERVAL 1 hour NO LIMITS
  FOR INTERVAL 1 day NO LIMITS
  TO admin_users;

-- Settings profiles for different user tiers
CREATE SETTINGS PROFILE IF NOT EXISTS interactive_profile SETTINGS
  max_threads = 8,
  max_memory_usage = 8000000000,  -- 8 GB
  max_execution_time = 300,  -- 5 minutes
  max_rows_to_read = 10000000,
  max_bytes_to_read = 10000000000,  -- 10 GB
  readonly = 1
  TO interactive_users;

CREATE SETTINGS PROFILE IF NOT EXISTS etl_profile SETTINGS
  max_threads = 16,
  max_memory_usage = 32000000000,  -- 32 GB
  max_execution_time = 7200,  -- 2 hours
  max_rows_to_read = 100000000,
  max_bytes_to_read = 100000000000,  -- 100 GB
  readonly = 0
  TO etl_users;

CREATE SETTINGS PROFILE IF NOT EXISTS batch_profile SETTINGS
  max_threads = 32,
  max_memory_usage = 64000000000,  -- 64 GB
  max_execution_time = 14400,  -- 4 hours
  max_rows_to_read = 1000000000,
  max_bytes_to_read = 500000000000,  -- 500 GB
  readonly = 0
  TO batch_users;

CREATE SETTINGS PROFILE IF NOT EXISTS admin_profile SETTINGS
  max_threads = 64,
  max_memory_usage = 128000000000,  -- 128 GB
  max_execution_time = 86400,  -- 24 hours
  readonly = 0
  TO admin_users;

-- Query killer for runaway queries
-- Requires ClickHouse keeper or zookeeper
CREATE TABLE IF NOT EXISTS query_killer_log (
  event_time DateTime,
  query_id String,
  user String,
  query String,
  reason String,
  duration_seconds UInt32,
  memory_usage UInt64
)
ENGINE = MergeTree()
ORDER BY event_time;
