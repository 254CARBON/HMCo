-- MLFlow Backend Database Setup Script
-- Run this script against the shared PostgreSQL instance

-- Create mlflow user if not exists
DO $$ BEGIN
  CREATE USER mlflow WITH PASSWORD 'mlflow-secure-password-change-me';
EXCEPTION WHEN DUPLICATE_OBJECT THEN
  RAISE NOTICE 'mlflow user already exists';
END $$;

-- Create mlflow database
CREATE DATABASE mlflow OWNER mlflow;

-- Grant privileges to mlflow user
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Connect to mlflow database for schema setup
\c mlflow

-- Create schema for MLFlow backend store
CREATE SCHEMA IF NOT EXISTS mlflow AUTHORIZATION mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlflow GRANT ALL ON TABLES TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlflow GRANT ALL ON SEQUENCES TO mlflow;

-- MLFlow automatically creates tables on first run, but we ensure proper schema
-- The following tables will be created by MLFlow:
-- - experiments (experiment metadata)
-- - runs (run/trial metadata)
-- - metrics (time-series metrics)
-- - params (parameters)
-- - tags (arbitrary key-value tags)
-- - model_versions (model registry)
-- - registered_models (model registry)

GRANT ALL PRIVILEGES ON SCHEMA mlflow TO mlflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mlflow TO mlflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA mlflow TO mlflow;

-- Verify setup
SELECT
  datname as database,
  usename as owner
FROM pg_database
JOIN pg_user ON pg_database.datdba = pg_user.usesysid
WHERE datname = 'mlflow';
