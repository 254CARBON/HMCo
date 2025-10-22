-- PostgreSQL initialization script for Iceberg REST Catalog
-- This script sets up the required database schema and permissions

-- Connect to iceberg_rest database
-- Note: This assumes the database was already created by postgres-shared-init ConfigMap

CREATE SCHEMA IF NOT EXISTS iceberg_catalog;

-- Grant privileges directly to the catalog login to avoid SET ROLE caching issues
GRANT USAGE, CREATE ON SCHEMA iceberg_catalog TO iceberg_user;
GRANT USAGE, CREATE ON SCHEMA public TO iceberg_user;
GRANT ALL ON ALL TABLES IN SCHEMA iceberg_catalog TO iceberg_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA iceberg_catalog TO iceberg_user;

-- The Iceberg REST Catalog will auto-create tables as needed:
-- - iceberg_tables: stores table metadata
-- - iceberg_namespaces: stores namespace information
-- - iceberg_views: stores view definitions (optional)

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant extension usage
GRANT USAGE ON SCHEMA public TO iceberg_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES FOR USER postgres IN SCHEMA public GRANT ALL ON TABLES TO iceberg_user;
ALTER DEFAULT PRIVILEGES FOR USER postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO iceberg_user;
ALTER DEFAULT PRIVILEGES FOR USER postgres IN SCHEMA iceberg_catalog GRANT ALL ON TABLES TO iceberg_user;
ALTER DEFAULT PRIVILEGES FOR USER postgres IN SCHEMA iceberg_catalog GRANT ALL ON SEQUENCES TO iceberg_user;

-- Ensure new sessions default to the Iceberg schema first
ALTER ROLE iceberg_user IN DATABASE iceberg_rest SET search_path = iceberg_catalog, public;

-- Test connectivity
SELECT current_user, current_database(), now();
