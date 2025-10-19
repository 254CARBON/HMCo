-- PostgreSQL initialization script for Iceberg REST Catalog
-- This script sets up the required database schema and permissions

-- Connect to iceberg_rest database
-- Note: This assumes the database was already created by postgres-shared-init ConfigMap

-- Create schema for Iceberg metadata
CREATE SCHEMA IF NOT EXISTS iceberg_catalog;

-- Grant permissions to iceberg_user
GRANT ALL PRIVILEGES ON SCHEMA iceberg_catalog TO iceberg_user;
GRANT USAGE ON SCHEMA public TO iceberg_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO iceberg_user;

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

-- Create an application role for better permission management
CREATE ROLE iceberg_app NOINHERIT;
GRANT iceberg_app TO iceberg_user;

-- Test connectivity
SELECT current_user, current_database(), now();
