-- Migration: Create providers and runs tables
-- Provides basic schema needed for ingestion orchestration.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS providers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  type VARCHAR(50) NOT NULL,
  status VARCHAR(50) DEFAULT 'inactive',
  uis TEXT NOT NULL,
  config JSONB DEFAULT '{}'::jsonb,
  schedule VARCHAR(255),
  last_run_at TIMESTAMP,
  next_run_at TIMESTAMP,
  total_runs INT DEFAULT 0,
  success_rate FLOAT DEFAULT 100.0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider_id UUID NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
  status VARCHAR(50) NOT NULL,
  started_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP,
  records_ingested INT DEFAULT 0,
  records_failed INT DEFAULT 0,
  duration INT,
  logs TEXT,
  error_message TEXT,
  parameters JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runs_provider_id ON runs(provider_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);

