-- Polygon Market Data Iceberg table definition
-- Run with Spark SQL or Trino:  CREATE TABLE IF NOT EXISTS raw.polygon_market_ohlc ...

CREATE TABLE IF NOT EXISTS iceberg.raw.polygon_market_ohlc (
  ticker STRING COMMENT 'Polygon ticker symbol (e.g. C:CL)',
  event_timestamp BIGINT COMMENT 'Epoch millis for the aggregated candle',
  event_time_utc TIMESTAMP COMMENT 'UTC timestamp derived from event_timestamp',
  trading_day DATE COMMENT 'Trading day (UTC)',
  open_price DOUBLE COMMENT 'Opening price',
  high_price DOUBLE COMMENT 'Session high price',
  low_price DOUBLE COMMENT 'Session low price',
  close_price DOUBLE COMMENT 'Closing price',
  volume BIGINT COMMENT 'Aggregated volume',
  transactions BIGINT COMMENT 'Number of transactions in the interval',
  vwap DOUBLE COMMENT 'Volume weighted average price',
  instrument_name STRING COMMENT 'Instrument name resolved from reference endpoint',
  primary_exchange STRING COMMENT 'Primary exchange code',
  ingest_batch_id STRING COMMENT 'Identifier for the ingestion batch',
  ingested_at TIMESTAMP COMMENT 'Ingestion timestamp (UTC)'
)
USING iceberg
PARTITIONED BY (trading_day, ticker)
TBLPROPERTIES (
  'format-version' = '2',
  'write.parquet.compression-codec' = 'zstd',
  'write.metadata.delete-after-commit.enabled' = 'true',
  'commit.retention.duration' = '7 d'
);
