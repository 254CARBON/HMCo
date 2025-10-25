"""
Spark job for ingesting Polygon.io daily aggregates into Iceberg.

This job fetches OHLCV data for configured tickers, normalises the payload,
applies light enrichment, and writes the result to an Iceberg table with the
schema expected by downstream analytics and quality checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

DEFAULT_TICKERS = ["C:CL", "C:NG", "C:HO", "C:RB"]
POLYGON_BASE_URL = "https://api.polygon.io"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polygon.io Spark ingestion job")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated list of Polygon tickers to ingest",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to lookback_days before end date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today in UTC.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of days to look back when start date is not provided.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("POLYGON_API_KEY"),
        help="Polygon API key. Defaults to POLYGON_API_KEY environment variable.",
    )
    parser.add_argument(
        "--iceberg-catalog",
        type=str,
        default=os.getenv("ICEBERG_CATALOG", "iceberg"),
        help="Iceberg catalog name configured in Spark.",
    )
    parser.add_argument(
        "--iceberg-namespace",
        type=str,
        default=os.getenv("ICEBERG_NAMESPACE", "raw"),
        help="Iceberg namespace (database) to write into.",
    )
    parser.add_argument(
        "--iceberg-table",
        type=str,
        default=os.getenv("ICEBERG_TABLE", "polygon_market_ohlc"),
        help="Iceberg table name inside the namespace.",
    )
    parser.add_argument(
        "--iceberg-uri",
        type=str,
        default=os.getenv("ICEBERG_REST_URI", "http://iceberg-rest-catalog:8181"),
        help="Iceberg REST catalog endpoint.",
    )
    parser.add_argument(
        "--iceberg-warehouse",
        type=str,
        default=os.getenv("ICEBERG_WAREHOUSE", "s3://iceberg-warehouse/"),
        help="Warehouse location backing the Iceberg catalog.",
    )
    parser.add_argument(
        "--s3-endpoint",
        type=str,
        default=os.getenv("SPARK_S3_ENDPOINT", "http://minio-service:9000"),
        help="S3/MinIO endpoint accessible from Spark executors.",
    )
    parser.add_argument(
        "--s3-access-key",
        type=str,
        default=os.getenv("AWS_ACCESS_KEY_ID"),
        help="Access key for MinIO/S3.",
    )
    parser.add_argument(
        "--s3-secret-key",
        type=str,
        default=os.getenv("AWS_SECRET_ACCESS_KEY"),
        help="Secret key for MinIO/S3.",
    )
    parser.add_argument(
        "--checkpoint-location",
        type=str,
        default=os.getenv("CHECKPOINT_LOCATION"),
        help="Optional checkpoint location for incremental processing.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Safety cap for Polygon pagination depth.",
    )
    parser.add_argument(
        "--request-limit",
        type=int,
        default=5000,
        help="Number of records requested per Polygon page.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def resolve_date_range(args: argparse.Namespace) -> tuple[str, str]:
    end_dt = (
        datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.end_date
        else datetime.utcnow()
    )
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=args.lookback_days)

    if start_dt > end_dt:
        raise ValueError("Start date must be on or before end date.")

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def create_spark_session(args: argparse.Namespace) -> SparkSession:
    app_name = f"polygon-market-ingestion-{args.iceberg_table}"
    builder = (
        SparkSession.builder.appName(app_name)
        .config(
            f"spark.sql.catalog.{args.iceberg_catalog}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(
            f"spark.sql.catalog.{args.iceberg_catalog}.type",
            "rest",
        )
        .config(
            f"spark.sql.catalog.{args.iceberg_catalog}.uri",
            args.iceberg_uri,
        )
        .config(
            f"spark.sql.catalog.{args.iceberg_catalog}.warehouse",
            args.iceberg_warehouse,
        )
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.shuffle.partitions", "48")
    )

    spark = builder.getOrCreate()

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()  # type: ignore[attr-defined]
    hadoop_conf.set("fs.s3a.endpoint", args.s3_endpoint)
    if args.s3_access_key:
        hadoop_conf.set("fs.s3a.access.key", args.s3_access_key)
    if args.s3_secret_key:
        hadoop_conf.set("fs.s3a.secret.key", args.s3_secret_key)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", str(args.s3_endpoint.startswith("https")).lower())

    return spark


def ensure_iceberg_table(spark: SparkSession, args: argparse.Namespace) -> None:
    full_table_name = f"{args.iceberg_catalog}.{args.iceberg_namespace}.{args.iceberg_table}"
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {full_table_name} (
      ticker STRING,
      event_timestamp BIGINT,
      event_time_utc TIMESTAMP,
      trading_day DATE,
      open_price DOUBLE,
      high_price DOUBLE,
      low_price DOUBLE,
      close_price DOUBLE,
      volume BIGINT,
      transactions BIGINT,
      vwap DOUBLE,
      instrument_name STRING,
      primary_exchange STRING,
      ingest_batch_id STRING,
      ingested_at TIMESTAMP
    )
    USING iceberg
    PARTITIONED BY (trading_day, ticker)
    TBLPROPERTIES (
      'format-version' = '2',
      'write.parquet.compression-codec' = 'zstd',
      'write.delete.mode' = 'merge-on-read'
    )
    """
    logging.info("Ensuring Iceberg table exists: %s", full_table_name)
    spark.sql(ddl)


def build_schema() -> StructType:
    return StructType(
        [
            StructField("ticker", StringType(), False),
            StructField("event_timestamp", LongType(), False),
            StructField("open_price", DoubleType(), True),
            StructField("high_price", DoubleType(), True),
            StructField("low_price", DoubleType(), True),
            StructField("close_price", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("transactions", DoubleType(), True),
            StructField("vwap", DoubleType(), True),
            StructField("instrument_name", StringType(), True),
            StructField("primary_exchange", StringType(), True),
        ]
    )


def fetch_polygon_aggregates(
    session: requests.Session,
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    max_pages: int,
    request_limit: int,
) -> List[Dict[str, Optional[float]]]:
    records: List[Dict[str, Optional[float]]] = []
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": request_limit,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    pages = 0
    while url and pages < max_pages:
        pages += 1
        response = session.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 429:
            raise RuntimeError("Polygon API rate limited the request; consider lowering concurrency.")
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) or []

        for item in results:
            records.append(
                {
                    "ticker": ticker,
                    "event_timestamp": item.get("t"),
                    "open_price": item.get("o"),
                    "high_price": item.get("h"),
                    "low_price": item.get("l"),
                    "close_price": item.get("c"),
                    "volume": item.get("v"),
                    "transactions": item.get("n"),
                    "vwap": item.get("vw"),
                    "instrument_name": item.get("T"),
                    "primary_exchange": item.get("x"),
                }
            )

        next_url = payload.get("next_url")
        url = (
            f"{next_url}&apiKey={api_key}"
            if next_url and "apiKey=" not in next_url
            else next_url
        )
        params = None  # Pagination handled by next_url

    logging.info(
        "Fetched %s records for %s across %s page(s)",
        len(records),
        ticker,
        pages,
    )
    return records


def fetch_all_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    api_key: str,
    max_pages: int,
    request_limit: int,
) -> List[Dict[str, Optional[float]]]:
    session = requests.Session()
    all_records: List[Dict[str, Optional[float]]] = []
    for ticker in tickers:
        ticker = ticker.strip()
        if not ticker:
            continue
        ticker_records = fetch_polygon_aggregates(
            session=session,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            max_pages=max_pages,
            request_limit=request_limit,
        )
        all_records.extend(ticker_records)
    return all_records


def to_dataframe(
    spark: SparkSession,
    records: List[Dict[str, Optional[float]]],
    ingest_batch_id: str,
) -> DataFrame:
    schema = build_schema()
    df = spark.createDataFrame(records, schema=schema)
    if df.rdd.isEmpty():
        logging.warning("No records retrieved from Polygon API.")
        return df

    df = (
        df.withColumn("event_time_utc", F.from_unixtime(F.col("event_timestamp") / 1000.0))
        .withColumn("trading_day", F.to_date("event_time_utc"))
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("ingest_batch_id", F.lit(ingest_batch_id))
    )

    df = df.dropDuplicates(["ticker", "event_timestamp"])
    return df.select(
        "ticker",
        "event_timestamp",
        "event_time_utc",
        "trading_day",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "transactions",
        "vwap",
        "instrument_name",
        "primary_exchange",
        "ingest_batch_id",
        "ingested_at",
    )


def write_to_iceberg(df: DataFrame, args: argparse.Namespace) -> int:
    if df.rdd.isEmpty():
        logging.warning("Skipping Iceberg write because the DataFrame is empty.")
        return 0

    target_table = f"{args.iceberg_catalog}.{args.iceberg_namespace}.{args.iceberg_table}"
    record_count = df.count()
    logging.info("Writing %s records to %s", record_count, target_table)

    (
        df.write.format("iceberg")
        .mode("append")
        .save(target_table)
    )
    return record_count


def main() -> None:
    configure_logging()
    args = parse_args()

    if not args.api_key:
        raise RuntimeError("Polygon API key is required (pass --api-key or set POLYGON_API_KEY).")

    start_date, end_date = resolve_date_range(args)
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    ingest_batch_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{len(tickers)}tickers"

    logging.info(
        "Starting Polygon ingestion for tickers=%s range=%s..%s",
        tickers,
        start_date,
        end_date,
    )

    raw_records = fetch_all_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        api_key=args.api_key,
        max_pages=args.max_pages,
        request_limit=args.request_limit,
    )

    spark = create_spark_session(args)
    ensure_iceberg_table(spark, args)

    df = to_dataframe(spark, raw_records, ingest_batch_id)
    record_count = write_to_iceberg(df, args)
    logging.info("Polygon ingestion complete. Records written: %s", record_count)
    spark.stop()


if __name__ == "__main__":
    main()
