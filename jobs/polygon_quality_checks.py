"""
Deequ quality checks for Polygon market data.

This Spark job evaluates completeness, uniqueness, and statistical thresholds
for the `raw.polygon_market_ohlc` Iceberg table and records the results in
`monitoring.polygon_quality_checks`.
"""

from __future__ import annotations

import argparse
import logging
import json
import os
from datetime import datetime, timezone
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

try:
    from pydeequ.checks import Check, CheckLevel
    from pydeequ.verification import VerificationResult, VerificationSuite
except ImportError as exc:  # pragma: no cover - explicit guidance
    raise RuntimeError(
        "pydeequ is required for Polygon quality checks. "
        "Install it with `pip install pydeequ` or add the Deequ jar to Spark."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polygon Deequ quality checks")
    parser.add_argument(
        "--catalog",
        default=os.getenv("ICEBERG_CATALOG", "iceberg"),
        help="Spark Iceberg catalog name.",
    )
    parser.add_argument(
        "--source-table",
        default=os.getenv("POLYGON_SOURCE_TABLE", "raw.polygon_market_ohlc"),
        help="Source table (namespace.table) containing Polygon data.",
    )
    parser.add_argument(
        "--target-table",
        default=os.getenv("POLYGON_QUALITY_TABLE", "monitoring.polygon_quality_checks"),
        help="Destination table for persisting Deequ results.",
    )
    parser.add_argument(
        "--iceberg-uri",
        default=os.getenv("ICEBERG_REST_URI", "http://iceberg-rest-catalog:8181"),
        help="Iceberg REST catalog endpoint.",
    )
    parser.add_argument(
        "--iceberg-warehouse",
        default=os.getenv("ICEBERG_WAREHOUSE", "s3://iceberg-warehouse/"),
        help="Warehouse backing the Iceberg catalog.",
    )
    parser.add_argument(
        "--s3-endpoint",
        default=os.getenv("SPARK_S3_ENDPOINT", "http://minio-service:9000"),
        help="S3/MinIO endpoint accessible by executors.",
    )
    parser.add_argument(
        "--s3-access-key",
        default=os.getenv("AWS_ACCESS_KEY_ID"),
        help="Access key for MinIO/S3.",
    )
    parser.add_argument(
        "--s3-secret-key",
        default=os.getenv("AWS_SECRET_ACCESS_KEY"),
        help="Secret key for MinIO/S3.",
    )
    parser.add_argument(
        "--check-level",
        choices=["warning", "error"],
        default="warning",
        help="Deequ check severity level.",
    )
    parser.add_argument(
        "--freshness-threshold-hours",
        type=int,
        default=6,
        help="Fail freshness check when latest trading day is older than this many hours.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [polygon-quality] %(message)s",
    )


def build_spark_session(args: argparse.Namespace) -> SparkSession:
    builder = (
        SparkSession.builder.appName("polygon-quality-checks")
        .config(
            f"spark.sql.catalog.{args.catalog}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(
            f"spark.sql.catalog.{args.catalog}.type",
            "rest",
        )
        .config(
            f"spark.sql.catalog.{args.catalog}.uri",
            args.iceberg_uri,
        )
        .config(
            f"spark.sql.catalog.{args.catalog}.warehouse",
            args.iceberg_warehouse,
        )
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.shuffle.partitions", "48")
    )
    spark = builder.getOrCreate()
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()  # type: ignore[attr-defined]
    hadoop_conf.set("fs.s3a.endpoint", args.s3_endpoint)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    if args.s3_access_key:
        hadoop_conf.set("fs.s3a.access.key", args.s3_access_key)
    if args.s3_secret_key:
        hadoop_conf.set("fs.s3a.secret.key", args.s3_secret_key)
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", str(args.s3_endpoint.startswith("https")).lower())
    return spark


def ensure_target_table(spark: SparkSession, args: argparse.Namespace) -> None:
    qualified_table = f"{args.catalog}.{args.target_table}"
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {qualified_table} (
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
    )
    """
    spark.sql(ddl)


def latest_trading_day_hours(spark: SparkSession, catalog: str, source_table: str) -> float:
    qualified_table = f"{catalog}.{source_table}"
    df = spark.table(qualified_table)
    latest = df.select(F.max("event_time_utc").alias("latest_ts")).collect()
    if not latest or latest[0]["latest_ts"] is None:
        return float("inf")

    latest_ts = latest[0]["latest_ts"]
    if isinstance(latest_ts, datetime):
        latest_dt = latest_ts.replace(tzinfo=timezone.utc)
    else:
        # Fallback to parsing string representation
        latest_dt = datetime.fromisoformat(str(latest_ts)).replace(tzinfo=timezone.utc)
    delta = datetime.utcnow().replace(tzinfo=timezone.utc) - latest_dt
    return delta.total_seconds() / 3600.0


def build_checks(spark: SparkSession, level: str) -> Check:
    check_level = CheckLevel.Warning if level == "warning" else CheckLevel.Error

    check = Check(spark, check_level, "polygon-market-quality")
    check = (
        check.isComplete("ticker")
        .isComplete("trading_day")
        .isComplete("close_price")
        .isNonNegative("open_price")
        .isNonNegative("high_price")
        .isNonNegative("low_price")
        .isNonNegative("close_price")
        .isNonNegative("volume")
        .hasMin("open_price", lambda v: v > 0, "open_price_positive")
        .hasMin("close_price", lambda v: v > 0, "close_price_positive")
        .hasNumberOfDistinctValues("ticker", lambda cnt: cnt >= 1, "tickers_present")
        .isContainedIn("primary_exchange", ["NYM", "NYE", "CME", "CBOT", "COM"], "exchange_whitelist")
        .hasUniqueness(["ticker", "event_timestamp"])
    )
    return check


def run_quality_checks(
    spark: SparkSession,
    args: argparse.Namespace,
) -> List[str]:
    qualified_source = f"{args.catalog}.{args.source_table}"
    df = spark.table(qualified_source)

    freshness_hours = latest_trading_day_hours(spark, args.catalog, args.source_table)
    logging.info("Latest trading data freshness: %.2f hours", freshness_hours)
    if freshness_hours > args.freshness_threshold_hours:
        logging.warning(
            "Freshness threshold exceeded (%.2f h > %s h)",
            freshness_hours,
            args.freshness_threshold_hours,
        )

    check = build_checks(spark, args.check_level)

    verification_result = (
        VerificationSuite(spark)
        .onData(df)
        .addCheck(check)
        .run()
    )

    results_df = VerificationResult.checkResultsAsDataFrame(spark, verification_result)
    metrics = VerificationResult.successMetricsAsDataFrame(spark, verification_result)

    metrics_json = metrics.toJSON().collect()
    metrics_blob = json.dumps([json.loads(row) for row in metrics_json]) if metrics_json else "{}"
    record_count = df.count()

    enriched_results = (
        results_df
        .withColumn("table_name", F.lit(args.source_table))
        .withColumn("check_timestamp", F.current_timestamp())
        .withColumn("trading_day", F.to_date(F.current_timestamp()))
        .withColumn("record_count", F.lit(record_count))
        .withColumn("freshness_hours", F.lit(freshness_hours))
        .withColumn("metrics", F.lit(metrics_blob))
    )

    qualified_target = f"{args.catalog}.{args.target_table}"
    logging.info("Persisting quality results to %s", qualified_target)
    enriched_results.write.format("iceberg").mode("append").save(qualified_target)

    failures = [
        row.check
        for row in results_df.filter(F.col("check_status") != "Success").select("check").distinct().collect()
    ]
    return failures


def main() -> None:
    configure_logging()
    args = parse_args()

    spark = build_spark_session(args)
    ensure_target_table(spark, args)

    failures = run_quality_checks(spark, args)
    if failures:
        logging.warning("Deequ checks failed: %s", failures)
    else:
        logging.info("All Deequ checks succeeded for Polygon dataset.")

    spark.stop()


if __name__ == "__main__":
    main()
