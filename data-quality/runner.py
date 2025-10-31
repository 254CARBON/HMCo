#!/usr/bin/env python3
"""
Data Quality Runner using Great Expectations
Pulls sample data from Iceberg/Trino or ClickHouse and runs validation suites
"""

import argparse
import sys
from pathlib import Path
import importlib.util

import pandas as pd
from sqlalchemy import create_engine
from great_expectations.dataset import PandasDataset


def load_suite(suite_name: str):
    """Load a GE expectations suite by name"""
    suite_path = Path(__file__).parent / "ge_expectations" / f"{suite_name}.py"
    
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite not found: {suite_path}")
    
    spec = importlib.util.spec_from_file_location(suite_name, suite_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.build_suite


def get_sample_data(table: str, engine_type: str = "trino", limit: int = 1000) -> pd.DataFrame:
    """Fetch sample data from Iceberg (via Trino) or ClickHouse"""
    
    if engine_type == "trino":
        # Trino connection for Iceberg tables
        engine = create_engine("trino://trino-coordinator:8080/iceberg")
        query = f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}"
    elif engine_type == "clickhouse":
        # ClickHouse connection
        engine = create_engine("clickhouse://clickhouse:8123/default")
        query = f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}"
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    return pd.read_sql(query, engine)


def main():
    parser = argparse.ArgumentParser(description="Run data quality checks")
    parser.add_argument("--suite", required=True, help="GE suite name (e.g., eia_daily)")
    parser.add_argument("--table", required=True, help="Table name (e.g., hub_curated.eia_daily_fuel)")
    parser.add_argument("--clickhouse", action="store_true", help="Use ClickHouse instead of Trino")
    parser.add_argument("--limit", type=int, default=1000, help="Sample size")
    
    args = parser.parse_args()
    
    engine_type = "clickhouse" if args.clickhouse else "trino"
    
    print(f"Loading suite: {args.suite}")
    suite_func = load_suite(args.suite)
    
    print(f"Fetching sample data from {args.table} via {engine_type}")
    df = get_sample_data(args.table, engine_type, args.limit)
    
    print(f"Running validation on {len(df)} rows")
    ge_df = PandasDataset(df)
    results = suite_func(ge_df)
    
    # Check results
    if results.success:
        print("✅ All expectations passed!")
        return 0
    else:
        print("❌ Some expectations failed:")
        for result in results.results:
            if not result.success:
                print(f"  - {result.expectation_config.expectation_type}: {result.exception_info}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
