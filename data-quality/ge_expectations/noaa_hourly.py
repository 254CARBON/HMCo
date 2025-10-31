"""
Great Expectations suite for NOAA hourly weather data
"""
from great_expectations.dataset import PandasDataset


def build_suite(df: PandasDataset):
    """Build expectations suite for NOAA hourly data"""
    
    # Column existence
    df.expect_table_columns_to_match_ordered_list([
        "ts", "grid_id", "temperature", "humidity", "wind_speed", "precipitation"
    ])
    
    # Nullability checks
    df.expect_column_values_to_not_be_null("ts")
    df.expect_column_values_to_not_be_null("grid_id")
    
    # Range checks for weather metrics
    df.expect_column_values_to_be_between("temperature", min_value=-100, max_value=150)
    df.expect_column_values_to_be_between("humidity", min_value=0, max_value=100)
    df.expect_column_values_to_be_between("wind_speed", min_value=0, max_value=200)
    df.expect_column_values_to_be_between("precipitation", min_value=0, max_value=100)
    
    # Uniqueness (primary key)
    df.expect_compound_columns_to_be_unique(["ts", "grid_id"])
    
    # Freshness check (data should be within last 2 hours)
    df.expect_column_max_to_be_between(
        "ts",
        min_value="now() - interval '2 hours'",
        max_value="now() + interval '1 hour'"
    )
    
    return df.validate()
