"""
Great Expectations suite for EIA daily fuel data
"""
from great_expectations.dataset import PandasDataset


def build_suite(df: PandasDataset):
    """Build expectations suite for EIA daily data"""
    
    # Column existence
    df.expect_table_columns_to_match_ordered_list([
        "ts", "region", "series", "value"
    ])
    
    # Nullability checks
    df.expect_column_values_to_not_be_null("ts")
    df.expect_column_values_to_not_be_null("region")
    df.expect_column_values_to_not_be_null("series")
    df.expect_column_values_to_not_be_null("value")
    
    # Range checks
    df.expect_column_values_to_be_between("value", min_value=-1e9, max_value=1e9)
    
    # Uniqueness (primary key)
    df.expect_compound_columns_to_be_unique(["ts", "region", "series"])
    
    # Freshness check (data should be within last 48 hours)
    df.expect_column_max_to_be_between(
        "ts",
        min_value="now() - interval '48 hours'",
        max_value="now() + interval '1 hour'"
    )
    
    return df.validate()
