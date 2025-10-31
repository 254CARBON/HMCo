"""
Great Expectations suite for ISO real-time LMP data (CAISO, MISO, SPP)
"""
from great_expectations.dataset import PandasDataset


def build_suite(df: PandasDataset):
    """Build expectations suite for RT LMP data"""
    
    # Column existence
    df.expect_table_columns_to_match_ordered_list([
        "ts", "iso", "node", "lmp", "congestion", "loss"
    ])
    
    # Nullability checks
    df.expect_column_values_to_not_be_null("ts")
    df.expect_column_values_to_not_be_null("iso")
    df.expect_column_values_to_not_be_null("node")
    df.expect_column_values_to_not_be_null("lmp")
    
    # Range checks for LMP components
    df.expect_column_values_to_be_between("lmp", min_value=-10000, max_value=10000)
    df.expect_column_values_to_be_between("congestion", min_value=-10000, max_value=10000)
    df.expect_column_values_to_be_between("loss", min_value=-10000, max_value=10000)
    
    # ISO domain check
    df.expect_column_values_to_be_in_set("iso", ["CAISO", "MISO", "SPP"])
    
    # Uniqueness (primary key)
    df.expect_compound_columns_to_be_unique(["ts", "iso", "node"])
    
    # Freshness check (data should be within last 15 minutes for RT)
    df.expect_column_max_to_be_between(
        "ts",
        min_value="now() - interval '15 minutes'",
        max_value="now() + interval '5 minutes'"
    )
    
    return df.validate()
