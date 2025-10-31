"""
Probabilistic data quality checks for anomaly and drift detection.
Statistical detectors that catch bad data beyond schema validation.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

try:
    from pyiceberg.catalog import load_catalog
    ICEBERG_AVAILABLE = True
except ImportError:
    ICEBERG_AVAILABLE = False


class SeasonalZScoreDetector:
    """Detect anomalies using seasonal z-scores (day-of-week, hour-of-day patterns)."""
    
    def __init__(self, threshold: float = 3.0, lookback_days: int = 30):
        self.threshold = threshold
        self.lookback_days = lookback_days
        
    def detect(self, df: pd.DataFrame, value_col: str, timestamp_col: str) -> pd.DataFrame:
        """
        Detect anomalies using seasonal z-scores.
        
        Args:
            df: DataFrame with time series data
            value_col: Column name for values to check
            timestamp_col: Column name for timestamps
            
        Returns:
            DataFrame with is_anomaly flag and anomaly_score
        """
        df = df.copy()
        df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
        df['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
        
        # Calculate seasonal statistics (by hour and day of week)
        seasonal_stats = df.groupby(['hour', 'day_of_week'])[value_col].agg(['mean', 'std']).reset_index()
        seasonal_stats.columns = ['hour', 'day_of_week', 'seasonal_mean', 'seasonal_std']
        
        # Join seasonal stats back
        df = df.merge(seasonal_stats, on=['hour', 'day_of_week'], how='left')
        
        # Calculate z-score
        df['z_score'] = (df[value_col] - df['seasonal_mean']) / (df['seasonal_std'] + 1e-9)
        df['anomaly_score'] = np.abs(df['z_score'])
        df['is_anomaly'] = df['anomaly_score'] > self.threshold
        
        return df


class EWMADetector:
    """Exponentially Weighted Moving Average for drift detection."""
    
    def __init__(self, span: int = 20, threshold_sigma: float = 3.0):
        self.span = span
        self.threshold_sigma = threshold_sigma
        
    def detect(self, df: pd.DataFrame, value_col: str, timestamp_col: str) -> pd.DataFrame:
        """
        Detect drift using EWMA.
        
        Args:
            df: DataFrame with time series data
            value_col: Column name for values to check
            timestamp_col: Column name for timestamps
            
        Returns:
            DataFrame with is_drift flag and drift_score
        """
        df = df.copy()
        df = df.sort_values(timestamp_col)
        
        # Calculate EWMA and EWMSTD
        df['ewma'] = df[value_col].ewm(span=self.span).mean()
        df['ewmstd'] = df[value_col].ewm(span=self.span).std()
        
        # Detect drift (deviation from EWMA)
        df['drift_score'] = np.abs(df[value_col] - df['ewma']) / (df['ewmstd'] + 1e-9)
        df['is_drift'] = df['drift_score'] > self.threshold_sigma
        
        return df


class KSTestDetector:
    """Kolmogorov-Smirnov test for distribution drift."""
    
    def __init__(self, p_value_threshold: float = 0.01, window_size: int = 1000):
        self.p_value_threshold = p_value_threshold
        self.window_size = window_size
        
    def detect(self, reference_data: np.ndarray, current_data: np.ndarray) -> Tuple[bool, float, float]:
        """
        Perform KS test to detect distribution drift.
        
        Args:
            reference_data: Historical reference distribution
            current_data: Current data to test
            
        Returns:
            (is_drift, ks_statistic, p_value)
        """
        ks_stat, p_value = stats.ks_2samp(reference_data, current_data)
        is_drift = p_value < self.p_value_threshold
        return is_drift, ks_stat, p_value


class IQROutlierDetector:
    """Interquartile Range outlier detection."""
    
    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        
    def detect(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df: DataFrame with data
            value_col: Column name for values to check
            
        Returns:
            DataFrame with is_outlier flag
        """
        df = df.copy()
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        
        df['is_outlier'] = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
        df['outlier_distance'] = np.where(
            df[value_col] < lower_bound,
            lower_bound - df[value_col],
            np.where(df[value_col] > upper_bound, df[value_col] - upper_bound, 0)
        )
        
        return df


class AnomalyEnsemble:
    """Ensemble of anomaly detectors with voting."""
    
    def __init__(self, min_votes: int = 2):
        self.detectors = {
            'seasonal_zscore': SeasonalZScoreDetector(),
            'ewma': EWMADetector(),
            'iqr': IQROutlierDetector()
        }
        self.min_votes = min_votes
        
    def detect(self, df: pd.DataFrame, value_col: str, timestamp_col: str) -> pd.DataFrame:
        """
        Run ensemble detection with voting.
        
        Args:
            df: DataFrame with time series data
            value_col: Column name for values to check
            timestamp_col: Column name for timestamps
            
        Returns:
            DataFrame with ensemble_anomaly flag and vote_count
        """
        df = df.copy()
        
        # Run all detectors
        df_seasonal = self.detectors['seasonal_zscore'].detect(df, value_col, timestamp_col)
        df_ewma = self.detectors['ewma'].detect(df, value_col, timestamp_col)
        df_iqr = self.detectors['iqr'].detect(df, value_col)
        
        # Count votes
        df['anomaly_votes'] = (
            df_seasonal['is_anomaly'].astype(int) +
            df_ewma['is_drift'].astype(int) +
            df_iqr['is_outlier'].astype(int)
        )
        
        df['ensemble_anomaly'] = df['anomaly_votes'] >= self.min_votes
        
        # Combine scores
        df['anomaly_score_max'] = np.maximum.reduce([
            df_seasonal['anomaly_score'],
            df_ewma['drift_score'],
            df_iqr['outlier_distance']
        ])
        
        return df


def quarantine_partition(
    table_name: str,
    partition_date: str,
    anomaly_summary: Dict,
    clickhouse_client
) -> bool:
    """
    Quarantine a partition that failed anomaly checks.
    
    Args:
        table_name: Name of the table
        partition_date: Partition date (YYYY-MM-DD)
        anomaly_summary: Summary of detected anomalies
        clickhouse_client: ClickHouse client
        
    Returns:
        True if quarantine successful
    """
    # Create quarantine table if not exists
    quarantine_table = f"{table_name}_quarantine"
    
    # Move partition to quarantine
    query = f"""
    ALTER TABLE {table_name}
    MOVE PARTITION '{partition_date}' TO TABLE {quarantine_table}
    """
    
    try:
        clickhouse_client.execute(query)
        
        # Log quarantine action
        log_query = f"""
        INSERT INTO data_quality_log (
            table_name, partition_date, action, anomaly_count, anomaly_summary, timestamp
        ) VALUES (
            '{table_name}', '{partition_date}', 'QUARANTINE',
            {anomaly_summary.get('anomaly_count', 0)},
            '{str(anomaly_summary)}', now()
        )
        """
        clickhouse_client.execute(log_query)
        
        return True
    except Exception as e:
        print(f"Error quarantining partition: {e}")
        return False


def tag_iceberg_snapshot(
    table_path: str,
    snapshot_id: int,
    quality_status: str,
    anomaly_summary: Dict
) -> bool:
    """
    Tag Iceberg snapshot with quality metadata.
    
    Args:
        table_path: Iceberg table path
        snapshot_id: Snapshot ID to tag
        quality_status: 'passed', 'quarantined', or 'failed'
        anomaly_summary: Summary of anomalies
        
    Returns:
        True if tagging successful
    """
    if not ICEBERG_AVAILABLE:
        print("pyiceberg not available, skipping snapshot tagging")
        return False
    
    try:
        catalog = load_catalog("default")
        table = catalog.load_table(table_path)
        
        # Update snapshot summary with quality tags
        table.update_snapshot_summary(
            snapshot_id=snapshot_id,
            updates={
                "quality.status": quality_status,
                "quality.anomaly_count": str(anomaly_summary.get('anomaly_count', 0)),
                "quality.timestamp": datetime.utcnow().isoformat(),
                "quality.summary": str(anomaly_summary)
            }
        )
        
        return True
    except Exception as e:
        print(f"Error tagging snapshot: {e}")
        return False
