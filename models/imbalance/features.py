"""
Feature Engineering for DA↔RT Imbalance Cost Model

Prepares features for quantile regression model to predict imbalance costs
and hedge schedule risk.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ImbalanceFeatureEngineer:
    """
    Feature engineering for DA↔RT imbalance cost prediction.
    
    Extracts features that capture schedule risk including:
    - DA/RT price spreads and volatility
    - Load forecast errors
    - Wind/solar forecast errors
    - Ramping requirements
    - Historical imbalance patterns
    """
    
    def __init__(self, lookback_hours: int = 168):
        """
        Initialize feature engineer.
        
        Args:
            lookback_hours: Hours of history for feature calculation (default 1 week)
        """
        self.lookback_hours = lookback_hours
    
    def extract_price_spread_features(
        self,
        da_lmp: pd.DataFrame,
        rt_lmp: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract DA-RT price spread features.
        
        Args:
            da_lmp: Day-ahead LMP data with columns: timestamp, node_id, lmp
            rt_lmp: Real-time LMP data with columns: timestamp, node_id, lmp
            
        Returns:
            spread_features: DataFrame with spread-based features
        """
        # Merge DA and RT prices
        df = da_lmp.merge(
            rt_lmp,
            on=['timestamp', 'node_id'],
            suffixes=('_da', '_rt'),
            how='inner'
        )
        
        # Calculate spread
        df['spread'] = df['lmp_rt'] - df['lmp_da']
        df['spread_abs'] = np.abs(df['spread'])
        df['spread_pct'] = (df['spread'] / df['lmp_da'].abs()).replace([np.inf, -np.inf], 0)
        
        # Rolling statistics
        for window in [24, 72, 168]:  # 1 day, 3 days, 1 week
            df[f'spread_mean_{window}h'] = df.groupby('node_id')['spread'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'spread_std_{window}h'] = df.groupby('node_id')['spread'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'spread_abs_mean_{window}h'] = df.groupby('node_id')['spread_abs'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Price volatility
        df['da_volatility_24h'] = df.groupby('node_id')['lmp_da'].transform(
            lambda x: x.rolling(24, min_periods=1).std()
        )
        df['rt_volatility_24h'] = df.groupby('node_id')['lmp_rt'].transform(
            lambda x: x.rolling(24, min_periods=1).std()
        )
        
        return df
    
    def extract_forecast_error_features(
        self,
        load_forecast: pd.DataFrame,
        load_actual: pd.DataFrame,
        renewable_forecast: Optional[pd.DataFrame] = None,
        renewable_actual: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract forecast error features.
        
        Args:
            load_forecast: Load forecast data
            load_actual: Actual load data
            renewable_forecast: Renewable forecast (optional)
            renewable_actual: Actual renewable generation (optional)
            
        Returns:
            forecast_features: DataFrame with forecast error features
        """
        # Load forecast error
        df = load_forecast.merge(
            load_actual,
            on=['timestamp', 'zone_id'],
            suffixes=('_fcst', '_actual'),
            how='inner'
        )
        
        df['load_error'] = df['load_actual'] - df['load_fcst']
        df['load_error_pct'] = (df['load_error'] / df['load_actual']).replace([np.inf, -np.inf], 0)
        df['load_error_abs'] = np.abs(df['load_error'])
        
        # Rolling forecast error statistics
        for window in [24, 72]:
            df[f'load_error_mean_{window}h'] = df.groupby('zone_id')['load_error'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'load_error_std_{window}h'] = df.groupby('zone_id')['load_error'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Renewable forecast error if available
        if renewable_forecast is not None and renewable_actual is not None:
            ren_df = renewable_forecast.merge(
                renewable_actual,
                on=['timestamp', 'resource_id'],
                suffixes=('_fcst', '_actual'),
                how='inner'
            )
            
            ren_df['renewable_error'] = ren_df['generation_actual'] - ren_df['generation_fcst']
            ren_df['renewable_error_pct'] = (
                ren_df['renewable_error'] / ren_df['generation_actual']
            ).replace([np.inf, -np.inf], 0)
            
            # Aggregate by zone
            ren_agg = ren_df.groupby(['timestamp', 'zone_id']).agg({
                'renewable_error': ['mean', 'std', 'sum'],
                'renewable_error_pct': ['mean', 'std']
            }).reset_index()
            
            ren_agg.columns = ['timestamp', 'zone_id'] + [
                f'renewable_{stat}_{metric}' 
                for metric, stat in ren_agg.columns[2:]
            ]
            
            df = df.merge(ren_agg, on=['timestamp', 'zone_id'], how='left')
        
        return df
    
    def extract_temporal_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract temporal features (hour of day, day of week, etc.).
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            df: DataFrame with additional temporal features
        """
        df = df.copy()
        
        # Hour of day (cyclical encoding)
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (cyclical encoding)
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Month (cyclical encoding)
        df['month'] = df['timestamp'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Is weekend
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Is peak hour (7-23)
        df['is_peak'] = ((df['hour'] >= 7) & (df['hour'] < 23)).astype(int)
        
        return df
    
    def extract_ramping_features(
        self,
        load_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract ramping requirement features.
        
        Args:
            load_data: Load data with timestamp and load columns
            
        Returns:
            ramping_features: DataFrame with ramping features
        """
        df = load_data.copy()
        
        # Calculate load ramps
        df['load_ramp_1h'] = df.groupby('zone_id')['load'].diff(1)
        df['load_ramp_3h'] = df.groupby('zone_id')['load'].diff(3)
        
        # Absolute ramps
        df['load_ramp_1h_abs'] = np.abs(df['load_ramp_1h'])
        df['load_ramp_3h_abs'] = np.abs(df['load_ramp_3h'])
        
        # Rolling max/min ramps
        for window in [24, 72]:
            df[f'max_ramp_{window}h'] = df.groupby('zone_id')['load_ramp_1h_abs'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            df[f'ramp_volatility_{window}h'] = df.groupby('zone_id')['load_ramp_1h'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        return df
    
    def compute_all_features(
        self,
        da_lmp: pd.DataFrame,
        rt_lmp: pd.DataFrame,
        load_forecast: pd.DataFrame,
        load_actual: pd.DataFrame,
        renewable_forecast: Optional[pd.DataFrame] = None,
        renewable_actual: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute all imbalance features.
        
        Args:
            da_lmp: Day-ahead LMP data
            rt_lmp: Real-time LMP data
            load_forecast: Load forecast
            load_actual: Actual load
            renewable_forecast: Renewable forecast (optional)
            renewable_actual: Actual renewable (optional)
            
        Returns:
            features_df: Complete feature DataFrame
        """
        logger.info("Computing imbalance cost features...")
        
        # Price spread features
        spread_features = self.extract_price_spread_features(da_lmp, rt_lmp)
        
        # Forecast error features
        forecast_features = self.extract_forecast_error_features(
            load_forecast,
            load_actual,
            renewable_forecast,
            renewable_actual
        )
        
        # Ramping features
        ramping_features = self.extract_ramping_features(load_actual)
        
        # Merge features
        features_df = spread_features.merge(
            forecast_features,
            on=['timestamp', 'zone_id'],
            how='inner'
        )
        
        features_df = features_df.merge(
            ramping_features,
            on=['timestamp', 'zone_id'],
            how='inner'
        )
        
        # Temporal features
        features_df = self.extract_temporal_features(features_df)
        
        # Drop NaN
        features_df = features_df.dropna()
        
        logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} samples")
        
        return features_df


def main():
    """Main feature engineering pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract imbalance cost features')
    parser.add_argument('--da-lmp', required=True, help='Day-ahead LMP CSV')
    parser.add_argument('--rt-lmp', required=True, help='Real-time LMP CSV')
    parser.add_argument('--load-forecast', required=True, help='Load forecast CSV')
    parser.add_argument('--load-actual', required=True, help='Actual load CSV')
    parser.add_argument('--output', required=True, help='Output features CSV')
    
    args = parser.parse_args()
    
    # Load data
    da_lmp = pd.read_csv(args.da_lmp, parse_dates=['timestamp'])
    rt_lmp = pd.read_csv(args.rt_lmp, parse_dates=['timestamp'])
    load_forecast = pd.read_csv(args.load_forecast, parse_dates=['timestamp'])
    load_actual = pd.read_csv(args.load_actual, parse_dates=['timestamp'])
    
    # Initialize feature engineer
    engineer = ImbalanceFeatureEngineer()
    
    # Compute features
    features = engineer.compute_all_features(
        da_lmp,
        rt_lmp,
        load_forecast,
        load_actual
    )
    
    # Save
    features.to_csv(args.output, index=False)
    print(f"Features saved to {args.output}")
    print(f"Shape: {features.shape}")


if __name__ == '__main__':
    main()
