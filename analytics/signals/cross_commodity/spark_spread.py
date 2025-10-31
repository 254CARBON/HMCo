"""
Cross-commodity signal engine: Gas-Power spark spreads
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SparkSpreadCalculator:
    """
    Calculate spark spreads and cross-commodity signals
    Target: >300bps IRR improvement in hedging
    """
    
    def __init__(self, clickhouse_client=None):
        self.clickhouse_client = clickhouse_client
        
    def calculate_spark_spread(
        self,
        power_lmp: float,
        gas_price: float,
        heat_rate: float = 7.0,  # MMBtu/MWh
        variable_om: float = 2.0  # $/MWh
    ) -> float:
        """
        Calculate spark spread
        
        Spark Spread = Power LMP - (Gas Price * Heat Rate + VOM)
        
        Args:
            power_lmp: Power price ($/MWh)
            gas_price: Gas price ($/MMBtu)
            heat_rate: Heat rate (MMBtu/MWh)
            variable_om: Variable O&M ($/MWh)
            
        Returns:
            Spark spread ($/MWh)
        """
        fuel_cost = gas_price * heat_rate
        spark_spread = power_lmp - fuel_cost - variable_om
        
        return spark_spread
    
    def calculate_implied_heat_rate(
        self,
        power_lmp: float,
        gas_price: float,
        variable_om: float = 2.0
    ) -> float:
        """
        Calculate implied heat rate from power and gas prices
        
        Args:
            power_lmp: Power price ($/MWh)
            gas_price: Gas price ($/MMBtu)
            variable_om: Variable O&M ($/MWh)
            
        Returns:
            Implied heat rate (MMBtu/MWh)
        """
        if gas_price <= 0:
            return 0.0
        
        implied_hr = (power_lmp - variable_om) / gas_price
        return implied_hr
    
    def calculate_carbon_adjusted_spread(
        self,
        spark_spread: float,
        carbon_price: float,
        carbon_intensity: float = 0.4  # tons CO2/MWh
    ) -> float:
        """
        Adjust spark spread for carbon costs
        
        Args:
            spark_spread: Base spark spread ($/MWh)
            carbon_price: Carbon price ($/ton)
            carbon_intensity: Emissions intensity (tons CO2/MWh)
            
        Returns:
            Carbon-adjusted spread ($/MWh)
        """
        carbon_cost = carbon_price * carbon_intensity
        adjusted_spread = spark_spread - carbon_cost
        
        return adjusted_spread
    
    def calculate_lng_netback(
        self,
        destination_price: float,
        shipping_cost: float,
        liquefaction_cost: float = 3.0,  # $/MMBtu
        regasification_cost: float = 0.5,  # $/MMBtu
        fx_rate: float = 1.0
    ) -> float:
        """
        Calculate LNG netback price
        
        Args:
            destination_price: Price at destination ($/MMBtu)
            shipping_cost: Shipping cost ($/MMBtu)
            liquefaction_cost: Liquefaction cost ($/MMBtu)
            regasification_cost: Regasification cost ($/MMBtu)
            fx_rate: FX rate adjustment
            
        Returns:
            Netback price ($/MMBtu)
        """
        netback = (destination_price - shipping_cost - 
                   liquefaction_cost - regasification_cost) * fx_rate
        
        return netback
    
    def compute_cross_asset_features(
        self,
        power_data: pd.DataFrame,
        gas_data: pd.DataFrame,
        carbon_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute full suite of cross-commodity features
        
        Args:
            power_data: Power LMP and load data
            gas_data: Gas prices and flows
            carbon_data: Optional carbon prices
            
        Returns:
            DataFrame with cross-asset features
        """
        logger.info("Computing cross-commodity features")
        
        # Merge datasets on timestamp
        df = power_data.merge(gas_data, on='timestamp', how='inner')
        
        if carbon_data is not None:
            df = df.merge(carbon_data, on='timestamp', how='left')
        
        # Calculate spark spreads
        df['spark_spread'] = df.apply(
            lambda row: self.calculate_spark_spread(
                row['power_lmp'],
                row['gas_price']
            ),
            axis=1
        )
        
        # Calculate implied heat rates
        df['implied_heat_rate'] = df.apply(
            lambda row: self.calculate_implied_heat_rate(
                row['power_lmp'],
                row['gas_price']
            ),
            axis=1
        )
        
        # Carbon-adjusted spreads
        if 'carbon_price' in df.columns:
            df['carbon_adjusted_spread'] = df.apply(
                lambda row: self.calculate_carbon_adjusted_spread(
                    row['spark_spread'],
                    row.get('carbon_price', 0)
                ),
                axis=1
            )
        
        # Add rolling statistics
        df['spark_spread_ma7'] = df['spark_spread'].rolling(7).mean()
        df['spark_spread_std7'] = df['spark_spread'].rolling(7).std()
        df['spark_spread_zscore'] = (df['spark_spread'] - df['spark_spread_ma7']) / df['spark_spread_std7']
        
        logger.info(f"Computed features for {len(df)} records")
        return df
    
    def identify_arbitrage_opportunities(
        self,
        features_df: pd.DataFrame,
        threshold_zscore: float = 2.0
    ) -> pd.DataFrame:
        """
        Identify potential arbitrage opportunities
        
        Args:
            features_df: DataFrame with cross-asset features
            threshold_zscore: Z-score threshold for opportunity
            
        Returns:
            DataFrame with opportunities
        """
        opportunities = features_df[
            abs(features_df['spark_spread_zscore']) > threshold_zscore
        ].copy()
        
        opportunities['opportunity_type'] = opportunities['spark_spread_zscore'].apply(
            lambda x: 'long_power_short_gas' if x > 0 else 'short_power_long_gas'
        )
        
        opportunities['potential_value'] = abs(opportunities['spark_spread_zscore']) * \
                                          opportunities['spark_spread_std7']
        
        logger.info(f"Identified {len(opportunities)} arbitrage opportunities")
        return opportunities
