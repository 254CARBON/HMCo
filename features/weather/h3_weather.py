"""
H3-based weather feature joining for grid nodes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class H3WeatherJoiner:
    """
    Joins weather data to grid nodes using H3 spatial indexing
    """
    
    def __init__(self, h3_resolution: int = 7):
        """
        Args:
            h3_resolution: H3 resolution (7 = ~5km hexagons)
        """
        self.h3_resolution = h3_resolution
        
    def join_weather_to_nodes(
        self,
        nodes: List[Dict],
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join weather features to grid nodes via H3 indexing
        
        Args:
            nodes: List of node dicts with lat/lon
            weather_df: Weather data with h3_index
            
        Returns:
            DataFrame with node_id and weather features
        """
        logger.info(f"Joining weather to {len(nodes)} nodes")
        
        # Convert nodes to DataFrame
        nodes_df = pd.DataFrame(nodes)
        
        # Compute H3 index for each node
        nodes_df['h3_index'] = nodes_df.apply(
            lambda row: self._lat_lon_to_h3(
                row['latitude'],
                row['longitude']
            ),
            axis=1
        )
        
        # Join with weather data
        result = nodes_df.merge(
            weather_df,
            on='h3_index',
            how='left'
        )
        
        logger.info(f"Weather joined to nodes: {len(result)} records")
        return result
    
    def _lat_lon_to_h3(self, lat: float, lon: float) -> str:
        """
        Convert lat/lon to H3 index
        Simplified mock - in production use h3-py library
        """
        try:
            # In production: import h3; return h3.geo_to_h3(lat, lon, self.h3_resolution)
            # Mock implementation
            h3_index = f"h3_{self.h3_resolution}_{int(lat*100)}_{int(lon*100)}"
            return h3_index
        except Exception as e:
            logger.warning(f"H3 conversion error: {e}")
            return "h3_unknown"
    
    def aggregate_weather_spatial(
        self,
        weather_df: pd.DataFrame,
        window_km: float = 50.0
    ) -> pd.DataFrame:
        """
        Spatially aggregate weather within a window
        
        Args:
            weather_df: Weather data with h3_index
            window_km: Spatial window in km
            
        Returns:
            Aggregated weather data
        """
        # Group by h3_index and timestamp, aggregate
        agg_cols = {
            'temperature': 'mean',
            'wind_speed': 'mean',
            'solar_irradiance': 'mean',
            'precipitation': 'sum'
        }
        
        result = weather_df.groupby(['h3_index', 'timestamp']).agg(agg_cols).reset_index()
        
        return result
