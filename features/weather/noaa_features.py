"""NOAA weather feature extraction"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NOAAFeatureExtractor:
    """Extract and engineer features from NOAA weather data"""
    
    def __init__(self, clickhouse_client=None):
        self.clickhouse_client = clickhouse_client
        
    def load_weather(
        self,
        start_time: datetime,
        end_time: datetime,
        iso_region: str
    ) -> pd.DataFrame:
        """Load NOAA weather data from ClickHouse"""
        query = f"""
        SELECT 
            timestamp,
            h3_index,
            temperature,
            wind_speed,
            wind_direction,
            solar_irradiance,
            precipitation,
            humidity,
            pressure
        FROM noaa_weather
        WHERE timestamp >= '{start_time}'
          AND timestamp < '{end_time}'
          AND iso_region = '{iso_region}'
        ORDER BY timestamp, h3_index
        """
        
        if self.clickhouse_client:
            try:
                return self.clickhouse_client.query_dataframe(query)
            except Exception as e:
                logger.error(f"Error loading weather: {e}")
        
        return self._generate_mock_weather(start_time, end_time)
    
    def engineer_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer weather features"""
        df = weather_df.copy()
        
        # Derived features
        df['wind_power_potential'] = 0.5 * 1.225 * (df['wind_speed'] ** 3)  # Air density * v^3
        df['solar_capacity_factor'] = df['solar_irradiance'] / 1000.0  # Normalize to 1kW/m²
        df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
        
        # Temporal features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['is_peak'] = df['hour'].isin(range(14, 21)).astype(int)
        
        return df
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity"""
        # Simplified heat index formula
        return temp + 0.5 * (humidity / 100.0) * (temp - 14.0)
    
    def _generate_mock_weather(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Generate mock weather data"""
        timestamps = pd.date_range(start_time, end_time, freq='1H')[:100]
        data = []
        
        for ts in timestamps:
            data.append({
                'timestamp': ts,
                'h3_index': f'h3_mock_{np.random.randint(100)}',
                'temperature': 20 + 10 * np.sin(ts.hour * np.pi / 12) + np.random.randn() * 2,
                'wind_speed': max(0, 8 + np.random.randn() * 3),
                'wind_direction': np.random.uniform(0, 360),
                'solar_irradiance': max(0, 500 * np.sin(ts.hour * np.pi / 12)) if 6 <= ts.hour <= 18 else 0,
                'precipitation': max(0, np.random.exponential(0.1)),
                'humidity': np.random.uniform(40, 80),
                'pressure': 1013 + np.random.randn() * 5
            })
        
        return pd.DataFrame(data)
