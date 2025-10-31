"""
Data preparation module for LMP nowcasting
Handles feature engineering, graph construction, and data loading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LMPDataPreparation:
    """
    Prepares data for LMP nowcasting model including:
    - Graph structure (nodes, edges, PTDF matrices)
    - Weather features (H3-joined NOAA data)
    - Historical LMP timeseries
    - Market features (load, generation, congestion)
    """
    
    def __init__(
        self,
        clickhouse_client=None,
        lookback_hours: int = 168,
        forecast_horizon_minutes: int = 60
    ):
        self.clickhouse_client = clickhouse_client
        self.lookback_hours = lookback_hours
        self.forecast_horizon_minutes = forecast_horizon_minutes
        
    def load_graph_topology(
        self,
        iso: str,
        include_ptdf: bool = True
    ) -> Dict:
        """
        Load network topology and PTDF/shift factors
        
        Args:
            iso: ISO name (CAISO, MISO, SPP, etc.)
            include_ptdf: Whether to include PTDF matrices
            
        Returns:
            Dict with nodes, edges, and PTDF data
        """
        logger.info(f"Loading graph topology for {iso}")
        
        # Query node mappings
        nodes_query = f"""
        SELECT 
            node_id,
            node_name,
            hub_id,
            zone_id,
            latitude,
            longitude,
            node_type
        FROM iso_node_mapping
        WHERE iso = '{iso}'
        ORDER BY node_id
        """
        
        # Placeholder for graph structure
        graph_data = {
            'iso': iso,
            'nodes': [],
            'edges': [],
            'ptdf_matrix': None
        }
        
        if self.clickhouse_client:
            # In production, query from ClickHouse
            try:
                nodes_df = self.clickhouse_client.query_dataframe(nodes_query)
                graph_data['nodes'] = nodes_df.to_dict('records')
                
                if include_ptdf:
                    # Load or compute PTDF/shift factors
                    # This would come from power flow analysis
                    graph_data['ptdf_matrix'] = self._compute_ptdf_estimates(nodes_df)
                    
            except Exception as e:
                logger.error(f"Error loading graph topology: {e}")
                # Return mock data for development
                graph_data = self._get_mock_graph_data(iso)
        else:
            # Mock data for testing
            graph_data = self._get_mock_graph_data(iso)
            
        logger.info(f"Loaded {len(graph_data['nodes'])} nodes for {iso}")
        return graph_data
    
    def load_weather_features(
        self,
        start_time: datetime,
        end_time: datetime,
        iso: str
    ) -> pd.DataFrame:
        """
        Load H3-joined NOAA weather data
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            iso: ISO region
            
        Returns:
            DataFrame with weather features per node/h3 cell
        """
        logger.info(f"Loading weather features from {start_time} to {end_time}")
        
        weather_query = f"""
        SELECT 
            timestamp,
            h3_index,
            temperature,
            wind_speed,
            wind_direction,
            solar_irradiance,
            precipitation,
            humidity
        FROM noaa_weather
        WHERE timestamp >= '{start_time}'
          AND timestamp < '{end_time}'
          AND iso_region = '{iso}'
        ORDER BY timestamp, h3_index
        """
        
        if self.clickhouse_client:
            try:
                weather_df = self.clickhouse_client.query_dataframe(weather_query)
                return weather_df
            except Exception as e:
                logger.error(f"Error loading weather data: {e}")
                return self._get_mock_weather_data(start_time, end_time)
        else:
            return self._get_mock_weather_data(start_time, end_time)
    
    def load_lmp_history(
        self,
        start_time: datetime,
        end_time: datetime,
        iso: str,
        nodes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load historical LMP data for training/validation
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            iso: ISO name
            nodes: Optional list of node IDs to filter
            
        Returns:
            DataFrame with LMP history
        """
        logger.info(f"Loading LMP history from {start_time} to {end_time}")
        
        node_filter = ""
        if nodes:
            node_list = "', '".join(nodes)
            node_filter = f"AND node_id IN ('{node_list}')"
        
        lmp_query = f"""
        SELECT 
            timestamp,
            node_id,
            lmp_value,
            energy_component,
            congestion_component,
            loss_component
        FROM iso_rt_lmp_canonical
        WHERE timestamp >= '{start_time}'
          AND timestamp < '{end_time}'
          AND iso = '{iso}'
          {node_filter}
        ORDER BY timestamp, node_id
        """
        
        if self.clickhouse_client:
            try:
                lmp_df = self.clickhouse_client.query_dataframe(lmp_query)
                return lmp_df
            except Exception as e:
                logger.error(f"Error loading LMP data: {e}")
                return self._get_mock_lmp_data(start_time, end_time)
        else:
            return self._get_mock_lmp_data(start_time, end_time)
    
    def prepare_training_dataset(
        self,
        iso: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Prepare complete training dataset with all features
        
        Args:
            iso: ISO name
            start_date: Training start date
            end_date: Training end date
            
        Returns:
            Dict containing graph, features, and targets
        """
        logger.info(f"Preparing training dataset for {iso}")
        
        # Load graph structure
        graph_data = self.load_graph_topology(iso, include_ptdf=True)
        
        # Load weather features
        weather_df = self.load_weather_features(start_date, end_date, iso)
        
        # Load LMP history
        lmp_df = self.load_lmp_history(start_date, end_date, iso)
        
        # Create temporal features
        temporal_features = self._create_temporal_features(lmp_df)
        
        # Combine all features
        dataset = {
            'graph': graph_data,
            'weather': weather_df,
            'lmp_history': lmp_df,
            'temporal_features': temporal_features,
            'metadata': {
                'iso': iso,
                'start_date': start_date,
                'end_date': end_date,
                'num_nodes': len(graph_data['nodes']),
                'num_samples': len(lmp_df)
            }
        }
        
        logger.info(f"Dataset prepared with {dataset['metadata']['num_samples']} samples")
        return dataset
    
    def _compute_ptdf_estimates(self, nodes_df: pd.DataFrame) -> np.ndarray:
        """Estimate PTDF/shift factors from topology"""
        n_nodes = len(nodes_df)
        # Simplified PTDF estimation - in production would use DC power flow
        ptdf = np.random.randn(n_nodes, n_nodes) * 0.1
        np.fill_diagonal(ptdf, 1.0)
        return ptdf
    
    def _create_temporal_features(self, lmp_df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features (hour, day of week, etc.)"""
        if 'timestamp' not in lmp_df.columns:
            return pd.DataFrame()
        
        temporal = pd.DataFrame()
        temporal['timestamp'] = lmp_df['timestamp']
        temporal['hour'] = pd.to_datetime(temporal['timestamp']).dt.hour
        temporal['day_of_week'] = pd.to_datetime(temporal['timestamp']).dt.dayofweek
        temporal['month'] = pd.to_datetime(temporal['timestamp']).dt.month
        temporal['is_weekend'] = temporal['day_of_week'].isin([5, 6]).astype(int)
        
        return temporal
    
    def _get_mock_graph_data(self, iso: str) -> Dict:
        """Generate mock graph data for testing"""
        n_nodes = 100
        nodes = [
            {
                'node_id': f'{iso}_NODE_{i:04d}',
                'node_name': f'Node {i}',
                'hub_id': f'HUB_{i % 10}',
                'zone_id': f'ZONE_{i % 5}',
                'latitude': 35.0 + np.random.randn() * 2,
                'longitude': -95.0 + np.random.randn() * 5,
                'node_type': np.random.choice(['gen', 'load', 'hub'])
            }
            for i in range(n_nodes)
        ]
        
        return {
            'iso': iso,
            'nodes': nodes,
            'edges': [],
            'ptdf_matrix': np.eye(n_nodes)
        }
    
    def _get_mock_weather_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Generate mock weather data for testing"""
        timestamps = pd.date_range(start_time, end_time, freq='5min')
        
        data = []
        for ts in timestamps[:100]:  # Limit for testing
            data.append({
                'timestamp': ts,
                'h3_index': f'h3_{np.random.randint(1000)}',
                'temperature': 20 + np.random.randn() * 5,
                'wind_speed': max(0, 10 + np.random.randn() * 3),
                'wind_direction': np.random.uniform(0, 360),
                'solar_irradiance': max(0, 500 + np.random.randn() * 200),
                'precipitation': max(0, np.random.exponential(0.1)),
                'humidity': np.random.uniform(30, 90)
            })
        
        return pd.DataFrame(data)
    
    def _get_mock_lmp_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Generate mock LMP data for testing"""
        timestamps = pd.date_range(start_time, end_time, freq='5min')
        
        data = []
        for ts in timestamps[:100]:  # Limit for testing
            for node_idx in range(10):
                base_lmp = 30 + np.random.randn() * 10
                data.append({
                    'timestamp': ts,
                    'node_id': f'NODE_{node_idx:04d}',
                    'lmp_value': base_lmp,
                    'energy_component': base_lmp * 0.7,
                    'congestion_component': base_lmp * 0.2,
                    'loss_component': base_lmp * 0.1
                })
        
        return pd.DataFrame(data)
