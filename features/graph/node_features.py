"""Node feature extraction for graph models"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class NodeFeatureExtractor:
    """Extract and engineer node-level features"""
    
    def extract_features(self, topology: Dict, temporal_data: pd.DataFrame) -> np.ndarray:
        """Extract node features from topology and temporal data"""
        num_nodes = topology['num_nodes']
        
        # Static features: node type, voltage, coordinates
        static_features = self._extract_static_features(topology['nodes'])
        
        # Dynamic features: load, generation, prices
        dynamic_features = self._extract_dynamic_features(temporal_data, num_nodes)
        
        # Combine
        features = np.concatenate([static_features, dynamic_features], axis=-1)
        
        logger.info(f"Extracted features shape: {features.shape}")
        return features
    
    def _extract_static_features(self, nodes: List[Dict]) -> np.ndarray:
        """Extract static node features"""
        features = []
        for node in nodes:
            feat = [
                node.get('latitude', 0.0),
                node.get('longitude', 0.0),
                node.get('voltage_level', 0) / 500.0,  # Normalize
                1.0 if node.get('node_type') == 'gen' else 0.0,
                1.0 if node.get('node_type') == 'load' else 0.0,
            ]
            features.append(feat)
        return np.array(features)
    
    def _extract_dynamic_features(self, data: pd.DataFrame, num_nodes: int) -> np.ndarray:
        """Extract dynamic features from temporal data"""
        # Placeholder: would aggregate recent LMP, load, generation
        return np.random.randn(num_nodes, 10)  # 10 dynamic features
