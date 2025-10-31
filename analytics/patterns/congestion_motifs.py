"""
Topology Motif Miner

Discovers recurrent congestion patterns using SAX/Matrix Profile + HDBSCAN.
DoD: Motif alerts capture ≥60% of high-congestion intervals with >2:1 precision:recall.
"""

import logging
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class CongestionMotifMiner:
    """
    Mine recurrent congestion patterns from LMP spreads and flow data.
    
    Uses time series motif discovery to identify repeating congestion patterns.
    """
    
    def __init__(self, window_size: int = 24, min_cluster_size: int = 3):
        self.window_size = window_size
        self.min_cluster_size = min_cluster_size
        self.motifs = []
    
    def extract_windows(
        self,
        lmp_spreads: pd.DataFrame,
        flow_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract sliding windows from time series data.
        
        Args:
            lmp_spreads: LMP spread time series
            flow_data: Flow data time series
            
        Returns:
            windows: Array of shape (n_windows, window_size * n_features)
        """
        # Combine features
        df = lmp_spreads.merge(flow_data, on='timestamp', how='inner')
        df = df.sort_values('timestamp')
        
        # Select numeric columns
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'iso']]
        features = df[feature_cols].values
        
        # Create sliding windows
        windows = []
        for i in range(len(features) - self.window_size + 1):
            window = features[i:i+self.window_size].flatten()
            windows.append(window)
        
        return np.array(windows)
    
    def discover_motifs(
        self,
        lmp_spreads: pd.DataFrame,
        flow_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Discover congestion motifs using clustering.
        
        Args:
            lmp_spreads: LMP spread time series
            flow_data: Flow data time series
            
        Returns:
            motifs: List of discovered motifs
        """
        logger.info("Discovering congestion motifs...")
        
        # Extract windows
        windows = self.extract_windows(lmp_spreads, flow_data)
        
        # Normalize windows
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        windows_scaled = scaler.fit_transform(windows)
        
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(windows_scaled)
        
        # Extract motifs (cluster centroids)
        motifs = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_windows = windows_scaled[labels == cluster_id]
            centroid = cluster_windows.mean(axis=0)
            
            motif = {
                'motif_id': f'motif_{cluster_id}',
                'cluster_id': cluster_id,
                'occurrence_count': len(cluster_windows),
                'avg_duration_hours': self.window_size,
                'signature': centroid.tolist()
            }
            motifs.append(motif)
        
        self.motifs = motifs
        logger.info(f"Discovered {len(motifs)} congestion motifs")
        
        return motifs
    
    def detect_motif(
        self,
        current_window: np.ndarray,
        threshold: float = 2.0
    ) -> Dict:
        """
        Detect if current window matches a known motif.
        
        Args:
            current_window: Current time series window
            threshold: Distance threshold for matching
            
        Returns:
            detection: Dict with motif_id and confidence
        """
        if not self.motifs:
            return None
        
        # Normalize window
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        window_scaled = scaler.fit_transform(current_window.reshape(1, -1))[0]
        
        # Find closest motif
        best_match = None
        best_distance = float('inf')
        
        for motif in self.motifs:
            distance = euclidean(window_scaled, motif['signature'])
            if distance < best_distance and distance < threshold:
                best_distance = distance
                best_match = motif
        
        if best_match:
            confidence = max(0, 1 - best_distance / threshold)
            return {
                'motif_id': best_match['motif_id'],
                'confidence_score': confidence,
                'expected_duration_hours': best_match['avg_duration_hours']
            }
        
        return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmp-spreads', required=True)
    parser.add_argument('--flow-data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    lmp_spreads = pd.read_csv(args.lmp_spreads, parse_dates=['timestamp'])
    flow_data = pd.read_csv(args.flow_data, parse_dates=['timestamp'])
    
    miner = CongestionMotifMiner()
    motifs = miner.discover_motifs(lmp_spreads, flow_data)
    
    pd.DataFrame(motifs).to_csv(args.output, index=False)
    print(f"Discovered {len(motifs)} motifs, saved to {args.output}")
