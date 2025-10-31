"""
Topology Signals Feature Engineering

Extracts co-movement patterns and flow proxy signals from network topology
for use in PTDF/LODF estimation.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TopologySignalExtractor:
    """
    Extracts topology-based signals for grid sensitivity models.
    
    Computes co-movement indicators, flow proxies, and network-based features
    to enhance PTDF/LODF predictions.
    """
    
    def __init__(
        self,
        lookback_window: int = 24,
        n_components: int = 5
    ):
        """
        Initialize topology signal extractor.
        
        Args:
            lookback_window: Hours of history for co-movement calculation
            n_components: Number of PCA components for dimensionality reduction
        """
        self.lookback_window = lookback_window
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        
    def build_network_graph(
        self,
        nodes: pd.DataFrame,
        lines: pd.DataFrame
    ) -> nx.Graph:
        """
        Build network graph from node and line data.
        
        Args:
            nodes: DataFrame with node_id, lat, lon, voltage_level
            lines: DataFrame with line_id, from_node, to_node, capacity
            
        Returns:
            G: NetworkX graph representation
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for _, node in nodes.iterrows():
            G.add_node(
                node['node_id'],
                voltage_level=node.get('voltage_level', 345),
                lat=node.get('lat', 0),
                lon=node.get('lon', 0)
            )
        
        # Add edges with attributes
        for _, line in lines.iterrows():
            G.add_edge(
                line['from_node'],
                line['to_node'],
                line_id=line['line_id'],
                capacity=line.get('capacity', 1000),
                length=line.get('length', 1.0)
            )
        
        logger.info(f"Built network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def compute_centrality_metrics(
        self,
        G: nx.Graph
    ) -> pd.DataFrame:
        """
        Compute network centrality metrics for each node.
        
        Args:
            G: NetworkX graph
            
        Returns:
            centrality_df: DataFrame with centrality metrics
        """
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        
        # Betweenness centrality (flow through node)
        betweenness_cent = nx.betweenness_centrality(G, weight='capacity')
        
        # Closeness centrality
        closeness_cent = nx.closeness_centrality(G, distance='length')
        
        # Eigenvector centrality
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector_cent = {node: 0.0 for node in G.nodes()}
        
        # PageRank
        pagerank = nx.pagerank(G, weight='capacity')
        
        # Combine into DataFrame
        centrality_df = pd.DataFrame({
            'node_id': list(G.nodes()),
            'topology_degree_centrality': [degree_cent[n] for n in G.nodes()],
            'topology_betweenness_centrality': [betweenness_cent[n] for n in G.nodes()],
            'topology_closeness_centrality': [closeness_cent[n] for n in G.nodes()],
            'topology_eigenvector_centrality': [eigenvector_cent[n] for n in G.nodes()],
            'topology_pagerank': [pagerank[n] for n in G.nodes()]
        })
        
        return centrality_df
    
    def compute_flow_comovement(
        self,
        flow_data: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute co-movement correlations between line flows.
        
        Args:
            flow_data: DataFrame with timestamp and flow columns
            window: Rolling window size (default: self.lookback_window)
            
        Returns:
            comovement_df: DataFrame with co-movement signals
        """
        if window is None:
            window = self.lookback_window
        
        # Get flow columns
        flow_cols = [col for col in flow_data.columns if col.startswith('flow_')]
        
        # Calculate rolling correlations
        comovement_features = []
        
        for i, col1 in enumerate(flow_cols):
            for col2 in flow_cols[i+1:]:
                # Rolling correlation
                corr_col = f'topology_corr_{col1}_{col2}'
                flow_data[corr_col] = (
                    flow_data[col1].rolling(window).corr(flow_data[col2].rolling(window))
                )
                comovement_features.append(corr_col)
        
        # Extract timestamp and co-movement features
        comovement_df = flow_data[['timestamp'] + comovement_features].copy()
        
        # Fill NaN with 0
        comovement_df = comovement_df.fillna(0)
        
        return comovement_df
    
    def compute_flow_proxies(
        self,
        flow_data: pd.DataFrame,
        lmp_data: pd.DataFrame,
        G: nx.Graph
    ) -> pd.DataFrame:
        """
        Compute flow proxy signals based on price differences.
        
        Args:
            flow_data: DataFrame with flow data
            lmp_data: DataFrame with LMP data
            G: Network graph
            
        Returns:
            proxy_df: DataFrame with flow proxy signals
        """
        # Merge data
        df = flow_data.merge(lmp_data, on='timestamp', how='inner')
        
        # Get LMP columns
        lmp_cols = [col for col in df.columns if col.startswith('lmp_')]
        
        # Compute price spreads for connected nodes
        proxy_features = []
        
        for edge in G.edges():
            from_node, to_node = edge
            from_lmp = f'lmp_{from_node}'
            to_lmp = f'lmp_{to_node}'
            
            if from_lmp in lmp_cols and to_lmp in lmp_cols:
                # Price spread as flow proxy
                spread_col = f'topology_spread_{from_node}_{to_node}'
                df[spread_col] = df[from_lmp] - df[to_lmp]
                proxy_features.append(spread_col)
                
                # Absolute spread
                abs_spread_col = f'topology_abs_spread_{from_node}_{to_node}'
                df[abs_spread_col] = np.abs(df[spread_col])
                proxy_features.append(abs_spread_col)
        
        # Extract proxy features
        proxy_df = df[['timestamp'] + proxy_features].copy()
        
        return proxy_df
    
    def compute_pca_features(
        self,
        flow_data: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Compute PCA-based features from flow data.
        
        Args:
            flow_data: DataFrame with flow data
            fit: Whether to fit PCA (True for training, False for inference)
            
        Returns:
            pca_df: DataFrame with PCA components
        """
        # Get flow columns
        flow_cols = [col for col in flow_data.columns if col.startswith('flow_')]
        
        # Extract flow values
        X = flow_data[flow_cols].fillna(0)
        
        # Scale
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # PCA
        if fit:
            self.pca = PCA(n_components=self.n_components)
            components = self.pca.fit_transform(X_scaled)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            components = self.pca.transform(X_scaled)
        
        # Create DataFrame
        pca_df = pd.DataFrame(
            components,
            columns=[f'topology_pca_{i}' for i in range(self.n_components)]
        )
        pca_df['timestamp'] = flow_data['timestamp'].values
        
        return pca_df
    
    def compute_all_signals(
        self,
        flow_data: pd.DataFrame,
        lmp_data: pd.DataFrame,
        nodes: pd.DataFrame,
        lines: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Compute all topology signals.
        
        Args:
            flow_data: Flow data
            lmp_data: LMP data
            nodes: Node metadata
            lines: Line metadata
            fit: Whether to fit scalers/PCA
            
        Returns:
            signals_df: Complete topology signals DataFrame
        """
        logger.info("Computing topology signals...")
        
        # Build network graph
        G = self.build_network_graph(nodes, lines)
        
        # Compute centrality metrics
        centrality_df = self.compute_centrality_metrics(G)
        
        # Compute co-movement
        comovement_df = self.compute_flow_comovement(flow_data)
        
        # Compute flow proxies
        proxy_df = self.compute_flow_proxies(flow_data, lmp_data, G)
        
        # Compute PCA features
        pca_df = self.compute_pca_features(flow_data, fit=fit)
        
        # Merge all signals
        signals_df = flow_data[['timestamp']].copy()
        
        # Add time-varying features
        signals_df = signals_df.merge(comovement_df, on='timestamp', how='left')
        signals_df = signals_df.merge(proxy_df, on='timestamp', how='left')
        signals_df = signals_df.merge(pca_df, on='timestamp', how='left')
        
        # Fill NaN
        signals_df = signals_df.fillna(0)
        
        logger.info(f"Generated {len(signals_df.columns)-1} topology signals")
        
        return signals_df


def main():
    """Main pipeline for topology signal extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract topology signals')
    parser.add_argument('--flow-data', required=True, help='Path to flow data CSV')
    parser.add_argument('--lmp-data', required=True, help='Path to LMP data CSV')
    parser.add_argument('--nodes', required=True, help='Path to nodes CSV')
    parser.add_argument('--lines', required=True, help='Path to lines CSV')
    parser.add_argument('--output', required=True, help='Path to output CSV')
    parser.add_argument('--lookback-window', type=int, default=24, help='Lookback window in hours')
    parser.add_argument('--n-components', type=int, default=5, help='Number of PCA components')
    
    args = parser.parse_args()
    
    # Load data
    flow_data = pd.read_csv(args.flow_data, parse_dates=['timestamp'])
    lmp_data = pd.read_csv(args.lmp_data, parse_dates=['timestamp'])
    nodes = pd.read_csv(args.nodes)
    lines = pd.read_csv(args.lines)
    
    # Initialize extractor
    extractor = TopologySignalExtractor(
        lookback_window=args.lookback_window,
        n_components=args.n_components
    )
    
    # Compute signals
    signals = extractor.compute_all_signals(
        flow_data,
        lmp_data,
        nodes,
        lines,
        fit=True
    )
    
    # Save
    signals.to_csv(args.output, index=False)
    print(f"Topology signals saved to {args.output}")
    print(f"Generated {len(signals.columns)-1} features for {len(signals)} timestamps")


if __name__ == '__main__':
    main()
