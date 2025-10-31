"""
Graph topology construction for power grid networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GraphTopology:
    """
    Manages power grid topology for graph neural networks
    
    Features:
    - Node-hub mappings
    - Edge construction (transmission lines)
    - Zone/region hierarchies
    - Topology changes over time
    """
    
    def __init__(self, clickhouse_client=None):
        self.clickhouse_client = clickhouse_client
        self.topology_cache = {}
        
    def load_topology(
        self,
        iso: str,
        include_zones: bool = True
    ) -> Dict:
        """
        Load network topology from database
        
        Args:
            iso: ISO name (CAISO, MISO, SPP, etc.)
            include_zones: Whether to include zone/hub hierarchies
            
        Returns:
            Dict with nodes, edges, and hierarchy
        """
        cache_key = f"{iso}_{include_zones}"
        if cache_key in self.topology_cache:
            return self.topology_cache[cache_key]
        
        logger.info(f"Loading topology for {iso}")
        
        # Load nodes
        nodes = self._load_nodes(iso)
        
        # Load edges (transmission lines)
        edges = self._load_edges(iso)
        
        # Load hierarchies
        hierarchies = {}
        if include_zones:
            hierarchies = self._load_hierarchies(iso, nodes)
        
        topology = {
            'iso': iso,
            'nodes': nodes,
            'edges': edges,
            'hierarchies': hierarchies,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
        
        self.topology_cache[cache_key] = topology
        logger.info(f"Loaded {len(nodes)} nodes, {len(edges)} edges for {iso}")
        
        return topology
    
    def _load_nodes(self, iso: str) -> List[Dict]:
        """Load node information"""
        if self.clickhouse_client:
            query = f"""
            SELECT DISTINCT
                node_id,
                node_name,
                hub_id,
                zone_id,
                latitude,
                longitude,
                node_type,
                voltage_level
            FROM iso_node_mapping
            WHERE iso = '{iso}'
            ORDER BY node_id
            """
            try:
                df = self.clickhouse_client.query_dataframe(query)
                return df.to_dict('records')
            except Exception as e:
                logger.error(f"Error loading nodes: {e}")
        
        # Mock data
        return self._generate_mock_nodes(iso)
    
    def _load_edges(self, iso: str) -> List[Tuple]:
        """Load transmission line edges"""
        # In production, load from transmission line database
        # For now, generate based on geographic proximity
        
        nodes = self._load_nodes(iso)
        edges = []
        
        # Create edges based on proximity (simplified)
        for i, node_i in enumerate(nodes):
            lat_i = node_i.get('latitude', 0)
            lon_i = node_i.get('longitude', 0)
            
            for j, node_j in enumerate(nodes[i+1:], start=i+1):
                lat_j = node_j.get('latitude', 0)
                lon_j = node_j.get('longitude', 0)
                
                # Simple distance calculation
                dist = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)
                
                # Connect nearby nodes (< 2 degree radius)
                if dist < 2.0:
                    edges.append((i, j))
        
        return edges
    
    def _load_hierarchies(
        self,
        iso: str,
        nodes: List[Dict]
    ) -> Dict:
        """Load zone/hub hierarchies"""
        hierarchies = {
            'zones': {},
            'hubs': {}
        }
        
        # Group nodes by zone
        for node in nodes:
            zone_id = node.get('zone_id')
            if zone_id:
                if zone_id not in hierarchies['zones']:
                    hierarchies['zones'][zone_id] = []
                hierarchies['zones'][zone_id].append(node['node_id'])
        
        # Group nodes by hub
        for node in nodes:
            hub_id = node.get('hub_id')
            if hub_id:
                if hub_id not in hierarchies['hubs']:
                    hierarchies['hubs'][hub_id] = []
                hierarchies['hubs'][hub_id].append(node['node_id'])
        
        return hierarchies
    
    def get_edge_index(self, topology: Dict) -> np.ndarray:
        """
        Convert edges to PyTorch Geometric format
        
        Returns:
            Edge index array [2, num_edges]
        """
        edges = topology['edges']
        
        if not edges:
            # Return self-loops if no edges
            num_nodes = topology['num_nodes']
            return np.array([
                list(range(num_nodes)),
                list(range(num_nodes))
            ])
        
        # Convert to bidirectional edges
        edge_list = []
        for i, j in edges:
            edge_list.append([i, j])
            edge_list.append([j, i])  # Add reverse edge
        
        edge_array = np.array(edge_list).T
        return edge_array
    
    def get_adjacency_matrix(self, topology: Dict) -> np.ndarray:
        """
        Get adjacency matrix for the graph
        
        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        num_nodes = topology['num_nodes']
        adj = np.zeros((num_nodes, num_nodes))
        
        for i, j in topology['edges']:
            adj[i, j] = 1
            adj[j, i] = 1
        
        return adj
    
    def _generate_mock_nodes(self, iso: str, num_nodes: int = 100) -> List[Dict]:
        """Generate mock node data for testing"""
        nodes = []
        
        # Base coordinates by ISO
        iso_coords = {
            'CAISO': (36.7, -119.7),
            'MISO': (41.8, -87.6),
            'SPP': (38.5, -98.0),
            'ERCOT': (31.0, -99.0),
            'PJM': (40.0, -79.0)
        }
        
        base_lat, base_lon = iso_coords.get(iso, (40.0, -95.0))
        
        for i in range(num_nodes):
            node = {
                'node_id': f'{iso}_NODE_{i:04d}',
                'node_name': f'Node {i}',
                'hub_id': f'HUB_{i % 10}',
                'zone_id': f'ZONE_{i % 5}',
                'latitude': base_lat + np.random.randn() * 2,
                'longitude': base_lon + np.random.randn() * 5,
                'node_type': np.random.choice(['gen', 'load', 'hub'], p=[0.2, 0.7, 0.1]),
                'voltage_level': np.random.choice([69, 138, 230, 345, 500])
            }
            nodes.append(node)
        
        return nodes
