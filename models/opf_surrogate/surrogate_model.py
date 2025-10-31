"""
Surrogate OPF model using graph neural networks
Fast LMP delta prediction for outage scenarios
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OPFSurrogate(nn.Module):
    """
    Graph-based surrogate for DC-OPF
    
    Predicts LMP deltas given:
    - Network topology
    - Generation/load conditions
    - Outages/constraints
    
    Target: <2s latency for scenarios with confidence bands
    """
    
    def __init__(
        self,
        node_features: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_quantiles: List[float] = [0.05, 0.5, 0.95]
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_quantiles = output_quantiles
        
        # Encoder for node features
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Quantile prediction heads
        self.quantile_predictors = nn.ModuleDict({
            f'q{int(q*100)}': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # LMP delta per node
            )
            for q in output_quantiles
        })
        
        # Confidence estimator
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        outage_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges]
            outage_mask: Binary mask for outaged lines [num_edges]
            
        Returns:
            Dict with quantile predictions and confidence
        """
        # Encode node features
        h = self.node_encoder(x)
        
        # Apply outage mask to edges if provided
        if outage_mask is not None:
            # Filter edges based on mask
            valid_edges = outage_mask > 0
            edge_index = edge_index[:, valid_edges]
        
        # Graph convolutions
        for conv in self.conv_layers:
            h = conv(h, edge_index)
            h = torch.relu(h)
        
        # Quantile predictions
        predictions = {}
        for q_name, predictor in self.quantile_predictors.items():
            pred = predictor(h).squeeze(-1)
            predictions[q_name] = pred
        
        # Confidence scores
        confidence = self.confidence_layer(h).squeeze(-1)
        predictions['confidence'] = confidence
        
        return predictions
    
    def predict_scenario(
        self,
        baseline_lmp: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        outages: List[Tuple[int, int]]
    ) -> Dict[str, np.ndarray]:
        """
        Predict LMP changes for an outage scenario
        
        Args:
            baseline_lmp: Baseline LMP values [num_nodes]
            x: Node features [num_nodes, features]
            edge_index: Graph edges [2, num_edges]
            outages: List of (from_node, to_node) outages
            
        Returns:
            Dict with predicted LMP deltas and confidence
        """
        # Create outage mask
        outage_mask = self._create_outage_mask(edge_index, outages)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(x, edge_index, outage_mask)
        
        # Compute new LMPs
        results = {}
        for q_name, delta in predictions.items():
            if q_name != 'confidence':
                new_lmp = baseline_lmp + delta.cpu().numpy()
                results[f'lmp_{q_name}'] = new_lmp
                results[f'delta_{q_name}'] = delta.cpu().numpy()
        
        results['confidence'] = predictions['confidence'].cpu().numpy()
        
        return results
    
    def _create_outage_mask(
        self,
        edge_index: torch.Tensor,
        outages: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Create binary mask for outaged transmission lines"""
        num_edges = edge_index.shape[1]
        mask = torch.ones(num_edges, dtype=torch.bool)
        
        # Mark outaged edges
        for from_node, to_node in outages:
            for i in range(num_edges):
                if (edge_index[0, i] == from_node and edge_index[1, i] == to_node) or \
                   (edge_index[0, i] == to_node and edge_index[1, i] == from_node):
                    mask[i] = False
        
        return mask
