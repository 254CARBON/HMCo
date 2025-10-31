"""Node embeddings using GraphSAGE for cold-start and similarity"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import logging

logger = logging.getLogger(__name__)


class NodeEmbedding(nn.Module):
    """
    GraphSAGE embeddings for nodes
    Target: k-NN nowcast -10% MAPE on sparse nodes, <20ms p95 ANN search
    """
    
    def __init__(self, in_channels: int = 32, hidden_channels: int = 128, out_channels: int = 256):
        super().__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Generate node embeddings"""
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x  # [num_nodes, 256]
    
    def get_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between embeddings"""
        return torch.nn.functional.cosine_similarity(embed1, embed2, dim=-1)
