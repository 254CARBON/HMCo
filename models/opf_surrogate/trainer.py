"""Training for OPF surrogate model"""

import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class OPFSurrogateTrainer:
    """Trainer for OPF surrogate model on DC-OPF samples"""
    
    def __init__(self, model_config: Dict):
        from .surrogate_model import OPFSurrogate
        self.model = OPFSurrogate(**model_config)
        
    def train(self, train_loader, val_loader, epochs: int = 50):
        """Train model on DC-OPF solution samples"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                x, edge_index, y = batch
                pred = self.model(x, edge_index)
                
                # Quantile loss
                loss = 0
                for q_name, q_pred in pred.items():
                    if q_name != 'confidence':
                        loss += torch.nn.functional.mse_loss(q_pred, y)
                
                loss.backward()
                optimizer.step()
            
            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")
