"""Probabilistic curve builder with conformal calibration"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ConformalQuantileRegressor(nn.Module):
    """
    Quantile regression with conformal prediction for calibrated intervals
    Target: 90% PI contains truth ≥90% of time
    """
    
    def __init__(self, input_dim: int = 64, num_quantiles: int = 99):
        super().__init__()
        
        self.quantiles = torch.linspace(0.01, 0.99, num_quantiles)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.quantile_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_quantiles)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quantiles"""
        h = self.encoder(x)
        quantiles = torch.stack([head(h) for head in self.quantile_heads], dim=-1)
        return quantiles.squeeze(1)
    
    def calibrate(self, predictions: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
        """Apply conformal calibration"""
        errors = torch.abs(actuals.unsqueeze(-1) - predictions)
        calibration_quantile = torch.quantile(errors, 0.95, dim=0)
        return calibration_quantile
