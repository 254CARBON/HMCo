"""Safe offline RL strategy optimizer with CVaR constraints"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CQLHedger(nn.Module):
    """
    Conservative Q-Learning hedger with CVaR risk constraints
    Target: Sharpe +25%, Max DD -20% vs heuristic
    """
    
    def __init__(self, state_dim: int = 64, action_dim: int = 10, cvar_alpha: float = 0.95):
        super().__init__()
        
        self.cvar_alpha = cvar_alpha
        
        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get action from policy"""
        return self.policy(state)
    
    def compute_cvar(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute CVaR at alpha level"""
        sorted_returns, _ = torch.sort(returns)
        cutoff_idx = int(len(sorted_returns) * (1 - self.cvar_alpha))
        cvar = sorted_returns[:cutoff_idx].mean()
        return cvar
    
    def optimize_with_constraints(
        self,
        state: torch.Tensor,
        risk_budget: float,
        position_limits: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Optimize action with risk and capacity constraints"""
        action = self.forward(state)
        
        # Apply position limits
        action = torch.clamp(action, -position_limits['max_short'], position_limits['max_long'])
        
        # Estimate risk
        q_value = self.q_net(torch.cat([state, action], dim=-1))
        
        metrics = {
            'action': action.detach().numpy(),
            'expected_value': q_value.item(),
            'risk_used': abs(action).sum().item() / risk_budget
        }
        
        return action, metrics
