"""Generative scenario factory using diffusion models"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DiffusionScenarioGenerator(nn.Module):
    """
    Diffusion model for generating plausible price/weather scenarios
    Target: >10% better generalization in regime shifts
    """
    
    def __init__(self, data_dim: int = 128, time_steps: int = 1000):
        super().__init__()
        
        self.time_steps = time_steps
        
        # Noise prediction network (U-Net style)
        self.noise_pred = nn.Sequential(
            nn.Linear(data_dim + 1, 256),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim)
        )
        
        # Beta schedule for diffusion
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, time_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise at timestep t"""
        t_embed = t.float().unsqueeze(-1) / self.time_steps
        x_t = torch.cat([x, t_embed], dim=-1)
        return self.noise_pred(x_t)
    
    def sample(
        self,
        num_samples: int,
        data_dim: int,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate scenarios via denoising
        
        Args:
            num_samples: Number of scenarios to generate
            data_dim: Dimensionality of each scenario
            condition: Optional conditioning (regime, outage, etc.)
            
        Returns:
            Generated scenarios [num_samples, data_dim]
        """
        device = next(self.parameters()).device
        
        # Start from noise
        x = torch.randn(num_samples, data_dim, device=device)
        
        # Denoising loop
        for t in reversed(range(self.time_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x, t_batch)
            
            # Apply conditioning if provided
            if condition is not None:
                noise_pred = noise_pred + 0.1 * condition
            
            # Denoise step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            beta_t = 1 - alpha_t / alpha_t_prev
            
            x = (x - beta_t * noise_pred / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t / alpha_t_prev)
            
            # Add noise (except last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
        
        return x
    
    def generate_scenarios(
        self,
        base_curve: np.ndarray,
        num_scenarios: int = 1000,
        regime: Optional[str] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate scenarios conditioned on regime
        
        Args:
            base_curve: Baseline forecast
            num_scenarios: Number of scenarios
            regime: Regime conditioning (normal, stressed, etc.)
            seed: Random seed
            
        Returns:
            Scenarios [num_scenarios, curve_length]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        data_dim = len(base_curve)
        
        # Regime conditioning
        condition = None
        if regime == 'stressed':
            condition = torch.ones(1, data_dim) * 0.5
        elif regime == 'transition':
            condition = torch.randn(1, data_dim) * 0.3
        
        # Generate
        with torch.no_grad():
            scenarios = self.sample(num_scenarios, data_dim, condition)
        
        # Add to baseline
        scenarios_np = scenarios.cpu().numpy()
        scenarios_np = base_curve[None, :] + scenarios_np
        
        logger.info(f"Generated {num_scenarios} scenarios for regime '{regime}'")
        return scenarios_np
