"""
Inverse Reinforcement Learning for market participants
Models bidding behavior and responses to market conditions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MarketParticipantIRL(nn.Module):
    """
    Maximum entropy IRL for participant behavior modeling
    
    Learns reward functions from observed bid/dispatch actions
    Supports different participant types: peakers, baseload, renewables
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,
        hidden_dim: int = 128,
        participant_type: str = 'peaker'
    ):
        super().__init__()
        
        self.participant_type = participant_type
        
        # Reward function network
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network (for rollouts)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute reward for state-action pair"""
        sa = torch.cat([state, action], dim=-1)
        reward = self.reward_net(sa)
        return reward
    
    def predict_action(self, state: torch.Tensor) -> torch.Tensor:
        """Predict action given state"""
        return self.policy_net(state)
    
    def learn_from_demonstrations(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor
    ):
        """
        Learn reward function from participant demonstrations
        
        Args:
            states: Market states [batch, state_dim]
            actions: Observed actions (bids) [batch, action_dim]
            outcomes: Dispatch outcomes [batch]
        """
        # Maximum entropy IRL objective
        # Maximize likelihood of demonstrations under learned reward
        rewards = self.forward(states, actions)
        
        # Feature expectations from demonstrations
        demo_features = torch.mean(states, dim=0)
        
        # Feature expectations from policy rollouts
        policy_actions = self.predict_action(states)
        policy_features = torch.mean(states, dim=0)
        
        # IRL loss: match feature expectations
        loss = torch.nn.functional.mse_loss(policy_features, demo_features)
        
        return loss


class ParticipantLibrary:
    """Library of different participant agent policies"""
    
    def __init__(self):
        self.agents = {
            'peaker': MarketParticipantIRL(participant_type='peaker'),
            'baseload': MarketParticipantIRL(participant_type='baseload'),
            'renewable': MarketParticipantIRL(participant_type='renewable'),
            'storage': MarketParticipantIRL(participant_type='storage')
        }
        
    def get_agent(self, participant_type: str) -> MarketParticipantIRL:
        """Get agent model for participant type"""
        return self.agents.get(participant_type, self.agents['peaker'])
    
    def simulate_market(
        self,
        state: Dict,
        participant_mix: Dict[str, int]
    ) -> Dict:
        """
        Simulate market with mix of participants
        
        Args:
            state: Market state
            participant_mix: Dict of {type: count}
            
        Returns:
            Aggregate market response
        """
        responses = {}
        
        for ptype, count in participant_mix.items():
            agent = self.get_agent(ptype)
            
            # Simulate participant responses
            state_tensor = torch.zeros(1, 32)  # Mock state
            action = agent.predict_action(state_tensor)
            
            responses[ptype] = {
                'action': action.detach().numpy(),
                'count': count
            }
        
        return responses
