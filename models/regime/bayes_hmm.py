"""
Bayesian Hidden Markov Model for regime detection
Identifies market regimes and structural breaks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BayesianHMM:
    """
    Bayesian HMM for market regime detection
    
    Identifies regimes: normal, stressed, transition
    Gates forecasts based on regime probabilities
    Target: ≥25% error reduction out-of-regime
    """
    
    def __init__(
        self,
        num_states: int = 3,
        feature_dim: int = 10
    ):
        self.num_states = num_states
        self.feature_dim = feature_dim
        
        # HMM parameters
        self.transition_matrix = np.ones((num_states, num_states)) / num_states
        self.emission_params = {
            'means': np.random.randn(num_states, feature_dim),
            'covs': [np.eye(feature_dim) for _ in range(num_states)]
        }
        self.initial_probs = np.ones(num_states) / num_states
        
        # Regime labels
        self.regime_labels = {
            0: 'normal',
            1: 'stressed',
            2: 'transition'
        }
        
    def fit(
        self,
        features: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Fit HMM to features using Baum-Welch algorithm
        
        Args:
            features: Time series features [T, feature_dim]
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
        """
        logger.info(f"Fitting Bayesian HMM with {self.num_states} states")
        
        T = len(features)
        
        for iteration in range(max_iter):
            # E-step: Forward-backward algorithm
            alpha, beta = self._forward_backward(features)
            
            # M-step: Update parameters
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            
            xi = self._compute_xi(features, alpha, beta)
            
            # Update transition matrix
            self.transition_matrix = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T
            
            # Update emission parameters
            for s in range(self.num_states):
                self.emission_params['means'][s] = (gamma[:, s:s+1] * features).sum(axis=0) / gamma[:, s].sum()
                
                diff = features - self.emission_params['means'][s]
                self.emission_params['covs'][s] = (gamma[:, s:s+1] * diff).T @ diff / gamma[:, s].sum()
            
            # Check convergence
            if iteration > 0 and np.abs(prev_ll - self._log_likelihood(features)) < tol:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            prev_ll = self._log_likelihood(features)
        
        logger.info("HMM fitting complete")
    
    def predict_regime(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for new features
        
        Args:
            features: Features [T, feature_dim]
            
        Returns:
            most_likely_states: Viterbi path [T]
            state_probs: State probabilities [T, num_states]
        """
        # Viterbi algorithm for most likely path
        most_likely_states = self._viterbi(features)
        
        # Forward algorithm for state probabilities
        alpha, _ = self._forward_backward(features)
        state_probs = alpha / alpha.sum(axis=1, keepdims=True)
        
        return most_likely_states, state_probs
    
    def gate_forecast(
        self,
        forecast: Dict,
        state_probs: np.ndarray,
        threshold: float = 0.7
    ) -> Dict:
        """
        Gate forecast based on regime probabilities
        
        Args:
            forecast: Forecast dict
            state_probs: Current state probabilities
            threshold: Confidence threshold for gating
            
        Returns:
            Gated forecast with flags
        """
        # Get current regime
        current_regime_idx = np.argmax(state_probs[-1])
        current_regime = self.regime_labels[current_regime_idx]
        regime_prob = state_probs[-1, current_regime_idx]
        
        # Determine gating action
        if regime_prob >= threshold:
            if current_regime == 'normal':
                action = 'allow'
                weight = 1.0
            elif current_regime == 'transition':
                action = 'down_weight'
                weight = 0.5
            else:  # stressed
                action = 'block'
                weight = 0.0
        else:
            action = 'down_weight'
            weight = 0.3
        
        # Apply gating
        gated_forecast = forecast.copy()
        gated_forecast['gate_action'] = action
        gated_forecast['gate_weight'] = weight
        gated_forecast['regime'] = current_regime
        gated_forecast['regime_prob'] = float(regime_prob)
        
        return gated_forecast
    
    def _forward_backward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-backward algorithm"""
        T = len(features)
        
        # Forward pass
        alpha = np.zeros((T, self.num_states))
        alpha[0] = self.initial_probs * self._emission_prob(features[0])
        
        for t in range(1, T):
            for s in range(self.num_states):
                alpha[t, s] = self._emission_prob(features[t], s) * \
                             (alpha[t-1] @ self.transition_matrix[:, s])
        
        # Backward pass
        beta = np.zeros((T, self.num_states))
        beta[-1] = 1
        
        for t in range(T-2, -1, -1):
            for s in range(self.num_states):
                beta[t, s] = (self.transition_matrix[s] * \
                             self._emission_prob(features[t+1]) * \
                             beta[t+1]).sum()
        
        return alpha, beta
    
    def _viterbi(self, features: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence"""
        T = len(features)
        
        # Initialize
        delta = np.zeros((T, self.num_states))
        psi = np.zeros((T, self.num_states), dtype=int)
        
        delta[0] = self.initial_probs * self._emission_prob(features[0])
        
        # Forward pass
        for t in range(1, T):
            for s in range(self.num_states):
                probs = delta[t-1] * self.transition_matrix[:, s]
                psi[t, s] = np.argmax(probs)
                delta[t, s] = np.max(probs) * self._emission_prob(features[t], s)
        
        # Backward pass
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path
    
    def _emission_prob(self, feature: np.ndarray, state: int = None) -> np.ndarray:
        """Compute emission probability (Gaussian)"""
        if state is None:
            # All states
            probs = np.zeros(self.num_states)
            for s in range(self.num_states):
                probs[s] = self._gaussian_pdf(
                    feature,
                    self.emission_params['means'][s],
                    self.emission_params['covs'][s]
                )
            return probs
        else:
            # Specific state
            return self._gaussian_pdf(
                feature,
                self.emission_params['means'][state],
                self.emission_params['covs'][state]
            )
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Multivariate Gaussian PDF"""
        d = len(x)
        diff = x - mean
        
        try:
            cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(d))
            det = np.linalg.det(cov + 1e-6 * np.eye(d))
            norm = 1.0 / np.sqrt((2 * np.pi) ** d * det)
            exponent = -0.5 * diff.T @ cov_inv @ diff
            return norm * np.exp(exponent)
        except:
            return 1e-10
    
    def _compute_xi(self, features: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute xi (state transition expectations)"""
        T = len(features)
        xi = np.zeros((T-1, self.num_states, self.num_states))
        
        for t in range(T-1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * \
                                 self._emission_prob(features[t+1], j) * beta[t+1, j]
            
            xi[t] = xi[t] / xi[t].sum()
        
        return xi
    
    def _log_likelihood(self, features: np.ndarray) -> float:
        """Compute log likelihood"""
        alpha, _ = self._forward_backward(features)
        return np.log(alpha[-1].sum())
