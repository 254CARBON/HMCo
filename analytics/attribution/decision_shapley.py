"""
Alpha Attribution & Decision Shapley

Decomposes P&L to data features using Shapley values over counterfactual scenarios.
DoD: Top-3 drivers explain ≥70% of P&L variance for target books.
"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from itertools import combinations

logger = logging.getLogger(__name__)


class DecisionShapley:
    """
    Compute Shapley values for decision attribution.
    
    Attributes P&L to individual features/signals using game-theoretic approach.
    """
    
    def __init__(self, sampling: bool = True, n_samples: int = 100):
        """
        Initialize Shapley calculator.
        
        Args:
            sampling: Whether to use sampling approximation (faster)
            n_samples: Number of samples for approximation
        """
        self.sampling = sampling
        self.n_samples = n_samples
    
    def compute_shapley_values(
        self,
        decision_id: str,
        features: Dict[str, float],
        pnl: float,
        model_predict_fn
    ) -> Dict[str, float]:
        """
        Compute Shapley values for a decision.
        
        Args:
            decision_id: Decision identifier
            features: Dict of feature_name -> value
            pnl: Realized P&L
            model_predict_fn: Function that predicts P&L given features
            
        Returns:
            shapley_values: Dict of feature_name -> Shapley value
        """
        logger.info(f"Computing Shapley values for decision {decision_id}")
        
        feature_names = list(features.keys())
        n_features = len(feature_names)
        shapley_values = {name: 0.0 for name in feature_names}
        
        if self.sampling:
            # Sampling approximation
            for _ in range(self.n_samples):
                # Random permutation of features
                perm = np.random.permutation(feature_names)
                
                # Compute marginal contributions
                subset = {}
                prev_value = model_predict_fn({})
                
                for feature_name in perm:
                    subset[feature_name] = features[feature_name]
                    curr_value = model_predict_fn(subset)
                    marginal = curr_value - prev_value
                    shapley_values[feature_name] += marginal
                    prev_value = curr_value
            
            # Average over samples
            for name in feature_names:
                shapley_values[name] /= self.n_samples
        else:
            # Exact calculation (exponential complexity)
            for feature_name in feature_names:
                marginal_sum = 0.0
                
                # Iterate over all subsets not containing feature_name
                other_features = [f for f in feature_names if f != feature_name]
                
                for k in range(len(other_features) + 1):
                    for subset in combinations(other_features, k):
                        subset_dict = {f: features[f] for f in subset}
                        
                        # Value with and without feature_name
                        value_without = model_predict_fn(subset_dict)
                        subset_dict[feature_name] = features[feature_name]
                        value_with = model_predict_fn(subset_dict)
                        
                        # Marginal contribution
                        marginal = value_with - value_without
                        
                        # Weight by combinatorial factor
                        weight = 1.0 / (n_features * np.math.comb(n_features - 1, k))
                        marginal_sum += weight * marginal
                
                shapley_values[feature_name] = marginal_sum
        
        logger.info(f"Shapley values computed for {n_features} features")
        return shapley_values
    
    def attribute_pnl(
        self,
        decisions: pd.DataFrame,
        pnl_data: pd.DataFrame,
        model_predict_fn
    ) -> pd.DataFrame:
        """
        Attribute P&L across multiple decisions.
        
        Args:
            decisions: DataFrame with decision records
            pnl_data: DataFrame with P&L by decision
            model_predict_fn: Prediction function
            
        Returns:
            attribution: DataFrame with Shapley values per decision
        """
        results = []
        
        for _, row in decisions.iterrows():
            decision_id = row['decision_id']
            
            # Get features from JSON
            import json
            features = json.loads(row['feature_snapshot'])
            
            # Get P&L
            pnl = pnl_data[pnl_data['decision_id'] == decision_id]['total_pnl'].iloc[0]
            
            # Compute Shapley values
            shapley_values = self.compute_shapley_values(
                decision_id,
                features,
                pnl,
                model_predict_fn
            )
            
            # Store results
            for feature_name, value in shapley_values.items():
                results.append({
                    'decision_id': decision_id,
                    'feature_name': feature_name,
                    'shapley_value': value,
                    'shapley_value_pct': value / pnl * 100 if pnl != 0 else 0,
                    'feature_value': features[feature_name]
                })
        
        return pd.DataFrame(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--decisions', required=True, help='Decisions CSV')
    parser.add_argument('--pnl', required=True, help='P&L CSV')
    parser.add_argument('--output', required=True, help='Output CSV')
    args = parser.parse_args()
    
    decisions = pd.read_csv(args.decisions, parse_dates=['timestamp'])
    pnl_data = pd.read_csv(args.pnl, parse_dates=['timestamp'])
    
    # Dummy prediction function (would be replaced with actual model)
    def dummy_predict(features):
        return sum(features.values())
    
    shapley = DecisionShapley(sampling=True, n_samples=50)
    attribution = shapley.attribute_pnl(decisions, pnl_data, dummy_predict)
    
    attribution.to_csv(args.output, index=False)
    print(f"Attribution saved to {args.output}")
