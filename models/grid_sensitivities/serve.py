"""
PTDF/LODF Serving Module

Real-time inference service for network sensitivity estimation.
Provides fast PTDF/LODF predictions for operational use.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class PTDFLODFServer:
    """
    Real-time serving for PTDF/LODF predictions.
    
    Provides low-latency predictions of nodal price changes based on
    line flow changes and network topology.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize PTDF/LODF server.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.ptdf_model = None
        self.lodf_model = None
        self.feature_cols = []
        self.target_cols = []
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        with open(self.model_path, 'rb') as f:
            models = pickle.load(f)
        
        self.ptdf_model = models['ptdf_model']
        self.lodf_model = models.get('lodf_model')
        self.feature_cols = models['feature_cols']
        self.target_cols = models['target_cols']
        
        logger.info(f"Models loaded from {self.model_path}")
    
    def predict_lmp_delta(
        self,
        flow_changes: Dict[str, float],
        topology_signals: Dict[str, float],
        use_lodf: bool = False
    ) -> Dict[str, float]:
        """
        Predict LMP changes from flow changes.
        
        Args:
            flow_changes: Dict of line_id -> flow change (MW)
            topology_signals: Dict of topology signal name -> value
            use_lodf: Whether to use LODF model (for outage scenarios)
            
        Returns:
            predictions: Dict of node_id -> predicted LMP change ($/MWh)
        """
        # Prepare feature vector
        features = {}
        
        # Add flow deltas
        for col in self.feature_cols:
            if col.startswith('delta_flow_'):
                line_id = col.replace('delta_flow_', '')
                features[col] = flow_changes.get(line_id, 0.0)
            elif col.startswith('topology_'):
                features[col] = topology_signals.get(col, 0.0)
            elif col.endswith('_lag1'):
                # For lag features, use 0 as default (or could cache previous values)
                features[col] = 0.0
        
        # Convert to DataFrame
        X = pd.DataFrame([features])[self.feature_cols]
        
        # Select model
        model = self.lodf_model if use_lodf and self.lodf_model else self.ptdf_model
        
        # Predict
        y_pred = model.predict(X)[0]
        
        # Convert to dict
        predictions = {
            col.replace('delta_lmp_', ''): float(pred)
            for col, pred in zip(self.target_cols, y_pred)
        }
        
        return predictions
    
    def predict_with_confidence(
        self,
        flow_changes: Dict[str, float],
        topology_signals: Dict[str, float],
        confidence_level: float = 0.95,
        use_lodf: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict LMP changes with confidence intervals.
        
        Args:
            flow_changes: Dict of line_id -> flow change (MW)
            topology_signals: Dict of topology signal name -> value
            confidence_level: Confidence level for intervals
            use_lodf: Whether to use LODF model
            
        Returns:
            results: Dict of node_id -> {mean, lower, upper, std}
        """
        # Get base prediction
        predictions = self.predict_lmp_delta(flow_changes, topology_signals, use_lodf)
        
        # Estimate confidence intervals using model coefficients variance
        # (simplified approach - could be enhanced with bootstrap)
        results = {}
        for node_id, pred in predictions.items():
            # Use a heuristic: std ≈ 10% of prediction magnitude
            std = abs(pred) * 0.1
            z_score = 1.96  # for 95% confidence
            
            results[node_id] = {
                'mean': pred,
                'lower': pred - z_score * std,
                'upper': pred + z_score * std,
                'std': std
            }
        
        return results
    
    def batch_predict(
        self,
        scenarios: List[Dict[str, Dict[str, float]]],
        use_lodf: bool = False
    ) -> List[Dict[str, float]]:
        """
        Batch prediction for multiple scenarios.
        
        Args:
            scenarios: List of dicts with 'flow_changes' and 'topology_signals'
            use_lodf: Whether to use LODF model
            
        Returns:
            predictions: List of prediction dicts
        """
        results = []
        
        for scenario in scenarios:
            pred = self.predict_lmp_delta(
                scenario['flow_changes'],
                scenario['topology_signals'],
                use_lodf
            )
            results.append(pred)
        
        return results
    
    def get_sensitivity_matrix(
        self,
        line_ids: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract PTDF sensitivity matrix.
        
        Args:
            line_ids: List of line IDs (None for all)
            node_ids: List of node IDs (None for all)
            
        Returns:
            sensitivity_matrix: DataFrame with lines as rows, nodes as columns
        """
        # Get model coefficients
        coef = self.ptdf_model.coef_
        
        # Extract flow-related features
        flow_feature_indices = [
            i for i, col in enumerate(self.feature_cols)
            if col.startswith('delta_flow_')
        ]
        
        # Build sensitivity matrix
        flow_cols = [
            self.feature_cols[i].replace('delta_flow_', '')
            for i in flow_feature_indices
        ]
        
        node_cols = [col.replace('delta_lmp_', '') for col in self.target_cols]
        
        # Extract relevant coefficients
        sensitivity = coef[flow_feature_indices, :]
        
        # Create DataFrame
        df = pd.DataFrame(sensitivity, index=flow_cols, columns=node_cols)
        
        # Filter if requested
        if line_ids is not None:
            df = df.loc[df.index.isin(line_ids)]
        if node_ids is not None:
            df = df.loc[:, df.columns.isin(node_ids)]
        
        return df
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check server health.
        
        Returns:
            status: Dict with health check results
        """
        status = {
            'ptdf_loaded': self.ptdf_model is not None,
            'lodf_loaded': self.lodf_model is not None,
            'features_loaded': len(self.feature_cols) > 0,
            'targets_loaded': len(self.target_cols) > 0
        }
        
        return status


def main():
    """Main serving pipeline for testing."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Serve PTDF/LODF predictions')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--flow-changes', type=str, help='JSON dict of flow changes')
    parser.add_argument('--topology-signals', type=str, help='JSON dict of topology signals')
    parser.add_argument('--use-lodf', action='store_true', help='Use LODF model')
    parser.add_argument('--sensitivity-matrix', action='store_true', help='Print sensitivity matrix')
    
    args = parser.parse_args()
    
    # Initialize server
    server = PTDFLODFServer(args.model_path)
    
    # Health check
    print("Health Check:")
    health = server.health_check()
    for key, value in health.items():
        print(f"  {key}: {'✓' if value else '✗'}")
    print()
    
    # Sensitivity matrix
    if args.sensitivity_matrix:
        print("Sensitivity Matrix (PTDF):")
        matrix = server.get_sensitivity_matrix()
        print(matrix.head())
        print(f"\nMatrix shape: {matrix.shape}")
        print()
    
    # Prediction
    if args.flow_changes and args.topology_signals:
        flow_changes = json.loads(args.flow_changes)
        topology_signals = json.loads(args.topology_signals)
        
        print("Predicting LMP changes...")
        predictions = server.predict_with_confidence(
            flow_changes,
            topology_signals,
            use_lodf=args.use_lodf
        )
        
        print("\nPredictions with confidence intervals:")
        for node_id, pred in predictions.items():
            print(f"  {node_id}:")
            print(f"    Mean: {pred['mean']:.2f} $/MWh")
            print(f"    95% CI: [{pred['lower']:.2f}, {pred['upper']:.2f}]")
            print(f"    Std: {pred['std']:.2f}")


if __name__ == '__main__':
    main()
