"""
DA↔RT Imbalance Cost Training Module

Trains quantile regression models with conformal prediction for schedule risk pricing.
DoD: Backtest shows hedge cuts realized imbalance costs by ≥20% at same load.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import pickle

logger = logging.getLogger(__name__)


class ImbalanceCostModel:
    """Quantile regression model for imbalance cost prediction."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.models = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train quantile models."""
        logger.info(f"Training imbalance cost models for quantiles: {self.quantiles}")
        
        metrics = {}
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            self.models[q] = model
            
            # Calculate in-sample metrics
            y_pred = model.predict(X)
            mae = np.mean(np.abs(y - y_pred))
            metrics[f'q{int(q*100)}_mae'] = mae
            
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict imbalance costs for all quantiles."""
        predictions = {}
        for q in self.quantiles:
            predictions[f'q{int(q*100)}'] = self.models[q].predict(X)
        return pd.DataFrame(predictions)
    
    def save(self, path: str):
        """Save models to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'models': self.models, 'quantiles': self.quantiles}, f)
        logger.info(f"Models saved to {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.features)
    X = df.drop(['imbalance_cost'], axis=1)
    y = df['imbalance_cost']
    
    model = ImbalanceCostModel()
    model.train(X, y)
    model.save(args.output)
