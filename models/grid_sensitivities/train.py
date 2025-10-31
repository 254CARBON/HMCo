"""
PTDF/LODF Training Module

Trains Power Transfer Distribution Factor (PTDF) and Line Outage Distribution 
Factor (LODF) models using compressed sensing and sparse VAR/Granger with ridge 
regression for near-real-time network sensitivity estimation.

DoD: On historical outages, predicted ΔLMP correlation ≥ 0.8; confidence tracks error.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import csr_matrix
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class PTDFLODFTrainer:
    """
    Trainer for PTDF/LODF sensitivity matrices using compressed sensing.
    
    Uses sparse VAR/Granger models with ridge regularization to estimate
    network sensitivities from historical flow and price data.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        n_splits: int = 5,
        confidence_level: float = 0.95
    ):
        """
        Initialize PTDF/LODF trainer.
        
        Args:
            alpha: Regularization strength
            l1_ratio: L1 penalty ratio for elastic net
            n_splits: Number of time series splits for cross-validation
            confidence_level: Confidence level for uncertainty estimation
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_splits = n_splits
        self.confidence_level = confidence_level
        self.ptdf_model = None
        self.lodf_model = None
        self.feature_cols = []
        self.target_cols = []
        
    def prepare_features(
        self,
        flow_data: pd.DataFrame,
        lmp_data: pd.DataFrame,
        topology_signals: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for PTDF/LODF training.
        
        Args:
            flow_data: Historical line flow data
            lmp_data: Historical LMP data
            topology_signals: Co-movement and flow proxy signals
            
        Returns:
            X: Feature matrix
            y: Target matrix (ΔLMP)
        """
        # Merge datasets on timestamp
        df = flow_data.merge(lmp_data, on='timestamp', how='inner')
        df = df.merge(topology_signals, on='timestamp', how='inner')
        
        # Calculate price deltas
        lmp_cols = [col for col in df.columns if col.startswith('lmp_')]
        for col in lmp_cols:
            df[f'delta_{col}'] = df[col].diff()
        
        # Calculate flow deltas
        flow_cols = [col for col in df.columns if col.startswith('flow_')]
        for col in flow_cols:
            df[f'delta_{col}'] = df[col].diff()
        
        # Feature columns: flow changes, topology signals, lagged values
        self.feature_cols = (
            [f'delta_{col}' for col in flow_cols] +
            [col for col in df.columns if col.startswith('topology_')] +
            [f'{col}_lag1' for col in flow_cols]
        )
        
        # Add lagged features
        for col in flow_cols:
            df[f'{col}_lag1'] = df[col].shift(1)
        
        # Target columns: LMP deltas
        self.target_cols = [f'delta_{col}' for col in lmp_cols]
        
        # Drop NaN rows
        df = df.dropna()
        
        X = df[self.feature_cols]
        y = df[self.target_cols]
        
        return X, y
    
    def train_ptdf(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Train PTDF model using ridge regression with compressed sensing.
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            metrics: Training metrics including correlation and R²
        """
        logger.info("Training PTDF model with compressed sensing...")
        
        # Use ridge regression for stability
        self.ptdf_model = Ridge(alpha=self.alpha, solver='sparse_cg')
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.ptdf_model.fit(X_train, y_train)
            y_pred = self.ptdf_model.predict(X_val)
            
            # Calculate correlation for each node
            correlations = [
                np.corrcoef(y_val.iloc[:, i], y_pred[:, i])[0, 1]
                for i in range(y_val.shape[1])
            ]
            cv_scores.append(np.mean(correlations))
        
        # Final training on full dataset
        self.ptdf_model.fit(X, y)
        y_pred = self.ptdf_model.predict(X)
        
        # Calculate metrics
        correlations = [
            np.corrcoef(y.iloc[:, i], y_pred[:, i])[0, 1]
            for i in range(y.shape[1])
        ]
        
        metrics = {
            'mean_correlation': np.mean(correlations),
            'min_correlation': np.min(correlations),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"PTDF training complete. Mean correlation: {metrics['mean_correlation']:.3f}")
        
        return metrics
    
    def train_lodf(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        outage_mask: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train LODF model for line outage scenarios.
        
        Args:
            X: Feature matrix
            y: Target matrix
            outage_mask: Binary mask indicating outage periods
            
        Returns:
            metrics: Training metrics
        """
        logger.info("Training LODF model for outage scenarios...")
        
        # Filter to outage periods if mask provided
        if outage_mask is not None:
            X = X[outage_mask]
            y = y[outage_mask]
        
        # Use Lasso for sparse LODF estimation
        self.lodf_model = Lasso(alpha=self.alpha, max_iter=5000)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.lodf_model.fit(X_train, y_train)
            y_pred = self.lodf_model.predict(X_val)
            
            correlations = [
                np.corrcoef(y_val.iloc[:, i], y_pred[:, i])[0, 1]
                for i in range(y_val.shape[1])
            ]
            cv_scores.append(np.mean(correlations))
        
        # Final training
        self.lodf_model.fit(X, y)
        y_pred = self.lodf_model.predict(X)
        
        # Calculate metrics
        correlations = [
            np.corrcoef(y.iloc[:, i], y_pred[:, i])[0, 1]
            for i in range(y.shape[1])
        ]
        
        metrics = {
            'mean_correlation': np.mean(correlations),
            'min_correlation': np.min(correlations),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"LODF training complete. Mean correlation: {metrics['mean_correlation']:.3f}")
        
        return metrics
    
    def estimate_confidence(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate confidence intervals for predictions.
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            confidence_df: DataFrame with confidence bounds
        """
        if self.ptdf_model is None:
            raise ValueError("Model not trained. Call train_ptdf first.")
        
        # Use bootstrap for confidence estimation
        n_bootstrap = 100
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]
            
            # Train model on bootstrap sample
            model = Ridge(alpha=self.alpha, solver='sparse_cg')
            model.fit(X_boot, y_boot)
            
            # Predict on original data
            y_pred = model.predict(X)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Calculate confidence bounds
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100
        
        confidence_df = pd.DataFrame({
            'mean': predictions.mean(axis=0).mean(axis=1),
            'lower': np.percentile(predictions, lower_percentile, axis=0).mean(axis=1),
            'upper': np.percentile(predictions, upper_percentile, axis=0).mean(axis=1),
            'std': predictions.std(axis=0).mean(axis=1)
        })
        
        return confidence_df
    
    def save_models(self, output_path: str):
        """Save trained models to disk."""
        models = {
            'ptdf_model': self.ptdf_model,
            'lodf_model': self.lodf_model,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols,
            'metadata': {
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio,
                'trained_at': datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(models, f)
        
        logger.info(f"Models saved to {output_path}")
    
    @staticmethod
    def load_models(input_path: str) -> 'PTDFLODFTrainer':
        """Load trained models from disk."""
        with open(input_path, 'rb') as f:
            models = pickle.load(f)
        
        trainer = PTDFLODFTrainer()
        trainer.ptdf_model = models['ptdf_model']
        trainer.lodf_model = models['lodf_model']
        trainer.feature_cols = models['feature_cols']
        trainer.target_cols = models['target_cols']
        
        logger.info(f"Models loaded from {input_path}")
        return trainer


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PTDF/LODF models')
    parser.add_argument('--flow-data', required=True, help='Path to flow data CSV')
    parser.add_argument('--lmp-data', required=True, help='Path to LMP data CSV')
    parser.add_argument('--topology-signals', required=True, help='Path to topology signals CSV')
    parser.add_argument('--output-model', required=True, help='Path to save trained model')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization strength')
    
    args = parser.parse_args()
    
    # Load data
    flow_data = pd.read_csv(args.flow_data, parse_dates=['timestamp'])
    lmp_data = pd.read_csv(args.lmp_data, parse_dates=['timestamp'])
    topology_signals = pd.read_csv(args.topology_signals, parse_dates=['timestamp'])
    
    # Initialize trainer
    trainer = PTDFLODFTrainer(alpha=args.alpha)
    
    # Prepare features
    X, y = trainer.prepare_features(flow_data, lmp_data, topology_signals)
    
    # Train models
    ptdf_metrics = trainer.train_ptdf(X, y)
    lodf_metrics = trainer.train_lodf(X, y)
    
    # Estimate confidence
    confidence = trainer.estimate_confidence(X, y)
    
    # Save models
    trainer.save_models(args.output_model)
    
    # Print results
    print("\nPTDF Metrics:")
    for key, value in ptdf_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nLODF Metrics:")
    for key, value in lodf_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nConfidence intervals estimated with {len(confidence)} samples")


if __name__ == '__main__':
    main()
