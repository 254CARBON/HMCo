"""
DA↔RT Imbalance Cost Inference Module

Real-time inference for imbalance cost prediction.
"""

import logging
import pickle
import pandas as pd

logger = logging.getLogger(__name__)


class ImbalanceCostPredictor:
    """Real-time predictor for imbalance costs."""
    
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.models = data['models']
        self.quantiles = data['quantiles']
        logger.info(f"Loaded models for quantiles: {self.quantiles}")
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict imbalance costs."""
        predictions = {}
        for q in self.quantiles:
            predictions[f'q{int(q*100)}'] = self.models[q].predict(features)
        return pd.DataFrame(predictions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    predictor = ImbalanceCostPredictor(args.model)
    features = pd.read_csv(args.features)
    predictions = predictor.predict(features)
    predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
