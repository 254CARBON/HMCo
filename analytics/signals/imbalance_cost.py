"""
Imbalance Cost Signals

Produces hub/node risk premia for DA↔RT imbalance costs.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ImbalanceCostSignals:
    """Generate risk premia signals for imbalance costs."""
    
    def compute_risk_premia(
        self,
        imbalance_history: pd.DataFrame,
        window: int = 720  # 30 days hourly
    ) -> pd.DataFrame:
        """
        Compute risk premia by node/hub.
        
        Args:
            imbalance_history: Historical imbalance costs
            window: Rolling window for statistics
            
        Returns:
            risk_premia: DataFrame with P10/P50/P90 and risk premium
        """
        logger.info("Computing imbalance risk premia...")
        
        results = []
        for (node_id, iso), group in imbalance_history.groupby(['node_id', 'iso']):
            group = group.sort_values('timestamp')
            
            # Calculate rolling quantiles
            p10 = group['imbalance_cost_per_mw'].rolling(window, min_periods=24).quantile(0.1)
            p50 = group['imbalance_cost_per_mw'].rolling(window, min_periods=24).quantile(0.5)
            p90 = group['imbalance_cost_per_mw'].rolling(window, min_periods=24).quantile(0.9)
            volatility = group['imbalance_cost_per_mw'].rolling(window, min_periods=24).std()
            
            # Risk premium as difference between P90 and P50
            risk_premium = p90 - p50
            
            results.append(pd.DataFrame({
                'timestamp': group['timestamp'],
                'iso': iso,
                'node_id': node_id,
                'p10': p10,
                'p50': p50,
                'p90': p90,
                'volatility': volatility,
                'risk_premium': risk_premium
            }))
        
        df = pd.concat(results, ignore_index=True)
        df = df.dropna()
        
        logger.info(f"Computed risk premia for {df['node_id'].nunique()} nodes")
        return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', required=True, help='Imbalance history CSV')
    parser.add_argument('--output', required=True, help='Output CSV')
    args = parser.parse_args()
    
    history = pd.read_csv(args.history, parse_dates=['timestamp'])
    signals = ImbalanceCostSignals()
    premia = signals.compute_risk_premia(history)
    premia.to_csv(args.output, index=False)
    print(f"Risk premia saved to {args.output}")
