"""
Unit tests for LMP nowcasting module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch


class TestLMPDataPreparation:
    """Test data preparation for LMP nowcasting"""
    
    def test_load_graph_topology(self):
        """Test graph topology loading"""
        from models.lmp_nowcast.dataprep import LMPDataPreparation
        
        dataprep = LMPDataPreparation()
        graph = dataprep.load_graph_topology('CAISO')
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert 'iso' in graph
        assert graph['iso'] == 'CAISO'
        assert len(graph['nodes']) > 0
    
    def test_prepare_training_dataset(self):
        """Test training dataset preparation"""
        from models.lmp_nowcast.dataprep import LMPDataPreparation
        
        dataprep = LMPDataPreparation()
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        dataset = dataprep.prepare_training_dataset('CAISO', start_date, end_date)
        
        assert 'graph' in dataset
        assert 'weather' in dataset
        assert 'lmp_history' in dataset
        assert 'metadata' in dataset
        assert dataset['metadata']['iso'] == 'CAISO'


class TestPhysicsGuidedSTTransformer:
    """Test physics-guided transformer model"""
    
    def test_model_forward(self):
        """Test forward pass"""
        from models.lmp_nowcast.trainer import PhysicsGuidedSTTransformer
        
        model = PhysicsGuidedSTTransformer(
            num_nodes=10,
            node_features=8,
            hidden_dim=32,
            num_layers=2,
            forecast_horizon=12
        )
        
        # Create mock input
        batch_size = 2
        seq_len = 12
        num_nodes = 10
        features = 8
        
        x = torch.randn(batch_size, seq_len, num_nodes, features)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        ptdf = torch.eye(num_nodes)
        
        # Forward pass
        predictions = model(x, edge_index, ptdf)
        
        assert 'q10' in predictions
        assert 'q50' in predictions
        assert 'q90' in predictions
        
        # Check shapes
        assert predictions['q50'].shape == (batch_size, num_nodes, 12)
    
    def test_quantile_loss(self):
        """Test quantile loss computation"""
        from models.lmp_nowcast.trainer import PhysicsGuidedSTTransformer
        
        model = PhysicsGuidedSTTransformer(num_nodes=10)
        
        pred = torch.randn(10, 12)
        target = torch.randn(10, 12)
        
        loss = model._quantile_loss(pred, target, 0.5)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestLMPNowcastInference:
    """Test inference engine"""
    
    @pytest.mark.skip(reason="Requires trained model")
    def test_predict(self):
        """Test prediction"""
        from models.lmp_nowcast.infer import LMPNowcastInference
        
        # Would need a trained model checkpoint
        # inference = LMPNowcastInference('model.pt')
        # results = inference.predict(features)
        pass
    
    def test_compute_metrics(self):
        """Test metrics computation"""
        from models.lmp_nowcast.infer import LMPNowcastInference
        
        # Create mock inference engine (without loading model)
        class MockInference:
            def compute_metrics(self, predictions, actuals):
                from models.lmp_nowcast.infer import LMPNowcastInference
                temp = LMPNowcastInference.__new__(LMPNowcastInference)
                return temp.compute_metrics(predictions, actuals)
        
        inference = MockInference()
        
        predictions = {
            'q10': np.array([[25, 26, 27]]),
            'q50': np.array([[30, 31, 32]]),
            'q90': np.array([[35, 36, 37]])
        }
        
        actuals = {
            'actual': np.array([[31, 30, 33]])
        }
        
        metrics = inference.compute_metrics(predictions, actuals)
        
        assert 'mape' in metrics
        assert 'coverage_80' in metrics
        assert metrics['mape'] >= 0


class TestGraphTopology:
    """Test graph topology module"""
    
    def test_load_topology(self):
        """Test topology loading"""
        from features.graph.topology import GraphTopology
        
        topo = GraphTopology()
        graph = topo.load_topology('CAISO')
        
        assert graph['num_nodes'] > 0
        assert graph['num_edges'] >= 0
        assert 'nodes' in graph
    
    def test_edge_index_format(self):
        """Test edge index conversion"""
        from features.graph.topology import GraphTopology
        
        topo = GraphTopology()
        graph = topo.load_topology('CAISO')
        
        edge_index = topo.get_edge_index(graph)
        
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0


class TestPTDFEstimator:
    """Test PTDF estimation"""
    
    def test_compute_ptdf(self):
        """Test PTDF computation"""
        from features.graph.ptdf import PTDFEstimator
        from features.graph.topology import GraphTopology
        
        topo = GraphTopology()
        graph = topo.load_topology('CAISO')
        
        ptdf_est = PTDFEstimator()
        ptdf = ptdf_est.compute_ptdf(graph)
        
        assert ptdf.shape[0] == graph['num_nodes']
        assert ptdf.shape[1] == graph['num_nodes']
        assert not np.isnan(ptdf).any()


class TestSparkSpread:
    """Test spark spread calculations"""
    
    def test_calculate_spark_spread(self):
        """Test spark spread calculation"""
        from analytics.signals.cross_commodity.spark_spread import SparkSpreadCalculator
        
        calc = SparkSpreadCalculator()
        
        spread = calc.calculate_spark_spread(
            power_lmp=45.0,
            gas_price=5.0,
            heat_rate=7.0,
            variable_om=2.0
        )
        
        expected = 45.0 - (5.0 * 7.0) - 2.0  # = 8.0
        assert abs(spread - expected) < 0.01
    
    def test_calculate_implied_heat_rate(self):
        """Test implied heat rate calculation"""
        from analytics.signals.cross_commodity.spark_spread import SparkSpreadCalculator
        
        calc = SparkSpreadCalculator()
        
        ihr = calc.calculate_implied_heat_rate(
            power_lmp=45.0,
            gas_price=5.0,
            variable_om=2.0
        )
        
        expected = (45.0 - 2.0) / 5.0  # = 8.6
        assert abs(ihr - expected) < 0.01
    
    def test_carbon_adjusted_spread(self):
        """Test carbon-adjusted spread"""
        from analytics.signals.cross_commodity.spark_spread import SparkSpreadCalculator
        
        calc = SparkSpreadCalculator()
        
        adjusted = calc.calculate_carbon_adjusted_spread(
            spark_spread=10.0,
            carbon_price=50.0,
            carbon_intensity=0.4
        )
        
        expected = 10.0 - (50.0 * 0.4)  # = -10.0
        assert abs(adjusted - expected) < 0.01


class TestRegimeDetection:
    """Test regime detection"""
    
    def test_bayes_hmm_initialization(self):
        """Test HMM initialization"""
        from models.regime.bayes_hmm import BayesianHMM
        
        hmm = BayesianHMM(num_states=3, feature_dim=10)
        
        assert hmm.num_states == 3
        assert hmm.feature_dim == 10
        assert hmm.transition_matrix.shape == (3, 3)
    
    def test_regime_prediction(self):
        """Test regime prediction"""
        from models.regime.bayes_hmm import BayesianHMM
        
        hmm = BayesianHMM(num_states=3, feature_dim=5)
        
        # Mock features
        features = np.random.randn(100, 5)
        
        # Should not crash
        states, probs = hmm.predict_regime(features)
        
        assert len(states) == 100
        assert probs.shape == (100, 3)
    
    def test_forecast_gating(self):
        """Test forecast gating"""
        from models.regime.bayes_hmm import BayesianHMM
        
        hmm = BayesianHMM(num_states=3)
        
        forecast = {'value': 45.0}
        state_probs = np.array([[0.1, 0.2, 0.7]])  # Mostly in state 2
        
        gated = hmm.gate_forecast(forecast, state_probs, threshold=0.7)
        
        assert 'gate_action' in gated
        assert 'gate_weight' in gated
        assert 'regime' in gated
        assert gated['gate_weight'] >= 0 and gated['gate_weight'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
