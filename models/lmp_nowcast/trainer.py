"""
Training module for LMP nowcasting
Physics-aware spatiotemporal transformer with graph attention
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PhysicsGuidedSTTransformer(pl.LightningModule):
    """
    Spatiotemporal Transformer with physics-aware constraints
    
    Features:
    - Graph attention for spatial dependencies
    - Temporal transformer for time series
    - DC-OPF style consistency penalty
    - Quantile regression outputs (P10, P50, P90)
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_features: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        forecast_horizon: int = 12,  # 60min / 5min intervals
        quantiles: List[float] = [0.1, 0.5, 0.9],
        physics_weight: float = 0.1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        
        # Node embedding
        self.node_embedding = nn.Embedding(num_nodes, node_features)
        
        # Spatial graph attention layers
        self.spatial_layers = nn.ModuleList([
            GATConv(
                node_features if i == 0 else hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=0.1
            )
            for i in range(num_layers)
        ])
        
        # Temporal transformer
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Quantile prediction heads
        self.quantile_heads = nn.ModuleDict({
            f'q{int(q*100)}': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, forecast_horizon)
            )
            for q in quantiles
        })
        
        # Physics constraint layer (PTDF-based)
        self.physics_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptdf_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [batch, seq_len, num_nodes, features]
            edge_index: Graph edges [2, num_edges]
            ptdf_matrix: Optional PTDF matrix [num_nodes, num_nodes]
            
        Returns:
            Dict with quantile predictions
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial processing with graph attention
        spatial_features = []
        for t in range(seq_len):
            # Process each timestep
            x_t = x[:, t]  # [batch, num_nodes, features]
            
            # Flatten batch dimension for graph processing
            # GATConv expects [num_nodes, features] so we process each batch item separately
            batch_outputs = []
            for b in range(batch_size):
                h = x_t[b]  # [num_nodes, features]
                
                # Apply graph attention layers
                for layer in self.spatial_layers:
                    h = layer(h, edge_index)
                    h = torch.relu(h)
                
                batch_outputs.append(h)
            
            # Stack batch back together [batch, num_nodes, hidden_dim]
            h_batched = torch.stack(batch_outputs, dim=0)
            spatial_features.append(h_batched)
        
        # Stack temporal features [batch, seq_len, num_nodes, hidden_dim]
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Reshape for temporal processing
        # [batch * num_nodes, seq_len, hidden_dim]
        h_temporal = spatial_features.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
        
        # Apply temporal transformer
        h_temporal = self.temporal_encoder(h_temporal)
        
        # Take last timestep for prediction
        h_final = h_temporal[:, -1, :]  # [batch * num_nodes, hidden_dim]
        
        # Apply physics constraints if PTDF available
        if ptdf_matrix is not None:
            h_physics = self._apply_physics_constraint(h_final, ptdf_matrix, batch_size)
            h_final = h_final + self.physics_weight * h_physics
        
        # Reshape back [batch, num_nodes, hidden_dim]
        h_final = h_final.reshape(batch_size, num_nodes, self.hidden_dim)
        
        # Generate quantile predictions
        predictions = {}
        for q_name, head in self.quantile_heads.items():
            # [batch, num_nodes, forecast_horizon]
            pred = head(h_final)
            predictions[q_name] = pred
        
        return predictions
    
    def _apply_physics_constraint(
        self,
        h: torch.Tensor,
        ptdf_matrix: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Apply DC-OPF style consistency penalty"""
        # Simplified physics constraint
        # In production: enforce power flow equations
        h_physics = self.physics_layer(h)
        
        # Reshape and apply PTDF
        h_reshaped = h_physics.reshape(batch_size, self.num_nodes, -1)
        
        # Matrix multiply with PTDF for flow consistency
        # This encourages predictions to respect network topology
        if ptdf_matrix.shape[0] == self.num_nodes:
            h_flow = torch.matmul(ptdf_matrix, h_reshaped)
            return h_flow.reshape(-1, self.hidden_dim)
        
        return h_physics
    
    def _quantile_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        quantile: float
    ) -> torch.Tensor:
        """Pinball loss for quantile regression"""
        errors = target - pred
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        """Training step with quantile loss"""
        x, edge_index, ptdf_matrix, y = batch
        
        # Forward pass
        predictions = self.forward(x, edge_index, ptdf_matrix)
        
        # Compute quantile losses
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            q_name = f'q{int(q*100)}'
            pred = predictions[q_name]
            loss = self._quantile_loss(pred, y, q)
            total_loss += loss
            self.log(f'train_loss_{q_name}', loss, prog_bar=True)
        
        total_loss = total_loss / len(self.quantiles)
        self.log('train_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with metrics"""
        x, edge_index, ptdf_matrix, y = batch
        
        predictions = self.forward(x, edge_index, ptdf_matrix)
        
        # Compute metrics
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            q_name = f'q{int(q*100)}'
            pred = predictions[q_name]
            loss = self._quantile_loss(pred, y, q)
            total_loss += loss
            self.log(f'val_loss_{q_name}', loss, prog_bar=True)
            
            # MAPE for median prediction
            if q == 0.5:
                mape = torch.mean(torch.abs((y - pred) / (y + 1e-8))) * 100
                self.log('val_mape', mape, prog_bar=True)
        
        total_loss = total_loss / len(self.quantiles)
        self.log('val_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class LMPNowcastTrainer:
    """
    High-level trainer for LMP nowcasting model
    Handles data loading, training, and model management
    """
    
    def __init__(
        self,
        model_config: Dict,
        data_config: Dict,
        training_config: Dict
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        
        self.model = None
        self.trainer = None
        
    def build_model(self) -> PhysicsGuidedSTTransformer:
        """Build the model from config"""
        self.model = PhysicsGuidedSTTransformer(**self.model_config)
        logger.info(f"Built model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train(
        self,
        train_loader,
        val_loader,
        mlflow_tracking: bool = True
    ):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            mlflow_tracking: Whether to log to MLflow
        """
        if self.model is None:
            self.build_model()
        
        # Setup callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='lmp-nowcast-{epoch:02d}-{val_loss:.4f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Setup logger
        logger_obj = None
        if mlflow_tracking:
            try:
                from pytorch_lightning.loggers import MLFlowLogger
                logger_obj = MLFlowLogger(
                    experiment_name='lmp-nowcast',
                    tracking_uri='http://mlflow:5000'
                )
            except ImportError:
                logger.warning("MLflow not available, skipping tracking")
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.training_config.get('max_epochs', 100),
            accelerator='auto',
            devices=1,
            callbacks=callbacks,
            logger=logger_obj,
            gradient_clip_val=1.0,
            log_every_n_steps=10
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.fit(self.model, train_loader, val_loader)
        logger.info("Training completed")
        
    def evaluate(self, test_loader) -> Dict:
        """Evaluate model on test set"""
        if self.trainer is None:
            raise ValueError("Must train model first")
        
        results = self.trainer.test(self.model, test_loader)
        return results[0]
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'data_config': self.data_config
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'LMPNowcastTrainer':
        """Load model from checkpoint"""
        checkpoint = torch.load(path)
        
        trainer = cls(
            model_config=checkpoint['model_config'],
            data_config=checkpoint['data_config'],
            training_config={}
        )
        
        trainer.build_model()
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {path}")
        return trainer
