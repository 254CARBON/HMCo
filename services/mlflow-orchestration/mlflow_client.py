"""
MLFlow Client Wrapper for DolphinScheduler Integration

This module provides convenient wrappers for tracking ML experiments within
DolphinScheduler workflows. It simplifies MLFlow API calls and provides
sensible defaults for the 254Carbon data platform.

Usage:
    import sys
    sys.path.insert(0, '/path/to/mlflow-orchestration')
    from mlflow_client import MLFlowClient
    
    client = MLFlowClient(experiment_name="my_experiment")
    client.log_params({"lr": 0.01, "epochs": 100})
    client.log_metrics({"loss": 0.5, "accuracy": 0.95})
    client.log_model(model, "model")
"""

import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient as MLFlowCoreClient

logger = logging.getLogger(__name__)


class MLFlowClient:
    """
    Wrapper around MLFlow tracking API for DolphinScheduler jobs.
    
    Attributes:
        tracking_uri: MLFlow tracking server URI
        experiment_name: Name of the experiment
        run: Current MLFlow run object
        client: Underlying MLFlow client
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize MLFlow client.
        
        Args:
            experiment_name: Name of the experiment to track
            tracking_uri: MLFlow tracking server URI (defaults to MLFLOW_TRACKING_URI env var)
            tags: Additional tags to attach to the run (recommended: source, version, etc.)
        """
        # Get tracking URI from environment or parameter
        self.tracking_uri = tracking_uri or os.environ.get(
            'MLFLOW_TRACKING_URI',
            'http://mlflow.data-platform.svc.cluster.local:5000'
        )
        
        # Set MLFlow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.experiment_name = experiment_name
        self.client = MLFlowCoreClient(tracking_uri=self.tracking_uri)
        self.run = None
        self.tags = tags or {}
        
        # Add default tags for DolphinScheduler context
        self._add_default_tags()
        
        logger.info(
            f"MLFlow client initialized: "
            f"experiment={experiment_name}, uri={self.tracking_uri}"
        )
    
    def _add_default_tags(self):
        """Add default tags from environment variables."""
        # Add DolphinScheduler context if available
        if 'DOLPHINSCHEDULER_PROCESS_ID' in os.environ:
            self.tags['dolphinscheduler_process_id'] = os.environ['DOLPHINSCHEDULER_PROCESS_ID']
        
        if 'DOLPHINSCHEDULER_TASK_NAME' in os.environ:
            self.tags['dolphinscheduler_task'] = os.environ['DOLPHINSCHEDULER_TASK_NAME']
        
        # Add timestamp
        self.tags['created_at'] = datetime.utcnow().isoformat()
        
        # Add source
        self.tags['source'] = 'dolphinscheduler'
    
    def start_run(self, run_name: Optional[str] = None) -> Run:
        """
        Start a new MLFlow run.
        
        Args:
            run_name: Name for the run (optional)
            
        Returns:
            MLFlow Run object
        """
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            experiment = mlflow.get_experiment(experiment_id)
        
        # Start new run
        self.run = mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name
        )
        
        # Set tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        
        logger.info(f"Started MLFlow run: {self.run.info.run_id}")
        return self.run
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLFlow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        if self.run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLFlow run: {self.run.info.run_id}")
        else:
            logger.warning("No active run to end")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (for time-series metrics)
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for key, value in metrics.items():
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
        
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file.
        
        Args:
            local_path: Path to local file or directory
            artifact_path: Optional path within artifact store
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if os.path.isfile(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        elif os.path.isdir(local_path):
            mlflow.log_artifacts(local_path, artifact_path)
        else:
            raise FileNotFoundError(f"Path not found: {local_path}")
        
        logger.info(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model,
        artifact_path: str,
        flavor: str = "sklearn"
    ):
        """
        Log a trained model.
        
        Args:
            model: The model object to log
            artifact_path: Path within artifact store (e.g., "model")
            flavor: MLFlow model flavor (sklearn, tensorflow, pytorch, etc.)
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif flavor == "tensorflow":
            mlflow.tensorflow.log_model(model, artifact_path)
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif flavor == "pyfunc":
            mlflow.pyfunc.log_model(model, artifact_path)
        else:
            raise ValueError(f"Unsupported model flavor: {flavor}")
        
        logger.info(f"Logged model ({flavor}): {artifact_path}")
    
    def set_tag(self, key: str, value: str):
        """
        Set a tag for the current run.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.set_tag(key, value)
    
    def get_run_id(self) -> Optional[str]:
        """
        Get the current run ID.
        
        Returns:
            MLFlow run ID or None if no active run
        """
        return self.run.info.run_id if self.run else None
    
    def get_experiment_id(self) -> Optional[str]:
        """
        Get the experiment ID.
        
        Returns:
            MLFlow experiment ID or None if experiment doesn't exist
        """
        if not self.run:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            return experiment.experiment_id if experiment else None
        return self.run.info.experiment_id


def setup_mlflow_for_dolphinscheduler(
    experiment_name: str,
    tags: Optional[Dict[str, str]] = None
) -> MLFlowClient:
    """
    Convenience function for setting up MLFlow in DolphinScheduler tasks.
    
    This function should be called at the start of a DolphinScheduler Python task.
    
    Args:
        experiment_name: Name of the experiment
        tags: Additional tags for the run
        
    Returns:
        Initialized MLFlowClient
        
    Example:
        client = setup_mlflow_for_dolphinscheduler(
            experiment_name="iris_classification",
            tags={
                "dataset": "iris",
                "model_type": "random_forest"
            }
        )
        client.start_run("my_run")
        client.log_params({"n_estimators": 100})
        client.log_metrics({"accuracy": 0.95})
        client.end_run()
    """
    return MLFlowClient(
        experiment_name=experiment_name,
        tags=tags
    )
