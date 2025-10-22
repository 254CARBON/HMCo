"""
MLFlow integration client for Spark jobs in 254Carbon platform.

This module provides utilities for tracking Spark job execution in MLFlow,
including automatic tagging with job context, parameter logging, and metric export.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.pyfunc import PythonModel

logger = logging.getLogger(__name__)


class SparkMLFlowClient:
    """Client for integrating Spark jobs with MLFlow tracking server."""
    
    def __init__(self, 
                 tracking_uri: str = None,
                 experiment_name: str = "spark-jobs",
                 registry_uri: str = None):
        """
        Initialize SparkMLFlowClient.
        
        Args:
            tracking_uri: MLFlow tracking server URI (defaults to env var)
            experiment_name: Name of the MLFlow experiment
            registry_uri: MLFlow model registry URI (optional)
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", 
            "http://mlflow-service:5000"
        )
        self.registry_uri = registry_uri or os.getenv(
            "MLFLOW_REGISTRY_URI",
            "http://mlflow-service:5000"
        )
        self.experiment_name = experiment_name
        self.run_id = None
        
        # Set MLFlow URIs
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        # Create/get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
        
        logger.info(f"MLFlow Client initialized: {self.tracking_uri}, "
                   f"Experiment: {experiment_name}")
    
    def start_spark_run(self,
                       job_name: str,
                       job_id: str = None,
                       tags: Dict[str, str] = None,
                       params: Dict[str, Any] = None) -> str:
        """
        Start a new MLFlow run for a Spark job.
        
        Args:
            job_name: Human-readable job name
            job_id: Unique job identifier (auto-generated if not provided)
            tags: Dictionary of tags to add to run
            params: Dictionary of parameters to log
            
        Returns:
            MLFlow run ID
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        # Start MLFlow run
        run = mlflow.start_run(experiment_id=self.experiment_id)
        self.run_id = run.info.run_id
        
        # Add base tags
        base_tags = {
            "spark_job": job_name,
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "platform": "254carbon",
            "framework": "spark"
        }
        
        # Merge with provided tags
        if tags:
            base_tags.update(tags)
        
        # Log tags
        for tag_name, tag_value in base_tags.items():
            mlflow.set_tag(tag_name, str(tag_value))
        
        # Log parameters
        if params:
            for param_name, param_value in params.items():
                try:
                    mlflow.log_param(param_name, str(param_value))
                except Exception as e:
                    logger.warning(f"Failed to log param {param_name}: {e}")
        
        logger.info(f"Started MLFlow run for {job_name}: {self.run_id}")
        return self.run_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """
        Log metrics from Spark job.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Step/iteration number (for time-series metrics)
        """
        if not self.run_id:
            logger.warning("No active MLFlow run, cannot log metrics")
            return
        
        for metric_name, metric_value in metrics.items():
            try:
                mlflow.log_metric(metric_name, float(metric_value), step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {metric_name}: {e}")
    
    def log_quality_metrics(self, 
                          quality_results: Dict[str, Any],
                          table_name: str):
        """
        Log Deequ quality check results.
        
        Args:
            quality_results: Dictionary of quality check results
            table_name: Name of table being checked
        """
        if not self.run_id:
            logger.warning("No active MLFlow run, cannot log quality metrics")
            return
        
        # Extract quality score
        quality_score = quality_results.get("quality_score", 0)
        mlflow.log_metric("quality_score", quality_score)
        
        # Log individual check results
        for check_name, check_result in quality_results.get("checks", {}).items():
            metric_name = f"quality_check_{table_name}_{check_name}"
            if isinstance(check_result, (int, float)):
                mlflow.log_metric(metric_name, check_result)
        
        # Log quality check summary
        mlflow.log_dict(quality_results, "quality_results.json")
        
        logger.info(f"Logged quality metrics for {table_name}")
    
    def log_data_quality_metadata(self,
                                 table_name: str,
                                 record_count: int,
                                 schema: Dict[str, str]):
        """
        Log data quality metadata.
        
        Args:
            table_name: Name of the table
            record_count: Number of records in table
            schema: Column schema as dictionary
        """
        if not self.run_id:
            return
        
        mlflow.log_param("table_name", table_name)
        mlflow.log_metric("record_count", record_count)
        mlflow.log_dict(schema, "table_schema.json")
    
    def log_spark_config(self, spark_config: Dict[str, str]):
        """
        Log Spark configuration parameters.
        
        Args:
            spark_config: Spark configuration dictionary
        """
        if not self.run_id:
            return
        
        # Log selected key configs
        key_configs = [
            "spark.sql.adaptive.enabled",
            "spark.sql.shuffle.partitions",
            "spark.serializer",
            "spark.sql.catalog.iceberg.type",
            "spark.hadoop.fs.s3a.endpoint"
        ]
        
        for config_key in key_configs:
            if config_key in spark_config:
                mlflow.log_param(config_key, str(spark_config[config_key]))
        
        # Log full config as artifact
        mlflow.log_dict(spark_config, "spark_config.json")
    
    def log_artifacts(self, artifact_path: str, artifact_type: str = "output"):
        """
        Log artifacts from Spark job.
        
        Args:
            artifact_path: Path to artifact file or directory
            artifact_type: Type of artifact (output, logs, model, etc)
        """
        if not self.run_id:
            logger.warning("No active MLFlow run, cannot log artifacts")
            return
        
        try:
            if os.path.isfile(artifact_path):
                mlflow.log_artifact(artifact_path, artifact_type)
            elif os.path.isdir(artifact_path):
                mlflow.log_artifacts(artifact_path, artifact_type)
            else:
                logger.warning(f"Artifact path not found: {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {artifact_path}: {e}")
    
    def end_run(self, status: str = "FINISHED", error_message: str = None):
        """
        End the current MLFlow run.
        
        Args:
            status: Run status (FINISHED, FAILED)
            error_message: Error message if run failed
        """
        if not self.run_id:
            return
        
        if error_message:
            mlflow.set_tag("error", error_message)
            mlflow.set_tag("status", status or "FAILED")
        else:
            mlflow.set_tag("status", status or "FINISHED")
        
        mlflow.end_run()
        logger.info(f"Ended MLFlow run: {self.run_id}")
        self.run_id = None
    
    def log_model(self,
                 model_path: Union[str, PythonModel],
                 model_flavor: str = "pyfunc",
                 registered_model_name: str = None):
        """
        Log trained model to MLFlow.
        
        Args:
            model_path: Path to model directory or PythonModel instance
            model_flavor: MLFlow model flavor (pyfunc, sklearn, etc)
            registered_model_name: Register model under this name
        """
        if not self.run_id:
            logger.warning("No active MLFlow run, cannot log model")
            return
        
        try:
            if model_flavor == "pyfunc":
                if isinstance(model_path, PythonModel):
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=model_path
                    )
                else:
                    if not isinstance(model_path, str):
                        raise ValueError(
                            "For pyfunc flavor provide a path to a saved model "
                            "directory or an instance of mlflow.pyfunc.PythonModel"
                        )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Pyfunc model path not found: {model_path}"
                        )
                    if not os.path.isdir(model_path):
                        raise ValueError(
                            "Pyfunc model path must be a directory created with "
                            "mlflow.pyfunc.save_model"
                        )
                    mlflow.log_artifacts(model_path, "model")
            elif model_flavor == "sklearn":
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                mlflow.sklearn.log_model(model, "model")
            
            if registered_model_name:
                mlflow.register_model(
                    f"runs:/{self.run_id}/model",
                    registered_model_name
                )
                logger.info(f"Registered model: {registered_model_name}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def get_run_info(self) -> Dict[str, Any]:
        """Get information about the current run."""
        if not self.run_id:
            return {}
        
        try:
            run = mlflow.get_run(self.run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return {}


def setup_mlflow_for_spark_job(job_name: str, 
                               job_id: str = None,
                               experiment_name: str = "spark-jobs",
                               tracking_uri: str = None) -> SparkMLFlowClient:
    """
    Convenience function to set up MLFlow for a Spark job.
    
    Args:
        job_name: Name of the Spark job
        job_id: Unique job identifier
        experiment_name: MLFlow experiment name
        tracking_uri: MLFlow tracking server URI
        
    Returns:
        Initialized SparkMLFlowClient
    """
    client = SparkMLFlowClient(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    client.start_spark_run(job_name, job_id)
    return client


if __name__ == "__main__":
    # Example usage
    client = SparkMLFlowClient(experiment_name="spark-examples")
    
    # Start a run
    run_id = client.start_spark_run(
        "example-etl-job",
        tags={"job_type": "etl", "data_source": "kafka"},
        params={"parallelism": 200, "shuffle_partitions": 200}
    )
    
    # Log metrics
    client.log_metrics({
        "processing_time_seconds": 123.45,
        "records_processed": 1000000,
        "throughput_rps": 8108
    })
    
    # End run
    client.end_run(status="FINISHED")
    
    print(f"MLFlow run created: {run_id}")
