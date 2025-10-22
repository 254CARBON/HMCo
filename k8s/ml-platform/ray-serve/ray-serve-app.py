"""
Ray Serve application for ML model serving with MLflow integration
"""
import os
import logging
from typing import Dict, Any
import numpy as np
import mlflow
from ray import serve
from feast import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class MLflowModelServing:
    """
    Ray Serve deployment that loads models from MLflow and serves predictions
    """
    
    def __init__(self):
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.data-platform.svc.cluster.local:5000")
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        self.models = {}
        self.feature_store = None
        
        # Initialize Feast feature store
        try:
            self.feature_store = FeatureStore(repo_path="/feast/feature_repo")
            logger.info("Feast feature store initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Feast: {e}")
        
        logger.info(f"MLflow Model Serving initialized with URI: {self.mlflow_uri}")
    
    def load_model(self, model_name: str, model_version: str = "latest"):
        """Load a model from MLflow registry"""
        try:
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            self.models[model_name] = model
            logger.info(f"Loaded model: {model_name} version {model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def get_features(self, entity_id: str, feature_view: str):
        """Fetch features from Feast feature store"""
        if not self.feature_store:
            return None
        
        try:
            entity_df = {"entity_id": [entity_id]}
            features = self.feature_store.get_online_features(
                features=[feature_view],
                entity_rows=entity_df
            ).to_dict()
            return features
        except Exception as e:
            logger.error(f"Failed to fetch features: {e}")
            return None
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle prediction requests
        
        Request format:
        {
            "model_name": "commodity_price_predictor",
            "data": {...} or "entity_id": "...",
            "use_feature_store": true/false
        }
        """
        try:
            model_name = request.get("model_name")
            if not model_name:
                return {"error": "model_name is required"}
            
            # Load model if not already loaded
            if model_name not in self.models:
                success = self.load_model(model_name)
                if not success:
                    return {"error": f"Failed to load model {model_name}"}
            
            model = self.models[model_name]
            
            # Get input data
            if request.get("use_feature_store"):
                entity_id = request.get("entity_id")
                feature_view = request.get("feature_view", f"{model_name}_features")
                features = self.get_features(entity_id, feature_view)
                if not features:
                    return {"error": "Failed to fetch features"}
                input_data = features
            else:
                input_data = request.get("data")
            
            if not input_data:
                return {"error": "No input data provided"}
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Format response
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            return {
                "model_name": model_name,
                "prediction": prediction,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e), "status": "failed"}


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.5},
)
class HealthCheck:
    """Health check endpoint"""
    
    async def __call__(self, request):
        return {"status": "healthy", "service": "ray-serve-ml"}


# Create deployments
deployment = MLflowModelServing.bind()
health_deployment = HealthCheck.bind()



