# MLFlow Integration Guide

This document covers integrating MLFlow with DolphinScheduler for ML workflow orchestration and DataHub for metadata governance.

## Overview

MLFlow integrates into the 254Carbon data platform as:

1. **Experiment Tracking Server**: Deployed in Kubernetes with PostgreSQL backend and MinIO artifact storage
2. **DolphinScheduler Integration**: Python SDK for logging ML experiments in orchestrated jobs
3. **DataHub Integration**: Automated metadata ingestion of models and experiments

## Part 1: DolphinScheduler Integration

### Architecture

```
DolphinScheduler Job
      ↓
MLFlow Python Client (mlflow_client.py)
      ↓
MLFlow Tracking Server (HTTP)
      ↓
PostgreSQL (metadata) + MinIO (artifacts)
      ↓
MLFlow UI & Registry
```

### Setup in DolphinScheduler

#### Step 1: Add MLFlow Dependencies to Workers

Ensure all DolphinScheduler worker nodes have MLFlow installed:

```bash
# On each worker node
pip install mlflow>=2.10.0 boto3>=1.26.0

# Or add to requirements file
echo "mlflow>=2.10.0" >> /opt/dolphinscheduler/requirements.txt
echo "boto3>=1.26.0" >> /opt/dolphinscheduler/requirements.txt
pip install -r /opt/dolphinscheduler/requirements.txt
```

#### Step 2: Copy MLFlow Integration Package

Copy the integration package to a shared location:

```bash
# Option A: Shared NFS mount
cp -r services/mlflow-orchestration /mnt/shared/mlflow-orchestration

# Option B: Bake into worker image
COPY services/mlflow-orchestration /opt/mlflow-orchestration
```

#### Step 3: Create DolphinScheduler Python Task

In DolphinScheduler UI, create a new Python task:

```python
#!/usr/bin/env python3
"""
Example: Train ML Model in DolphinScheduler
"""

import sys
import os

# Add MLFlow integration to path
sys.path.insert(0, '/mnt/shared/mlflow-orchestration')
# OR: sys.path.insert(0, '/opt/mlflow-orchestration')

from mlflow_client import setup_mlflow_for_dolphinscheduler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main task function."""
    
    # Initialize MLFlow client
    client = setup_mlflow_for_dolphinscheduler(
        experiment_name="model_training",
        tags={
            "task": "iris_classification",
            "environment": "production"
        }
    )
    
    try:
        # Start tracking
        client.start_run("iris_v1")
        logger.info("MLFlow run started")
        
        # Your training code
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log parameters
        params = {"n_estimators": 100, "max_depth": 10}
        client.log_params(params)
        
        # Train and evaluate
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        
        # Log metrics
        client.log_metrics({"accuracy": score})
        
        # Log model
        client.log_model(model, "model", flavor="sklearn")
        
        # Mark as complete
        client.end_run("FINISHED")
        logger.info("MLFlow run completed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        client.end_run("FAILED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

### Accessing Tracked Experiments

#### Via MLFlow UI

1. Navigate to `https://mlflow.254carbon.com`
2. Authenticate with Cloudflare Access
3. Select experiment from the list
4. View runs, parameters, metrics, and artifacts

#### Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://mlflow.data-platform.svc.cluster.local:5000")

# List experiments
experiments = client.search_experiments()

# Search runs
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.accuracy > 0.9"
)

# Get run details
run = client.get_run(run_id)
print(f"Parameters: {run.data.params}")
print(f"Metrics: {run.data.metrics}")
```

## Part 2: DataHub Integration

### Architecture

```
MLFlow Models (PostgreSQL + MinIO)
      ↓
DataHub Ingestion Recipe
      ↓
DataHub GMS
      ↓
DataHub UI (Metadata Catalog)
```

### Setup: MLFlow Ingestion Recipe

#### Step 1: Create Recipe Configuration

Create `k8s/datahub/mlflow-ingestion-recipe.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-ingestion-recipe
  namespace: data-platform
data:
  recipe.yml: |
    source:
      type: mlflow
      config:
        tracking_uri: http://mlflow.data-platform.svc.cluster.local:5000
        # Optional: Specific experiments to ingest
        experiments:
          - name: "model_training"
          - name: "experiment_*"
        # Include run details and artifacts
        include_run_details: true
        include_artifacts: true

    sink:
      type: datahub-rest
      config:
        server: http://datahub-gms.data-platform.svc.cluster.local:8080

    transformers:
      - type: add_dataset_properties
        config:
          properties:
            classification: "ML_MODEL"
            domain: "machine-learning"
      
      - type: mark_dataset_status
        config:
          status: ACTIVE
```

#### Step 2: Deploy DataHub Ingestion Job

```bash
kubectl apply -f k8s/datahub/mlflow-ingestion-recipe.yaml

# Create CronJob for periodic ingestion
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mlflow-datahub-ingestion
  namespace: data-platform
spec:
  schedule: "0 * * * *"  # Every hour
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: default
          containers:
          - name: datahub-ingestion
            image: acryldata/datahub-ingestion:0.12.0
            env:
            - name: DATAHUB_GMS_URL
              value: http://datahub-gms.data-platform.svc.cluster.local:8080
            volumeMounts:
            - name: recipe
              mountPath: /etc/datahub
            command:
            - datahub
            - ingest
            - -c
            - /etc/datahub/recipe.yml
          volumes:
          - name: recipe
            configMap:
              name: mlflow-ingestion-recipe
          restartPolicy: OnFailure
EOF
```

#### Step 3: Verify Ingestion

```bash
# Check job logs
kubectl logs -n data-platform -l job-name=mlflow-datahub-ingestion

# Access DataHub UI
# Navigate to: https://datahub.254carbon.com
# Search for MLFlow models and experiments
```

### Model Lineage in DataHub

Configure MLFlow to track lineage automatically:

```python
# In MLFlow tracking code
import mlflow
from mlflow.entities import Dataset

# Log input dataset
dataset = Dataset(
    source="postgresql://...",
    schema="features",
    name="iris_features"
)
mlflow.log_input(dataset, context="training")

# Train model...

# Output model will be tracked with lineage
```

### DataHub Integration Benefits

- **Model Governance**: Centralized registry of all ML models
- **Lineage Tracking**: Understand data → feature → model relationships
- **Ownership**: Assign ownership and tags to models
- **Discovery**: Data scientists can find existing models
- **Quality Metrics**: Track model performance over time

## Part 3: End-to-End ML Pipeline Example

### Complete Workflow

```python
#!/usr/bin/env python3
"""
Complete ML Pipeline with MLFlow Tracking and DolphinScheduler
"""

import sys
sys.path.insert(0, '/mnt/shared/mlflow-orchestration')

from mlflow_client import setup_mlflow_for_dolphinscheduler
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load training data."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model_type, params):
    """Train model with specified parameters."""
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1": f1_score(y_test, predictions, average="weighted"),
    }


def main():
    """Main pipeline."""
    
    # Initialize MLFlow
    client = setup_mlflow_for_dolphinscheduler(
        experiment_name="iris_comparison",
        tags={"stage": "model_selection"}
    )
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        logger.info(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
        
        # Model configurations to try
        models_to_train = [
            {
                "type": "random_forest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            },
            {
                "type": "gradient_boosting",
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            }
        ]
        
        best_score = 0
        best_run = None
        
        # Train and evaluate each model
        for model_config in models_to_train:
            model_type = model_config["type"]
            params = model_config["params"]
            
            # Start tracking run
            run_name = f"{model_type}_v1"
            client.start_run(run_name)
            logger.info(f"Training {model_type}...")
            
            try:
                # Log parameters
                client.log_params(params)
                
                # Train model
                model = train_model(model_type, params)
                model.fit(X_train, y_train)
                logger.info(f"Model trained: {model_type}")
                
                # Evaluate
                metrics = evaluate_model(model, X_test, y_test)
                client.log_metrics(metrics)
                logger.info(f"Metrics: {metrics}")
                
                # Log model
                client.log_model(model, "model", flavor="sklearn")
                logger.info("Model logged to MLFlow")
                
                # Track best model
                if metrics["accuracy"] > best_score:
                    best_score = metrics["accuracy"]
                    best_run = {
                        "model_type": model_type,
                        "accuracy": metrics["accuracy"],
                        "run_id": client.get_run_id()
                    }
                
                client.set_tag("status", "success")
                client.end_run("FINISHED")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}", exc_info=True)
                client.set_tag("status", "failed")
                client.end_run("FAILED")
        
        # Log summary
        logger.info(f"Best model: {best_run['model_type']} with accuracy {best_run['accuracy']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

## Part 4: Best Practices

### Experiment Management

1. **Naming Convention**: Use descriptive experiment names
   - ✅ `iris_classification_v1`
   - ❌ `test_123`

2. **Tagging**: Always tag runs with context
   ```python
   tags = {
       "task": "classification",
       "dataset": "iris",
       "model_type": "random_forest",
       "environment": "dev"
   }
   ```

3. **Run Names**: Include model version and timestamp
   - ✅ `rf_v1_2024_01_15`
   - ❌ `run1`

### Parameter Logging

1. **Log all hyperparameters**, even defaults
2. **Use consistent naming**: `learning_rate`, not `lr` or `learn_rate`
3. **Document parameter meanings** in experiment tags

### Metric Logging

1. **Log multiple evaluation metrics**: accuracy, precision, recall, F1
2. **Include data split sizes** for reproducibility
3. **Log time-series metrics** for training progress
4. **Add context tags** for interpretation

### Model Versioning

1. **Always log models** to artifact store
2. **Use consistent artifact paths**: `model/`, `preprocessing/`
3. **Log model metadata**: input schema, output format
4. **Test model reproducibility**: load and predict from artifact

### Production Deployment

1. **Version models systematically**: v1.0, v1.1, v2.0
2. **Track deployment status** with tags
3. **Log A/B test results** with meaningful metrics
4. **Monitor model drift** and retrain triggers

## Troubleshooting

### MLFlow Server Not Accessible

```bash
# Check if pod is running
kubectl get pods -n data-platform -l app=mlflow

# Check logs
kubectl logs -n data-platform -l app=mlflow

# Test connection
kubectl port-forward -n data-platform svc/mlflow 5000:5000
curl http://localhost:5000/health
```

### DolphinScheduler Task Fails to Log

```bash
# Check environment variables
echo $MLFLOW_TRACKING_URI

# Verify connectivity from worker
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Check worker logs
# In DolphinScheduler UI: View logs for failed task
```

### DataHub Ingestion Fails

```bash
# Check ingestion job logs
kubectl logs -n data-platform -l job-name=mlflow-datahub-ingestion

# Verify DataHub GMS is reachable
kubectl exec -n data-platform mlflow-<pod> -- \
  curl http://datahub-gms.data-platform.svc.cluster.local:8080

# Check recipe configuration
kubectl get configmap mlflow-ingestion-recipe -n data-platform -o yaml
```

## Resources

- [MLFlow Documentation](https://mlflow.org/docs/latest/)
- [MLFlow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [DolphinScheduler Python Task](https://dolphinscheduler.apache.org/docs/2.1.0/user_doc/guide/task/python.html)
- [DataHub Ingestion](https://datahubproject.io/docs/generated/ingestion/)
