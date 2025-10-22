# MLFlow Orchestration Integration

**MLflow Status**: ✅ Deployed and Operational  
**Access**: http://mlflow.data-platform.svc.cluster.local:5000 | https://mlflow.254carbon.com

Python utilities for integrating MLFlow with DolphinScheduler workflows in the 254Carbon data platform.

## Overview

This package provides a simplified Python client for tracking ML experiments within DolphinScheduler jobs. It handles:

- Experiment and run lifecycle management
- Parameter and metric logging
- Model and artifact versioning
- Automatic tagging with DolphinScheduler context
- Error handling and logging

## Installation

### In DolphinScheduler Python Tasks

1. **Option A: Install from PyPI (if packaged)**
```bash
pip install mlflow-orchestration
```

2. **Option B: Local import in DolphinScheduler task**
```python
import sys
sys.path.insert(0, '/path/to/mlflow-orchestration')
from mlflow_client import setup_mlflow_for_dolphinscheduler
```

### Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from mlflow_client import setup_mlflow_for_dolphinscheduler

# Setup MLFlow client
client = setup_mlflow_for_dolphinscheduler(
    experiment_name="iris_classification",
    tags={
        "dataset": "iris",
        "model_type": "random_forest"
    }
)

# Start a tracked run
client.start_run("baseline_v1")

# Log hyperparameters
client.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
})

# Train model and log metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Log metrics
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
client.log_metrics({"accuracy": accuracy, "test_samples": len(y_test)})

# Log model
client.log_model(model, "model", flavor="sklearn")

# Finish run
client.end_run()
```

### DolphinScheduler Task Example

Create a Python task in DolphinScheduler:

```python
#!/usr/bin/env python3
"""
DolphinScheduler Task: Train ML Model and Track with MLFlow
"""

import sys
sys.path.insert(0, '/home/dolphinscheduler/mlflow-orchestration')

from mlflow_client import setup_mlflow_for_dolphinscheduler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """Main training function."""
    
    # Initialize MLFlow tracking
    client = setup_mlflow_for_dolphinscheduler(
        experiment_name="model_training",
        tags={
            "task": "train_iris_model",
            "version": "v1.0"
        }
    )
    
    try:
        # Start tracking run
        client.start_run("iris_model_training")
        logger.info("MLFlow run started")
        
        # Your training code here
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42
        }
        client.log_params(params)
        logger.info(f"Logged parameters: {params}")
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Evaluate and log metrics
        predictions = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted"),
            "recall": recall_score(y_test, predictions, average="weighted"),
            "f1": f1_score(y_test, predictions, average="weighted"),
            "test_samples": len(y_test)
        }
        client.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")
        
        # Log model
        client.log_model(model, "model", flavor="sklearn")
        logger.info("Model logged to MLFlow artifact store")
        
        # Set final tag
        client.set_tag("status", "success")
        
        # Complete run
        client.end_run("FINISHED")
        logger.info("MLFlow run completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        client.end_run("FAILED")
        return 1

if __name__ == "__main__":
    exit_code = train_model()
    sys.exit(exit_code)
```

## API Reference

### MLFlowClient

#### Constructor

```python
MLFlowClient(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
)
```

**Parameters:**
- `experiment_name`: Name of the experiment (required)
- `tracking_uri`: MLFlow tracking server URI (defaults to `MLFLOW_TRACKING_URI` env var or cluster default)
- `tags`: Additional tags to attach to all runs

**Example:**
```python
client = MLFlowClient(
    experiment_name="my_experiment",
    tracking_uri="http://mlflow.data-platform.svc.cluster.local:5000",
    tags={"project": "254carbon", "team": "data-science"}
)
```

#### Methods

##### start_run(run_name: Optional[str] = None) → Run

Start a new tracking run.

```python
run = client.start_run("my_run_v1")
```

##### end_run(status: str = "FINISHED")

End the current tracking run.

```python
client.end_run("FINISHED")  # Success
client.end_run("FAILED")    # Failed
client.end_run("KILLED")    # Killed
```

##### log_params(params: Dict[str, Any])

Log hyperparameters.

```python
client.log_params({
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 32
})
```

##### log_metrics(metrics: Dict[str, float], step: Optional[int] = None)

Log metrics.

```python
# Single step
client.log_metrics({"accuracy": 0.95, "loss": 0.05})

# Time-series metrics
for epoch in range(10):
    client.log_metrics({"train_loss": loss}, step=epoch)
```

##### log_artifact(local_path: str, artifact_path: Optional[str] = None)

Log artifact file or directory.

```python
# Log single file
client.log_artifact("/tmp/model.pkl", "artifacts")

# Log directory
client.log_artifact("/tmp/plots", "visualizations")
```

##### log_model(model, artifact_path: str, flavor: str = "sklearn")

Log a trained model.

```python
# Scikit-learn
client.log_model(model, "model", flavor="sklearn")

# TensorFlow
client.log_model(model, "model", flavor="tensorflow")

# PyTorch
client.log_model(model, "model", flavor="pytorch")
```

Supported flavors: `sklearn`, `tensorflow`, `pytorch`, `pyfunc`

##### set_tag(key: str, value: str)

Set a tag on the current run.

```python
client.set_tag("deployment_status", "ready_for_production")
```

##### get_run_id() → Optional[str]

Get current run ID.

```python
run_id = client.get_run_id()
print(f"Run ID: {run_id}")
```

##### get_experiment_id() → Optional[str]

Get experiment ID.

```python
exp_id = client.get_experiment_id()
print(f"Experiment ID: {exp_id}")
```

### Helper Functions

#### setup_mlflow_for_dolphinscheduler(experiment_name: str, tags: Optional[Dict[str, str]] = None) → MLFlowClient

Convenience function for DolphinScheduler tasks.

```python
client = setup_mlflow_for_dolphinscheduler(
    experiment_name="my_experiment",
    tags={"task": "training"}
)
```

## Environment Variables

### Required

- `MLFLOW_TRACKING_URI`: MLFlow tracking server URI
  - Default: `http://mlflow.data-platform.svc.cluster.local:5000`
  - Example: `http://mlflow:5000` or `postgresql://...`

### Optional (from DolphinScheduler)

- `DOLPHINSCHEDULER_PROCESS_ID`: Process ID (auto-tagged)
- `DOLPHINSCHEDULER_TASK_NAME`: Task name (auto-tagged)

### S3/MinIO (auto-injected)

- `AWS_ACCESS_KEY_ID`: MinIO access key
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key
- `MLFLOW_S3_ENDPOINT_URL`: MinIO endpoint

## Integration with DataHub

Models logged to MLFlow can be automatically ingested into DataHub for governance. See `docs/mlflow/integration-guide.md` for setup instructions.

## Troubleshooting

### Connection Refused

**Error**: `ConnectionError: Unable to connect to MLFlow server`

**Solution**:
1. Verify MLFlow pod is running: `kubectl get pods -n data-platform -l app=mlflow`
2. Check tracking URI: `echo $MLFLOW_TRACKING_URI`
3. Test connectivity: `curl http://mlflow.data-platform.svc.cluster.local:5000/health`

### Authentication Failed

**Error**: `401 Unauthorized` from MinIO

**Solution**:
1. Verify MinIO credentials in pod: `kubectl exec -it <mlflow-pod> -- env | grep AWS`
2. Check MinIO service: `kubectl get svc -n data-platform | grep minio`
3. Verify S3 endpoint: `echo $MLFLOW_S3_ENDPOINT_URL`

### Model Upload Failed

**Error**: `S3 upload error` or `bucket not found`

**Solution**:
1. Create bucket: `mc mb local/mlflow-artifacts`
2. Enable versioning: `mc version enable local/mlflow-artifacts`
3. Check credentials and permissions

## Examples

See `examples/` directory for complete working examples:
- `simple_classification.py` - Basic scikit-learn classification
- `time_series_metrics.py` - Time-series metric logging
- `model_comparison.py` - Comparing multiple models
- `advanced_tracking.py` - Advanced artifact and model logging

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black mlflow_client.py

# Lint
pylint mlflow_client.py
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review MLFlow documentation: https://mlflow.org/docs
3. Check cluster logs: `kubectl logs -n data-platform -l app=mlflow`
