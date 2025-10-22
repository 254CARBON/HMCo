#!/usr/bin/env python3
"""
Simple MLflow Tracking Example
Tests MLflow deployment with a basic scikit-learn model
"""

import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set tracking URI
MLFLOW_TRACKING_URI = os.environ.get(
    'MLFLOW_TRACKING_URI',
    'http://mlflow.data-platform.svc.cluster.local:5000'
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print("="*50)

# Set experiment name
experiment_name = "iris_classification_test"
mlflow.set_experiment(experiment_name)

print(f"Experiment: {experiment_name}")
print("="*50)

# Load data
print("\nLoading Iris dataset...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Start MLflow run
with mlflow.start_run(run_name="random_forest_baseline") as run:
    print(f"\n{' Starting MLflow Run ':=^50}")
    print(f"Run ID: {run.info.run_id}")
    
    # Define hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42
    }
    
    # Log parameters
    print("\nLogging parameters...")
    mlflow.log_params(params)
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "test_samples": len(y_test)
    }
    
    # Log metrics
    print("\nLogging metrics...")
    mlflow.log_metrics(metrics)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Log model
    print("\nLogging model...")
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="iris_random_forest"
    )
    print("✓ Model logged")
    
    # Log tags
    mlflow.set_tags({
        "dataset": "iris",
        "model_type": "random_forest",
        "framework": "scikit-learn",
        "test": "deployment_verification"
    })
    
    print(f"\n{' Run Complete ':=^50}")
    print(f"\nView results at:")
    print(f"  {MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

print("\n" + "="*50)
print("✓ MLflow tracking test completed successfully!")
print("="*50)

sys.exit(0)


