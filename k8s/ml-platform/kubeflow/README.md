# Kubeflow Pipelines & ML Training Platform

**Platform**: 254Carbon Advanced Analytics Platform  
**Component**: ML Orchestration & Training  
**Technology**: Kubeflow 2.0, Katib 0.16, Training Operators  
**Status**: Implementation Phase 2

---

## Overview

Kubeflow provides a comprehensive ML platform with:

- **Kubeflow Pipelines**: ML workflow orchestration
- **Katib**: Hyperparameter tuning and AutoML
- **Training Operators**: Distributed training for PyTorch and TensorFlow
- **Experiments**: Experiment tracking and comparison
- **Model Registry**: Integration with MLflow

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  ML Pipeline UI (Web Interface)                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Kubeflow Pipelines API Server                               │
│  - Pipeline Management                                       │
│  - Workflow Execution (Argo Workflows)                       │
└──────┬──────────────────────────────────────┬────────────────┘
       │                                       │
       │ Submit                                │ Monitor
       ↓                                       ↓
┌──────────────────┐                ┌───────────────────────────┐
│  Training Jobs   │                │  Katib Experiments        │
│  - PyTorchJob    │                │  - Hyperparameter Tuning  │
│  - TFJob         │                │  - AutoML                 │
└────────┬─────────┘                └───────────────────────────┘
         │
         │ Log Results
         ↓
┌────────────────────────────────────────────────────────────────┐
│  ML Metadata (MySQL) + Artifacts (MinIO)                       │
└────────────────────────────────────────────────────────────────┘
         │
         │ Track
         ↓
┌────────────────────────────────────────────────────────────────┐
│  MLflow (Model Registry & Tracking)                            │
└────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Kubeflow Pipelines

Orchestrates ML workflows with versioned, reproducible pipelines.

**Features**:
- Visual pipeline designer
- Reusable pipeline components
- Experiment tracking
- Scheduled and triggered runs
- Integration with MLflow

### 2. Katib

AutoML platform for hyperparameter tuning.

**Algorithms**:
- Random Search
- Grid Search
- Bayesian Optimization
- Hyperband
- TPE (Tree-structured Parzen Estimator)
- CMA-ES

### 3. Training Operators

Distributed training for deep learning frameworks.

**Supported**:
- **PyTorch**: Distributed training with torch.distributed
- **TensorFlow**: Parameter servers and distributed strategies

### 4. ML Metadata

Stores pipeline execution metadata in MySQL.

**Tracks**:
- Pipeline runs
- Artifacts
- Metrics
- Parameters

## Deployment

### Prerequisites

```bash
# Ensure MinIO is running
kubectl get pods -n data-platform -l app=minio

# Ensure MLflow is running
kubectl get pods -n data-platform -l app=mlflow

# Create MinIO bucket for pipelines
kubectl exec -n data-platform -it deploy/minio -- \
  mc mb local/mlpipelines
```

### Deploy Kubeflow

```bash
# 1. Create namespace and RBAC
kubectl apply -f k8s/ml-platform/kubeflow/namespace.yaml

# 2. Deploy Kubeflow Pipelines
kubectl apply -f k8s/ml-platform/kubeflow/kubeflow-pipelines.yaml

# 3. Wait for pipelines to be ready
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=600s

# 4. Deploy Katib
kubectl apply -f k8s/ml-platform/kubeflow/katib.yaml

# 5. Deploy Training Operators
kubectl apply -f k8s/ml-platform/training-operators/pytorch-operator.yaml
kubectl apply -f k8s/ml-platform/training-operators/tensorflow-operator.yaml

# 6. Verify deployment
kubectl get pods -n kubeflow
```

### Verify Installation

```bash
# Check all components
kubectl get pods -n kubeflow

# Port-forward to UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Access at http://localhost:8080

# Check Katib UI
kubectl port-forward -n kubeflow svc/katib-ui 8081:80
```

## Usage

### Create a Simple Pipeline

```python
from kfp import dsl, compiler

@dsl.component
def train_model(dataset: str) -> str:
    """Train a commodity price prediction model"""
    import mlflow
    from sklearn.ensemble import RandomForestRegressor
    
    mlflow.set_tracking_uri("http://mlflow.data-platform.svc.cluster.local:5000")
    
    with mlflow.start_run():
        # Training code here
        model = RandomForestRegressor()
        # ... train model ...
        
        mlflow.sklearn.log_model(model, "model")
        return "model_trained"

@dsl.component
def evaluate_model(model_path: str) -> float:
    """Evaluate the trained model"""
    # Evaluation code
    return 0.95

@dsl.pipeline(
    name="Commodity Price Prediction Pipeline",
    description="Train and evaluate commodity price prediction model"
)
def commodity_price_pipeline(dataset: str = "commodity_prices"):
    train_task = train_model(dataset=dataset)
    eval_task = evaluate_model(model_path=train_task.output)

# Compile pipeline
compiler.Compiler().compile(
    commodity_price_pipeline,
    'commodity_price_pipeline.yaml'
)
```

### Upload and Run Pipeline

```bash
# Upload pipeline
kfp pipeline upload \
  --pipeline-name "Commodity Price Prediction" \
  commodity_price_pipeline.yaml

# Create a run
kfp run submit \
  --experiment-name "Commodity Experiments" \
  --pipeline-name "Commodity Price Prediction" \
  --run-name "initial-training"
```

### Hyperparameter Tuning with Katib

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: commodity-price-hp-tuning
  namespace: kubeflow
spec:
  algorithm:
    algorithmName: tpe
  parallelTrialCount: 3
  maxTrialCount: 12
  maxFailedTrialCount: 3
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  parameters:
  - name: learning_rate
    parameterType: double
    feasibleSpace:
      min: "0.001"
      max: "0.1"
  - name: num_estimators
    parameterType: int
    feasibleSpace:
      min: "10"
      max: "200"
  trialTemplate:
    primaryContainerName: training
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
            - name: training
              image: python:3.10
              command:
              - python
              - train.py
              - --learning-rate=${trialParameters.learningRate}
              - --num-estimators=${trialParameters.numEstimators}
            restartPolicy: Never
```

Apply the experiment:

```bash
kubectl apply -f commodity-hp-tuning.yaml

# Monitor progress
kubectl get experiments -n kubeflow
kubectl describe experiment commodity-price-hp-tuning -n kubeflow
```

### Distributed PyTorch Training

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed-training
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
            command:
            - python
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
            command:
            - python
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 1
```

```bash
kubectl apply -f pytorch-job.yaml
kubectl get pytorchjobs -n kubeflow
kubectl logs -n kubeflow pytorch-distributed-training-master-0
```

### TensorFlow Distributed Training

```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: tensorflow-distributed
  namespace: kubeflow
spec:
  tfReplicaSpecs:
    PS:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
            - python
            - train.py
            resources:
              limits:
                memory: 4Gi
                cpu: 2
    Worker:
      replicas: 4
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
            - python
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: 8Gi
                cpu: 4
    Chief:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
            - python
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 1
```

## Integration with DolphinScheduler

Create a DolphinScheduler task to trigger Kubeflow pipelines:

```python
from dolphinscheduler import Task

class KubeflowPipelineTask(Task):
    def execute(self, context):
        import kfp
        
        client = kfp.Client(
            host="http://ml-pipeline.kubeflow.svc.cluster.local:8888"
        )
        
        run = client.create_run_from_pipeline_func(
            commodity_price_pipeline,
            arguments={"dataset": "commodity_prices"}
        )
        
        # Wait for completion
        client.wait_for_run_completion(run.run_id, timeout=3600)
        
        return run.run_id
```

## Monitoring

### Pipeline Metrics

```bash
# View pipeline runs
kfp run list

# Get run details
kfp run get <run-id>

# View metrics
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
curl http://localhost:8888/apis/v1beta1/runs
```

### Katib Experiments

```bash
# List experiments
kubectl get experiments -n kubeflow

# Get experiment status
kubectl describe experiment <experiment-name> -n kubeflow

# View best trial
kubectl get trials -n kubeflow -l experiment=<experiment-name> \
  --sort-by=.status.observation.metrics[0].latest
```

### Training Job Logs

```bash
# PyTorch job logs
kubectl logs -n kubeflow <pytorchjob-name>-master-0

# TensorFlow job logs
kubectl logs -n kubeflow <tfjob-name>-chief-0
kubectl logs -n kubeflow <tfjob-name>-worker-0
```

## Best Practices

1. **Version Control**: Store pipeline definitions in Git
2. **Experiment Tracking**: Use MLflow alongside Kubeflow for unified tracking
3. **Resource Limits**: Set appropriate CPU/GPU limits for training jobs
4. **Checkpointing**: Implement model checkpointing for long-running training
5. **Distributed Training**: Use distributed training for large datasets/models
6. **Hyperparameter Tuning**: Start with random search, then Bayesian optimization
7. **Pipeline Components**: Create reusable components for common tasks

## Troubleshooting

### Pipeline Fails to Run

```bash
# Check pipeline API server logs
kubectl logs -n kubeflow -l app=ml-pipeline --tail=100

# Check workflow status
kubectl get workflows -n kubeflow

# Describe failed workflow
kubectl describe workflow <workflow-name> -n kubeflow
```

### Katib Experiment Not Progressing

```bash
# Check controller logs
kubectl logs -n kubeflow -l app=katib-controller

# Check trials
kubectl get trials -n kubeflow

# Check suggestion service
kubectl logs -n kubeflow -l suggestion=<experiment-name>
```

### Training Job Issues

```bash
# Check operator logs
kubectl logs -n kubeflow -l app=pytorch-operator
kubectl logs -n kubeflow -l app=tf-job-operator

# Check pod events
kubectl describe pod <job-pod> -n kubeflow

# Check resource availability
kubectl top nodes
```

## Next Steps

- [ ] Integrate with Seldon Core for model deployment (Phase 2.2)
- [ ] Create commodity-specific pipeline templates
- [ ] Set up automated retraining workflows
- [ ] Implement A/B testing for models
- [ ] Add model explainability components
- [ ] Create Grafana dashboards for ML metrics

## Resources

- **Kubeflow Pipelines**: https://www.kubeflow.org/docs/components/pipelines/
- **Katib**: https://www.kubeflow.org/docs/components/katib/
- **Training Operators**: https://www.kubeflow.org/docs/components/training/
- **KFP SDK**: https://kubeflow-pipelines.readthedocs.io/



