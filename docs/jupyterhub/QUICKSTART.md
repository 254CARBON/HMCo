# JupyterHub Quick Start Guide

Get up and running with JupyterHub in 5 minutes!

## Step 1: Access JupyterHub

Navigate to: **https://jupyter.254carbon.com**

## Step 2: Authenticate

1. You'll see Cloudflare Access login page
2. Enter your email address
3. Check your email for one-time code
4. Enter the code to authenticate

## Step 3: Start Your Server

1. Click **"Start My Server"** button
2. Select server options (default is fine):
   - Environment: JupyterLab
   - CPU: 2 cores
   - Memory: 8GB
   - Storage: 10GB
3. Click **"Start"**
4. Wait for server to spawn (usually 30-60 seconds)

## Step 4: Access JupyterLab

Once spawned, you'll be in **JupyterLab** interface:

```
┌─ JupyterLab ─────────────────────────────┐
│ File Edit View Run Kernel Tabs Help      │
├──────────────────┬──────────────────────┤
│                  │                      │
│ File Browser     │   Notebook Editor    │
│                  │                      │
│ (Open files)     │  (Write & run code)  │
│                  │                      │
└──────────────────┴──────────────────────┘
```

## Quick Examples

### Example 1: Query Data with Trino

Create a new notebook and run:

```python
import pandas as pd
from connect_trino import get_connection

# Connect to Trino
conn = get_connection()
cursor = conn.cursor()

# Query Iceberg table
cursor.execute("""
    SELECT * 
    FROM iceberg.default.sample_data 
    LIMIT 100
""")

# Convert to DataFrame
df = pd.DataFrame(cursor.fetchall())
print(df)
```

### Example 2: Access Object Storage (MinIO)

```python
from connect_minio import get_client

# Connect to MinIO
client = get_client()

# List buckets
buckets = client.list_buckets()
print("Available buckets:")
for bucket in buckets.buckets:
    print(f"  - {bucket.name}")

# List files in bucket
objects = client.list_objects("data-lake", prefix="raw/")
print("\nFiles in data-lake/raw/:")
for obj in objects:
    print(f"  - {obj.object_name} ({obj.size} bytes)")

# Upload file
client.fput_object("data-lake", "my-data.csv", "/path/to/local/file.csv")

# Download file
client.fget_object("data-lake", "my-data.csv", "/tmp/downloaded.csv")
```

### Example 3: Track ML Experiments with MLflow

```python
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow.ml-platform:5000")
mlflow.set_experiment("iris-classification")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow run
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy:.4f}")
```

### Example 4: Distributed Computing with Ray

```python
import ray
import time

# Initialize Ray (connects to existing cluster)
ray.init(address="ray://ray-cluster-head.data-platform:6379", ignore_reinit_error=True)

# Define a distributed task
@ray.remote
def expensive_function(x):
    time.sleep(1)
    return x * x

# Run tasks in parallel
results = ray.get([expensive_function.remote(i) for i in range(10)])
print(f"Results: {results}")

# Shutdown Ray
# ray.shutdown()
```

## File Organization

Your home directory is organized as:

```
/home/jovyan/
├── work/              (Persistent storage for notebooks)
│   ├── notebook1.ipynb
│   ├── data/
│   └── results/
├── platform-config/   (Platform connection scripts)
│   ├── connect-trino.py
│   ├── connect-minio.py
│   ├── connect-mlflow.py
│   └── ...
└── .jupyter/          (Jupyter configuration)
    └── lab/
```

All files in `/home/jovyan/` are automatically saved and persist across sessions!

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | `Shift + Enter` |
| Add cell below | `B` |
| Add cell above | `A` |
| Delete cell | `D, D` |
| Open command palette | `Ctrl + Shift + C` |
| Open file browser | `Ctrl + Shift + F` |
| Toggle line numbers | `Ctrl + L` |

## Stop Your Server

When you're done:

1. Click **"File"** menu
2. Select **"Log Out"**
3. Your notebook server will stop automatically

## Common Issues

### Q: I'm getting "Connection refused" errors

**A:** This usually means the platform service isn't responding. Wait a moment and try again, or check:
```python
import socket
socket.gethostbyname("trino.data-platform")
```

### Q: My notebook keeps restarting

**A:** You may be running out of memory. Check with:
```bash
free -h
```

Consider requesting more memory when starting your server.

### Q: I can't see my files after logging back in

**A:** Files are stored in `/home/jovyan/work/`. Make sure you saved them there!

### Q: How do I install additional packages?

**A:** Use pip in a notebook cell:
```python
!pip install --quiet numpy pandas scikit-learn
```

Or use conda if available:
```python
!conda install -c conda-forge package-name
```

## Next Steps

1. **Explore example notebooks**: Check `/opt/notebooks/examples/`
2. **Read documentation**: https://docs.254carbon.com
3. **Join community**: Slack channel #data-science
4. **Get help**: platform@254carbon.com

## Resources

- [JupyterLab Docs](https://jupyterlab.readthedocs.io/)
- [Jupyter Notebook Docs](https://jupyter-notebook.readthedocs.io/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)

## Tips & Tricks

### Tip 1: Save Notebooks as Python Scripts
```
File → Export Notebook As → Executable Script
```

### Tip 2: Share Notebooks
Upload to shared storage:
```python
import shutil
shutil.copy("/home/jovyan/work/notebook.ipynb", "/mnt/shared-data/shared-notebook.ipynb")
```

### Tip 3: Schedule Recurring Tasks
Use DolphinScheduler to run notebooks on a schedule:
```python
# Export notebook as script, then create DolphinScheduler task
```

### Tip 4: Version Control
```bash
!cd /home/jovyan && git init
!git add work/notebook.ipynb
!git commit -m "Initial version"
```

## Security Reminders

⚠️ **Never commit secrets!**

```python
# BAD
connection = trino.connect(
    host="trino.data-platform",
    user="admin",
    password="secret123"  # Don't do this!
)

# GOOD
import os
password = os.getenv("TRINO_PASSWORD")
```

⚠️ **Don't download sensitive data**

Always work with data in the cluster!

## Getting Help

Having issues? Check:

1. **Logs**: `kubectl logs -n jupyter <pod-name>`
2. **Status page**: https://status.254carbon.com
3. **Documentation**: https://docs.254carbon.com
4. **Contact support**: platform@254carbon.com

---

**Happy Data Science! 🚀**
