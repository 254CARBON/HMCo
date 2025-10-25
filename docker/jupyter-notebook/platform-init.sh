#!/bin/bash
# Platform initialization script for JupyterHub notebooks

set -e

echo "Initializing 254Carbon platform environment..."

# Create configuration directory
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab

# Create Jupyter Lab settings
cat > ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.json << 'EOF'
{
  "kernelShutdownConfirmation": false,
  "notebookVersion": 5
}
EOF

# Create git configuration
git config --global user.email "jupyter@254carbon.com"
git config --global user.name "Jupyter User"

# Create platform config directory
mkdir -p ~/platform-config

# Export environment variables for platform services
cat > ~/platform-config/platform-env.sh << 'EOF'
#!/bin/bash
# Platform services environment configuration

export TRINO_HOST="${TRINO_HOST:-trino.data-platform}"
export TRINO_PORT="${TRINO_PORT:-8080}"
export TRINO_CATALOG="${TRINO_CATALOG:-iceberg}"
export TRINO_SCHEMA="${TRINO_SCHEMA:-default}"

export MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio.data-platform:9000}"
export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow.ml-platform:5000}"
export MLFLOW_ARTIFACT_URI="${MLFLOW_ARTIFACT_URI:-s3://mlflow-artifacts}"

export POSTGRES_HOST="${POSTGRES_HOST:-postgres-shared.data-platform}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_DATABASE="${POSTGRES_DATABASE:-jupyter}"
export POSTGRES_USER="${POSTGRES_USER:-jupyter_user}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"

export DATAHUB_REST_API="${DATAHUB_REST_API:-http://datahub-gms.data-platform:8080}"

export RAY_CLUSTER_HEAD="${RAY_CLUSTER_HEAD:-ray-cluster-head.data-platform:6379}"

export KAFKA_BROKERS="${KAFKA_BROKERS:-kafka-cluster-kafka-bootstrap.data-platform:9092}"

echo "Platform environment variables loaded!"
EOF

chmod +x ~/platform-config/platform-env.sh

# Create example connection scripts
cat > ~/platform-config/connect-trino.py << 'EOF'
"""
Trino connection helper
Usage: from connect_trino import get_connection
"""
from trino.dbapi import connect

def get_connection():
    """Get connection to Trino cluster"""
    import os
    return connect(
        host=os.getenv("TRINO_HOST", "trino.data-platform"),
        port=int(os.getenv("TRINO_PORT", 8080)),
        user="jupyter",
        catalog=os.getenv("TRINO_CATALOG", "iceberg"),
        schema=os.getenv("TRINO_SCHEMA", "default"),
    )

# Quick test
if __name__ == "__main__":
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    print("Trino connection successful!")
    print(cursor.fetchall())
EOF

cat > ~/platform-config/connect-minio.py << 'EOF'
"""
MinIO connection helper
Usage: from connect_minio import get_client
"""
from minio import Minio
import os

def get_client():
    """Get MinIO client"""
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio.data-platform:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

# Quick test
if __name__ == "__main__":
    client = get_client()
    buckets = client.list_buckets()
    print("MinIO buckets:")
    for bucket in buckets.buckets:
        print(f"  - {bucket.name}")
EOF

cat > ~/platform-config/connect-mlflow.py << 'EOF'
"""
MLflow connection helper
Usage: import mlflow; mlflow.set_tracking_uri(get_tracking_uri())
"""
import os

def get_tracking_uri():
    """Get MLflow tracking URI"""
    return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.ml-platform:5000")

def set_experiment(name):
    """Create or get experiment"""
    import mlflow
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(name)

# Quick test
if __name__ == "__main__":
    import mlflow
    tracking_uri = get_tracking_uri()
    print(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("Default")
    print(f"Default experiment: {experiment.experiment_id if experiment else 'not found'}")
EOF

cat > ~/platform-config/connect-postgres.py << 'EOF'
"""
PostgreSQL connection helper
Usage: from connect_postgres import get_connection
"""
import psycopg2
import os

def get_connection():
    """Get connection to PostgreSQL"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres-shared.data-platform"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        database=os.getenv("POSTGRES_DATABASE", "jupyter"),
        user=os.getenv("POSTGRES_USER", "jupyter_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )

# Quick test
if __name__ == "__main__":
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    print("PostgreSQL connection successful!")
    print(cursor.fetchone()[0])
    cursor.close()
    conn.close()
EOF

cat > ~/platform-config/connect-datahub.py << 'EOF'
"""
DataHub connection helper
Usage: from connect_datahub import get_emitter
"""
from datahub.emitter.rest_emitter import DatahubRestEmitter
import os

def get_emitter():
    """Get DataHub REST emitter"""
    return DatahubRestEmitter(
        gms_server=os.getenv("DATAHUB_REST_API", "http://datahub-gms.data-platform:8080")
    )

# Quick test
if __name__ == "__main__":
    emitter = get_emitter()
    print(f"DataHub emitter created successfully")
    print(f"Server: {emitter.server}")
EOF

# Create startup notebook
cat > ~/work/00_Platform_Setup.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 254Carbon Platform Setup\n",
    "\n",
    "This notebook demonstrates how to connect to all platform services from JupyterHub."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Platform Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, os.path.expanduser('~/platform-config'))\n",
    "import os\n",
    "\n",
    "# Load platform environment variables\n",
    "exec(open(os.path.expanduser('~/platform-config/platform-env.sh')).read())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test Trino Connection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from connect_trino import get_connection\n",
    "\n",
    "conn = get_connection()\n",
    "cursor = conn.cursor()\n",
    "cursor.execute(\"SELECT 1 as test_value\")\n",
    "print(\"Trino connection test:\")\n",
    "print(cursor.fetchall())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test MinIO Connection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from connect_minio import get_client\n",
    "\n",
    "client = get_client()\n",
    "buckets = client.list_buckets()\n",
    "print(\"MinIO buckets:\")\n",
    "for bucket in buckets.buckets:\n",
    "    print(f\"  - {bucket.name}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test MLflow Connection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mlflow\n",
    "from connect_mlflow import get_tracking_uri\n",
    "\n",
    "mlflow.set_tracking_uri(get_tracking_uri())\n",
    "print(f\"MLflow tracking URI: {mlflow.get_tracking_uri()}\")\n",
    "\n",
    "# List experiments\n",
    "experiments = mlflow.search_experiments()\n",
    "print(f\"\\nAvailable experiments ({len(experiments)}):\")\n",
    "for exp in experiments:\n",
    "    print(f\"  - {exp.name} (ID: {exp.experiment_id})\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "Platform initialization complete!"
echo ""
echo "Available tools in ~/platform-config/:"
echo "  - connect-trino.py"
echo "  - connect-minio.py"
echo "  - connect-mlflow.py"
echo "  - connect-postgres.py"
echo "  - connect-datahub.py"
echo "  - platform-env.sh"
echo ""
echo "Get started with: 00_Platform_Setup.ipynb in ~/work/"
