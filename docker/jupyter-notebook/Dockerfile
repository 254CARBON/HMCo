# Custom JupyterHub Notebook Image
# Based on Jupyter Data Science Notebook with platform-specific libraries

FROM jupyter/datascience-notebook:latest@sha256:476c6e673e7d5d8b5059f8680b1c6a988942a79263da651bf302dc696ab311f2

LABEL maintainer="254Carbon Platform Team <platform@254carbon.com>"
LABEL description="JupyterHub notebook image with integrated platform service libraries"

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    postgresql-client \
    mysql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Upgrade pip and install base packages
RUN pip install --quiet --upgrade pip setuptools wheel

# Data Processing & Analysis
RUN pip install --quiet \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    statsmodels \
    sympy

# Data visualization
RUN pip install --quiet \
    matplotlib \
    seaborn \
    plotly \
    bokeh \
    altair \
    holoviews

# Database & Query Engines
RUN pip install --quiet \
    trino \
    presto-python-client \
    psycopg2-binary \
    pymysql \
    sqlalchemy \
    sqlalchemy-trino \
    pyodbc

# Cloud Storage
RUN pip install --quiet \
    s3fs \
    boto3 \
    minio \
    adlfs \
    gcsfs

# Data Formats
RUN pip install --quiet \
    pyarrow \
    delta-spark \
    iceberg-python \
    parquet \
    fastparquet

# Big Data Processing
RUN pip install --quiet \
    pyspark \
    kafka-python \
    confluent-kafka

# ML & Deep Learning
RUN pip install --quiet \
    tensorflow \
    torch==2.8.0 \
    torchvision \
    xgboost \
    lightgbm \
    catboost

# ML Tracking & Management
RUN pip install --quiet \
    mlflow>=2.22.2 \
    wandb \
    optuna \
    ray[tune] \
    ray[serve]

# Data Governance & Catalog
RUN pip install --quiet \
    datahub-client \
    datahub-python-sdk

# Data Quality & Validation
RUN pip install --quiet \
    great-expectations \
    dbt-core \
    dbt-trino \
    dbt-postgres

# Distributed Computing
RUN pip install --quiet \
    dask \
    dask-distributed \
    distributed

# Jupyter Extensions & Tools
RUN pip install --quiet \
    jupyterlab \
    jupyterlab-git \
    jupyterlab-code-formatter \
    black \
    jupyterlab-lsp \
    python-lsp-server \
    python-lsp-black

# Development Tools
RUN pip install --quiet \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Utilities
RUN pip install --quiet \
    python-dotenv \
    pyyaml \
    requests>=2.32.4 \
    aiohttp \
    httpx

USER root

# Create directories for mounted volumes and examples
RUN mkdir -p /mnt/shared-data \
    && mkdir -p /opt/notebooks/examples \
    && chown -R jovyan:jovyan /mnt/shared-data \
    && chown -R jovyan:jovyan /opt/notebooks

USER jovyan

# Copy platform configuration scripts
COPY --chown=jovyan:jovyan platform-init.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/platform-init.sh

# Copy example notebooks
COPY --chown=jovyan:jovyan examples/ /opt/notebooks/examples/

# Set up JupyterLab as default
ENV JUPYTER_ENABLE_LAB=1

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8888/lab/api/status || exit 1

# Start JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
