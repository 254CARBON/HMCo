# Custom MLflow image with PostgreSQL and S3 support
FROM ghcr.io/mlflow/mlflow:v2.10.0

USER root

# Install PostgreSQL client and Python drivers
RUN pip install --no-cache-dir \
    psycopg2-binary \
    boto3 \
    pymysql \
    cryptography

# Create non-root user for running MLflow
RUN useradd -m -u 1000 mlflow

# Set working directory
WORKDIR /home/mlflow

# Switch to non-root user
USER mlflow

# Expose port
EXPOSE 5000

# Default command (can be overridden)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]


