# MLflow Helm Chart

This Helm chart deploys MLflow, a platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

## Chart Details

This chart includes the following components:

*   **MLflow Deployment:** The core MLflow server.
*   **MLflow Service:** Exposes the MLflow server.
*   **MLflow Ingress:** Configures ingress for the MLflow UI.
*   **MLflow ConfigMap:** Contains configuration for the MLflow server.
*   **MLflow Secrets:** Manages secrets for the database connection.
*   **ServiceMonitor:** Configures Prometheus monitoring for the MLflow server.

## Prerequisites

*   Kubernetes 1.19+
*   Helm 3.2.0+
*   A running PostgreSQL database.

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```
