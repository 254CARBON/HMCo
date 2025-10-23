# 254Carbon Advanced Analytics Platform

This repository contains the infrastructure-as-code for the 254Carbon Advanced Analytics Platform, a comprehensive, cloud-native environment for data analytics and machine learning.

## Overview

The platform is designed to be a scalable and resilient environment for data scientists and analysts to develop and deploy analytics and machine learning models. It is built on Kubernetes and leverages a variety of open-source technologies, including:

*   **Data Platform:** A suite of tools for data storage, processing, and visualization, including a Data Lake with MinIO and Iceberg, a Data Warehouse with Doris, and a Data Query Engine with Trino. Data discovery and governance is provided by DataHub.
*   **ML Platform:** A comprehensive environment for machine learning, including MLflow for experiment tracking, Kubeflow for pipelines, and Ray for distributed computing.
*   **Foundational Infrastructure:** A robust set of foundational services, including a service mesh with Istio, monitoring with Prometheus and Grafana, secrets management with Vault, and automated deployments with ArgoCD.

## Getting Started

All platform components are managed as Helm charts and deployed via ArgoCD. The ArgoCD applications are defined in `k8s/gitops/argocd-applications.yaml`. For runbooks, SSO guides, and troubleshooting, start with the curated documentation index at [`docs/index.md`](docs/index.md).

### Prerequisites

*   A running Kubernetes cluster.
*   `kubectl` configured to connect to your cluster.
*   `helm` for managing charts.
*   `argocd` CLI for managing ArgoCD applications.

### Deployment

1.  **Install ArgoCD:** Apply the `argocd-install.yaml` manifest to your cluster.
2.  **Apply the AppProject:** Apply the `argocd-applications.yaml` file to create the `production` AppProject (pre-scoped to namespaces including `data-platform`, `monitoring`, `istio-system`, and supporting services).
3.  **Deploy the Applications:** The applications are configured with `selfHeal: true` and will be automatically deployed by ArgoCD in the correct order.

## Repository Structure

*   `helm/`: Contains all the Helm charts for the platform components.
    *   `charts/`: Contains the individual charts for each service.
        *   `data-platform/`: The umbrella chart for the data platform.
        *   `ml-platform/`: The umbrella chart for the machine learning platform.
        *   ... and many more.
*   `docs/`: Contains detailed documentation for the platform.
*   `services/`: Contains source code for custom services.
*   `workflows/`: Contains workflow definitions for the platform.
