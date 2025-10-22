# Data Lake Helm Chart

This Helm chart deploys the services that constitute the Data Lake, including MinIO for object storage and Iceberg for table format.

## Chart Details

This chart includes the following components:

*   **MinIO:** A high-performance, S3-compatible object store.
*   **Iceberg REST:** A REST service for managing Iceberg tables.
*   **Data Lifecycle Policies:** ConfigMaps defining data lifecycle policies.
*   **Initialization Jobs:** Jobs for initializing MinIO and Iceberg.

## Prerequisites

*   Kubernetes 1.19+
*   Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```
