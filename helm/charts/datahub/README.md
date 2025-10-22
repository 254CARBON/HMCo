# DataHub Helm Chart

This Helm chart deploys DataHub, a modern data discovery platform.

## Chart Details

This chart includes the following components:

*   **DataHub Frontend:** The user interface for DataHub.
*   **DataHub GMS:** The General Metadata Service, the core of DataHub.
*   **Elasticsearch:** Used for searching metadata.
*   **Neo4j:** Used for storing the metadata graph.
*   **Prerequisites:** A job that sets up Elasticsearch and Neo4j.

## Prerequisites

*   Kubernetes 1.19+
*   Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```
