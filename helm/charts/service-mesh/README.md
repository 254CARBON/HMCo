# Service Mesh Helm Chart

This Helm chart deploys the Istio service mesh.

## Chart Details

This chart includes the following components:

*   **Istio Operator:** Manages the lifecycle of the Istio control plane.
*   **Istio CRDs:** The Custom Resource Definitions required for Istio.
*   **Observability:** Configurations for Jaeger, Kiali, and telemetry.
*   **Security:** Authorization policies and peer authentication.
*   **Traffic Management:** Destination rules and virtual services.

## Prerequisites

*   Kubernetes 1.19+
*   Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```
