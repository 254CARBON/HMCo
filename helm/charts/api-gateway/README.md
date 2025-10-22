# API Gateway Helm Chart

This Helm chart deploys the API Gateway service, which is built on [Kong](https://konghq.com/). It provides a centralized entry point for all API traffic, handling routing, authentication, and other cross-cutting concerns.

## Chart Details

This chart includes the following components:

*   **Kong Deployment:** The core proxy that handles API traffic.
*   **Kong Services:** Kubernetes services for exposing the Kong proxy and admin APIs.
*   **Kong Routes:** Defines how requests are routed to upstream services.
*   **Kong Plugins:** Configures plugins for features like JWT authentication.
*   **Configuration Job:** A one-time job for applying initial configuration to Kong.

## Prerequisites

*   Kubernetes 1.19+
*   Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```
