# Monitoring Helm Chart

This Helm chart deploys a comprehensive monitoring stack, including Prometheus for metrics collection, Grafana for visualization, and Alertmanager for handling alerts.

## Chart Details

This chart includes the following components:

*   **Prometheus:** A powerful open-source monitoring and alerting toolkit.
*   **Grafana:** A multi-platform open source analytics and interactive visualization web application.
*   **Alertmanager:** Handles alerts sent by client applications such as the Prometheus server.
*   **Dashboards:** Pre-configured Grafana dashboards for monitoring various components.
*   **Alerting Rules:** A set of pre-configured alerting rules for Prometheus.

## Prerequisites

*   Kubernetes 1.19+
*   Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```
