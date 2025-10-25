# Kubecost Deployment & Budget Monitoring

Kubecost is now part of the monitoring Helm chart and deploys alongside the rest of the observability stack via ArgoCD. This document lists the knobs that control the deployment, monthly budget thresholds, and the dashboards/alerts that ship with the chart.

## Helm Configuration Summary

- **Dependency:** The monitoring chart vendors the upstream `kubecost/cost-analyzer` Helm chart (`alias: kubecost`). Dependencies are managed with `helm dependency update` in `helm/charts/monitoring`.
- **Base values:** Global defaults live in `helm/charts/monitoring/values.yaml`. Key toggles:
  - `kubecost.enabled` – controls whether Kubecost is deployed.
  - `kubecost.global.prometheus.fqdn` – points Kubecost to the platform Prometheus (`http://prometheus.monitoring.svc:9090`).
  - `kubecost.global.notifications.alertmanager` – routes Kubecost internal alerts to the existing Alertmanager service.
  - `kubecost.persistentVolume.storageClass` – uses `local-storage-standard` for Kubecost ETL state.
- **Production overrides:** Budget thresholds, alert contacts, and ingress hostnames are defined in `helm/charts/monitoring/values/prod.yaml`:
  - `kubecost.global.notifications.alerts` describes Kubecost-native budget alerts sent via email.
  - `kubecostBudgetThresholds` lists the same monthly budgets the PrometheusRule uses for warning/critical firing (namespaces, dollar amounts, and contact emails).
  - `kubecost.ingress` exposes the UI at `https://kubecost.254carbon.com` (TLS secret `kubecost-tls`).

> **Sync tip:** When budgets change, update both the `kubecost.global.notifications.alerts` entries and the matching `kubecostBudgetThresholds` block so Kubecost email alerts and Prometheus alerting stay in lock-step.

## Dashboards

- `helm/charts/monitoring/templates/kubecost-grafana-dashboards.yaml` provisions the **Kubecost Budget Overview** dashboard. Panels include namespace-level spend stats, budget consumption gauges (80% warning / 100% critical), and daily cost trends.
- Dashboards are automatically mounted into the platform Grafana (`grafana` deployment in `monitoring` namespace). After Argo sync, check Grafana under *FinOps* → *Kubecost Budget Overview*.

## Alerting

- **PrometheusRule:** `helm/charts/monitoring/templates/kubecost-alerts.yaml` renders warning and critical alerts per namespace using `kubecost_namespace_total_cost` metrics. Alerts fire when spend crosses 80% and 100% of the configured monthly budgets and route through the existing Alertmanager pipeline.
- **Kubecost-native alerts:** The upstream chart consumes the same thresholds via `kubecost.global.notifications.alerts`, driving Kubecost’s internal notification engine (email/webhook).

## Deployment & Verification Checklist

1. **Argo sync:** ArgoCD will pick up the new dependency automatically. Force a sync of the `monitoring` application if Kubecost was previously absent.
2. **Pods:** Confirm Kubecost pods are running:
   ```bash
   kubectl -n monitoring get pods -l app.kubernetes.io/instance=kubecost
   ```
3. **Ingress/DNS:** Verify the ingress resolves and presents TLS (`kubecost.254carbon.com`).
4. **Metrics wiring:** Ensure Prometheus is scraping Kubecost (`kubectl -n monitoring get servicemonitors | grep kubecost`) and that `kubecost_namespace_total_cost` appears in Prometheus.
5. **Dashboards:** Log into Grafana and open *Kubecost Budget Overview* to confirm panels render data.
6. **Alerts:** Temporarily lower a budget threshold (e.g., set `warningPercent` to `0.01`) and run `kubectl -n monitoring get prometheusrule kubecost-budget-alerts -o yaml` to verify the rendered expressions before reverting.

For advanced customization (cloud billing ingestion, multi-cluster federation, etc.) consult the upstream chart values exposed via the `kubecost` block in `helm/charts/monitoring/values.yaml`.
