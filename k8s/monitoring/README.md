Observability Stack — Prometheus Operator, Loki, Alertmanager, Grafana

Overview
- Prometheus Operator via kube-prometheus-stack (Helm) with cross-namespace ServiceMonitor/PodMonitor discovery.
- Loki + Promtail (Helm) with persistence and retention.
- Alertmanager (Helm) with example email/PagerDuty routes.
- Grafana is deployed separately in this repo and already wired for Cloudflare Access SSO.

Files
- `k8s/monitoring/kube-prometheus-stack-values.yaml` — tuned Prometheus Operator config
- `k8s/monitoring/loki-values.yaml` — loki-stack config (persistence, retention, promtail)
- `k8s/monitoring/alertmanager-values.yaml` — base Alertmanager config
- `k8s/monitoring/monitors-logging.yaml` — ServiceMonitor/PodMonitor for Loki/Promtail
- `k8s/monitoring/ingress-monitors.yaml` — PodMonitor for NGINX Ingress Controller metrics
- `k8s/monitoring/ingress-slo-recording-rules.yaml` — ingress path SLIs (p95/p99, error ratio)
- `k8s/monitoring/ingress-slo-alerts.yaml` — service‑specific ingress SLO thresholds
- Existing CRs: ServiceMonitor/PrometheusRule across data-platform and monitoring namespaces

Prereqs
- Helm repos
  - `helm repo add prometheus-community https://prometheus-community.github.io/helm-charts`
  - `helm repo add grafana https://grafana.github.io/helm-charts`
  - `helm repo update`

Install/Upgrade
1) Prometheus Operator (kube-prometheus-stack)
   helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
     -n monitoring --create-namespace \
     -f k8s/monitoring/kube-prometheus-stack-values.yaml

2) Loki + Promtail
   helm upgrade --install loki grafana/loki-stack \
     -n monitoring -f k8s/monitoring/loki-values.yaml

3) Alertmanager (optional standalone)
   helm upgrade --install alertmanager prometheus-community/alertmanager \
     -n monitoring -f k8s/monitoring/alertmanager-values.yaml

4) CRDs for scraping rules (already in repo)
   kubectl apply -f k8s/monitoring/servicemonitors.yaml
   kubectl apply -f k8s/monitoring/iceberg-monitoring.yaml
   kubectl apply -f k8s/monitoring/spark-servicemonitor.yaml
   kubectl apply -f k8s/monitoring/alerting-rules.yaml
   kubectl apply -f k8s/monitoring/monitors-logging.yaml
   kubectl apply -f k8s/monitoring/ingress-monitors.yaml
   kubectl apply -f k8s/monitoring/ingress-slo-recording-rules.yaml
   kubectl apply -f k8s/monitoring/ingress-slo-alerts.yaml

5) Grafana (already deployed in repo)
   - Dashboards auto-provision from ConfigMaps here (label `grafana_dashboard: "1"`).
   - Ingress for `grafana.254carbon.com` is in `k8s/ingress/ingress-rules.yaml` with Cloudflare Access.

Notes
- The Operator values enable discovery from all namespaces; keep using namespaced ServiceMonitors and PrometheusRules.
- Existing manual `prometheus.yaml` and `alert-manager.yaml` remain for dev/offline use; prefer Helm/operator in production.
- Loki retention is set to 14 days; adjust `retention_period` and PVC sizes per capacity planning.
- For Alertmanager secrets (SMTP/PagerDuty), wire real secrets from Vault and avoid committing credentials.

Validation
- Targets: `kubectl -n monitoring port-forward svc/prometheus-operated 9090 &` → http://localhost:9090/targets
- Loki: `kubectl -n monitoring port-forward svc/loki 3100 &` → query `{namespace="data-platform"}` in Grafana Explore
- Alerts: `kubectl -n monitoring port-forward svc/alertmanager-operated 9093 &` → http://localhost:9093
