# Performance Metrics and Optimization Targets

Objective
- Establish clear SLIs/SLOs, Prometheus queries, and optimization targets across the platform to guide tuning and capacity planning.

Scope
- User-facing services: Grafana, DataHub (frontend+GMS), Trino, Doris, Superset, LakeFS, MinIO, Vault, Iceberg REST Catalog.
- Infra: Kubernetes API/cluster health, Prometheus/Loki stack, ingress path latency.

Global SLO Targets (Initial)
- Availability: 99.9% monthly per user-facing service (error budget: 43.2 minutes/month).
- Error rate: < 0.1% (5xx) of requests per service (rolling 30d).
- Latency p95: < 300 ms for UI/API endpoints; p99 < 800 ms for steady-state traffic.
- Resource headroom: Node CPU < 70%, node memory < 80% sustained (5m), PVC usage < 85%.
- Logging: Retention 14 days; query response p95 < 1.5 s for typical log queries.

Service-Specific Targets (Starting Points)
- Grafana: p95 dashboard render < 1 s; availability 99.9%.
- DataHub: p95 API latency < 300 ms; error rate < 0.2%; availability 99.9%.
- Trino: p95 query completion < 5 s for interactive queries; failure rate < 0.5%.
- ClickHouse: p95 query completion < 1 s for OLAP slices; ingestion lag < 10 s.
- Superset: p95 dashboard load < 2 s (cached); API error rate < 0.2%.
- MinIO: p95 object GET < 100 ms intra-cluster; 99.9% availability.
- Vault: p95 secret read < 50 ms; unsealed=1; availability 99.99%.
- LakeFS: p95 API < 250 ms; error rate < 0.2%.
- Iceberg REST Catalog: p95 < 250 ms; error rate < 0.2% (already instrumented).

How We Measure (PromQL)
- Error rate (per job):
  - sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)
    /
    sum(rate(http_requests_total[5m])) by (job)
- Latency p95 (per job):
  - histogram_quantile(0.95, sum by (job, le) (rate(http_request_duration_seconds_bucket[5m])))
- Availability (per job):
  - avg_over_time(up[5m]) by (job)
- Node resource headroom:
  - CPU: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))*100)
  - Memory: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))*100
- PVC usage:
  - kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes

Error Budget Burn Alerts (99.9% SLO example)
- Fast burn (critical): 2h/6h windows; alert when error_ratio_2h > 0.0144 and error_ratio_6h > 0.006.
- Slow burn (warning): 24h/3d windows; alert when error_ratio_24h > 0.003 and error_ratio_3d > 0.001.

Optimization Levers
- Ingress: enable keepalive, gzip/ Brotli, caching for static assets; CDN config via Cloudflare.
- Services: enable HTTP connection pooling, tune thread/concurrency pools, enable result caching.
- DB/query engines (Trino/ClickHouse): tune memory pools, spill to disk thresholds, worker count.
- Storage: size PVCs with >20% headroom, enable parallelism, revisit MinIO erasure coding.
- Kubernetes: right-size requests/limits; anti-affinity for HA control planes; pod disruption budgets.
- Observability: downsample/recording rules for heavy queries; increase retention judiciously.

Acceptance & Review
- Weekly dashboard review for SLO compliance; monthly target re-baseline using recorded SLIs.
- Error budget policy: if budget burn exceeds warning threshold, trigger focused reliability sprint.

Implementation Artifacts
- Prometheus recording rules: `k8s/monitoring/slo-recording-rules.yaml`
- SLO alerts: `k8s/monitoring/slo-alerts.yaml`
- Grafana SLO dashboard: `k8s/monitoring/grafana-dashboards.yaml` (slo-overview.json)
- Synthetic canary checks: `helm/charts/monitoring/values.yaml` (`blackboxExporter` section)

Deployment
- kubectl apply -f k8s/monitoring/slo-recording-rules.yaml
- kubectl apply -f k8s/monitoring/slo-alerts.yaml
- kubectl apply -f k8s/monitoring/grafana-dashboards.yaml

Validation
- In Grafana, open “SLO Overview” and verify time series.
- Query Prometheus for:
  - sli:http_error_ratio:5m
  - sli:http_latency_p95:5m
  - sli:service_availability:5m
- Verify synthetic canaries:
  - `kubectl get probes -n monitoring`
  - `kubectl logs -n monitoring deploy/blackbox-exporter`
