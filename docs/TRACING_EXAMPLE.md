# Distributed Tracing Example - Portal Request Path

## Overview

This document demonstrates the distributed tracing setup for the 254Carbon platform, specifically showing how a portal request flows through the service mesh with complete observability.

## Example Trace: Portal Homepage Request

### Trace Flow

A typical request to the portal (`https://254carbon.com`) flows through the following components:

1. **Ingress Gateway** (istio-ingressgateway)
   - Receives external HTTPS request
   - TLS termination
   - Routes to portal-services

2. **Portal Service** (portal-services pod)
   - Next.js application serving the portal
   - Fetches service catalog data
   - Renders homepage

3. **Service Mesh** (Istio/Envoy sidecars)
   - All traffic instrumented with tracing
   - Spans collected at each hop
   - Metrics exported to Prometheus

### Trace Details

#### Trace ID Example
```
Trace ID: 7f8a9b2c3d4e5f6a1b2c3d4e5f6a7b8c
Span Count: 5
Total Duration: 127ms
```

#### Span Breakdown

| Span | Service | Operation | Duration | Tags |
|------|---------|-----------|----------|------|
| 1 | istio-ingressgateway | ingress | 127ms | http.method=GET, http.url=/, http.status_code=200 |
| 2 | portal-services | portal.request | 119ms | component=portal, version=1.2.0 |
| 3 | portal-services.envoy | envoy.proxy | 5ms | istio.mesh_id=254carbon |
| 4 | portal-services | next.render | 89ms | page=index, framework=nextjs |
| 5 | portal-services | services.fetch | 18ms | endpoint=services.json, cache=miss |

### Custom Tags for Enhanced Observability

Our telemetry configuration adds custom tags to all traces:

- **environment**: `production`
- **platform**: `254carbon`
- **service_mesh_version**: `istio-1.20`
- **request_path**: Actual request path (e.g., `/`, `/api/health`)
- **response_code**: HTTP response code (e.g., `200`, `404`, `500`)
- **service_type**: For portal, set to `portal` (critical tier)
- **slo_tier**: `critical` for portal service

### Viewing Traces

#### Via Jaeger UI

1. Access Jaeger at `https://jaeger.254carbon.com` (if exposed)
2. Select service: `portal-services`
3. Filter by:
   - Min duration: 100ms (to find slow requests)
   - Tags: `slo_tier=critical`
   - HTTP status: `500` (to find errors)

#### Via Grafana

The Grafana dashboard at `k8s/monitoring/grafana-dashboard-configmap.yaml` includes a panel (ID: 13) titled "Distributed Traces - Portal Request Latency" that visualizes:
- Request latency over time
- Trace statistics (mean, max)
- Direct links to Jaeger for detailed trace inspection

### Sampling Configuration

| Service | Namespace | Sampling Rate | Rationale |
|---------|-----------|---------------|-----------|
| portal-services | default | 50% | Critical user-facing service, need high visibility |
| mlflow | ml-platform | 25% | Important but less critical than portal |
| clickhouse | data-platform | 25% | Database, high traffic but need trace samples |
| redis | data-platform | 0% | Very high traffic, disabled for performance |
| istiod | istio-system | 100% | Control plane, full tracing |

## Metrics Integration

### Tracing Metrics Exported

The telemetry configuration ensures the following metrics are exported alongside traces:

1. **REQUEST_COUNT**: Total requests per service
2. **REQUEST_DURATION**: Request latency histogram
3. **REQUEST_SIZE**: Request payload size
4. **RESPONSE_SIZE**: Response payload size
5. **TCP_OPENED_CONNECTIONS**: TCP connection establishment
6. **TCP_CLOSED_CONNECTIONS**: TCP connection termination

### Correlation with SLOs

Traces are correlated with SLO metrics defined in `helm/charts/monitoring/templates/slo-rules.yaml`:

- **Availability SLO**: Traces with response_code >= 500 indicate availability issues
- **Latency SLO**: Trace duration correlated with p95/p99 latency metrics
- **Error Budget**: Failed traces contribute to error budget burn calculations

## Logging Integration

### Loki Configuration

Access logs are exported to Loki (via stdout, scraped by Promtail/Fluentd):

- **All 5xx responses**: Full logging for error investigation
- **Sample of 2xx/3xx responses**: 10% sampling to reduce volume
- **Structured format**: JSON logs with trace_id, span_id for correlation

### Log-Trace Correlation

Each log line includes:
```json
{
  "timestamp": "2025-10-31T04:00:00.000Z",
  "level": "info",
  "service": "portal-services",
  "trace_id": "7f8a9b2c3d4e5f6a1b2c3d4e5f6a7b8c",
  "span_id": "1b2c3d4e5f6a7b8c",
  "message": "Request completed",
  "duration_ms": 127,
  "status_code": 200,
  "path": "/",
  "method": "GET"
}
```

## Validation

### Testing Tracing Setup

1. **Generate test traffic**:
   ```bash
   curl -v https://254carbon.com/
   ```

2. **Find trace in Jaeger**:
   - Look for service: `portal-services`
   - Find trace with operation: `portal.request`
   - Verify all spans are present

3. **Verify metrics**:
   ```promql
   # Check if traces are being collected
   sum(rate(istio_requests_total{destination_service="portal-services"}[5m]))
   
   # Check trace sampling
   sum(rate(traces_sampled_total{service="portal-services"}[5m]))
   ```

4. **Check logs**:
   ```logql
   {kubernetes_namespace_name="default", app="portal-services"} |~ "trace_id"
   ```

## Troubleshooting

### No Traces Appearing

1. Check Jaeger deployment: `kubectl get pods -n istio-system | grep jaeger`
2. Verify telemetry config: `kubectl get telemetry -A`
3. Check Envoy proxy logs: `kubectl logs <pod> -c istio-proxy`

### Missing Spans

1. Verify service mesh injection: `kubectl get pods -n default -o jsonpath='{.items[*].spec.containers[*].name}'`
2. Check if sidecar is running: Look for `istio-proxy` container
3. Review sampling rate: Increase if traces are missing

### High Cardinality Issues

If metrics storage becomes problematic:
1. Reduce sampling percentage
2. Add more selective filters in telemetry config
3. Disable tracing for high-traffic internal services (like redis)

## Next Steps

1. **Add exemplars**: Link metrics to traces in Grafana
2. **Custom instrumentation**: Add application-level spans for business logic
3. **Trace-based alerts**: Alert on anomalous trace patterns
4. **SLO dashboards**: Create per-service SLO dashboards with trace links
