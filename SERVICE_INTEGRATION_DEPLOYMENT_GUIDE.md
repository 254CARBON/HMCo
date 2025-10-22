# Service Integration Enhancement - Deployment Guide

**Platform**: 254Carbon Data Platform  
**Version**: 1.0.0  
**Date**: October 22, 2025  
**Status**: Ready for Deployment

---

## Executive Summary

This guide provides comprehensive deployment instructions for the enhanced service integration architecture, including:

- **Istio Service Mesh**: Secure service-to-service communication with mTLS
- **Kong API Gateway**: Unified API management and authentication
- **Event-Driven Architecture**: Kafka-based asynchronous communication
- **Enhanced Observability**: Distributed tracing, metrics, and visualization

## Prerequisites

### System Requirements
- Kubernetes cluster (v1.24+)
- 20 CPU cores available
- 40GB RAM available
- 100GB storage for Kafka and PostgreSQL
- Helm 3.x installed
- kubectl configured

### Existing Services
- All 35+ services deployed in data-platform namespace
- Kafka cluster (3 brokers) operational
- Prometheus and Grafana monitoring stack
- Network policies configured

## Phase 1: Service Mesh Deployment (Day 1-2)

### Step 1.1: Deploy Istio Operator

```bash
# Apply Istio operator
kubectl apply -f k8s/service-mesh/istio-operator.yaml

# Wait for operator to be ready
kubectl wait --for=condition=available --timeout=300s \
  deployment/istio-operator -n istio-operator

# Verify operator
kubectl get pods -n istio-operator
```

### Step 1.2: Install Istio Control Plane

```bash
# Apply Istio configuration
kubectl apply -f k8s/service-mesh/istio-config.yaml

# Wait for istiod to be ready
kubectl wait --for=condition=available --timeout=300s \
  deployment/istiod -n istio-system

# Verify installation
kubectl get pods -n istio-system
istioctl version
```

### Step 1.3: Enable Sidecar Injection

```bash
# Label namespaces for automatic injection
kubectl label namespace data-platform istio-injection=enabled
kubectl label namespace monitoring istio-injection=enabled

# Verify labels
kubectl get namespace -L istio-injection
```

### Step 1.4: Deploy Observability Tools

```bash
# Deploy Kiali
kubectl apply -f k8s/service-mesh/observability/kiali.yaml

# Deploy Jaeger
kubectl apply -f k8s/service-mesh/observability/jaeger.yaml

# Deploy telemetry configuration
kubectl apply -f k8s/service-mesh/observability/telemetry.yaml

# Verify deployment
kubectl get pods -n istio-system
```

### Step 1.5: Apply Security Policies

```bash
# Apply peer authentication (mTLS)
kubectl apply -f k8s/service-mesh/security/peer-authentication.yaml

# Apply authorization policies
kubectl apply -f k8s/service-mesh/security/authorization-policies.yaml

# Verify policies
kubectl get peerauthentication -n data-platform
kubectl get authorizationpolicy -n data-platform
```

### Step 1.6: Configure Traffic Management

```bash
# Apply destination rules
kubectl apply -f k8s/service-mesh/traffic-management/destination-rules.yaml

# Apply virtual services
kubectl apply -f k8s/service-mesh/traffic-management/virtual-services.yaml

# Verify configuration
kubectl get destinationrule -n data-platform
kubectl get virtualservice -n data-platform
```

### Step 1.7: Update Network Policies

```bash
# Apply Istio-compatible network policies
kubectl apply -f k8s/service-mesh/network-policies-istio.yaml

# Verify policies
kubectl get networkpolicy -n data-platform
kubectl get networkpolicy -n istio-system
```

### Step 1.8: Restart Services for Sidecar Injection

```bash
# Restart all deployments in data-platform
kubectl rollout restart deployment -n data-platform

# Monitor rollout status
kubectl rollout status deployment -n data-platform

# Verify sidecars are injected
kubectl get pods -n data-platform -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].name}{"\n"}{end}'
```

### Step 1.9: Verify Service Mesh

```bash
# Check mesh status
istioctl proxy-status

# Verify mTLS
istioctl authn tls-check <pod-name>.<namespace>

# Check configuration
istioctl analyze -n data-platform

# Access Kiali dashboard
kubectl port-forward -n istio-system svc/kiali 20001:20001
# Open http://localhost:20001
```

## Phase 2: API Gateway Deployment (Day 3-4)

### Step 2.1: Deploy Kong Infrastructure

```bash
# Deploy Kong with PostgreSQL
kubectl apply -f k8s/api-gateway/kong-deployment.yaml

# Wait for PostgreSQL
kubectl wait --for=condition=ready --timeout=300s \
  pod -l app=kong-postgres -n kong

# Wait for migrations
kubectl wait --for=condition=complete --timeout=300s \
  job/kong-migrations -n kong

# Wait for Kong
kubectl wait --for=condition=available --timeout=300s \
  deployment/kong -n kong

# Verify deployment
kubectl get pods -n kong
```

### Step 2.2: Register Services with Kong

```bash
# Apply Kong services
kubectl apply -f k8s/api-gateway/kong-services.yaml

# Apply Kong routes
kubectl apply -f k8s/api-gateway/kong-routes.yaml

# Verify services and routes
kubectl get kongservice -n kong
kubectl get kongroute -n kong
```

### Step 2.3: Configure Plugins

```bash
# Apply Kong plugins
kubectl apply -f k8s/api-gateway/kong-plugins.yaml

# Verify plugins
kubectl get kongplugin -n kong
kubectl get kongclusterplugin
```

### Step 2.4: Verify API Gateway

```bash
# Port-forward to Kong admin API
kubectl port-forward -n kong svc/kong-admin 8001:8001

# List services
curl http://localhost:8001/services

# List routes
curl http://localhost:8001/routes

# Test API endpoint
curl http://localhost:8000/api/services/status
```

## Phase 3: Event-Driven Architecture (Day 5-6)

### Step 3.1: Create Kafka Topics

```bash
# Apply topic configurations
kubectl apply -f k8s/event-driven/kafka-topics.yaml

# Wait for job to complete
kubectl wait --for=condition=complete --timeout=300s \
  job/kafka-topics-creator -n data-platform

# Verify topics
kubectl exec -n data-platform kafka-0 -- \
  kafka-topics --bootstrap-server kafka-service:9092 --list
```

### Step 3.2: Register Event Schemas

```bash
# Upload schemas to Schema Registry
# (Assuming schema-registry service is available)

for schema in k8s/event-driven/schemas/*.avsc; do
  schema_name=$(basename $schema .avsc)
  curl -X POST http://schema-registry:8081/subjects/${schema_name}/versions \
    -H "Content-Type: application/vnd.schemaregistry.v1+json" \
    -d @${schema}
done

# Verify schemas
curl http://schema-registry:8081/subjects
```

### Step 3.3: Deploy Event Processors

```bash
# Deploy Flink jobs for stream processing
kubectl apply -f k8s/event-driven/flink-jobs/

# Verify Flink deployments
kubectl get flinkdeployment -n data-platform
```

### Step 3.4: Verify Event Infrastructure

```bash
# Check Kafka cluster health
kubectl exec -n data-platform kafka-0 -- \
  kafka-broker-api-versions --bootstrap-server kafka-service:9092

# Monitor consumer lag
kubectl exec -n data-platform kafka-0 -- \
  kafka-consumer-groups --bootstrap-server kafka-service:9092 --list
```

## Phase 4: Monitoring and Observability (Day 7)

### Step 4.1: Access Dashboards

```bash
# Kiali (Service Mesh Visualization)
kubectl port-forward -n istio-system svc/kiali 20001:20001
# Open http://localhost:20001

# Jaeger (Distributed Tracing)
kubectl port-forward -n istio-system svc/jaeger-query 16686:16686
# Open http://localhost:16686

# Kong Admin (API Gateway)
kubectl port-forward -n kong svc/kong-admin 8001:8001
# Open http://localhost:8001

# Grafana (Metrics)
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open http://localhost:3000
```

### Step 4.2: Verify Metrics Collection

```bash
# Check ServiceMonitors
kubectl get servicemonitor -n istio-system
kubectl get servicemonitor -n kong
kubectl get servicemonitor -n data-platform

# Query Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090
```

### Step 4.3: Configure Alerts

```bash
# Verify PrometheusRules
kubectl get prometheusrule -n monitoring

# Check AlertManager
kubectl get pods -n monitoring -l app=alertmanager
```

## Verification and Testing

### Service Mesh Verification

```bash
# Test mTLS between services
kubectl exec -n data-platform <pod1> -c app -- \
  curl -v http://service2.data-platform.svc.cluster.local:8080

# Check traffic routing
kubectl exec -n data-platform <pod> -c app -- \
  curl -H "x-test: canary" http://service.data-platform.svc.cluster.local:8080

# Verify circuit breaker
# (Simulate failures and check Kiali for ejected endpoints)
```

### API Gateway Verification

```bash
# Test rate limiting
for i in {1..100}; do
  curl -w "%{http_code}\n" -o /dev/null -s \
    http://kong-proxy/api/services
done

# Test authentication
curl -H "apikey: test-key" \
  http://kong-proxy/api/datahub/entities

# Check metrics
curl http://kong-admin:8001/metrics
```

### Event System Verification

```bash
# Produce test event
kubectl exec -n data-platform kafka-0 -- \
  kafka-console-producer --bootstrap-server kafka-service:9092 \
  --topic data-ingestion < test-event.json

# Consume events
kubectl exec -n data-platform kafka-0 -- \
  kafka-console-consumer --bootstrap-server kafka-service:9092 \
  --topic data-ingestion --from-beginning --max-messages 10

# Check consumer lag
kubectl exec -n data-platform kafka-0 -- \
  kafka-consumer-groups --bootstrap-server kafka-service:9092 \
  --group test-group --describe
```

## Performance Tuning

### Istio Sidecar Optimization

```yaml
# Add annotations to reduce sidecar resources
metadata:
  annotations:
    sidecar.istio.io/proxyCPU: "50m"
    sidecar.istio.io/proxyCPULimit: "200m"
    sidecar.istio.io/proxyMemory: "64Mi"
    sidecar.istio.io/proxyMemoryLimit: "256Mi"
```

### Kong Performance

```bash
# Increase worker processes
kubectl set env deployment/kong -n kong \
  KONG_NGINX_WORKER_PROCESSES=4

# Enable caching
kubectl apply -f k8s/api-gateway/kong-cache-plugin.yaml
```

### Kafka Optimization

```bash
# Increase batch size for producers
# Tune consumer fetch sizes
# Enable compression at topic level
```

## Rollback Procedures

### Rollback Service Mesh

```bash
# Remove sidecar injection label
kubectl label namespace data-platform istio-injection-

# Restart pods
kubectl rollout restart deployment -n data-platform

# Remove Istio
kubectl delete -f k8s/service-mesh/istio-config.yaml
kubectl delete -f k8s/service-mesh/istio-operator.yaml
```

### Rollback API Gateway

```bash
# Remove Kong routes
kubectl delete kongroute --all -n kong

# Remove Kong services
kubectl delete kongservice --all -n kong

# Delete Kong deployment
kubectl delete -f k8s/api-gateway/kong-deployment.yaml
```

## Troubleshooting

### Common Issues

#### Sidecars Not Injecting
```bash
# Check injection label
kubectl get namespace data-platform -o yaml | grep istio-injection

# Check webhook
kubectl get mutatingwebhookconfiguration

# Manual injection
istioctl kube-inject -f deployment.yaml | kubectl apply -f -
```

#### mTLS Connection Failures
```bash
# Check peer authentication
kubectl get peerauthentication -n data-platform

# Verify certificates
istioctl proxy-config secret <pod>.<namespace>

# Check destination rules
kubectl get destinationrule -n data-platform
```

#### Kong Database Connection Issues
```bash
# Check PostgreSQL
kubectl logs -n kong -l app=kong-postgres

# Check migrations
kubectl logs -n kong job/kong-migrations

# Reset database
kubectl delete job kong-migrations -n kong
kubectl apply -f k8s/api-gateway/kong-deployment.yaml
```

#### Kafka Topic Creation Failures
```bash
# Check Kafka connectivity
kubectl exec -n data-platform kafka-0 -- \
  kafka-broker-api-versions --bootstrap-server kafka-service:9092

# Manual topic creation
kubectl exec -n data-platform kafka-0 -- \
  kafka-topics --bootstrap-server kafka-service:9092 \
  --create --topic test-topic --partitions 3 --replication-factor 3
```

## Monitoring and Alerting

### Key Metrics to Monitor

**Service Mesh:**
- `istio_requests_total`
- `istio_request_duration_milliseconds`
- `istio_tcp_connections_opened_total`

**API Gateway:**
- `kong_http_status`
- `kong_latency`
- `kong_bandwidth`

**Event System:**
- `kafka_consumer_lag`
- `kafka_broker_topics_partitions`
- `kafka_producer_record_send_rate`

### Alerts

```yaml
# High consumer lag
- alert: KafkaConsumerLagHigh
  expr: kafka_consumer_lag > 10000
  for: 5m

# Service mesh high error rate
- alert: IstioHighErrorRate
  expr: rate(istio_requests_total{response_code=~"5.*"}[5m]) > 0.05
  for: 5m

# Kong API errors
- alert: KongHighErrorRate
  expr: rate(kong_http_status{code=~"5.*"}[5m]) > 0.05
  for: 5m
```

## Success Criteria

### Phase 1: Service Mesh
- ✅ All pods have Envoy sidecars
- ✅ mTLS enabled (PERMISSIVE mode initially)
- ✅ Kiali showing service graph
- ✅ Distributed tracing working
- ✅ No increase in error rates

### Phase 2: API Gateway
- ✅ Kong proxy responding
- ✅ All services registered
- ✅ Rate limiting working
- ✅ Metrics collection active
- ✅ <10ms added latency

### Phase 3: Events
- ✅ All topics created
- ✅ Schemas registered
- ✅ Events flowing through Kafka
- ✅ Consumer lag < 1000
- ✅ Stream processing jobs running

## Next Steps

1. **Gradually enable STRICT mTLS** mode per service
2. **Enable authentication** on Kong routes
3. **Implement event producers** in services
4. **Create custom Grafana dashboards**
5. **Document API contracts** in Kong dev portal
6. **Train team** on new architecture

## Support

For issues or questions:
1. Check logs: `kubectl logs -n <namespace> <pod> -c <container>`
2. Use istioctl: `istioctl analyze -n data-platform`
3. Review Kiali service graph for traffic issues
4. Check Kong admin API for configuration issues
5. Monitor Kafka consumer lag and broker health

---

**Deployment Status**: Ready for Production  
**Estimated Deployment Time**: 7 days (phased rollout)  
**Team Size Required**: 2-3 engineers  
**Risk Level**: Medium (gradual rollout mitigates risk)



