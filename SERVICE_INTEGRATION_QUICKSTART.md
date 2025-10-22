# Service Integration - Quick Reference

**Platform**: 254Carbon Data Platform  
**Version**: 1.0.0  
**Date**: October 22, 2025

---

## 🚀 Quick Deploy

```bash
# 1. Service Mesh (5 minutes)
kubectl apply -f k8s/service-mesh/istio-operator.yaml
kubectl apply -f k8s/service-mesh/istio-config.yaml
kubectl label namespace data-platform istio-injection=enabled
kubectl rollout restart deployment -n data-platform

# 2. API Gateway (5 minutes)
kubectl apply -f k8s/api-gateway/kong-deployment.yaml
kubectl apply -f k8s/api-gateway/kong-services.yaml
kubectl apply -f k8s/api-gateway/kong-routes.yaml

# 3. Event Infrastructure (2 minutes)
kubectl apply -f k8s/event-driven/kafka-topics.yaml

# 4. Observability (2 minutes)
kubectl apply -f k8s/service-mesh/observability/
```

## 🔍 Quick Check

```bash
# Service Mesh
kubectl get pods -n istio-system
istioctl proxy-status

# API Gateway
kubectl get pods -n kong
curl http://kong-admin:8001/status

# Events
kubectl exec kafka-0 -- kafka-topics --list

# Health
kubectl get pods -n data-platform | grep -c "2/2"
```

## 📊 Dashboards

| Tool | URL | Purpose |
|------|-----|---------|
| Kiali | https://kiali.254carbon.com | Service mesh visualization |
| Jaeger | https://jaeger.254carbon.com | Distributed tracing |
| Kong | https://kong.254carbon.com | API gateway admin |
| Grafana | https://grafana.254carbon.com | Metrics dashboards |

## 🔧 Common Commands

### Service Mesh

```bash
# Check sidecar injection
kubectl get pods -n data-platform -o wide

# Verify mTLS
istioctl authn tls-check <pod>.<namespace>

# Analyze configuration
istioctl analyze -n data-platform

# Check proxy status
istioctl proxy-status
```

### API Gateway

```bash
# Port-forward to admin API
kubectl port-forward -n kong svc/kong-admin 8001:8001

# List services
curl http://localhost:8001/services

# Test rate limiting
for i in {1..100}; do curl http://kong-proxy/api/services; done

# Check metrics
curl http://localhost:8001/metrics | grep kong_
```

### Events

```bash
# List topics
kubectl exec kafka-0 -- kafka-topics --bootstrap-server kafka-service:9092 --list

# Check consumer lag
kubectl exec kafka-0 -- kafka-consumer-groups --bootstrap-server kafka-service:9092 --describe --group <group>

# Produce test event
echo '{"test":"data"}' | kubectl exec -i kafka-0 -- kafka-console-producer --broker-list kafka-service:9092 --topic data-ingestion

# Consume events
kubectl exec -it kafka-0 -- kafka-console-consumer --bootstrap-server kafka-service:9092 --topic data-ingestion --from-beginning
```

## 🐛 Troubleshooting

### Sidecars Not Injecting

```bash
# Check label
kubectl get namespace data-platform --show-labels

# Manual injection
istioctl kube-inject -f deployment.yaml | kubectl apply -f -
```

### Service Unreachable

```bash
# Check virtual services
kubectl get virtualservice -n data-platform

# Check destination rules
kubectl get destinationrule -n data-platform

# Test from pod
kubectl exec -it <pod> -c app -- curl http://service:8080
```

### Kong Issues

```bash
# Check logs
kubectl logs -n kong -l app=kong

# Reset database
kubectl delete job kong-migrations -n kong
kubectl apply -f k8s/api-gateway/kong-deployment.yaml
```

### Event Issues

```bash
# Check Kafka health
kubectl exec kafka-0 -- kafka-broker-api-versions --bootstrap-server kafka-service:9092

# Describe topic
kubectl exec kafka-0 -- kafka-topics --bootstrap-server kafka-service:9092 --describe --topic <topic>
```

## 📈 Key Metrics

```promql
# Service mesh request rate
rate(istio_requests_total[5m])

# API gateway latency
histogram_quantile(0.99, kong_latency_bucket)

# Consumer lag
kafka_consumer_lag

# Error rate
rate(istio_requests_total{response_code=~"5.."}[5m])
```

## 🔐 Security

```bash
# Check mTLS mode
kubectl get peerauthentication -n data-platform

# Verify certificates
istioctl proxy-config secret <pod>.<namespace>

# Check authorization policies
kubectl get authorizationpolicy -n data-platform
```

## 📚 Documentation

- **Service Mesh**: `k8s/service-mesh/README.md`
- **API Gateway**: `k8s/api-gateway/README.md`
- **Events**: `k8s/event-driven/README.md`
- **Deployment Guide**: `SERVICE_INTEGRATION_DEPLOYMENT_GUIDE.md`
- **Implementation Summary**: `SERVICE_INTEGRATION_IMPLEMENTATION_COMPLETE.md`

## 🆘 Support

1. Check logs: `kubectl logs <pod> -c <container>`
2. Run diagnostics: `istioctl analyze`
3. Review dashboards: Kiali, Jaeger, Grafana
4. Check documentation in `k8s/*/README.md`

## 🎯 Success Criteria

- [ ] All pods have 2 containers (app + istio-proxy)
- [ ] Kiali shows service graph
- [ ] Jaeger shows traces
- [ ] Kong admin API responds
- [ ] All Kafka topics exist
- [ ] No increase in error rates
- [ ] Latency < 10ms added

## ⚡ Performance Tuning

```yaml
# Reduce sidecar resources
metadata:
  annotations:
    sidecar.istio.io/proxyCPU: "50m"
    sidecar.istio.io/proxyMemory: "64Mi"

# Increase Kong workers
env:
- name: KONG_NGINX_WORKER_PROCESSES
  value: "4"

# Tune Kafka consumers
fetch.min.bytes: 1048576
fetch.max.wait.ms: 500
```

## 🔄 Rollback

```bash
# Remove service mesh
kubectl label namespace data-platform istio-injection-
kubectl rollout restart deployment -n data-platform
kubectl delete -f k8s/service-mesh/

# Remove API gateway
kubectl delete -f k8s/api-gateway/

# Topics remain (no rollback needed)
```

---

**Quick Start Time**: ~15 minutes  
**Full Deployment Time**: ~2 hours  
**Team Training**: 1-2 days



