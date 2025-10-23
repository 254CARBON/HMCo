# 254Carbon Platform - Complete Deployment Guide

**Last Updated**: October 22, 2025  
**Status**: Production Ready

---

## Quick Deployment

For the Advanced Analytics Platform, use the automated deployment script:

```bash
# Deploy everything
./scripts/deploy-advanced-analytics-platform.sh

# Deploy specific phase
./scripts/deploy-advanced-analytics-platform.sh --phase=1

# Deploy specific component
./scripts/deploy-advanced-analytics-platform.sh --component=ray-serve
```

---

## Component Deployment Guides

### Core Platform (Already Deployed)
- **Service Mesh (Istio)**: See `SERVICE_INTEGRATION_QUICKSTART.md`
- **API Gateway (Kong)**: See `k8s/api-gateway/README.md`
- **Event System (Kafka)**: See `k8s/event-driven/README.md`
- **Commodity Platform**: See `COMMODITY_QUICKSTART.md`

### Advanced Analytics Platform (New)

#### Phase 1: Real-time ML Pipeline
- **Ray Serve**: `k8s/ml-platform/ray-serve/README.md`
- **Feast**: `k8s/ml-platform/feast-feature-store/README.md`
- **CDC Connectors**: `k8s/streaming/connectors/README.md`

#### Phase 2: ML/AI Platform
- **Kubeflow**: `k8s/ml-platform/kubeflow/README.md`
- **Seldon Core**: `k8s/ml-platform/seldon-core/README.md`

#### Phase 3-5: Governance, Observability, UX
- See `ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md` for complete details

---

## Post-Deployment

1. **Verify Services**
   ```bash
   kubectl get pods -A | grep -E "ray-serve|feast|kubeflow|seldon"
   ```

2. **Access UIs**
   - Ray Serve: `kubectl port-forward -n data-platform svc/ray-serve-service 8265:8265`
   - Kubeflow: `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`
   - Feast: `kubectl port-forward -n data-platform svc/feast-server 8080:8080`

3. **Run Tests**
   - See individual component READMEs for testing procedures

4. **Monitor**
   - Grafana: https://grafana.254carbon.com
   - Prometheus: https://prometheus.254carbon.com

---

## Troubleshooting

For component-specific issues, refer to:
- Component README files in `k8s/*/README.md`
- `ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md`
- Platform logs: `kubectl logs -n <namespace> <pod-name>`

---

## Documentation Index

### Quick Start Guides
- `SERVICE_INTEGRATION_QUICKSTART.md` - Service mesh, API gateway, events
- `COMMODITY_QUICKSTART.md` - Commodity data platform
- `ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md` - ML/AI platform

### Component Documentation
All components have detailed README files in their respective directories:
- `k8s/ml-platform/*/README.md`
- `k8s/streaming/connectors/README.md`
- `k8s/api-gateway/README.md`
- `k8s/event-driven/README.md`

### Architecture & Planning
- `advanced-analytics-platform.plan.md` - Implementation plan
- `KIND_TO_BARE_METAL_MIGRATION_PLAN.md` - Infrastructure migration

---

**For detailed implementation summaries, see:**
- `ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md` (Latest)
- Main `README.md` (Platform overview)




