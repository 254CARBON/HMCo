# Schema Registry Setup Guide

**Service**: Confluent Schema Registry  
**Version**: 7.4.0  
**Status**: ✅ Operational  
**Deployment Date**: October 21, 2025

---

## Overview

Schema Registry provides a centralized repository for managing and validating schemas for Apache Kafka. It ensures data compatibility and evolution across Kafka topics, which is critical for DataHub metadata events.

### Service Details
- **Image**: `confluentinc/cp-schema-registry:7.4.0`
- **Replicas**: 1
- **Namespace**: `data-platform`
- **Service Name**: `schema-registry-service`
- **Port**: 8081
- **Protocol**: HTTP/REST

---

## Architecture

```
┌─────────────────┐
│  Kafka Topics   │
│  (_schemas)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────┐
│ Schema Registry │◄──────│  DataHub MCE     │
│  (Port 8081)    │       │  Consumer        │
└─────────────────┘       └──────────────────┘
```

### Key Features
- Schema validation and compatibility checking
- RESTful API for schema management
- Integration with Kafka for schema storage
- Support for Avro, JSON Schema, and Protobuf formats

---

## Configuration

### Environment Variables

```yaml
- name: SCHEMA_REGISTRY_HOST_NAME
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
      
- name: SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS
  value: "kafka-service.data-platform.svc.cluster.local:9092"
  
- name: SCHEMA_REGISTRY_KAFKASTORE_TOPIC
  value: "_schemas"
  
- name: SCHEMA_REGISTRY_KAFKASTORE_TOPIC_REPLICATION_FACTOR
  value: "1"
  
- name: SCHEMA_REGISTRY_LISTENERS
  value: "http://0.0.0.0:8081"
  
- name: SCHEMA_REGISTRY_DEBUG
  value: "true"
```

### Security Context

Schema Registry runs with restricted Pod Security settings:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
```

Container-level security:

```yaml
securityContext:
  allowPrivilegeEscalation: false
  runAsNonRoot: true
  runAsUser: 1000
  capabilities:
    drop:
    - ALL
  seccompProfile:
    type: RuntimeDefault
```

### Init Container

The deployment includes an init container that waits for Kafka to be ready:

```yaml
initContainers:
- name: wait-for-kafka
  image: busybox:1.35
  command: ['sh', '-c', "echo 'Waiting for Kafka...' && for i in {1..60}; do nc -z kafka-service.data-platform.svc.cluster.local 9092 && echo 'Kafka is ready!' && exit 0; sleep 2; done; echo 'Kafka startup timeout'; exit 1"]
```

---

## API Reference

### Base URL
```
http://schema-registry-service.data-platform.svc.cluster.local:8081
```

### Common Endpoints

#### List All Subjects
```bash
curl http://schema-registry-service:8081/subjects
```

#### Get Schema by ID
```bash
curl http://schema-registry-service:8081/schemas/ids/1
```

#### Register New Schema
```bash
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data '{"schema": "{\"type\":\"string\"}"}' \
  http://schema-registry-service:8081/subjects/test-value/versions
```

#### Check Compatibility
```bash
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data '{"schema": "{\"type\":\"string\"}"}' \
  http://schema-registry-service:8081/compatibility/subjects/test-value/versions/latest
```

---

## Health Checks

### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /subjects
    port: 8081
    scheme: HTTP
  initialDelaySeconds: 10
  periodSeconds: 10
```

### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /subjects
    port: 8081
    scheme: HTTP
  initialDelaySeconds: 60
  periodSeconds: 30
```

### Manual Health Check
```bash
kubectl exec -it -n data-platform deploy/schema-registry -- \
  curl http://localhost:8081/subjects
```

---

## Integration with DataHub

DataHub MCE (Metadata Change Event) Consumer uses Schema Registry to validate and process metadata events.

### DataHub Configuration

```yaml
env:
- name: KAFKA_SCHEMAREGISTRY_URL
  value: "http://schema-registry-service:8081"
```

### Verification

Check DataHub MCE Consumer logs:

```bash
kubectl logs -n data-platform -l app=datahub-mce-consumer --tail=50 | grep schema-registry
```

Expected output:
```
Ready: http://schema-registry-service:8081.
schema.registry.url = [http://schema-registry-service:8081]
```

---

## Troubleshooting

### Pod Not Starting

**Check Pod Status**:
```bash
kubectl get pods -n data-platform -l app=schema-registry
```

**View Logs**:
```bash
kubectl logs -n data-platform -l app=schema-registry
```

**Check Events**:
```bash
kubectl describe pod -n data-platform -l app=schema-registry
```

### Connection Refused

If services report "connection refused" to Schema Registry:

1. **Verify Pod is Running**:
   ```bash
   kubectl get pods -n data-platform -l app=schema-registry
   ```

2. **Check Service**:
   ```bash
   kubectl get svc schema-registry-service -n data-platform
   ```

3. **Test Connectivity**:
   ```bash
   kubectl run -it --rm debug --image=busybox --restart=Never -n data-platform -- \
     nc -zv schema-registry-service 8081
   ```

### Kafka Connection Issues

**Check Init Container Logs**:
```bash
kubectl logs -n data-platform -l app=schema-registry -c wait-for-kafka
```

**Verify Kafka Service**:
```bash
kubectl get svc kafka-service -n data-platform
```

**Test Kafka Connectivity**:
```bash
kubectl run -it --rm kafka-test --image=busybox --restart=Never -n data-platform -- \
  nc -zv kafka-service 9092
```

---

## Monitoring

### Prometheus Metrics

Schema Registry exposes JMX metrics that can be scraped by Prometheus. Add to ServiceMonitor:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: schema-registry
  namespace: data-platform
spec:
  selector:
    matchLabels:
      app: schema-registry
  endpoints:
  - port: schema-registry
    path: /metrics
```

### Key Metrics to Monitor

- `kafka_schema_registry_registered_count` - Number of registered schemas
- `kafka_schema_registry_request_rate` - Request rate
- `kafka_schema_registry_error_rate` - Error rate
- `jvm_memory_used_bytes` - JVM memory usage

### Grafana Dashboard

Create alerts for:
- Schema Registry pod not ready
- High error rate (>5%)
- JVM memory usage >80%
- Kafka connection failures

---

## Backup and Recovery

### Schema Backup

Schemas are stored in Kafka topic `_schemas`. To back up:

```bash
# Export all schemas
kubectl exec -it -n data-platform deploy/schema-registry -- \
  curl http://localhost:8081/subjects | jq -r '.[]' | \
  while read subject; do
    echo "Backing up $subject"
    curl "http://schema-registry-service:8081/subjects/$subject/versions/latest" \
      > "schema-backup-$subject.json"
  done
```

### Recovery

To restore schemas:

```bash
# Restore from backup
for file in schema-backup-*.json; do
  subject=$(echo $file | sed 's/schema-backup-//;s/.json//')
  curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
    --data @$file \
    "http://schema-registry-service:8081/subjects/$subject/versions"
done
```

---

## Security Considerations

### Network Policies

Schema Registry is protected by network policies:

```yaml
# Allow DataHub MCE Consumer
- from:
  - podSelector:
      matchLabels:
        app: datahub-mce-consumer
  ports:
  - protocol: TCP
    port: 8081

# Allow DataHub GMS
- from:
  - podSelector:
      matchLabels:
        app: datahub-gms
  ports:
  - protocol: TCP
    port: 8081
```

### RBAC

Schema Registry runs with minimal permissions:
- No cluster-level access
- Limited to data-platform namespace
- Read-only access to ConfigMaps

### Authentication

For production, enable authentication:

```yaml
env:
- name: SCHEMA_REGISTRY_AUTHENTICATION_METHOD
  value: "BASIC"
- name: SCHEMA_REGISTRY_AUTHENTICATION_ROLES
  value: "admin,developer,reader"
- name: SCHEMA_REGISTRY_AUTHENTICATION_REALM
  value: "SchemaRegistry"
```

---

## Performance Tuning

### Resource Limits

Current configuration:

```yaml
resources:
  limits:
    memory: "1Gi"
    cpu: "500m"
  requests:
    memory: "512Mi"
    cpu: "250m"
```

### JVM Tuning

For high-load environments, increase heap:

```yaml
env:
- name: SCHEMA_REGISTRY_HEAP_OPTS
  value: "-Xms1024m -Xmx2048m"
```

### Caching

Enable schema caching:

```yaml
env:
- name: SCHEMA_REGISTRY_SCHEMA_CACHE_SIZE
  value: "1000"
- name: SCHEMA_REGISTRY_SCHEMA_CACHE_EXPIRY_SECS
  value: "3600"
```

---

## References

### Documentation
- [Confluent Schema Registry Docs](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Schema Registry API Reference](https://docs.confluent.io/platform/current/schema-registry/develop/api.html)
- [DataHub Integration](https://datahubproject.io/docs/metadata-ingestion/)

### Configuration Files
- Deployment: `k8s/shared/kafka/schema-registry.yaml`
- Service: Included in deployment manifest
- Network Policy: `k8s/networking/data-platform-network-policies.yaml`

### Related Services
- Kafka: `k8s/shared/kafka/kafka.yaml`
- DataHub: `k8s/datahub/datahub.yaml`
- Zookeeper: `k8s/shared/zookeeper/zookeeper.yaml`

---

**Last Updated**: October 21, 2025  
**Maintained By**: Platform Team  
**Status**: Production Ready ✅


