# DataHub Helm Chart Values - Simplified with default MySQL
# Reference: https://artifacthub.io/packages/helm/datahub/datahub
{{- $kafkaVals := default (dict) .Values.kafka }}
{{- $kafkaBootstrap := default "kafka-service:9092" (get $kafkaVals "bootstrapServers") }}
{{- $schemaRegistryUrl := default "http://schema-registry-service:8081" (get $kafkaVals "schemaRegistryUrl") }}

global:
  # Use existing Kafka
  kafka:
    bootstrap:
      server: {{ $kafkaBootstrap | quote }}
    schemaregistry:
      url: {{ $schemaRegistryUrl | quote }}
  
  # Use existing Elasticsearch
  elasticsearch:
    host: "elasticsearch-service"
    port: "9200"
    useSSL: "false"
  
  # Use existing Neo4j
  neo4j:
    host: "graphdb-service:7474"
    uri: "bolt://graphdb-service:7687"
    username: "neo4j"
    password:
      value: "datahub_password"

# Disable external dependencies
elasticsearch:
  enabled: false

neo4j:
  enabled: false

cp-helm-charts:
  enabled: false

kafka:
  enabled: false

# Keep PostgreSQL disabled
postgresql:
  enabled: false

# Enable MySQL with default settings (Helm chart will handle it)
# Not setting mysql.enabled - let chart use defaults

# GMS Configuration
datahub-gms:
  enabled: true
  replicaCount: 1

# Frontend Configuration  
datahub-frontend:
  enabled: true
  replicaCount: 1

# Consumers
datahub-mae-consumer:
  enabled: true

datahub-mce-consumer:
  enabled: true
