# DataHub Helm Chart Values - Using Existing Infrastructure
# Reference: https://artifacthub.io/packages/helm/datahub/datahub
{{- $kafkaVals := default (dict) .Values.kafka }}
{{- $kafkaBootstrap := default "kafka-service:9092" (get $kafkaVals "bootstrapServers") }}
{{- $schemaRegistryUrl := default "http://schema-registry-service:8081" (get $kafkaVals "schemaRegistryUrl") }}

# Global configuration
global:
  # DataHub system configuration
  datahub:
    appVersion: "head"
    systemUpdate:
      enabled: true
  
  # PostgreSQL configuration (using existing postgres-shared)
  sql:
    datasource:
      host: "postgres-shared-service"
      port: "5432"
      url: "jdbc:postgresql://postgres-shared-service:5432/datahub"
      driver: "org.postgresql.Driver"
      username: "datahub"
      password:
        secretRef: "postgres-shared-secret"
        secretKey: "password"
  
  # Elasticsearch configuration (using existing service)
  elasticsearch:
    host: "elasticsearch-service"
    port: "9200"
    useSSL: "false"
  
  # Neo4j configuration (using existing service)
  graph:
    neo4j:
      host: "graphdb-service:7474"
      uri: "bolt://graphdb-service:7687"
      username: "neo4j"
      password:
        value: "datahub_password"
  
  # Kafka configuration (using existing service)
  kafka:
    bootstrap:
      server: {{ $kafkaBootstrap | quote }}
    schemaregistry:
      url: {{ $schemaRegistryUrl | quote }}

# Disable all prerequisite charts (using existing infrastructure)
elasticsearch:
  enabled: false

neo4j:
  enabled: false

mysql:
  enabled: false

postgresql:
  enabled: false

cp-helm-charts:
  enabled: false

kafka:
  enabled: false

# DataHub GMS (Graph Metadata Service)
datahub-gms:
  enabled: true
  replicaCount: 1
  
  image:
    repository: acryldata/datahub-gms
    tag: "head"
  
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: "1"
      memory: 2Gi
  
  # S3/MinIO configuration
  extraEnvs:
    - name: AWS_REGION
      value: "us-east-1"
    - name: AWS_ENDPOINT_URL
      value: "http://minio-service:9000"
    - name: S3_ENDPOINT
      value: "http://minio-service:9000"
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: minio-secret
          key: access-key
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: minio-secret
          key: secret-key
    - name: DATAHUB_SECRET
      valueFrom:
        secretKeyRef:
          name: datahub-secret
          key: DATAHUB_SECRET

# DataHub Frontend
datahub-frontend:
  enabled: true
  replicaCount: 1
  
  image:
    repository: acryldata/datahub-frontend-react
    tag: "head"
  
  resources:
    requests:
      cpu: 250m
      memory: 512Mi
    limits:
      cpu: 500m
      memory: 1Gi
  
  extraEnvs:
    - name: DATAHUB_SECRET
      valueFrom:
        secretKeyRef:
          name: datahub-secret
          key: DATAHUB_SECRET

# MAE Consumer
datahub-mae-consumer:
  enabled: true
  replicaCount: 1
  
  image:
    repository: acryldata/datahub-mae-consumer
    tag: "head"
  
  resources:
    requests:
      cpu: 250m
      memory: 512Mi
    limits:
      cpu: 500m
      memory: 1Gi
  
  extraEnvs:
    - name: AWS_REGION
      value: "us-east-1"
    - name: AWS_ENDPOINT_URL
      value: "http://minio-service:9000"
    - name: S3_ENDPOINT
      value: "http://minio-service:9000"
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: minio-secret
          key: access-key
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: minio-secret
          key: secret-key

# MCE Consumer
datahub-mce-consumer:
  enabled: true
  replicaCount: 1
  
  image:
    repository: acryldata/datahub-mce-consumer
    tag: "head"
  
  resources:
    requests:
      cpu: 250m
      memory: 512Mi
    limits:
      cpu: 500m
      memory: 1Gi
  
  extraEnvs:
    - name: AWS_REGION
      value: "us-east-1"
    - name: AWS_ENDPOINT_URL
      value: "http://minio-service:9000"
    - name: S3_ENDPOINT
      value: "http://minio-service:9000"
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: minio-secret
          key: access-key
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: minio-secret
          key: secret-key

# System update job (handles database initialization)
datahubSystemUpdate:
  enabled: true

# Elasticsearch setup job
elasticsearchSetupJob:
  enabled: true
  extraEnvs:
    - name: USE_AWS_ELASTICSEARCH
      value: "false"

# MySQL setup job (disabled - using PostgreSQL)
mysqlSetupJob:
  enabled: false

# PostgreSQL setup job
postgresqlSetupJob:
  enabled: true

# Kafka setup job  
kafkaSetupJob:
  enabled: true
