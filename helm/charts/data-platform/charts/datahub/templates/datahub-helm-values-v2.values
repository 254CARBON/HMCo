# DataHub Helm Chart Values - Using Existing PostgreSQL
#
# Reference: https://artifacthub.io/packages/helm/datahub/datahub
{{- $kafkaVals := default (dict) .Values.kafka }}
{{- $kafkaBootstrap := default "kafka-service:9092" (get $kafkaVals "bootstrapServers") }}
{{- $schemaRegistryUrl := default "http://schema-registry-service:8081" (get $kafkaVals "schemaRegistryUrl") }}

global:
  # Database Configuration - PostgreSQL
  sql:
    datasource:
      host: "postgres-shared-service:5432"
      hostForpostgresqlClient: "postgres-shared-service:5432"
      port: "5432"
      url: "jdbc:postgresql://postgres-shared-service:5432/datahub?sslmode=disable"
      driver: "org.postgresql.Driver"
      username: "datahub"
      password:
        secretRef: postgres-shared-secret
        secretKey: password
  
  # Kafka Configuration
  kafka:
    bootstrap:
      server: {{ $kafkaBootstrap | quote }}
    schemaregistry:
      url: {{ $schemaRegistryUrl | quote }}
  
  # Elasticsearch Configuration
  elasticsearch:
    host: "elasticsearch-service"
    port: "9200"
    useSSL: "false"
  
  # Neo4j Configuration  
  neo4j:
    host: "graphdb-service:7474"
    uri: "bolt://graphdb-service:7687"
    username: "neo4j"
    password:
      value: "datahub_password"
  
  datahub:
    systemUpdate:
      enabled: true

# Disable all dependency charts
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

# Disable prerequisites setup - we're using existing services
elasticsearchSetupJob:
  enabled: false

mysqlSetupJob:
  enabled: false

postgresqlSetupJob:
  enabled: false

kafkaSetupJob:
  enabled: false

# GMS Configuration
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

# Frontend Configuration  
datahub-frontend:
  enabled: true
  replicaCount: 1
  
  image:
    repository: acryldata/datahub-frontend-react
    tag: "head"

# Consumers
datahub-mae-consumer:
  enabled: true
  image:
    repository: acryldata/datahub-mae-consumer
    tag: "head"

datahub-mce-consumer:
  enabled: true
  image:
    repository: acryldata/datahub-mce-consumer
    tag: "head"
