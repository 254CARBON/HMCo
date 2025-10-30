# DataHub Helm Chart Values - Using Existing PostgreSQL (No Additional Storage)
# Reference: https://artifacthub.io/packages/helm/datahub/datahub

# Disable ALL prerequisite charts to avoid storage quota issues
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

# Disable setup jobs that expect prerequisites-mysql
elasticsearchSetupJob:
  enabled: false

mysqlSetupJob:
  enabled: false

postgresqlSetupJob:
  enabled: false

# Only enable DataHub components - let them initialize the database on first run
datahub-gms:
  enabled: true
  replicaCount: 1
  
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: "1"
      memory: 2Gi

datahub-frontend:
  enabled: true
  replicaCount: 1

datahub-mae-consumer:
  enabled: true

datahub-mce-consumer:
  enabled: true

