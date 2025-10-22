# DataHub Prerequisites Chart Values
# This deploys MySQL for DataHub

# Enable MySQL
mysql:
  enabled: true
  auth:
    rootPassword: "datahub_mysql_root"
    database: "datahub"
    username: "datahub"
    password: "datahub_mysql"
  primary:
    persistence:
      enabled: true
      storageClass: "local-path"
      size: 5Gi

# Disable other services (we're using existing ones)
elasticsearch:
  enabled: false

neo4j:
  enabled: false

kafka:
  enabled: false

cp-helm-charts:
  enabled: false

