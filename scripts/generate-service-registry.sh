#!/usr/bin/env bash
set -euo pipefail

# Generate a services.json registry from Kubernetes ingress rules.
# Requires: yq (v4). Usage:
#   scripts/generate-service-registry.sh [INGRESS_FILE] [OUTPUT_JSON]

INGRESS_FILE="${1:-k8s/ingress/ingress-rules.yaml}"
OUT_FILE="${2:-prototype/terminal/services.json}"

if ! command -v yq >/dev/null 2>&1; then
  echo "error: 'yq' is required. Install yq v4 (https://mikefarah.gitbook.io/yq/)" >&2
  exit 1
fi

tmp=$(mktemp)

# Extract hosts and ingress names
yq -r '. as $i | select(.kind=="Ingress") | [(.spec.tls[].hosts[]? // empty), (.spec.rules[].host? // empty)] | unique | map({host: ., id: (split(".")[0]), ingress: $i.metadata.name}) | .[] | @base64' "$INGRESS_FILE" > "$tmp"

declare -A CATEGORY DISPLAY DESC

# Curated mappings for friendlier names and categories
CATEGORY[datahub]="Catalog";      DISPLAY[datahub]="DataHub";             DESC[datahub]="Data catalog and lineage platform."
CATEGORY[superset]="BI";          DISPLAY[superset]="Apache Superset";    DESC[superset]="Dashboards and data exploration."
CATEGORY[grafana]="Monitoring";   DISPLAY[grafana]="Grafana";             DESC[grafana]="Observability dashboards and alerts."
CATEGORY[doris]="OLAP";           DISPLAY[doris]="Apache Doris";          DESC[doris]="MPP analytics database frontend."
CATEGORY[trino]="SQL";            DISPLAY[trino]="Trino";                 DESC[trino]="SQL query engine coordinator UI."
CATEGORY[vault]="Security";       DISPLAY[vault]="Vault";                 DESC[vault]="Secrets management console."
CATEGORY[lakefs]="Data Lake";     DISPLAY[lakefs]="lakeFS";               DESC[lakefs]="Git-like data lake management."
CATEGORY[dolphin]="Orchestration";DISPLAY[dolphin]="DolphinScheduler";    DESC[dolphin]="Workflow and job scheduler."
CATEGORY[dolphinscheduler]="Orchestration"; DISPLAY[dolphinscheduler]="DolphinScheduler"; DESC[dolphinscheduler]="Workflow and job scheduler."
CATEGORY[minio]="Storage";        DISPLAY[minio]="MinIO Console";         DESC[minio]="Object storage management console."

icon_for() {
  case "$1" in
    datahub) echo "📚";; superset) echo "📊";; grafana) echo "📈";; doris) echo "🧮";;
    trino) echo "🧠";; vault) echo "🔐";; lakefs) echo "🌊";; dolphin|dolphinscheduler) echo "🐬";;
    minio) echo "🗄️";; *) echo "🔗";;
  esac
}

to_json_array() {
  local first=1
  echo "["
  while read -r line; do
    # decode base64 json {host,id,ingress}
    local obj=$(echo "$line" | base64 -d)
    local host=$(echo "$obj" | yq -r '.host')
    local id=$(echo "$obj" | yq -r '.id')
    local name=${DISPLAY[$id]:-}
    if [[ -z "$name" ]]; then
      # Capitalize id
      name=$(echo "$id" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
    fi
    local category=${CATEGORY[$id]:-Misc}
    local desc=${DESC[$id]:-"Service entry for $id."}
    local icon=$(icon_for "$id")
    local url="https://$host/"
    if [[ $first -eq 0 ]]; then echo ","; fi
    first=0
    printf '  {"id":"%s","name":"%s","url":"%s","category":"%s","description":"%s","icon":"%s"}' \
      "$id" "$name" "$url" "$category" "$desc" "$icon"
  done < "$tmp"
  echo; echo "]"
}

mkdir -p "$(dirname "$OUT_FILE")"
to_json_array > "$OUT_FILE"
echo "Wrote $OUT_FILE from $INGRESS_FILE"

