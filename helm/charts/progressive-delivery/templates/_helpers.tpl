{{/*
Expand the name of the chart.
*/}}
{{- define "progressive-delivery.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "progressive-delivery.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "progressive-delivery.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "progressive-delivery.labels" -}}
helm.sh/chart: {{ include "progressive-delivery.chart" . }}
{{ include "progressive-delivery.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "progressive-delivery.selectorLabels" -}}
app.kubernetes.io/name: {{ include "progressive-delivery.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "progressive-delivery.serviceAccountName" -}}
{{- if .Values.argoRollouts.serviceAccount.create }}
{{- default (include "progressive-delivery.fullname" .) .Values.argoRollouts.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.argoRollouts.serviceAccount.name }}
{{- end }}
{{- end }}
