{{/*
Fully qualified app name
*/}}
{{- define "jupyterhub.fullname" -}}
jupyterhub
{{- end }}

{{/*
Chart name
*/}}
{{- define "jupyterhub.name" -}}
jupyterhub
{{- end }}

{{/*
Chart version
*/}}
{{- define "jupyterhub.chart" -}}
jupyterhub-0.1.0
{{- end }}

{{/*
Common labels
*/}}
{{- define "jupyterhub.labels" -}}
app.kubernetes.io/name: jupyterhub
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: "1.0"
app.kubernetes.io/managed-by: Helm
{{- end }}

{{/*
Selector labels
*/}}
{{- define "jupyterhub.selectorLabels" -}}
app.kubernetes.io/name: jupyterhub
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "jupyterhub.serviceAccountName" -}}
jupyterhub
{{- end }}
