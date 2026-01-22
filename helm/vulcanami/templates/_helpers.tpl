{{/*
Expand the name of the chart.
*/}}
{{- define "vulcanami.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "vulcanami.fullname" -}}
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
{{- define "vulcanami.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "vulcanami.labels" -}}
helm.sh/chart: {{ include "vulcanami.chart" . }}
{{ include "vulcanami.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "vulcanami.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vulcanami.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "vulcanami.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "vulcanami.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate the full image reference with optional digest pinning.
SECURITY: Supports image digest pinning for production immutability.

Usage:
  {{ include "vulcanami.image" . }}

Examples:
  - tag only: ghcr.io/org/image:v1.0.0
  - tag with digest: ghcr.io/org/image:v1.0.0@sha256:abc123...
  - tag@digest in tag field: ghcr.io/org/image:v1.0.0@sha256:abc123...
  - separate digest field: ghcr.io/org/image:v1.0.0@sha256:def456...
*/}}
{{- define "vulcanami.image" -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- $digest := .Values.image.digest | default "" -}}
{{- if $digest -}}
  {{- printf "%s:%s@%s" $repository $tag $digest -}}
{{- else if contains "@sha256:" $tag -}}
  {{- printf "%s:%s" $repository $tag -}}
{{- else -}}
  {{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end }}
