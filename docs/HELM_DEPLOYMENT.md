# VulcanAMI Helm Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Image Digest Pinning](#image-digest-pinning)
3. [Required Values Validation](#required-values-validation)
4. [Deployment Examples](#deployment-examples)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools
- Kubernetes 1.23+ cluster
- Helm 3.8+ (for values schema validation)
- kubectl configured for target cluster
- Access to image registry (ghcr.io)

### Required Secrets
Before deployment, create the following Kubernetes secrets:

```bash
# JWT secret (min 32 characters)
kubectl create secret generic vulcanami-secrets \
  --from-literal=jwt-secret-key=$(openssl rand -base64 32) \
  --namespace vulcanami

# Database credentials
kubectl create secret generic postgres-secrets \
  --from-literal=postgres-password=$(openssl rand -base64 32) \
  --namespace vulcanami

# Optional: Image pull secret for private registry
kubectl create secret docker-registry ghcr-credentials \
  --docker-server=ghcr.io \
  --docker-username=<github-username> \
  --docker-password=<github-pat> \
  --namespace vulcanami
```

---

## Image Digest Pinning

### Why Digest Pinning?

**Security Risk**: Image tags (e.g., `v1.0.0`) are mutable and can be re-pushed with different content.

**Solution**: Use cryptographic digests for immutability:
```
ghcr.io/org/image:v1.0.0@sha256:abc123...
```

### Getting Image Digests

```bash
# Method 1: Pull and inspect
docker pull ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0

# Method 2: Use crane (no pull required)
crane digest ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0

# Method 3: Use skopeo
skopeo inspect docker://ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0 | jq -r '.Digest'
```

### Deployment with Digests

**Option 1: Tag with embedded digest**
```bash
helm install vulcanami ./helm/vulcanami \
  --set image.tag="v1.0.0@sha256:abc123def456..." \
  --namespace vulcanami
```

**Option 2: Separate digest field**
```bash
helm install vulcanami ./helm/vulcanami \
  --set image.tag="v1.0.0" \
  --set image.digest="sha256:abc123def456..." \
  --namespace vulcanami
```

**Option 3: values.yaml**
```yaml
image:
  repository: ghcr.io/musicmonk42/vulcanami_llm-api
  tag: "v1.0.0"
  digest: "sha256:abc123def456..."
```

---

## Required Values Validation

### Helm Values Schema

The chart includes `values.schema.json` that enforces:

✅ **REQUIRED**: Image tag must be set (cannot be `REPLACE_ME`)  
✅ **REQUIRED**: Tag cannot be `latest`, `stable`, or `main`  
✅ **REQUIRED**: Resource limits and requests must be defined  
✅ **REQUIRED**: Security contexts must be properly configured  
✅ **ENFORCED**: `runAsNonRoot: true` (cannot be disabled)  
✅ **ENFORCED**: `readOnlyRootFilesystem: true` (cannot be disabled)  
✅ **ENFORCED**: `allowPrivilegeEscalation: false` (cannot be disabled)  

### Validation Commands

```bash
# Lint chart (validates schema)
helm lint ./helm/vulcanami --values prod-values.yaml --strict

# Dry-run to see rendered manifests
helm install vulcanami ./helm/vulcanami \
  --values prod-values.yaml \
  --dry-run --debug \
  --namespace vulcanami

# Template and validate with kubeval
helm template vulcanami ./helm/vulcanami --values prod-values.yaml \
  | kubeval --strict
```

### Common Validation Errors

❌ **Error**: "image.tag: Does not match pattern"
```
image:
  tag: "latest"  # ❌ Not allowed in production
```
✅ **Fix**:
```
image:
  tag: "v1.0.0"  # ✅ Specific version required
```

❌ **Error**: "image.tag is required"
```
image:
  tag: "REPLACE_ME"  # ❌ Placeholder not replaced
```
✅ **Fix**:
```bash
helm install --set image.tag=v1.0.0 ...
```

❌ **Error**: "resources.limits is required"
```
resources: {}  # ❌ No limits defined
```
✅ **Fix**:
```
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi
```

---

## Deployment Examples

### Development Environment

```bash
helm install vulcanami ./helm/vulcanami \
  --values helm/vulcanami/values.yaml \
  --set image.tag=v1.0.0-dev \
  --set autoscaling.enabled=false \
  --set replicaCount=1 \
  --set ingress.enabled=false \
  --namespace vulcanami-dev \
  --create-namespace
```

### Staging Environment

```bash
helm install vulcanami ./helm/vulcanami \
  --values environments/staging-values.yaml \
  --set image.tag="v1.0.0@sha256:abc123..." \
  --set ingress.hosts[0].host=staging-api.vulcanami.io \
  --namespace vulcanami-staging \
  --create-namespace \
  --wait --timeout 10m
```

### Production Environment (with digest pinning)

```bash
# Get latest digest
DIGEST=$(crane digest ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0)

# Deploy with digest pinning
helm install vulcanami ./helm/vulcanami \
  --values environments/production-values.yaml \
  --set image.tag="v1.0.0" \
  --set image.digest="${DIGEST}" \
  --set ingress.hosts[0].host=api.vulcanami.io \
  --set ingress.tls[0].secretName=vulcanami-tls \
  --set ingress.tls[0].hosts[0]=api.vulcanami.io \
  --namespace vulcanami-prod \
  --create-namespace \
  --wait --timeout 15m
```

### Upgrade with Zero Downtime

```bash
# Ensure PodDisruptionBudget allows rolling updates
helm upgrade vulcanami ./helm/vulcanami \
  --values environments/production-values.yaml \
  --set image.tag="v1.1.0" \
  --set image.digest="sha256:newdigest..." \
  --namespace vulcanami-prod \
  --wait --timeout 15m \
  --atomic --cleanup-on-fail
```

---

## Security Best Practices

### 1. Always Use Digest Pinning in Production
```yaml
# ❌ BAD: Tags can be re-pushed
image:
  tag: "v1.0.0"

# ✅ GOOD: Digest guarantees immutability
image:
  tag: "v1.0.0"
  digest: "sha256:abc123..."
```

### 2. Enforce TLS for Ingress
```yaml
ingress:
  enabled: true
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  tls:
    - secretName: vulcanami-tls
      hosts:
        - api.vulcanami.io
```

### 3. Set Resource Limits
```yaml
resources:
  limits:
    cpu: 2000m      # Hard limit
    memory: 4Gi
  requests:
    cpu: 500m       # Guaranteed resources
    memory: 1Gi
```

### 4. Enable Pod Disruption Budget
```yaml
podDisruptionBudget:
  enabled: true
  minAvailable: 1  # At least 1 pod always running
```

### 5. Use Network Policies
```yaml
networkPolicy:
  enabled: true
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
  egress:
    - to:
      - namespaceSelector: {}
      ports:
      - protocol: TCP
        port: 443  # HTTPS
      - protocol: UDP
        port: 53   # DNS
```

---

## Troubleshooting

### Image Pull Errors

```bash
# Check image pull secret
kubectl get secret ghcr-credentials -n vulcanami

# Test image pull manually
kubectl run test-pull --image=ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0 \
  --image-pull-policy=Always --restart=Never -n vulcanami

# Check events
kubectl get events -n vulcanami --sort-by='.lastTimestamp'
```

### Digest Mismatch

```
Error: Failed to pull image "...": rpc error: manifest unknown
```

**Cause**: Digest doesn't match the tag in the registry.

**Solution**: Re-fetch the digest:
```bash
# Get current digest
crane digest ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0

# Update deployment
helm upgrade vulcanami ./helm/vulcanami \
  --set image.digest="<new-digest>" \
  --reuse-values
```

### Health Check Failures

```bash
# Check pod logs
kubectl logs -n vulcanami -l app.kubernetes.io/name=vulcanami --tail=100

# Check readiness probe
kubectl describe pod -n vulcanami -l app.kubernetes.io/name=vulcanami | grep -A 10 "Readiness"

# Test health endpoint manually
kubectl port-forward -n vulcanami svc/vulcanami 8000:8000
curl http://localhost:8000/health/ready
```

### Values Schema Validation Errors

```bash
# Get detailed error messages
helm lint ./helm/vulcanami --values prod-values.yaml --strict --debug

# Validate specific value
helm template vulcanami ./helm/vulcanami --show-only templates/deployment.yaml \
  --set image.tag=v1.0.0 \
  --debug
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Get image digest
        id: digest
        run: |
          DIGEST=$(crane digest ghcr.io/${{ github.repository }}:${{ github.ref_name }})
          echo "digest=${DIGEST}" >> $GITHUB_OUTPUT
      
      - name: Deploy with Helm
        run: |
          helm upgrade vulcanami ./helm/vulcanami \
            --install \
            --values environments/production-values.yaml \
            --set image.tag="${{ github.ref_name }}" \
            --set image.digest="${{ steps.digest.outputs.digest }}" \
            --namespace vulcanami-prod \
            --wait --timeout 15m \
            --atomic
```

---

## Additional Resources

- [Helm Values Schema Documentation](https://helm.sh/docs/topics/charts/#schema-files)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Image Digest Pinning](https://docs.docker.com/engine/reference/commandline/pull/#pull-an-image-by-digest-immutable-identifier)
- [Pod Disruption Budgets](https://kubernetes.io/docs/tasks/run-application/configure-pdb/)
