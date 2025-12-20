# Deployment Guide

## Overview

This guide covers deploying VulcanAMI/Graphix Vulcan to different environments using various deployment methods.

## Pre-Deployment Validation

Before deploying to any environment, run the comprehensive test suite to ensure everything is configured correctly:

```bash
# Quick pre-deployment check
./quick_test.sh quick

# Full validation (recommended)
./test_full_cicd.sh

# Test specific components
./quick_test.sh docker      # Docker configurations
./quick_test.sh k8s         # Kubernetes manifests
./quick_test.sh security    # Security settings

# Run pytest test suite
pytest tests/test_cicd_reproducibility.py -v
```

**Pre-Deployment Checklist:**
- [ ] All tests pass (`./test_full_cicd.sh`)
- [ ] Docker builds successfully
- [ ] All environment variables documented
- [ ] Secrets stored securely (not in code)
- [ ] Version tagged in git
- [ ] Documentation up to date
- [x] Requirements-hashed.txt up-to-date with SHA256 hashes
- [x] Kubernetes manifests validated with kustomize
- [x] All Docker Compose files validated

For detailed testing instructions, see **[TESTING_GUIDE.md](TESTING_GUIDE.md)**.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.24+ (for K8s deployment)
- Helm 3.10+ (for Helm deployment)
- kubectl configured
- Access to container registry (GitHub Container Registry, Docker Hub, etc.)

## Deployment Methods

### 1. Docker Compose (Development/Testing)

#### Quick Start

```bash
# Clone repository
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

# Generate secrets
make generate-secrets > .env

# Edit .env with generated secrets
vim .env

# Start all services
make up

# Check status
make ps

# View logs
make logs-compose
```

#### Production Docker Compose

```bash
# Create production .env
cat > .env << 'EOF'
# Required production secrets
JWT_SECRET_KEY=<your-secure-jwt-secret>
BOOTSTRAP_KEY=<your-bootstrap-key>
POSTGRES_PASSWORD=<secure-password>
REDIS_PASSWORD=<secure-password>
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=<secure-password>
GRAFANA_PASSWORD=<secure-password>

# Configuration
VERSION=latest
REGISTRY=ghcr.io
IMAGE_NAME=musicmonk42/vulcanami_llm
ENVIRONMENT=production
LOG_LEVEL=INFO

# Security: Network binding (set to 0.0.0.0 for container networking)
HOST=0.0.0.0
API_HOST=0.0.0.0

# Optional: HuggingFace Model Pinning (recommended for production)
# VULCAN_TEXT_MODEL_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594
# VULCAN_AUDIO_MODEL_REVISION=<commit-hash>
# VULCAN_BERT_MODEL_REVISION=<commit-hash>
EOF

# Start production stack
docker compose -f docker-compose.prod.yml up -d

# Monitor startup
docker compose -f docker-compose.prod.yml logs -f
```

#### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| API Gateway | 8000 | Main API endpoint |
| DQS Service | 8080 | Data Quality System |
| PII Service | 8082 | PII Detection |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |
| MinIO | 9000 | Object storage |
| MinIO Console | 9001 | MinIO UI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |

### 2. Kubernetes Deployment

#### Image Tagging Strategy

For reproducible deployments, the kustomize overlays use `IMAGE_TAG` as a placeholder that must be replaced with a specific version during deployment. The CI/CD pipeline automatically replaces this placeholder, but for manual deployments, you need to set the image tag:

```bash
# Option 1: Edit kustomization.yaml directly
cd k8s/overlays/<environment>  # development, staging, or production
# Edit kustomization.yaml and replace IMAGE_TAG with your version (e.g., v1.0.0)

# Option 2: Use kustomize edit command for specific environment
cd k8s/overlays/production
kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0

# Or for development
cd k8s/overlays/development
kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:develop-abc1234

# Option 3: Use sed for automated replacement (works for any overlay)
VERSION=v1.0.0
ENVIRONMENT=production  # or development, staging
sed -i "s|newTag: IMAGE_TAG|newTag: $VERSION|g" k8s/overlays/$ENVIRONMENT/kustomization.yaml

# Option 4: Replace in all overlays at once
VERSION=v1.0.0
for env in development staging production; do
  sed -i "s|newTag: IMAGE_TAG|newTag: $VERSION|g" k8s/overlays/$env/kustomization.yaml
done
```

**Best Practice**: Always use specific version tags (e.g., `v1.0.0`, `main-abc1234`) instead of `latest` for production deployments to ensure reproducibility.

#### Using kubectl with Kustomize

##### Development Deployment

```bash
# Create namespace (kustomize will create vulcanami-development)
# Create secrets in development namespace
kubectl create secret generic vulcanami-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
  --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=MINIO_SECRET_KEY=$(openssl rand -base64 24) \
  -n vulcanami-development

# Apply development configuration
kubectl apply -k k8s/overlays/development/

# Check deployment status
kubectl get all -n vulcanami-development

# Check pods
kubectl get pods -n vulcanami-development

# View logs
kubectl logs -f deployment/dev-vulcanami-api -n vulcanami-development
```

##### Production Deployment

```bash
# Create secrets in production namespace
kubectl create secret generic vulcanami-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
  --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=MINIO_SECRET_KEY=$(openssl rand -base64 24) \
  -n vulcanami-production

# Apply production configuration
kubectl apply -k k8s/overlays/production/

# Check deployment status
kubectl get all -n vulcanami-production

# Check pods
kubectl get pods -n vulcanami-production

# View logs
kubectl logs -f deployment/prod-vulcanami-api -n vulcanami-production
```

#### Using Helm

**Note:** Helm deployments can use any namespace you choose. The examples below use `vulcanami` namespace, which is independent of the kustomize overlay namespaces (`vulcanami-development` and `vulcanami-production`). You can choose a different namespace based on your requirements.

```bash
# Add repository (if published)
helm repo add vulcanami https://charts.vulcanami.io
helm repo update

# Or install from local chart
cd helm/vulcanami

# Create values file for environment
cat > values-prod.yaml << 'EOF'
replicaCount: 5

image:
  tag: "v1.0.0"

ingress:
  enabled: true
  hosts:
    - host: api.vulcanami.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

config:
  environment: production
  logLevel: INFO

secrets:
  jwtSecretKey: "your-secret"
  bootstrapKey: "your-secret"
  postgresPassword: "your-secret"
  redisPassword: "your-secret"
  minioSecretKey: "your-secret"
EOF

# Install
helm install vulcanami . \
  --namespace vulcanami \
  --create-namespace \
  --values values-prod.yaml

# Upgrade
helm upgrade vulcanami . \
  --namespace vulcanami \
  --values values-prod.yaml

# Uninstall
helm uninstall vulcanami -n vulcanami
```

### 3. AWS EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster \
  --name vulcanami-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name vulcanami-prod

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/aws/deploy.yaml

# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Deploy application
kubectl apply -k k8s/overlays/production/

# Get LoadBalancer URL
kubectl get svc -n ingress-nginx
```

### 4. Azure AKS Deployment

#### Prerequisites

1. **Azure CLI**: Install from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
2. **Azure Subscription**: Active Azure subscription with appropriate permissions
3. **Azure Service Principal**: For GitHub Actions automation (see below)

#### Manual Deployment

```bash
# Login to Azure
az login

# Create resource group
az group create --name vulcanami-prod --location eastus

# Create Azure Container Registry (ACR)
az acr create \
  --resource-group vulcanami-prod \
  --name vulcanamiregistry \
  --sku Standard

# Create AKS cluster
az aks create \
  --resource-group vulcanami-prod \
  --name vulcanami-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --attach-acr vulcanamiregistry \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group vulcanami-prod --name vulcanami-cluster

# Install ingress
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --create-namespace \
  --namespace ingress-nginx

# Deploy application
kubectl apply -k k8s/overlays/production/
```

#### GitHub Actions Automated Deployment

This repository includes a GitHub Actions workflow (`.github/workflows/azure-kubernetes-service-helm.yml`) for automated CI/CD to Azure AKS.

**Step 1: Create Azure Service Principal**

```bash
# Login to Azure
az login

# Get your subscription ID
az account show --query id --output tsv

# Create Service Principal with Contributor role
az ad sp create-for-rbac \
  --name "github-actions-vulcanami" \
  --role contributor \
  --scopes /subscriptions/{YOUR_SUBSCRIPTION_ID} \
  --sdk-auth

# Output will show:
# {
#   "clientId": "xxxx",
#   "clientSecret": "xxxx",
#   "subscriptionId": "xxxx",
#   "tenantId": "xxxx",
#   ...
# }
```

**Step 2: Configure GitHub Repository Secrets**

Add the following secrets to your GitHub repository:
- Go to: **Repository Settings** → **Secrets and variables** → **Actions** → **Repository secrets**
- Click "New repository secret" and add:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `AZURE_CLIENT_ID` | `{clientId}` from output | Service Principal Application ID |
| `AZURE_TENANT_ID` | `{tenantId}` from output | Azure Active Directory Tenant ID |
| `AZURE_SUBSCRIPTION_ID` | `{subscriptionId}` from output | Azure Subscription ID |

**⚠️ Important:** Without these secrets, the workflow will fail at the "Azure login" step.

**Step 3: Update Workflow Configuration**

Edit `.github/workflows/azure-kubernetes-service-helm.yml` and update environment variables:

```yaml
env:
  AZURE_CONTAINER_REGISTRY: "vulcanamiregistry"  # Your ACR name (without .azurecr.io)
  CONTAINER_NAME: "vulcanami-llm"                 # Your container image name
  RESOURCE_GROUP: "vulcanami-prod"                # Your resource group name
  CLUSTER_NAME: "vulcanami-cluster"               # Your AKS cluster name
  CHART_PATH: "helm/vulcanami"                    # Path to your Helm chart
  CHART_OVERRIDE_PATH: "helm/vulcanami/values-prod.yaml"  # Values override file
```

**Step 4: Trigger Deployment**

The workflow automatically runs on:
- Push to `main` branch
- Manual trigger via GitHub Actions UI

**Monitoring:**
- View workflow runs in the **Actions** tab
- Check deployment status and logs
- Monitor AKS cluster with `kubectl` or Azure Portal

**For detailed troubleshooting**, see the Azure AKS Deployment Workflow section in [CI_CD.md](CI_CD.md).

### 5. Google GKE Deployment

```bash
# Create cluster
gcloud container clusters create vulcanami-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials vulcanami-prod --zone us-central1-a

# Deploy application
kubectl apply -k k8s/overlays/production/
```

## Configuration

### Environment Variables

Required environment variables for all deployments:

```bash
# Security (REQUIRED)
JWT_SECRET_KEY=<base64-encoded-secret-48-chars>
BOOTSTRAP_KEY=<base64-encoded-secret-32-chars>

# Database
POSTGRES_HOST=postgres-service
POSTGRES_PORT=5432
POSTGRES_DB=vulcanami
POSTGRES_USER=vulcanami
POSTGRES_PASSWORD=<secure-password>

# Redis
REDIS_HOST=redis-service
REDIS_PORT=6379
REDIS_PASSWORD=<secure-password>

# MinIO
MINIO_ENDPOINT=http://minio-service:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=<secure-password>

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

### Secrets Management

#### Kubernetes Secrets

```bash
# Using kubectl (development)
kubectl create secret generic vulcanami-secrets \
  --from-env-file=.env \
  -n vulcanami-development

# Using kubectl (production)
kubectl create secret generic vulcanami-secrets \
  --from-env-file=.env \
  -n vulcanami-production

# Using sealed-secrets
kubeseal --format yaml < secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

#### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: vulcanami-secrets
  namespace: vulcanami-production  # or vulcanami-development
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: vulcanami-secrets
  data:
  - secretKey: JWT_SECRET_KEY
    remoteRef:
      key: vulcanami/jwt-secret
  - secretKey: POSTGRES_PASSWORD
    remoteRef:
      key: vulcanami/postgres-password
```

## Monitoring

### Prometheus

Access Prometheus:
- Local: `http://localhost:9090`
- Kubernetes: Port-forward or use Ingress

```bash
# For development
kubectl port-forward -n vulcanami-development svc/prometheus 9090:9090

# For production
kubectl port-forward -n vulcanami-production svc/prometheus 9090:9090
```

### Grafana

Access Grafana:
- Local: `http://localhost:3000`
- Kubernetes: Port-forward or use Ingress

```bash
# For development
kubectl port-forward -n vulcanami-development svc/grafana 3000:3000

# For production
kubectl port-forward -n vulcanami-production svc/grafana 3000:3000
```

Default credentials:
- Username: `admin`
- Password: Set in environment

### Logging

View application logs:

```bash
# Docker Compose
docker compose logs -f api-gateway

# Kubernetes - Development
kubectl logs -f deployment/dev-vulcanami-api -n vulcanami-development

# Kubernetes - Production
kubectl logs -f deployment/prod-vulcanami-api -n vulcanami-production

# Stream logs from all pods (development)
kubectl logs -f -l app=vulcanami-api -n vulcanami-development --all-containers=true

# Stream logs from all pods (production)
kubectl logs -f -l app=vulcanami-api -n vulcanami-production --all-containers=true
```

## Scaling

### Horizontal Pod Autoscaler

HPA is configured automatically with Helm/Kustomize:

```bash
# Check HPA status (development)
kubectl get hpa -n vulcanami-development

# Check HPA status (production)
kubectl get hpa -n vulcanami-production

# Manually scale (development)
kubectl scale deployment dev-vulcanami-api --replicas=3 -n vulcanami-development

# Manually scale (production)
kubectl scale deployment prod-vulcanami-api --replicas=10 -n vulcanami-production
```

### Vertical Scaling

Update resource limits:

```yaml
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

## Backup and Restore

### Database Backup

```bash
# Backup PostgreSQL (development)
kubectl exec -n vulcanami-development postgres-0 -- pg_dump -U vulcanami vulcanami > backup.sql

# Backup PostgreSQL (production)
kubectl exec -n vulcanami-production postgres-0 -- pg_dump -U vulcanami vulcanami > backup.sql

# Restore (development)
kubectl exec -i -n vulcanami-development postgres-0 -- psql -U vulcanami vulcanami < backup.sql

# Restore (production)
kubectl exec -i -n vulcanami-production postgres-0 -- psql -U vulcanami vulcanami < backup.sql
```

### Application State

```bash
# Backup MinIO data
mc mirror minio/vulcanami-hot ./backup/

# Restore
mc mirror ./backup/ minio/vulcanami-hot
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status (replace namespace as needed)
kubectl describe pod <pod-name> -n vulcanami-development  # or vulcanami-production

# Check events
kubectl get events -n vulcanami-development --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n vulcanami-development
```

### Service Not Accessible

```bash
# Check service (replace namespace as needed)
kubectl get svc -n vulcanami-development  # or vulcanami-production

# Check ingress
kubectl get ingress -n vulcanami-development
kubectl describe ingress vulcanami-ingress -n vulcanami-development

# Test from inside cluster (development)
kubectl run -it --rm debug --image=alpine --restart=Never -n vulcanami-development -- sh
apk add curl
curl http://dev-vulcanami-api.vulcanami-development.svc.cluster.local:8000/health

# Test from inside cluster (production)
kubectl run -it --rm debug --image=alpine --restart=Never -n vulcanami-production -- sh
apk add curl
curl http://prod-vulcanami-api.vulcanami-production.svc.cluster.local:8000/health
```

### Database Connection Issues

```bash
# Test database connectivity (development)
kubectl run -it --rm psql --image=postgres:14 --restart=Never -n vulcanami-development -- \
  psql -h postgres-service.vulcanami-development.svc.cluster.local -U vulcanami -d vulcanami

# Test database connectivity (production)
kubectl run -it --rm psql --image=postgres:14 --restart=Never -n vulcanami-production -- \
  psql -h postgres-service.vulcanami-production.svc.cluster.local -U vulcanami -d vulcanami

# Check database logs (replace namespace as needed)
kubectl logs -n vulcanami-development postgres-0  # or vulcanami-production
```

## Security Hardening

1. **Use TLS/SSL**: Configure cert-manager and Let's Encrypt
2. **Network Policies**: Restrict pod-to-pod communication
3. **Pod Security Policies**: Enforce security standards
4. **RBAC**: Use least-privilege access
5. **Secrets Management**: Use external secret stores
6. **Image Scanning**: Scan images for vulnerabilities
7. **Update Regularly**: Keep dependencies and base images updated

## Performance Tuning

1. **Resource Limits**: Set appropriate CPU/memory limits
2. **Connection Pooling**: Configure database connection pools
3. **Caching**: Enable Redis caching
4. **CDN**: Use CDN for static assets
5. **Load Balancing**: Use proper load balancer configuration
6. **Database Optimization**: Create indexes, optimize queries

## Rolling Updates

```bash
# Update image version (development)
kubectl set image deployment/dev-vulcanami-api \
  api=ghcr.io/musicmonk42/vulcanami_llm-api:v1.1.0 \
  -n vulcanami-development

# Update image version (production)
kubectl set image deployment/prod-vulcanami-api \
  api=ghcr.io/musicmonk42/vulcanami_llm-api:v1.1.0 \
  -n vulcanami-production

# Check rollout status (development)
kubectl rollout status deployment/dev-vulcanami-api -n vulcanami-development

# Check rollout status (production)
kubectl rollout status deployment/prod-vulcanami-api -n vulcanami-production

# Rollback if needed (development)
kubectl rollout undo deployment/dev-vulcanami-api -n vulcanami-development

# Rollback if needed (production)
kubectl rollout undo deployment/prod-vulcanami-api -n vulcanami-production
```

## Health Checks

All services expose health endpoints:
- Liveness: `/health`
- Readiness: `/health`
- Metrics: `/metrics`

Test health:
```bash
curl http://api.vulcanami.example.com/health
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/musicmonk42/VulcanAMI_LLM/issues
- Documentation: See README.md
- Email: support@vulcanami.io
