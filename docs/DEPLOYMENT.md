# Deployment Guide

## Overview

This guide covers deploying VulcanAMI/Graphix Vulcan to different environments using various deployment methods.

**✨ New**: See [METAPROGRAMMING_DEPLOYMENT.md](METAPROGRAMMING_DEPLOYMENT.md) for metaprogramming features deployment integration (autonomous graph evolution, 8 new node handlers).

## Pre-Deployment Validation

Before deploying to any environment, run the comprehensive test suite to ensure everything is configured correctly:

```bash
# Quick pre-deployment check
./quick_test.sh quick

# Full validation (recommended)
./test_full_cicd.sh

# Test specific components
./quick_test.sh docker # Docker configurations
./quick_test.sh k8s # Kubernetes manifests
./quick_test.sh security # Security settings

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

#### JWT Secret Configuration

**IMPORTANT**: The application requires a JWT secret for authentication. The entrypoint script validates the secret on startup:

**Limited Mode (Health Checks Only)**:
- If NO valid JWT secret is provided, the application starts in LIMITED MODE
- Health endpoints (`/health/live`, `/health/ready`, `/health`) work normally
- Authentication features are DISABLED
- Protected endpoints return 401 Unauthorized
- **Use case**: Initial deployment, health check validation, troubleshooting

**Full Mode (Production)**:
- Requires ONE of these environment variables:
 - `JWT_SECRET`
 - `JWT_SECRET_KEY` 
 - `GRAPHIX_JWT_SECRET`
- Secret requirements:
 - Minimum 32 characters
 - No weak patterns (password, 123456, etc.)
 - URL-safe characters recommended
- All features enabled including authentication

**Generate a secure JWT secret**:
```bash
# Generate URL-safe secret (recommended)
openssl rand -base64 48 | tr -d '+/'

# Or use Python
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

**Example .env configuration**:
```bash
# For production - REQUIRED
JWT_SECRET=$(openssl rand -base64 48 | tr -d '+/')

# Alternative variable names (any one of these works)
# JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '+/')
# GRAPHIX_JWT_SECRET=$(openssl rand -base64 48 | tr -d '+/')
```

**Startup validation messages**:
```
# With valid JWT (FULL MODE):
✅ Verified JWT secret in variable: JWT_SECRET (rotate secrets periodically)

# Without valid JWT (LIMITED MODE):
⚠️ WARNING: No valid JWT secret provided.
⚠️ Application will start in LIMITED MODE without JWT authentication
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
cd k8s/overlays/<environment> # development, staging, or production
# Edit kustomization.yaml and replace IMAGE_TAG with your version (e.g., v1.0.0)

# Option 2: Use kustomize edit command for specific environment
cd k8s/overlays/production
kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0

# Or for development
cd k8s/overlays/development
kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:develop-abc1234

# Option 3: Use sed for automated replacement (works for any overlay)
VERSION=v1.0.0
ENVIRONMENT=production # or development, staging
sed -i "s|newTag: IMAGE_TAG|newTag: $VERSION|g" k8s/overlays/$ENVIRONMENT/kustomization.yaml

# Option 4: Replace in all overlays at once
VERSION=v1.0.0
for env in development staging production; do
 sed -i "s|newTag: IMAGE_TAG|newTag: $VERSION|g" k8s/overlays/$env/kustomization.yaml
done
```

**Best Practice**: Always use specific version tags (e.g., `v1.0.0`, `main-abc1234`) instead of `latest` for production deployments to ensure reproducibility.

#### Quick Start with Automated Scripts

For the fastest and most reliable deployment, use the provided automation scripts:

```bash
# Step 1: Validate your cluster is ready
./scripts/validate-deployment.sh development

# Step 2: Deploy to development
./scripts/deploy.sh development --image-tag v1.0.0

# Or deploy to production
./scripts/deploy.sh production --image-tag v1.0.0
```

The deployment script provides:
- ✅ Comprehensive pre-deployment validation
- ✅ Automatic secret checking
- ✅ Rollout status monitoring
- ✅ Health checks after deployment
- ✅ Automatic rollback on failure
- ✅ Dry-run mode for testing

**Script Options:**
```bash
# Dry-run mode (test without applying changes)
./scripts/deploy.sh development --dry-run

# Skip validation checks (not recommended)
./scripts/deploy.sh production --skip-validation --image-tag v1.0.0

# Custom namespace
./scripts/deploy.sh staging --namespace my-custom-namespace --image-tag v1.0.0

# Longer rollout timeout for slow clusters
./scripts/deploy.sh production --wait-timeout 900 --image-tag v1.0.0
```

#### Prerequisites for Kubernetes Deployment

Before deploying, ensure you have:

**Required:**
- ✅ Kubernetes cluster 1.24+ (EKS, AKS, GKE, or on-premises)
- ✅ kubectl 1.24+ installed and configured
- ✅ kustomize 4.5+ installed
- ✅ At least 235Gi available storage (20Gi postgres + 5Gi redis + 50Gi milvus + 100Gi minio + 60Gi app)
- ✅ OpenSSL for secret generation

**Recommended:**
- ⚠️ NGINX Ingress Controller (for external access)
- ⚠️ cert-manager (for TLS certificates)
- ⚠️ Prometheus + Grafana (for monitoring)
- ⚠️ Storage class with ReadWriteOnce support

**Check your cluster:**
```bash
# Verify cluster access
kubectl cluster-info

# Check kubectl version
kubectl version --client

# Verify storage classes
kubectl get storageclasses

# Check available nodes
kubectl get nodes

# Run comprehensive validation
./scripts/validate-deployment.sh development
```

#### Secret Management

Secrets must be created before deployment. You have three options:

**Option 1: Use the provided script (Recommended)**
```bash
# Generate secrets and create in cluster
./scripts/generate-secrets.sh | kubectl apply -f - -n vulcanami-development

# For production with custom API keys
./scripts/generate-secrets.sh | \
 sed 's/OPENAI_API_KEY: ""/OPENAI_API_KEY: "sk-your-key"/' | \
 kubectl apply -f - -n vulcanami-production
```

**Option 2: Manual creation with kubectl**
```bash
kubectl create secret generic vulcanami-secrets \
 --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
 --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
 --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
 --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
 --from-literal=MINIO_ROOT_USER=minioadmin \
 --from-literal=MINIO_ROOT_PASSWORD=$(openssl rand -base64 32) \
 --from-literal=MINIO_SECRET_KEY=$(openssl rand -base64 24) \
 --from-literal=OPENAI_API_KEY=sk-your-openai-api-key \
 --from-literal=VULCAN_LLM_API_KEY=sk-your-openai-api-key \
 --from-literal=AWS_ACCESS_KEY_ID=minioadmin \
 --from-literal=AWS_SECRET_ACCESS_KEY=$(openssl rand -base64 32) \
 -n vulcanami-development
```

**Option 3: External Secret Manager (Production)**
```bash
# Using AWS Secrets Manager with External Secrets Operator
# Install External Secrets Operator first
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/crds/bundle.yaml

# Then create ExternalSecret resource
# See: https://external-secrets.io/latest/
```

**⚠️ Security Best Practices:**
- Never commit secrets to git
- Use different secrets for each environment
- Rotate secrets regularly (every 90 days recommended)
- Use external secret managers (AWS Secrets Manager, HashiCorp Vault) for production
- Limit RBAC access to secrets

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
# "clientId": "xxxx",
# "clientSecret": "xxxx",
# "subscriptionId": "xxxx",
# "tenantId": "xxxx",
# ...
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
 AZURE_CONTAINER_REGISTRY: "vulcanamiregistry" # Your ACR name (without .azurecr.io)
 CONTAINER_NAME: "vulcanami-llm" # Your container image name
 RESOURCE_GROUP: "vulcanami-prod" # Your resource group name
 CLUSTER_NAME: "vulcanami-cluster" # Your AKS cluster name
 CHART_PATH: "helm/vulcanami" # Path to your Helm chart
 CHART_OVERRIDE_PATH: "helm/vulcanami/values-prod.yaml" # Values override file
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
 namespace: vulcanami-production # or vulcanami-development
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
kubectl describe pod <pod-name> -n vulcanami-development # or vulcanami-production

# Check events
kubectl get events -n vulcanami-development --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n vulcanami-development
```

### Service Not Accessible

```bash
# Check service (replace namespace as needed)
kubectl get svc -n vulcanami-development # or vulcanami-production

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
kubectl logs -n vulcanami-development postgres-0 # or vulcanami-production
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
- Liveness: `/health/live` - Basic health check (service is running)
- Readiness: `/health/ready` - Service is ready to handle traffic
- Metrics: `/metrics` - Prometheus metrics endpoint

Test health endpoints:
```bash
# API health check
kubectl port-forward -n vulcanami-development svc/vulcanami-api 8000:8000
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Or use kubectl exec
kubectl exec -n vulcanami-development deploy/vulcanami-api -- curl -s http://localhost:8000/health/live
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Pods stuck in Pending state

**Symptoms:**
```bash
$ kubectl get pods -n vulcanami-development
NAME READY STATUS RESTARTS AGE
vulcanami-api-xxx 0/1 Pending 0 5m
```

**Diagnosis:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n vulcanami-development

# Common causes:
# 1. Insufficient resources
# 2. Storage class not available
# 3. Node selector/affinity mismatch
```

**Solutions:**
```bash
# Check node resources
kubectl top nodes

# Check PVC status
kubectl get pvc -n vulcanami-development

# Check storage class
kubectl get storageclasses
```

#### Issue: Milvus bootstrap fails

**Symptoms:**
```bash
# Init container fails
kubectl logs <pod-name> -n vulcanami-development -c milvus-bootstrap
Error: Cannot connect to milvus-service:19530
```

**Solutions:**
```bash
# 1. Check Milvus pod is running
kubectl get pods -n vulcanami-development -l app=milvus

# 2. Check Milvus service
kubectl get svc -n vulcanami-development milvus-service

# 3. Check Milvus logs
kubectl logs -n vulcanami-development -l app=milvus --tail=100

# 4. Test connectivity from API pod
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 nc -zv milvus-service 19530
```

#### Issue: MinIO bucket not created

**Symptoms:**
```bash
# API logs show S3 errors
Error: Bucket 'vulcanami-memory' does not exist
```

**Solutions:**
```bash
# 1. Check MinIO bucket setup job
kubectl get jobs -n vulcanami-development -l app=minio

# 2. Check job logs
kubectl logs -n vulcanami-development job/minio-bucket-setup

# 3. Manually create bucket if job failed
kubectl exec -n vulcanami-development -it sts/minio -- \
 mc alias set myminio http://localhost:9000 minioadmin <password>
kubectl exec -n vulcanami-development -it sts/minio -- \
 mc mb myminio/vulcanami-memory

# 4. Restart the bucket setup job
kubectl delete job minio-bucket-setup -n vulcanami-development
kubectl apply -k k8s/overlays/development/
```

#### Issue: Image pull errors

**Symptoms:**
```bash
# Pod shows ImagePullBackOff
Status: ImagePullBackOff
```

**Solutions:**
```bash
# 1. Check if image exists
docker pull ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0

# 2. Check image pull secrets (if registry is private)
kubectl get secrets -n vulcanami-development

# 3. Create image pull secret if needed
kubectl create secret docker-registry ghcr-secret \
 --docker-server=ghcr.io \
 --docker-username=<your-username> \
 --docker-password=<your-token> \
 -n vulcanami-development

# 4. Patch service account to use secret
kubectl patch serviceaccount default \
 -n vulcanami-development \
 -p '{"imagePullSecrets": [{"name": "ghcr-secret"}]}'
```

#### Issue: Secrets contain placeholder values

**Symptoms:**
```bash
# Validation warns about placeholder values
Secret key 'POSTGRES_PASSWORD' contains placeholder value
```

**Solutions:**
```bash
# Delete and recreate secrets
kubectl delete secret vulcanami-secrets -n vulcanami-development

# Regenerate with proper values
./scripts/generate-secrets.sh | kubectl apply -f - -n vulcanami-development

# Or update specific secret keys
kubectl patch secret vulcanami-secrets -n vulcanami-development \
 -p="{\"data\":{\"POSTGRES_PASSWORD\":\"$(echo -n 'your-password' | base64)\"}}"
```

#### Issue: Network policy blocking connections

**Symptoms:**
```bash
# Timeout errors between services
Error: timeout connecting to postgres-service:5432
```

**Solutions:**
```bash
# 1. Check network policies
kubectl get networkpolicies -n vulcanami-development

# 2. Describe policy to see rules
kubectl describe networkpolicy -n vulcanami-development

# 3. Temporarily disable network policy for testing
kubectl delete networkpolicy <policy-name> -n vulcanami-development

# 4. Re-enable after fixing
kubectl apply -k k8s/overlays/development/
```

#### Issue: Persistent volume claims not binding

**Symptoms:**
```bash
$ kubectl get pvc -n vulcanami-development
NAME STATUS VOLUME CAPACITY
milvus-storage-milvus-0 Pending 
```

**Solutions:**
```bash
# 1. Check if storage class exists
kubectl get storageclasses

# 2. Check PVC events
kubectl describe pvc milvus-storage-milvus-0 -n vulcanami-development

# 3. If storage class is wrong, update manifests
# Edit k8s/base/*-deployment.yaml and change storageClassName

# 4. Check if cluster has available storage
kubectl get pv
```

### Debugging Commands

```bash
# Get all resources in namespace
kubectl get all -n vulcanami-development

# Check pod logs
kubectl logs -n vulcanami-development <pod-name>
kubectl logs -n vulcanami-development <pod-name> -c <container-name>
kubectl logs -n vulcanami-development <pod-name> --previous # Previous container

# Follow logs in real-time
kubectl logs -f -n vulcanami-development -l app=vulcanami-api

# Execute commands in pod
kubectl exec -it -n vulcanami-development <pod-name> -- /bin/bash

# Check pod resource usage
kubectl top pods -n vulcanami-development

# Check node resource usage
kubectl top nodes

# Get pod YAML
kubectl get pod <pod-name> -n vulcanami-development -o yaml

# Check events
kubectl get events -n vulcanami-development --sort-by='.lastTimestamp'

# Port forward for local testing
kubectl port-forward -n vulcanami-development svc/vulcanami-api 8000:8000
kubectl port-forward -n vulcanami-development svc/minio-service 9000:9000 9001:9001

# Scale deployment
kubectl scale deployment vulcanami-api -n vulcanami-development --replicas=0
kubectl scale deployment vulcanami-api -n vulcanami-development --replicas=3
```

## Monitoring and Observability

### Prometheus Metrics

All services expose Prometheus metrics:

```bash
# Check metrics endpoint
kubectl port-forward -n vulcanami-development svc/vulcanami-api 8000:8000
curl http://localhost:8000/metrics

# Milvus metrics
kubectl port-forward -n vulcanami-development svc/milvus-service 9091:9091
curl http://localhost:9091/metrics
```

### Install Prometheus Stack (Recommended)

```bash
# Add Prometheus helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
 --namespace monitoring \
 --create-namespace \
 --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Access Grafana dashboard
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials: admin / prom-operator
```

### Key Metrics to Monitor

**Application Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `http_requests_in_progress` - Active requests
- `vulcan_llm_inference_duration_seconds` - LLM inference time
- `vulcan_memory_operations_total` - Memory system operations

**System Metrics:**
- `container_cpu_usage_seconds_total` - CPU usage
- `container_memory_working_set_bytes` - Memory usage
- `kube_pod_status_phase` - Pod status
- `kube_deployment_status_replicas_available` - Available replicas

**Database Metrics:**
- `pg_stat_database_*` - PostgreSQL statistics
- `redis_connected_clients` - Redis connections
- `milvus_*` - Milvus vector operations

### Logging

```bash
# View API logs
kubectl logs -n vulcanami-development -l app=vulcanami-api --tail=100 -f

# View all pod logs
kubectl logs -n vulcanami-development --all-containers=true --tail=100

# Filter logs by severity
kubectl logs -n vulcanami-development -l app=vulcanami-api | grep ERROR

# Export logs to file
kubectl logs -n vulcanami-development -l app=vulcanami-api --tail=1000 > api.log
```

### Centralized Logging (Optional)

Install EFK Stack (Elasticsearch, Fluentd, Kibana):

```bash
# Add elastic helm repo
helm repo add elastic https://helm.elastic.co
helm repo update

# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
 --namespace logging \
 --create-namespace

# Install Fluentd
kubectl apply -f https://raw.githubusercontent.com/fluent/fluentd-kubernetes-daemonset/master/fluentd-daemonset-elasticsearch.yaml

# Install Kibana
helm install kibana elastic/kibana \
 --namespace logging \
 --set elasticsearchHosts=http://elasticsearch-master:9200
```

## Rollback Procedures

### Rollback a Failed Deployment

```bash
# Check rollout history
kubectl rollout history deployment/vulcanami-api -n vulcanami-development

# Rollback to previous version
kubectl rollout undo deployment/vulcanami-api -n vulcanami-development

# Rollback to specific revision
kubectl rollout undo deployment/vulcanami-api -n vulcanami-development --to-revision=3

# Check rollout status
kubectl rollout status deployment/vulcanami-api -n vulcanami-development
```

### Complete Environment Rollback

```bash
# 1. Tag current state
kubectl get all -n vulcanami-development -o yaml > backup-$(date +%Y%m%d-%H%M%S).yaml

# 2. Delete current deployment
kubectl delete -k k8s/overlays/development/

# 3. Restore previous version
git checkout <previous-commit>
kubectl apply -k k8s/overlays/development/

# 4. Verify rollback
./scripts/validate-deployment.sh development
```

### Emergency Rollback

If deployment is completely broken:

```bash
# Scale down all deployments immediately
kubectl scale deployment --all --replicas=0 -n vulcanami-development

# Delete problematic resources
kubectl delete deployment vulcanami-api -n vulcanami-development

# Restore from backup
kubectl apply -f backup-<timestamp>.yaml

# Or redeploy from last known good version
git checkout <last-good-commit>
./scripts/deploy.sh development --image-tag <last-good-version>
```

## Upgrade Procedures

### Minor Version Upgrade

```bash
# 1. Backup current state
kubectl get all -n vulcanami-production -o yaml > backup-pre-upgrade.yaml

# 2. Update image tag
cd k8s/overlays/production
kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:v1.1.0

# 3. Run validation
./scripts/validate-deployment.sh production

# 4. Apply upgrade
kubectl apply -k k8s/overlays/production/

# 5. Monitor rollout
kubectl rollout status deployment/prod-vulcanami-api -n vulcanami-production

# 6. Verify health
kubectl exec -n vulcanami-production deploy/prod-vulcanami-api -- \
 curl -s http://localhost:8000/health/ready
```

### Major Version Upgrade

For major version upgrades (e.g., v1.x to v2.x):

```bash
# 1. Review CHANGELOG.md for breaking changes

# 2. Test upgrade in development first
./scripts/deploy.sh development --image-tag v2.0.0

# 3. Run integration tests
kubectl exec -n vulcanami-development deploy/vulcanami-api -- python -m pytest /app/tests/

# 4. Backup production data
kubectl exec -n vulcanami-production sts/postgres-0 -- \
 pg_dump -U vulcanami vulcanami > backup-$(date +%Y%m%d).sql

# 5. Schedule maintenance window

# 6. Deploy to production with canary strategy (if using service mesh)
# Or use blue-green deployment

# 7. Monitor logs and metrics closely
kubectl logs -f -n vulcanami-production -l app=vulcanami-api
```

### Zero-Downtime Upgrade Strategy

```bash
# 1. Ensure replicas > 1
kubectl scale deployment vulcanami-api -n vulcanami-production --replicas=5

# 2. Update image with rolling update strategy
kubectl set image deployment/vulcanami-api \
 api=ghcr.io/musicmonk42/vulcanami_llm-api:v1.1.0 \
 -n vulcanami-production

# 3. Monitor rollout
kubectl rollout status deployment/vulcanami-api -n vulcanami-production

# 4. Verify no downtime
# Monitor http_requests_total metric for continuous traffic
```

## Performance Tuning

### Resource Optimization

```bash
# Monitor resource usage
kubectl top pods -n vulcanami-production

# Adjust resource requests/limits based on actual usage
kubectl set resources deployment vulcanami-api \
 -n vulcanami-production \
 --limits=cpu=4000m,memory=8Gi \
 --requests=cpu=1000m,memory=2Gi
```

### Horizontal Pod Autoscaling

```bash
# Create HPA based on CPU
kubectl autoscale deployment vulcanami-api \
 -n vulcanami-production \
 --cpu-percent=70 \
 --min=3 \
 --max=20

# Or create HPA based on custom metrics
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
 name: vulcanami-api-hpa
 namespace: vulcanami-production
spec:
 scaleTargetRef:
 apiVersion: apps/v1
 kind: Deployment
 name: vulcanami-api
 minReplicas: 3
 maxReplicas: 20
 metrics:
 - type: Resource
 resource:
 name: cpu
 target:
 type: Utilization
 averageUtilization: 70
 - type: Resource
 resource:
 name: memory
 target:
 type: Utilization
 averageUtilization: 80
EOF

# Check HPA status
kubectl get hpa -n vulcanami-production
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/musicmonk42/VulcanAMI_LLM/issues
- Documentation: See README.md
- Email: support@vulcanami.io
