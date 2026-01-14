# New Engineer Setup Guide

Welcome to VulcanAMI! This guide will help you get started with Docker, Kubernetes, and Helm deployments.

## Quick Start (5 minutes)

### Prerequisites Check

Run this command to verify you have everything installed:

```bash
# Check all prerequisites
./scripts/check-prerequisites.sh
```

**Required tools:**
- Docker 20.10+ 
- Docker Compose v2+
- kubectl 1.24+ (for Kubernetes deployment)
- Helm 3.10+ (for Helm deployment)
- Git

**Optional (for Azure AKS deployment):**
- Azure CLI 2.0+ (https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

### Installation Links (if needed)

- **Docker**: https://docs.docker.com/get-docker/
- **kubectl**: https://kubernetes.io/docs/tasks/tools/
- **Helm**: https://helm.sh/docs/intro/install/

## Step-by-Step Setup

### 1. Clone and Validate

```bash
# Clone the repository
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

# Validate all configurations
./scripts/validate-all.sh
```

The validation script checks:
- ✓ Docker configurations
- ✓ Docker Compose files
- ✓ Kubernetes manifests
- ✓ Helm charts
- ✓ Required files

### 2. Choose Your Deployment Method

Pick the method that matches your needs:

#### A. Docker Compose (Recommended for Development)

**Best for:** Local development, testing, quick demos

```bash
# 1. Create environment file
cp .env.example .env

# 2. Generate secure secrets
./scripts/generate-secrets.sh >> .env

# 3. Edit .env and verify the secrets look good
cat .env

# 4. Start all services
docker compose -f docker-compose.dev.yml up -d

# 5. Check status
docker compose -f docker-compose.dev.yml ps

# 6. Access services
# - API Gateway: http://localhost:8000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

**Stop services:**
```bash
docker compose -f docker-compose.dev.yml down
```

#### B. Kubernetes with Kustomize

**Best for:** Kubernetes clusters, staging/production

```bash
# 1. Create namespace
kubectl create namespace vulcanami

# 2. Create secrets
kubectl create secret generic vulcanami-secrets \
 --from-literal=jwt-secret-key=$(openssl rand -base64 48) \
 --from-literal=bootstrap-key=$(openssl rand -base64 32) \
 --from-literal=postgres-password=$(openssl rand -base64 32) \
 --from-literal=redis-password=$(openssl rand -base64 32) \
 -n vulcanami

# 3. Deploy with kustomize
kubectl apply -k k8s/overlays/development

# 4. Check status
kubectl get pods -n vulcanami

# 5. Port forward to access services
kubectl port-forward -n vulcanami svc/api-gateway 8000:8000
```

**Cleanup:**
```bash
kubectl delete namespace vulcanami
```

#### C. Helm Chart

**Best for:** Production Kubernetes, GitOps workflows

```bash
# 1. Create namespace
kubectl create namespace vulcanami

# 2. Install chart with secrets
helm install vulcanami ./helm/vulcanami \
 --namespace vulcanami \
 --set image.tag=v1.0.0 \
 --set secrets.jwtSecretKey=$(openssl rand -base64 48 | tr -d '\n') \
 --set secrets.bootstrapKey=$(openssl rand -base64 32 | tr -d '\n') \
 --set secrets.postgresPassword=$(openssl rand -base64 32 | tr -d '\n') \
 --set secrets.redisPassword=$(openssl rand -base64 32 | tr -d '\n')

# 3. Check deployment
helm status vulcanami -n vulcanami
kubectl get pods -n vulcanami

# 4. Access the application
kubectl port-forward -n vulcanami svc/vulcanami 8000:8000
```

**Upgrade:**
```bash
helm upgrade vulcanami ./helm/vulcanami -n vulcanami
```

**Uninstall:**
```bash
helm uninstall vulcanami -n vulcanami
```

#### D. Azure Kubernetes Service (AKS) with GitHub Actions

**Best for:** Production deployment with automated CI/CD on Azure

This repository includes a GitHub Actions workflow for automated deployment to Azure AKS.

**Prerequisites:**
1. Azure subscription with AKS cluster and ACR created
2. Azure Service Principal for GitHub Actions
3. GitHub repository secrets configured

**Setup Steps:**

```bash
# 1. Create Azure Service Principal
az login
az ad sp create-for-rbac \
 --name "github-actions-vulcanami" \
 --role contributor \
 --scopes /subscriptions/{YOUR_SUBSCRIPTION_ID} \
 --sdk-auth

# 2. Copy the output (clientId, tenantId, subscriptionId)
# 3. Add these secrets to your GitHub repository:
# Settings > Secrets and variables > Actions > Repository secrets
# - AZURE_CLIENT_ID (from clientId)
# - AZURE_TENANT_ID (from tenantId)
# - AZURE_SUBSCRIPTION_ID (from subscriptionId)

# 4. Update workflow configuration
# Edit .github/workflows/azure-kubernetes-service-helm.yml
# Update these environment variables:
# - AZURE_CONTAINER_REGISTRY
# - CONTAINER_NAME
# - RESOURCE_GROUP
# - CLUSTER_NAME
# - CHART_PATH
# - CHART_OVERRIDE_PATH
```

**Trigger Deployment:**
- Automatically on push to `main` branch
- Manually via GitHub Actions UI

**Monitor Deployment:**
```bash
# View workflow runs in GitHub Actions tab
# Or check deployment locally:
az aks get-credentials --resource-group vulcanami-prod --name vulcanami-cluster
kubectl get pods -n vulcanami
```

**⚠️ Important:** Without the required Azure secrets, the workflow will fail with a clear error message. See [DEPLOYMENT.md](DEPLOYMENT.md#4-azure-aks-deployment) for detailed Azure setup.

## Validation & Testing

### Quick Validation (30 seconds)

```bash
# Run quick checks
./scripts/validate-all.sh
```

### Build Docker Images

```bash
# Build main Dockerfile
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:dev .

# Build all service images
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-api:dev -f docker/api/Dockerfile .
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-dqs:dev -f docker/dqs/Dockerfile .
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-pii:dev -f docker/pii/Dockerfile .
```

### Test Services

```bash
# After starting services, test health endpoints

# Docker Compose
curl http://localhost:8000/health

# Kubernetes/Helm (with port-forward)
curl http://localhost:8000/health
```

## Common Issues & Solutions

### Issue: "REJECT_INSECURE_JWT" build error

**Problem:** Docker build fails with JWT acknowledgment error

**Solution:** All Dockerfiles require explicit acknowledgment that you're not embedding secrets:

```bash
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:dev .
```

### Issue: "required variable X is missing"

**Problem:** Docker Compose fails with missing environment variables

**Solution:** For `docker-compose.prod.yml`, all secrets are required:

```bash
# Create .env file with all required values
cat > .env << 'EOF'
JWT_SECRET_KEY=$(openssl rand -base64 48)
BOOTSTRAP_KEY=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)
EOF

# For development, use docker-compose.dev.yml which has defaults
docker compose -f docker-compose.dev.yml up -d
```

### Issue: Helm validation errors

**Problem:** Helm lint shows "secrets.X is required" errors

**Solution:** These are informational warnings. Provide secrets during install:

```bash
helm install vulcanami ./helm/vulcanami \
 --set secrets.jwtSecretKey=YOUR_SECRET \
 --set secrets.bootstrapKey=YOUR_SECRET \
 --set secrets.postgresPassword=YOUR_SECRET \
 --set secrets.redisPassword=YOUR_SECRET
```

### Issue: Port already in use

**Problem:** Services fail to start because ports are occupied

**Solution:** Check what's using the ports:

```bash
# Check port usage
sudo lsof -i :8000
sudo lsof -i :5432
sudo lsof -i :6379

# Kill processes or stop conflicting services
# Or change ports in docker-compose.yml
```

### Issue: FAISS AVX512 or LLVM warnings on startup

**Problem:** Log shows messages about FAISS AVX512 fallback or LLVM execution engine

**Solution:** These are informational messages, not errors:

- **FAISS AVX512 warning**: System automatically uses best available instruction set (typically AVX2). This is expected and optimal for your CPU.
- **LLVM execution engine**: If creation fails, IR generation and compilation still work. Only JIT execution is unavailable, which doesn't affect most functionality.

No action needed. For production optimization:
```bash
# Optional: Install FAISS for better vector search performance
pip install faiss-cpu

# Check CPU capabilities
cat /proc/cpuinfo | grep -E 'avx|avx2|avx512'
```

See `docs/CONFIGURATION.md` section 9 for details.

### Issue: Kubernetes connection refused

**Problem:** kubectl commands fail with "connection refused"

**Solution:** 

```bash
# Check if kubectl is configured
kubectl cluster-info

# If not configured, you need a Kubernetes cluster
# Options: minikube, kind, Docker Desktop, or cloud provider

# Quick local cluster with minikube
minikube start
```

## Architecture Overview

### Services

- **API Gateway** (`:8000`) - Main application API
- **DQS Service** (`:8080`) - Data Quality System
- **PII Service** (`:8082`) - PII Detection
- **PostgreSQL** (`:5432`) - Database
- **Redis** (`:6379`) - Cache
- **MinIO** (`:9000`, `:9001`) - Object Storage
- **Prometheus** (`:9090`) - Metrics
- **Grafana** (`:3000`) - Dashboards

### Networks

**Docker Compose:**
- `vulcanami-network` - Main application network
- `monitoring-network` - Monitoring services
- `storage-network` - Storage services (internal)

**Kubernetes:**
- Default namespace networking with NetworkPolicies

## Security Notes

### DO NOT commit secrets!

- Never commit `.env` files
- Never commit secrets in code
- Use environment variables or secret management systems

### Generate strong secrets:

```bash
# JWT Secret (minimum 32 characters)
openssl rand -base64 48 | tr -d '+/'

# Other secrets
openssl rand -base64 32 | tr -d '+/'
```

### Network Binding

- **Development (bare metal):** Use `127.0.0.1` (localhost only)
- **Docker/Containers:** Use `0.0.0.0` (required for container networking)
- **Production:** Use reverse proxy (nginx) with `0.0.0.0` binding in containers

## Next Steps

1. **Read the documentation:**
 - [DEPLOYMENT.md](DEPLOYMENT.md) - Detailed deployment guide
 - [DOCKER_BUILD_GUIDE.md](DOCKER_BUILD_GUIDE.md) - Docker build details
 - [INFRASTRUCTURE_SECURITY_GUIDE.md](INFRASTRUCTURE_SECURITY_GUIDE.md) - Security best practices

2. **Explore the codebase:**
 - `src/` - Application source code
 - `docker/` - Service-specific Dockerfiles
 - `k8s/` - Kubernetes manifests
 - `helm/` - Helm charts
 - `configs/` - Configuration files

3. **Run tests:**
 ```bash
 # Quick test
 ./quick_test.sh
 
 # Full test suite
 ./test_full_cicd.sh
 ```

4. **Set up your IDE:**
 - Install Python 3.10+
 - Install dependencies: `pip install -r requirements.txt`
 - Configure linting: `pylint`, `black`, `mypy`

## Getting Help

- **Documentation:** Check the `*.md` files in the repository root
- **Scripts:** Look in `scripts/` directory for automation tools
- **Issues:** Check existing GitHub Issues
- **Validation:** Run `./scripts/validate-all.sh` to diagnose problems

## Useful Commands

```bash
# Docker Compose
docker compose -f docker-compose.dev.yml up -d # Start services
docker compose -f docker-compose.dev.yml down # Stop services
docker compose -f docker-compose.dev.yml ps # Check status
docker compose -f docker-compose.dev.yml logs -f # View logs

# Kubernetes
kubectl get pods -n vulcanami # List pods
kubectl describe pod POD_NAME -n vulcanami # Pod details
kubectl logs POD_NAME -n vulcanami # Pod logs
kubectl port-forward svc/SERVICE 8000:8000 # Port forward

# Helm
helm list -n vulcanami # List releases
helm status RELEASE -n vulcanami # Release status
helm upgrade RELEASE ./helm/vulcanami # Upgrade
helm rollback RELEASE -n vulcanami # Rollback

# Docker
docker ps # List containers
docker images # List images
docker logs CONTAINER_NAME # Container logs
docker exec -it CONTAINER_NAME bash # Shell access
```

## Summary

**To get started quickly:**

```bash
# 1. Clone repo
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

# 2. Validate everything works
./scripts/validate-all.sh

# 3. Start services (pick one)
docker compose -f docker-compose.dev.yml up -d # Docker Compose
# OR
kubectl apply -k k8s/overlays/development # Kubernetes
# OR 
helm install vulcanami ./helm/vulcanami --set... # Helm

# 4. Access the application
curl http://localhost:8000/health
```

Welcome aboard! 🚀
