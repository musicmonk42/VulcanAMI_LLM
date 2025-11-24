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

#### Using kubectl with Kustomize

```bash
# Create namespace
kubectl create namespace vulcanami

# Create secrets
kubectl create secret generic vulcanami-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
  --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=MINIO_SECRET_KEY=$(openssl rand -base64 24) \
  -n vulcanami

# Development
kubectl apply -k k8s/overlays/development/

# Production
kubectl apply -k k8s/overlays/production/

# Check deployment status
kubectl get all -n vulcanami

# Check pods
kubectl get pods -n vulcanami

# View logs
kubectl logs -f deployment/vulcanami-api -n vulcanami
```

#### Using Helm

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

```bash
# Create resource group
az group create --name vulcanami-prod --location eastus

# Create AKS cluster
az aks create \
  --resource-group vulcanami-prod \
  --name vulcanami-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
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
# Using kubectl
kubectl create secret generic vulcanami-secrets \
  --from-env-file=.env \
  -n vulcanami

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
  namespace: vulcanami
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
kubectl port-forward -n vulcanami svc/prometheus 9090:9090
```

### Grafana

Access Grafana:
- Local: `http://localhost:3000`
- Kubernetes: Port-forward or use Ingress

```bash
kubectl port-forward -n vulcanami svc/grafana 3000:3000
```

Default credentials:
- Username: `admin`
- Password: Set in environment

### Logging

View application logs:

```bash
# Docker Compose
docker-compose logs -f api-gateway

# Kubernetes
kubectl logs -f deployment/vulcanami-api -n vulcanami

# Stream logs from all pods
kubectl logs -f -l app=vulcanami-api -n vulcanami --all-containers=true
```

## Scaling

### Horizontal Pod Autoscaler

HPA is configured automatically with Helm/Kustomize:

```bash
# Check HPA status
kubectl get hpa -n vulcanami

# Manually scale
kubectl scale deployment vulcanami-api --replicas=10 -n vulcanami
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
# Backup PostgreSQL
kubectl exec -n vulcanami postgres-0 -- pg_dump -U vulcanami vulcanami > backup.sql

# Restore
kubectl exec -i -n vulcanami postgres-0 -- psql -U vulcanami vulcanami < backup.sql
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
# Check pod status
kubectl describe pod <pod-name> -n vulcanami

# Check events
kubectl get events -n vulcanami --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n vulcanami
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n vulcanami

# Check ingress
kubectl get ingress -n vulcanami
kubectl describe ingress vulcanami-ingress -n vulcanami

# Test from inside cluster
kubectl run -it --rm debug --image=alpine --restart=Never -- sh
apk add curl
curl http://vulcanami-api.vulcanami.svc.cluster.local:8000/health
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm psql --image=postgres:14 --restart=Never -- \
  psql -h postgres-service.vulcanami.svc.cluster.local -U vulcanami -d vulcanami

# Check database logs
kubectl logs -n vulcanami postgres-0
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
# Update image version
kubectl set image deployment/vulcanami-api \
  api=ghcr.io/musicmonk42/vulcanami_llm-api:v1.1.0 \
  -n vulcanami

# Check rollout status
kubectl rollout status deployment/vulcanami-api -n vulcanami

# Rollback if needed
kubectl rollout undo deployment/vulcanami-api -n vulcanami
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
