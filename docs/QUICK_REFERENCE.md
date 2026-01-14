# Quick Reference: Docker, Kubernetes, and Helm

## Prerequisites
```bash
./scripts/check-prerequisites.sh
```

## Validation
```bash
./scripts/validate-all.sh
```

## Generate Secrets
```bash
./scripts/generate-secrets.sh > .env
# Edit .env to verify secrets
```

---

## Docker Compose (Development)

### Start Services
```bash
docker compose -f docker-compose.dev.yml up -d
```

### Check Status
```bash
docker compose -f docker-compose.dev.yml ps
```

### View Logs
```bash
docker compose -f docker-compose.dev.yml logs -f
```

### Stop Services
```bash
docker compose -f docker-compose.dev.yml down
```

### Access Services
- API Gateway: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- MinIO: http://localhost:9001

---

## Docker Compose (Production)

### Prerequisites
```bash
# Create .env with all required secrets
cp .env.example .env
./scripts/generate-secrets.sh >> .env
```

### Start Services
```bash
docker compose -f docker-compose.prod.yml up -d
```

### Validate Config (Before Starting)
```bash
docker compose -f docker-compose.prod.yml config
```

---

## Kubernetes with Kustomize

### Create Namespace
```bash
kubectl create namespace vulcanami
```

### Create Secrets
```bash
kubectl create secret generic vulcanami-secrets \
 --from-literal=jwt-secret-key=$(openssl rand -base64 48) \
 --from-literal=bootstrap-key=$(openssl rand -base64 32) \
 --from-literal=postgres-password=$(openssl rand -base64 32) \
 --from-literal=redis-password=$(openssl rand -base64 32) \
 -n vulcanami
```

### Deploy Development
```bash
kubectl apply -k k8s/overlays/development
```

### Deploy Production
```bash
kubectl apply -k k8s/overlays/production
```

### Check Status
```bash
kubectl get all -n vulcanami
kubectl get pods -n vulcanami -w
```

### Port Forward
```bash
kubectl port-forward -n vulcanami svc/api-gateway 8000:8000
```

### View Logs
```bash
kubectl logs -n vulcanami -l app=vulcanami -f
```

### Delete
```bash
kubectl delete namespace vulcanami
```

---

## Helm Chart

### Validate Chart
```bash
helm lint helm/vulcanami
```

### Template (Test Rendering)
```bash
helm template vulcanami ./helm/vulcanami \
 --set image.tag=v1.0.0 \
 --set secrets.jwtSecretKey=YOUR_SECRET \
 --set secrets.bootstrapKey=YOUR_SECRET \
 --set secrets.postgresPassword=YOUR_SECRET \
 --set secrets.redisPassword=YOUR_SECRET
```

### Install
```bash
# Create namespace first
kubectl create namespace vulcanami

# Install with secrets
helm install vulcanami ./helm/vulcanami \
 --namespace vulcanami \
 --set image.tag=v1.0.0 \
 --set secrets.jwtSecretKey=$(openssl rand -base64 48 | tr -d '\n') \
 --set secrets.bootstrapKey=$(openssl rand -base64 32 | tr -d '\n') \
 --set secrets.postgresPassword=$(openssl rand -base64 32 | tr -d '\n') \
 --set secrets.redisPassword=$(openssl rand -base64 32 | tr -d '\n')
```

### Status
```bash
helm status vulcanami -n vulcanami
helm get values vulcanami -n vulcanami
```

### Upgrade
```bash
helm upgrade vulcanami ./helm/vulcanami -n vulcanami
```

### Rollback
```bash
helm rollback vulcanami -n vulcanami
```

### Uninstall
```bash
helm uninstall vulcanami -n vulcanami
```

---

## Docker Build

### Build Main Image
```bash
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:latest .
```

### Build Service Images
```bash
# API Gateway
docker build --build-arg REJECT_INSECURE_JWT=ack \
 -t vulcanami-api:latest -f docker/api/Dockerfile .

# DQS Service
docker build --build-arg REJECT_INSECURE_JWT=ack \
 -t vulcanami-dqs:latest -f docker/dqs/Dockerfile .

# PII Service
docker build --build-arg REJECT_INSECURE_JWT=ack \
 -t vulcanami-pii:latest -f docker/pii/Dockerfile .
```

### Run Image
```bash
docker run -d \
 -e JWT_SECRET_KEY=$(openssl rand -base64 48) \
 -p 5000:5000 \
 vulcanami:latest
```

---

## Health Checks

### Docker Compose
```bash
curl http://localhost:8000/health
```

### Kubernetes
```bash
kubectl get pods -n vulcanami
kubectl exec -n vulcanami POD_NAME -- curl localhost:8000/health
```

---

## Troubleshooting

### Docker Build Fails
```bash
# Check Docker is running
docker ps

# Check disk space
df -h

# Clean up old images
docker system prune -a
```

### Compose Validation Fails
```bash
# Check syntax
docker compose -f docker-compose.dev.yml config

# For production, ensure .env exists with all required vars
cat .env
```

### Kubernetes Connection Issues
```bash
# Check cluster access
kubectl cluster-info

# Check context
kubectl config current-context

# List contexts
kubectl config get-contexts
```

### Helm Template Errors
```bash
# Lint chart
helm lint helm/vulcanami

# Test template with debug
helm template vulcanami ./helm/vulcanami --debug
```

---

## Security Best Practices

1. **Never commit .env files**
2. **Generate strong random secrets** - use `openssl rand -base64 48`
3. **Use specific image tags** - avoid `latest` in production
4. **Rotate secrets regularly**
5. **Use secret management** - AWS Secrets Manager, HashiCorp Vault
6. **Network binding:**
 - Development: `127.0.0.1` (localhost only)
 - Docker/Kubernetes: `0.0.0.0` (container networking)
7. **Pin model versions** - set HuggingFace revision hashes

---

## Getting Help

- **Documentation**: See `*.md` files in repository root
- **Validation**: Run `./scripts/validate-all.sh`
- **New Engineer Guide**: Read `NEW_ENGINEER_SETUP.md`
- **Prerequisites**: Run `./scripts/check-prerequisites.sh`

---

## Common Commands Reference

```bash
# Validation
./scripts/validate-all.sh # Validate all configurations
./scripts/check-prerequisites.sh # Check installed tools
./scripts/generate-secrets.sh # Generate secrets

# Docker
docker ps # List running containers
docker logs CONTAINER_NAME # View logs
docker exec -it CONTAINER bash # Shell access

# Docker Compose
docker compose ps # List services
docker compose logs -f # Follow logs
docker compose restart SERVICE # Restart service

# Kubernetes
kubectl get pods -n vulcanami # List pods
kubectl describe pod POD -n vulcanami # Pod details
kubectl logs -f POD -n vulcanami # Follow logs
kubectl exec -it POD -- bash # Shell access

# Helm
helm list -n vulcanami # List releases
helm history vulcanami -n vulcanami # Release history
helm get manifest vulcanami -n vulcanami # View manifests
```

---

**Last Updated**: December 15, 2025
**Tested With**: Docker 28.0.4, Compose v2.38.2, kubectl v1.34.2, Helm v3.19.2
