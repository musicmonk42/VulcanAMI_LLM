# Docker Build Guide

## Quick Start

**✅ Status**: All Docker configurations validated and working correctly. Requirements file with SHA256 hashes is up-to-date.

### Prerequisites
- Docker 20.10+ installed
- Docker Compose v2+
- Network access for downloading dependencies

### Validate Configurations
```bash
# Quick validation (no build)
./quick_docker_validation.sh

# Comprehensive validation
./validate_cicd_docker.sh

# Full build testing (requires network access)
./test_docker_build.sh
```

### Build and Run

#### 1. Setup Environment
```bash
# Copy example environment file
cp .env.example .env

# Generate secure secrets
export JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '+/')
export BOOTSTRAP_KEY=$(openssl rand -base64 32 | tr -d '+/')
export POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d '+/')
export REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '+/')

# Update .env file with your secrets
# DO NOT commit .env to git!

# Security: Set network binding
# - Development (local): Use 127.0.0.1 (localhost only - secure default)
# - Docker/Container: Use 0.0.0.0 (container networking)
export HOST=0.0.0.0  # For Docker
export API_HOST=0.0.0.0  # For Docker

# Optional: Pin HuggingFace models to specific versions (production recommended)
# export VULCAN_TEXT_MODEL_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594
```

#### 2. Build Images

Using Makefile:
```bash
# Build main image
make docker-build

# Build all service images
make docker-build-all

# Build without cache
make docker-build-no-cache
```

Using Docker directly:
```bash
# Main application
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:latest .

# API Gateway
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-api:latest -f docker/api/Dockerfile .

# DQS Service
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-dqs:latest -f docker/dqs/Dockerfile .

# PII Service
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-pii:latest -f docker/pii/Dockerfile .
```

#### 3. Run Services

Development environment:
```bash
# Using Makefile
make up-build

# Using Docker Compose
docker compose -f docker-compose.dev.yml up -d
```

Production environment:
```bash
# Ensure .env has all required secrets
docker compose -f docker-compose.prod.yml up -d
```

## Docker Images

### Main Application (`Dockerfile`)
- **Base**: python:3.10.11-slim
- **User**: graphix (UID 1001)
- **Ports**: 8000
- **Features**:
  - Multi-stage build (builder + runtime)
  - Hash-verified dependencies
  - JWT validation on startup
  - Healthcheck enabled
  - Non-root execution
- **Includes**: `src/`, `configs/` directories

### API Gateway (`docker/api/Dockerfile`)
- **User**: apiuser (UID 1001)
- **Ports**: 8000 (API), 9148 (metrics)
- **Command**: `uvicorn src.api_gateway:app`
- **Includes**: `src/`, `configs/` directories

### DQS Service (`docker/dqs/Dockerfile`)
- **User**: dqs (UID 1001)
- **Ports**: 8080 (API), 9145 (metrics)
- **Command**: `uvicorn src.dqs_service:app`
- **Includes**: PostgreSQL client libraries, `src/`, `configs/` directories

### PII Service (`docker/pii/Dockerfile`)
- **User**: pii (UID 1001)
- **Ports**: 8082 (API), 9147 (metrics)
- **Command**: `uvicorn src.pii_service:app`
- **Volumes**: `/models` for PII detection models
- **Includes**: `src/`, `configs/` directories

## Security Features

### 1. JWT Validation
All Docker images require a strong JWT secret at runtime:
- Minimum 32 characters
- No weak patterns (password, secret, etc.)
- Validated by entrypoint.sh

Provide via environment variable:
```bash
export JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '+/')
```

### 2. Build-time Security
- Must acknowledge no embedded secrets: `--build-arg REJECT_INSECURE_JWT=ack`
- Multi-stage builds minimize attack surface
- All images run as non-root users
- OS packages updated during build
- No secrets in image layers

### 3. Runtime Security
- Healthchecks monitor service health
- Resource limits prevent DoS
- Internal networks isolate backend services
- TLS/HTTPS in production (via nginx)

## Reproducibility

### Hashed Dependencies
All Python dependencies verified with SHA256 hashes:
```bash
# File: requirements-hashed.txt
# Contains: 4007+ SHA256 hashes
```

Update hashed requirements:
```bash
pip install pip-tools
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
```

### Pinned Versions
All base images and dependencies use exact versions:
- Python: `3.10.11-slim`
- PostgreSQL: `14-alpine`
- Redis: `7-alpine`
- MinIO: `RELEASE.2025-01-10T00-00-00Z`
- etc.

### Image Tagging
Always use semantic versioning in production:
```bash
# Good
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:v1.0.0 .
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:$(git rev-parse --short HEAD) .

# Bad (never in production - uses :latest tag)
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:latest .
```

## Docker Compose Services

### Development (`docker-compose.dev.yml`)
- Development-friendly configuration
- Volume mounts for live code reloading
- Debug logging enabled
- No resource constraints

### Production (`docker-compose.prod.yml`)
**Services**:
- **postgres**: Data persistence
- **redis**: Caching and sessions
- **minio**: Object storage
- **api-gateway**: Main API
- **dqs-service**: Data quality system
- **pii-service**: PII detection
- **prometheus**: Metrics collection
- **grafana**: Visualization
- **nginx**: Reverse proxy

**Networks**:
- `vulcanami-prod`: External access
- `backend`: Internal services (isolated)

**Volumes**:
- All data persisted to named volumes
- Survives container restarts

## Environment Variables

### Required (Production)
```bash
JWT_SECRET_KEY        # API authentication (>=32 chars)
BOOTSTRAP_KEY         # Initial setup key
POSTGRES_PASSWORD     # Database password
REDIS_PASSWORD        # Cache password
MINIO_ROOT_USER       # Object storage user
MINIO_ROOT_PASSWORD   # Object storage password
GRAFANA_PASSWORD      # Dashboard password
```

### Optional
```bash
ENVIRONMENT           # development|production
LOG_LEVEL             # DEBUG|INFO|WARNING|ERROR
API_PORT              # Default: 8000
DATABASE_URL          # Override default connection
```

## Validation Scripts

### quick_docker_validation.sh
Fast validation without building:
- Checks Dockerfile syntax
- Validates configurations
- Tests security features
- ~10 seconds

### validate_cicd_docker.sh
Comprehensive validation:
- Docker configurations
- Compose files
- Kubernetes manifests
- Helm charts
- GitHub Actions
- Security checks
- ~30 seconds

### test_docker_build.sh
Full build testing:
- Builds all images
- Tests all features
- Security validation
- Reproducibility checks
- ~15+ minutes (requires network)

## Troubleshooting

### Build fails with SSL errors
**Issue**: Certificate verification errors during pip install

**Solution**: This typically happens in CI environments with self-signed certificates. The Docker configurations are correct. Build in a standard environment with proper SSL certificates.

### Build fails with REJECT_INSECURE_JWT error
**Issue**: Build fails with "Refusing to build: set --build-arg REJECT_INSECURE_JWT=ack"

**Solution**: This is a security check to ensure you acknowledge that no JWT secrets are embedded in the image.

For local Docker builds:
```bash
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:latest .
```

For Railway deployment:
Set `REJECT_INSECURE_JWT=ack` as an environment variable in your Railway service's Variables tab. Railway passes environment variables to Docker builds automatically.

For Google Cloud Build / Cloud Run:
A `cloudbuild.yaml` file is included in the repository that automatically passes the build argument:
```bash
gcloud builds submit --config cloudbuild.yaml .
```
Or for Cloud Run source-based deployments, the `cloudbuild.yaml` is automatically detected.

For docker-compose:
```yaml
services:
  your-service:
    build:
      context: .
      args:
        REJECT_INSECURE_JWT: ack
```

### Entrypoint fails with JWT error
**Issue**: "ERROR: No valid JWT secret provided"

**Solution**: Set a strong JWT secret:
```bash
export JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '+/')
docker run -e JWT_SECRET_KEY="$JWT_SECRET_KEY" vulcanami:latest
```

### Docker Compose fails with missing variables
**Issue**: "required variable X is missing"

**Solution**: Create .env file with all required variables:
```bash
cp .env.example .env
# Edit .env and add your secrets
```

### Permission denied errors
**Issue**: Files owned by root

**Solution**: All images use non-root users. Check volume permissions:
```bash
chown -R 1001:1001 /path/to/volume
```

## Monitoring and Health

### Healthchecks
All services have healthchecks:
```bash
# Check service health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View healthcheck logs
docker inspect --format='{{.State.Health.Status}}' container_name
```

### Metrics
Services expose Prometheus metrics:
- API Gateway: `:9148/metrics`
- DQS Service: `:9145/metrics`
- PII Service: `:9147/metrics`

Access Grafana dashboard:
```bash
# Development
http://localhost:3000

# Login: admin / [GRAFANA_PASSWORD from .env]
```

## Best Practices

### Development
1. Use `docker-compose.dev.yml`
2. Mount source code as volumes for live reload
3. Enable debug logging
4. Don't commit .env files

### Production
1. Use `docker-compose.prod.yml`
2. Store secrets in secret management (AWS Secrets Manager, Vault)
3. Use specific version tags (never `latest`)
4. Enable monitoring and alerting
5. Regular security updates
6. Backup volumes regularly

### CI/CD
1. Validate configurations in CI
2. Build and scan images
3. Tag images with git SHA
4. Deploy with specific tags
5. Run smoke tests after deployment

### Railway Deployment
Railway deployment is configured via `railway.toml`. However, Railway does NOT support `[build.args]` in the TOML configuration. Instead, you must set the `REJECT_INSECURE_JWT` environment variable in Railway's service settings.

**Setting up Railway for this repository:**

1. Connect your Railway service to this repository
2. Go to your service's "Variables" tab in Railway dashboard
3. Add the following environment variable:
   ```
   REJECT_INSECURE_JWT=ack
   ```
4. Railway will automatically pass this to the Docker build process

**Required Runtime Environment Variables** (also set in Railway's Variables tab):
- `JWT_SECRET_KEY` - Required, minimum 32 characters
- `BOOTSTRAP_KEY` - Required for initial setup

The `railway.toml` file configures the Dockerfile location and deployment settings:

## Additional Resources

- [REPRODUCIBLE_BUILDS.md](REPRODUCIBLE_BUILDS.md) - Reproducibility guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment documentation
- [INFRASTRUCTURE_SECURITY_GUIDE.md](INFRASTRUCTURE_SECURITY_GUIDE.md) - Security guide

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run validation scripts
3. Review documentation
4. Check GitHub Issues

---

**Last Updated**: December 4, 2025  
**Docker Version**: 28.0.4  
**Compose Version**: v2.38.2
