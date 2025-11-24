# CI/CD Pipeline Documentation

## Overview

This repository includes a comprehensive CI/CD pipeline using GitHub Actions. The pipeline automates testing, building, security scanning, and deployment of the VulcanAMI/Graphix Vulcan platform.

**✅ Validation Status**: All CI/CD configurations have been validated and are production-ready. 

## Testing and Validation

To validate your local setup and ensure reproducibility:

```bash
# Quick validation (recommended before commits)
./quick_test.sh quick

# Full comprehensive test suite
./test_full_cicd.sh

# Run pytest test suite
pytest tests/test_cicd_reproducibility.py -v

# Run existing validation script
./validate_cicd_docker.sh
```

For detailed testing instructions, see **[TESTING_GUIDE.md](TESTING_GUIDE.md)**.

## Pipeline Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main`, `develop`, `feature/**`, `hotfix/**` branches
- Pull requests to `main` and `develop`
- Manual workflow dispatch

**Jobs:**
- **Lint**: Code quality checks (Black, isort, Flake8, Pylint, Bandit)
- **Test**: Run tests on Python 3.11 and 3.12 with PostgreSQL and Redis
- **Integration Test**: Run integration tests with Docker Compose services
- **Build Validation**: Validate Docker image builds
- **Check Dependencies**: Scan for vulnerable dependencies

**Environment Variables Required:**
- `JWT_SECRET_KEY` (auto-generated in CI)
- `BOOTSTRAP_KEY` (auto-generated in CI)
- Database and Redis credentials (configured for test services)

### 2. Docker Build Workflow (`.github/workflows/docker.yml`)

**Triggers:**
- Push to `main`, `develop` branches
- Tags matching `v*.*.*`
- Pull requests to `main`
- Manual workflow dispatch

**Jobs:**
- **Build and Push**: Build multi-architecture (AMD64, ARM64) images for all services
- **Scan Images**: Security scanning with Trivy
- **Test Docker Compose**: Validate Docker Compose configurations (using v2 syntax)
- **Push to Registries**: Push versioned images on release tags

**Note**: This workflow uses Docker Compose v2 (`docker compose`) which is bundled with Docker Engine.

**Services Built:**
- `vulcanami-main`: Main application
- `vulcanami-api`: API Gateway
- `vulcanami-dqs`: Data Quality System
- `vulcanami-pii`: PII Detection Service

**Registry Credentials Required:**
- `GITHUB_TOKEN` (automatically provided)
- `DOCKERHUB_USERNAME` (optional)
- `DOCKERHUB_TOKEN` (optional)

### 3. Security Scanning Workflow (`.github/workflows/security.yml`)

**Triggers:**
- Push to `main`, `develop` branches
- Pull requests to `main`
- Daily schedule (2 AM UTC)
- Manual workflow dispatch

**Scans Performed:**
1. **CodeQL Analysis**: Static code analysis for security vulnerabilities
2. **Dependency Scan**: Check Python packages with pip-audit and Safety
3. **Secret Scan**: Detect exposed secrets with TruffleHog and GitLeaks
4. **SAST Scan**: Security scanning with Bandit and Semgrep
5. **Container Scan**: Docker image scanning with Trivy and Grype
6. **Infrastructure Scan**: IaC scanning with Checkov and Kubesec
7. **License Check**: Verify license compliance

**Results Location:**
- GitHub Security tab
- Workflow artifacts
- SARIF reports uploaded to GitHub

### 4. Deployment Workflow (`.github/workflows/deploy.yml`)

**Triggers:**
- Push to `main` (staging), `develop` (development)
- Tags matching `v*.*.*` (production)
- Manual workflow dispatch with environment selection

**Deployment Methods:**
1. **Kubernetes**: Deploy using kubectl with kustomize overlays
2. **Helm**: Deploy using Helm charts
3. **Docker Compose**: Deploy to development VM via SSH

**Environment Configuration:**
- `development`: Auto-deploys from `develop` branch
- `staging`: Auto-deploys from `main` branch
- `production`: Auto-deploys from version tags

**Required Secrets:**
- `KUBE_CONFIG`: Kubernetes config for kubectl
- `JWT_SECRET_KEY`: Application JWT secret
- `POSTGRES_PASSWORD`: Database password
- `REDIS_PASSWORD`: Redis password
- `MINIO_PASSWORD`: MinIO password
- `DEV_HOST`, `DEV_USERNAME`, `DEV_SSH_KEY`: For Docker Compose deployment
- `SLACK_WEBHOOK_URL` (optional): For deployment notifications

### 5. Release Management Workflow (`.github/workflows/release.yml`)

**Triggers:**
- Push of version tags (`v*.*.*`)
- Manual workflow dispatch

**Jobs:**
1. **Create Release**: Generate GitHub release with changelog
2. **Build Artifacts**: Build Python packages and source archives
3. **Publish to PyPI**: Publish package to Python Package Index
4. **Notify Release**: Send notifications to Slack

**Required Secrets:**
- `PYPI_API_TOKEN` (optional): For PyPI publishing
- `SLACK_WEBHOOK_URL` (optional): For notifications

### 6. Dependabot Configuration (`.github/dependabot.yml`)

**Automated Dependency Updates:**
- Python packages (weekly on Monday)
- GitHub Actions (weekly on Monday)
- Docker base images (weekly on Monday)
- Terraform modules (weekly on Monday)

## Setting Up CI/CD

### 1. Configure Repository Secrets

Navigate to `Settings > Secrets and variables > Actions` and add:

```bash
# Required secrets
JWT_SECRET_KEY=$(openssl rand -base64 48)
BOOTSTRAP_KEY=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
MINIO_PASSWORD=$(openssl rand -base64 24)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Optional secrets
DOCKERHUB_USERNAME=your-dockerhub-username
DOCKERHUB_TOKEN=your-dockerhub-token
PYPI_API_TOKEN=your-pypi-token
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
CODECOV_TOKEN=your-codecov-token

# Kubernetes deployment
KUBE_CONFIG=<base64-encoded-kubeconfig>

# Development VM deployment
DEV_HOST=dev.example.com
DEV_USERNAME=deploy
DEV_SSH_KEY=<private-ssh-key>
```

### 2. Configure Kubernetes

For Kubernetes deployments, create secrets:

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
```

### 3. Environment-Specific Configuration

Update kustomize overlays for each environment:

- **Development**: `k8s/overlays/development/`
- **Staging**: `k8s/overlays/staging/`
- **Production**: `k8s/overlays/production/`

### 4. Enable GitHub Container Registry

1. Generate a Personal Access Token with `write:packages` permission
2. Add as `GHCR_TOKEN` secret (or use default `GITHUB_TOKEN`)

## Makefile Commands

The project includes a comprehensive Makefile for local development:

```bash
# Development
make install          # Install dependencies
make install-dev      # Install dev dependencies
make setup           # Setup dev environment

# Code Quality
make format          # Format code
make lint           # Run linters
make lint-security  # Security linting
make type-check     # Type checking

# Testing
make test           # Run all tests
make test-cov       # Run tests with coverage
make test-integration  # Integration tests

# Docker
make docker-build       # Build Docker image
make docker-run         # Run container
make docker-build-all   # Build all service images

# Docker Compose (v2 syntax)
make up             # Start dev services with 'docker compose'
make down           # Stop services
make logs-compose   # View logs

# Kubernetes
make k8s-apply      # Apply K8s manifests
make k8s-status     # Check status

# Helm
make helm-install   # Install with Helm
make helm-template  # Show Helm template

# CI/CD Validation
make ci-local       # Run CI locally
make validate-cicd  # Validate all CI/CD configurations
make generate-secrets  # Generate secrets
```

**New**: Use `make validate-cicd` to run comprehensive validation of Docker, Docker Compose, GitHub Actions, Kubernetes, and Helm configurations.

## Monitoring and Observability

### Metrics

All services expose Prometheus metrics:
- API Gateway: `:9148/metrics`
- DQS Service: `:9145/metrics`
- PII Service: `:9147/metrics`

### Logs

Structured logging in JSON format:
- Application logs: Stdout/Stderr
- Audit logs: SQLite database
- Security events: Slack alerts (optional)

### Health Checks

Health endpoints:
- `/health`: Liveness and readiness
- `/metrics`: Prometheus metrics

## Troubleshooting

### Build Failures

1. Check workflow logs in GitHub Actions
2. Verify required secrets are set
3. Test Docker build locally:
   ```bash
   make docker-build
   ```

### Deployment Failures

1. Check Kubernetes pod status:
   ```bash
   kubectl get pods -n vulcanami
   kubectl describe pod <pod-name> -n vulcanami
   kubectl logs <pod-name> -n vulcanami
   ```

2. Verify secrets exist:
   ```bash
   kubectl get secrets -n vulcanami
   ```

### Test Failures

1. Run tests locally:
   ```bash
   make test-cov
   ```

2. Check test logs in GitHub Actions artifacts

### Security Scan Failures

1. Review security scan results in GitHub Security tab
2. Check workflow artifacts for detailed reports
3. Update vulnerable dependencies:
   ```bash
   pip install --upgrade <package>
   ```

## Best Practices

1. **Never commit secrets** to version control
2. **Use environment-specific configurations** for different deployments
3. **Run tests locally** before pushing
4. **Review security scan results** regularly
5. **Keep dependencies updated** (Dependabot helps)
6. **Use semantic versioning** for releases
7. **Write meaningful commit messages** for better changelogs
8. **Tag releases** for production deployments
9. **Use Docker Compose v2** (`docker compose` not `docker-compose`)
10. **Validate configurations** before deploying with `./validate_cicd_docker.sh`

## Validation Tool

### Comprehensive CI/CD Validation

The repository includes a comprehensive validation script that checks all aspects of CI/CD, Docker, and reproducibility:

```bash
./validate_cicd_docker.sh
```

**What it validates:**
1. ✅ Prerequisites (Docker, Docker Compose v2, kubectl, helm, pip-tools)
2. ✅ Requirements files (requirements.txt and requirements-hashed.txt with SHA256)
3. ✅ Docker configurations (Dockerfiles, security practices, healthchecks)
4. ✅ Docker Compose files (dev and prod configurations)
5. ✅ GitHub Actions workflows (YAML validity, modern syntax)
6. ✅ Kubernetes manifests (all K8s resources)
7. ✅ Helm charts (lint validation)
8. ✅ Entrypoint script (JWT validation logic)
9. ✅ Security configuration (.gitignore, no hardcoded secrets)
10. ✅ Reproducibility (pinned versions, hashed dependencies)

**Expected output:**
```
========================================
Validation Summary
========================================

Passed:   42
Warnings: 3
Failed:   0

✓ All critical checks passed!
```

Run this validation script:
- Before pushing changes
- After updating dependencies
- Before deploying to production
- As part of your CI/CD pipeline

### Dependency Management with Hashes

For reproducible builds, this repository uses hash-verified dependencies:

```bash
# Generate or update hashed requirements
pip install pip-tools
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt

# Docker builds automatically use hashed requirements when present
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:latest .
```

The Docker build will:
1. Check for `requirements-hashed.txt`
2. Use hash verification if available (secure)
3. Fall back to standard install if not (prints warning)

**Always generate hashed requirements for production!**

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Kustomize Documentation](https://kustomize.io/)
