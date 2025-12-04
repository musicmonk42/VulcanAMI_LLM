# CI/CD Quick Start Guide

## 🚀 Quick Start

### Generate Secrets
```bash
# Option 1: Use .env.example as template
cp .env.example .env
# Then edit .env file and replace placeholder values

# Option 2: Auto-generate secrets
make generate-secrets > .env
# Edit .env file with your favorite editor
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies (testing, linting, code quality tools)
pip install -r requirements-dev.txt

# Start all services
make up

# Run tests
make test

# Check status
make ps

# View logs
make logs-compose

# Stop services
make down
```

### Docker Build
```bash
# Build main image
make docker-build

# Build all service images
make docker-build-all

# Run container
make docker-run
```

### Kubernetes Deployment
```bash
# Apply manifests (development)
make k8s-apply

# Check status
make k8s-status

# View logs
make k8s-logs
```

### Helm Deployment
```bash
# Install with Helm
make helm-install

# View template
make helm-template

# Uninstall
make helm-uninstall
```

## 📋 Available Make Commands

Run `make help` to see all available commands.

### Categories:
- **Development**: install, install-dev, setup
- **Code Quality**: format, lint, lint-security, type-check
- **Testing**: test, test-cov, test-fast, test-integration
- **Docker**: docker-build, docker-run, docker-shell, docker-build-all
- **Docker Compose**: up, down, ps, logs-compose, restart
- **Kubernetes**: k8s-apply, k8s-status, k8s-logs, k8s-delete
- **Helm**: helm-install, helm-uninstall, helm-template
- **CI/CD**: ci-local, ci-security
- **Utilities**: clean, version, generate-secrets, env-example

## 🔐 Required Secrets

**Important Security Note:** All placeholder values in `.env.example` and test files are clearly marked with comments indicating they are not real secrets. This helps prevent false positive security alerts during automated scanning.

### GitHub Repository Secrets

Set these in `Settings > Secrets and variables > Actions`:

```
JWT_SECRET_KEY          # Generate with: openssl rand -base64 48
BOOTSTRAP_KEY           # Generate with: openssl rand -base64 32
POSTGRES_PASSWORD       # Generate with: openssl rand -base64 32
REDIS_PASSWORD          # Generate with: openssl rand -base64 32
MINIO_PASSWORD          # Generate with: openssl rand -base64 24
GRAFANA_PASSWORD        # Generate with: openssl rand -base64 16
```

Optional:
```
DOCKERHUB_USERNAME      # For Docker Hub registry
DOCKERHUB_TOKEN         # Docker Hub access token
PYPI_API_TOKEN         # For PyPI publishing
SLACK_WEBHOOK_URL      # For notifications
CODECOV_TOKEN          # For code coverage
KUBE_CONFIG            # For K8s deployment
```

### Kubernetes Secrets

```bash
kubectl create secret generic vulcanami-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
  --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=MINIO_SECRET_KEY=$(openssl rand -base64 24) \
  -n vulcanami
```

## 🔄 Workflows

### Continuous Integration (ci.yml)
- Triggered on: push, pull_request
- Jobs: lint, test (Python 3.11 & 3.12), integration-test, build-validation
- Services: PostgreSQL, Redis

### Docker Build (docker.yml)
- Triggered on: push to main/develop, tags, pull_request
- Builds: Multi-arch (AMD64, ARM64) images
- Services: main, api, dqs, pii
- Security: Trivy scanning, SBOM generation

### Security Scanning (security.yml)
- Triggered on: push, pull_request, schedule (daily)
- Scans: CodeQL, dependencies, secrets, SAST, containers, IaC, licenses

### Deployment (deploy.yml)
- Triggered on: push to main/develop, tags
- Methods: Kubernetes, Helm, Docker Compose
- Environments: development, staging, production

### Release Management (release.yml)
- Triggered on: version tags (v*.*.*)
- Creates: GitHub release, artifacts, PyPI package
- Notifications: Slack

## 📦 Deployment Environments

### Development
- Branch: `develop`
- Auto-deploy: Yes
- Services: Full stack with dev tools

### Staging
- Branch: `main`
- Auto-deploy: Yes
- Services: Production-like environment

### Production
- Tags: `v*.*.*`
- Auto-deploy: Yes (with approval)
- Services: Production stack

## 🐛 Troubleshooting

### Workflow Failures
```bash
# Check workflow status
gh workflow list

# View workflow run
gh run view <run-id>

# Re-run failed jobs
gh run rerun <run-id>
```

### Local Testing
```bash
# Run CI checks locally
make ci-local

# Run security scans
make ci-security

# Test Docker build
make docker-build
```

### Deployment Issues
```bash
# Check K8s pods
kubectl get pods -n vulcanami

# View pod logs
kubectl logs -f <pod-name> -n vulcanami

# Describe pod
kubectl describe pod <pod-name> -n vulcanami
```

## 📚 Documentation

- [CI_CD.md](./CI_CD.md) - Detailed CI/CD documentation
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment guide
- [README.md](./README.md) - Project overview

## 🛠️ Tools Required

- Docker 20.10+
- Docker Compose 2.0+
- kubectl 1.24+ (for K8s)
- Helm 3.10+ (for Helm)
- Python 3.11+
- Make
- Git

## 📞 Support

- Issues: https://github.com/musicmonk42/VulcanAMI_LLM/issues
- Documentation: See docs in repository
- Email: support@vulcanami.io
