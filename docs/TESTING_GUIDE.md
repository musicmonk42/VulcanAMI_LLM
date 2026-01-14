# Comprehensive CI/CD and Reproducibility Testing Guide

This guide explains how to run all conceivable tests to ensure this repository can be uploaded and reproduced in any CI/CD environment (GitHub Actions, Docker, Kubernetes, etc.).

## Quick Start

Run all tests with a single command:

```bash
./test_full_cicd.sh
```

This executes a comprehensive test suite covering all aspects of CI/CD and reproducibility.

## Development Environment Setup

Before running tests, set up your development environment:

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# 2. Install production dependencies
pip install -r requirements.txt

# 3. Install development dependencies (testing, linting, code quality tools)
pip install -r requirements-dev.txt

# 4. Verify installation
pytest --version
black --version
pip-compile --version
```

### Development Dependencies

The `requirements-dev.txt` file includes additional development tools:
- **Code Quality**: black, isort, flake8, pylint, mypy
- **Security**: bandit
- **Dependency Management**: pip-tools (for regenerating hashed requirements)
- **Development Tools**: ipython, ipdb
- **Documentation**: sphinx, sphinx-rtd-theme

**Note**: Testing tools (pytest, pytest-cov, pytest-asyncio, pytest-timeout, coverage) are already included in requirements.txt

### Regenerating Hashed Requirements

When dependencies change, regenerate the hashed requirements file:

```bash
# Install pip-tools (included in requirements-dev.txt)
pip install pip-tools

# Generate hashed requirements
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt

# Update to latest versions
pip-compile --upgrade requirements.txt -o requirements-hashed.txt
```

## Test Suites Available

### 1. Full Shell-Based Test Suite (`test_full_cicd.sh`)

**Purpose**: Comprehensive validation of all CI/CD components without requiring full dependency installation.

**What it tests**:
- Environment prerequisites (Docker, Python, Git, kubectl, helm)
- Repository structure (README, Dockerfile, requirements, Makefile, etc.)
- Configuration file validation (docker-compose, .gitignore, .dockerignore)
- GitHub Actions workflows (YAML validation)
- Docker build tests (security features, non-root user, healthchecks)
- Python dependencies and security
- Kubernetes manifest validation
- Helm chart validation
- Security configuration (no committed secrets, JWT validation)
- Existing validation scripts
- pytest CI/CD tests
- Documentation validation
- Reproducibility verification

**Usage**:
```bash
chmod +x test_full_cicd.sh
./test_full_cicd.sh
```

**Output**:
- Real-time colored console output
- Detailed logs in `test_results_<timestamp>/full_test.log`
- Summary report in `test_results_<timestamp>/summary.txt`

### 2. pytest CI/CD Test Suite (`tests/test_cicd_reproducibility.py`)

**Purpose**: Detailed programmatic tests using pytest framework.

**What it tests**:
- Docker configurations (security, health checks, multi-stage builds)
- Dependency management (hashed requirements, pinned versions)
- CI/CD workflows (GitHub Actions validation)
- Kubernetes manifests (YAML structure, resource definitions)
- Helm charts (lint validation, values files)
- Security configurations (gitignore, no secrets, bandit)
- Reproducibility features (pinned versions, Makefile, documentation)
- Validation scripts (existence and executability)
- End-to-end tests (Docker build and run)

**Usage**:
```bash
# Install minimal dependencies
pip install pytest pytest-cov pytest-timeout pyyaml python-dotenv

# Run all tests
pytest tests/test_cicd_reproducibility.py -v

# Run without slow tests (e.g., Docker builds)
pytest tests/test_cicd_reproducibility.py -v -m "not slow"

# Run specific test class
pytest tests/test_cicd_reproducibility.py::TestDockerConfigurations -v

# Run with coverage
pytest tests/test_cicd_reproducibility.py --cov=. --cov-report=html
```

### 3. Existing Validation Script (`validate_cicd_docker.sh`)

**Purpose**: Original validation script for CI/CD and Docker configurations.

**Usage**:
```bash
./validate_cicd_docker.sh
```

**What it checks**:
- Prerequisites (Docker, Docker Compose v2, yamllint, kubectl, helm)
- Requirements files (requirements.txt, requirements-hashed.txt)
- Docker configurations (Dockerfile, service Dockerfiles)
- Docker Compose files (dev and prod)
- GitHub Actions workflows
- Kubernetes manifests
- Helm charts
- Entrypoint script
- Security configuration
- Reproducibility configuration

### 4. Comprehensive Test Runner (`run_comprehensive_tests.sh`)

**Purpose**: Existing comprehensive test script.

**Usage**:
```bash
./run_comprehensive_tests.sh
```

## Test Categories

### A. Docker and Container Tests

1. **Docker Build Tests**
 ```bash
 # Build main image
 docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:test .
 
 # Build service images
 docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-api:test -f docker/api/Dockerfile .
 docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-dqs:test -f docker/dqs/Dockerfile .
 docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-pii:test -f docker/pii/Dockerfile .
 ```

2. **Docker Run Tests**
 ```bash
 # Generate secure secrets
 JWT_SECRET=$(openssl rand -base64 48)
 BOOTSTRAP_KEY=$(openssl rand -base64 32)
 
 # Run container
 docker run --rm \
 -e JWT_SECRET_KEY=$JWT_SECRET \
 -e BOOTSTRAP_KEY=$BOOTSTRAP_KEY \
 vulcanami:test python --version
 ```

3. **Docker Compose Tests**
 ```bash
 # Validate configurations
 docker compose -f docker-compose.dev.yml config
 docker compose -f docker-compose.prod.yml config
 
 # Start services
 docker compose -f docker-compose.dev.yml up -d
 
 # Check health
 docker compose -f docker-compose.dev.yml ps
 
 # Cleanup
 docker compose -f docker-compose.dev.yml down -v
 ```

### B. Kubernetes and Helm Tests

1. **Kubernetes Manifest Validation**
 ```bash
 # Validate YAML syntax
 yamllint k8s/
 
 # Validate with kubectl (dry-run)
 kubectl apply -f k8s/base --dry-run=client
 
 # Check with kustomize
 kubectl apply -k k8s/overlays/development --dry-run=client
 ```

2. **Helm Chart Tests**
 ```bash
 # Lint chart
 helm lint helm/vulcanami
 
 # Template output (dry-run)
 helm template vulcanami helm/vulcanami
 
 # Validate with real cluster (if available)
 helm install vulcanami helm/vulcanami --dry-run --debug
 ```

### C. Dependency and Security Tests

1. **Dependency Verification**
 ```bash
 # Check for hashed requirements
 grep -c "sha256:" requirements-hashed.txt
 
 # Verify no unpinned dependencies
 grep -v "^#" requirements.txt | grep -v "^$" | grep -v "==" | grep -v ">="
 
 # Check for vulnerabilities (if pip-audit installed)
 pip-audit -r requirements.txt
 ```

2. **Security Scans**
 ```bash
 # Bandit security scan
 bandit -r src/ -ll
 
 # Check for secrets in git
 git grep -i "password\|secret\|api_key" src/
 
 # Trivy container scan (if installed)
 trivy image vulcanami:test
 ```

### D. GitHub Actions Workflow Tests

1. **Workflow Validation**
 ```bash
 # Validate YAML syntax
 yamllint .github/workflows/
 
 # Check with Python YAML parser
 python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
 ```

2. **Local Workflow Testing** (with act, if installed)
 ```bash
 # Install act: https://github.com/nektos/act
 
 # Run CI workflow locally
 act -j lint
 act -j test
 ```

### E. Reproducibility Tests

1. **Version Pinning Verification**
 ```bash
 # Check Dockerfile uses specific Python version
 grep "FROM python:[0-9]" Dockerfile
 
 # Check all dependencies are pinned
 grep -c "==" requirements.txt
 
 # Verify git commit hash
 git rev-parse HEAD
 ```

2. **Build Reproducibility**
 ```bash
 # Build image twice with same tag
 docker build --build-arg REJECT_INSECURE_JWT=ack -t test:1 .
 IMAGE1=$(docker images --no-trunc test:1 -q)
 
 docker build --build-arg REJECT_INSECURE_JWT=ack -t test:2 .
 IMAGE2=$(docker images --no-trunc test:2 -q)
 
 # Compare (note: timestamps may differ, but layers should match)
 docker history test:1
 docker history test:2
 ```

## Using Makefile

The project includes a comprehensive Makefile for common operations:

```bash
# Show all available targets
make help

# Development setup
make install # Install Python dependencies
make install-dev # Install dev dependencies
make setup # Full development setup

# Code quality
make format # Format code
make lint # Run linters
make lint-security # Run security linters
make type-check # Run type checking

# Testing
make test # Run tests
make test-cov # Run tests with coverage
make test-fast # Run fast tests only

# Docker
make docker-build # Build Docker image
make docker-run # Run Docker container
make docker-stop # Stop container

# Docker Compose
make up # Start all services
make down # Stop all services
make logs-compose # View logs

# Kubernetes
make k8s-apply # Apply K8s manifests
make k8s-status # Check K8s status

# Helm
make helm-install # Install with Helm
make helm-uninstall # Uninstall Helm release

# CI/CD
make ci-local # Run CI checks locally
make validate-cicd # Validate CI/CD configuration

# Full pipeline
make all # Run full build pipeline
```

## Continuous Integration

### GitHub Actions

The repository includes multiple CI/CD workflows:

1. **CI Workflow** (`.github/workflows/ci.yml`)
 - Runs on: push to main/develop, pull requests
 - Tests: lint, unit tests, integration tests
 - Matrix: Python 3.11

2. **Docker Workflow** (`.github/workflows/docker.yml`)
 - Runs on: push to main/develop, version tags
 - Builds: Multi-architecture images (AMD64, ARM64)
 - Pushes: GitHub Container Registry, Docker Hub

3. **Security Workflow** (`.github/workflows/security.yml`)
 - Runs on: push, pull requests, daily schedule
 - Scans: CodeQL, dependencies, secrets, containers

4. **Deployment Workflow** (`.github/workflows/deploy.yml`)
 - Runs on: version tags, manual trigger
 - Deploys: Kubernetes, Helm, Docker Compose

### Running Workflows Locally

To test workflows before pushing:

```bash
# Run full CI/CD validation
./test_full_cicd.sh

# Run pytest tests
pytest tests/test_cicd_reproducibility.py -v

# Run existing validation
./validate_cicd_docker.sh
```

## Common Issues and Solutions

### Issue: Docker build fails with JWT secret error
**Solution**: Always provide `--build-arg REJECT_INSECURE_JWT=ack`
```bash
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:test .
```

### Issue: Docker Compose validation fails with missing environment variables
**Solution**: Set dummy values or use .env file
```bash
export POSTGRES_PASSWORD="dummy"
export JWT_SECRET_KEY="dummy"
docker compose config
```

### Issue: pytest can't find modules
**Solution**: Install minimal dependencies
```bash
pip install pytest pytest-cov pyyaml python-dotenv
```

### Issue: Kubernetes manifests have multiple documents
**Solution**: Use `yaml.safe_load_all()` instead of `yaml.safe_load()`
```python
with open('manifest.yaml') as f:
 docs = list(yaml.safe_load_all(f))
```

## Pre-Deployment Checklist

Before deploying or uploading to a new environment:

- [ ] Run `./test_full_cicd.sh` - all tests pass
- [ ] Run `pytest tests/test_cicd_reproducibility.py` - all tests pass
- [ ] Run `./validate_cicd_docker.sh` - no critical failures
- [ ] Docker build succeeds without errors
- [ ] Docker image runs successfully
- [ ] Docker Compose starts all services
- [ ] All environment variables documented
- [ ] Secrets stored securely (not in code)
- [ ] Documentation up to date
- [ ] Version tagged in git
- [ ] requirements-hashed.txt is current
- [ ] GitHub Actions workflows validated

## Test Output Interpretation

### Success Indicators
- ✓ (green) = Test passed
- All critical tests show ✓
- Exit code: 0
- Message: "All Critical Tests Passed!"

### Warning Indicators
- ⊘ (yellow) = Test skipped (often expected)
- Optional components not present
- Non-critical validations

### Failure Indicators
- ✗ (red) = Test failed
- Exit code: non-zero
- Message: "Some Tests Failed!"
- Review detailed logs in `test_results_*/`

## Additional Resources

- **CI/CD Documentation**: `CI_CD.md`
- **Reproducible Builds**: `REPRODUCIBLE_BUILDS.md`
- **Deployment Guide**: `DEPLOYMENT.md`
- **Security Guide**: `INFRASTRUCTURE_SECURITY_GUIDE.md`
- **Quick Start**: `QUICKSTART.md`

## Support and Troubleshooting

For issues or questions:
1. Check test output logs in `test_results_*/`
2. Review GitHub Actions workflow runs
3. Consult documentation files
4. Check Docker logs: `docker logs <container>`
5. Validate configurations manually with provided commands

## Contributing

When adding new features:
1. Update relevant tests in `tests/test_cicd_reproducibility.py`
2. Add validation to `test_full_cicd.sh` if needed
3. Update documentation
4. Run full test suite before committing
5. Ensure GitHub Actions workflows pass

---

**Last Updated**: 2025-11-25 
**Maintained By**: VulcanAMI Team 
**Status**: ✅ All tests passing, ready for production deployment
