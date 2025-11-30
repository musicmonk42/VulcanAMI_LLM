################################################################################
# VulcanAMI / Graphix Vulcan - Comprehensive Makefile
# Usage: make <target> [ARGS]
################################################################################

# Project configuration
PROJECT_NAME ?= vulcanami
IMAGE_PREFIX ?= vulcanami
REGISTRY ?= ghcr.io
REGISTRY_USERNAME ?= musicmonk42
TAG ?= latest
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
PYTHON_VERSION ?= 3.11
DOCKER_COMPOSE_DEV ?= docker-compose.dev.yml
DOCKER_COMPOSE_PROD ?= docker-compose.prod.yml

# Service images
IMAGE_MAIN ?= $(IMAGE_PREFIX)-main
IMAGE_API ?= $(IMAGE_PREFIX)-api
IMAGE_DQS ?= $(IMAGE_PREFIX)-dqs
IMAGE_PII ?= $(IMAGE_PREFIX)-pii

# Container names
CONTAINER_MAIN ?= $(IMAGE_MAIN)
CONTAINER_API ?= $(IMAGE_API)
CONTAINER_DQS ?= $(IMAGE_DQS)
CONTAINER_PII ?= $(IMAGE_PII)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.DEFAULT_GOAL := help

################################################################################
# Help
################################################################################

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)VulcanAMI / Graphix Vulcan - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Variables:$(NC)"
	@echo "  IMAGE_PREFIX=$(IMAGE_PREFIX)"
	@echo "  TAG=$(TAG)"
	@echo "  VERSION=$(VERSION)"
	@echo "  PYTHON_VERSION=$(PYTHON_VERSION)"

################################################################################
# Development Environment
################################################################################

.PHONY: install
install: ## Install Python dependencies
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@if [ -f setup.py ]; then \
		echo "$(GREEN)Installing local package...$(NC)"; \
		pip install -e .; \
	fi
	@echo "$(GREEN)Downloading spacy language model...$(NC)"
	python -m spacy download en_core_web_sm || echo "$(YELLOW)Spacy model download failed (non-critical)$(NC)"

.PHONY: install-dev
install-dev: install ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	pip install pytest pytest-cov pytest-asyncio pytest-timeout
	pip install black isort flake8 pylint mypy bandit
	pip install pre-commit

.PHONY: setup
setup: install-dev ## Setup development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	pre-commit install || echo "pre-commit not available"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "No .env.example found"; \
	fi

################################################################################
# Code Quality
################################################################################

.PHONY: format
format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ tests/ || true
	isort src/ tests/ || true

.PHONY: lint
lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(NC)"
	black --check src/ tests/ || true
	isort --check-only src/ tests/ || true
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
	pylint src/ --exit-zero || true

.PHONY: lint-security
lint-security: ## Run security linters
	@echo "$(GREEN)Running security linters...$(NC)"
	bandit -r src/ -ll || true

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(NC)"
	mypy src/ --ignore-missing-imports || true

################################################################################
# Testing
################################################################################

.PHONY: test
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ \
		--cov=src \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		-v

.PHONY: test-fast
test-fast: ## Run tests without slow tests
	@echo "$(GREEN)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	pytest tests/integration/ -v || echo "No integration tests found"

################################################################################
# Docker - Single Image
################################################################################

.PHONY: docker-build
docker-build: ## Build main Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build \
		--build-arg REJECT_INSECURE_JWT=ack \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE_MAIN):$(TAG) \
		-t $(IMAGE_MAIN):$(VERSION) \
		.

.PHONY: docker-build-no-cache
docker-build-no-cache: ## Build Docker image without cache
	@echo "$(GREEN)Building Docker image (no cache)...$(NC)"
	docker build --no-cache \
		--build-arg REJECT_INSECURE_JWT=ack \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE_MAIN):$(TAG) \
		.

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run --rm -it \
		--name $(CONTAINER_MAIN) \
		-e JWT_SECRET_KEY=$$(openssl rand -base64 48) \
		-e BOOTSTRAP_KEY=$$(openssl rand -base64 32) \
		-p 5000:5000 \
		$(IMAGE_MAIN):$(TAG)

.PHONY: docker-shell
docker-shell: ## Get shell in Docker container
	@echo "$(GREEN)Starting shell in container...$(NC)"
	docker run --rm -it \
		--entrypoint /bin/bash \
		-v "$$(pwd)":/app \
		$(IMAGE_MAIN):$(TAG)

.PHONY: docker-logs
docker-logs: ## Show Docker container logs
	docker logs -f $(CONTAINER_MAIN)

.PHONY: docker-stop
docker-stop: ## Stop Docker container
	@echo "$(GREEN)Stopping container...$(NC)"
	-docker stop $(CONTAINER_MAIN)
	-docker rm $(CONTAINER_MAIN)

################################################################################
# Docker - Multi-Service Build
################################################################################

.PHONY: docker-build-all
docker-build-all: ## Build all service Docker images
	@echo "$(GREEN)Building all Docker images...$(NC)"
	docker build --build-arg REJECT_INSECURE_JWT=ack -t $(IMAGE_MAIN):$(TAG) -f Dockerfile .
	docker build --build-arg REJECT_INSECURE_JWT=ack -t $(IMAGE_API):$(TAG) -f docker/api/Dockerfile . || true
	docker build --build-arg REJECT_INSECURE_JWT=ack -t $(IMAGE_DQS):$(TAG) -f docker/dqs/Dockerfile . || true
	docker build --build-arg REJECT_INSECURE_JWT=ack -t $(IMAGE_PII):$(TAG) -f docker/pii/Dockerfile . || true

.PHONY: docker-push-all
docker-push-all: ## Push all Docker images to registry
	@echo "$(GREEN)Pushing all images to registry...$(NC)"
	docker tag $(IMAGE_MAIN):$(TAG) $(REGISTRY)/$(REGISTRY_USERNAME)/$(IMAGE_MAIN):$(TAG)
	docker tag $(IMAGE_MAIN):$(TAG) $(REGISTRY)/$(REGISTRY_USERNAME)/$(IMAGE_MAIN):$(VERSION)
	docker push $(REGISTRY)/$(REGISTRY_USERNAME)/$(IMAGE_MAIN):$(TAG)
	docker push $(REGISTRY)/$(REGISTRY_USERNAME)/$(IMAGE_MAIN):$(VERSION)

################################################################################
# Docker Compose
################################################################################

.PHONY: up
up: ## Start all services with docker-compose (dev)
	@echo "$(GREEN)Starting development services...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_DEV) up -d

.PHONY: up-build
up-build: ## Build and start all services
	@echo "$(GREEN)Building and starting services...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_DEV) up -d --build

.PHONY: down
down: ## Stop all services
	@echo "$(GREEN)Stopping services...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_DEV) down

.PHONY: down-volumes
down-volumes: ## Stop all services and remove volumes
	@echo "$(YELLOW)Stopping services and removing volumes...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_DEV) down -v

.PHONY: ps
ps: ## Show running services
	docker compose -f $(DOCKER_COMPOSE_DEV) ps

.PHONY: logs-compose
logs-compose: ## Show logs from all services
	docker compose -f $(DOCKER_COMPOSE_DEV) logs -f

.PHONY: restart
restart: down up ## Restart all services

################################################################################
# Docker Compose - Production
################################################################################

.PHONY: prod-up
prod-up: ## Start production services
	@echo "$(GREEN)Starting production services...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_PROD) up -d

.PHONY: prod-down
prod-down: ## Stop production services
	@echo "$(GREEN)Stopping production services...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_PROD) down

.PHONY: prod-logs
prod-logs: ## Show production logs
	docker compose -f $(DOCKER_COMPOSE_PROD) logs -f

################################################################################
# Kubernetes
################################################################################

.PHONY: k8s-apply
k8s-apply: ## Apply Kubernetes manifests
	@echo "$(GREEN)Applying Kubernetes manifests...$(NC)"
	kubectl apply -k k8s/overlays/development/ || kubectl apply -k k8s/base/

.PHONY: k8s-delete
k8s-delete: ## Delete Kubernetes resources
	@echo "$(YELLOW)Deleting Kubernetes resources...$(NC)"
	kubectl delete -k k8s/overlays/development/ || kubectl delete -k k8s/base/

.PHONY: k8s-status
k8s-status: ## Show Kubernetes deployment status
	@echo "$(GREEN)Kubernetes Status:$(NC)"
	kubectl get all -n vulcanami-development || kubectl get all

.PHONY: k8s-logs
k8s-logs: ## Show Kubernetes pod logs
	kubectl logs -f -l app=vulcanami-api -n vulcanami-development || true

################################################################################
# Helm
################################################################################

.PHONY: helm-install
helm-install: ## Install with Helm
	@echo "$(GREEN)Installing with Helm...$(NC)"
	helm upgrade --install vulcanami ./helm/vulcanami \
		--create-namespace \
		--namespace vulcanami-development

.PHONY: helm-uninstall
helm-uninstall: ## Uninstall Helm release
	@echo "$(YELLOW)Uninstalling Helm release...$(NC)"
	helm uninstall vulcanami -n vulcanami-development

.PHONY: helm-template
helm-template: ## Show Helm template output
	helm template vulcanami ./helm/vulcanami

################################################################################
# CI/CD
################################################################################

.PHONY: ci-local
ci-local: lint test ## Run CI checks locally
	@echo "$(GREEN)Running CI checks locally...$(NC)"

.PHONY: ci-security
ci-security: lint-security ## Run security scans
	@echo "$(GREEN)Running security scans...$(NC)"
	trivy fs . || echo "Trivy not installed"

################################################################################
# Cleanup
################################################################################

.PHONY: clean
clean: docker-stop ## Clean up containers and temporary files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml || true
	rm -rf dist build || true

.PHONY: clean-all
clean-all: clean down-volumes ## Clean everything including volumes
	@echo "$(RED)Cleaning everything...$(NC)"
	docker system prune -f || true

################################################################################
# Database
################################################################################

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	alembic upgrade head || echo "Alembic not configured"

.PHONY: db-reset
db-reset: ## Reset database
	@echo "$(YELLOW)Resetting database...$(NC)"
	rm -f *.db *.db-journal || true

################################################################################
# Utilities
################################################################################

.PHONY: version
version: ## Show version information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Tag: $(TAG)"
	@echo "Python: $(PYTHON_VERSION)"

.PHONY: env-example
env-example: ## Create .env.example file
	@echo "$(GREEN)Creating .env.example...$(NC)"
	@echo "# JWT Configuration" > .env.example
	@echo "JWT_SECRET_KEY=generate-with-openssl-rand-base64-48" >> .env.example
	@echo "BOOTSTRAP_KEY=generate-with-openssl-rand-base64-32" >> .env.example
	@echo "" >> .env.example
	@echo "# Database" >> .env.example
	@echo "POSTGRES_DB=vulcanami" >> .env.example
	@echo "POSTGRES_USER=vulcanami" >> .env.example
	@echo "POSTGRES_PASSWORD=change-me-in-production" >> .env.example
	@echo "" >> .env.example
	@echo "# Redis" >> .env.example
	@echo "REDIS_PASSWORD=change-me-in-production" >> .env.example
	@echo "" >> .env.example
	@echo "# MinIO" >> .env.example
	@echo "MINIO_ROOT_USER=minioadmin" >> .env.example
	@echo "MINIO_ROOT_PASSWORD=change-me-in-production" >> .env.example
	@echo "" >> .env.example
	@echo "# Grafana" >> .env.example
	@echo "GRAFANA_USER=admin" >> .env.example
	@echo "GRAFANA_PASSWORD=change-me-in-production" >> .env.example
	@echo "" >> .env.example
	@echo "# Application" >> .env.example
	@echo "LOG_LEVEL=INFO" >> .env.example
	@echo "ENVIRONMENT=development" >> .env.example

################################################################################
# CI/CD and Validation
################################################################################

.PHONY: validate-cicd
validate-cicd: ## Validate CI/CD, Docker, and reproducibility configuration
	@echo "$(GREEN)Running comprehensive CI/CD validation...$(NC)"
	@chmod +x validate_cicd_docker.sh
	./validate_cicd_docker.sh

.PHONY: validate-docker
validate-docker: ## Validate Docker and Docker Compose configurations
	@echo "$(GREEN)Validating Docker configurations...$(NC)"
	docker compose -f $(DOCKER_COMPOSE_DEV) config > /dev/null && echo "$(GREEN)✓ docker-compose.dev.yml is valid$(NC)"
	@# Production compose requires env vars; use dummy values for validation syntax check
	@# Note: These values are for syntax validation only and meet minimum length requirements
	@JWT_SECRET_KEY=dummy-jwt-secret-key-for-validation-only-32-chars-minimum \
	 BOOTSTRAP_KEY=dummy-bootstrap-key-for-validation-only-32-chars-min \
	 POSTGRES_PASSWORD=dummy-postgres-password-validation-32char \
	 REDIS_PASSWORD=dummy-redis-password-validation-32chars \
	 MINIO_ROOT_USER=minioadmin \
	 MINIO_ROOT_PASSWORD=dummy-minio-password-validation-32char \
	 GRAFANA_PASSWORD=dummy-grafana-password-validation-32chars \
	 docker compose -f $(DOCKER_COMPOSE_PROD) config > /dev/null && echo "$(GREEN)✓ docker-compose.prod.yml is valid$(NC)"

.PHONY: generate-hashed-requirements
generate-hashed-requirements: ## Generate requirements-hashed.txt with SHA256 hashes
	@echo "$(GREEN)Generating hashed requirements...$(NC)"
	pip install pip-tools
	pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
	@echo "$(GREEN)✓ requirements-hashed.txt generated$(NC)"

.PHONY: generate-secrets
generate-secrets: ## Generate secure secrets for .env
	@echo "$(GREEN)Generating secure secrets...$(NC)"
	@echo "Add these to your .env file:"
	@echo ""
	@echo "JWT_SECRET_KEY=$$(openssl rand -base64 48)"
	@echo "BOOTSTRAP_KEY=$$(openssl rand -base64 32)"
	@echo "POSTGRES_PASSWORD=$$(openssl rand -base64 32)"
	@echo "REDIS_PASSWORD=$$(openssl rand -base64 32)"
	@echo "MINIO_ROOT_PASSWORD=$$(openssl rand -base64 24)"
	@echo "GRAFANA_PASSWORD=$$(openssl rand -base64 16)"

################################################################################
# Phony Targets
################################################################################

.PHONY: all
all: install lint test docker-build ## Run full build pipeline

################################################################################
# End of Makefile
################################################################################
