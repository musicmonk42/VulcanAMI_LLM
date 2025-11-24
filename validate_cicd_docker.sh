#!/bin/bash
################################################################################
# Comprehensive CI/CD and Docker Reproducibility Validation Script
# This script validates all aspects of the CI/CD pipeline, Docker
# configurations, and reproducibility setup
################################################################################

# Don't exit on error, we want to collect all results
set +e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# 1. Prerequisites Check
################################################################################
print_header "1. Checking Prerequisites"

if command_exists docker; then
    print_success "Docker is installed ($(docker --version))"
else
    print_error "Docker is not installed"
fi

if docker compose version >/dev/null 2>&1; then
    print_success "Docker Compose is installed ($(docker compose version))"
else
    print_error "Docker Compose is not installed"
fi

if command_exists yamllint; then
    print_success "yamllint is installed"
else
    print_warning "yamllint is not installed (optional)"
fi

if command_exists kubectl; then
    print_success "kubectl is installed"
else
    print_warning "kubectl is not installed (optional)"
fi

if command_exists helm; then
    print_success "helm is installed"
else
    print_warning "helm is not installed (optional)"
fi

if command_exists pip-compile; then
    print_success "pip-tools is installed"
else
    print_warning "pip-tools is not installed (needed for requirements-hashed.txt generation)"
fi

################################################################################
# 2. Requirements Files Validation
################################################################################
print_header "2. Validating Requirements Files"

if [ -f "requirements.txt" ]; then
    print_success "requirements.txt exists"
else
    print_error "requirements.txt not found"
fi

if [ -f "requirements-hashed.txt" ]; then
    if grep -qE '^[^#].*--hash=sha256:' requirements-hashed.txt; then
        print_success "requirements-hashed.txt contains SHA256 hashes"
    else
        print_error "requirements-hashed.txt does not contain proper hashes"
    fi
else
    print_error "requirements-hashed.txt not found"
fi

################################################################################
# 3. Docker Configuration Validation
################################################################################
print_header "3. Validating Docker Configurations"

# Check main Dockerfile
if [ -f "Dockerfile" ]; then
    print_success "Main Dockerfile exists"
    
    # Check for security best practices
    if grep -q "USER graphix" Dockerfile; then
        print_success "Dockerfile uses non-root user"
    else
        print_error "Dockerfile does not specify non-root user"
    fi
    
    if grep -q "HEALTHCHECK" Dockerfile; then
        print_success "Dockerfile has HEALTHCHECK"
    else
        print_warning "Dockerfile missing HEALTHCHECK"
    fi
    
    if grep -q "REJECT_INSECURE_JWT" Dockerfile; then
        print_success "Dockerfile validates JWT secret configuration"
    else
        print_warning "Dockerfile missing JWT secret validation"
    fi
else
    print_error "Main Dockerfile not found"
fi

# Check service Dockerfiles
for service in api dqs pii; do
    if [ -f "docker/$service/Dockerfile" ]; then
        print_success "docker/$service/Dockerfile exists"
    else
        print_warning "docker/$service/Dockerfile not found"
    fi
done

################################################################################
# 4. Docker Compose Validation
################################################################################
print_header "4. Validating Docker Compose Files"

# Create temporary .env for validation
cat > .env.test << 'EOF'
JWT_SECRET_KEY=test-secret-key-minimum-32-characters-long-for-validation
BOOTSTRAP_KEY=test-bootstrap-key-minimum-32-chars
POSTGRES_PASSWORD=test-postgres-password-minimum-32-chars
REDIS_PASSWORD=test-redis-password-minimum-32-chars
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=test-minio-password-minimum-24-chars
GRAFANA_PASSWORD=test-grafana-password
EOF

if [ -f "docker-compose.dev.yml" ]; then
    if docker compose -f docker-compose.dev.yml --env-file .env.test config > /dev/null 2>&1; then
        print_success "docker-compose.dev.yml is valid"
    else
        print_error "docker-compose.dev.yml has errors"
    fi
else
    print_error "docker-compose.dev.yml not found"
fi

if [ -f "docker-compose.prod.yml" ]; then
    if docker compose -f docker-compose.prod.yml --env-file .env.test config > /dev/null 2>&1; then
        print_success "docker-compose.prod.yml is valid"
    else
        print_error "docker-compose.prod.yml has errors"
    fi
else
    print_error "docker-compose.prod.yml not found"
fi

rm -f .env.test

################################################################################
# 5. GitHub Actions Workflows Validation
################################################################################
print_header "5. Validating GitHub Actions Workflows"

workflow_files=(
    ".github/workflows/ci.yml"
    ".github/workflows/docker.yml"
    ".github/workflows/security.yml"
    ".github/workflows/deploy.yml"
    ".github/workflows/release.yml"
    ".github/workflows/infrastructure-validation.yml"
)

for workflow in "${workflow_files[@]}"; do
    if [ -f "$workflow" ]; then
        if command_exists yamllint; then
            if yamllint -d relaxed "$workflow" > /dev/null 2>&1; then
                print_success "$workflow is valid YAML"
            else
                print_warning "$workflow has YAML lint warnings"
            fi
        else
            print_success "$workflow exists"
        fi
        
        # Check for deprecated docker-compose usage
        if grep -q "docker-compose " "$workflow"; then
            print_warning "$workflow uses deprecated 'docker-compose' command (should be 'docker compose')"
        fi
    else
        print_error "$workflow not found"
    fi
done

################################################################################
# 6. Kubernetes Manifests Validation
################################################################################
print_header "6. Validating Kubernetes Manifests"

if [ -d "k8s/base" ]; then
    k8s_files=$(find k8s/base -name "*.yaml" ! -name "kustomization.yaml" 2>/dev/null)
    if [ -n "$k8s_files" ]; then
        for k8s_file in $k8s_files; do
            if command_exists yamllint; then
                if yamllint -d relaxed "$k8s_file" > /dev/null 2>&1; then
                    print_success "$k8s_file is valid YAML"
                else
                    print_warning "$k8s_file has YAML lint warnings"
                fi
            else
                print_success "$k8s_file exists"
            fi
        done
    else
        print_warning "No Kubernetes manifests found in k8s/base"
    fi
else
    print_warning "k8s/base directory not found"
fi

################################################################################
# 7. Helm Charts Validation
################################################################################
print_header "7. Validating Helm Charts"

if [ -d "helm/vulcanami" ]; then
    if command_exists helm; then
        if helm lint helm/vulcanami > /dev/null 2>&1; then
            print_success "Helm chart passes lint"
        else
            print_error "Helm chart has lint errors"
        fi
    else
        print_success "helm/vulcanami exists (helm not available for linting)"
    fi
else
    print_warning "helm/vulcanami directory not found"
fi

################################################################################
# 8. Entrypoint Script Validation
################################################################################
print_header "8. Validating Entrypoint Script"

if [ -f "entrypoint.sh" ]; then
    print_success "entrypoint.sh exists"
    
    # Check if it's executable
    if [ -x "entrypoint.sh" ]; then
        print_success "entrypoint.sh is executable"
    else
        print_warning "entrypoint.sh is not executable"
    fi
    
    # Test JWT validation (without secret)
    if bash entrypoint.sh echo "test" 2>&1 | grep -q "No valid JWT secret"; then
        print_success "entrypoint.sh properly validates JWT secrets"
    else
        print_warning "entrypoint.sh JWT validation may not be working"
    fi
else
    print_error "entrypoint.sh not found"
fi

################################################################################
# 9. Security Configuration Check
################################################################################
print_header "9. Checking Security Configuration"

# Check .gitignore
if [ -f ".gitignore" ]; then
    if grep -q "^\.env$" .gitignore; then
        print_success ".gitignore properly excludes .env files"
    else
        print_error ".gitignore does not exclude .env files"
    fi
    
    if grep -q "^\*\.key$" .gitignore || grep -q "^\*\.pem$" .gitignore; then
        print_success ".gitignore excludes key/certificate files"
    else
        print_warning ".gitignore may not exclude all sensitive files"
    fi
else
    print_error ".gitignore not found"
fi

# Check for hardcoded secrets in files (basic check)
echo "Checking for potential hardcoded secrets..."
if grep -rE "(password|secret|key).*=.*['\"].*['\"]" \
    --include="*.yml" --include="*.yaml" --include="*.py" \
    --exclude-dir=".git" --exclude-dir="venv" --exclude-dir=".venv" \
    . 2>/dev/null | grep -vE "(# |## |password_hash|secret_key_base|example|TODO|NOTE)" | head -5; then
    print_warning "Potential hardcoded secrets found (review above lines)"
else
    print_success "No obvious hardcoded secrets found"
fi

################################################################################
# 10. Reproducibility Checks
################################################################################
print_header "10. Checking Reproducibility Configuration"

# Check if Python version is pinned
if grep -q "python-version: '3.11'" .github/workflows/*.yml 2>/dev/null; then
    print_success "GitHub Actions use pinned Python version"
else
    print_warning "GitHub Actions may not use pinned Python version"
fi

# Check if Docker base images use tags
if grep -E "FROM python:3\\.11" Dockerfile > /dev/null 2>&1; then
    print_success "Dockerfile uses specific Python version"
else
    print_warning "Dockerfile may use 'latest' tag (not reproducible)"
fi

# Check documentation
if [ -f "REPRODUCIBLE_BUILDS.md" ]; then
    print_success "REPRODUCIBLE_BUILDS.md documentation exists"
else
    print_warning "REPRODUCIBLE_BUILDS.md documentation not found"
fi

################################################################################
# Summary
################################################################################
print_header "Validation Summary"

echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the errors above.${NC}"
    exit 1
fi
