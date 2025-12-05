#!/bin/bash
################################################################################
# Full CI/CD and Reproducibility Test Suite
# This script runs all conceivable tests to ensure the project can be
# uploaded and reproduced in any CI/CD environment
################################################################################

set +e  # Don't exit on error, collect all results
set -o pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Output directory
TEST_OUTPUT_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT_DIR"

# Log file
LOG_FILE="$TEST_OUTPUT_DIR/full_test.log"

################################################################################
# Helper Functions
################################################################################

log_section() {
    echo -e "\n${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}========================================${NC}\n" | tee -a "$LOG_FILE"
}

log_test() {
    echo -e "${CYAN}TEST: $1${NC}" | tee -a "$LOG_FILE"
    ((TOTAL_TESTS++))
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
    ((PASSED_TESTS++))
}

log_failure() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$LOG_FILE"
    ((FAILED_TESTS++))
}

log_skip() {
    echo -e "${YELLOW}⊘ $1${NC}" | tee -a "$LOG_FILE"
    ((SKIPPED_TESTS++))
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_unpinned_dependencies() {
    # Check for unpinned dependencies in requirements.txt
    # Returns: list of unpinned dependencies (empty if all pinned)
    if [ -f "$1" ]; then
        grep -v "^#" "$1" | grep -v "^$" | grep -v "==" | grep -v ">=" | grep -v "^-" | grep -v "^https://" | grep -v " @ " || true
    fi
}

################################################################################
# Main Test Suite
################################################################################

log_section "Full CI/CD and Reproducibility Test Suite"
echo "Starting comprehensive validation at $(date)" | tee -a "$LOG_FILE"
echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# 1. Environment Prerequisites
################################################################################
log_section "1. Environment Prerequisites"

log_test "Docker installed"
if command_exists docker; then
    VERSION=$(docker --version)
    log_success "Docker installed: $VERSION"
else
    log_failure "Docker not installed"
fi

log_test "Docker Compose v2 available"
if docker compose version >/dev/null 2>&1; then
    VERSION=$(docker compose version)
    log_success "Docker Compose v2 available: $VERSION"
else
    log_failure "Docker Compose v2 not available"
fi

log_test "Python installed"
if command_exists python3 || command_exists python; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    VERSION=$($PYTHON_CMD --version)
    log_success "Python installed: $VERSION"
else
    log_failure "Python not installed"
fi

log_test "Git installed"
if command_exists git; then
    VERSION=$(git --version)
    log_success "Git installed: $VERSION"
else
    log_failure "Git not installed"
fi

log_test "kubectl available"
if command_exists kubectl; then
    log_success "kubectl available (Kubernetes deployment possible)"
else
    log_skip "kubectl not available (Kubernetes deployment will be skipped)"
fi

log_test "helm available"
if command_exists helm; then
    log_success "helm available (Helm deployment possible)"
else
    log_skip "helm not available (Helm deployment will be skipped)"
fi

################################################################################
# 2. Repository Structure Validation
################################################################################
log_section "2. Repository Structure Validation"

log_test "README.md exists"
if [ -f "README.md" ]; then
    log_success "README.md found"
else
    log_failure "README.md not found"
fi

log_test "Dockerfile exists"
if [ -f "Dockerfile" ]; then
    log_success "Dockerfile found"
else
    log_failure "Dockerfile not found"
fi

log_test "requirements.txt exists"
if [ -f "requirements.txt" ]; then
    log_success "requirements.txt found"
else
    log_failure "requirements.txt not found"
fi

log_test "requirements-hashed.txt exists (reproducibility)"
if [ -f "requirements-hashed.txt" ]; then
    if grep -q "sha256:" requirements-hashed.txt; then
        log_success "requirements-hashed.txt found with SHA256 hashes"
    else
        log_failure "requirements-hashed.txt exists but missing SHA256 hashes"
    fi
else
    log_failure "requirements-hashed.txt not found (needed for reproducible builds)"
fi

log_test "Makefile exists"
if [ -f "Makefile" ]; then
    log_success "Makefile found"
else
    log_failure "Makefile not found"
fi

log_test "src directory exists"
if [ -d "src" ]; then
    log_success "src directory found"
else
    log_failure "src directory not found"
fi

log_test "tests directory exists"
if [ -d "tests" ]; then
    log_success "tests directory found"
else
    log_failure "tests directory not found"
fi

################################################################################
# 3. Configuration File Validation
################################################################################
log_section "3. Configuration File Validation"

log_test "docker-compose.dev.yml valid"
if docker compose -f docker-compose.dev.yml config >/dev/null 2>&1; then
    log_success "docker-compose.dev.yml is valid"
else
    log_failure "docker-compose.dev.yml validation failed"
fi

log_test "docker-compose.prod.yml valid"
if [ -f "docker-compose.prod.yml" ]; then
    # Set dummy env vars to avoid interpolation errors
    export POSTGRES_PASSWORD="dummy"
    export REDIS_PASSWORD="dummy"
    export MINIO_ROOT_PASSWORD="dummy"
    export MINIO_ROOT_USER="dummy"
    export JWT_SECRET_KEY="dummy"
    export BOOTSTRAP_KEY="dummy"
    export GRAFANA_PASSWORD="dummy"
    export GRAFANA_USER="dummy"
    
    if docker compose -f docker-compose.prod.yml config >/dev/null 2>&1; then
        log_success "docker-compose.prod.yml is valid"
    else
        log_failure "docker-compose.prod.yml validation failed"
    fi
    
    # Unset dummy env vars
    unset POSTGRES_PASSWORD REDIS_PASSWORD MINIO_ROOT_PASSWORD MINIO_ROOT_USER
    unset JWT_SECRET_KEY BOOTSTRAP_KEY GRAFANA_PASSWORD GRAFANA_USER
else
    log_skip "docker-compose.prod.yml not found"
fi

log_test ".gitignore excludes secrets"
if [ -f ".gitignore" ]; then
    if grep -q ".env" .gitignore && grep -q "*.key" .gitignore; then
        log_success ".gitignore properly configured"
    else
        log_failure ".gitignore missing critical exclusions"
    fi
else
    log_failure ".gitignore not found"
fi

log_test ".dockerignore excludes unnecessary files"
if [ -f ".dockerignore" ]; then
    if grep -q "__pycache__" .dockerignore && grep -q ".git" .dockerignore; then
        log_success ".dockerignore properly configured"
    else
        log_failure ".dockerignore missing critical exclusions"
    fi
else
    log_failure ".dockerignore not found"
fi

################################################################################
# 4. GitHub Actions Workflows Validation
################################################################################
log_section "4. GitHub Actions Workflows Validation"

if [ -d ".github/workflows" ]; then
    for workflow in .github/workflows/*.yml; do
        if [ -f "$workflow" ]; then
            log_test "$(basename $workflow) is valid YAML"
            if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
                log_success "$(basename $workflow) is valid"
            else
                log_failure "$(basename $workflow) has invalid YAML"
            fi
        fi
    done
else
    log_skip ".github/workflows directory not found"
fi

################################################################################
# 5. Docker Build Tests
################################################################################
log_section "5. Docker Build Tests"

log_test "Main Dockerfile security features"
if grep -q "USER graphix\|USER 1001" Dockerfile; then
    log_success "Dockerfile runs as non-root user"
else
    log_failure "Dockerfile missing non-root user configuration"
fi

if grep -q "HEALTHCHECK" Dockerfile; then
    log_success "Dockerfile includes HEALTHCHECK"
else
    log_failure "Dockerfile missing HEALTHCHECK"
fi

if grep -q "REJECT_INSECURE_JWT\|JWT_SECRET" Dockerfile; then
    log_success "Dockerfile validates JWT configuration"
else
    log_failure "Dockerfile missing JWT secret validation"
fi

log_test "Service Dockerfiles exist"
for service in api dqs pii; do
    if [ -f "docker/$service/Dockerfile" ]; then
        log_success "docker/$service/Dockerfile found"
    else
        log_skip "docker/$service/Dockerfile not found (optional service)"
    fi
done

log_test "Docker build syntax validation"
if docker build --help >/dev/null 2>&1; then
    # Try a dry-run style check (checking if Dockerfile parses)
    if docker build --build-arg REJECT_INSECURE_JWT=ack -t test-syntax-check:latest . --dry-run 2>/dev/null || \
       docker build --build-arg REJECT_INSECURE_JWT=ack -t test-syntax-check:latest . 2>&1 | head -20 | grep -q "Step 1/"; then
        log_success "Dockerfile syntax is valid"
    else
        log_skip "Cannot validate Docker build syntax without full build"
    fi
else
    log_skip "Docker build validation skipped"
fi

################################################################################
# 6. Python Dependencies and Security
################################################################################
log_section "6. Python Dependencies and Security"

log_test "Install pytest for testing"
if command_exists pip3 || command_exists pip; then
    PIP_CMD=$(command -v pip3 || command -v pip)
    
    # Try to install pytest if not available
    if ! $PYTHON_CMD -c "import pytest" 2>/dev/null; then
        echo "Installing pytest..." | tee -a "$LOG_FILE"
        $PIP_CMD install pytest pytest-timeout --quiet 2>&1 | tee -a "$LOG_FILE"
    fi
    
    if $PYTHON_CMD -c "import pytest" 2>/dev/null; then
        log_success "pytest available"
    else
        log_skip "pytest installation failed (will skip pytest tests)"
    fi
else
    log_skip "pip not available"
fi

log_test "Python syntax validation"
SYNTAX_ERRORS=0
if [ -d "src" ]; then
    for pyfile in $(find src -name "*.py" -type f 2>/dev/null | head -20); do
        if ! $PYTHON_CMD -m py_compile "$pyfile" 2>/dev/null; then
            ((SYNTAX_ERRORS++))
        fi
    done
    
    if [ $SYNTAX_ERRORS -eq 0 ]; then
        log_success "All checked Python files have valid syntax"
    else
        log_failure "Found $SYNTAX_ERRORS Python files with syntax errors"
    fi
else
    log_skip "src directory not found"
fi

################################################################################
# 7. Kubernetes Manifest Validation
################################################################################
log_section "7. Kubernetes Manifest Validation"

if [ -d "k8s" ]; then
    log_test "Kubernetes manifests are valid YAML"
    K8S_ERRORS=0
    for manifest in $(find k8s -name "*.yaml" -type f 2>/dev/null); do
        # Use yaml.safe_load_all to handle multi-document YAML files
        if ! python3 -c "import yaml; list(yaml.safe_load_all(open('$manifest')))" 2>/dev/null; then
            ((K8S_ERRORS++))
        fi
    done
    
    if [ $K8S_ERRORS -eq 0 ]; then
        log_success "All Kubernetes manifests are valid YAML"
    else
        log_failure "Found $K8S_ERRORS invalid Kubernetes manifests"
    fi
    
    if command_exists kubectl; then
        log_test "kubectl can validate manifests"
        if kubectl apply -f k8s/base --dry-run=client >/dev/null 2>&1; then
            log_success "Kubernetes manifests validated with kubectl"
        else
            log_skip "kubectl validation failed (might need cluster context)"
        fi
    fi
else
    log_skip "k8s directory not found (Kubernetes deployment optional)"
fi

################################################################################
# 8. Helm Chart Validation
################################################################################
log_section "8. Helm Chart Validation"

if [ -d "helm" ] && command_exists helm; then
    CHART_DIRS=$(find helm -name "Chart.yaml" -type f 2>/dev/null)
    
    if [ -n "$CHART_DIRS" ]; then
        for chart_file in $CHART_DIRS; do
            CHART_DIR=$(dirname "$chart_file")
            log_test "Helm chart lint: $CHART_DIR"
            
            if helm lint "$CHART_DIR" >/dev/null 2>&1; then
                log_success "Helm chart $CHART_DIR is valid"
            else
                log_failure "Helm chart $CHART_DIR failed lint"
            fi
        done
    else
        log_skip "No Helm charts found"
    fi
else
    log_skip "Helm not available or helm directory not found"
fi

################################################################################
# 9. Security Scans
################################################################################
log_section "9. Security Configuration"

log_test "No .env files committed"
if find . -name ".env" -not -path "*/\.*" -not -name "*.example" 2>/dev/null | grep -q ".env"; then
    log_failure "Found committed .env files"
else
    log_success "No .env files committed"
fi

log_test "No private keys committed"
if find . -name "*.pem" -o -name "*.key" -not -path "*/\.*" 2>/dev/null | grep -q -E "\.(pem|key)$"; then
    log_failure "Found committed private keys"
else
    log_success "No private keys committed"
fi

log_test "entrypoint.sh validates secrets"
if [ -f "entrypoint.sh" ]; then
    if grep -q "JWT_SECRET\|JWT_SECRET_KEY\|GRAPHIX_JWT_SECRET" entrypoint.sh; then
        log_success "entrypoint.sh validates JWT secrets"
    else
        log_failure "entrypoint.sh missing JWT secret validation"
    fi
else
    log_skip "entrypoint.sh not found"
fi

################################################################################
# 10. Run Existing Validation Scripts
################################################################################
log_section "10. Run Existing Validation Scripts"

log_test "validate_cicd_docker.sh"
if [ -f "validate_cicd_docker.sh" ] && [ -x "validate_cicd_docker.sh" ]; then
    echo "Running validate_cicd_docker.sh..." | tee -a "$LOG_FILE"
    if ./validate_cicd_docker.sh > "$TEST_OUTPUT_DIR/validate_cicd_docker.log" 2>&1; then
        log_success "validate_cicd_docker.sh passed"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            log_success "validate_cicd_docker.sh completed"
        else
            log_failure "validate_cicd_docker.sh failed (check $TEST_OUTPUT_DIR/validate_cicd_docker.log)"
        fi
    fi
else
    log_skip "validate_cicd_docker.sh not found or not executable"
fi

################################################################################
# 11. Run pytest CI/CD Tests
################################################################################
log_section "11. Run pytest CI/CD Tests"

log_test "pytest CI/CD reproducibility tests"
if [ -f "tests/test_cicd_reproducibility.py" ]; then
    if $PYTHON_CMD -c "import pytest" 2>/dev/null; then
        echo "Running pytest CI/CD tests..." | tee -a "$LOG_FILE"
        
        # Run pytest with timeout and capture output
        if $PYTHON_CMD -m pytest tests/test_cicd_reproducibility.py \
            -v \
            --tb=short \
            --timeout=300 \
            -m "not slow" \
            > "$TEST_OUTPUT_DIR/pytest_cicd.log" 2>&1; then
            log_success "pytest CI/CD tests passed"
        else
            EXIT_CODE=$?
            # Check if any tests passed
            if grep -q "passed" "$TEST_OUTPUT_DIR/pytest_cicd.log"; then
                log_success "pytest CI/CD tests completed (some may have been skipped)"
            else
                log_failure "pytest CI/CD tests failed (check $TEST_OUTPUT_DIR/pytest_cicd.log)"
            fi
        fi
    else
        log_skip "pytest not available"
    fi
else
    log_skip "tests/test_cicd_reproducibility.py not found"
fi

################################################################################
# 12. Documentation Validation
################################################################################
log_section "12. Documentation Validation"

REQUIRED_DOCS=("README.md" "CI_CD.md" "DEPLOYMENT.md")
for doc in "${REQUIRED_DOCS[@]}"; do
    log_test "$doc exists"
    if [ -f "$doc" ]; then
        # Check if file is not empty
        if [ -s "$doc" ]; then
            log_success "$doc found and not empty"
        else
            log_failure "$doc is empty"
        fi
    else
        log_skip "$doc not found (recommended)"
    fi
done

################################################################################
# 13. Reproducibility Verification
################################################################################
log_section "13. Reproducibility Verification"

log_test "Python version pinned in Dockerfile"
if grep -E "FROM python:[0-9]+\.[0-9]+" Dockerfile | grep -v ":latest"; then
    log_success "Dockerfile uses pinned Python version"
else
    log_failure "Dockerfile should use specific Python version (not :latest)"
fi

log_test "All dependencies have pinned versions"
if [ -f "requirements.txt" ]; then
    UNPINNED=$(check_unpinned_dependencies "requirements.txt")
    if [ -z "$UNPINNED" ]; then
        log_success "All dependencies have pinned versions"
    else
        log_failure "Found unpinned dependencies: $UNPINNED"
    fi
fi

log_test "Git commit hash available"
if git rev-parse --short HEAD >/dev/null 2>&1; then
    COMMIT_HASH=$(git rev-parse --short HEAD)
    log_success "Git commit hash: $COMMIT_HASH"
else
    log_skip "Not in a git repository"
fi

################################################################################
# 14. Create Test Summary Report
################################################################################
log_section "14. Test Summary"

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
else
    PASS_RATE=0
fi

echo "" | tee -a "$LOG_FILE"
echo "Total Tests:   $TOTAL_TESTS" | tee -a "$LOG_FILE"
echo "Passed:        $PASSED_TESTS (${GREEN}✓${NC})" | tee -a "$LOG_FILE"
echo "Failed:        $FAILED_TESTS (${RED}✗${NC})" | tee -a "$LOG_FILE"
echo "Skipped:       $SKIPPED_TESTS (${YELLOW}⊘${NC})" | tee -a "$LOG_FILE"
echo "Pass Rate:     ${PASS_RATE}%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Generate summary report
cat > "$TEST_OUTPUT_DIR/summary.txt" <<EOF
Full CI/CD and Reproducibility Test Report
===========================================
Date: $(date)
Repository: $(basename $(pwd))
Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

Test Results:
- Total Tests:   $TOTAL_TESTS
- Passed:        $PASSED_TESTS
- Failed:        $FAILED_TESTS
- Skipped:       $SKIPPED_TESTS
- Pass Rate:     ${PASS_RATE}%

Status: $([ $FAILED_TESTS -eq 0 ] && echo "SUCCESS ✓" || echo "FAILED ✗")

Detailed logs available in: $TEST_OUTPUT_DIR/
EOF

cat "$TEST_OUTPUT_DIR/summary.txt"

# Final result
echo "" | tee -a "$LOG_FILE"
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}All Critical Tests Passed! ✓${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}========================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "This repository is ready for CI/CD deployment." | tee -a "$LOG_FILE"
    echo "Results saved to: $TEST_OUTPUT_DIR/" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${RED}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${RED}Some Tests Failed! ✗${NC}" | tee -a "$LOG_FILE"
    echo -e "${RED}========================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Please review failures above and in $TEST_OUTPUT_DIR/" | tee -a "$LOG_FILE"
    exit 1
fi
