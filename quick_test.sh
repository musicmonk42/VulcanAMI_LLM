#!/bin/bash
################################################################################
# Quick Test Runner - Run specific test categories quickly
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}VulcanAMI Quick Test Runner${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all              Run all tests (same as test_full_cicd.sh)"
    echo "  docker           Run Docker-related tests only"
    echo "  k8s              Run Kubernetes tests only"
    echo "  security         Run security tests only"
    echo "  dependencies     Run dependency tests only"
    echo "  workflows        Run GitHub Actions workflow tests only"
    echo "  pytest           Run pytest test suite"
    echo "  quick            Run quick validation (no slow tests)"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker        # Test Docker configurations and builds"
    echo "  $0 quick         # Quick validation before commit"
    echo "  $0 all           # Full comprehensive test suite"
}

test_docker() {
    echo -e "${BLUE}Running Docker Tests${NC}"
    
    # Check Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Docker not installed${NC}"
        exit 1
    fi
    
    # Validate Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}Dockerfile not found${NC}"
        exit 1
    fi
    
    echo "✓ Docker installed"
    echo "✓ Dockerfile found"
    
    # Check Dockerfile security features
    if grep -q "USER graphix\|USER 1001" Dockerfile; then
        echo "✓ Dockerfile uses non-root user"
    else
        echo -e "${RED}✗ Dockerfile missing non-root user${NC}"
        exit 1
    fi
    
    if grep -q "HEALTHCHECK" Dockerfile; then
        echo "✓ Dockerfile has HEALTHCHECK"
    else
        echo -e "${YELLOW}⚠ Dockerfile missing HEALTHCHECK${NC}"
    fi
    
    # Validate docker-compose files
    if docker compose -f docker-compose.dev.yml config >/dev/null 2>&1; then
        echo "✓ docker-compose.dev.yml is valid"
    else
        echo -e "${RED}✗ docker-compose.dev.yml validation failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All Docker tests passed!${NC}"
}

test_k8s() {
    echo -e "${BLUE}Running Kubernetes Tests${NC}"
    
    if [ ! -d "k8s" ]; then
        echo -e "${YELLOW}k8s directory not found (optional)${NC}"
        exit 0
    fi
    
    # Validate YAML syntax
    K8S_ERRORS=0
    for manifest in $(find k8s -name "*.yaml" -type f 2>/dev/null); do
        if ! python3 -c "import yaml; list(yaml.safe_load_all(open('$manifest')))" 2>/dev/null; then
            echo -e "${RED}✗ Invalid YAML: $manifest${NC}"
            ((K8S_ERRORS++))
        fi
    done
    
    if [ $K8S_ERRORS -eq 0 ]; then
        echo "✓ All Kubernetes manifests are valid YAML"
    else
        echo -e "${RED}Found $K8S_ERRORS invalid manifests${NC}"
        exit 1
    fi
    
    # Validate Helm chart if available
    if [ -d "helm" ] && command -v helm >/dev/null 2>&1; then
        CHART_DIRS=$(find helm -name "Chart.yaml" -type f 2>/dev/null)
        if [ -n "$CHART_DIRS" ]; then
            for chart_file in $CHART_DIRS; do
                CHART_DIR=$(dirname "$chart_file")
                if helm lint "$CHART_DIR" >/dev/null 2>&1; then
                    echo "✓ Helm chart $CHART_DIR is valid"
                else
                    echo -e "${RED}✗ Helm chart $CHART_DIR failed lint${NC}"
                    exit 1
                fi
            done
        fi
    fi
    
    echo -e "${GREEN}All Kubernetes tests passed!${NC}"
}

test_security() {
    echo -e "${BLUE}Running Security Tests${NC}"
    
    # Check no .env files committed
    if find . -name ".env" -not -path "*/\.*" -not -name "*.example" 2>/dev/null | grep -q ".env"; then
        echo -e "${RED}✗ Found committed .env files${NC}"
        exit 1
    fi
    echo "✓ No .env files committed"
    
    # Check no private keys
    if find . -name "*.pem" -o -name "*.key" -not -path "*/\.*" 2>/dev/null | grep -q -E "\.(pem|key)$"; then
        echo -e "${RED}✗ Found committed private keys${NC}"
        exit 1
    fi
    echo "✓ No private keys committed"
    
    # Check .gitignore
    if [ -f ".gitignore" ]; then
        if grep -q ".env" .gitignore && grep -q "*.key" .gitignore; then
            echo "✓ .gitignore properly configured"
        else
            echo -e "${YELLOW}⚠ .gitignore missing critical exclusions${NC}"
        fi
    else
        echo -e "${RED}✗ .gitignore not found${NC}"
        exit 1
    fi
    
    # Check entrypoint validates secrets
    if [ -f "entrypoint.sh" ]; then
        if grep -q "JWT_SECRET\|JWT_SECRET_KEY" entrypoint.sh; then
            echo "✓ entrypoint.sh validates JWT secrets"
        else
            echo -e "${YELLOW}⚠ entrypoint.sh missing JWT validation${NC}"
        fi
    fi
    
    echo -e "${GREEN}All security tests passed!${NC}"
}

test_dependencies() {
    echo -e "${BLUE}Running Dependency Tests${NC}"
    
    # Check requirements.txt
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}✗ requirements.txt not found${NC}"
        exit 1
    fi
    echo "✓ requirements.txt found"
    
    # Check requirements-hashed.txt
    if [ ! -f "requirements-hashed.txt" ]; then
        echo -e "${YELLOW}⚠ requirements-hashed.txt not found (needed for reproducibility)${NC}"
    else
        if grep -q "sha256:" requirements-hashed.txt; then
            echo "✓ requirements-hashed.txt has SHA256 hashes"
        else
            echo -e "${RED}✗ requirements-hashed.txt missing hashes${NC}"
            exit 1
        fi
    fi
    
    # Check for unpinned dependencies
    UNPINNED=$(grep -v "^#" requirements.txt | grep -v "^$" | grep -v "==" | grep -v ">=" | grep -v "^-" | grep -v "^https://" || true)
    if [ -z "$UNPINNED" ]; then
        echo "✓ All dependencies are pinned"
    else
        echo -e "${RED}✗ Found unpinned dependencies${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All dependency tests passed!${NC}"
}

test_workflows() {
    echo -e "${BLUE}Running GitHub Actions Workflow Tests${NC}"
    
    if [ ! -d ".github/workflows" ]; then
        echo -e "${RED}✗ .github/workflows directory not found${NC}"
        exit 1
    fi
    
    ERRORS=0
    for workflow in .github/workflows/*.yml; do
        if [ -f "$workflow" ]; then
            if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
                echo "✓ $(basename $workflow) is valid"
            else
                echo -e "${RED}✗ $(basename $workflow) has invalid YAML${NC}"
                ((ERRORS++))
            fi
        fi
    done
    
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}All workflow tests passed!${NC}"
    else
        exit 1
    fi
}

test_pytest() {
    echo -e "${BLUE}Running pytest Test Suite${NC}"
    
    # Check if pytest is available
    if ! python3 -c "import pytest" 2>/dev/null; then
        echo "Installing pytest..."
        pip install pytest pytest-cov pytest-timeout pyyaml python-dotenv --quiet
    fi
    
    # Run pytest
    if [ -f "tests/test_cicd_reproducibility.py" ]; then
        python3 -m pytest tests/test_cicd_reproducibility.py -v --tb=short -m "not slow"
    else
        echo -e "${RED}✗ tests/test_cicd_reproducibility.py not found${NC}"
        exit 1
    fi
}

test_quick() {
    echo -e "${BLUE}Running Quick Validation${NC}"
    
    # Essential checks only
    ERRORS=0
    
    # Check key files exist
    for file in Dockerfile requirements.txt Makefile README.md; do
        if [ -f "$file" ]; then
            echo "✓ $file found"
        else
            echo -e "${RED}✗ $file not found${NC}"
            ((ERRORS++))
        fi
    done
    
    # Quick Docker validation
    if docker compose -f docker-compose.dev.yml config >/dev/null 2>&1; then
        echo "✓ docker-compose.dev.yml valid"
    else
        echo -e "${RED}✗ docker-compose.dev.yml invalid${NC}"
        ((ERRORS++))
    fi
    
    # Quick security check
    if find . -name ".env" -not -path "*/\.*" -not -name "*.example" 2>/dev/null | grep -q ".env"; then
        echo -e "${RED}✗ .env files committed${NC}"
        ((ERRORS++))
    else
        echo "✓ No .env files committed"
    fi
    
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}Quick validation passed!${NC}"
    else
        echo -e "${RED}Quick validation failed!${NC}"
        exit 1
    fi
}

test_all() {
    echo -e "${BLUE}Running Full Test Suite${NC}"
    ./test_full_cicd.sh
}

# Main script
case "${1:-help}" in
    all)
        test_all
        ;;
    docker)
        test_docker
        ;;
    k8s)
        test_k8s
        ;;
    security)
        test_security
        ;;
    dependencies)
        test_dependencies
        ;;
    workflows)
        test_workflows
        ;;
    pytest)
        test_pytest
        ;;
    quick)
        test_quick
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
