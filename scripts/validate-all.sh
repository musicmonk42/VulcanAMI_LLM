#!/bin/bash
################################################################################
# VulcanAMI Validation Script
# 
# Validates Docker, Kubernetes, and Helm configurations
# Safe to run - does not build or deploy anything
################################################################################

# Don't exit on errors - we want to collect all validation results
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

echo "================================================================================"
echo "  VulcanAMI Configuration Validation"
echo "================================================================================"
echo ""

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((CHECKS_WARNING++))
}

check_info() {
    echo -e "  ℹ $1"
}

# Change to repo root
cd "$ROOT_DIR"

################################################################################
# 1. Prerequisites Check
################################################################################
print_header "1. Checking Prerequisites"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    check_pass "Docker installed (version $DOCKER_VERSION)"
else
    check_fail "Docker not installed - https://docs.docker.com/get-docker/"
fi

if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version 2>&1 | head -n1 | awk '{print $NF}' || echo "installed")
    check_pass "Docker Compose installed (version $COMPOSE_VERSION)"
else
    check_fail "Docker Compose not installed"
fi

if command -v kubectl &> /dev/null; then
    KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | cut -d' ' -f3 || echo "installed")
    check_pass "kubectl installed (version $KUBECTL_VERSION)"
else
    check_warn "kubectl not installed - needed for Kubernetes deployments"
    check_info "Install: https://kubernetes.io/docs/tasks/tools/"
fi

if command -v helm &> /dev/null; then
    HELM_VERSION=$(helm version --short 2>/dev/null | cut -d'v' -f2 | cut -d'+' -f1)
    check_pass "Helm installed (version $HELM_VERSION)"
else
    check_warn "Helm not installed - needed for Helm deployments"
    check_info "Install: https://helm.sh/docs/intro/install/"
fi

################################################################################
# 2. Docker Configuration Validation
################################################################################
print_header "2. Validating Docker Configurations"

# Check main Dockerfile
if [ -f "Dockerfile" ]; then
    check_pass "Main Dockerfile exists"
    
    # Check for security features
    if grep -q "REJECT_INSECURE_JWT" Dockerfile; then
        check_pass "Security: JWT validation build arg present"
    else
        check_warn "Security: JWT validation build arg missing"
    fi
    
    if grep -q "USER" Dockerfile; then
        check_pass "Security: Non-root user configured"
    else
        check_warn "Security: Running as root (not recommended)"
    fi
    
    if grep -q "HEALTHCHECK" Dockerfile; then
        check_pass "Healthcheck configured"
    else
        check_warn "No healthcheck configured"
    fi
else
    check_fail "Main Dockerfile not found"
fi

# Check service Dockerfiles
for dockerfile in docker/api/Dockerfile docker/dqs/Dockerfile docker/pii/Dockerfile; do
    if [ -f "$dockerfile" ]; then
        SERVICE=$(basename $(dirname $dockerfile))
        check_pass "Service Dockerfile: $SERVICE"
    else
        check_warn "Service Dockerfile missing: $dockerfile"
    fi
done

################################################################################
# 3. Docker Compose Validation
################################################################################
print_header "3. Validating Docker Compose Files"

# Validate docker-compose.dev.yml
if [ -f "docker-compose.dev.yml" ]; then
    check_pass "docker-compose.dev.yml exists"
    
    if docker compose -f docker-compose.dev.yml config > /dev/null 2>&1; then
        check_pass "docker-compose.dev.yml syntax is valid"
        
        # Count services
        SERVICE_COUNT=$(docker compose -f docker-compose.dev.yml config --services 2>/dev/null | wc -l)
        check_info "Found $SERVICE_COUNT services defined"
    else
        check_fail "docker-compose.dev.yml has syntax errors"
    fi
else
    check_fail "docker-compose.dev.yml not found"
fi

# Validate docker-compose.prod.yml
if [ -f "docker-compose.prod.yml" ]; then
    check_pass "docker-compose.prod.yml exists"
    
    # Production compose requires env vars, so we can't fully validate
    # Just check file exists and basic structure
    if grep -q "services:" docker-compose.prod.yml; then
        check_pass "docker-compose.prod.yml has services defined"
    else
        check_fail "docker-compose.prod.yml missing services section"
    fi
    
    # Check for required env vars documentation
    if grep -q "required" docker-compose.prod.yml; then
        check_pass "Required environment variables documented"
    else
        check_warn "Required environment variables not clearly marked"
    fi
else
    check_fail "docker-compose.prod.yml not found"
fi

################################################################################
# 4. Kubernetes Configuration Validation
################################################################################
print_header "4. Validating Kubernetes Configurations"

if [ -d "k8s" ]; then
    check_pass "Kubernetes directory exists"
    
    # Check base directory
    if [ -d "k8s/base" ]; then
        check_pass "k8s/base directory exists"
        
        # Check for kustomization.yaml
        if [ -f "k8s/base/kustomization.yaml" ]; then
            check_pass "kustomization.yaml exists"
            
            # Validate with kustomize (if kubectl available)
            if command -v kubectl &> /dev/null; then
                if kubectl kustomize k8s/base > /dev/null 2>&1; then
                    check_pass "Kustomize base configuration is valid"
                else
                    check_fail "Kustomize base configuration has errors"
                fi
            fi
        else
            check_warn "kustomization.yaml not found in k8s/base"
        fi
        
        # Check for essential manifests
        for manifest in namespace.yaml deployment.yaml service.yaml; do
            FOUND=false
            for file in k8s/base/*.yaml; do
                if [ -f "$file" ] && grep -q "kind: $(echo $manifest | sed 's/.yaml//' | sed 's/-/ /' | awk '{for(i=1;i<=NF;i++){$i=toupper(substr($i,1,1)) substr($i,2)}}1' | sed 's/ //')" "$file" 2>/dev/null; then
                    FOUND=true
                    break
                fi
            done
            
            if [ "$FOUND" = true ]; then
                check_pass "Found manifest type: $(basename $manifest .yaml)"
            fi
        done
    else
        check_warn "k8s/base directory not found"
    fi
    
    # Check overlays
    if [ -d "k8s/overlays" ]; then
        check_pass "k8s/overlays directory exists"
        
        for overlay in development staging production; do
            if [ -d "k8s/overlays/$overlay" ]; then
                check_pass "Overlay configured: $overlay"
            fi
        done
    else
        check_warn "k8s/overlays directory not found"
    fi
else
    check_warn "Kubernetes directory not found (k8s/)"
fi

################################################################################
# 5. Helm Chart Validation
################################################################################
print_header "5. Validating Helm Chart"

if [ -d "helm/vulcanami" ]; then
    check_pass "Helm chart directory exists"
    
    # Check Chart.yaml
    if [ -f "helm/vulcanami/Chart.yaml" ]; then
        check_pass "Chart.yaml exists"
        
        # Extract version
        if grep -q "version:" helm/vulcanami/Chart.yaml; then
            CHART_VERSION=$(grep "^version:" helm/vulcanami/Chart.yaml | awk '{print $2}')
            check_info "Chart version: $CHART_VERSION"
        fi
    else
        check_fail "Chart.yaml not found"
    fi
    
    # Check values.yaml
    if [ -f "helm/vulcanami/values.yaml" ]; then
        check_pass "values.yaml exists"
    else
        check_fail "values.yaml not found"
    fi
    
    # Check templates directory
    if [ -d "helm/vulcanami/templates" ]; then
        check_pass "templates directory exists"
        
        TEMPLATE_COUNT=$(find helm/vulcanami/templates -name "*.yaml" -type f | wc -l)
        check_info "Found $TEMPLATE_COUNT template files"
    else
        check_fail "templates directory not found"
    fi
    
    # Run helm lint (if helm available)
    if command -v helm &> /dev/null; then
        check_info "Running helm lint..."
        
        # Lint with required values to avoid validation errors
        if helm lint helm/vulcanami \
            --set image.tag=test \
            --set secrets.jwtSecretKey=test \
            --set secrets.bootstrapKey=test \
            --set secrets.postgresPassword=test \
            --set secrets.redisPassword=test > /dev/null 2>&1; then
            check_pass "Helm chart passes lint checks"
        else
            # Check if it's just warnings
            LINT_OUTPUT=$(helm lint helm/vulcanami \
                --set image.tag=test \
                --set secrets.jwtSecretKey=test \
                --set secrets.bootstrapKey=test \
                --set secrets.postgresPassword=test \
                --set secrets.redisPassword=test 2>&1)
            
            if echo "$LINT_OUTPUT" | grep -q "0 chart(s) failed"; then
                check_pass "Helm chart passes lint (with warnings)"
            else
                check_fail "Helm chart has lint errors"
            fi
        fi
        
        # Test template rendering
        if helm template test helm/vulcanami \
            --set image.tag=test \
            --set secrets.jwtSecretKey=test \
            --set secrets.bootstrapKey=test \
            --set secrets.postgresPassword=test \
            --set secrets.redisPassword=test > /dev/null 2>&1; then
            check_pass "Helm templates render successfully"
        else
            check_fail "Helm template rendering failed"
        fi
    fi
else
    check_warn "Helm chart not found (helm/vulcanami/)"
fi

################################################################################
# 6. Documentation Check
################################################################################
print_header "6. Checking Documentation"

DOCS=(
    "README.md:Main documentation"
    "DEPLOYMENT.md:Deployment guide"
    "DOCKER_BUILD_GUIDE.md:Docker build guide"
    "NEW_ENGINEER_SETUP.md:New engineer setup"
    ".env.example:Environment example"
)

for doc in "${DOCS[@]}"; do
    FILE="${doc%%:*}"
    DESC="${doc##*:}"
    
    if [ -f "$FILE" ]; then
        check_pass "$DESC ($FILE)"
    else
        check_warn "$DESC not found ($FILE)"
    fi
done

################################################################################
# 7. Security Check
################################################################################
print_header "7. Security Configuration Check"

# Check .gitignore for secrets
if [ -f ".gitignore" ]; then
    if grep -q "\.env" .gitignore && grep -q "secrets" .gitignore; then
        check_pass "Secrets excluded from git (.gitignore)"
    else
        check_warn "Make sure .env and secrets are in .gitignore"
    fi
else
    check_fail ".gitignore not found"
fi

# Check for accidentally committed secrets
if [ -f ".env" ]; then
    check_warn ".env file exists - make sure it's not committed to git!"
else
    check_pass "No .env file in repo (good - use .env.example)"
fi

# Check for example env file
if [ -f ".env.example" ]; then
    check_pass ".env.example exists for reference"
    
    # Make sure it doesn't contain real secrets
    if grep -q "changeme\|example\|REPLACE_ME" .env.example; then
        check_pass ".env.example contains placeholder values"
    else
        check_warn ".env.example might contain real secrets!"
    fi
else
    check_warn ".env.example not found - create one for new engineers"
fi

################################################################################
# Summary
################################################################################
echo ""
echo "================================================================================"
echo "  Validation Summary"
echo "================================================================================"
echo ""
echo -e "  ${GREEN}✓ Passed:${NC}  $CHECKS_PASSED"
echo -e "  ${RED}✗ Failed:${NC}  $CHECKS_FAILED"
echo -e "  ${YELLOW}⚠ Warnings:${NC} $CHECKS_WARNING"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  All critical checks passed! ✓${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Read NEW_ENGINEER_SETUP.md for deployment instructions"
    echo "  2. Choose deployment method (Docker Compose, Kubernetes, or Helm)"
    echo "  3. Generate secrets and configure .env file"
    echo "  4. Deploy!"
    echo ""
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  Validation failed with $CHECKS_FAILED error(s) ✗${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Please fix the errors above before deploying."
    echo ""
    exit 1
fi
