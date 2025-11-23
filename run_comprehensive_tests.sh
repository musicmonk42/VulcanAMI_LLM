#!/bin/bash
###############################################################################
# Comprehensive CI/CD Testing Script for VulcanAMI_LLM
# This script runs all tests and validations to ensure reproducibility
###############################################################################

set +e  # Exit on error
set -o pipefail  # Fail on pipe errors

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Output directory for test results
TEST_OUTPUT_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VulcanAMI_LLM Comprehensive Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

###############################################################################
# Helper Functions
###############################################################################

log_section() {
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED_TESTS++))
}

log_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED_TESTS++))
}

log_skip() {
    echo -e "${YELLOW}⊘ $1${NC}"
    ((SKIPPED_TESTS++))
}

log_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local output_file="$TEST_OUTPUT_DIR/${test_name// /_}.log"
    
    ((TOTAL_TESTS++))
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    
    if eval "$test_command" > "$output_file" 2>&1; then
        log_success "$test_name"
        return 0
    else
        log_failure "$test_name (see $output_file)"
        return 1
    fi
}

###############################################################################
# Environment Setup Check
###############################################################################

log_section "1. Environment Validation"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION"
if [[ $PYTHON_VERSION == *"3.11"* ]] || [[ $PYTHON_VERSION == *"3.12"* ]]; then
    log_success "Python version check"
else
    log_failure "Python version check (expected 3.11 or 3.12)"
fi

# Check if pytest is installed
if python -m pytest --version > /dev/null 2>&1; then
    log_success "pytest installed"
else
    log_failure "pytest not installed"
    exit 1
fi

# Check disk space
AVAILABLE_SPACE=$(df -h / | awk 'NR==2 {print $4}')
echo "Available disk space: $AVAILABLE_SPACE"
log_info "Available disk space: $AVAILABLE_SPACE"

###############################################################################
# 2. Code Linting and Formatting
###############################################################################

log_section "2. Code Linting and Formatting"

# Black formatting check (if available)
if command -v black &> /dev/null; then
    run_test "Black formatting check" "black --check src/ tests/ --quiet" || true
else
    log_skip "Black not installed"
fi

# isort check (if available)
if command -v isort &> /dev/null; then
    run_test "isort import sorting check" "isort --check-only src/ tests/ --quiet" || true
else
    log_skip "isort not installed"
fi

# Flake8 linting (if available)
if command -v flake8 &> /dev/null; then
    run_test "Flake8 linting" "flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics" || true
else
    log_skip "Flake8 not installed"
fi

###############################################################################
# 3. Security Scanning
###############################################################################

log_section "3. Security Scanning"

# Bandit security check (if available)
if command -v bandit &> /dev/null; then
    run_test "Bandit security scan" "bandit -r src/ -ll -f json -o $TEST_OUTPUT_DIR/bandit_report.json" || true
else
    log_skip "Bandit not installed"
fi

# Check for hardcoded secrets
if grep -r "sk-[a-zA-Z0-9]\{32,\}" src/ tests/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__"; then
    log_failure "Found potential hardcoded API keys"
else
    log_success "No hardcoded API keys found"
fi

# Check for TODO/FIXME comments that might indicate bugs
TODO_COUNT=$(grep -r "TODO\|FIXME\|XXX\|HACK\|BUG" src/ --include="*.py" 2>/dev/null | wc -l || echo 0)
log_info "Found $TODO_COUNT TODO/FIXME comments in source code"

###############################################################################
# 4. Configuration Validation
###############################################################################

log_section "4. Configuration Validation"

# Check Dockerfile exists and is valid
if [ -f "Dockerfile" ]; then
    log_success "Dockerfile exists"
    
    # Check for security best practices in Dockerfile
    if grep -q "USER" Dockerfile && ! grep -q "USER root" Dockerfile; then
        log_success "Dockerfile uses non-root user"
    else
        log_failure "Dockerfile should use non-root user"
    fi
    
    if grep -q "REJECT_INSECURE_JWT" Dockerfile; then
        log_success "Dockerfile has JWT security check"
    else
        log_failure "Dockerfile missing JWT security check"
    fi
else
    log_failure "Dockerfile not found"
fi

# Check docker-compose files
if [ -f "docker-compose.dev.yml" ]; then
    log_success "docker-compose.dev.yml exists"
else
    log_failure "docker-compose.dev.yml not found"
fi

if [ -f "docker-compose.prod.yml" ]; then
    log_success "docker-compose.prod.yml exists"
else
    log_failure "docker-compose.prod.yml not found"
fi

# Check CI/CD workflows
if [ -d ".github/workflows" ]; then
    WORKFLOW_COUNT=$(ls -1 .github/workflows/*.yml 2>/dev/null | wc -l)
    log_success "Found $WORKFLOW_COUNT GitHub Actions workflows"
    
    # Check for required workflows
    for workflow in "ci.yml" "docker.yml" "security.yml"; do
        if [ -f ".github/workflows/$workflow" ]; then
            log_success "Workflow $workflow exists"
        else
            log_failure "Workflow $workflow not found"
        fi
    done
else
    log_failure "GitHub Actions workflows directory not found"
fi

# Check pytest configuration
if [ -f "pytest.ini" ]; then
    log_success "pytest.ini configuration exists"
else
    log_failure "pytest.ini not found"
fi

# Check requirements files
if [ -f "requirements.txt" ]; then
    REQ_COUNT=$(grep -c "^[a-zA-Z]" requirements.txt || echo 0)
    log_success "requirements.txt exists ($REQ_COUNT packages)"
else
    log_failure "requirements.txt not found"
fi

###############################################################################
# 5. Unit Tests
###############################################################################

log_section "5. Unit Tests Execution"

# Run pytest with coverage (limited to avoid memory/space issues)
log_info "Running unit tests (this may take several minutes)..."

# First, do a dry run to collect tests
TEST_COUNT=$(python -m pytest tests/ --collect-only -q 2>/dev/null | grep "test session starts" -A 100 | grep "collected" | awk '{print $2}' || echo "unknown")
log_info "Collected $TEST_COUNT tests"

# Run a subset of fast tests first
if python -m pytest tests/ -v --tb=short -x --maxfail=5 -k "not slow and not integration" \
    --timeout=60 \
    > "$TEST_OUTPUT_DIR/unit_tests.log" 2>&1; then
    log_success "Unit tests passed"
else
    log_failure "Unit tests failed (see $TEST_OUTPUT_DIR/unit_tests.log)"
fi

###############################################################################
# 6. Docker Build Validation
###############################################################################

log_section "6. Docker Build Validation"

# Check if Docker is available
if command -v docker &> /dev/null; then
    log_info "Docker is available - performing build validation"
    
    # Validate Dockerfile syntax
    if docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami-test:validation . \
        > "$TEST_OUTPUT_DIR/docker_build.log" 2>&1; then
        log_success "Docker image builds successfully"
        
        # Clean up test image
        docker rmi vulcanami-test:validation > /dev/null 2>&1 || true
    else
        log_failure "Docker build failed (see $TEST_OUTPUT_DIR/docker_build.log)"
    fi
else
    log_skip "Docker not available in environment"
fi

###############################################################################
# 7. Source Code Structure Validation
###############################################################################

log_section "7. Source Code Structure Validation"

# Check for required directories
REQUIRED_DIRS=("src" "tests" "configs" "docker")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        FILE_COUNT=$(find "$dir" -type f 2>/dev/null | wc -l)
        log_success "Directory $dir exists ($FILE_COUNT files)"
    else
        log_failure "Required directory $dir not found"
    fi
done

# Check Python package structure
if [ -f "src/__init__.py" ]; then
    log_success "src/ is a Python package"
else
    log_info "src/__init__.py not found (may be intentional)"
fi

# Count Python files
PY_FILES=$(find src/ -name "*.py" 2>/dev/null | wc -l)
TEST_FILES=$(find tests/ -name "test_*.py" 2>/dev/null | wc -l)
log_info "Found $PY_FILES Python source files"
log_info "Found $TEST_FILES test files"

# Check for __pycache__ pollution
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ $PYCACHE_COUNT -gt 0 ]; then
    log_info "Found $PYCACHE_COUNT __pycache__ directories (consider adding to .gitignore)"
fi

###############################################################################
# 8. Import and Dependency Checks
###############################################################################

log_section "8. Import and Dependency Validation"

# Test if main modules can be imported
log_info "Testing module imports..."

TEST_IMPORTS=(
    "src.agent_registry"
    "src.consensus_engine"
    "src.execution.execution_engine"
    "src.governance.governance_loop"
)

for module in "${TEST_IMPORTS[@]}"; do
    if python -c "import sys; sys.path.insert(0, 'src'); import ${module//src./}" 2>/dev/null; then
        log_success "Import $module"
    else
        log_failure "Failed to import $module"
    fi
done

###############################################################################
# 9. Git and Version Control
###############################################################################

log_section "9. Git and Version Control Validation"

# Check git status
if git status > /dev/null 2>&1; then
    log_success "Git repository initialized"
    
    # Check for uncommitted changes
    if [ -z "$(git status --porcelain)" ]; then
        log_success "Working directory is clean"
    else
        log_info "Working directory has changes"
    fi
    
    # Check current branch
    CURRENT_BRANCH=$(git branch --show-current)
    log_info "Current branch: $CURRENT_BRANCH"
    
    # Check for .gitignore
    if [ -f ".gitignore" ]; then
        log_success ".gitignore exists"
    else
        log_failure ".gitignore not found"
    fi
else
    log_failure "Not a git repository"
fi

###############################################################################
# 10. Documentation Validation
###############################################################################

log_section "10. Documentation Validation"

# Check for required documentation
REQUIRED_DOCS=("README.md" "CI_CD.md" "DEPLOYMENT.md")
for doc in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        LINES=$(wc -l < "$doc")
        log_success "$doc exists ($LINES lines)"
    else
        log_failure "$doc not found"
    fi
done

# Check for Makefile
if [ -f "Makefile" ]; then
    TARGET_COUNT=$(grep -c "^[a-zA-Z_-]*:" Makefile || echo 0)
    log_success "Makefile exists ($TARGET_COUNT targets)"
else
    log_failure "Makefile not found"
fi

###############################################################################
# Final Summary
###############################################################################

log_section "Test Summary"

echo ""
echo -e "${BLUE}Total Tests Run: ${NC}$TOTAL_TESTS"
echo -e "${GREEN}Passed: ${NC}$PASSED_TESTS"
echo -e "${RED}Failed: ${NC}$FAILED_TESTS"
echo -e "${YELLOW}Skipped: ${NC}$SKIPPED_TESTS"
echo ""

SUCCESS_RATE=$((PASSED_TESTS * 100 / (PASSED_TESTS + FAILED_TESTS)))
echo -e "${BLUE}Success Rate: ${NC}${SUCCESS_RATE}%"
echo ""

# Save summary to file
cat > "$TEST_OUTPUT_DIR/summary.txt" << EOF
VulcanAMI_LLM Comprehensive Test Summary
========================================
Date: $(date)
Total Tests: $TOTAL_TESTS
Passed: $PASSED_TESTS
Failed: $FAILED_TESTS
Skipped: $SKIPPED_TESTS
Success Rate: ${SUCCESS_RATE}%

All test logs are available in: $TEST_OUTPUT_DIR/
EOF

log_info "Test results saved to: $TEST_OUTPUT_DIR/"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}\n"
    exit 0
else
    echo -e "\n${RED}✗ Some tests failed. Review the logs for details.${NC}\n"
    exit 1
fi
