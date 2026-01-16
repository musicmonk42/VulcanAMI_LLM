#!/bin/bash
################################################################################
# Architecture Consolidation Validation Script
# 
# This script validates that the migration from vulcan.reasoning.integration
# to vulcan.reasoning.unified is complete and correct across all deployment
# files, documentation, and configurations.
#
# Exit codes:
#   0 = All checks passed
#   1 = One or more checks failed
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FAILED_CHECKS=0
TOTAL_CHECKS=0

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Architecture Consolidation Validation${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to run a check
check() {
    local name="$1"
    local description="$2"
    shift 2
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$TOTAL_CHECKS] ${description}... "
    
    if "$@" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# Function to run a negative check (should NOT find anything)
check_not_found() {
    local name="$1"
    local description="$2"
    local pattern="$3"
    shift 3
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$TOTAL_CHECKS] ${description}... "
    
    # Check if pattern is found in any of the files
    local found=0
    for file in "$@"; do
        if [ -f "$file" ] && grep -q "$pattern" "$file" 2>/dev/null; then
            found=1
            break
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo -e "  ${YELLOW}Found '$pattern' in: $file${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

echo -e "${YELLOW}Phase 1: Python Module Validation${NC}"
echo "-----------------------------------"

# Check that key Python files compile
check "compile_reasoning_init" \
    "Compile src/vulcan/reasoning/__init__.py" \
    python -m py_compile src/vulcan/reasoning/__init__.py

check "compile_unified_init" \
    "Compile src/vulcan/reasoning/unified/__init__.py" \
    python -m py_compile src/vulcan/reasoning/unified/__init__.py

check "compile_singletons" \
    "Compile src/vulcan/reasoning/singletons.py" \
    python -m py_compile src/vulcan/reasoning/singletons.py

check "compile_unified_chat" \
    "Compile src/vulcan/endpoints/unified_chat.py" \
    python -m py_compile src/vulcan/endpoints/unified_chat.py

check "compile_agent_pool" \
    "Compile src/vulcan/orchestrator/agent_pool.py" \
    python -m py_compile src/vulcan/orchestrator/agent_pool.py

check "compile_query_router" \
    "Compile src/vulcan/routing/query_router.py" \
    python -m py_compile src/vulcan/routing/query_router.py

check "compile_manager" \
    "Compile src/vulcan/server/startup/manager.py" \
    python -m py_compile src/vulcan/server/startup/manager.py

echo ""
echo -e "${YELLOW}Phase 2: Legacy Package Deletion Verification${NC}"
echo "-----------------------------------------------"

# Verify integration package directory is deleted
check "integration_dir_deleted" \
    "Verify src/vulcan/reasoning/integration/ is deleted" \
    test ! -d src/vulcan/reasoning/integration

# Verify no integration package files remain
check "no_integration_files" \
    "Verify no integration package files remain" \
    test ! -f src/vulcan/reasoning/integration/__init__.py

echo ""
echo -e "${YELLOW}Phase 3: Docker & Container Configuration${NC}"
echo "------------------------------------------"

# Check Dockerfile doesn't reference integration
check_not_found "dockerfile_integration" \
    "Dockerfile has no integration references" \
    "vulcan\.reasoning\.integration\|ReasoningIntegration" \
    Dockerfile

# Check docker-compose files don't reference integration
check_not_found "docker_compose_dev_integration" \
    "docker-compose.dev.yml has no integration references" \
    "vulcan\.reasoning\.integration\|ReasoningIntegration" \
    docker-compose.dev.yml

check_not_found "docker_compose_prod_integration" \
    "docker-compose.prod.yml has no integration references" \
    "vulcan\.reasoning\.integration\|ReasoningIntegration" \
    docker-compose.prod.yml

echo ""
echo -e "${YELLOW}Phase 4: Kubernetes & Helm Configuration${NC}"
echo "-----------------------------------------"

# Check that K8s manifests don't reference integration
HELM_FILES=$(find helm/ -name "*.yaml" -o -name "*.yml" 2>/dev/null || true)
K8S_FILES=$(find k8s/ -name "*.yaml" -o -name "*.yml" 2>/dev/null || true)

if [ -n "$HELM_FILES" ]; then
    for file in $HELM_FILES; do
        check_not_found "helm_$(basename $file)" \
            "Helm file $(basename $file) has no integration references" \
            "vulcan\.reasoning\.integration\|ReasoningIntegration" \
            "$file"
    done
else
    echo "  ${YELLOW}No Helm files found (OK)${NC}"
fi

if [ -n "$K8S_FILES" ]; then
    for file in $K8S_FILES; do
        check_not_found "k8s_$(basename $file)" \
            "K8s file $(basename $file) has no integration references" \
            "vulcan\.reasoning\.integration\|ReasoningIntegration" \
            "$file"
    done
else
    echo "  ${YELLOW}No K8s files found (OK)${NC}"
fi

echo ""
echo -e "${YELLOW}Phase 5: Makefile & CI/CD Scripts${NC}"
echo "----------------------------------"

# Check Makefile
check_not_found "makefile_integration" \
    "Makefile has no integration references" \
    "vulcan\.reasoning\.integration" \
    Makefile

# Check GitHub workflows (if they exist)
if [ -d .github/workflows ]; then
    WORKFLOW_FILES=$(find .github/workflows -name "*.yml" -o -name "*.yaml" 2>/dev/null || true)
    if [ -n "$WORKFLOW_FILES" ]; then
        for file in $WORKFLOW_FILES; do
            check_not_found "workflow_$(basename $file)" \
                "Workflow $(basename $file) has no integration references" \
                "vulcan\.reasoning\.integration" \
                "$file"
        done
    fi
fi

echo ""
echo -e "${YELLOW}Phase 6: Documentation Consistency${NC}"
echo "-----------------------------------"

# Check ARCHITECTURE_OVERVIEW.md was updated
check "arch_overview_updated" \
    "ARCHITECTURE_OVERVIEW.md references UnifiedReasoner" \
    grep -q "UnifiedReasoner" docs/ARCHITECTURE_OVERVIEW.md

# Verify it doesn't reference ReasoningIntegration in diagrams
if grep -q "QueryRouter → ReasoningIntegration" docs/ARCHITECTURE_OVERVIEW.md 2>/dev/null; then
    echo -e "  ${RED}✗ FAIL: ARCHITECTURE_OVERVIEW.md still has old diagram${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
else
    echo -e "  ${GREEN}✓ PASS: ARCHITECTURE_OVERVIEW.md diagrams updated${NC}"
fi

echo ""
echo -e "${YELLOW}Phase 7: Import Compatibility Verification${NC}"
echo "-------------------------------------------"

# Test that compatibility imports work
echo -n "Testing compatibility imports... "
if python3 -c "
import sys
sys.path.insert(0, 'src')
from vulcan.reasoning import apply_reasoning
from vulcan.reasoning import get_reasoning_integration
from vulcan.reasoning import observe_query_start
from vulcan.reasoning import ensure_reasoning_type_enum
print('All imports successful')
" 2>&1 | grep -q "All imports successful"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Test that old imports fail as expected
echo -n "Testing old imports correctly fail... "
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from vulcan.reasoning.integration import apply_reasoning
    print('UNEXPECTED: Old import succeeded')
    sys.exit(1)
except ImportError:
    print('Expected import failure')
    sys.exit(0)
" 2>&1 | grep -q "Expected import failure"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}======================================${NC}"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✓ All $TOTAL_CHECKS checks passed!${NC}"
    echo ""
    echo -e "${GREEN}Architecture consolidation is complete and production-ready.${NC}"
    exit 0
else
    echo -e "${RED}✗ $FAILED_CHECKS of $TOTAL_CHECKS checks failed${NC}"
    echo ""
    echo -e "${YELLOW}Please review the failed checks above and address any issues.${NC}"
    exit 1
fi
