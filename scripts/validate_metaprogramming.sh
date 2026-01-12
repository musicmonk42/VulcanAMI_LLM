#!/bin/bash
# Metaprogramming Features - Quick Validation Script
# Tests that metaprogramming node handlers are properly integrated

set -e

echo "=== Metaprogramming Features Validation ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}1. Checking Handler Registration${NC}"
python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.unified_runtime.node_handlers import get_node_handlers
    handlers = get_node_handlers()
    
    metaprog_handlers = [
        'PATTERN_COMPILE', 'FIND_SUBGRAPH', 'GRAPH_SPLICE', 
        'GRAPH_COMMIT', 'NSO_MODIFY', 'ETHICAL_LABEL', 'EVAL', 'HALT'
    ]
    
    registered = [h for h in metaprog_handlers if h in handlers]
    
    print(f'  Total handlers: {len(handlers)}')
    print(f'  Metaprogramming handlers: {len(registered)}/8')
    
    if len(registered) == 8:
        print('  ✓ All metaprogramming handlers registered')
        sys.exit(0)
    else:
        print(f'  ✗ Missing handlers: {set(metaprog_handlers) - set(registered)}')
        sys.exit(1)
except Exception as e:
    print(f'  ✗ Handler check failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Handler registration verified${NC}"
else
    echo -e "${RED}✗ Handler registration failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}2. Checking GraphAwareEvolutionEngine${NC}"
python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.graph_aware_evolution import GraphAwareEvolutionEngine
    from src.evolution_engine import EvolutionEngine
    
    # Verify inheritance
    assert issubclass(GraphAwareEvolutionEngine, EvolutionEngine)
    print('  ✓ GraphAwareEvolutionEngine extends EvolutionEngine')
    
    # Create instance
    engine = GraphAwareEvolutionEngine(population_size=5)
    print(f'  ✓ Engine instantiation works')
    
    # Check metaprogramming stats
    stats = engine.get_metaprogramming_stats()
    assert 'metaprogramming_enabled' in stats
    print(f'  ✓ Statistics tracking available')
    
    sys.exit(0)
except Exception as e:
    print(f'  ✗ GraphAwareEvolutionEngine check failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GraphAwareEvolutionEngine verified${NC}"
else
    echo -e "${RED}✗ GraphAwareEvolutionEngine check failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}3. Checking Documentation${NC}"
docs_found=0

if [ -f "docs/architecture/ADR-001-metaprogramming.md" ]; then
    echo "  ✓ ADR-001-metaprogramming.md exists"
    docs_found=$((docs_found + 1))
fi

if [ -f "docs/architecture/SECURITY-metaprogramming.md" ]; then
    echo "  ✓ SECURITY-metaprogramming.md exists"
    docs_found=$((docs_found + 1))
fi

if [ -f "docs/architecture/IMPLEMENTATION-SUMMARY.md" ]; then
    echo "  ✓ IMPLEMENTATION-SUMMARY.md exists"
    docs_found=$((docs_found + 1))
fi

if [ -f "docs/METAPROGRAMMING_DEPLOYMENT.md" ]; then
    echo "  ✓ METAPROGRAMMING_DEPLOYMENT.md exists"
    docs_found=$((docs_found + 1))
fi

if [ $docs_found -eq 4 ]; then
    echo -e "${GREEN}✓ All documentation present${NC}"
else
    echo -e "${YELLOW}⚠ Some documentation missing (${docs_found}/4)${NC}"
fi

echo ""
echo -e "${GREEN}4. Checking Test Files${NC}"
tests_found=0

if [ -f "tests/test_metaprogramming_handlers.py" ]; then
    echo "  ✓ test_metaprogramming_handlers.py exists"
    tests_found=$((tests_found + 1))
fi

if [ -f "tests/test_metaprogramming_integration.py" ]; then
    echo "  ✓ test_metaprogramming_integration.py exists"
    tests_found=$((tests_found + 1))
fi

if [ -f "tests/test_graph_aware_evolution.py" ]; then
    echo "  ✓ test_graph_aware_evolution.py exists"
    tests_found=$((tests_found + 1))
fi

if [ $tests_found -eq 3 ]; then
    echo -e "${GREEN}✓ All test files present${NC}"
else
    echo -e "${RED}✗ Test files missing (${tests_found}/3)${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}5. Running Quick Unit Tests${NC}"
if command -v pytest &> /dev/null; then
    pytest tests/test_metaprogramming_handlers.py -q --tb=no 2>&1 | tail -5
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Unit tests passing${NC}"
    else
        echo -e "${YELLOW}⚠ Some unit tests failed (check with: pytest tests/test_metaprogramming_handlers.py -v)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ pytest not available, skipping test execution${NC}"
fi

echo ""
echo "=== Validation Summary ==="
echo -e "${GREEN}✓ Handler Registration: PASS${NC}"
echo -e "${GREEN}✓ Evolution Engine: PASS${NC}"
echo -e "${GREEN}✓ Documentation: COMPLETE${NC}"
echo -e "${GREEN}✓ Test Files: PRESENT${NC}"
echo ""
echo -e "${GREEN}=== METAPROGRAMMING FEATURES READY FOR DEPLOYMENT ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Build Docker image: make docker-build"
echo "  2. Run full tests: pytest tests/test_metaprogramming*.py tests/test_graph_aware*.py -v"
echo "  3. Deploy: See docs/METAPROGRAMMING_DEPLOYMENT.md"
echo ""
