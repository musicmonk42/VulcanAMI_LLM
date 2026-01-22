"""
Tests for Bug Fixes in Vulcan Reasoning Pipeline

This test file validates the fixes for the following bugs:
1. Bug #1: Message format mismatch in MathematicalComputationTool
2. Bug #2: Missing tool name mappings in Orchestrator  
3. Bug #3: Query misclassification - symbolic logic routed to philosophical
4. Bug #4: Safety filter false positive on educational causal content
"""

import re
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "vulcan"


# ============================================================================
# BUG #1: Message Format Tests
# ============================================================================

def test_graphix_llm_client_message_format():
    """Test that mathematical computation tool wraps prompts correctly."""
    file_path = SRC_DIR / "reasoning" / "mathematical_computation.py"
    with open(file_path, 'r') as f:
        source = f.read()
    
    # Check that the fix is in place - look for the wrapped message format
    # More flexible pattern that handles different quote styles and whitespace
    pattern = r'llm\.chat\s*\(\s*\[\s*\{\s*["\']role["\']\s*:\s*["\']user["\']\s*,\s*["\']content["\']\s*:\s*prompt\s*\}\s*\]\s*\)'
    if re.search(pattern, source):
        print("✓ Bug #1: Mathematical computation tool has message format fix")
        return True
    else:
        print("✗ Bug #1: Message format fix not found")
        return False


# ============================================================================
# BUG #2: Tool Name Mappings Tests
# ============================================================================

def test_tool_name_mappings():
    """Test that new tool name aliases are in the mapping."""
    file_path = SRC_DIR / "reasoning" / "unified" / "orchestrator.py"
    with open(file_path, 'r') as f:
        source = f.read()
    
    mappings = {
        'fol_solver': 'SYMBOLIC',
        'dag_analyzer': 'CAUSAL',
        'meta_reasoning': 'PHILOSOPHICAL',
    }
    
    all_found = True
    for tool_name, expected_type in mappings.items():
        pattern = rf"'{tool_name}':\s*ReasoningType\.{expected_type}"
        if re.search(pattern, source):
            print(f"✓ Bug #2: {tool_name} → {expected_type} mapping found")
        else:
            print(f"✗ Bug #2: {tool_name} → {expected_type} mapping NOT found")
            all_found = False
    
    return all_found


# ============================================================================
# BUG #3: Symbolic Logic Classification Tests
# ============================================================================

def test_symbolic_logic_prompt_patterns():
    """Test that LLM router prompt includes symbolic logic patterns."""
    file_path = SRC_DIR / "routing" / "routing_prompts.py"
    with open(file_path, 'r') as f:
        source = f.read()
    
    patterns_to_check = [
        'rule chaining',
        'nonmonotonic',
        'rule-based reasoning',
        'Rule-based reasoning',
    ]
    
    found_count = 0
    for pattern in patterns_to_check:
        if pattern.lower() in source.lower():
            found_count += 1
    
    if found_count >= 2:  # At least 2 patterns should be present
        print(f"✓ Bug #3: LLM router prompt includes symbolic logic patterns ({found_count}/{len(patterns_to_check)})")
        return True
    else:
        print(f"✗ Bug #3: Not enough symbolic logic patterns found ({found_count}/{len(patterns_to_check)})")
        return False


# ============================================================================
# BUG #4: Safety Filter Educational Content Tests
# ============================================================================

def test_pearl_style_causal_patterns():
    """Test that safety validator includes Pearl-style causal patterns."""
    file_path = SRC_DIR / "safety" / "safety_validator.py"
    with open(file_path, 'r') as f:
        source = f.read()
    
    patterns_to_check = [
        'pearl',
        'confounding.*causation',
        'you.*observe.*dataset',
        'causal.*arrow',
        r'→|->',  # Causal arrow notation (Unicode or ASCII alternative)
    ]
    
    found_count = 0
    for pattern in patterns_to_check:
        if re.search(pattern, source, re.IGNORECASE):
            found_count += 1
    
    if found_count >= 3:  # At least 3 patterns should be present
        print(f"✓ Bug #4: Safety validator includes Pearl-style patterns ({found_count}/{len(patterns_to_check)})")
        return True
    else:
        print(f"✗ Bug #4: Not enough Pearl-style patterns found ({found_count}/{len(patterns_to_check)})")
        return False


def test_causal_education_keywords():
    """Test that causal education keywords include Pearl-style terms."""
    file_path = SRC_DIR / "safety" / "safety_validator.py"
    with open(file_path, 'r') as f:
        source = f.read()
    
    keywords_to_check = [
        'pearl-style',
        'confounding vs causation',
        'causal arrow',
    ]
    
    found_count = 0
    for keyword in keywords_to_check:
        if keyword.lower() in source.lower():
            found_count += 1
    
    if found_count >= 2:  # At least 2 keywords should be present
        print(f"✓ Bug #4: Causal education keywords updated ({found_count}/{len(keywords_to_check)})")
        return True
    else:
        print(f"✗ Bug #4: Not enough causal keywords found ({found_count}/{len(keywords_to_check)})")
        return False


if __name__ == "__main__":
    # Run tests manually
    print("\n" + "="*70)
    print("Running Bug Fix Validation Tests")
    print("="*70 + "\n")
    
    results = []
    
    print("Testing Bug #1: Message Format Mismatch")
    print("-" * 70)
    results.append(test_graphix_llm_client_message_format())
    print()
    
    print("Testing Bug #2: Tool Name Mappings")
    print("-" * 70)
    results.append(test_tool_name_mappings())
    print()
    
    print("Testing Bug #3: Symbolic Logic Classification")
    print("-" * 70)
    results.append(test_symbolic_logic_prompt_patterns())
    print()
    
    print("Testing Bug #4: Safety Filter Educational Content")
    print("-" * 70)
    result1 = test_pearl_style_causal_patterns()
    result2 = test_causal_education_keywords()
    results.append(result1 and result2)
    print()
    
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All bug fixes validated successfully!")
        print("="*70)
        exit(0)
    else:
        print("✗ Some tests failed - please review the changes")
        print("="*70)
        exit(1)

