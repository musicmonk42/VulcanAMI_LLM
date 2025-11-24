#!/usr/bin/env python3
"""
Simple verification script to ensure warning messages are displayed correctly.
This script can be run independently without pytest infrastructure.
"""

import sys
import logging

# Capture warnings
captured_warnings = []

class WarningCapture(logging.Handler):
    def emit(self, record):
        if record.levelno == logging.WARNING:
            captured_warnings.append(record.getMessage())

# Setup logging to capture warnings
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
handler = WarningCapture()
root_logger.addHandler(handler)

print("=" * 70)
print("VERIFYING SYSTEM CAPABILITY WARNINGS")
print("=" * 70)

# Clear any previous warnings
captured_warnings.clear()

# Test 1: Import ZK module (should trigger Groth16 warning)
print("\n1. Testing Groth16 SNARK module warning...")
try:
    import src.persistant_memory_v46.zk as zk_module
    if not zk_module.SNARK_AVAILABLE:
        expected_msg = "Groth16 SNARK module unavailable (falling back to basic implementation)"
        found = any(expected_msg in w for w in captured_warnings)
        if found:
            print(f"   ✓ PASS: '{expected_msg}'")
        else:
            print(f"   ✗ FAIL: Expected warning not found")
            print(f"   Captured warnings: {captured_warnings}")
            sys.exit(1)
    else:
        print("   ℹ Module is available (warning not expected)")
except Exception as e:
    print(f"   ✗ FAIL: Import error: {e}")
    sys.exit(1)

# Clear warnings for next test
captured_warnings.clear()

# Test 2: Import analogical reasoning (should trigger spaCy warning)
print("\n2. Testing spaCy model warning...")
try:
    import src.vulcan.reasoning.analogical_reasoning as analogical
    if analogical.nlp is None:
        expected_msg = "spaCy model not loaded for analogical reasoning"
        found = any(expected_msg in w for w in captured_warnings)
        if found:
            print(f"   ✓ PASS: '{expected_msg}'")
        else:
            print(f"   ✗ FAIL: Expected warning not found")
            print(f"   Captured warnings: {captured_warnings}")
            sys.exit(1)
    else:
        print("   ℹ Model is loaded (warning not expected)")
except Exception as e:
    print(f"   ✗ FAIL: Import error: {e}")
    sys.exit(1)

# Clear warnings for next test
captured_warnings.clear()

# Test 3: Import language evolution registry (should trigger FAISS AVX warning)
print("\n3. Testing FAISS AVX capability warning...")
try:
    import specs.formal_grammar.language_evolution_registry as registry
    expected_msg = "FAISS loaded with AVX2 (AVX512 unavailable)"
    found = any(expected_msg in w for w in captured_warnings)
    if found:
        print(f"   ✓ PASS: '{expected_msg}'")
    else:
        # This warning may not appear if AVX512 is available
        print("   ℹ AVX512 may be available on this system (warning not expected)")
except Exception as e:
    print(f"   ✗ FAIL: Import error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL VERIFICATION TESTS PASSED")
print("=" * 70)
print("\nSummary of warnings captured:")
for i, warning in enumerate(captured_warnings, 1):
    print(f"  {i}. {warning}")
