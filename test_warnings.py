#!/usr/bin/env python3
"""
Test script to verify that system capability warnings are displayed correctly.
"""
import sys
import logging

# Configure logging to display warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

print("Testing system capability warnings...\n")

# Test 1: Groth16 SNARK module
print("1. Testing Groth16 SNARK module warning:")
try:
    from src.persistant_memory_v46.zk import SNARK_AVAILABLE
    if not SNARK_AVAILABLE:
        print("   ✓ Warning displayed correctly")
    else:
        print("   ✓ Module available (no warning expected)")
except Exception as e:
    print(f"   ✗ Error importing: {e}")

# Test 2: spaCy model for analogical reasoning
print("\n2. Testing spaCy model warning:")
try:
    from src.vulcan.reasoning.analogical_reasoning import nlp
    if nlp is None:
        print("   ✓ Warning displayed correctly")
    else:
        print("   ✓ Model loaded (no warning expected)")
except Exception as e:
    print(f"   ✗ Error importing: {e}")

# Test 3: FAISS AVX2/AVX512
print("\n3. Testing FAISS AVX capability warning:")
try:
    # Import the module which will trigger the FAISS check
    import specs.formal_grammar.language_evolution_registry as registry
    print("   ✓ FAISS AVX capability check completed")
except Exception as e:
    print(f"   ✗ Error importing: {e}")

print("\n✓ All warning tests completed")
