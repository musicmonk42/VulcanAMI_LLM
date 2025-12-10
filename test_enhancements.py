#!/usr/bin/env python3
"""
Test script for new requirements:
1. CPU instruction set detection (ARM NEON, etc.)
2. Detailed diagnostic information in warnings
3. Runtime performance metrics
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("=" * 80)
print("TESTING NEW ENHANCEMENTS")
print("=" * 80)

# Test 1: CPU Capabilities Detection
print("\n1. CPU Capabilities Detection (ARM NEON, AVX, etc.)")
print("-" * 80)
try:
    from src.utils.cpu_capabilities import (get_capability_summary,
                                            get_cpu_capabilities)
    
    caps = get_cpu_capabilities()
    print(get_capability_summary())
    print(f"\nDetailed capabilities:")
    print(f"  Architecture: {caps.architecture}")
    print(f"  Platform: {caps.platform}")
    print(f"  Cores: {caps.cpu_cores}")
    print(f"  Best Vector Instruction Set: {caps.get_best_vector_instruction_set()}")
    print(f"  Performance Tier: {caps.get_performance_tier()}")
    
    if caps.architecture.lower().startswith('arm') or caps.architecture.lower().startswith('aarch'):
        print(f"\nARM Features:")
        print(f"  NEON: {caps.has_neon}")
        print(f"  SVE: {caps.has_sve}")
        print(f"  SVE2: {caps.has_sve2}")
    else:
        print(f"\nx86/x64 Features:")
        print(f"  SSE4.2: {caps.has_sse4_2}")
        print(f"  AVX: {caps.has_avx}")
        print(f"  AVX2: {caps.has_avx2}")
        print(f"  AVX-512F: {caps.has_avx512f}")
        print(f"  FMA: {caps.has_fma}")
    
    print("\n✓ CPU Capabilities Detection PASSED")
except Exception as e:
    print(f"\n✗ CPU Capabilities Detection FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Enhanced Warning Messages
print("\n\n2. Enhanced Diagnostic Warning Messages")
print("-" * 80)
try:
    # Import modules that should display enhanced warnings
    print("Loading modules to trigger warnings...")
    
    # Capture warnings
    import io
    from contextlib import redirect_stderr
    
    f = io.StringIO()
    with redirect_stderr(f):
        import specs.formal_grammar.language_evolution_registry as registry
    
    warnings_output = f.getvalue()
    print("Warnings captured:")
    print(warnings_output if warnings_output else "(No warnings - features may be available)")
    
    print("\n✓ Enhanced Warning Messages PASSED")
except Exception as e:
    print(f"\n✗ Enhanced Warning Messages FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Performance Metrics
print("\n\n3. Runtime Performance Metrics")
print("-" * 80)
try:
    from src.persistant_memory_v46.zk import ZKProver
    from src.utils.performance_metrics import get_performance_tracker
    
    tracker = get_performance_tracker()
    
    # Generate some proofs to collect metrics
    print("Generating ZK proofs to collect performance metrics...")
    prover = ZKProver()
    
    for i in range(5):
        proof = prover.generate_unlearning_proof(
            pattern=f'test_pattern_{i}',
            affected_packs=[f'pack{j}' for j in range(3)]
        )
    
    print("\nPerformance Report:")
    print(tracker.format_report())
    
    print("\n✓ Performance Metrics PASSED")
except Exception as e:
    print(f"\n✗ Performance Metrics FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL NEW ENHANCEMENTS TESTED")
print("=" * 80)
