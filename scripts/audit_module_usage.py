#!/usr/bin/env python3
"""
Audit which modules are initialized vs actually used in request handling.

This script analyzes the VULCAN codebase to identify modules that are:
1. Initialized at startup but never called during request processing
2. Imported but instantiation result is discarded
3. Have methods that are never invoked

Usage:
    python scripts/audit_module_usage.py
"""

import re
from pathlib import Path
from collections import defaultdict


# Modules to audit
MODULES = {
    'TournamentManager': {
        'file': 'src/tournament_manager.py',
        'class': 'TournamentManager',
        'expected_methods': ['run_tournament', 'run_adaptive_tournament', 'select_winner', 'evaluate_proposals']
    },
    'EvolutionEngine': {
        'file': 'src/evolution_engine.py',
        'class': 'EvolutionEngine',
        'expected_methods': ['evolve', 'mutate', 'crossover', 'run_generation', 'run_evolution']
    },
    'ConsensusEngine': {
        'file': 'src/consensus_engine.py',
        'class': 'ConsensusEngine',
        'expected_methods': ['reach_consensus', 'vote', 'aggregate', 'evaluate_proposal']
    },
    'GovernanceLoop': {
        'file': 'src/governance_loop.py',
        'class': 'GovernanceLoop',
        'expected_methods': ['enforce_policies', 'check_compliance', 'add_policy', 'start', 'stop']
    },
    'Superoptimizer': {
        'file': 'src/superoptimizer.py',
        'class': 'Superoptimizer',
        'expected_methods': ['optimize', 'search', 'evaluate', 'optimize_graph']
    }
}


def find_all_calls(src_dir: str, class_name: str, method_names: list) -> dict:
    """Find all calls to a class's methods across the codebase."""
    results = {
        'instantiations': [],  # Where class is created
        'method_calls': defaultdict(list),  # Where methods are called
        'imports': [],  # Where class is imported
        'stored_assignments': [],  # Where instance is stored (e.g., self.x = Class())
    }

    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"Warning: {src_dir} does not exist")
        return results

    for py_file in src_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Check for imports
                if f"from" in line and class_name in line:
                    results['imports'].append((str(py_file), i, line.strip()))
                elif f"import" in line and class_name in line:
                    results['imports'].append((str(py_file), i, line.strip()))

                # Check for instantiation with storage
                # Patterns: self.x = Class(), var = Class(), app.state.x = Class()
                storage_pattern = rf'(self\.\w+|app\.state\.\w+|\w+)\s*=\s*{class_name}\s*\('
                if re.search(storage_pattern, line):
                    results['stored_assignments'].append((str(py_file), i, line.strip()))

                # Check for bare instantiation (result discarded)
                # Pattern: Class() at start of line or after logger.info
                bare_pattern = rf'^\s*{class_name}\s*\('
                if re.match(bare_pattern, line.strip()) and '=' not in line:
                    results['instantiations'].append((str(py_file), i, line.strip(), 'DISCARDED'))
                elif f"{class_name}(" in line:
                    results['instantiations'].append((str(py_file), i, line.strip(), 'STORED'))

                # Check for method calls
                for method in method_names:
                    # Pattern: .method( or var.method( or self.x.method(
                    if f".{method}(" in line:
                        results['method_calls'][method].append((str(py_file), i, line.strip()))
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")

    return results


def audit_module(name: str, config: dict) -> dict:
    """Audit a single module's usage."""
    print(f"\n{'='*60}")
    print(f"AUDITING: {name}")
    print(f"{'='*60}")

    results = find_all_calls('src', config['class'], config['expected_methods'])

    print(f"\n📦 Imports ({len(results['imports'])}):")
    for file, line, code in results['imports'][:5]:
        # Shorten path for readability
        short_file = file.replace('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/', '')
        print(f"   {short_file}:{line}")
        print(f"      {code[:80]}")

    print(f"\n🔨 Instantiations ({len(results['instantiations'])}):")
    for item in results['instantiations'][:5]:
        file, line, code = item[0], item[1], item[2]
        status = item[3] if len(item) > 3 else 'UNKNOWN'
        short_file = file.replace('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/', '')
        print(f"   [{status}] {short_file}:{line}")
        print(f"      {code[:80]}")

    print(f"\n💾 Stored Assignments ({len(results['stored_assignments'])}):")
    for file, line, code in results['stored_assignments'][:5]:
        short_file = file.replace('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/', '')
        print(f"   {short_file}:{line}")
        print(f"      {code[:80]}")

    print(f"\n📞 Method Calls:")
    total_calls = 0
    for method, calls in results['method_calls'].items():
        total_calls += len(calls)
        if calls:
            print(f"   .{method}() - {len(calls)} call(s)")
            for file, line, code in calls[:2]:
                short_file = file.replace('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/', '')
                print(f"      {short_file}:{line} - {code[:60]}...")
        else:
            print(f"   .{method}() - ❌ NEVER CALLED")

    # Determine status
    has_storage = len(results['stored_assignments']) > 0
    has_calls = total_calls > 0

    if has_storage and has_calls:
        status = "✅ USED (stored and methods called)"
    elif has_storage and not has_calls:
        status = "⚠️  INITIALIZED BUT METHODS NEVER CALLED"
    elif not has_storage and has_calls:
        status = "❓ METHODS CALLED BUT NO STORED INSTANCE FOUND"
    else:
        status = "❌ NOT INTEGRATED"

    print(f"\n📊 Status: {status}")

    return {
        'name': name,
        'imports': len(results['imports']),
        'instantiations': len(results['instantiations']),
        'stored': len(results['stored_assignments']),
        'method_calls': total_calls,
        'status': status
    }


def main():
    print("=" * 60)
    print("VULCAN Module Usage Audit")
    print("=" * 60)
    print("\nThis audit checks which modules are initialized at startup")
    print("but never actually used during request processing.\n")

    summary = []
    for name, config in MODULES.items():
        result = audit_module(name, config)
        summary.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Module':<20} {'Imports':<8} {'Created':<8} {'Stored':<8} {'Calls':<8} {'Status'}")
    print("-" * 90)
    for r in summary:
        print(f"{r['name']:<20} {r['imports']:<8} {r['instantiations']:<8} {r['stored']:<8} {r['method_calls']:<8} {r['status']}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    unused_modules = [r for r in summary if 'NEVER CALLED' in r['status'] or 'NOT INTEGRATED' in r['status']]
    if unused_modules:
        print("\nModules that need wiring into request flow:")
        for r in unused_modules:
            print(f"  - {r['name']}: {r['status']}")
    else:
        print("\n✅ All modules appear to be integrated!")

    return summary


if __name__ == "__main__":
    main()
