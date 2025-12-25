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
        'expected_methods': ['run_tournament', 'run_adaptive_tournament', 'select_winner', 'evaluate_proposals'],
        'request_flow_file': 'src/graphix_arena.py',
        'integration_note': 'Used in Arena /tournament endpoint, should appear in logs as "[trace] Tournament complete:"'
    },
    'EvolutionEngine': {
        'file': 'src/evolution_engine.py',
        'class': 'EvolutionEngine',
        'expected_methods': ['evolve', 'mutate', 'crossover', 'run_generation', 'run_evolution'],
        'request_flow_file': 'src/graphix_arena.py',
        'integration_note': 'Integrated into run_shadow_task for generator/evolver agents, logs "[EvolutionEngine] Starting evolution"'
    },
    'ConsensusEngine': {
        'file': 'src/consensus_engine.py',
        'class': 'ConsensusEngine',
        'expected_methods': ['reach_consensus', 'vote', 'aggregate', 'evaluate_proposal'],
        'request_flow_file': 'src/memory/governed_unlearning.py',
        'integration_note': 'Used in governed unlearning/training paths, not typical request flow'
    },
    'GovernanceLoop': {
        'file': 'src/governance_loop.py',
        'class': 'GovernanceLoop',
        'expected_methods': ['enforce_policies', 'check_compliance', 'add_policy', 'start', 'stop'],
        'request_flow_file': 'src/unified_runtime/runtime_extensions.py',
        'integration_note': 'Policy enforcement system; vulcan.routing.governance_logger is separate audit logger'
    },
    'Superoptimizer': {
        'file': 'src/superoptimizer.py',
        'class': 'Superoptimizer',
        'expected_methods': ['optimize', 'search', 'evaluate', 'optimize_graph', 'generate_kernel'],
        'request_flow_file': 'src/full_platform.py',
        'integration_note': 'Stored in app.state but not accessed; for kernel optimization, not typical request flow'
    },
    # Strategy Module Components
    'StrategyOrchestrator': {
        'file': 'src/strategies/strategy_orchestrator.py',
        'class': 'StrategyOrchestrator',
        'expected_methods': ['analyze', 'record_execution', 'get_statistics', 'get_health_status', 'get_drift_status'],
        'request_flow_file': 'src/vulcan/routing/query_router.py',
        'integration_note': 'Main entry point for strategies module - wires cost model, drift monitor, VOI gate, etc.'
    },
    'StochasticCostModel': {
        'file': 'src/strategies/cost_model.py',
        'class': 'StochasticCostModel',
        'expected_methods': ['predict_cost', 'update', 'get_statistics', 'save_model', 'load_model'],
        'request_flow_file': 'src/strategies/strategy_orchestrator.py',
        'integration_note': 'Uncertainty-aware cost prediction; used via StrategyOrchestrator'
    },
    'DistributionMonitor': {
        'file': 'src/strategies/distribution_monitor.py',
        'class': 'DistributionMonitor',
        'expected_methods': ['detect_shift', 'update', 'get_drift_summary', 'get_statistics'],
        'request_flow_file': 'src/strategies/strategy_orchestrator.py',
        'integration_note': 'Drift detection (KS, Wasserstein, MMD); used via StrategyOrchestrator'
    },
    'ToolMonitor': {
        'file': 'src/strategies/tool_monitor.py',
        'class': 'ToolMonitor',
        'expected_methods': ['record_execution', 'get_health_status', 'get_statistics', 'export_metrics'],
        'request_flow_file': 'src/strategies/strategy_orchestrator.py',
        'integration_note': 'Tool health, latency tracking; used via StrategyOrchestrator'
    },
    'ValueOfInformationGate': {
        'file': 'src/strategies/value_of_information.py',
        'class': 'ValueOfInformationGate',
        'expected_methods': ['should_gather_more', 'evaluate', 'get_statistics'],
        'request_flow_file': 'src/strategies/strategy_orchestrator.py',
        'integration_note': 'Decides when to gather more info vs proceed; used via StrategyOrchestrator'
    },
    # SelfOptimizer
    'SelfOptimizer': {
        'file': 'src/evolve/self_optimizer.py',
        'class': 'SelfOptimizer',
        'expected_methods': ['start', 'stop', 'collect_metrics', 'optimize', 'get_recommendations'],
        'request_flow_file': 'src/vulcan/main.py',
        'integration_note': 'Autonomous performance tuning; started in lifespan, stopped on shutdown'
    },
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


def check_request_flow_integration(class_name: str, request_flow_file: str) -> dict:
    """Check if a module is integrated into the request flow file."""
    result = {
        'integrated': False,
        'location': None,
        'usage_type': None
    }
    
    try:
        file_path = Path(request_flow_file)
        if not file_path.exists():
            return result
            
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check for various integration patterns
        patterns = [
            (rf'self\.{class_name.lower()}', 'instance_attribute'),
            (rf'self\.\w*{class_name.lower()}\w*', 'instance_attribute'),
            (rf'app\.state\.{class_name.lower()}', 'app_state'),
            (rf'{class_name}\(', 'direct_instantiation'),
        ]
        
        for pattern, usage_type in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result['integrated'] = True
                result['usage_type'] = usage_type
                # Find line number
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        result['location'] = i
                        break
                break
                
    except Exception as e:
        print(f"Warning: Could not check {request_flow_file}: {e}")
        
    return result


def audit_module(name: str, config: dict) -> dict:
    """Audit a single module's usage."""
    print(f"\n{'='*60}")
    print(f"AUDITING: {name}")
    print(f"{'='*60}")

    results = find_all_calls('src', config['class'], config['expected_methods'])
    request_flow = check_request_flow_integration(config['class'], config.get('request_flow_file', ''))

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

    # Check request flow integration
    print(f"\n🔄 Request Flow Integration:")
    if request_flow['integrated']:
        print(f"   ✅ Integrated in {config.get('request_flow_file', 'N/A')}")
        print(f"      Usage: {request_flow['usage_type']} at line {request_flow['location']}")
    else:
        print(f"   ⚠️  Not found in {config.get('request_flow_file', 'N/A')}")
    
    print(f"\n📝 Integration Note:")
    print(f"   {config.get('integration_note', 'N/A')}")

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
        'request_flow_integrated': request_flow['integrated'],
        'status': status,
        'integration_note': config.get('integration_note', '')
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
    print("REQUEST FLOW INTEGRATION STATUS")
    print("=" * 60)
    
    for r in summary:
        status_icon = "✅" if r['request_flow_integrated'] else "⚠️"
        print(f"\n{status_icon} {r['name']}:")
        print(f"   {r['integration_note']}")

    return summary


if __name__ == "__main__":
    main()
