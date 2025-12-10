#!/usr/bin/env python3
"""
Simple test to verify that the import statements added to dependencies.py work correctly.
This tests the imports directly without requiring numpy or other complex dependencies.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing import fixes for dependencies.py...")
print("=" * 60)

# Test each import individually
imports_to_test = [
    ("vulcan.learning.continual_learning", "ContinualLearner", "LEARNING:continual"),
    ("vulcan.planning", "DistributedCoordinator", "DISTRIBUTED:distributed"),
    ("vulcan.world_model.meta_reasoning.self_improvement_drive", "SelfImprovementDrive", "META_REASONING:self_improvement_drive"),
    ("vulcan.world_model.meta_reasoning.motivational_introspection", "MotivationalIntrospection", "META_REASONING:motivational_introspection"),
    ("vulcan.world_model.meta_reasoning.objective_hierarchy", "ObjectiveHierarchy", "META_REASONING:objective_hierarchy"),
    ("vulcan.world_model.meta_reasoning.objective_negotiator", "ObjectiveNegotiator", "META_REASONING:objective_negotiator"),
    ("vulcan.world_model.meta_reasoning.goal_conflict_detector", "GoalConflictDetector", "META_REASONING:goal_conflict_detector"),
    ("vulcan.world_model.meta_reasoning.preference_learner", "PreferenceLearner", "META_REASONING:preference_learner"),
    ("vulcan.world_model.meta_reasoning.value_evolution_tracker", "ValueEvolutionTracker", "META_REASONING:value_evolution_tracker"),
    ("vulcan.world_model.meta_reasoning.ethical_boundary_monitor", "EthicalBoundaryMonitor", "META_REASONING:ethical_boundary_monitor"),
    ("vulcan.world_model.meta_reasoning.curiosity_reward_shaper", "CuriosityRewardShaper", "META_REASONING:curiosity_reward_shaper"),
    ("vulcan.world_model.meta_reasoning.internal_critic", "InternalCritic", "META_REASONING:internal_critic"),
    ("vulcan.world_model.meta_reasoning.auto_apply_policy", "Policy", "META_REASONING:auto_apply_policy"),
    ("vulcan.world_model.meta_reasoning.counterfactual_objectives", "CounterfactualObjectiveReasoner", "META_REASONING:counterfactual_objectives"),
]

results = {}
failed_imports = []
successful_imports = []

for module_name, class_name, dep_name in imports_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        results[dep_name] = "✓ Import OK"
        successful_imports.append(dep_name)
        print(f"✓ {dep_name}: Successfully imported {class_name} from {module_name}")
    except ImportError as e:
        results[dep_name] = f"✗ Import FAILED: {e}"
        failed_imports.append(dep_name)
        print(f"✗ {dep_name}: Failed to import {class_name} from {module_name}")
        print(f"   Error: {e}")
    except Exception as e:
        results[dep_name] = f"✗ Error: {e}"
        failed_imports.append(dep_name)
        print(f"✗ {dep_name}: Unexpected error")
        print(f"   Error: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total imports tested: {len(imports_to_test)}")
print(f"Successful imports: {len(successful_imports)}")
print(f"Failed imports: {len(failed_imports)}")

if failed_imports:
    print("\nFailed imports:")
    for dep in failed_imports:
        print(f"  - {dep}")
    print("\n" + "=" * 60)
    print("TEST FAILED")
    print("=" * 60)
    sys.exit(1)
else:
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    sys.exit(0)
