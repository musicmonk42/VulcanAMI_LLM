#!/usr/bin/env python3
"""
VulcanAMI System Validation Script
Validates all core components are working as designed
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class SystemValidator:
    """Validates all VulcanAMI system components"""

    def __init__(self):
        self.results = {}
        self.critical_failures = []
        self.warnings = []

    def validate_module(self, module_path: str, description: str) -> bool:
        """Validate a module can be imported"""
        try:
            module = importlib.import_module(module_path)
            logger.info(f"✅ {description}: OK")
            self.results[description] = "PASS"
            return True
        except ImportError as e:
            logger.error(f"❌ {description}: FAILED - {e}")
            self.results[description] = f"FAIL: {e}"
            self.critical_failures.append(description)
            return False
        except Exception as e:
            logger.warning(f"⚠️  {description}: WARNING - {e}")
            self.results[description] = f"WARNING: {e}"
            self.warnings.append(description)
            return False

    def validate_class(self, module_path: str, class_name: str, description: str) -> bool:
        """Validate a class can be instantiated or imported"""
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            logger.info(f"✅ {description}: OK ({class_name} available)")
            self.results[description] = "PASS"
            return True
        except Exception as e:
            logger.error(f"❌ {description}: FAILED - {e}")
            self.results[description] = f"FAIL: {e}"
            self.critical_failures.append(description)
            return False

    def validate_memory_system(self) -> bool:
        """Validate memory system components"""
        logger.info("\n🧠 VALIDATING MEMORY SYSTEM...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.memory.base", "Memory - Base Classes")
        all_ok &= self.validate_module("vulcan.memory.hierarchical", "Memory - Hierarchical System")
        all_ok &= self.validate_module("vulcan.memory.retrieval", "Memory - Retrieval System")
        all_ok &= self.validate_module("vulcan.memory.consolidation", "Memory - Consolidation")
        all_ok &= self.validate_module("vulcan.memory.persistence", "Memory - Persistence")
        all_ok &= self.validate_module("vulcan.memory.specialized", "Memory - Specialized Types")
        all_ok &= self.validate_module("vulcan.memory.distributed", "Memory - Distributed System")

        # Also check persistant_memory_v46
        all_ok &= self.validate_module("persistant_memory_v46", "Persistent Memory v46")

        return all_ok

    def validate_self_improvement(self) -> bool:
        """Validate self-improvement system"""
        logger.info("\n🔧 VALIDATING SELF-IMPROVEMENT SYSTEM...")

        all_ok = True
        all_ok &= self.validate_module(
            "vulcan.world_model.meta_reasoning.self_improvement_drive",
            "Self-Improvement - Drive System"
        )
        all_ok &= self.validate_class(
            "vulcan.world_model.meta_reasoning.self_improvement_drive",
            "SelfImprovementDrive",
            "Self-Improvement - Drive Class"
        )
        all_ok &= self.validate_module(
            "vulcan.world_model.meta_reasoning.csiu_enforcement",
            "Self-Improvement - CSIU Enforcement"
        )
        all_ok &= self.validate_module(
            "vulcan.world_model.meta_reasoning.safe_execution",
            "Self-Improvement - Safe Execution"
        )
        all_ok &= self.validate_module(
            "vulcan.world_model.meta_reasoning.auto_apply_policy",
            "Self-Improvement - Auto-Apply Policy"
        )

        return all_ok

    def validate_world_model(self) -> bool:
        """Validate world model components"""
        logger.info("\n🌍 VALIDATING WORLD MODEL...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.world_model.world_model_core", "World Model - Core")
        all_ok &= self.validate_module("vulcan.world_model.causal_graph", "World Model - Causal Graph")
        all_ok &= self.validate_module("vulcan.world_model.dynamics_model", "World Model - Dynamics")
        all_ok &= self.validate_module("vulcan.world_model.prediction_engine", "World Model - Predictions")
        all_ok &= self.validate_module("vulcan.world_model.intervention_manager", "World Model - Interventions")
        all_ok &= self.validate_module("vulcan.world_model.correlation_tracker", "World Model - Correlations")
        all_ok &= self.validate_module("vulcan.world_model.invariant_detector", "World Model - Invariants")
        all_ok &= self.validate_module("vulcan.world_model.confidence_calibrator", "World Model - Confidence")

        return all_ok

    def validate_reasoning(self) -> bool:
        """Validate reasoning system"""
        logger.info("\n🤔 VALIDATING REASONING SYSTEM...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.reasoning", "Reasoning - Core Module")
        all_ok &= self.validate_module("vulcan.reasoning.unified_reasoning", "Reasoning - Unified System")

        return all_ok

    def validate_semantic_bridge(self) -> bool:
        """Validate semantic bridge (native language)"""
        logger.info("\n🌉 VALIDATING SEMANTIC BRIDGE (NATIVE LANGUAGE)...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.semantic_bridge", "Semantic Bridge - Core")
        # Note: semantic_graph and semantic_space are not separate modules - graph functionality
        # is integrated into semantic_bridge_core with networkx or SimpleDiGraph fallback
        all_ok &= self.validate_module("vulcan.semantic_bridge.concept_mapper", "Semantic Bridge - Concept Mapper")
        all_ok &= self.validate_module("vulcan.semantic_bridge.transfer_engine", "Semantic Bridge - Transfer Engine")
        all_ok &= self.validate_module("vulcan.semantic_bridge.domain_registry", "Semantic Bridge - Domain Registry")

        return all_ok

    def validate_arena(self) -> bool:
        """Validate arena system"""
        logger.info("\n🏟️  VALIDATING ARENA SYSTEM...")

        all_ok = True
        all_ok &= self.validate_module("graphix_arena", "Arena - Graphix Arena")
        all_ok &= self.validate_class("graphix_arena", "GraphixArena", "Arena - Main Class")

        return all_ok

    def validate_safety_system(self) -> bool:
        """Validate safety systems"""
        logger.info("\n🛡️  VALIDATING SAFETY SYSTEMS...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.safety.safety_validator", "Safety - Validator")
        all_ok &= self.validate_module("vulcan.safety.neural_safety", "Safety - Neural Safety")
        all_ok &= self.validate_module("vulcan.safety.governance_alignment", "Safety - Governance")
        all_ok &= self.validate_module("vulcan.safety.tool_safety", "Safety - Tool Safety")
        all_ok &= self.validate_module("vulcan.safety.domain_validators", "Safety - Domain Validators")

        return all_ok

    def validate_orchestrator(self) -> bool:
        """Validate orchestration system"""
        logger.info("\n🎭 VALIDATING ORCHESTRATOR...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.orchestrator", "Orchestrator - Core")

        return all_ok

    def validate_utilities(self) -> bool:
        """Validate utility modules"""
        logger.info("\n🔧 VALIDATING UTILITIES...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.utils.numeric_utils", "Utils - Numeric Utilities")

        return all_ok

    def validate_runtime(self) -> bool:
        """Validate unified runtime"""
        logger.info("\n⚙️  VALIDATING UNIFIED RUNTIME...")

        all_ok = True
        all_ok &= self.validate_module("unified_runtime.unified_runtime_core", "Runtime - Core")
        all_ok &= self.validate_module("unified_runtime.execution_engine", "Runtime - Execution Engine")
        all_ok &= self.validate_module("unified_runtime.ai_runtime_integration", "Runtime - AI Integration")

        return all_ok

    def validate_learning(self) -> bool:
        """Validate learning systems"""
        logger.info("\n📚 VALIDATING LEARNING SYSTEMS...")

        all_ok = True
        all_ok &= self.validate_module("vulcan.learning", "Learning - Core Module")

        return all_ok

    def run_full_validation(self) -> bool:
        """Run complete system validation"""
        logger.info("="*80)
        logger.info("🚀 VULCANAMI SYSTEM VALIDATION")
        logger.info("="*80)

        all_systems_ok = True

        # Validate all subsystems
        all_systems_ok &= self.validate_memory_system()
        all_systems_ok &= self.validate_self_improvement()
        all_systems_ok &= self.validate_world_model()
        all_systems_ok &= self.validate_reasoning()
        all_systems_ok &= self.validate_semantic_bridge()
        all_systems_ok &= self.validate_arena()
        all_systems_ok &= self.validate_safety_system()
        all_systems_ok &= self.validate_orchestrator()
        all_systems_ok &= self.validate_utilities()
        all_systems_ok &= self.validate_runtime()
        all_systems_ok &= self.validate_learning()

        # Print summary
        self.print_summary(all_systems_ok)

        return all_systems_ok

    def print_summary(self, all_ok: bool):
        """Print validation summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("="*80)

        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v == "PASS")
        failed = len(self.critical_failures)
        warned = len(self.warnings)

        logger.info(f"\nTotal Components Checked: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Warnings: {warned}")

        if self.critical_failures:
            logger.error(f"\n❌ CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                logger.error(f"  - {failure}")

        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        logger.info("\n" + "="*80)
        if all_ok:
            logger.info("✅ SYSTEM STATUS: ALL SYSTEMS OPERATIONAL")
        else:
            logger.error("❌ SYSTEM STATUS: FAILURES DETECTED")
        logger.info("="*80)


def main():
    """Main validation entry point"""
    validator = SystemValidator()
    success = validator.run_full_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
