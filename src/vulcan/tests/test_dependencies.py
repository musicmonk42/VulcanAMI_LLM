# ============================================================
# VULCAN-AGI Orchestrator - Dependencies Tests
# Comprehensive test suite for dependencies.py
# FIXED: Corrected test_complex_dependency_setup to match actual shutdown behavior
# ============================================================

from vulcan.orchestrator.dependencies import (
    DependencyCategory,
    EnhancedCollectiveDeps,
    create_full_deps,
    create_minimal_deps,
    get_status_symbol,
    print_dependency_report,
    safe_print,
    validate_dependencies,
)
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

# Add src directory to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import components to test

# ============================================================
# MOCK COMPONENTS
# ============================================================


class MockComponent:
    """Generic mock component with shutdown support"""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._shutdown_called = False

    def shutdown(self):
        """Mock shutdown method"""
        self._shutdown_called = True


class MockMetrics:
    """Mock metrics collector"""

    def __init__(self):
        self._shutdown_called = False
        self.counters = {}

    def increment_counter(self, name: str):
        self.counters[name] = self.counters.get(name, 0) + 1

    def shutdown(self):
        self._shutdown_called = True


# ============================================================
# TEST: UTILITY FUNCTIONS
# ============================================================


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_safe_print_normal_text(self):
        """Test safe_print with normal ASCII text"""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            safe_print("Hello World")
            output = sys.stdout.getvalue()
            self.assertIn("Hello World", output)
        finally:
            sys.stdout = old_stdout

    def test_safe_print_unicode_text(self):
        """Test safe_print with Unicode text"""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            safe_print("Hello ✓ World")
            output = sys.stdout.getvalue()
            # Should print something (either Unicode or ASCII fallback)
            self.assertGreater(len(output), 0)
        finally:
            sys.stdout = old_stdout

    def test_get_status_symbol_success(self):
        """Test get_status_symbol for success"""
        symbol = get_status_symbol(True)
        self.assertIsNotNone(symbol)
        self.assertGreater(len(symbol), 0)
        # Should be either Unicode checkmark or [OK]
        self.assertTrue(symbol in ["✓", "[OK]"])

    def test_get_status_symbol_failure(self):
        """Test get_status_symbol for failure"""
        symbol = get_status_symbol(False)
        self.assertIsNotNone(symbol)
        self.assertGreater(len(symbol), 0)
        # Should be either Unicode X or [MISSING]
        self.assertTrue(symbol in ["✗", "[MISSING]"])


# ============================================================
# TEST: DEPENDENCY CATEGORY
# ============================================================


class TestDependencyCategory(unittest.TestCase):
    """Test DependencyCategory class"""

    def test_category_constants(self):
        """Test that all category constants are defined"""
        self.assertEqual(DependencyCategory.CORE, "core")
        self.assertEqual(DependencyCategory.SAFETY, "safety")
        self.assertEqual(DependencyCategory.MEMORY, "memory")
        self.assertEqual(DependencyCategory.PROCESSING, "processing")
        self.assertEqual(DependencyCategory.REASONING, "reasoning")
        self.assertEqual(DependencyCategory.LEARNING, "learning")
        self.assertEqual(DependencyCategory.PLANNING, "planning")
        self.assertEqual(DependencyCategory.DISTRIBUTED, "distributed")


# ============================================================
# TEST: ENHANCED COLLECTIVE DEPS - INITIALIZATION
# ============================================================


class TestEnhancedCollectiveDepsInit(unittest.TestCase):
    """Test EnhancedCollectiveDeps initialization"""

    def test_default_initialization(self):
        """Test default initialization"""
        deps = EnhancedCollectiveDeps()

        self.assertIsNotNone(deps)
        self.assertTrue(deps._initialized)
        self.assertFalse(deps._shutdown)
        self.assertIsNotNone(deps.metrics)

    def test_initialization_with_components(self):
        """Test initialization with specific components"""
        mock_env = Mock()
        mock_safety = Mock()

        deps = EnhancedCollectiveDeps(env=mock_env, safety_validator=mock_safety)

        self.assertEqual(deps.env, mock_env)
        self.assertEqual(deps.safety_validator, mock_safety)

    def test_post_init_called(self):
        """Test that __post_init__ is called"""
        deps = EnhancedCollectiveDeps()
        self.assertTrue(deps._initialized)

    def test_all_fields_initialized(self):
        """Test that all fields exist"""
        deps = EnhancedCollectiveDeps()

        # Core
        self.assertTrue(hasattr(deps, "env"))
        self.assertTrue(hasattr(deps, "metrics"))

        # Safety
        self.assertTrue(hasattr(deps, "safety_validator"))
        self.assertTrue(hasattr(deps, "governance"))
        self.assertTrue(hasattr(deps, "nso_aligner"))
        self.assertTrue(hasattr(deps, "explainer"))

        # Memory
        self.assertTrue(hasattr(deps, "ltm"))
        self.assertTrue(hasattr(deps, "am"))
        self.assertTrue(hasattr(deps, "compressed_memory"))

        # Processing
        self.assertTrue(hasattr(deps, "multimodal"))

        # Reasoning
        self.assertTrue(hasattr(deps, "probabilistic"))
        self.assertTrue(hasattr(deps, "symbolic"))
        self.assertTrue(hasattr(deps, "causal"))
        self.assertTrue(hasattr(deps, "abstract"))
        self.assertTrue(hasattr(deps, "cross_modal"))

        # Learning
        self.assertTrue(hasattr(deps, "continual"))
        self.assertTrue(hasattr(deps, "compositional"))
        self.assertTrue(hasattr(deps, "meta_cognitive"))
        self.assertTrue(hasattr(deps, "world_model"))

        # Planning
        self.assertTrue(hasattr(deps, "goal_system"))
        self.assertTrue(hasattr(deps, "resource_compute"))

        # Distributed
        self.assertTrue(hasattr(deps, "distributed"))


# ============================================================
# TEST: VALIDATION
# ============================================================


class TestValidation(unittest.TestCase):
    """Test dependency validation"""

    def test_validate_empty_deps(self):
        """Test validation of empty dependencies"""
        deps = EnhancedCollectiveDeps()
        report = deps.validate()

        self.assertIsInstance(report, dict)
        self.assertIn(DependencyCategory.CORE, report)
        self.assertIn(DependencyCategory.SAFETY, report)
        self.assertIn(DependencyCategory.MEMORY, report)

    def test_validate_partial_deps(self):
        """Test validation with some dependencies set"""
        deps = EnhancedCollectiveDeps(safety_validator=Mock(), ltm=Mock())
        report = deps.validate()

        # Safety validator should not be in missing list
        self.assertNotIn("safety_validator", report[DependencyCategory.SAFETY])

        # LTM should not be in missing list
        self.assertNotIn("ltm", report[DependencyCategory.MEMORY])

        # But other components should be missing
        self.assertIn("governance", report[DependencyCategory.SAFETY])
        self.assertIn("am", report[DependencyCategory.MEMORY])

    def test_validate_full_deps(self):
        """Test validation with all dependencies set"""
        deps = EnhancedCollectiveDeps(
            env=Mock(),
            safety_validator=Mock(),
            governance=Mock(),
            nso_aligner=Mock(),
            explainer=Mock(),
            ltm=Mock(),
            am=Mock(),
            compressed_memory=Mock(),
            multimodal=Mock(),
            probabilistic=Mock(),
            symbolic=Mock(),
            causal=Mock(),
            abstract=Mock(),
            cross_modal=Mock(),
            continual=Mock(),
            compositional=Mock(),
            meta_cognitive=Mock(),
            world_model=Mock(),
            goal_system=Mock(),
            resource_compute=Mock(),
            experiment_generator=Mock(),  # FIXED
            problem_executor=Mock(),  # FIXED
            # Mock all meta-reasoning components
            self_improvement_drive=Mock(),
            motivational_introspection=Mock(),
            objective_hierarchy=Mock(),
            objective_negotiator=Mock(),
            goal_conflict_detector=Mock(),
            preference_learner=Mock(),
            value_evolution_tracker=Mock(),
            ethical_boundary_monitor=Mock(),
            curiosity_reward_shaper=Mock(),
            internal_critic=Mock(),
            auto_apply_policy=Mock(),
            validation_tracker=Mock(),
            transparency_interface=Mock(),
            counterfactual_objectives=Mock(),
            distributed=Mock(),  # ADDED MISSING
        )
        report = deps.validate()

        # Check that no required dependencies are missing
        for category, missing_deps in report.items():
            # Metrics is auto-created, so CORE might not be empty
            if category == DependencyCategory.CORE:
                self.assertEqual(
                    len(missing_deps),
                    0,
                    f"Category {category} has missing deps: {missing_deps}",
                )
            elif category == DependencyCategory.META_REASONING:
                # Filter out import failures
                missing_initialized = [
                    d for d in missing_deps if "(import failed)" not in d
                ]
                self.assertEqual(
                    len(missing_initialized),
                    0,
                    f"Category {category} has missing deps: {missing_initialized}",
                )
            else:
                self.assertEqual(
                    len(missing_deps),
                    0,
                    f"Category {category} has missing deps: {missing_deps}",
                )

    def test_get_missing_dependencies(self):
        """Test getting list of missing dependencies"""
        deps = EnhancedCollectiveDeps()
        # FIXED: Get missing from status report
        status = deps.get_status()
        missing = [
            dep
            for cat_deps in status["missing_by_category"].values()
            for dep in cat_deps
        ]

        self.assertIsInstance(missing, list)
        self.assertGreater(len(missing), 0)
        self.assertIn("safety_validator", missing)
        self.assertIn("ltm", missing)

    def test_get_available_dependencies(self):
        """Test getting list of available dependencies"""
        deps = EnhancedCollectiveDeps(safety_validator=Mock(), ltm=Mock())
        # FIXED: Renamed method call
        available = deps.get_available_components()

        self.assertIsInstance(available, set)
        self.assertIn("safety_validator", available)
        self.assertIn("ltm", available)
        self.assertIn("metrics", available)

    def test_is_complete_empty(self):
        """Test is_complete with empty dependencies"""
        deps = EnhancedCollectiveDeps()
        self.assertFalse(deps.is_complete())

    def test_is_complete_partial(self):
        """Test is_complete with partial dependencies"""
        deps = EnhancedCollectiveDeps(safety_validator=Mock(), ltm=Mock())
        self.assertFalse(deps.is_complete())

    def test_is_complete_full(self):
        """Test is_complete with all required dependencies"""
        deps = EnhancedCollectiveDeps(
            env=Mock(),  # Added env
            safety_validator=Mock(),
            governance=Mock(),
            nso_aligner=Mock(),
            explainer=Mock(),
            ltm=Mock(),
            am=Mock(),
            compressed_memory=Mock(),
            multimodal=Mock(),
            probabilistic=Mock(),
            symbolic=Mock(),
            causal=Mock(),
            abstract=Mock(),
            cross_modal=Mock(),
            continual=Mock(),
            compositional=Mock(),
            meta_cognitive=Mock(),
            world_model=Mock(),
            goal_system=Mock(),
            resource_compute=Mock(),
            experiment_generator=Mock(),  # FIXED
            problem_executor=Mock(),  # FIXED
            # Mock all meta-reasoning components
            self_improvement_drive=Mock(),
            motivational_introspection=Mock(),
            objective_hierarchy=Mock(),
            objective_negotiator=Mock(),
            goal_conflict_detector=Mock(),
            preference_learner=Mock(),
            value_evolution_tracker=Mock(),
            ethical_boundary_monitor=Mock(),
            curiosity_reward_shaper=Mock(),
            internal_critic=Mock(),
            auto_apply_policy=Mock(),
            validation_tracker=Mock(),
            transparency_interface=Mock(),
            counterfactual_objectives=Mock(),
            distributed=Mock(),  # ADDED MISSING
        )
        self.assertTrue(deps.is_complete())


# ============================================================
# TEST: STATUS
# ============================================================


class TestStatus(unittest.TestCase):
    """Test status reporting"""

    def test_get_status(self):
        """Test get_status method"""
        deps = EnhancedCollectiveDeps()
        status = deps.get_status()

        self.assertIsInstance(status, dict)
        self.assertIn("initialized", status)
        self.assertIn("shutdown", status)
        self.assertIn("complete", status)
        self.assertIn("total_dependencies", status)
        self.assertIn("available_count", status)
        self.assertIn("missing_count", status)
        self.assertIn("available_by_category", status)
        self.assertIn("missing_by_category", status)
        # FIXED: This key doesn't exist
        # self.assertIn('metrics_available', status)
        self.assertIn("distributed_enabled", status)

    def test_status_initialized(self):
        """Test that status shows initialized"""
        deps = EnhancedCollectiveDeps()
        status = deps.get_status()
        self.assertTrue(status["initialized"])

    def test_status_not_shutdown(self):
        """Test that status shows not shutdown initially"""
        deps = EnhancedCollectiveDeps()
        status = deps.get_status()
        self.assertFalse(status["shutdown"])

    def test_status_metrics_available(self):
        """Test that metrics are shown as available"""
        deps = EnhancedCollectiveDeps()
        status = deps.get_status()
        # FIXED: Check the correct structure
        self.assertIn("metrics", status["available_by_category"]["core"])

    def test_status_distributed_disabled(self):
        """Test that distributed is shown as disabled by default"""
        deps = EnhancedCollectiveDeps()
        status = deps.get_status()
        self.assertFalse(status["distributed_enabled"])

    def test_status_distributed_enabled(self):
        """Test that distributed is shown as enabled when set"""
        deps = EnhancedCollectiveDeps(distributed=Mock())
        status = deps.get_status()
        self.assertTrue(status["distributed_enabled"])


# ============================================================
# TEST: SHUTDOWN
# ============================================================


class TestShutdown(unittest.TestCase):
    """Test shutdown functionality"""

    def test_shutdown_all(self):
        """Test shutdown_all method"""
        mock_metrics = MockMetrics()
        mock_distributed = MockComponent("distributed")
        mock_ltm = MockComponent("ltm")

        deps = EnhancedCollectiveDeps(distributed=mock_distributed, ltm=mock_ltm)
        deps.metrics = mock_metrics

        deps.shutdown_all()

        self.assertTrue(deps._shutdown)
        self.assertTrue(mock_metrics._shutdown_called)
        self.assertTrue(mock_distributed._shutdown_called)
        self.assertTrue(mock_ltm._shutdown_called)

    def test_shutdown_all_idempotent(self):
        """Test that shutdown_all can be called multiple times"""
        deps = EnhancedCollectiveDeps()

        deps.shutdown_all()
        # Should not raise exception
        deps.shutdown_all()

        self.assertTrue(deps._shutdown)

    def test_shutdown_with_missing_shutdown_method(self):
        """Test shutdown when components don't have shutdown method"""
        mock_component = Mock(spec=[])  # No shutdown method

        deps = EnhancedCollectiveDeps(safety_validator=mock_component)

        # Should not raise exception
        deps.shutdown_all()
        self.assertTrue(deps._shutdown)

    def test_shutdown_handles_errors(self):
        """Test that shutdown handles component errors gracefully"""
        mock_metrics = Mock()
        mock_metrics.shutdown.side_effect = Exception("Shutdown error")

        deps = EnhancedCollectiveDeps()
        deps.metrics = mock_metrics

        # Should not raise exception
        deps.shutdown_all()
        self.assertTrue(deps._shutdown)

    def test_destructor_calls_shutdown(self):
        """Test that destructor calls shutdown"""
        mock_metrics = MockMetrics()
        deps = EnhancedCollectiveDeps()
        deps.metrics = mock_metrics

        # Trigger destructor
        deps.__del__()

        self.assertTrue(mock_metrics._shutdown_called)


# ============================================================
# TEST: FACTORY FUNCTIONS
# ============================================================


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions"""

    def test_create_minimal_deps(self):
        """Test create_minimal_deps function"""
        deps = create_minimal_deps()

        self.assertIsNotNone(deps)
        self.assertIsInstance(deps, EnhancedCollectiveDeps)
        self.assertIsNotNone(deps.metrics)

    def test_create_full_deps_no_args(self):
        """Test create_full_deps with no arguments"""
        deps = create_full_deps()

        self.assertIsNotNone(deps)
        self.assertIsInstance(deps, EnhancedCollectiveDeps)

    def test_create_full_deps_with_env(self):
        """Test create_full_deps with environment"""
        mock_env = Mock()
        deps = create_full_deps(env=mock_env)

        self.assertEqual(deps.env, mock_env)

    def test_create_full_deps_with_components(self):
        """Test create_full_deps with component kwargs"""
        mock_safety = Mock()
        mock_ltm = Mock()

        deps = create_full_deps(safety_validator=mock_safety, ltm=mock_ltm)

        self.assertEqual(deps.safety_validator, mock_safety)
        self.assertEqual(deps.ltm, mock_ltm)

    def test_create_full_deps_unknown_component(self):
        """Test create_full_deps with unknown component"""
        # Should log warning but not fail
        deps = create_full_deps(unknown_component=Mock())

        self.assertIsNotNone(deps)
        self.assertFalse(hasattr(deps, "unknown_component"))

    def test_validate_dependencies_all_present(self):
        """Test validate_dependencies with all dependencies"""
        deps = EnhancedCollectiveDeps(
            env=Mock(),  # Added env
            safety_validator=Mock(),
            governance=Mock(),
            nso_aligner=Mock(),
            explainer=Mock(),
            ltm=Mock(),
            am=Mock(),
            compressed_memory=Mock(),
            multimodal=Mock(),
            probabilistic=Mock(),
            symbolic=Mock(),
            causal=Mock(),
            abstract=Mock(),
            cross_modal=Mock(),
            continual=Mock(),
            compositional=Mock(),
            meta_cognitive=Mock(),
            world_model=Mock(),
            goal_system=Mock(),
            resource_compute=Mock(),
            experiment_generator=Mock(),  # FIXED
            problem_executor=Mock(),  # FIXED
            distributed=Mock(),  # ADDED MISSING
        )

        result = validate_dependencies(deps)
        self.assertTrue(result)

    def test_validate_dependencies_missing(self):
        """Test validate_dependencies with missing dependencies"""
        deps = EnhancedCollectiveDeps()

        result = validate_dependencies(deps)
        self.assertFalse(result)

    def test_validate_dependencies_specific_categories(self):
        """Test validate_dependencies with specific categories"""
        deps = EnhancedCollectiveDeps(
            safety_validator=Mock(),
            governance=Mock(),
            nso_aligner=Mock(),
            explainer=Mock(),
        )

        # Should pass for SAFETY category
        result = validate_dependencies(deps, [DependencyCategory.SAFETY])
        self.assertTrue(result)

        # Should fail for MEMORY category
        result = validate_dependencies(deps, [DependencyCategory.MEMORY])
        self.assertFalse(result)


# ============================================================
# TEST: PRINT DEPENDENCY REPORT
# ============================================================


class TestPrintDependencyReport(unittest.TestCase):
    """Test print_dependency_report function"""

    def test_print_dependency_report_empty(self):
        """Test printing report for empty dependencies"""
        deps = EnhancedCollectiveDeps()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            print_dependency_report(deps)
            output = sys.stdout.getvalue()

            self.assertIn("VULCAN-AGI DEPENDENCIES REPORT", output)
            self.assertIn("Initialized:", output)
            # FIXED: Match exact text
            self.assertIn("Complete (all components loaded):", output)
        finally:
            sys.stdout = old_stdout

    def test_print_dependency_report_full(self):
        """Test printing report for full dependencies"""
        deps = EnhancedCollectiveDeps(
            safety_validator=Mock(), ltm=Mock(), multimodal=Mock()
        )

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            print_dependency_report(deps)
            output = sys.stdout.getvalue()

            self.assertIn("AVAILABLE DEPENDENCIES BY CATEGORY:", output)
            # FIXED: Match exact text
            self.assertIn("MISSING/NOT INITIALIZED DEPENDENCIES BY CATEGORY:", output)
        finally:
            sys.stdout = old_stdout


# ============================================================
# TEST: INTEGRATION
# ============================================================


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_full_lifecycle(self):
        """Test full dependency lifecycle"""
        # Create
        deps = create_minimal_deps()
        self.assertTrue(deps._initialized)

        # Add components
        deps.safety_validator = Mock()
        deps.ltm = Mock()

        # Validate
        # FIXED: Renamed method call
        available = deps.get_available_components()
        self.assertIn("safety_validator", available)

        # Get status
        status = deps.get_status()
        self.assertTrue(status["initialized"])
        self.assertFalse(status["shutdown"])

        # Shutdown
        deps.shutdown_all()
        self.assertTrue(deps._shutdown)

    def test_complex_dependency_setup(self):
        """
        Test complex dependency setup and validation
        FIXED: Verify shutdown only for components in the shutdown loop
        """
        # Create with some dependencies
        deps = create_full_deps(
            safety_validator=MockComponent("safety"),
            governance=MockComponent("governance"),
            ltm=MockComponent("ltm"),
            am=MockComponent("am"),
            multimodal=MockComponent("multimodal"),
            continual=MockComponent("continual"),
            world_model=MockComponent("world_model"),
        )

        # Check status
        status = deps.get_status()
        self.assertGreater(status["available_count"], 0)
        self.assertGreater(status["missing_count"], 0)

        # Add more components
        deps.nso_aligner = MockComponent("nso")
        deps.explainer = MockComponent("explainer")
        deps.compressed_memory = MockComponent("compressed_memory")
        deps.meta_cognitive = MockComponent("meta_cognitive")

        # Validate specific category
        result = validate_dependencies(deps, [DependencyCategory.SAFETY])
        self.assertTrue(result)

        # Shutdown
        deps.shutdown_all()

        # Verify shutdown was called on components that ARE in the shutdown loop
        # According to shutdown_all() in dependencies.py, these are shut down:
        # - metrics (always present)
        # - distributed (if present)
        # - memory systems: ltm, am, compressed_memory
        # - learning systems: continual, world_model, meta_cognitive

        # Memory systems
        self.assertTrue(deps.ltm._shutdown_called, "ltm shutdown should be called")
        self.assertTrue(deps.am._shutdown_called, "am shutdown should be called")
        self.assertTrue(
            deps.compressed_memory._shutdown_called,
            "compressed_memory shutdown should be called",
        )

        # Learning systems
        self.assertTrue(
            deps.continual._shutdown_called, "continual shutdown should be called"
        )
        self.assertTrue(
            deps.world_model._shutdown_called, "world_model shutdown should be called"
        )
        self.assertTrue(
            deps.meta_cognitive._shutdown_called,
            "meta_cognitive shutdown should be called",
        )

        # Note: The following components ARE in the shutdown loop per dependencies.py
        # - safety_validator
        # - governance
        # - nso_aligner
        # - explainer
        # - multimodal

        # FIXED: Test behavior matches source code. These *are* in the shutdown loop.
        self.assertTrue(
            deps.safety_validator._shutdown_called,
            "safety_validator *is* in the shutdown loop",
        )
        self.assertTrue(
            deps.governance._shutdown_called, "governance *is* in the shutdown loop"
        )
        self.assertTrue(
            deps.multimodal._shutdown_called, "multimodal *is* in the shutdown loop"
        )


# ============================================================
# TEST SUITE RUNNER
# ============================================================


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    test_suite.addTest(unittest.makeSuite(TestDependencyCategory))
    test_suite.addTest(unittest.makeSuite(TestEnhancedCollectiveDepsInit))
    test_suite.addTest(unittest.makeSuite(TestValidation))
    test_suite.addTest(unittest.makeSuite(TestStatus))
    test_suite.addTest(unittest.makeSuite(TestShutdown))
    test_suite.addTest(unittest.makeSuite(TestFactoryFunctions))
    test_suite.addTest(unittest.makeSuite(TestPrintDependencyReport))
    test_suite.addTest(unittest.makeSuite(TestIntegration))

    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
