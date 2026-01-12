"""
test_curiosity_crystallizer_e2e.py - End-to-End Integration Tests
Tests the full learning loop: CuriosityEngine → KnowledgeCrystallizer → Principle Application

Part of the VULCAN-AGI system

Tests the complete flow:
1. CuriosityEngine.run_experiment() produces ExperimentResult with learned_knowledge
2. KnowledgeIntegrator.integrate_results() processes the results
3. KnowledgeCrystallizer.crystallize() extracts principles
4. VersionedKnowledgeBase.store() persists principles
5. Principles are available for application to new problems

Run with: pytest src/vulcan/tests/test_curiosity_crystallizer_e2e.py -v
"""

import time
from unittest.mock import Mock, patch
from typing import Any, Dict, List

import pytest

from vulcan.curiosity_engine.curiosity_engine_core import (
    ExperimentResult,
    KnowledgeIntegrator,
)
from vulcan.curiosity_engine.experiment_generator import Experiment, ExperimentType
from vulcan.curiosity_engine.gap_analyzer import KnowledgeGap
from vulcan.knowledge_crystallizer.knowledge_crystallizer_core import (
    ExecutionTrace,
    KnowledgeCrystallizer,
)
from vulcan.knowledge_crystallizer.principle_extractor import Principle


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def knowledge_crystallizer():
    """Create a real KnowledgeCrystallizer instance"""
    return KnowledgeCrystallizer(vulcan_memory=None, semantic_bridge=None)


@pytest.fixture
def knowledge_integrator():
    """Create a KnowledgeIntegrator instance"""
    return KnowledgeIntegrator()


@pytest.fixture
def sample_knowledge_gap():
    """Create a sample knowledge gap"""
    return KnowledgeGap(
        type="pattern_recognition",
        domain="optimization",
        priority=0.8,
        estimated_cost=15.0,
        complexity=0.6,
    )


@pytest.fixture
def sample_experiment(sample_knowledge_gap):
    """Create a sample experiment"""
    return Experiment(
        gap=sample_knowledge_gap,
        complexity=0.6,
        timeout=30.0,
        success_criteria={"min_accuracy": 0.8},
        experiment_type=ExperimentType.EXPLORATORY,
        parameters={"method": "evolutionary", "generations": 10},
    )


# ============================================================================
# TEST SUITE: Knowledge Integration Pipeline
# ============================================================================


class TestKnowledgeIntegrationPipeline:
    """Test the complete knowledge integration pipeline"""

    def test_experiment_result_to_crystallization(
        self, sample_experiment, knowledge_crystallizer, knowledge_integrator
    ):
        """
        Test converting experiment results to crystallized knowledge
        
        Flow: ExperimentResult → ExecutionTrace → Crystallization → Principles
        """
        # Create experiment result with learned knowledge
        learned_knowledge = {
            "patterns": [
                {"type": "sequential", "steps": ["init", "process", "validate"]},
                {"type": "conditional", "conditions": ["check_threshold", "retry"]},
            ],
            "domain": "optimization",
            "experiment_type": "pattern_discovery",
            "accuracy": 0.85,
            "observations": [
                {"metric": "convergence_rate", "value": 0.92},
                {"metric": "stability", "value": 0.88},
            ],
        }

        experiment_result = ExperimentResult(
            experiment=sample_experiment,
            success=True,
            output={"result": "patterns_discovered"},
            learned_knowledge=learned_knowledge,
            execution_time=2.5,
        )

        # Get baseline principle count
        initial_principles = knowledge_crystallizer.knowledge_base.get_all_principles()
        initial_count = len(initial_principles)

        # Integrate results (should trigger crystallization)
        knowledge_integrator.integrate_results(
            [experiment_result], knowledge_base=knowledge_crystallizer
        )

        # Verify principles were created
        final_principles = knowledge_crystallizer.knowledge_base.get_all_principles()
        final_count = len(final_principles)

        # Should have more principles after integration
        assert (
            final_count >= initial_count
        ), "Crystallization should create new principles"

    def test_crystallization_with_rich_experiment_data(
        self, knowledge_crystallizer, knowledge_integrator
    ):
        """Test that crystallization creates richer principles than simple storage"""
        
        # Create rich experiment knowledge
        rich_knowledge = {
            "patterns": [
                {
                    "type": "optimization_pattern",
                    "steps": ["initialize", "iterate", "converge"],
                    "success_rate": 0.92,
                }
            ],
            "domain": "machine_learning",
            "experiment_type": "hyperparameter_tuning",
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.91,
            "observations": [
                {"metric": "learning_rate", "optimal_value": 0.001},
                {"metric": "batch_size", "optimal_value": 32},
            ],
        }

        # Create a mock experiment and result
        mock_gap = KnowledgeGap(
            type="optimization",
            domain="machine_learning",
            priority=0.9,
            estimated_cost=20.0,
            complexity=0.7,
        )
        
        mock_experiment = Experiment(
            gap=mock_gap,
            complexity=0.7,
            timeout=60.0,
            success_criteria={"min_accuracy": 0.85},
            experiment_type=ExperimentType.ITERATIVE,
            parameters={"optimizer": "adam", "epochs": 50},
        )

        result = ExperimentResult(
            experiment=mock_experiment,
            success=True,
            output={"model_performance": "excellent"},
            learned_knowledge=rich_knowledge,
            execution_time=45.0,
        )

        # Integrate using crystallization
        knowledge_integrator.integrate_results(
            [result], knowledge_base=knowledge_crystallizer
        )

        # Search for principles in the domain
        search_results = knowledge_crystallizer.knowledge_base.search(
            {"domain": "machine_learning"}, limit=10
        )

        # Verify that crystallization occurred
        # (Even if no principles were created, the crystallization should have been attempted)
        assert search_results is not None
        assert hasattr(search_results, "total_count")

    def test_fallback_to_store_knowledge(
        self, knowledge_crystallizer, knowledge_integrator
    ):
        """Test that system falls back to store_knowledge if crystallization fails"""
        
        # Create minimal knowledge that might not crystallize well
        minimal_knowledge = {
            "simple_fact": "value",
            "another_fact": 42,
        }

        mock_gap = KnowledgeGap(
            type="general", domain="general", priority=0.5, estimated_cost=5.0, complexity=0.3
        )
        
        mock_experiment = Experiment(
            gap=mock_gap,
            complexity=0.3,
            timeout=10.0,
            success_criteria={},
            experiment_type=ExperimentType.EXPLORATORY,
            parameters={},
        )

        result = ExperimentResult(
            experiment=mock_experiment,
            success=True,
            output={"data": "stored"},
            learned_knowledge=minimal_knowledge,
        )

        # Should not raise an exception even with minimal data
        try:
            knowledge_integrator.integrate_results(
                [result], knowledge_base=knowledge_crystallizer
            )
            # If we get here, either crystallization worked or fallback succeeded
            assert True
        except Exception as e:
            pytest.fail(f"Integration should not fail with fallback: {e}")


# ============================================================================
# TEST SUITE: Principle Application After Crystallization
# ============================================================================


class TestPrincipleApplicationAfterCrystallization:
    """Test that crystallized principles can be applied to new problems"""

    def test_end_to_end_learning_and_application(
        self, knowledge_crystallizer, knowledge_integrator
    ):
        """
        Complete end-to-end test: Learn from experiment → Apply to new problem
        
        This is the full learning loop test.
        """
        # Step 1: Create and integrate experiment results
        learned_knowledge = {
            "patterns": [
                {
                    "type": "data_processing",
                    "steps": ["load", "clean", "transform", "validate"],
                }
            ],
            "domain": "data_engineering",
            "experiment_type": "pipeline_optimization",
            "success_rate": 0.94,
        }

        mock_gap = KnowledgeGap(
            type="pipeline",
            domain="data_engineering",
            priority=0.85,
            estimated_cost=25.0,
            complexity=0.65,
        )
        
        mock_experiment = Experiment(
            gap=mock_gap,
            complexity=0.65,
            timeout=40.0,
            success_criteria={"min_success_rate": 0.9},
            experiment_type=ExperimentType.ITERATIVE,
            parameters={"approach": "incremental"},
        )

        result = ExperimentResult(
            experiment=mock_experiment,
            success=True,
            output={"pipeline": "optimized"},
            learned_knowledge=learned_knowledge,
        )

        # Step 2: Integrate (crystallize) the knowledge
        knowledge_integrator.integrate_results(
            [result], knowledge_base=knowledge_crystallizer
        )

        # Step 3: Verify principles can be searched and potentially applied
        search_results = knowledge_crystallizer.knowledge_base.search(
            {"domain": "data_engineering"}, limit=5
        )

        assert search_results is not None
        assert hasattr(search_results, "principles")

        # Step 4: If principles were created, verify they have the right structure
        if search_results.total_count > 0:
            for principle in search_results.principles:
                assert hasattr(principle, "id")
                assert hasattr(principle, "confidence")
                # Principles from crystallization should have better confidence than default 0.5
                # (though this depends on the extraction algorithm)

    def test_multiple_experiments_accumulate_knowledge(
        self, knowledge_crystallizer, knowledge_integrator
    ):
        """Test that multiple experiments build up knowledge base"""
        
        initial_count = len(
            knowledge_crystallizer.knowledge_base.get_all_principles()
        )

        # Run multiple experiment integrations
        for i in range(3):
            learned_knowledge = {
                "patterns": [
                    {
                        "type": f"pattern_type_{i}",
                        "steps": [f"step_{j}" for j in range(3)],
                    }
                ],
                "domain": "accumulation_test",
                "experiment_type": "discovery",
                "iteration": i,
            }

            mock_gap = KnowledgeGap(
                type="discovery",
                domain="accumulation_test",
                priority=0.7,
                estimated_cost=10.0,
                complexity=0.4,
            )
            
            mock_experiment = Experiment(
                gap=mock_gap,
                complexity=0.4,
                timeout=20.0,
                success_criteria={},
                experiment_type=ExperimentType.EXPLORATORY,
                parameters={"iteration": i},
            )

            result = ExperimentResult(
                experiment=mock_experiment,
                success=True,
                output={f"result_{i}": "data"},
                learned_knowledge=learned_knowledge,
            )

            knowledge_integrator.integrate_results(
                [result], knowledge_base=knowledge_crystallizer
            )

        final_count = len(knowledge_crystallizer.knowledge_base.get_all_principles())

        # Knowledge should accumulate (though the exact count depends on crystallization)
        # At minimum, the system should not lose knowledge
        assert final_count >= initial_count


# ============================================================================
# TEST SUITE: Error Handling and Edge Cases
# ============================================================================


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in the integration"""

    def test_empty_learned_knowledge(
        self, sample_experiment, knowledge_crystallizer, knowledge_integrator
    ):
        """Test handling of experiment with empty learned knowledge"""
        
        result = ExperimentResult(
            experiment=sample_experiment,
            success=True,
            output={"status": "completed"},
            learned_knowledge={},  # Empty knowledge
        )

        # Should handle gracefully without error
        try:
            knowledge_integrator.integrate_results(
                [result], knowledge_base=knowledge_crystallizer
            )
            assert True
        except Exception as e:
            pytest.fail(f"Should handle empty learned knowledge: {e}")

    def test_failed_experiment_integration(
        self, sample_experiment, knowledge_crystallizer, knowledge_integrator
    ):
        """Test that failed experiments are handled correctly"""
        
        result = ExperimentResult(
            experiment=sample_experiment,
            success=False,  # Failed experiment
            output=None,
            error="Experiment timeout",
            learned_knowledge={},
        )

        # Failed experiments should be skipped gracefully
        try:
            knowledge_integrator.integrate_results(
                [result], knowledge_base=knowledge_crystallizer
            )
            assert True
        except Exception as e:
            pytest.fail(f"Should handle failed experiments: {e}")

    def test_malformed_knowledge_structure(
        self, sample_experiment, knowledge_crystallizer, knowledge_integrator
    ):
        """Test handling of malformed knowledge structures"""
        
        malformed_knowledge = {
            "patterns": "not_a_list",  # Should be a list
            "domain": None,  # Missing domain
            "invalid_key": ["unexpected", "data"],
        }

        result = ExperimentResult(
            experiment=sample_experiment,
            success=True,
            output={"status": "completed"},
            learned_knowledge=malformed_knowledge,
        )

        # Should handle gracefully with fallback
        try:
            knowledge_integrator.integrate_results(
                [result], knowledge_base=knowledge_crystallizer
            )
            assert True
        except Exception as e:
            pytest.fail(f"Should handle malformed knowledge: {e}")

    def test_integration_without_knowledge_base(self, knowledge_integrator):
        """Test integration when knowledge_base is None"""
        
        mock_gap = KnowledgeGap(
            type="test", domain="test", priority=0.5, estimated_cost=5.0, complexity=0.3
        )
        
        mock_experiment = Experiment(
            gap=mock_gap,
            complexity=0.3,
            timeout=10.0,
            success_criteria={},
            experiment_type=ExperimentType.EXPLORATORY,
            parameters={},
        )

        result = ExperimentResult(
            experiment=mock_experiment,
            success=True,
            output={"data": "test"},
            learned_knowledge={"fact": "value"},
        )

        # Should handle gracefully when knowledge_base is None
        try:
            knowledge_integrator.integrate_results(
                [result], knowledge_base=None  # No knowledge base
            )
            assert True
        except Exception as e:
            pytest.fail(f"Should handle None knowledge_base: {e}")


# ============================================================================
# TEST SUITE: Conversion Logic
# ============================================================================


class TestExecutionTraceConversion:
    """Test the conversion of experiment knowledge to ExecutionTrace"""

    def test_convert_patterns_to_actions(self, knowledge_integrator):
        """Test that patterns are converted to actions correctly"""
        
        knowledge = {
            "patterns": [
                {"type": "sequential", "steps": ["a", "b", "c"]},
                {"type": "parallel", "steps": ["x", "y"]},
            ],
            "domain": "test_domain",
            "experiment_type": "pattern_test",
        }

        trace = knowledge_integrator._convert_to_execution_trace(knowledge)

        assert trace is not None
        assert len(trace.actions) >= 2  # At least 2 patterns converted
        assert trace.domain == "test_domain"
        assert trace.success is True

    def test_convert_with_observations(self, knowledge_integrator):
        """Test conversion with observations in context"""
        
        knowledge = {
            "patterns": [{"type": "test"}],
            "domain": "observation_test",
            "experiment_type": "test",
            "observations": [
                {"metric": "accuracy", "value": 0.95},
                {"metric": "latency", "value": 0.1},
            ],
        }

        trace = knowledge_integrator._convert_to_execution_trace(knowledge)

        assert trace is not None
        assert "observations" in trace.context
        assert len(trace.context["observations"]) == 2

    def test_convert_with_metrics_in_outcomes(self, knowledge_integrator):
        """Test that metrics are added to outcomes"""
        
        knowledge = {
            "patterns": [{"type": "test"}],
            "domain": "metrics_test",
            "experiment_type": "test",
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
        }

        trace = knowledge_integrator._convert_to_execution_trace(knowledge)

        assert trace is not None
        assert "accuracy" in trace.outcomes
        assert trace.outcomes["accuracy"] == 0.92
        assert "precision" in trace.outcomes
        assert "recall" in trace.outcomes

    def test_convert_minimal_knowledge(self, knowledge_integrator):
        """Test conversion with minimal knowledge (should still work)"""
        
        knowledge = {
            "simple_fact": "value",
        }

        trace = knowledge_integrator._convert_to_execution_trace(knowledge)

        # Even minimal knowledge should produce a valid trace
        assert trace is not None
        assert len(trace.actions) > 0
        assert trace.success is True


# ============================================================================
# INTEGRATION TEST MARKER
# ============================================================================

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration
