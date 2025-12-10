"""
test_curiosity_engine_integration.py - Full integration tests for Curiosity Engine
Part of the VULCAN-AGI system test suite

Tests the complete system working together:
- GapAnalyzer -> DependencyGraph -> ExperimentGenerator -> CuriosityEngine
"""

import threading
import time
from unittest.mock import Mock

import pytest

from vulcan.curiosity_engine.curiosity_engine_core import (
    CuriosityEngine, ExplorationFrontier)
from vulcan.curiosity_engine.dependency_graph import (
    CycleAwareDependencyGraph, DependencyAnalyzer, ROICalculator)
from vulcan.curiosity_engine.experiment_generator import (
    Experiment, ExperimentGenerator, ExperimentType, FailureType, IterativeExperimentDesigner)
from vulcan.curiosity_engine.exploration_budget import (CostEstimator,
                                                        DynamicBudget,
                                                        ResourceMonitor)
from vulcan.curiosity_engine.gap_analyzer import (GapAnalyzer, KnowledgeGap)


class TestGapAnalyzerToGraph:
    """Test integration between GapAnalyzer and DependencyGraph"""

    def test_gaps_to_dependency_graph(self):
        """Test adding gaps to dependency graph"""
        # Create gap analyzer and generate gaps
        analyzer = GapAnalyzer(min_frequency=0.1)

        # Record failures
        for i in range(10):
            analyzer.record_failure(
                "decomposition",
                {
                    "pattern": "hierarchical",
                    "domain": "planning",
                    "complexity": 0.7,
                    "missing_concepts": ["goal_decomposition"],
                },
            )

        gaps = analyzer.analyze_decomposition_failures()

        # Create dependency graph
        graph = CycleAwareDependencyGraph()

        # Add gaps to graph
        for gap in gaps:
            graph.add_node(gap)

        # Verify gaps were added
        assert graph.storage.node_count() >= len(gaps)

    def test_dependency_analysis_on_gaps(self):
        """Test analyzing dependencies between gaps"""
        # Create gaps
        gap1 = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            gap_id="gap_decomp",
        )

        gap2 = KnowledgeGap(
            type="semantic",
            domain="planning",
            priority=0.6,
            estimated_cost=15.0,
            gap_id="gap_semantic",
            metadata={"concept": "goal_decomposition"},
        )

        # Create dependency analyzer
        dep_analyzer = DependencyAnalyzer()

        # Find dependencies
        gap1.metadata["missing_concepts"] = ["goal_decomposition"]
        dependencies = dep_analyzer.find_dependencies(gap1)

        # Should find semantic dependencies
        assert isinstance(dependencies, list)

    def test_cycle_detection_in_gap_dependencies(self):
        """Test cycle detection when adding gap dependencies"""
        graph = CycleAwareDependencyGraph()

        gap1 = KnowledgeGap(
            type="decomposition",
            domain="A",
            priority=0.8,
            estimated_cost=20.0,
            gap_id="gap_a",
        )

        gap2 = KnowledgeGap(
            type="causal", domain="B", priority=0.7, estimated_cost=15.0, gap_id="gap_b"
        )

        gap3 = KnowledgeGap(
            type="transfer",
            domain="C",
            priority=0.6,
            estimated_cost=10.0,
            gap_id="gap_c",
        )

        # Add nodes
        graph.add_node(gap1)
        graph.add_node(gap2)
        graph.add_node(gap3)

        # Add edges that would create a cycle
        graph.add_edge(gap1, gap2)
        graph.add_edge(gap2, gap3)

        # This should detect potential cycle
        would_cycle = graph.would_create_cycle(gap3, gap1)
        assert would_cycle is True


class TestGraphToExperimentGenerator:
    """Test integration between DependencyGraph and ExperimentGenerator"""

    def test_prioritized_gaps_to_experiments(self):
        """Test generating experiments from prioritized gaps"""
        # Create gaps with dependencies
        gap1 = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            complexity=0.7,
        )

        gap2 = KnowledgeGap(
            type="causal",
            domain="physics",
            priority=0.9,
            estimated_cost=30.0,
            complexity=0.8,
        )

        # Create dependency graph
        graph = CycleAwareDependencyGraph()
        graph.add_node(gap1)
        graph.add_node(gap2)

        # Calculate ROI
        roi_calc = ROICalculator()
        gap1.adjusted_roi = roi_calc.get_adjusted_roi(gap1, graph)
        gap2.adjusted_roi = roi_calc.get_adjusted_roi(gap2, graph)

        # Generate experiments
        generator = ExperimentGenerator()

        experiments1 = generator.generate_for_gap(gap1)
        experiments2 = generator.generate_for_gap(gap2)

        assert len(experiments1) > 0
        assert len(experiments2) > 0
        assert all(isinstance(e, Experiment) for e in experiments1)
        assert all(isinstance(e, Experiment) for e in experiments2)

    def test_topological_order_experiment_generation(self):
        """Test generating experiments in topological order"""
        # Create gaps with clear dependencies
        gap1 = KnowledgeGap(
            type="semantic",
            domain="planning",
            priority=0.5,
            estimated_cost=10.0,
            gap_id="gap_sem",
        )

        gap2 = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            gap_id="gap_dec",
        )

        # Create dependency graph
        graph = CycleAwareDependencyGraph()
        graph.add_node(gap1)
        graph.add_node(gap2)
        graph.add_edge(gap2, gap1)  # gap2 depends on gap1

        # Get topological order
        sorted_gaps = graph.topological_sort()

        # Generate experiments in order
        generator = ExperimentGenerator()
        experiments = []

        for gap in sorted_gaps:
            gap_experiments = generator.generate_for_gap(gap)
            experiments.extend(gap_experiments)

        assert len(experiments) > 0


class TestExperimentGeneratorToBudget:
    """Test integration between ExperimentGenerator and Budget management"""

    def test_experiment_cost_estimation(self):
        """Test cost estimation for generated experiments"""
        gap = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            complexity=0.7,
        )

        # Generate experiments
        generator = ExperimentGenerator()
        experiments = generator.generate_for_gap(gap)

        # Estimate costs
        cost_estimator = CostEstimator()

        for experiment in experiments:
            cost = cost_estimator.estimate_experiment_cost(experiment)
            assert cost > 0
            assert isinstance(cost, float)

    def test_budget_constrained_experiment_selection(self):
        """Test selecting experiments within budget"""
        # FIX: Start with a larger budget to ensure some experiments can be selected
        budget = DynamicBudget(base_allocation=200.0)

        gap = KnowledgeGap(
            type="causal",
            domain="physics",
            priority=0.9,
            estimated_cost=30.0,
            complexity=0.8,
        )

        # Generate experiments
        generator = ExperimentGenerator()
        experiments = generator.generate_for_gap(gap)

        # Estimate costs
        cost_estimator = CostEstimator()

        # Select experiments within budget
        selected = []
        for experiment in experiments:
            cost = cost_estimator.estimate_experiment_cost(experiment)
            if budget.can_afford(cost):
                success = budget.consume(cost)
                if success:
                    selected.append(experiment)

        # FIX: More lenient assertion - either we selected some or we had none to select
        assert len(selected) >= 0
        assert budget.get_available() >= 0

        # If we generated experiments, we should have been able to select at least one
        if len(experiments) > 0:
            assert len(selected) > 0, (
                f"Should have selected at least one experiment from {len(experiments)} available with budget 200.0"
            )


class TestBudgetToResourceMonitor:
    """Test integration between Budget and ResourceMonitor"""

    def test_budget_adjustment_based_on_resources(self):
        """Test adjusting budget based on resource availability"""
        budget = DynamicBudget(base_allocation=100.0)
        monitor = ResourceMonitor(sampling_interval=0.1)

        # Get current load
        load = monitor.get_current_load()

        # Adjust budget based on load
        budget.adjust_for_load(load)

        # Budget should still be valid
        assert budget.get_available() > 0

    def test_resource_prediction_for_experiments(self):
        """Test predicting resource needs for experiments"""
        monitor = ResourceMonitor()

        # Get current snapshot
        snapshot = monitor.get_resource_snapshot()

        assert snapshot.cpu_percent >= 0
        assert snapshot.memory_percent >= 0

        # Predict future load
        predicted_load, confidence = monitor.predict_future_load(horizon_minutes=5)

        assert 0 <= predicted_load <= 1.0
        assert 0 <= confidence <= 1.0


class TestFullCuriosityEnginePipeline:
    """Test complete Curiosity Engine pipeline"""

    def test_end_to_end_learning_cycle(self):
        """Test complete learning cycle from gap identification to execution"""
        # Initialize engine with mocked components
        knowledge_mock = Mock()
        decomposer_mock = Mock()
        world_model_mock = Mock()

        engine = CuriosityEngine(
            knowledge=knowledge_mock,
            decomposer=decomposer_mock,
            world_model=world_model_mock,
        )

        # Record some failures
        engine.gap_analyzer.record_failure(
            "decomposition",
            {
                "pattern": "hierarchical",
                "domain": "planning",
                "complexity": 0.7,
                "missing_concepts": ["goal_decomposition"],
            },
        )

        for i in range(5):
            engine.gap_analyzer.record_failure(
                "prediction",
                {
                    "cause": "temperature",
                    "effect": "pressure",
                    "magnitude": 0.4,
                    "domain": "physics",
                    "variables": ["T", "P"],
                },
            )

        # Run learning cycle
        result = engine.run_learning_cycle(max_experiments=3)

        # Verify results
        assert isinstance(result, dict)
        assert "strategy_used" in result
        assert "gaps_identified" in result
        assert "experiments_run" in result
        assert result["experiments_run"] >= 0

    def test_strategy_selection_adaptation(self):
        """Test that engine adapts strategy based on context"""
        engine = CuriosityEngine()

        # Test different contexts
        contexts = [
            {"resource_load": 0.2, "available_budget": 100},  # Should use comprehensive
            {"resource_load": 0.9, "available_budget": 100},  # Should use minimal
            {"resource_load": 0.5, "available_budget": 5},  # Should use efficient
        ]

        for context in contexts:
            strategy = engine.select_exploration_strategy(context)
            assert strategy in [
                "gap_driven",
                "minimal",
                "efficient",
                "balanced",
                "comprehensive",
            ]

    def test_gap_identification_with_cycle_detection(self):
        """Test gap identification with dependency cycle detection"""
        engine = CuriosityEngine()

        # Record various failures
        for i in range(10):
            engine.gap_analyzer.record_failure(
                "decomposition",
                {"pattern": "hierarchical", "domain": "planning", "complexity": 0.7},
            )

        # Identify gaps with cycle detection
        gaps = engine.identify_gaps_with_cycle_detection()

        assert isinstance(gaps, list)
        # Should handle cycles if any
        assert not engine.gap_graph.has_cycles()

    def test_experiment_execution_with_safety(self):
        """Test safe experiment execution"""
        engine = CuriosityEngine()

        gap = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            complexity=0.6,
        )

        # Generate experiment
        experiments = engine.experiment_generator.generate_for_gap(gap)

        if experiments:
            experiment = experiments[0]

            # Execute safely
            result = engine.run_experiment_sandboxed(experiment)

            assert result is not None
            assert hasattr(result, "success")
            assert hasattr(result, "experiment")

    def test_knowledge_integration_after_experiments(self):
        """Test integrating results back into knowledge systems"""
        knowledge_mock = Mock()
        knowledge_mock.store_knowledge = Mock()

        engine = CuriosityEngine(knowledge=knowledge_mock)

        gap = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            complexity=0.5,
        )

        experiment = Experiment(
            gap=gap,
            complexity=0.5,
            timeout=30.0,
            success_criteria={"min_accuracy": 0.7},
            experiment_type=ExperimentType.DECOMPOSITION,
        )

        # Run experiment
        result = engine.run_experiment_sandboxed(experiment)

        # Update from results
        engine.update_from_experiment_results([result])

        # Verify update was called
        assert engine.learning_rate >= 0


class TestExplorationFrontier:
    """Test ExplorationFrontier integration"""

    def test_frontier_tracking_with_experiments(self):
        """Test tracking exploration frontier during experiments"""
        frontier = ExplorationFrontier()

        # Add explored regions
        region1 = frontier.add_explored_region(
            "planning", {"hierarchical", "goal_oriented"}
        )
        frontier.add_explored_region("planning", {"temporal", "scheduling"})
        frontier.add_explored_region("physics", {"causal", "predictive"})

        # Get frontier regions
        frontier_regions = frontier.get_unexplored_neighbors()

        assert isinstance(frontier_regions, list)

    def test_exploration_value_estimation(self):
        """Test estimating value of exploring regions"""
        frontier = ExplorationFrontier()

        # Add explored regions
        frontier.add_explored_region("domain_a", {"pattern1", "pattern2"})
        frontier.add_explored_region("domain_b", {"pattern3", "pattern4"})

        # Get all regions
        regions = frontier.region_manager.get_all_regions()

        # Estimate values
        for region_id, region in regions.items():
            value = frontier.estimate_exploration_value(region)
            assert 0 <= value <= 1.0

    def test_frontier_update_with_new_knowledge(self):
        """Test updating frontier with newly learned knowledge"""
        frontier = ExplorationFrontier()

        # Initial exploration
        frontier.add_explored_region("planning", {"hierarchical"})

        # New knowledge discovered
        new_knowledge = {
            "planning": {"patterns": ["temporal", "resource_allocation"], "value": 0.8}
        }

        # Update frontier
        frontier.update_frontier(new_knowledge)

        # Verify update
        regions = frontier.region_manager.get_all_regions()
        assert len(regions) > 0


class TestIterativeExperimentDesign:
    """Test iterative experiment design integration"""

    def test_failure_analysis_and_adaptation(self):
        """Test analyzing failures and adapting experiments"""
        designer = IterativeExperimentDesigner()

        gap = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            complexity=0.7,
        )

        # Generate iterative experiments
        experiments = designer.generate_iterative_experiments(gap, max_iterations=3)

        assert len(experiments) == 3
        assert all(isinstance(e, Experiment) for e in experiments)

        # Simulate failure
        failure_result = {"success": False, "error": "timeout", "data": {}}

        # Analyze failure
        analysis = designer.analyze_failure(failure_result, experiments[0])

        assert analysis.type in FailureType.__members__.values()
        assert len(analysis.suggested_adjustments) > 0

    def test_experiment_pivoting_on_failure(self):
        """Test pivoting experiment strategy after failure"""
        designer = IterativeExperimentDesigner()

        gap = KnowledgeGap(
            type="causal", domain="physics", priority=0.9, estimated_cost=30.0
        )

        experiment = Experiment(
            gap=gap,
            complexity=0.8,
            timeout=30.0,
            success_criteria={"causal_strength": 0.7},
            experiment_type=ExperimentType.CAUSAL,
        )

        # Analyze failure
        failure_result = {"success": False, "error": "wrong_approach", "accuracy": 0.2}

        analysis = designer.analyze_failure(failure_result, experiment)

        # Pivot experiment
        pivoted = designer.pivot_experiment_strategy(experiment, analysis)

        assert pivoted.iteration == experiment.iteration + 1
        assert (
            pivoted.experiment_type != experiment.experiment_type
            or pivoted.parameters != experiment.parameters
        )


class TestConcurrentOperations:
    """Test concurrent operations across all components"""

    def test_concurrent_gap_analysis(self):
        """Test concurrent gap analysis"""
        analyzer = GapAnalyzer()

        def record_failures():
            for i in range(50):
                analyzer.record_failure(
                    "decomposition",
                    {"pattern": "test", "domain": "test", "complexity": 0.5},
                )

        threads = [threading.Thread(target=record_failures) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Analyze gaps
        gaps = analyzer.get_all_gaps()

        # Should handle concurrent access
        assert isinstance(gaps, list)

    def test_concurrent_experiment_generation(self):
        """Test concurrent experiment generation"""
        generator = ExperimentGenerator()

        gaps = [
            KnowledgeGap(
                type="decomposition",
                domain=f"domain_{i}",
                priority=0.7,
                estimated_cost=20.0,
                complexity=0.6,
            )
            for i in range(5)
        ]

        def generate_experiments(gap):
            return generator.generate_for_gap(gap)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(generate_experiments, gaps))

        assert len(results) == 5
        assert all(isinstance(exps, list) for exps in results)

    def test_concurrent_budget_operations(self):
        """Test concurrent budget operations"""
        budget = DynamicBudget(base_allocation=100.0)

        def consume_budget():
            for _ in range(10):
                if budget.can_afford(5.0):
                    budget.consume(5.0)
                time.sleep(0.01)

        threads = [threading.Thread(target=consume_budget) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Budget should never go negative
        assert budget.get_available() >= 0


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""

    def test_malformed_gap_handling(self):
        """Test handling malformed gap data"""
        analyzer = GapAnalyzer()

        # Record malformed failures
        analyzer.record_failure("decomposition", None)
        analyzer.record_failure("decomposition", {})
        analyzer.record_failure("decomposition", {"invalid": "data"})

        # Should not crash
        gaps = analyzer.analyze_decomposition_failures()
        assert isinstance(gaps, list)

    def test_experiment_failure_recovery(self):
        """Test recovering from experiment failures"""
        engine = CuriosityEngine()

        gap = KnowledgeGap(
            type="decomposition",
            domain="planning",
            priority=0.8,
            estimated_cost=20.0,
            complexity=0.7,
        )

        # Generate failing experiment
        experiment = Experiment(
            gap=gap,
            complexity=0.8,
            timeout=0.001,  # Very short timeout
            success_criteria={"min_accuracy": 0.99},  # Very high threshold
            experiment_type=ExperimentType.DECOMPOSITION,
        )

        # Execute - should handle failure gracefully
        result = engine.run_experiment_sandboxed(experiment)

        assert result is not None
        # Should not crash on failure
        engine.update_from_experiment_results([result])

    def test_budget_exhaustion_handling(self):
        """Test handling budget exhaustion"""
        budget = DynamicBudget(base_allocation=10.0)

        # Exhaust budget
        for _ in range(5):
            budget.consume(3.0)

        # Try to consume more
        result = budget.consume(10.0)

        assert result is False
        assert budget.get_available() >= 0

    def test_cycle_breaking_recovery(self):
        """Test recovering from dependency cycles"""
        graph = CycleAwareDependencyGraph()

        gap1 = KnowledgeGap(
            type="decomposition",
            domain="A",
            priority=0.8,
            estimated_cost=20.0,
            gap_id="gap_a",
        )

        gap2 = KnowledgeGap(
            type="causal", domain="B", priority=0.7, estimated_cost=15.0, gap_id="gap_b"
        )

        gap3 = KnowledgeGap(
            type="transfer",
            domain="C",
            priority=0.6,
            estimated_cost=10.0,
            gap_id="gap_c",
        )

        # Add nodes and edges
        graph.add_node(gap1)
        graph.add_node(gap2)
        graph.add_node(gap3)

        # Create cycle using weak edges
        graph.add_edge(gap1, gap2)
        graph.add_edge(gap2, gap3)
        graph.add_weak_edge(gap3, gap1)  # Weak edge to create cycle

        # Break cycles
        if graph.has_cycles():
            broken = graph.break_cycles_minimum_cost()
            assert len(broken) > 0

        # Should be acyclic now
        assert not graph.has_cycles()


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics"""

    def test_large_gap_set_handling(self):
        """Test handling large numbers of gaps"""
        analyzer = GapAnalyzer()

        # Create many gaps
        for i in range(100):
            analyzer.record_failure(
                "decomposition",
                {
                    "pattern": f"pattern_{i % 10}",
                    "domain": f"domain_{i % 5}",
                    "complexity": 0.5 + (i % 10) * 0.05,
                },
            )

        start_time = time.time()
        gaps = analyzer.get_all_gaps()
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 5.0
        assert isinstance(gaps, list)

    def test_experiment_generation_performance(self):
        """Test experiment generation performance"""
        generator = ExperimentGenerator()

        gaps = [
            KnowledgeGap(
                type="decomposition",
                domain=f"domain_{i}",
                priority=0.7,
                estimated_cost=20.0,
                complexity=0.6,
            )
            for i in range(50)
        ]

        start_time = time.time()

        for gap in gaps:
            experiments = generator.generate_for_gap(gap)
            assert len(experiments) > 0

        elapsed = time.time() - start_time

        # Should generate experiments efficiently
        assert elapsed < 10.0

    def test_dependency_graph_scaling(self):
        """Test dependency graph with many nodes"""
        graph = CycleAwareDependencyGraph()

        # Add many gaps
        gaps = [
            KnowledgeGap(
                type="decomposition",
                domain=f"domain_{i}",
                priority=0.7,
                estimated_cost=20.0,
                gap_id=f"gap_{i}",
            )
            for i in range(100)
        ]

        start_time = time.time()

        for gap in gaps:
            graph.add_node(gap)

        # Add some edges
        for i in range(0, len(gaps) - 1, 2):
            graph.add_edge(gaps[i], gaps[i + 1])

        elapsed = time.time() - start_time

        # Should handle many nodes efficiently
        assert elapsed < 5.0
        assert graph.storage.node_count() == 100


class TestSystemConfiguration:
    """Test system configuration and initialization"""

    def test_curiosity_engine_initialization_variants(self):
        """Test different initialization configurations"""
        # Minimal initialization
        engine1 = CuriosityEngine()
        assert engine1 is not None

        # With components
        knowledge_mock = Mock()
        decomposer_mock = Mock()
        world_model_mock = Mock()

        engine2 = CuriosityEngine(
            knowledge=knowledge_mock,
            decomposer=decomposer_mock,
            world_model=world_model_mock,
        )
        assert engine2.knowledge is not None
        assert engine2.decomposer is not None
        assert engine2.world_model is not None

    def test_component_configuration(self):
        """Test configuring individual components"""
        # Configure gap analyzer
        analyzer = GapAnalyzer(
            anomaly_threshold=0.3, min_frequency=0.2, max_history=5000
        )
        assert analyzer.anomaly_threshold == 0.3

        # Configure experiment generator
        generator = ExperimentGenerator(default_timeout=60.0, max_complexity=0.9)
        assert generator.default_timeout == 60.0

        # Configure budget
        budget = DynamicBudget(base_allocation=200.0, enable_recovery=True)
        assert budget.base_allocation == 200.0

    def test_system_statistics_collection(self):
        """Test collecting statistics from all components"""
        engine = CuriosityEngine()

        # Record some activity
        engine.gap_analyzer.record_failure(
            "decomposition", {"pattern": "test", "domain": "test", "complexity": 0.5}
        )

        # Get statistics from components
        gap_stats = engine.gap_analyzer.get_statistics()
        graph_stats = engine.gap_graph.get_statistics()
        budget_stats = engine.exploration_budget.get_statistics()

        assert isinstance(gap_stats, dict)
        assert isinstance(graph_stats, dict)
        assert isinstance(budget_stats, dict)


@pytest.fixture
def complete_system():
    """Fixture providing a complete, initialized system"""
    knowledge_mock = Mock()
    decomposer_mock = Mock()
    world_model_mock = Mock()

    engine = CuriosityEngine(
        knowledge=knowledge_mock,
        decomposer=decomposer_mock,
        world_model=world_model_mock,
    )

    return engine


class TestCompleteWorkflows:
    """Test complete end-to-end workflows"""

    def test_discovery_to_execution_workflow(self, complete_system):
        """Test complete workflow from discovery to execution"""
        engine = complete_system

        # Phase 1: Record observations and failures
        for i in range(10):
            engine.gap_analyzer.record_failure(
                "decomposition",
                {
                    "pattern": "hierarchical",
                    "domain": "planning",
                    "complexity": 0.6 + i * 0.02,
                    "missing_concepts": ["goal_decomposition"],
                },
            )

        # Phase 2: Identify and prioritize gaps
        gaps = engine.identify_gaps_with_cycle_detection()
        assert len(gaps) >= 0

        # Phase 3: Generate experiments
        if gaps:
            experiments = engine.generate_targeted_experiments(gaps[0])
            assert len(experiments) >= 0

            # Phase 4: Execute experiments
            if experiments:
                result = engine.run_experiment_sandboxed(experiments[0])
                assert result is not None

                # Phase 5: Integrate results
                engine.update_from_experiment_results([result])

    def test_iterative_learning_workflow(self, complete_system):
        """Test iterative learning over multiple cycles"""
        engine = complete_system

        results = []

        # Run multiple learning cycles
        for cycle in range(3):
            # Add some failures each cycle
            for i in range(5):
                engine.gap_analyzer.record_failure(
                    "prediction",
                    {
                        "cause": "x",
                        "effect": "y",
                        "magnitude": 0.4 + cycle * 0.1,
                        "domain": "physics",
                        "variables": ["x", "y"],
                    },
                )

            # Run learning cycle
            result = engine.run_learning_cycle(max_experiments=2)
            results.append(result)

        # Verify learning occurred
        assert len(results) == 3
        assert all("experiments_run" in r for r in results)

    def test_adaptive_strategy_workflow(self, complete_system):
        """Test adaptive strategy selection over time"""
        engine = complete_system

        strategies_used = []

        # Simulate different resource conditions
        for i in range(5):
            # Vary resource load
            context = {
                "resource_load": 0.2 + i * 0.15,
                "available_budget": 100 - i * 20,
                "recent_success_rate": 0.5 + i * 0.05,
            }

            strategy = engine.select_exploration_strategy(context)
            strategies_used.append(strategy)

        # Should adapt strategies
        assert len(set(strategies_used)) > 1  # Used different strategies


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])
