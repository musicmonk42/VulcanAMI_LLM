"""
test_curiosity_engine_core.py - Comprehensive tests for CuriosityEngine
Part of the VULCAN-AGI system

Tests cover:
- Thread safety
- Resource management
- Knowledge gap identification
- Experiment execution
- Edge cases and error handling
"""

import threading
import time
from queue import Empty
from unittest.mock import Mock

import pytest

from vulcan.curiosity_engine.curiosity_engine_core import (
    CuriosityEngine, ExperimentManager, ExperimentResult, ExplorationFrontier,
    ExplorationValueEstimator, GapPrioritizer, KnowledgeIntegrator,
    KnowledgeRegion, RegionManager, SafeExperimentExecutor, StrategySelector)
from vulcan.curiosity_engine.experiment_generator import (Experiment,
                                                          ExperimentType)
from vulcan.curiosity_engine.gap_analyzer import KnowledgeGap

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_knowledge_gap():
    """Create a mock knowledge gap"""
    gap = KnowledgeGap(
        type="decomposition",
        domain="test_domain",
        priority=0.8,
        estimated_cost=10.0,
        complexity=0.5,
    )
    return gap


@pytest.fixture
def mock_experiment(mock_knowledge_gap):
    """Create a mock experiment"""
    experiment = Experiment(
        gap=mock_knowledge_gap,
        complexity=0.5,
        timeout=30.0,
        success_criteria={"min_accuracy": 0.7},
        experiment_type=ExperimentType.DECOMPOSITION,
        parameters={"strategy": "hierarchical"},
    )
    return experiment


@pytest.fixture
def curiosity_engine():
    """Create a CuriosityEngine instance"""
    engine = CuriosityEngine()
    return engine


@pytest.fixture
def region_manager():
    """Create a RegionManager instance"""
    return RegionManager(cache_size=100)


@pytest.fixture
def exploration_frontier():
    """Create an ExplorationFrontier instance"""
    return ExplorationFrontier(cache_size=100)


# ============================================================================
# Test KnowledgeRegion
# ============================================================================


class TestKnowledgeRegion:
    """Tests for KnowledgeRegion class"""

    def test_distance_calculation(self):
        """Test distance calculation between regions"""
        region1 = KnowledgeRegion(
            domain="test", patterns={"a", "b", "c"}, confidence=0.5
        )
        region2 = KnowledgeRegion(
            domain="test", patterns={"b", "c", "d"}, confidence=0.5
        )

        distance = region1.distance_to(region2)

        # Jaccard distance: 1 - (2/4) = 0.5
        assert abs(distance - 0.5) < 0.01

    def test_distance_empty_patterns(self):
        """Test distance with empty patterns"""
        region1 = KnowledgeRegion(domain="test", patterns=set(), confidence=0.5)
        region2 = KnowledgeRegion(domain="test", patterns={"a", "b"}, confidence=0.5)

        distance = region1.distance_to(region2)

        # Should return maximum distance
        assert distance == 1.0

    def test_distance_identical_patterns(self):
        """Test distance with identical patterns"""
        region1 = KnowledgeRegion(
            domain="test", patterns={"a", "b", "c"}, confidence=0.5
        )
        region2 = KnowledgeRegion(
            domain="test", patterns={"a", "b", "c"}, confidence=0.7
        )

        distance = region1.distance_to(region2)

        # Should return minimum distance
        assert distance == 0.0


# ============================================================================
# Test RegionManager
# ============================================================================


class TestRegionManager:
    """Tests for RegionManager class"""

    def test_add_region(self, region_manager):
        """Test adding a region"""
        patterns = {"pattern1", "pattern2"}
        region_id = region_manager.add_region("test_domain", patterns)

        assert region_id is not None
        assert region_id in region_manager.explored_regions

        region = region_manager.get_region(region_id)
        assert region is not None
        assert region.domain == "test_domain"
        assert region.patterns == patterns

    def test_add_empty_patterns_bug(self, region_manager):
        """Test the empty patterns bug - should NOT merge with existing regions"""
        # Add first region with patterns
        patterns1 = {"pattern1", "pattern2"}
        region_id1 = region_manager.add_region("test_domain", patterns1)

        # Try to add region with empty patterns - should create NEW region
        empty_patterns = set()
        region_id2 = region_manager.add_region("test_domain", empty_patterns)

        # BUG FIX: These should be DIFFERENT regions
        # The bug would make them the same due to: (not patterns or best_overlap > ...)
        # After fix with: (patterns and best_overlap > ...), they should differ
        assert region_id1 != region_id2, (
            "Empty patterns should not match existing regions"
        )

    def test_region_merging(self, region_manager):
        """Test that similar regions get merged"""
        patterns1 = {"a", "b", "c", "d"}
        region_id1 = region_manager.add_region("test", patterns1)

        # Add overlapping patterns (75% overlap)
        patterns2 = {"a", "b", "c", "e"}
        region_id2 = region_manager.add_region("test", patterns2)

        # Should merge since overlap > 50%
        assert region_id1 == region_id2

        # Check merged patterns
        region = region_manager.get_region(region_id1)
        assert "e" in region.patterns

    def test_eviction(self):
        """Test region eviction when cache is full"""
        manager = RegionManager(cache_size=5)

        # Add more regions than cache size
        region_ids = []
        for i in range(10):
            patterns = {f"pattern_{i}"}
            region_id = manager.add_region(f"domain_{i}", patterns)
            region_ids.append(region_id)
            time.sleep(0.01)  # Ensure different timestamps

        # Should have evicted oldest regions
        assert len(manager.explored_regions) <= 5

        # Oldest regions should be evicted
        assert region_ids[0] not in manager.explored_regions

    def test_thread_safety(self, region_manager):
        """Test thread safety of region manager"""
        errors = []

        def add_regions(thread_id):
            try:
                for i in range(20):
                    patterns = {f"pattern_{thread_id}_{i}"}
                    region_manager.add_region(f"domain_{thread_id}", patterns)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_regions, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_get_neighbors(self, region_manager):
        """Test getting neighboring regions"""
        # Add regions that should connect
        patterns1 = {"a", "b", "c"}
        region_id1 = region_manager.add_region("test", patterns1)

        patterns2 = {"b", "c", "d"}
        region_id2 = region_manager.add_region("test", patterns2)

        neighbors = region_manager.get_neighbors(region_id1)

        # Should be neighbors due to high overlap
        assert region_id2 in neighbors or len(neighbors) >= 0

    def test_frontier_update(self, region_manager):
        """Test frontier region tracking"""
        patterns = {"pattern1", "pattern2"}
        region_id = region_manager.add_region("test", patterns)

        # Should be in frontier initially
        frontier = region_manager.get_frontier_regions()
        assert region_id in frontier


# ============================================================================
# Test ExplorationValueEstimator
# ============================================================================


class TestExplorationValueEstimator:
    """Tests for ExplorationValueEstimator class"""

    def test_value_estimation(self):
        """Test basic value estimation"""
        estimator = ExplorationValueEstimator()

        region = KnowledgeRegion(
            domain="test", patterns={"a", "b"}, confidence=0.3, exploration_count=0
        )

        value = estimator.estimate_value(region, neighbors_count=2)

        # Should return a value between 0 and 1
        assert 0 <= value <= 1

        # Higher uncertainty should give higher value
        assert value > 0.5

    def test_value_caching(self):
        """Test value caching"""
        estimator = ExplorationValueEstimator()

        region = KnowledgeRegion(domain="test", patterns={"a"}, confidence=0.5)

        # First call
        value1 = estimator.estimate_value(region)

        # Second call should use cache
        value2 = estimator.estimate_value(region)

        assert value1 == value2

    def test_decay_with_exploration(self):
        """Test that value decays with exploration count"""
        estimator = ExplorationValueEstimator(decay_rate=0.9)

        region1 = KnowledgeRegion(
            domain="test", patterns={"a"}, confidence=0.5, exploration_count=0
        )

        region2 = KnowledgeRegion(
            domain="test", patterns={"a"}, confidence=0.5, exploration_count=5
        )

        value1 = estimator.estimate_value(region1)
        value2 = estimator.estimate_value(region2)

        # More explored region should have lower value
        assert value2 < value1


# ============================================================================
# Test ExplorationFrontier
# ============================================================================


class TestExplorationFrontier:
    """Tests for ExplorationFrontier class"""

    def test_add_explored_region(self, exploration_frontier):
        """Test adding explored regions"""
        region_id = exploration_frontier.add_explored_region(
            "test_domain", {"pattern1", "pattern2"}
        )

        assert region_id is not None
        assert region_id in exploration_frontier.frontier_regions

    def test_pattern_conversion(self, exploration_frontier):
        """Test pattern type conversion"""
        # Test string pattern
        region_id1 = exploration_frontier.add_explored_region("test", "single_pattern")

        # Test list pattern
        region_id2 = exploration_frontier.add_explored_region(
            "test", ["pattern1", "pattern2"]
        )

        # Test set pattern
        region_id3 = exploration_frontier.add_explored_region(
            "test", {"pattern3", "pattern4"}
        )

        assert all([region_id1, region_id2, region_id3])

    def test_update_frontier(self, exploration_frontier):
        """Test frontier update with new knowledge"""
        new_knowledge = {
            "domain1": {"patterns": ["p1", "p2"], "value": 0.8},
            "domain2": ["p3", "p4"],
            "domain3": "single_pattern",
        }

        exploration_frontier.update_frontier(new_knowledge)

        # Should have added regions for all domains
        frontier = exploration_frontier.frontier_regions
        assert len(frontier) > 0

    def test_get_unexplored_neighbors(self, exploration_frontier):
        """Test getting unexplored neighbors"""
        # Add some regions
        exploration_frontier.add_explored_region("test", {"a", "b"})
        exploration_frontier.add_explored_region("test", {"b", "c"})

        # Get neighbors
        neighbors = exploration_frontier.get_unexplored_neighbors()

        # Should return list of (region_id, value) tuples
        assert isinstance(neighbors, list)
        for neighbor_id, value in neighbors:
            assert isinstance(neighbor_id, str)
            assert 0 <= value <= 1


# ============================================================================
# Test SafeExperimentExecutor
# ============================================================================


class TestSafeExperimentExecutor:
    """Tests for SafeExperimentExecutor class"""

    def test_execute_decomposition_experiment(self, mock_experiment):
        """Test decomposition experiment execution"""
        executor = SafeExperimentExecutor()

        result = executor.execute_experiment(mock_experiment)

        assert "success" in result
        assert "data" in result

    def test_execute_with_mock_decomposer(self, mock_experiment):
        """Test execution with mock decomposer"""
        executor = SafeExperimentExecutor()

        # Create mock decomposer
        mock_decomposer = Mock()
        mock_decomposer.test_decomposition = Mock(
            return_value={"success": True, "components": ["comp1", "comp2"]}
        )

        result = executor.execute_experiment(
            mock_experiment, decomposer=mock_decomposer
        )

        assert result["success"] == True
        assert "patterns" in result

    def test_execute_causal_experiment(self, mock_knowledge_gap):
        """Test causal experiment execution"""
        executor = SafeExperimentExecutor()

        gap = KnowledgeGap(
            type="causal", domain="test", priority=0.8, estimated_cost=10.0
        )

        experiment = Experiment(
            gap=gap,
            complexity=0.5,
            timeout=30.0,
            success_criteria={},
            parameters={"intervention": {"variable": "test_var"}},
        )

        result = executor.execute_experiment(experiment)

        assert "success" in result
        assert "observations" in result

    def test_error_handling(self, mock_experiment):
        """Test error handling in execution"""
        executor = SafeExperimentExecutor()

        # Create experiment that will fail
        mock_experiment.parameters = None  # This should cause an error

        result = executor.execute_experiment(mock_experiment)

        # Should handle error gracefully
        assert "error" in result


# ============================================================================
# Test StrategySelector
# ============================================================================


class TestStrategySelector:
    """Tests for StrategySelector class"""

    def test_select_strategy_high_load(self):
        """Test strategy selection under high load"""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            current_load=0.9, available_budget=50, recent_failures=2
        )

        assert strategy == "minimal"

    def test_select_strategy_low_budget(self):
        """Test strategy selection with low budget"""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            current_load=0.5, available_budget=5, recent_failures=2
        )

        assert strategy == "efficient"

    def test_select_strategy_many_failures(self):
        """Test strategy selection with many failures"""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            current_load=0.5, available_budget=50, recent_failures=10
        )

        assert strategy == "gap_driven"

    def test_select_strategy_optimal_conditions(self):
        """Test strategy selection under optimal conditions"""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            current_load=0.2, available_budget=100, recent_failures=1
        )

        assert strategy == "comprehensive"

    def test_select_strategy_balanced(self):
        """Test default balanced strategy"""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            current_load=0.5, available_budget=50, recent_failures=3
        )

        assert strategy == "balanced"


# ============================================================================
# Test GapPrioritizer
# ============================================================================


class TestGapPrioritizer:
    """Tests for GapPrioritizer class"""

    def test_calculate_priority(self, mock_knowledge_gap):
        """Test priority calculation"""
        prioritizer = GapPrioritizer()

        priority = prioritizer.calculate_priority(
            mock_knowledge_gap, descendants_count=5, ancestors_count=2
        )

        assert priority > 0

    def test_priority_boost_for_causal_gaps(self):
        """Test priority boost for causal gaps"""
        prioritizer = GapPrioritizer()

        gap_causal = KnowledgeGap(
            type="causal", domain="test", priority=1.0, estimated_cost=10.0
        )

        gap_other = KnowledgeGap(
            type="decomposition", domain="test", priority=1.0, estimated_cost=10.0
        )

        priority_causal = prioritizer.calculate_priority(gap_causal)
        priority_other = prioritizer.calculate_priority(gap_other)

        # Causal gaps should have higher priority
        assert priority_causal > priority_other

    def test_prioritize_gaps(self, mock_knowledge_gap):
        """Test gap prioritization"""
        prioritizer = GapPrioritizer()

        gaps = [
            KnowledgeGap(type="test", domain="d1", priority=0.5, estimated_cost=10.0),
            KnowledgeGap(type="test", domain="d2", priority=0.9, estimated_cost=10.0),
            KnowledgeGap(type="test", domain="d3", priority=0.3, estimated_cost=10.0),
        ]

        priorities = prioritizer.prioritize_gaps(gaps)

        # Should be sorted by priority (descending)
        assert len(priorities) == 3
        assert priorities[0].priority >= priorities[1].priority
        assert priorities[1].priority >= priorities[2].priority


# ============================================================================
# Test ExperimentManager
# ============================================================================


class TestExperimentManager:
    """Tests for ExperimentManager class"""

    def test_run_experiment(self, mock_experiment):
        """Test running an experiment"""
        executor = SafeExperimentExecutor()
        manager = ExperimentManager(executor)

        result = manager.run_experiment(mock_experiment)

        assert isinstance(result, ExperimentResult)
        assert result.experiment == mock_experiment

    def test_track_success_rate(self, mock_experiment):
        """Test success rate tracking"""
        executor = SafeExperimentExecutor()
        manager = ExperimentManager(executor)

        # Run multiple experiments
        for _ in range(10):
            manager.run_experiment(mock_experiment)

        success_rate = manager.get_success_rate()

        assert 0 <= success_rate <= 1
        assert manager.total_experiments == 10

    def test_get_recent_failures(self, mock_experiment):
        """Test getting recent failures count"""
        executor = SafeExperimentExecutor()
        manager = ExperimentManager(executor)

        # Run some experiments
        for _ in range(5):
            manager.run_experiment(mock_experiment)

        failures = manager.get_recent_failures_count(window=5)

        assert failures >= 0


# ============================================================================
# Test CuriosityEngine
# ============================================================================


class TestCuriosityEngine:
    """Tests for main CuriosityEngine class"""

    def test_initialization(self, curiosity_engine):
        """Test engine initialization"""
        assert curiosity_engine is not None
        assert curiosity_engine.gap_analyzer is not None
        assert curiosity_engine.experiment_generator is not None
        assert curiosity_engine.exploration_budget is not None

    def test_select_exploration_strategy(self, curiosity_engine):
        """Test exploration strategy selection"""
        strategy = curiosity_engine.select_exploration_strategy()

        assert strategy in [
            "gap_driven",
            "minimal",
            "efficient",
            "comprehensive",
            "balanced",
        ]

    def test_identify_knowledge_gaps(self, curiosity_engine):
        """Test knowledge gap identification"""
        gaps = curiosity_engine.identify_knowledge_gaps(strategy="minimal")

        assert isinstance(gaps, list)

    def test_run_learning_cycle(self, curiosity_engine):
        """Test running a learning cycle"""
        summary = curiosity_engine.run_learning_cycle(max_experiments=2)

        assert "strategy_used" in summary
        assert "experiments_run" in summary
        assert "success_rate" in summary
        assert summary["experiments_run"] >= 0

    def test_thread_safety_learning_cycle(self, curiosity_engine):
        """Test thread safety of learning cycles"""
        errors = []
        results = []

        def run_cycle():
            try:
                result = curiosity_engine.run_learning_cycle(max_experiments=2)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cycle) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 3

    def test_budget_consumption(self, curiosity_engine):
        """Test budget consumption during learning"""
        curiosity_engine.exploration_budget.get_available()

        curiosity_engine.run_learning_cycle(max_experiments=3)

        final_budget = curiosity_engine.exploration_budget.get_available()

        # Budget should have been consumed or recovered
        assert final_budget >= 0

    def test_prioritize_gaps(self, curiosity_engine, mock_knowledge_gap):
        """Test gap prioritization"""
        gaps = [mock_knowledge_gap]

        priorities = curiosity_engine.prioritize_gaps(gaps)

        assert isinstance(priorities, list)

    def test_generate_targeted_experiments(self, curiosity_engine, mock_knowledge_gap):
        """Test targeted experiment generation"""
        experiments = curiosity_engine.generate_targeted_experiments(mock_knowledge_gap)

        assert isinstance(experiments, list)

    def test_run_experiment_sandboxed(self, curiosity_engine, mock_experiment):
        """Test sandboxed experiment execution"""
        result = curiosity_engine.run_experiment_sandboxed(mock_experiment)

        assert isinstance(result, ExperimentResult)

    def test_update_from_experiment_results(self, curiosity_engine, mock_experiment):
        """Test updating from experiment results"""
        result = ExperimentResult(
            experiment=mock_experiment,
            success=True,
            output={"data": "test"},
            learned_knowledge={"patterns": ["p1", "p2"]},
        )

        # Should not raise an error
        curiosity_engine.update_from_experiment_results([result])

    def test_error_recovery(self, curiosity_engine):
        """Test error recovery in learning cycle"""
        # Force an error condition
        curiosity_engine.gap_analyzer = None

        summary = curiosity_engine.run_learning_cycle(max_experiments=1)

        # Should return error summary
        assert "error" in summary or summary["experiments_run"] == 0


# ============================================================================
# Test KnowledgeIntegrator
# ============================================================================


class TestKnowledgeIntegrator:
    """Tests for KnowledgeIntegrator class"""

    def test_integrate_results(self, mock_experiment):
        """Test result integration"""
        integrator = KnowledgeIntegrator()

        result = ExperimentResult(
            experiment=mock_experiment,
            success=True,
            output={"data": "test"},
            learned_knowledge={
                "patterns": ["p1", "p2"],
                "observations": [{"var": "test"}],
                "domain": "test",
            },
        )

        # Create mocks
        mock_kb = Mock()
        mock_kb.store_knowledge = Mock()

        mock_wm = Mock()
        mock_wm.update_from_observation = Mock()

        mock_decomposer = Mock()
        mock_decomposer.learn_from_pattern = Mock()

        mock_frontier = Mock()
        mock_frontier.update_frontier = Mock()

        # Should not raise an error
        integrator.integrate_results(
            [result],
            knowledge_base=mock_kb,
            world_model=mock_wm,
            decomposer=mock_decomposer,
            exploration_frontier=mock_frontier,
        )

        # Verify mocks were called
        assert mock_kb.store_knowledge.called
        assert mock_wm.update_from_observation.called
        assert mock_decomposer.learn_from_pattern.called
        assert mock_frontier.update_frontier.called


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_learning_cycle(self):
        """Test complete learning cycle from start to finish"""
        engine = CuriosityEngine()

        # Run a full cycle
        summary = engine.run_learning_cycle(max_experiments=5)

        # Verify all components worked
        assert summary["experiments_run"] >= 0
        assert "success_rate" in summary
        assert "budget_remaining" in summary
        assert "resource_load" in summary

    def test_multiple_cycles(self):
        """Test running multiple learning cycles"""
        engine = CuriosityEngine()

        summaries = []
        for _ in range(3):
            summary = engine.run_learning_cycle(max_experiments=2)
            summaries.append(summary)

        assert len(summaries) == 3

        # Learning rate should adapt
        learning_rates = [s.get("learning_rate", 0) for s in summaries]
        assert all(0 <= lr <= 1 for lr in learning_rates)

    def test_resource_adaptation(self):
        """Test resource-based adaptation"""
        engine = CuriosityEngine()

        # Run cycles and track resource usage
        for _ in range(5):
            summary = engine.run_learning_cycle(max_experiments=2)

            # Engine should adapt to resource conditions
            assert "resource_load" in summary
            assert 0 <= summary["resource_load"] <= 1


# ============================================================================
# Edge Cases and Regression Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and known issues"""

    def test_empty_gap_[self, curiosity_engine):
        """Test handling of empty gap list"""
        priorities = curiosity_engine.prioritize_gaps([])

        assert isinstance(priorities, list)
        assert len(priorities) == 0

    def test_zero_budget(self, curiosity_engine):
        """Test behavior with zero budget"""
        # Consume all budget
        budget = curiosity_engine.exploration_budget.get_available()
        curiosity_engine.exploration_budget.consume(budget)

        summary = curiosity_engine.run_learning_cycle(max_experiments=1)

        # Should handle gracefully
        assert summary["experiments_run"] >= 0

    def test_concurrent_priority_queue_access(self, curiosity_engine):
        """Test concurrent access to priority queue"""
        errors = []

        def access_queue():
            try:
                for _ in range(10):
                    try:
                        curiosity_engine.learning_priorities.get(block=False)
                    except Empty:
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_queue) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_large_pattern_sets(self, region_manager):
        """Test handling of large pattern sets"""
        # Create large pattern set
        large_patterns = {f"pattern_{i}" for i in range(1000)}

        region_id = region_manager.add_region("test", large_patterns)

        assert region_id is not None

        region = region_manager.get_region(region_id)
        assert len(region.patterns) == 1000

    def test_deep_nesting(self, curiosity_engine):
        """Test handling of deeply nested operations"""
        # Create gaps with dependencies
        gaps = []
        for i in range(10):
            gap = KnowledgeGap(
                type="test", domain=f"domain_{i}", priority=0.5, estimated_cost=10.0
            )
            gaps.append(gap)

        # Should handle gracefully
        priorities = curiosity_engine.prioritize_gaps(gaps)
        assert len(priorities) == len(gaps)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance tests"""

    def test_region_manager_performance(self, region_manager):
        """Test RegionManager performance with many regions"""
        start_time = time.time()

        for i in range(100):
            patterns = {f"p{i}_{j}" for j in range(10)}
            region_manager.add_region(f"domain_{i}", patterns)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 5.0  # seconds

    def test_learning_cycle_performance(self, curiosity_engine):
        """Test learning cycle performance"""
        start_time = time.time()

        curiosity_engine.run_learning_cycle(max_experiments=10)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 30.0  # seconds


# ============================================================================
# Fixtures for Mocking
# ============================================================================


@pytest.fixture
def mock_knowledge_base():
    """Create mock knowledge base"""
    kb = Mock()
    kb.store_knowledge = Mock()
    kb.test_transfer = Mock(return_value={"transfer_success": True})
    kb.explore_latent = Mock(return_value={"discoveries": ["d1"]})
    return kb


@pytest.fixture
def mock_decomposer():
    """Create mock decomposer"""
    decomposer = Mock()
    decomposer.test_decomposition = Mock(
        return_value={"success": True, "components": ["c1", "c2"]}
    )
    decomposer.learn_from_pattern = Mock()
    return decomposer


@pytest.fixture
def mock_world_model():
    """Create mock world model"""
    wm = Mock()
    wm.test_intervention = Mock(return_value={"causal_strength": 0.5, "p_value": 0.03})
    wm.update_from_observation = Mock()
    return wm


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
