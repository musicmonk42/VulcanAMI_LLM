"""
test_exploration_budget.py - Comprehensive tests for ExplorationBudget
Part of the VULCAN-AGI system

Tests cover:
- Budget tracking and management
- Resource monitoring
- Cost estimation and calibration
- Dynamic budget adjustment
- Thread safety
- Performance
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from collections import defaultdict, deque

from vulcan.curiosity_engine.exploration_budget import (
    ResourceType,
    ResourceSnapshot,
    CostHistory,
    BudgetTracker,
    BudgetRecovery,
    LoadAdjuster,
    EfficiencyTracker,
    DynamicBudget,
    ResourceSampler,
    ResourcePredictor,
    ResourceAdvisor,
    ResourceMonitor,
    CostCalibrator,
    CostEstimator,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def resource_snapshot():
    """Create a sample resource snapshot"""
    return ResourceSnapshot(
        timestamp=time.time(),
        cpu_percent=0.5,
        memory_percent=0.6,
        disk_usage=0.4,
        network_bandwidth=50.0,
        active_processes=100,
        gpu_percent=0.3,
        gpu_memory=0.4
    )


@pytest.fixture
def cost_history():
    """Create sample cost history"""
    return CostHistory(
        experiment_type="decomposition",
        predicted_cost=10.0,
        actual_cost=12.0,
        domain="test_domain",
        timestamp=time.time(),
        accuracy=0.85
    )


@pytest.fixture
def budget_tracker():
    """Create BudgetTracker instance"""
    return BudgetTracker(initial_budget=100.0)


@pytest.fixture
def budget_recovery():
    """Create BudgetRecovery instance"""
    return BudgetRecovery()


@pytest.fixture
def load_adjuster():
    """Create LoadAdjuster instance"""
    return LoadAdjuster()


@pytest.fixture
def efficiency_tracker():
    """Create EfficiencyTracker instance"""
    return EfficiencyTracker()


@pytest.fixture
def dynamic_budget():
    """Create DynamicBudget instance"""
    return DynamicBudget(base_allocation=100.0)


@pytest.fixture
def resource_sampler():
    """Create ResourceSampler instance"""
    return ResourceSampler(enable_gpu=False)


@pytest.fixture
def resource_predictor():
    """Create ResourcePredictor instance"""
    return ResourcePredictor()


@pytest.fixture
def resource_advisor():
    """Create ResourceAdvisor instance"""
    return ResourceAdvisor()


@pytest.fixture
def resource_monitor():
    """Create ResourceMonitor instance"""
    return ResourceMonitor()


@pytest.fixture
def cost_calibrator():
    """Create CostCalibrator instance"""
    return CostCalibrator()


@pytest.fixture
def cost_estimator():
    """Create CostEstimator instance"""
    return CostEstimator()


@pytest.fixture
def mock_experiment():
    """Create a mock experiment"""
    exp = Mock()
    exp.experiment_id = "test_exp_001"
    exp.complexity = 0.5
    exp.timeout = 30.0
    exp.experiment_type = Mock()
    exp.experiment_type.__str__ = lambda self: "ExperimentType.DECOMPOSITION"
    exp.iteration = 0
    exp.gap = Mock()
    exp.gap.domain = "test_domain"
    exp.parameters = {
        'sample_size': 100,
        'iterations': 50
    }
    return exp


# ============================================================================
# Test ResourceType
# ============================================================================

class TestResourceType:
    """Tests for ResourceType enum"""
    
    def test_resource_types_exist(self):
        """Test that all resource types exist"""
        assert ResourceType.CPU
        assert ResourceType.MEMORY
        assert ResourceType.DISK
        assert ResourceType.NETWORK
        assert ResourceType.TIME
        assert ResourceType.GPU
    
    def test_resource_type_values(self):
        """Test resource type values"""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.DISK.value == "disk"


# ============================================================================
# Test ResourceSnapshot
# ============================================================================

class TestResourceSnapshot:
    """Tests for ResourceSnapshot class"""
    
    def test_create_snapshot(self):
        """Test creating a resource snapshot"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.5,
            memory_percent=0.6,
            disk_usage=0.4,
            network_bandwidth=100.0,
            active_processes=50
        )
        
        assert snapshot.cpu_percent == 0.5
        assert snapshot.memory_percent == 0.6
        assert snapshot.disk_usage == 0.4
        assert snapshot.active_processes == 50
    
    def test_to_dict(self, resource_snapshot):
        """Test converting snapshot to dictionary"""
        snapshot_dict = resource_snapshot.to_dict()
        
        assert 'timestamp' in snapshot_dict
        assert 'cpu_percent' in snapshot_dict
        assert 'memory_percent' in snapshot_dict
        assert snapshot_dict['cpu_percent'] == 0.5
    
    def test_get_bottleneck_cpu(self):
        """Test identifying CPU bottleneck"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.95,
            memory_percent=0.5,
            disk_usage=0.4,
            network_bandwidth=50.0,
            active_processes=100
        )
        
        bottleneck = snapshot.get_bottleneck()
        assert bottleneck == ResourceType.CPU
    
    def test_get_bottleneck_memory(self):
        """Test identifying memory bottleneck"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.5,
            memory_percent=0.9,
            disk_usage=0.4,
            network_bandwidth=50.0,
            active_processes=100
        )
        
        bottleneck = snapshot.get_bottleneck()
        assert bottleneck == ResourceType.MEMORY
    
    def test_get_bottleneck_none(self):
        """Test no bottleneck when resources are healthy"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.5,
            memory_percent=0.5,
            disk_usage=0.5,
            network_bandwidth=50.0,
            active_processes=100
        )
        
        bottleneck = snapshot.get_bottleneck()
        assert bottleneck is None


# ============================================================================
# Test CostHistory
# ============================================================================

class TestCostHistory:
    """Tests for CostHistory class"""
    
    def test_create_cost_history(self):
        """Test creating cost history"""
        history = CostHistory(
            experiment_type="decomposition",
            predicted_cost=10.0,
            actual_cost=12.0,
            domain="test",
            timestamp=time.time()
        )
        
        assert history.experiment_type == "decomposition"
        assert history.predicted_cost == 10.0
        assert history.actual_cost == 12.0
    
    def test_error_calculation(self):
        """Test error calculation"""
        history = CostHistory(
            experiment_type="test",
            predicted_cost=10.0,
            actual_cost=12.0,
            domain="test",
            timestamp=time.time()
        )
        
        error = history.error()
        # |10 - 12| / 12 = 2/12 = 0.1666...
        assert abs(error - 0.1666) < 0.001
    
    def test_relative_error_calculation(self):
        """Test relative error calculation"""
        history = CostHistory(
            experiment_type="test",
            predicted_cost=10.0,
            actual_cost=12.0,
            domain="test",
            timestamp=time.time()
        )
        
        rel_error = history.relative_error()
        # (12 - 10) / 10 = 0.2
        assert abs(rel_error - 0.2) < 0.001
    
    def test_error_with_zero_actual(self):
        """Test error calculation with zero actual cost"""
        history = CostHistory(
            experiment_type="test",
            predicted_cost=10.0,
            actual_cost=0.0,
            domain="test",
            timestamp=time.time()
        )
        
        error = history.error()
        assert error == 0.0


# ============================================================================
# Test BudgetTracker
# ============================================================================

class TestBudgetTracker:
    """Tests for BudgetTracker class"""
    
    def test_initialization(self, budget_tracker):
        """Test tracker initialization"""
        assert budget_tracker.current_budget == 100.0
        assert budget_tracker.reserved_budget == 0.0
        assert budget_tracker.total_consumed == 0.0
    
    def test_get_available(self, budget_tracker):
        """Test getting available budget"""
        available = budget_tracker.get_available()
        assert available == 100.0
    
    def test_consume_success(self, budget_tracker):
        """Test successful budget consumption"""
        success = budget_tracker.consume(30.0)
        
        assert success is True
        assert budget_tracker.current_budget == 70.0
        assert budget_tracker.total_consumed == 30.0
    
    def test_consume_failure(self, budget_tracker):
        """Test failed budget consumption"""
        success = budget_tracker.consume(150.0)
        
        assert success is False
        assert budget_tracker.current_budget == 100.0
        assert budget_tracker.total_consumed == 0.0
    
    def test_reserve_success(self, budget_tracker):
        """Test successful budget reservation"""
        success = budget_tracker.reserve(30.0, "res_001")
        
        assert success is True
        assert budget_tracker.reserved_budget == 30.0
        assert budget_tracker.get_available() == 70.0
    
    def test_reserve_failure(self, budget_tracker):
        """Test failed budget reservation"""
        success = budget_tracker.reserve(150.0, "res_001")
        
        assert success is False
        assert budget_tracker.reserved_budget == 0.0
    
    def test_release_reservation(self, budget_tracker):
        """Test releasing reservation"""
        budget_tracker.reserve(30.0, "res_001")
        budget_tracker.release_reservation(30.0, "res_001")
        
        assert budget_tracker.reserved_budget == 0.0
        assert budget_tracker.get_available() == 100.0
    
    def test_add_budget(self, budget_tracker):
        """Test adding budget"""
        budget_tracker.add_budget(50.0)
        
        assert budget_tracker.current_budget == 150.0
    
    def test_set_budget(self, budget_tracker):
        """Test setting budget"""
        budget_tracker.set_budget(200.0)
        
        assert budget_tracker.current_budget == 200.0
    
    def test_get_statistics(self, budget_tracker):
        """Test getting statistics"""
        budget_tracker.consume(30.0)
        budget_tracker.reserve(20.0, "res_001")
        
        stats = budget_tracker.get_statistics()
        
        assert 'current_budget' in stats
        assert 'available' in stats
        assert 'reserved' in stats
        assert stats['current_budget'] == 70.0
        assert stats['reserved'] == 20.0
    
    def test_thread_safety(self, budget_tracker):
        """Test thread safety"""
        errors = []
        
        def consume_budget(thread_id):
            try:
                for _ in range(10):
                    budget_tracker.consume(1.0)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=consume_budget, args=(i,)) for i in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ============================================================================
# Test BudgetRecovery
# ============================================================================

class TestBudgetRecovery:
    """Tests for BudgetRecovery class"""
    
    def test_initialization(self, budget_recovery):
        """Test recovery initialization"""
        assert budget_recovery.recovery_rate == 0.1
        assert budget_recovery.enabled is True
    
    def test_calculate_recovery_not_ready(self, budget_recovery):
        """Test recovery calculation when not ready"""
        recovery = budget_recovery.calculate_recovery(50.0, 100.0)
        
        # Should be 0 immediately after initialization
        assert recovery == 0.0
    
    def test_calculate_recovery_ready(self, budget_recovery):
        """Test recovery calculation when ready"""
        # Set last recovery time to past
        budget_recovery.last_recovery_time = time.time() - 61.0
        
        recovery = budget_recovery.calculate_recovery(50.0, 100.0)
        
        # Should recover some budget
        assert recovery > 0.0
        assert recovery <= 50.0
    
    def test_set_enabled(self, budget_recovery):
        """Test enabling/disabling recovery"""
        budget_recovery.set_enabled(False)
        assert budget_recovery.enabled is False
        
        recovery = budget_recovery.calculate_recovery(50.0, 100.0)
        assert recovery == 0.0
    
    def test_reset_timer(self, budget_recovery):
        """Test resetting recovery timer"""
        old_time = budget_recovery.last_recovery_time
        time.sleep(0.1)
        
        budget_recovery.reset_timer()
        
        assert budget_recovery.last_recovery_time > old_time


# ============================================================================
# Test LoadAdjuster
# ============================================================================

class TestLoadAdjuster:
    """Tests for LoadAdjuster class"""
    
    def test_initialization(self, load_adjuster):
        """Test adjuster initialization"""
        assert load_adjuster.adjustment_rate == 0.1
    
    def test_calculate_adjustment_low_load(self, load_adjuster):
        """Test adjustment for low system load"""
        new_budget = load_adjuster.calculate_adjustment(
            system_load=0.2,
            current_budget=100.0,
            min_budget=10.0,
            max_budget=200.0
        )
        
        # Should increase budget
        assert new_budget > 100.0
    
    def test_calculate_adjustment_high_load(self, load_adjuster):
        """Test adjustment for high system load"""
        new_budget = load_adjuster.calculate_adjustment(
            system_load=0.9,
            current_budget=100.0,
            min_budget=10.0,
            max_budget=200.0
        )
        
        # Should decrease budget
        assert new_budget < 100.0
    
    def test_calculate_adjustment_normal_load(self, load_adjuster):
        """Test adjustment for normal system load"""
        new_budget = load_adjuster.calculate_adjustment(
            system_load=0.5,
            current_budget=100.0,
            min_budget=10.0,
            max_budget=200.0
        )
        
        # Should be similar to current
        assert abs(new_budget - 100.0) < 20.0
    
    def test_calculate_adjustment_clamping(self, load_adjuster):
        """Test that adjustment respects limits"""
        # Test max limit
        new_budget = load_adjuster.calculate_adjustment(
            system_load=0.1,
            current_budget=190.0,
            min_budget=10.0,
            max_budget=200.0
        )
        assert new_budget <= 200.0
        
        # Test min limit
        new_budget = load_adjuster.calculate_adjustment(
            system_load=0.95,
            current_budget=15.0,
            min_budget=10.0,
            max_budget=200.0
        )
        assert new_budget >= 10.0
    
    def test_calculate_adjustment_invalid_load(self, load_adjuster):
        """Test adjustment with invalid load values"""
        # Test negative load
        new_budget = load_adjuster.calculate_adjustment(
            system_load=-0.5,
            current_budget=100.0,
            min_budget=10.0,
            max_budget=200.0
        )
        assert new_budget > 0
        
        # Test load > 1.0
        new_budget = load_adjuster.calculate_adjustment(
            system_load=1.5,
            current_budget=100.0,
            min_budget=10.0,
            max_budget=200.0
        )
        assert new_budget > 0


# ============================================================================
# Test EfficiencyTracker
# ============================================================================

class TestEfficiencyTracker:
    """Tests for EfficiencyTracker class"""
    
    def test_initialization(self, efficiency_tracker):
        """Test tracker initialization"""
        assert efficiency_tracker is not None
    
    def test_update(self, efficiency_tracker):
        """Test updating efficiency"""
        efficiency_tracker.update(experiments_run=10, successes=7)
        
        avg = efficiency_tracker.get_average_efficiency()
        assert abs(avg - 0.7) < 0.01
    
    def test_calculate_base_adjustment_high_efficiency(self, efficiency_tracker):
        """Test base adjustment with high efficiency"""
        # Add high efficiency scores
        for _ in range(20):
            efficiency_tracker.update(10, 8)
        
        new_base = efficiency_tracker.calculate_base_adjustment(
            current_base=100.0,
            min_base=10.0,
            max_base=200.0
        )
        
        # Should increase
        assert new_base > 100.0
    
    def test_calculate_base_adjustment_low_efficiency(self, efficiency_tracker):
        """Test base adjustment with low efficiency"""
        # Add low efficiency scores
        for _ in range(20):
            efficiency_tracker.update(10, 2)
        
        new_base = efficiency_tracker.calculate_base_adjustment(
            current_base=100.0,
            min_base=10.0,
            max_base=200.0
        )
        
        # Should decrease
        assert new_base < 100.0
    
    def test_get_average_efficiency_empty(self, efficiency_tracker):
        """Test getting average efficiency when empty"""
        avg = efficiency_tracker.get_average_efficiency()
        assert avg == 0.5


# ============================================================================
# Test DynamicBudget
# ============================================================================

class TestDynamicBudget:
    """Tests for DynamicBudget class"""
    
    def test_initialization(self, dynamic_budget):
        """Test budget initialization"""
        assert dynamic_budget.base_allocation == 100.0
        assert dynamic_budget.enable_recovery is True
    
    def test_get_available(self, dynamic_budget):
        """Test getting available budget"""
        available = dynamic_budget.get_available()
        assert available > 0
        assert available <= 100.0
    
    def test_can_afford(self, dynamic_budget):
        """Test checking affordability"""
        assert dynamic_budget.can_afford(50.0) is True
        assert dynamic_budget.can_afford(150.0) is False
    
    def test_consume(self, dynamic_budget):
        """Test consuming budget"""
        success = dynamic_budget.consume(30.0)
        
        assert success is True
        assert dynamic_budget.get_available() < 100.0
    
    def test_reserve_and_release(self, dynamic_budget):
        """Test reserving and releasing budget"""
        res_id = dynamic_budget.reserve(30.0)
        
        assert res_id is not None
        assert dynamic_budget.get_available() < 100.0
        
        dynamic_budget.release_reservation(30.0, res_id)
        
        # Available should increase
        available_after = dynamic_budget.get_available()
        assert available_after > 0
    
    def test_adjust_for_load(self, dynamic_budget):
        """Test adjusting for system load"""
        initial = dynamic_budget.tracker.current_budget
        
        # High load should decrease budget
        dynamic_budget.adjust_for_load(0.9)
        
        after = dynamic_budget.tracker.current_budget
        assert after < initial
    
    def test_update_efficiency(self, dynamic_budget):
        """Test updating efficiency"""
        dynamic_budget.update_efficiency(experiments_run=10, successes=7)
        
        stats = dynamic_budget.get_statistics()
        assert 'efficiency' in stats
    
    def test_get_statistics(self, dynamic_budget):
        """Test getting statistics"""
        stats = dynamic_budget.get_statistics()
        
        assert 'current_budget' in stats
        assert 'base_allocation' in stats
        assert 'efficiency' in stats
        assert stats['base_allocation'] == 100.0
    
    def test_reset(self, dynamic_budget):
        """Test resetting budget"""
        dynamic_budget.consume(30.0)
        dynamic_budget.reset()
        
        available = dynamic_budget.get_available()
        assert available == 100.0
    
    def test_thread_safety(self, dynamic_budget):
        """Test thread safety"""
        errors = []
        
        def perform_operations(thread_id):
            try:
                for _ in range(5):
                    dynamic_budget.consume(1.0)
                    dynamic_budget.reserve(1.0)
                    dynamic_budget.get_available()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=perform_operations, args=(i,)) for i in range(3)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ============================================================================
# Test ResourceSampler
# ============================================================================

class TestResourceSampler:
    """Tests for ResourceSampler class"""
    
    def test_initialization(self, resource_sampler):
        """Test sampler initialization"""
        assert resource_sampler is not None
        assert resource_sampler.enable_gpu is False
    
    def test_sample(self, resource_sampler):
        """Test sampling resources"""
        snapshot = resource_sampler.sample()
        
        assert isinstance(snapshot, ResourceSnapshot)
        assert 0.0 <= snapshot.cpu_percent <= 1.0
        assert 0.0 <= snapshot.memory_percent <= 1.0
        assert snapshot.active_processes >= 0


# ============================================================================
# Test ResourcePredictor
# ============================================================================

class TestResourcePredictor:
    """Tests for ResourcePredictor class"""
    
    def test_initialization(self, resource_predictor):
        """Test predictor initialization"""
        assert resource_predictor is not None
    
    def test_add_snapshot(self, resource_predictor, resource_snapshot):
        """Test adding snapshot"""
        resource_predictor.add_snapshot(resource_snapshot)
        
        assert len(resource_predictor.history) == 1
    
    def test_predict_load_insufficient_history(self, resource_predictor):
        """Test prediction with insufficient history"""
        predicted, confidence = resource_predictor.predict_load()
        
        assert 0.0 <= predicted <= 1.0
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_load_with_history(self, resource_predictor):
        """Test prediction with sufficient history"""
        # Add multiple snapshots
        for i in range(20):
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=0.5 + i * 0.01,
                memory_percent=0.6,
                disk_usage=0.4,
                network_bandwidth=50.0,
                active_processes=100
            )
            resource_predictor.add_snapshot(snapshot)
        
        predicted, confidence = resource_predictor.predict_load()
        
        assert 0.0 <= predicted <= 1.0
        assert confidence > 0.2
    
    def test_calculate_trend(self, resource_predictor):
        """Test trend calculation"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trend = resource_predictor._calculate_trend(values)
        
        # Should be positive upward trend
        assert trend > 0
    
    def test_calculate_trend_flat(self, resource_predictor):
        """Test trend calculation for flat values"""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        trend = resource_predictor._calculate_trend(values)
        
        # Should be near zero
        assert abs(trend) < 0.1
    
    def test_calculate_trend_single_value(self, resource_predictor):
        """Test trend calculation with single value"""
        values = np.array([5.0])
        trend = resource_predictor._calculate_trend(values)
        
        assert trend == 0.0


# ============================================================================
# Test ResourceAdvisor
# ============================================================================

class TestResourceAdvisor:
    """Tests for ResourceAdvisor class"""
    
    def test_initialization(self, resource_advisor):
        """Test advisor initialization"""
        assert resource_advisor is not None
    
    def test_recommend_adjustment_critical_cpu(self, resource_advisor):
        """Test recommendation for critical CPU usage"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.95,
            memory_percent=0.5,
            disk_usage=0.4,
            network_bandwidth=50.0,
            active_processes=100
        )
        
        recommendation = resource_advisor.recommend_adjustment(
            snapshot, predicted_load=0.9, confidence=0.8
        )
        
        assert recommendation['action'] == 'reduce'
        assert recommendation['adjustment_factor'] < 1.0
    
    def test_recommend_adjustment_low_usage(self, resource_advisor):
        """Test recommendation for low resource usage"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.2,
            memory_percent=0.2,
            disk_usage=0.2,
            network_bandwidth=10.0,
            active_processes=50
        )
        
        recommendation = resource_advisor.recommend_adjustment(
            snapshot, predicted_load=0.3, confidence=0.7
        )
        
        assert recommendation['action'] == 'increase'
        assert recommendation['adjustment_factor'] > 1.0


# ============================================================================
# Test ResourceMonitor
# ============================================================================

class TestResourceMonitor:
    """Tests for ResourceMonitor class"""
    
    def test_initialization(self, resource_monitor):
        """Test monitor initialization"""
        assert resource_monitor is not None
    
    def test_get_current_load(self, resource_monitor):
        """Test getting current load"""
        load = resource_monitor.get_current_load()
        
        assert 0.0 <= load <= 1.0
    
    def test_get_current_load_caching(self, resource_monitor):
        """Test load caching"""
        load1 = resource_monitor.get_current_load()
        load2 = resource_monitor.get_current_load()
        
        # Should be cached
        assert load1 == load2
    
    def test_predict_future_load(self, resource_monitor):
        """Test predicting future load"""
        predicted, confidence = resource_monitor.predict_future_load(horizon_minutes=10)
        
        assert 0.0 <= predicted <= 1.0
        assert 0.0 <= confidence <= 1.0
    
    def test_recommend_budget_adjustment(self, resource_monitor):
        """Test recommending budget adjustment"""
        recommendation = resource_monitor.recommend_budget_adjustment()
        
        assert 'action' in recommendation
        assert recommendation['action'] in ['maintain', 'reduce', 'increase']
    
    def test_get_resource_snapshot(self, resource_monitor):
        """Test getting resource snapshot"""
        snapshot = resource_monitor.get_resource_snapshot()
        
        assert isinstance(snapshot, ResourceSnapshot)
    
    def test_get_resource_trends(self, resource_monitor):
        """Test getting resource trends"""
        # Add some samples
        for _ in range(10):
            resource_monitor.get_current_load()
            time.sleep(0.01)
        
        trends = resource_monitor.get_resource_trends()
        
        # May be empty if not enough history
        if trends:
            assert 'cpu' in trends or 'memory' in trends


# ============================================================================
# Test CostCalibrator
# ============================================================================

class TestCostCalibrator:
    """Tests for CostCalibrator class"""
    
    def test_initialization(self, cost_calibrator):
        """Test calibrator initialization"""
        assert cost_calibrator.learning_rate == 0.1
        assert cost_calibrator.calibration_factor == 1.0
    
    def test_get_confidence_interval(self, cost_calibrator):
        """Test getting confidence interval"""
        lower, upper = cost_calibrator.get_confidence_interval('unknown_type')
        
        assert lower == 0.8
        assert upper == 1.2
    
    def test_calibrate_empty_history(self, cost_calibrator):
        """Test calibration with empty history"""
        base_costs = {'decomposition': 10.0}
        domain_costs = {'test': 1.0}
        
        updated_base, updated_domain = cost_calibrator.calibrate(
            [], base_costs, domain_costs
        )
        
        assert updated_base == base_costs
        assert updated_domain == domain_costs
    
    def test_calibrate_with_history(self, cost_calibrator):
        """Test calibration with history"""
        history = [
            CostHistory('decomposition', 10.0, 12.0, 'test', time.time()),
            CostHistory('decomposition', 10.0, 11.0, 'test', time.time()),
            CostHistory('decomposition', 10.0, 13.0, 'test', time.time())
        ]
        
        base_costs = {'decomposition': 10.0}
        domain_costs = {'test': 1.0}
        
        updated_base, updated_domain = cost_calibrator.calibrate(
            history, base_costs, domain_costs
        )
        
        # Should adjust costs based on actual vs predicted
        assert updated_base is not None
        assert updated_domain is not None


# ============================================================================
# Test CostEstimator
# ============================================================================

class TestCostEstimator:
    """Tests for CostEstimator class"""
    
    def test_initialization(self, cost_estimator):
        """Test estimator initialization"""
        assert cost_estimator is not None
        assert 'decomposition' in cost_estimator.base_costs
    
    def test_estimate_learning_cost(self, cost_estimator):
        """Test estimating learning cost"""
        cost = cost_estimator.estimate_learning_cost(
            gap_type='decomposition',
            complexity=0.5,
            priority=0.8,
            domain='test_domain'
        )
        
        assert cost > 0
    
    def test_estimate_experiment_cost(self, cost_estimator, mock_experiment):
        """Test estimating experiment cost"""
        cost = cost_estimator.estimate_experiment_cost(mock_experiment)
        
        assert cost > 0
    
    def test_estimate_experiment_cost_caching(self, cost_estimator, mock_experiment):
        """Test experiment cost caching"""
        cost1 = cost_estimator.estimate_experiment_cost(mock_experiment)
        cost2 = cost_estimator.estimate_experiment_cost(mock_experiment)
        
        # Should be cached
        assert cost1 == cost2
    
    def test_calibrate_from_history(self, cost_estimator):
        """Test calibrating from history"""
        history = [
            CostHistory('decomposition', 10.0, 12.0, 'test', time.time()),
            CostHistory('causal', 20.0, 22.0, 'test', time.time())
        ]
        
        cost_estimator.calibrate_from_history(history)
        
        # Should update calibration
        assert cost_estimator.calibrator.calibration_factor > 0
    
    def test_adjust_for_domain_novelty(self, cost_estimator):
        """Test adjusting for domain novelty"""
        adjusted = cost_estimator.adjust_for_domain_novelty('new_domain', 10.0)
        
        assert adjusted > 0
    
    def test_update_from_actual(self, cost_estimator):
        """Test updating from actual cost"""
        cost_estimator.update_from_actual(
            experiment_type='decomposition',
            predicted=10.0,
            actual=12.0,
            domain='test'
        )
        
        assert len(cost_estimator.cost_history) == 1
    
    def test_get_accuracy_stats(self, cost_estimator):
        """Test getting accuracy statistics"""
        stats = cost_estimator.get_accuracy_stats()
        
        assert 'mean_error' in stats
        assert 'confidence' in stats
    
    def test_get_domain_statistics(self, cost_estimator):
        """Test getting domain statistics"""
        # Add some history
        cost_estimator.update_from_actual('decomposition', 10.0, 12.0, 'test')
        
        stats = cost_estimator.get_domain_statistics()
        
        if stats:
            assert 'test' in stats
    
    def test_thread_safety(self, cost_estimator, mock_experiment):
        """Test thread safety"""
        errors = []
        
        def estimate_costs(thread_id):
            try:
                for _ in range(5):
                    cost_estimator.estimate_experiment_cost(mock_experiment)
                    cost_estimator.estimate_learning_cost(
                        'decomposition', 0.5, 0.8, 'test'
                    )
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=estimate_costs, args=(i,)) for i in range(3)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_budget_lifecycle(self, dynamic_budget, cost_estimator, mock_experiment):
        """Test complete budget lifecycle"""
        # Estimate cost
        cost = cost_estimator.estimate_experiment_cost(mock_experiment)
        
        # Check affordability
        assert dynamic_budget.can_afford(cost)
        
        # Reserve budget
        res_id = dynamic_budget.reserve(cost)
        assert res_id is not None
        
        # Consume budget
        success = dynamic_budget.consume(cost)
        assert success is True
        
        # Release reservation
        dynamic_budget.release_reservation(cost, res_id)
    
    def test_resource_monitoring_and_adjustment(self, resource_monitor, dynamic_budget):
        """Test resource monitoring with budget adjustment"""
        # Get current load
        load = resource_monitor.get_current_load()
        
        # Adjust budget based on load
        dynamic_budget.adjust_for_load(load)
        
        # Get recommendation
        recommendation = resource_monitor.recommend_budget_adjustment()
        
        assert recommendation['action'] in ['maintain', 'reduce', 'increase']
    
    def test_cost_estimation_and_calibration(self, cost_estimator, mock_experiment):
        """Test cost estimation and calibration workflow"""
        # Initial estimate
        predicted_cost = cost_estimator.estimate_experiment_cost(mock_experiment)
        
        # Simulate actual cost
        actual_cost = predicted_cost * 1.2
        
        # Update with actual
        cost_estimator.update_from_actual(
            'decomposition',
            predicted_cost,
            actual_cost,
            'test_domain'
        )
        
        # Check history
        assert len(cost_estimator.cost_history) > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests"""
    
    def test_large_scale_operations(self, dynamic_budget):
        """Test large scale budget operations"""
        start_time = time.time()
        
        # Perform many operations
        for i in range(100):
            dynamic_budget.consume(0.1)
            dynamic_budget.reserve(0.1)
            dynamic_budget.get_available()
        
        elapsed = time.time() - start_time
        
        # Should be fast
        assert elapsed < 1.0
    
    def test_resource_sampling_performance(self, resource_sampler):
        """Test resource sampling performance"""
        start_time = time.time()
        
        # Sample many times
        for _ in range(50):
            resource_sampler.sample()
        
        elapsed = time.time() - start_time
        
        # Should be reasonably fast (allow more time for slower systems)
        # Resource sampling involves psutil calls which can vary by system
        assert elapsed < 10.0
    
    def test_cost_estimation_performance(self, cost_estimator, mock_experiment):
        """Test cost estimation performance"""
        start_time = time.time()
        
        # Estimate many times
        for _ in range(100):
            cost_estimator.estimate_experiment_cost(mock_experiment)
        
        elapsed = time.time() - start_time
        
        # Should be fast with caching
        assert elapsed < 1.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])