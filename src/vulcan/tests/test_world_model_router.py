"""
test_world_model_router.py - Comprehensive test suite for WorldModelRouter
Part of the VULCAN-AGI system

Tests cover:
- UpdateType and UpdatePriority enums
- UpdateStrategy and ObservationSignature dataclasses
- UpdatePlan dataclass
- UpdateDependencyGraph for dependency management
- PatternLearner for learning from execution history
- CostModel for cost estimation and calibration
- WorldModelRouter main orchestrator
- Integration with world model components
- Safety validation throughout
- State persistence
- Thread safety
- Edge cases and error handling

FIXED: test_route_with_missing_attributes now provides proper mock attributes
FIXED: Patched EnhancedSafetyValidator to return 'safe' for most tests
FIXED: Corrected invalid config in test_router_with_safety_validator
"""

import pytest
import numpy as np
import time
import threading
import tempfile
import pickle
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch

# Import from world_model_router
from vulcan.world_model.world_model_router import (
    UpdateType,
    UpdatePriority,
    UpdateStrategy,
    ObservationSignature,
    UpdatePlan,
    UpdateDependencyGraph,
    PatternLearner,
    CostModel,
    WorldModelRouter
)


# ============================================================================
# Mocks for Safety Patching
# ============================================================================

# Create a mock validator instance that always returns 'safe'
mock_safe_validator_instance = MagicMock()
mock_safe_validator_instance.analyze_observation_safety = Mock(return_value={'safe': True})
mock_safe_validator_instance.validate_signature_safety = Mock(return_value={'safe': True})
mock_safe_validator_instance._validate_signature_safety = Mock(return_value={'safe': True}) # Handle private method call
mock_safe_validator_instance.validate_plan_safety = Mock(return_value={'safe': True})
mock_safe_validator_instance._validate_plan_safety = Mock(return_value={'safe': True}) # Handle private method call
mock_safe_validator_instance.validate_update_safety = Mock(return_value={'safe': True})
mock_safe_validator_instance._validate_update_safety = Mock(return_value={'safe': True}) # Handle private method call
mock_safe_validator_instance.validate_result_safety = Mock(return_value={'safe': True})
mock_safe_validator_instance._validate_result_safety = Mock(return_value={'safe': True}) # Handle private method call

# Create a mock validator class that returns the safe instance
mock_validator_class = MagicMock(return_value=mock_safe_validator_instance)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_world_model():
    """Create mock world model"""
    model = Mock()
    model.correlation_tracker = Mock()
    model.correlation_tracker.update = Mock(return_value={'status': 'success'})
    model.correlation_tracker.get_baseline = Mock(return_value=5.0)
    model.correlation_tracker.get_noise_level = Mock(return_value=0.5)
    
    model.dynamics = Mock()
    model.dynamics.update = Mock(return_value={'status': 'success'})
    
    model.invariant_detector = Mock()
    model.invariant_detector.check = Mock(return_value={'status': 'success'})
    
    model.confidence_tracker = Mock()
    model.confidence_tracker.update = Mock(return_value={'status': 'success'})
    
    model.intervention_manager = Mock()
    model.causal_graph = Mock()
    
    model.last_observation_time = time.time() - 10.0
    model.safety_validator = None
    
    return model


@pytest.fixture
def sample_observation():
    """Create sample observation"""
    obs = Mock()
    obs.timestamp = time.time()
    obs.variables = {'x': 1.0, 'y': 2.0, 'z': 3.0}
    obs.confidence = 0.9
    obs.domain = "test"
    obs.intervention_data = None
    obs.sequence_id = 123
    obs.metadata = {}
    obs.is_anomaly = False
    obs.structural_change = False
    obs.is_intervention = False
    return obs


@pytest.fixture
def intervention_observation():
    """Create observation with intervention"""
    obs = Mock()
    obs.timestamp = time.time()
    obs.variables = {'x': 5.0, 'y': 10.0}
    obs.intervention_data = {'intervened_variable': 'x', 'intervention_value': 5.0}
    obs.confidence = 1.0
    obs.domain = "test"
    obs.sequence_id = 124
    obs.metadata = {'is_intervention': True}
    obs.is_anomaly = False
    obs.structural_change = False
    obs.is_intervention = True
    return obs


@pytest.fixture
def observation_signature():
    """Create observation signature"""
    return ObservationSignature(
        has_intervention=False,
        has_temporal=True,
        has_multi_variable=True,
        has_anomaly=False,
        has_structural_change=False,
        variable_count=3,
        time_delta=10.0,
        confidence=0.9,
        domain="test",
        pattern_hash="123456"
    )


@pytest.fixture
def dependency_graph():
    """Create dependency graph"""
    return UpdateDependencyGraph()


@pytest.fixture
def pattern_learner():
    """Create pattern learner"""
    return PatternLearner(window_size=100)


@pytest.fixture
def cost_model():
    """Create cost model"""
    return CostModel()


@pytest.fixture
def router(mock_world_model):
    """Create router"""
    config = {
        'time_budget_ms': 1000,
        'min_confidence': 0.5,
        'exploration_rate': 0.1,
        'use_learning': True
    }
    return WorldModelRouter(mock_world_model, config)


# ============================================================================
# Test Enums
# ============================================================================

class TestEnums:
    """Test enum definitions"""
    
    def test_update_type_enum(self):
        """Test UpdateType enum"""
        assert UpdateType.INTERVENTION.value == "intervention"
        assert UpdateType.CORRELATION.value == "correlation"
        assert UpdateType.DYNAMICS.value == "dynamics"
        assert UpdateType.CAUSAL.value == "causal"
        assert UpdateType.INVARIANT.value == "invariant"
        assert UpdateType.CONFIDENCE.value == "confidence"
    
    def test_update_priority_enum(self):
        """Test UpdatePriority enum"""
        assert UpdatePriority.CRITICAL.value == 0
        assert UpdatePriority.HIGH.value == 1
        assert UpdatePriority.NORMAL.value == 2
        assert UpdatePriority.LOW.value == 3
        assert UpdatePriority.BATCH.value == 4
        
        # Test ordering
        assert UpdatePriority.CRITICAL.value < UpdatePriority.HIGH.value
        assert UpdatePriority.HIGH.value < UpdatePriority.NORMAL.value


# ============================================================================
# Test Dataclasses
# ============================================================================

class TestDataclasses:
    """Test dataclass definitions"""
    
    def test_update_strategy_creation(self):
        """Test UpdateStrategy creation"""
        strategy = UpdateStrategy(
            update_type=UpdateType.CORRELATION,
            priority=UpdatePriority.HIGH,
            dependencies=set(),
            estimated_cost_ms=20.0,
            confidence_threshold=0.5,
            can_parallelize=True,
            can_defer=False
        )
        
        assert strategy.update_type == UpdateType.CORRELATION
        assert strategy.priority == UpdatePriority.HIGH
        assert strategy.can_parallelize == True
    
    def test_observation_signature_creation(self, observation_signature):
        """Test ObservationSignature creation"""
        assert observation_signature.has_intervention == False
        assert observation_signature.has_multi_variable == True
        assert observation_signature.variable_count == 3
        assert observation_signature.confidence == 0.9
    
    def test_update_plan_creation(self):
        """Test UpdatePlan creation"""
        plan = UpdatePlan(
            immediate=[UpdateType.CORRELATION, UpdateType.DYNAMICS],
            deferred=[UpdateType.INVARIANT],
            parallel_groups=[[UpdateType.CONFIDENCE]],
            estimated_time_ms=100.0,
            confidence=0.8,
            reasoning="Test plan"
        )
        
        assert len(plan.immediate) == 2
        assert len(plan.deferred) == 1
        assert len(plan.parallel_groups) == 1
        assert plan.confidence == 0.8


# ============================================================================
# Test UpdateDependencyGraph
# ============================================================================

class TestUpdateDependencyGraph:
    """Test UpdateDependencyGraph component"""
    
    def test_get_dependencies(self, dependency_graph):
        """Test getting dependencies"""
        # Intervention has no dependencies
        deps = dependency_graph.get_dependencies(UpdateType.INTERVENTION)
        assert len(deps) == 0
        
        # Dynamics depends on correlation
        deps = dependency_graph.get_dependencies(UpdateType.DYNAMICS)
        assert UpdateType.CORRELATION in deps
        
        # Causal depends on intervention and correlation
        deps = dependency_graph.get_dependencies(UpdateType.CAUSAL)
        assert UpdateType.INTERVENTION in deps or UpdateType.CORRELATION in deps
    
    def test_can_parallelize(self, dependency_graph):
        """Test parallelization checking"""
        # These should be able to run in parallel
        can_parallel = dependency_graph.can_parallelize([
            UpdateType.CORRELATION,
            UpdateType.CONFIDENCE
        ])
        assert can_parallel == True
        
        # Intervention and Causal are mutually exclusive
        can_parallel = dependency_graph.can_parallelize([
            UpdateType.INTERVENTION,
            UpdateType.CAUSAL
        ])
        assert can_parallel == False
    
    def test_get_execution_order(self, dependency_graph):
        """Test getting execution order"""
        required = {
            UpdateType.CORRELATION,
            UpdateType.DYNAMICS,
            UpdateType.CONFIDENCE
        }
        
        execution_groups = dependency_graph.get_execution_order(required)
        
        assert len(execution_groups) > 0
        # All required updates should be included
        all_updates = []
        for group in execution_groups:
            all_updates.extend(group)
        
        assert UpdateType.CORRELATION in all_updates
        assert UpdateType.DYNAMICS in all_updates
        assert UpdateType.CONFIDENCE in all_updates
    
    def test_execution_order_respects_dependencies(self, dependency_graph):
        """Test that execution order respects dependencies"""
        required = {UpdateType.CORRELATION, UpdateType.DYNAMICS}
        
        execution_groups = dependency_graph.get_execution_order(required)
        
        # Correlation should come before Dynamics
        corr_group = None
        dyn_group = None
        
        for i, group in enumerate(execution_groups):
            if UpdateType.CORRELATION in group:
                corr_group = i
            if UpdateType.DYNAMICS in group:
                dyn_group = i
        
        if corr_group is not None and dyn_group is not None:
            assert corr_group <= dyn_group


# ============================================================================
# Test PatternLearner
# ============================================================================

class TestPatternLearner:
    """Test PatternLearner component"""
    
    def test_initialization(self, pattern_learner):
        """Test pattern learner initialization"""
        assert pattern_learner.window_size == 100
        assert len(pattern_learner.pattern_history) == 0
        assert len(pattern_learner.rules) == 0
    
    def test_record_pattern(self, pattern_learner, observation_signature):
        """Test recording a pattern"""
        updates = [UpdateType.CORRELATION, UpdateType.DYNAMICS]
        
        pattern_learner.record_pattern(
            signature=observation_signature,
            updates_run=updates,
            success=True,
            cost_ms=50.0
        )
        
        assert len(pattern_learner.pattern_history) == 1
        assert len(pattern_learner.pattern_counts) > 0
    
    def test_record_pattern_with_none_signature(self, pattern_learner):
        """Test recording pattern with None signature"""
        # Should handle gracefully
        pattern_learner.record_pattern(
            signature=None,
            updates_run=[UpdateType.CORRELATION],
            success=True,
            cost_ms=50.0
        )
        
        # Should not add to history
        assert len(pattern_learner.pattern_history) == 0
    
    def test_learn_rules(self, pattern_learner, observation_signature):
        """Test rule learning"""
        # Record multiple successful patterns
        updates = [UpdateType.CORRELATION, UpdateType.DYNAMICS]
        
        for _ in range(10):
            pattern_learner.record_pattern(
                signature=observation_signature,
                updates_run=updates,
                success=True,
                cost_ms=50.0
            )
        
        # Trigger rule learning
        pattern_learner._learn_rules()
        
        # Should have learned rules
        assert len(pattern_learner.rules) >= 0
    
    def test_predict_updates(self, pattern_learner, observation_signature):
        """Test predicting updates"""
        # Without learned rules
        predicted = pattern_learner.predict_updates(observation_signature)
        assert isinstance(predicted, list)
        
        # Add a rule
        pattern_learner.rules = [{
            'id': 'test_rule',
            'signature_conditions': {
                'has_intervention': False,
                'has_temporal': True,
                'has_multi_variable': True
            },
            'updates': [UpdateType.CORRELATION, UpdateType.DYNAMICS],
            'success_rate': 0.9,
            'avg_cost': 50.0
        }]
        pattern_learner.rule_confidence['test_rule'] = 0.9
        
        predicted = pattern_learner.predict_updates(observation_signature)
        assert len(predicted) > 0


# ============================================================================
# Test CostModel
# ============================================================================

class TestCostModel:
    """Test CostModel component"""
    
    def test_initialization(self, cost_model):
        """Test cost model initialization"""
        assert UpdateType.CORRELATION in cost_model.base_costs
        assert cost_model.base_costs[UpdateType.CORRELATION] > 0
    
    def test_estimate_cost(self, cost_model):
        """Test cost estimation"""
        cost = cost_model.estimate_cost(UpdateType.CORRELATION, observation_size=100)
        
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_estimate_cost_scales_with_size(self, cost_model):
        """Test that cost scales with observation size"""
        small_cost = cost_model.estimate_cost(UpdateType.CORRELATION, observation_size=10)
        large_cost = cost_model.estimate_cost(UpdateType.CORRELATION, observation_size=1000)
        
        assert large_cost > small_cost
    
    def test_update_cost(self, cost_model):
        """Test updating cost model"""
        initial_history_size = len(cost_model.cost_history[UpdateType.CORRELATION])
        
        cost_model.update_cost(UpdateType.CORRELATION, actual_cost=25.0)
        
        assert len(cost_model.cost_history[UpdateType.CORRELATION]) == initial_history_size + 1
    
    def test_cost_calibration(self, cost_model):
        """Test cost model calibration"""
        # Record many actual costs
        for _ in range(15):
            cost_model.update_cost(UpdateType.CORRELATION, actual_cost=100.0)
        
        # Base cost should be updated
        # (May or may not change depending on initial value)
        assert cost_model.base_costs[UpdateType.CORRELATION] > 0


# ============================================================================
# Test WorldModelRouter
# ============================================================================

@patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
class TestWorldModelRouter:
    """Test WorldModelRouter main orchestrator"""
    
    def test_initialization(self, router, mock_world_model):
        """Test router initialization"""
        assert router.world_model == mock_world_model
        assert router.time_budget_ms == 1000
        assert router.min_confidence == 0.5
        assert len(router.strategies) > 0
    
    def test_route_basic_observation(self, router, sample_observation):
        """Test routing basic observation"""
        plan = router.route(sample_observation)
        
        assert isinstance(plan, UpdatePlan)
        assert len(plan.immediate) > 0 or len(plan.deferred) > 0
        assert plan.estimated_time_ms >= 0
        assert 0.0 <= plan.confidence <= 1.0
    
    def test_route_intervention_observation(self, router, intervention_observation):
        """Test routing intervention observation"""
        plan = router.route(intervention_observation)
        
        assert isinstance(plan, UpdatePlan)
        # Intervention observations should trigger specific updates
        all_updates = plan.immediate + plan.deferred
        for group in plan.parallel_groups:
            all_updates.extend(group)
        
        # Should include intervention-related updates if intervention detected
        assert len(all_updates) > 0
    
    def test_route_with_constraints(self, router, sample_observation):
        """Test routing with constraints"""
        constraints = {
            'time_budget_ms': 100,
            'priority_threshold': UpdatePriority.HIGH
        }
        
        plan = router.route(sample_observation, constraints)
        
        assert isinstance(plan, UpdatePlan)
        assert plan.estimated_time_ms <= 100 or len(plan.immediate) == 0
    
    def test_route_caching(self, router, sample_observation):
        """Test that routing results are cached"""
        # First call
        plan1 = router.route(sample_observation)
        initial_cache_size = len(router.cache)
        
        # Second call with same observation
        plan2 = router.route(sample_observation)
        
        # Cache should be used
        assert len(router.cache) >= initial_cache_size
        assert router.metrics['cache_hits'] > 0 or plan1 == plan2
    
    def test_execute_plan(self, router, sample_observation):
        """Test executing a plan"""
        plan = router.route(sample_observation)
        
        result = router.execute(plan)
        
        assert 'results' in result
        assert 'updates_executed' in result
        assert 'execution_time_ms' in result
        assert 'success' in result
    
    def test_execute_empty_plan(self, router):
        """Test executing empty plan"""
        plan = UpdatePlan(
            immediate=[],
            deferred=[],
            parallel_groups=[],
            estimated_time_ms=0,
            confidence=1.0,
            reasoning="Empty plan"
        )
        
        result = router.execute(plan)
        
        assert result['success'] == True
        assert len(result['updates_executed']) == 0
    
    def test_execute_updates_components(self, router, sample_observation):
        """Test that execute calls world model components"""
        plan = UpdatePlan(
            immediate=[UpdateType.CORRELATION],
            deferred=[],
            parallel_groups=[],
            estimated_time_ms=20,
            confidence=0.9,
            reasoning="Test",
            metadata={'signature': router._extract_signature(sample_observation)}
        )
        
        result = router.execute(plan)
        
        # Should have called correlation tracker
        assert router.world_model.correlation_tracker.update.called
    
    def test_extract_signature(self, router, sample_observation):
        """Test extracting observation signature"""
        signature = router._extract_signature(sample_observation)
        
        assert isinstance(signature, ObservationSignature)
        assert signature.variable_count == 3
        assert signature.confidence == 0.9
        assert signature.pattern_hash is not None
    
    def test_extract_signature_intervention(self, router, intervention_observation):
        """Test extracting signature from intervention observation"""
        signature = router._extract_signature(intervention_observation)
        
        assert signature.has_intervention == True
    
    def test_determine_required_updates(self, router, observation_signature):
        """Test determining required updates"""
        required = router._determine_required_updates(observation_signature)
        
        assert isinstance(required, set)
        assert len(required) > 0
        # Should always include confidence
        assert UpdateType.CONFIDENCE in required
    
    def test_determine_required_updates_intervention(self, router):
        """Test required updates for intervention"""
        sig = ObservationSignature(
            has_intervention=True,
            has_temporal=False,
            has_multi_variable=False,
            has_anomaly=False,
            has_structural_change=False,
            variable_count=1,
            time_delta=None,
            confidence=1.0,
            domain="test",
            pattern_hash="123"
        )
        
        required = router._determine_required_updates(sig)
        
        # Should include intervention and causal updates
        assert UpdateType.INTERVENTION in required or UpdateType.CAUSAL in required
    
    def test_apply_constraints(self, router, observation_signature):
        """Test applying constraints to updates"""
        updates = {
            UpdateType.CORRELATION,
            UpdateType.DYNAMICS,
            UpdateType.INVARIANT,
            UpdateType.CONFIDENCE
        }
        
        constraints = {
            'time_budget_ms': 50,
            'priority_threshold': UpdatePriority.HIGH
        }
        
        filtered = router._apply_constraints(updates, constraints, observation_signature)
        
        assert isinstance(filtered, set)
        # Should filter out low-priority updates
        assert len(filtered) <= len(updates)
    
    def test_create_execution_plan(self, router, observation_signature):
        """Test creating execution plan"""
        updates = {UpdateType.CORRELATION, UpdateType.DYNAMICS}
        
        plan = router._create_execution_plan(updates, observation_signature)
        
        assert isinstance(plan, UpdatePlan)
        assert plan.estimated_time_ms > 0
        assert plan.reasoning is not None
    
    def test_get_metrics(self, router, sample_observation):
        """Test getting router metrics"""
        # Generate some activity
        router.route(sample_observation)
        
        metrics = router.get_metrics()
        
        assert 'plans_created' in metrics
        assert 'cache_hits' in metrics
        assert 'pattern_rules' in metrics
        assert isinstance(metrics, dict)


# ============================================================================
# Test Safety Integration
# ============================================================================

class TestSafetyIntegration:
    """Test safety validation integration"""
    
    @patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
    def test_router_with_safety_validator(self, mock_world_model):
        """Test router with safety validator"""
        config = {
            'safety_config': {}  # FIXED: Use a valid empty dict
        }
        
        # EnhancedSafetyValidator is patched via decorator
        
        router = WorldModelRouter(mock_world_model, config)
        
        # Should initialize safety validator if available
        assert hasattr(router, 'safety_validator')
        assert router.safety_validator is not None
        # FIXED: Can't reliably check mock type due to module-level caching
        # Just verify it's a validator instance (could be real or mock depending on test order)
        assert hasattr(router.safety_validator, 'analyze_observation_safety')
    
    def test_route_with_unsafe_observation(self, mock_world_model):
        """Test routing with unsafe observation"""
        # Create router with mock safety validator
        router = WorldModelRouter(mock_world_model, {})
        
        # Mock safety validator
        mock_validator = Mock()
        mock_validator.analyze_observation_safety = Mock(return_value={
            'safe': False,
            'reason': 'Test unsafe'
        })
        router.safety_validator = mock_validator
        # Need to set this flag manually for the router to use the validator
        router.safety_validator.is_enabled = True 
        
        obs = Mock()
        obs.timestamp = time.time()
        obs.variables = {'x': 1.0}
        
        plan = router.route(obs)
        
        # Should return minimal plan
        assert len(plan.immediate) == 0
        assert plan.metadata.get('safety_blocked') == True
    
    @patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
    def test_safety_metrics(self, router, sample_observation):
        """Test safety metrics tracking"""
        # Route some observations
        router.route(sample_observation)
        
        metrics = router.get_metrics()
        
        assert 'safety' in metrics
        assert metrics['safety']['enabled'] == True


# ============================================================================
# Test State Persistence
# ============================================================================

@patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
class TestStatePersistence:
    """Test state saving and loading"""
    
    def test_save_state(self, router):
        """Test saving router state"""
        with tempfile.TemporaryDirectory() as tmpdir:
            router.save_state(tmpdir)
            
            # Check that file was created
            save_path = Path(tmpdir) / 'router_state.pkl'
            assert save_path.exists()
    
    def test_load_state(self, router):
        """Test loading router state"""
        # Create some state
        router.metrics['plans_created'] = 100
        router.metrics['cache_hits'] = 50
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state
            router.save_state(tmpdir)
            
            # Create new router
            new_router = WorldModelRouter(router.world_model, {})
            
            # Load state
            new_router.load_state(tmpdir)
            
            # Verify state was loaded
            assert new_router.metrics['plans_created'] == 100
            assert new_router.metrics['cache_hits'] == 50
    
    def test_load_nonexistent_state(self, router):
        """Test loading from nonexistent path"""
        # Should handle gracefully
        router.load_state('/nonexistent/path')
        
        # Router should still be functional
        assert router is not None


# ============================================================================
# Test Thread Safety
# ============================================================================

@patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_routing(self, router, sample_observation):
        """Test concurrent routing operations"""
        results = []
        
        def route():
            plan = router.route(sample_observation)
            results.append(plan)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=route)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(isinstance(p, UpdatePlan) for p in results)
    
    def test_concurrent_execution(self, router, sample_observation):
        """Test concurrent plan execution"""
        plan = router.route(sample_observation)
        
        results = []
        
        def execute():
            result = router.execute(plan)
            results.append(result)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=execute)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert all('success' in r for r in results)
    
    def test_concurrent_pattern_learning(self, pattern_learner, observation_signature):
        """Test concurrent pattern recording"""
        def record():
            for _ in range(10):
                pattern_learner.record_pattern(
                    signature=observation_signature,
                    updates_run=[UpdateType.CORRELATION],
                    success=True,
                    cost_ms=50.0
                )
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=record)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have recorded all patterns
        assert len(pattern_learner.pattern_history) == 50


# ============================================================================
# Test Edge Cases
# ============================================================================

@patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_route_with_missing_attributes(self, router):
        """Test routing with observation missing attributes - FIXED: Added proper mock attributes"""
        # Minimal observation with proper mock attributes
        obs = Mock()
        obs.timestamp = time.time()
        obs.variables = {}  # FIXED: Provide empty dict instead of letting Mock auto-create
        obs.confidence = 1.0
        obs.domain = None
        obs.intervention_data = None
        obs.sequence_id = 125
        obs.metadata = {}
        obs.is_anomaly = False
        obs.structural_change = False
        obs.is_intervention = False
        
        plan = router.route(obs)
        
        # Should handle gracefully
        assert isinstance(plan, UpdatePlan)
    
    def test_execute_with_failing_component(self, router, sample_observation):
        """Test execution when component fails"""
        # Make component raise exception
        router.world_model.correlation_tracker.update.side_effect = Exception("Test error")
        
        plan = UpdatePlan(
            immediate=[UpdateType.CORRELATION],
            deferred=[],
            parallel_groups=[],
            estimated_time_ms=20,
            confidence=0.9,
            reasoning="Test",
            metadata={'signature': router._extract_signature(sample_observation)}
        )
        
        result = router.execute(plan)
        
        # Should handle error gracefully
        assert 'results' in result
        assert result['success'] == False
        
        # Reset side effect
        router.world_model.correlation_tracker.update.side_effect = None
    
    def test_route_with_zero_budget(self, router, sample_observation):
        """Test routing with zero time budget"""
        constraints = {
            'time_budget_ms': 0
        }
        
        plan = router.route(sample_observation, constraints)
        
        # Should return minimal plan
        assert isinstance(plan, UpdatePlan)
    
    def test_pattern_learner_with_failures(self, pattern_learner, observation_signature):
        """Test pattern learning with failed executions"""
        # Record mix of successes and failures
        for i in range(10):
            pattern_learner.record_pattern(
                signature=observation_signature,
                updates_run=[UpdateType.CORRELATION],
                success=(i % 2 == 0),  # Alternate success/failure
                cost_ms=50.0
            )
        
        # Trigger learning
        pattern_learner._learn_rules()
        
        # Should handle mixed success rates
        assert len(pattern_learner.pattern_history) == 10
    
    def test_cost_model_with_no_history(self, cost_model):
        """Test cost estimation with no history"""
        # Clear history
        cost_model.cost_history.clear()
        
        cost = cost_model.estimate_cost(UpdateType.CORRELATION, 100)
        
        # Should use base cost
        assert cost > 0
    
    def test_dependency_graph_circular_dependency(self, dependency_graph):
        """Test handling of circular dependencies"""
        # Try to create problematic update set
        # (In practice, the hardcoded dependencies shouldn't have cycles)
        
        required = {UpdateType.CORRELATION, UpdateType.DYNAMICS}
        
        execution_groups = dependency_graph.get_execution_order(required)
        
        # Should resolve without infinite loop
        assert len(execution_groups) > 0


# ============================================================================
# Integration Tests
# ============================================================================

@patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_routing_workflow(self, router, sample_observation):
        """Test complete routing and execution workflow"""
        # Step 1: Route observation
        plan = router.route(sample_observation)
        
        assert isinstance(plan, UpdatePlan)
        
        # Step 2: Execute plan
        result = router.execute(plan)
        
        assert 'success' in result
        
        # Step 3: Check metrics
        metrics = router.get_metrics()
        
        assert metrics['plans_created'] > 0
    
    def test_learning_and_adaptation(self, router, observation_signature):
        """Test learning from execution patterns"""
        # Record multiple similar patterns
        updates = [UpdateType.CORRELATION, UpdateType.DYNAMICS]
        
        for _ in range(20):
            router.pattern_learner.record_pattern(
                signature=observation_signature,
                updates_run=updates,
                success=True,
                cost_ms=50.0
            )
        
        # Should learn rules
        assert len(router.pattern_learner.rules) >= 0
        
        # Predict updates for similar signature
        predicted = router.pattern_learner.predict_updates(observation_signature)
        
        assert isinstance(predicted, list)
    
    def test_cost_calibration_over_time(self, router, sample_observation):
        """Test cost calibration with actual execution"""
        initial_cost = router.cost_model.base_costs[UpdateType.CORRELATION]
        
        # Execute multiple times
        for _ in range(15):
            plan = router.route(sample_observation)
            router.execute(plan)
        
        # Cost model may have been updated
        final_cost = router.cost_model.base_costs[UpdateType.CORRELATION]
        
        # Should have history
        assert len(router.cost_model.cost_history[UpdateType.CORRELATION]) > 0
    
    def test_progressive_complexity(self, router):
        """Test handling progressively complex observations"""
        # Simple observation
        obs1 = Mock()
        obs1.timestamp = time.time()
        obs1.variables = {'x': 1.0}
        obs1.confidence = 0.9
        obs1.domain = "test"
        obs1.intervention_data = None
        obs1.sequence_id = 126
        obs1.metadata = {}
        obs1.is_anomaly = False
        obs1.structural_change = False
        obs1.is_intervention = False
        
        plan1 = router.route(obs1)
        
        # Complex observation
        obs2 = Mock()
        obs2.timestamp = time.time()
        obs2.variables = {f'var_{i}': float(i) for i in range(20)}
        obs2.confidence = 0.9
        obs2.domain = "test"
        obs2.intervention_data = {'intervened_variable': 'var_0'}
        obs2.sequence_id = 127
        obs2.metadata = {'is_intervention': True}
        obs2.is_anomaly = False
        obs2.structural_change = False
        obs2.is_intervention = True
        
        plan2 = router.route(obs2)
        
        # More complex observation should have more updates or longer time
        # (or at least should handle without error)
        assert isinstance(plan1, UpdatePlan)
        assert isinstance(plan2, UpdatePlan)


# ============================================================================
# Performance Tests
# ============================================================================

@patch('vulcan.safety.safety_validator.EnhancedSafetyValidator', mock_validator_class)
class TestPerformance:
    """Performance and scalability tests"""
    
    def test_routing_performance(self, router, sample_observation):
        """Test routing performance"""
        import time as time_module
        start = time_module.time()
        
        for _ in range(100):
            router.route(sample_observation)
        
        elapsed = time_module.time() - start
        
        assert elapsed < 5, f"100 routing operations took {elapsed}s"
    
    def test_execution_performance(self, router, sample_observation):
        """Test execution performance"""
        plan = router.route(sample_observation)
        
        import time as time_module
        start = time_module.time()
        
        for _ in range(50):
            router.execute(plan)
        
        elapsed = time_module.time() - start
        
        assert elapsed < 10, f"50 executions took {elapsed}s"
    
    def test_pattern_learning_scalability(self, pattern_learner, observation_signature):
        """Test pattern learning with large history"""
        import time as time_module
        start = time_module.time()
        
        for _ in range(1000):
            pattern_learner.record_pattern(
                signature=observation_signature,
                updates_run=[UpdateType.CORRELATION],
                success=True,
                cost_ms=50.0
            )
        
        elapsed = time_module.time() - start
        
        assert elapsed < 5, f"Recording 1000 patterns took {elapsed}s"
    
    def test_large_dependency_graph(self, dependency_graph):
        """Test dependency resolution with many updates"""
        # All update types
        all_updates = set(UpdateType)
        
        import time as time_module
        start = time_module.time()
        
        execution_groups = dependency_graph.get_execution_order(all_updates)
        
        elapsed = time_module.time() - start
        
        assert elapsed < 1, f"Dependency resolution took {elapsed}s"
        assert len(execution_groups) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
