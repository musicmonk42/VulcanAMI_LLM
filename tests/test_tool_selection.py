"""
Comprehensive test suite for tool_selection.yaml
Validates optimization parameters, calibration settings, and strategy configuration.

Run with:
    pytest test_tool_selection.py -v
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set


# Test Fixtures
@pytest.fixture
def tool_selection_config():
    """Load the tool selection YAML file."""
    config_path = Path(__file__).parent / "configs" / "tool_selection.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / ".." / "configs" / "tool_selection.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def defaults(tool_selection_config):
    """Extract defaults section."""
    return tool_selection_config.get('defaults', {})


@pytest.fixture
def utility_weights(defaults):
    """Extract utility weights."""
    return defaults.get('utility_weights', {})


@pytest.fixture
def calibration(defaults):
    """Extract calibration settings."""
    return defaults.get('calibration', {})


@pytest.fixture
def portfolio_strategies(defaults):
    """Extract portfolio strategies."""
    return defaults.get('portfolio_strategies', {})


@pytest.fixture
def cost_model(defaults):
    """Extract cost model settings."""
    return defaults.get('cost_model', {})


# Test YAML Structure
class TestYAMLStructure:
    def test_yaml_loads_successfully(self, tool_selection_config):
        """Test that YAML file loads without errors."""
        assert tool_selection_config is not None
        assert isinstance(tool_selection_config, dict)
    
    def test_defaults_section_exists(self, tool_selection_config):
        """Test that defaults section exists."""
        assert 'defaults' in tool_selection_config
        assert isinstance(tool_selection_config['defaults'], dict)
    
    def test_top_level_structure(self, tool_selection_config):
        """Test that top-level structure is correct."""
        # Should have 'defaults' at minimum
        assert len(tool_selection_config) > 0


# Test Utility Weights
class TestUtilityWeights:
    def test_utility_weights_exist(self, defaults):
        """Test that utility_weights section exists."""
        assert 'utility_weights' in defaults
    
    def test_required_weight_fields(self, utility_weights):
        """Test that all required weight fields are present."""
        required_weights = ['quality', 'time_penalty', 'energy_penalty', 'risk_penalty']
        for weight in required_weights:
            assert weight in utility_weights, f"Missing weight: {weight}"
    
    def test_weights_are_numeric(self, utility_weights):
        """Test that all weights are numeric."""
        for key, value in utility_weights.items():
            assert isinstance(value, (int, float)), \
                f"Weight {key} should be numeric, got {type(value)}"
    
    def test_weights_are_positive(self, utility_weights):
        """Test that all weights are positive."""
        for key, value in utility_weights.items():
            assert value > 0, f"Weight {key} should be positive, got {value}"
    
    def test_weights_are_reasonable(self, utility_weights):
        """Test that weights are in reasonable range."""
        for key, value in utility_weights.items():
            assert 0 < value <= 100, \
                f"Weight {key} should be between 0 and 100, got {value}"
    
    def test_all_weights_equal_suggests_tuning_needed(self, utility_weights):
        """Test if all weights are equal (suggests default values)."""
        values = list(utility_weights.values())
        unique_values = set(values)
        
        if len(unique_values) == 1:
            pytest.skip(
                f"Warning: All weights are identical ({values[0]}). "
                "Consider tuning weights based on workload priorities."
            )
    
    def test_weight_ratios_make_sense(self, utility_weights):
        """Test that weight ratios are reasonable."""
        # Quality and risk should typically be weighted higher than penalties
        quality = utility_weights.get('quality', 1.0)
        risk = utility_weights.get('risk_penalty', 1.0)
        time = utility_weights.get('time_penalty', 1.0)
        energy = utility_weights.get('energy_penalty', 1.0)
        
        # This is a guideline, not a hard rule
        if quality == time == energy == risk:
            pytest.skip(
                "Warning: Consider whether equal weights reflect actual priorities. "
                "Typically quality/risk are weighted higher than time/energy penalties."
            )


# Test Calibration Settings
class TestCalibration:
    def test_calibration_exists(self, defaults):
        """Test that calibration section exists."""
        assert 'calibration' in defaults
    
    def test_min_samples_defined(self, calibration):
        """Test that min_samples is defined."""
        assert 'min_samples' in calibration
    
    def test_min_samples_positive(self, calibration):
        """Test that min_samples is positive."""
        min_samples = calibration['min_samples']
        assert min_samples > 0, "min_samples must be positive"
    
    def test_min_samples_reasonable(self, calibration):
        """Test that min_samples is reasonable."""
        min_samples = calibration['min_samples']
        assert 10 <= min_samples <= 10000, \
            f"min_samples should be 10-10000, got {min_samples}"
    
    def test_retrain_interval_defined(self, calibration):
        """Test that retrain_interval is defined."""
        assert 'retrain_interval' in calibration
    
    def test_retrain_interval_positive(self, calibration):
        """Test that retrain_interval is positive."""
        retrain = calibration['retrain_interval']
        assert retrain > 0, "retrain_interval must be positive"
    
    def test_retrain_interval_reasonable(self, calibration):
        """Test that retrain_interval is reasonable."""
        retrain = calibration['retrain_interval']
        assert 1 <= retrain <= 100000, \
            f"retrain_interval should be 1-100000, got {retrain}"
    
    def test_temperature_scaling_defined(self, calibration):
        """Test that temperature_scaling is defined."""
        assert 'temperature_scaling' in calibration
    
    def test_temperature_scaling_is_boolean(self, calibration):
        """Test that temperature_scaling is boolean."""
        temp_scaling = calibration['temperature_scaling']
        assert isinstance(temp_scaling, bool), \
            "temperature_scaling should be boolean"
    
    def test_isotonic_regression_defined(self, calibration):
        """Test that isotonic_regression is defined."""
        assert 'isotonic_regression' in calibration
    
    def test_isotonic_regression_is_boolean(self, calibration):
        """Test that isotonic_regression is boolean."""
        iso_reg = calibration['isotonic_regression']
        assert isinstance(iso_reg, bool), \
            "isotonic_regression should be boolean"
    
    def test_min_samples_less_than_retrain_interval(self, calibration):
        """Test that min_samples is compatible with retrain_interval."""
        min_samples = calibration['min_samples']
        retrain = calibration['retrain_interval']
        
        if min_samples > retrain:
            pytest.skip(
                f"Warning: min_samples ({min_samples}) > retrain_interval ({retrain}). "
                "You'll accumulate samples before first retraining."
            )
    
    def test_calibration_methods_enabled(self, calibration):
        """Test that at least one calibration method is enabled."""
        temp_scaling = calibration.get('temperature_scaling', False)
        iso_reg = calibration.get('isotonic_regression', False)
        
        assert temp_scaling or iso_reg, \
            "At least one calibration method should be enabled"


# Test Portfolio Strategies
class TestPortfolioStrategies:
    def test_portfolio_strategies_exist(self, defaults):
        """Test that portfolio_strategies section exists."""
        assert 'portfolio_strategies' in defaults
    
    def test_enabled_strategies_defined(self, portfolio_strategies):
        """Test that enabled strategies are defined."""
        assert 'enabled' in portfolio_strategies
    
    def test_enabled_is_list(self, portfolio_strategies):
        """Test that enabled is a list."""
        enabled = portfolio_strategies['enabled']
        assert isinstance(enabled, list), "enabled should be a list"
    
    def test_at_least_one_strategy_enabled(self, portfolio_strategies):
        """Test that at least one strategy is enabled."""
        enabled = portfolio_strategies['enabled']
        assert len(enabled) > 0, "At least one strategy should be enabled"
    
    def test_strategy_names_valid(self, portfolio_strategies):
        """Test that strategy names are valid."""
        valid_strategies = {
            'single',
            'speculative_parallel',
            'sequential_refinement',
            'ensemble',
            'cascade',
            'hybrid'
        }
        
        enabled = portfolio_strategies['enabled']
        for strategy in enabled:
            assert strategy in valid_strategies, \
                f"Invalid strategy: {strategy}. Valid: {valid_strategies}"
    
    def test_no_duplicate_strategies(self, portfolio_strategies):
        """Test that no strategies are duplicated."""
        enabled = portfolio_strategies['enabled']
        assert len(enabled) == len(set(enabled)), \
            f"Duplicate strategies found: {enabled}"
    
    def test_strategy_consistency(self, portfolio_strategies):
        """Test that strategy combination makes sense."""
        enabled = portfolio_strategies['enabled']
        
        # If both parallel and sequential are enabled, warn
        has_parallel = 'speculative_parallel' in enabled
        has_sequential = 'sequential_refinement' in enabled
        
        if has_parallel and has_sequential:
            pytest.skip(
                "Info: Both parallel and sequential strategies enabled. "
                "Ensure scheduler can choose appropriately based on workload."
            )
    
    def test_single_strategy_included(self, portfolio_strategies):
        """Test that single strategy is included."""
        enabled = portfolio_strategies['enabled']
        assert 'single' in enabled, \
            "'single' strategy should always be enabled as baseline"


# Test Cost Model
class TestCostModel:
    def test_cost_model_exists(self, defaults):
        """Test that cost_model section exists."""
        assert 'cost_model' in defaults
    
    def test_track_variance_defined(self, cost_model):
        """Test that track_variance is defined."""
        assert 'track_variance' in cost_model
    
    def test_track_variance_is_boolean(self, cost_model):
        """Test that track_variance is boolean."""
        track_var = cost_model['track_variance']
        assert isinstance(track_var, bool), "track_variance should be boolean"
    
    def test_cold_start_penalty_defined(self, cost_model):
        """Test that cold_start_penalty_ms is defined."""
        assert 'cold_start_penalty_ms' in cost_model
    
    def test_cold_start_penalty_positive(self, cost_model):
        """Test that cold_start_penalty_ms is positive."""
        penalty = cost_model['cold_start_penalty_ms']
        assert penalty >= 0, "cold_start_penalty_ms should be non-negative"
    
    def test_cold_start_penalty_reasonable(self, cost_model):
        """Test that cold_start_penalty_ms is reasonable."""
        penalty = cost_model['cold_start_penalty_ms']
        assert 0 <= penalty <= 10000, \
            f"cold_start_penalty_ms should be 0-10000ms, got {penalty}"
    
    def test_health_check_interval_defined(self, cost_model):
        """Test that health_check_interval is defined."""
        assert 'health_check_interval' in cost_model
    
    def test_health_check_interval_positive(self, cost_model):
        """Test that health_check_interval is positive."""
        interval = cost_model['health_check_interval']
        assert interval > 0, "health_check_interval must be positive"
    
    def test_health_check_interval_reasonable(self, cost_model):
        """Test that health_check_interval is reasonable."""
        interval = cost_model['health_check_interval']
        assert 1 <= interval <= 3600, \
            f"health_check_interval should be 1-3600 seconds, got {interval}"


# Test Missing Features
class TestMissingFeatures:
    def test_no_workload_classification(self, tool_selection_config):
        """Test that workload classification is missing."""
        assert 'workload_types' not in tool_selection_config, \
            "Workload classification not yet implemented"
    
    def test_no_learning_rate(self, calibration):
        """Test that learning rate is missing."""
        assert 'learning_rate' not in calibration, \
            "Learning rate configuration not yet implemented"
    
    def test_no_exploration_strategy(self, tool_selection_config):
        """Test that exploration strategy is missing."""
        assert 'exploration' not in tool_selection_config, \
            "Exploration strategy not yet implemented"
    
    def test_no_cost_budget(self, tool_selection_config):
        """Test that cost budget is missing."""
        assert 'budget_constraints' not in tool_selection_config, \
            "Budget constraints not yet implemented"
    
    def test_no_fallback_strategy(self, portfolio_strategies):
        """Test that fallback strategy is missing."""
        assert 'fallback' not in portfolio_strategies, \
            "Fallback strategy not yet implemented"
    
    def test_no_timeout_config(self, tool_selection_config):
        """Test that timeout configuration is missing."""
        assert 'timeout_ms' not in tool_selection_config.get('defaults', {}), \
            "Timeout configuration not yet implemented"


# Test Data Types
class TestDataTypes:
    def test_all_numeric_fields_are_numeric(self, defaults):
        """Test that numeric fields have correct types."""
        def check_numeric_recursive(d, path=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    check_numeric_recursive(value, f"{path}.{key}")
                elif key in ['min_samples', 'retrain_interval', 'cold_start_penalty_ms', 
                            'health_check_interval']:
                    assert isinstance(value, (int, float)), \
                        f"{path}.{key} should be numeric, got {type(value)}"
        
        check_numeric_recursive(defaults)
    
    def test_all_boolean_fields_are_boolean(self, defaults):
        """Test that boolean fields have correct types."""
        def check_boolean_recursive(d, path=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    check_boolean_recursive(value, f"{path}.{key}")
                elif key in ['temperature_scaling', 'isotonic_regression', 'track_variance']:
                    assert isinstance(value, bool), \
                        f"{path}.{key} should be boolean, got {type(value)}"
        
        check_boolean_recursive(defaults)


# Test Logical Consistency
class TestLogicalConsistency:
    def test_calibration_sample_size_sufficient(self, calibration):
        """Test that min_samples is sufficient for calibration."""
        min_samples = calibration['min_samples']
        
        if min_samples < 30:
            pytest.skip(
                f"Warning: min_samples ({min_samples}) is small. "
                "Consider 30+ samples for reliable calibration."
            )
    
    def test_health_check_not_too_frequent(self, cost_model):
        """Test that health checks aren't too frequent."""
        interval = cost_model['health_check_interval']
        
        if interval < 10:
            pytest.skip(
                f"Warning: health_check_interval ({interval}s) is very frequent. "
                "This may add overhead."
            )
    
    def test_cold_start_penalty_reflects_reality(self, cost_model):
        """Test that cold start penalty is realistic."""
        penalty = cost_model['cold_start_penalty_ms']
        
        if penalty == 0:
            pytest.skip(
                "Warning: cold_start_penalty_ms is 0. "
                "Most systems have some cold start cost."
            )
        
        if penalty < 10:
            pytest.skip(
                f"Warning: cold_start_penalty_ms ({penalty}ms) seems low. "
                "Typical cold starts are 50-500ms."
            )


# Test Integration with Hardware Profiles
class TestHardwareIntegration:
    def test_utility_function_compatible(self, utility_weights):
        """Test that utility function is compatible with hardware metrics."""
        # Utility function should consider: quality, time, energy, risk
        # Hardware profiles provide: latency, throughput, energy_per_op
        
        required_weights = ['quality', 'time_penalty', 'energy_penalty']
        for weight in required_weights:
            assert weight in utility_weights, \
                f"Utility function missing {weight} for hardware selection"
    
    def test_risk_penalty_defined_for_exotic_hardware(self, utility_weights):
        """Test that risk penalty exists for exotic hardware."""
        assert 'risk_penalty' in utility_weights, \
            "risk_penalty needed to account for experimental hardware reliability"


# Test Strategy Configuration
class TestStrategyConfiguration:
    def test_strategies_match_expected_patterns(self, portfolio_strategies):
        """Test that strategy names follow expected patterns."""
        enabled = portfolio_strategies['enabled']
        
        for strategy in enabled:
            # Should be lowercase with underscores
            assert strategy.islower(), f"Strategy {strategy} should be lowercase"
            assert ' ' not in strategy, f"Strategy {strategy} should not contain spaces"
    
    def test_strategy_parameters_missing(self, portfolio_strategies):
        """Test that strategy parameters are missing (document limitation)."""
        enabled = portfolio_strategies['enabled']
        
        # Each strategy could have parameters
        for strategy in enabled:
            if strategy == 'speculative_parallel':
                assert 'max_parallel' not in portfolio_strategies, \
                    "Strategy parameters not yet implemented"
            if strategy == 'sequential_refinement':
                assert 'max_iterations' not in portfolio_strategies, \
                    "Strategy parameters not yet implemented"


# Test Optimization Parameters
class TestOptimizationParameters:
    def test_utility_weights_sum_not_required(self, utility_weights):
        """Test that weights don't need to sum to 1 (can be scaled)."""
        total = sum(utility_weights.values())
        # Any positive sum is valid - will be normalized at runtime
        assert total > 0
    
    def test_can_compute_utility_score(self, utility_weights):
        """Test that utility score can be computed."""
        # Simulate hardware metrics
        simulated_metrics = {
            'quality_score': 0.8,
            'time_cost': 100,  # ms
            'energy_cost': 50,  # nJ
            'risk_score': 0.2
        }
        
        # Compute utility (simplified)
        quality_component = utility_weights['quality'] * simulated_metrics['quality_score']
        time_component = utility_weights['time_penalty'] / simulated_metrics['time_cost']
        energy_component = utility_weights['energy_penalty'] / simulated_metrics['energy_cost']
        risk_component = utility_weights['risk_penalty'] * (1 - simulated_metrics['risk_score'])
        
        utility = quality_component + time_component + energy_component + risk_component
        
        assert utility > 0, "Utility score should be computable and positive"


# Test Recommended Enhancements
class TestRecommendedEnhancements:
    def test_should_add_workload_classification(self, tool_selection_config):
        """Document that workload classification should be added."""
        recommended_config = {
            'workload_types': {
                'latency_sensitive': {
                    'utility_weights': {
                        'quality': 1.0,
                        'time_penalty': 2.0,
                        'energy_penalty': 0.5,
                        'risk_penalty': 1.5
                    }
                },
                'throughput_intensive': {
                    'utility_weights': {
                        'quality': 1.5,
                        'time_penalty': 0.5,
                        'energy_penalty': 0.3,
                        'risk_penalty': 1.0
                    }
                }
            }
        }
        
        # Document this is missing
        assert 'workload_types' not in tool_selection_config
    
    def test_should_add_exploration_strategy(self, tool_selection_config):
        """Document that exploration strategy should be added."""
        recommended_config = {
            'exploration': {
                'epsilon': 0.1,
                'decay_rate': 0.99,
                'min_epsilon': 0.01
            }
        }
        
        assert 'exploration' not in tool_selection_config
    
    def test_should_add_budget_constraints(self, tool_selection_config):
        """Document that budget constraints should be added."""
        recommended_config = {
            'budget_constraints': {
                'max_cost_per_request_usd': 0.01,
                'max_latency_ms': 1000,
                'max_energy_uj': 10000
            }
        }
        
        assert 'budget_constraints' not in tool_selection_config
    
    def test_should_add_strategy_parameters(self, portfolio_strategies):
        """Document that strategy parameters should be added."""
        recommended_config = {
            'strategy_config': {
                'speculative_parallel': {
                    'max_parallel': 3,
                    'timeout_multiplier': 1.5
                },
                'sequential_refinement': {
                    'max_iterations': 5,
                    'improvement_threshold': 0.1
                }
            }
        }
        
        assert 'strategy_config' not in portfolio_strategies


# Test Configuration Completeness
class TestConfigurationCompleteness:
    def test_all_sections_present(self, defaults):
        """Test that all major sections are present."""
        required_sections = [
            'utility_weights',
            'calibration',
            'portfolio_strategies',
            'cost_model'
        ]
        
        for section in required_sections:
            assert section in defaults, f"Missing section: {section}"
    
    def test_no_empty_sections(self, defaults):
        """Test that no sections are empty."""
        for key, value in defaults.items():
            if isinstance(value, dict):
                assert len(value) > 0, f"Section {key} is empty"
            elif isinstance(value, list):
                assert len(value) > 0, f"Section {key} is empty"


# Integration Tests
class TestIntegration:
    def test_config_supports_scheduler(self, tool_selection_config):
        """Test that config provides all info needed for a scheduler."""
        defaults = tool_selection_config['defaults']
        
        # Scheduler needs:
        # 1. Utility weights to score options
        assert 'utility_weights' in defaults
        
        # 2. Calibration to improve predictions
        assert 'calibration' in defaults
        
        # 3. Portfolio strategies to choose execution mode
        assert 'portfolio_strategies' in defaults
        
        # 4. Cost model to estimate costs
        assert 'cost_model' in defaults
    
    def test_can_simulate_tool_selection(self, utility_weights, portfolio_strategies):
        """Test that tool selection can be simulated."""
        # Simulate tools
        tools = {
            'tool_a': {'quality': 0.9, 'time': 100, 'energy': 50, 'risk': 0.1},
            'tool_b': {'quality': 0.7, 'time': 50, 'energy': 30, 'risk': 0.2}
        }
        
        # Compute scores
        scores = {}
        for tool_name, metrics in tools.items():
            score = (
                utility_weights['quality'] * metrics['quality'] -
                utility_weights['time_penalty'] * metrics['time'] / 1000 -
                utility_weights['energy_penalty'] * metrics['energy'] / 1000 +
                utility_weights['risk_penalty'] * (1 - metrics['risk'])
            )
            scores[tool_name] = score
        
        # Should be able to rank tools
        best_tool = max(scores, key=scores.get)
        assert best_tool in tools
    
    def test_strategy_selection_possible(self, portfolio_strategies):
        """Test that strategy selection is possible."""
        enabled = portfolio_strategies['enabled']
        
        # For a given workload, should be able to select strategy
        workload_characteristics = {
            'uncertainty': 'high',
            'latency_requirement': 'strict',
            'cost_budget': 'flexible'
        }
        
        # High uncertainty + flexible budget -> speculative_parallel
        if 'speculative_parallel' in enabled:
            selected_strategy = 'speculative_parallel'
        else:
            selected_strategy = enabled[0]
        
        assert selected_strategy in enabled


# Test Documentation and Clarity
class TestDocumentation:
    def test_config_is_self_documenting(self, tool_selection_config):
        """Test that configuration keys are self-explanatory."""
        # Read raw YAML to check for comments
        config_path = Path(__file__).parent / "configs" / "tool_selection.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent / ".." / "configs" / "tool_selection.yaml"
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Should have some comments explaining configuration
        comment_count = content.count('#')
        
        if comment_count < 5:
            pytest.skip(
                f"Warning: Only {comment_count} comments found. "
                "Add more documentation to explain configuration choices."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])