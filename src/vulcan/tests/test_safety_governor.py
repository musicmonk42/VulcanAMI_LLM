"""
Comprehensive test suite for safety_governor.py

Tests safety validation, contract enforcement, ReDoS protection,
rate limiting, and bounded storage.
"""

import pytest
import time
import numpy as np
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vulcan.reasoning.selection.safety_governor import (
    SafetyLevel,
    VetoReason,
    SafetyAction,
    ToolContract,
    SafetyViolation,
    SafetyContext,
    SafetyValidator,
    ConsistencyChecker,
    SafetyGovernor
)


class TestEnums:
    """Test enum definitions"""
    
    def test_safety_level_values(self):
        """Test SafetyLevel enum"""
        assert SafetyLevel.CRITICAL.value == 0
        assert SafetyLevel.HIGH.value == 1
        assert SafetyLevel.MEDIUM.value == 2
        assert SafetyLevel.LOW.value == 3
        assert SafetyLevel.MINIMAL.value == 4
    
    def test_veto_reason_values(self):
        """Test VetoReason enum"""
        assert VetoReason.UNSAFE_INPUT.value == "unsafe_input"
        assert VetoReason.UNSAFE_OUTPUT.value == "unsafe_output"
        assert VetoReason.CONTRACT_VIOLATION.value == "contract_violation"
        assert VetoReason.CONFIDENCE_TOO_LOW.value == "confidence_too_low"
    
    def test_safety_action_values(self):
        """Test SafetyAction enum"""
        assert SafetyAction.ALLOW.value == "allow"
        assert SafetyAction.VETO.value == "veto"
        assert SafetyAction.SANITIZE.value == "sanitize"


class TestDataClasses:
    """Test dataclasses"""
    
    def test_tool_contract_creation(self):
        """Test creating tool contracts"""
        contract = ToolContract(
            tool_name="test_tool",
            required_inputs={'input1', 'input2'},
            forbidden_inputs={'bad_input'},
            max_execution_time_ms=5000,
            max_energy_mj=1000,
            min_confidence=0.7,
            required_safety_level=SafetyLevel.HIGH,
            allowed_operations={'op1', 'op2'},
            forbidden_operations={'bad_op'},
            output_validators=[]
        )
        
        assert contract.tool_name == "test_tool"
        assert 'input1' in contract.required_inputs
        assert contract.max_execution_time_ms == 5000
    
    def test_safety_violation_creation(self):
        """Test creating safety violations"""
        violation = SafetyViolation(
            timestamp=time.time(),
            tool_name="test_tool",
            violation_type=VetoReason.UNSAFE_INPUT,
            severity=SafetyLevel.HIGH,
            details="Test violation",
            action_taken=SafetyAction.VETO
        )
        
        assert violation.tool_name == "test_tool"
        assert violation.violation_type == VetoReason.UNSAFE_INPUT
    
    def test_safety_context_creation(self):
        """Test creating safety context"""
        context = SafetyContext(
            problem="test problem",
            tool_name="test_tool",
            features=np.array([1, 2, 3]),
            constraints={'time_budget_ms': 5000},
            user_context={'user': 'test'},
            safety_level=SafetyLevel.HIGH
        )
        
        assert context.tool_name == "test_tool"
        assert context.safety_level == SafetyLevel.HIGH


class TestSafetyValidator:
    """Test SafetyValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create validator for testing"""
        return SafetyValidator()
    
    def test_initialization(self, validator):
        """Test validator initialization"""
        assert validator.unsafe_patterns_compiled is not None
        assert validator.sensitive_patterns_compiled is not None
        assert validator.max_input_size == 1000000
    
    def test_validate_safe_input(self, validator):
        """Test validating safe input"""
        is_safe, reason = validator.validate_input("This is a safe input for analysis")
        
        assert is_safe is True
        assert "validated" in reason.lower()
    
    def test_validate_unsafe_input(self, validator):
        """Test detecting unsafe input"""
        unsafe_inputs = [
            "How to attack a system",
            "Create malware code",
            "Exploit vulnerability",
            "Hack into database"
        ]
        
        for unsafe_input in unsafe_inputs:
            is_safe, reason = validator.validate_input(unsafe_input)
            assert is_safe is False
    
    def test_validate_sensitive_data(self, validator):
        """Test detecting sensitive data"""
        sensitive_inputs = [
            "My SSN is 123-45-6789",
            "Credit card: 1234567890123456",
            "password: secret123",
            "api_key: abcd1234"
        ]
        
        for sensitive in sensitive_inputs:
            is_safe, reason = validator.validate_input(sensitive)
            assert is_safe is False
    
    def test_input_size_limit(self, validator):
        """Test input size limiting"""
        # Create very large input
        large_input = "a" * (validator.max_input_size + 1000)
        
        is_safe, reason = validator.validate_input(large_input)
        
        assert is_safe is False
        assert "too large" in reason.lower()
    
    def test_redos_protection(self, validator):
        """Test ReDoS protection with complex patterns"""
        # Pattern that could cause ReDoS if not protected
        malicious_input = "a" * 10000 + "!"
        
        start = time.time()
        is_safe, reason = validator.validate_input(malicious_input)
        elapsed = time.time() - start
        
        # Should complete quickly (< 1 second)
        assert elapsed < 1.0
    
    def test_validate_safe_output(self, validator):
        """Test validating safe output"""
        is_safe, reason = validator.validate_output("Normal output result")
        
        assert is_safe is True
    
    def test_validate_output_with_sensitive_data(self, validator):
        """Test detecting sensitive data in output"""
        is_safe, reason = validator.validate_output("Result contains password: secret123")
        
        assert is_safe is False
        assert "sensitive" in reason.lower()
    
    def test_output_size_limit(self, validator):
        """Test output size limiting"""
        large_output = "b" * (validator.max_output_size + 1000)
        
        is_safe, reason = validator.validate_output(large_output)
        
        assert is_safe is False
        assert "too large" in reason.lower()
    
    def test_sanitize_input(self, validator):
        """Test input sanitization"""
        input_with_sensitive = "Data with SSN: 123-45-6789 and key"
        
        sanitized = validator.sanitize_input(input_with_sensitive)
        
        assert "123-45-6789" not in str(sanitized)
        assert "[REDACTED]" in str(sanitized)
    
    def test_sanitize_oversized_input(self, validator):
        """Test sanitizing oversized input"""
        large_input = "x" * (validator.max_input_size + 1000)
        
        sanitized = validator.sanitize_input(large_input)
        
        assert len(str(sanitized)) <= validator.max_input_size


class TestConsistencyChecker:
    """Test ConsistencyChecker"""
    
    @pytest.fixture
    def checker(self):
        """Create consistency checker for testing"""
        return ConsistencyChecker()
    
    def test_single_output_consistent(self, checker):
        """Test consistency with single output"""
        outputs = {'tool1': True}
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is True
        assert confidence == 1.0
    
    def test_boolean_consistency_agreement(self, checker):
        """Test boolean consistency when all agree"""
        outputs = {
            'tool1': True,
            'tool2': True,
            'tool3': True
        }
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is True
        assert confidence == 1.0
    
    def test_boolean_consistency_disagreement(self, checker):
        """Test boolean consistency with disagreement"""
        outputs = {
            'tool1': True,
            'tool2': True,
            'tool3': False
        }
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is False
        assert 0 < confidence < 1
    
    def test_numerical_consistency(self, checker):
        """Test numerical consistency"""
        # Similar values
        outputs = {
            'tool1': 10.0,
            'tool2': 10.1,
            'tool3': 9.9
        }
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is True
        assert confidence > 0.8
    
    def test_numerical_inconsistency(self, checker):
        """Test numerical inconsistency"""
        # Widely varying values
        outputs = {
            'tool1': 10.0,
            'tool2': 100.0,
            'tool3': 5.0
        }
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is False
    
    def test_string_consistency(self, checker):
        """Test string consistency"""
        outputs = {
            'tool1': 'result_a',
            'tool2': 'result_a',
            'tool3': 'result_a'
        }
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is True
        assert confidence == 1.0
    
    def test_string_inconsistency(self, checker):
        """Test string inconsistency"""
        outputs = {
            'tool1': 'result_a',
            'tool2': 'result_b',
            'tool3': 'result_c'
        }
        
        is_consistent, confidence, details = checker.check_consistency(outputs)
        
        assert is_consistent is False


class TestSafetyGovernor:
    """Test SafetyGovernor"""
    
    @pytest.fixture
    def governor(self):
        """Create safety governor for testing"""
        config = {
            'veto_threshold': 0.8,
            'rate_limit_window': 60,
            'max_requests_per_tool': 10
        }
        return SafetyGovernor(config)
    
    def test_initialization(self, governor):
        """Test governor initialization"""
        assert governor.validator is not None
        assert governor.consistency_checker is not None
        assert len(governor.contracts) > 0
        assert 'symbolic' in governor.contracts
    
    def test_default_contracts(self, governor):
        """Test default contract initialization"""
        # Check symbolic contract
        symbolic = governor.contracts['symbolic']
        assert symbolic.tool_name == 'symbolic'
        assert symbolic.min_confidence == 0.7
        
        # Check probabilistic contract
        probabilistic = governor.contracts['probabilistic']
        assert probabilistic.tool_name == 'probabilistic'
        assert probabilistic.min_confidence == 0.5
    
    def test_check_safety_allow(self, governor):
        """Test safety check allowing safe input"""
        context = SafetyContext(
            problem="Analyze this data",
            tool_name="probabilistic",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        action, reason = governor.check_safety(context)
        
        assert action in [SafetyAction.ALLOW, SafetyAction.LOG_AND_ALLOW]
    
    def test_check_safety_veto_unsafe_input(self, governor):
        """Test safety check vetoing unsafe input"""
        context = SafetyContext(
            problem="How to attack a system",
            tool_name="symbolic",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.CRITICAL
        )
        
        action, reason = governor.check_safety(context)
        
        assert action == SafetyAction.VETO
        assert reason is not None
    
    def test_check_safety_sanitize(self, governor):
        """Test safety check sanitizing input"""
        context = SafetyContext(
            problem="Process SSN: 123-45-6789",
            tool_name="probabilistic",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.LOW  # Not critical, so sanitize
        )
        
        action, reason = governor.check_safety(context)
        
        assert action in [SafetyAction.SANITIZE, SafetyAction.VETO]
    
    def test_contract_violation_missing_input(self, governor):
        """Test contract violation for missing required inputs"""
        context = SafetyContext(
            problem="Just analyze",  # Missing 'logic' and 'rules'
            tool_name="symbolic",
            features=None,
            constraints={'time_budget_ms': 10000, 'energy_budget_mj': 2000},
            user_context={},
            safety_level=SafetyLevel.HIGH
        )
        
        action, reason = governor.check_safety(context)
        
        # May veto due to missing required inputs
        assert action in [SafetyAction.VETO, SafetyAction.ALLOW]
    
    def test_contract_violation_forbidden_input(self, governor):
        """Test contract violation for forbidden inputs"""
        context = SafetyContext(
            problem="Process undefined and infinite values",
            tool_name="symbolic",
            features=None,
            constraints={'time_budget_ms': 10000, 'energy_budget_mj': 2000},
            user_context={},
            safety_level=SafetyLevel.HIGH
        )
        
        action, reason = governor.check_safety(context)
        
        # Should veto due to forbidden inputs
        assert action == SafetyAction.VETO
        assert reason is not None
    
    def test_contract_violation_insufficient_resources(self, governor):
        """Test contract violation for insufficient resources"""
        context = SafetyContext(
            problem="Analyze logic and rules",
            tool_name="symbolic",
            features=None,
            constraints={'time_budget_ms': 100, 'energy_budget_mj': 10},  # Too low
            user_context={},
            safety_level=SafetyLevel.HIGH
        )
        
        action, reason = governor.check_safety(context)
        
        assert action == SafetyAction.VETO
        assert "insufficient" in reason.lower()
    
    def test_rate_limiting(self, governor):
        """Test rate limiting enforcement"""
        context = SafetyContext(
            problem="Test problem",
            tool_name="test_tool",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.LOW
        )
        
        # Make requests up to limit
        for _ in range(governor.max_requests_per_tool):
            governor._is_rate_limited("test_tool")
        
        # Next request should be rate limited
        is_limited = governor._is_rate_limited("test_tool")
        
        assert is_limited is True
    
    def test_filter_candidates(self, governor):
        """Test filtering tool candidates"""
        candidates = [
            {'tool': 'probabilistic', 'confidence': 0.8},
            {'tool': 'symbolic', 'confidence': 0.9},
            {'tool': 'unknown_tool', 'confidence': 0.7}
        ]
        
        context = SafetyContext(
            problem="Analyze data",
            tool_name="",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        filtered = governor.filter_candidates(candidates, context)
        
        # Should filter based on safety checks
        assert isinstance(filtered, list)
    
    def test_validate_output_success(self, governor):
        """Test successful output validation"""
        class MockOutput:
            confidence = 0.9
        
        context = SafetyContext(
            problem="test",
            tool_name="probabilistic",
            features=None,
            constraints={},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        is_valid, reason = governor.validate_output(
            "probabilistic",
            MockOutput(),
            context
        )
        
        assert is_valid is True
    
    def test_validate_output_low_confidence(self, governor):
        """Test output validation with low confidence"""
        class MockOutput:
            confidence = 0.3  # Below threshold
        
        context = SafetyContext(
            problem="test",
            tool_name="probabilistic",
            features=None,
            constraints={},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        is_valid, reason = governor.validate_output(
            "probabilistic",
            MockOutput(),
            context
        )
        
        assert is_valid is False
        assert "confidence" in reason.lower()
    
    def test_check_consensus(self, governor):
        """Test consensus checking"""
        outputs = {
            'tool1': 10.0,
            'tool2': 10.5,
            'tool3': 9.8
        }
        
        is_consistent, confidence, details = governor.check_consensus(outputs)
        
        assert isinstance(is_consistent, bool)
        assert 0 <= confidence <= 1
    
    def test_violation_recording(self, governor):
        """Test violation recording"""
        initial_count = len(governor.violations)
        
        governor._record_violation(
            "test_tool",
            VetoReason.UNSAFE_INPUT,
            "Test violation"
        )
        
        assert len(governor.violations) == initial_count + 1
        assert governor.violation_counts['test_tool'] > 0
    
    def test_bounded_violation_storage(self, governor):
        """Test that violation storage is bounded"""
        # Add many violations
        for i in range(governor.max_violations + 100):
            governor._record_violation(
                f"tool_{i % 10}",
                VetoReason.UNSAFE_INPUT,
                f"Violation {i}"
            )
        
        # Should not exceed max
        assert len(governor.violations) <= governor.max_violations
    
    def test_safety_caching(self, governor):
        """Test safety check caching"""
        context = SafetyContext(
            problem="Cached problem",
            tool_name="probabilistic",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        # First call
        result1 = governor.check_safety(context)
        cache_size_1 = len(governor.safety_cache)
        
        # Second call (should use cache)
        result2 = governor.check_safety(context)
        cache_size_2 = len(governor.safety_cache)
        
        assert result1 == result2
        assert cache_size_2 == cache_size_1
    
    def test_cache_eviction(self, governor):
        """Test cache size limiting"""
        governor.max_cache_size = 10
        
        # Fill cache beyond limit
        for i in range(20):
            context = SafetyContext(
                problem=f"Problem {i}",
                tool_name="probabilistic",
                features=None,
                constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
                user_context={},
                safety_level=SafetyLevel.MEDIUM
            )
            governor.check_safety(context)
        
        # Cache should be limited
        assert len(governor.safety_cache) <= governor.max_cache_size
    
    def test_statistics(self, governor):
        """Test getting statistics"""
        # Generate some violations
        governor._record_violation(
            "test_tool",
            VetoReason.UNSAFE_INPUT,
            "Test"
        )
        
        stats = governor.get_statistics()
        
        assert 'total_violations' in stats
        assert 'violations_by_tool' in stats
        assert 'recent_violations' in stats
        assert stats['total_violations'] > 0
    
    def test_audit_trail_export(self, governor, tmp_path):
        """Test exporting audit trail"""
        # Generate some audit entries
        context = SafetyContext(
            problem="test",
            tool_name="probabilistic",
            features=None,
            constraints={},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        governor.check_safety(context)
        
        # Export
        export_path = tmp_path / "audit.json"
        governor.export_audit_trail(str(export_path))
        
        assert export_path.exists()
    
    def test_clear_cache(self, governor):
        """Test cache clearing"""
        # Add some cached items
        context = SafetyContext(
            problem="test",
            tool_name="probabilistic",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        governor.check_safety(context)
        
        assert len(governor.safety_cache) > 0
        
        governor.clear_cache()
        
        assert len(governor.safety_cache) == 0
    
    def test_reset_statistics(self, governor):
        """Test resetting statistics"""
        # Generate violations
        governor._record_violation("tool", VetoReason.UNSAFE_INPUT, "test")
        
        assert len(governor.violations) > 0
        
        governor.reset_statistics()
        
        assert len(governor.violations) == 0
        assert len(governor.violation_counts) == 0
    
    def test_thread_safety(self, governor):
        """Test thread-safe operations"""
        results = []
        errors = []
        
        def check_safety():
            try:
                for i in range(10):
                    context = SafetyContext(
                        problem=f"Problem {i}",
                        tool_name="probabilistic",
                        features=None,
                        constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
                        user_context={},
                        safety_level=SafetyLevel.MEDIUM
                    )
                    result = governor.check_safety(context)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=check_safety) for _ in range(3)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) > 0


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_safety_workflow(self):
        """Test complete safety checking workflow"""
        governor = SafetyGovernor()
        
        # 1. Check input safety
        context = SafetyContext(
            problem="Analyze this dataset with logic",
            tool_name="symbolic",
            features=np.array([1, 2, 3]),
            constraints={'time_budget_ms': 10000, 'energy_budget_mj': 2000},
            user_context={'user': 'test'},
            safety_level=SafetyLevel.HIGH
        )
        
        action, reason = governor.check_safety(context)
        
        assert action in [SafetyAction.ALLOW, SafetyAction.VETO, SafetyAction.LOG_AND_ALLOW]
        
        # 2. Validate output
        class MockOutput:
            confidence = 0.9
            value = "result"
        
        if action == SafetyAction.ALLOW:
            is_valid, reason = governor.validate_output(
                "symbolic",
                MockOutput(),
                context
            )
            assert isinstance(is_valid, bool)
        
        # 3. Check consensus
        outputs = {
            'tool1': 42,
            'tool2': 43,
            'tool3': 41
        }
        
        is_consistent, confidence, details = governor.check_consensus(outputs)
        assert isinstance(is_consistent, bool)
        
        # 4. Get statistics
        stats = governor.get_statistics()
        assert 'total_violations' in stats


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_problem(self):
        """Test with empty problem"""
        governor = SafetyGovernor()
        
        context = SafetyContext(
            problem="",
            tool_name="probabilistic",
            features=None,
            constraints={},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        action, reason = governor.check_safety(context)
        
        # Should handle gracefully
        assert isinstance(action, SafetyAction)
    
    def test_none_output_validation(self):
        """Test validating None output"""
        governor = SafetyGovernor()
        
        context = SafetyContext(
            problem="test",
            tool_name="probabilistic",
            features=None,
            constraints={},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        is_valid, reason = governor.validate_output(
            "probabilistic",
            None,
            context
        )
        
        # None output should be handled
        assert isinstance(is_valid, bool)
    
    def test_unknown_tool(self):
        """Test with unknown tool"""
        governor = SafetyGovernor()
        
        context = SafetyContext(
            problem="test",
            tool_name="unknown_tool",
            features=None,
            constraints={'time_budget_ms': 5000, 'energy_budget_mj': 1000},
            user_context={},
            safety_level=SafetyLevel.MEDIUM
        )
        
        action, reason = governor.check_safety(context)
        
        # Should allow or handle gracefully
        assert isinstance(action, SafetyAction)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])