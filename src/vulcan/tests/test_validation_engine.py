"""
test_validation_engine.py - Tests for validation_engine.py
"""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add the knowledge_crystallizer directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'knowledge_crystallizer'))

from validation_engine import (
    Principle, ValidationResult, ValidationResults, DomainTestCase,
    KnowledgeValidator, DomainValidator, FailureAnalysis,
    ValidationLevel, DomainCategory, TestResult
)


class TestPrinciple:
    """Test Principle dataclass"""
    
    def test_principle_creation_basic(self):
        """Test basic principle creation"""
        p = Principle(
            id="test_1",
            core_pattern=Mock(),
            confidence=0.8
        )
        assert p.id == "test_1"
        assert p.confidence == 0.8
        assert p.execution_type == "function"
    
    def test_principle_validation_empty_id(self):
        """Test that empty ID raises error"""
        with pytest.raises(ValueError, match="id must be non-empty"):
            Principle(id="", core_pattern=Mock(), confidence=0.5)
    
    def test_principle_validation_invalid_confidence(self):
        """Test that invalid confidence raises error"""
        with pytest.raises(ValueError, match="between 0 and 1"):
            Principle(id="test", core_pattern=Mock(), confidence=1.5)
    
    def test_principle_execute_no_logic(self):
        """Test execute without logic raises error"""
        p = Principle(id="test", core_pattern=Mock(), confidence=0.5)
        with pytest.raises(NotImplementedError):
            p.execute({'input': 1})
    
    def test_principle_execute_function_type(self):
        """Test execute with callable function"""
        def logic(inputs):
            return {'output': inputs['input'] * 2}
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            execution_logic=logic,
            execution_type="function"
        )
        
        result = p.execute({'input': 5})
        assert result == {'output': 10}
    
    def test_principle_execute_code_string(self):
        """Test execute with code string"""
        # Use safe code that doesn't trigger security checks
        # The 'inputs' dict is provided in the namespace by execute()
        code = """
x_val = inputs.get('x', 0)
y_val = inputs.get('y', 0)
output = {'result': x_val + y_val}
"""
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            execution_logic=code,
            execution_type="code_string"
        )
        
        result = p.execute({'x': 5, 'y': 3})
        assert result == {'result': 8}
    
    def test_principle_execute_code_string_dangerous(self):
        """Test that dangerous code is blocked"""
        code = "import os; output = {}"
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            execution_logic=code,
            execution_type="code_string"
        )
        
        with pytest.raises(ValueError, match="Dangerous operation"):
            p.execute({'input': 1})
    
    def test_principle_execute_rule_spec(self):
        """Test execute with rule specification"""
        rules = {
            'output_val': {
                'type': 'direct',
                'source': 'input_val'
            }
        }
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            execution_logic=rules,
            execution_type="rule_spec"
        )
        
        result = p.execute({'input_val': 42})
        assert result == {'output_val': 42}


class TestValidationResult:
    """Test ValidationResult dataclass"""
    
    def test_validation_result_creation(self):
        """Test creating validation result"""
        vr = ValidationResult(
            is_valid=True,
            confidence=0.9,
            errors=[],
            warnings=["Minor issue"]
        )
        assert vr.is_valid
        assert vr.confidence == 0.9
        assert len(vr.warnings) == 1
    
    def test_validation_result_to_dict(self):
        """Test converting to dict"""
        vr = ValidationResult(
            is_valid=False,
            confidence=0.3,
            errors=["Error 1", "Error 2"]
        )
        d = vr.to_dict()
        assert d['is_valid'] == False
        assert d['confidence'] == 0.3
        assert len(d['errors']) == 2


class TestValidationResults:
    """Test ValidationResults (multi-domain)"""
    
    def test_add_success(self):
        """Test adding successful domain"""
        vr = ValidationResults()
        vr.add_success('domain1', 0.9)
        
        assert 'domain1' in vr.successful_domains
        assert vr.domain_scores['domain1'] == 0.9
        assert vr.success_rate == 1.0
    
    def test_add_failure(self):
        """Test adding failed domain"""
        vr = ValidationResults()
        failure = FailureAnalysis(
            failure_type='test_failure',
            error_message='Test error'
        )
        vr.add_failure('domain1', failure)
        
        assert 'domain1' in vr.failed_domains
        assert vr.domain_scores['domain1'] == 0.0
        assert vr.success_rate == 0.0
    
    def test_mixed_results(self):
        """Test mixed success and failure"""
        vr = ValidationResults()
        vr.add_success('domain1', 0.8)
        vr.add_success('domain2', 0.9)
        
        failure = FailureAnalysis(failure_type='test')
        vr.add_failure('domain3', failure)
        
        assert vr.success_rate == 2/3
        assert len(vr.successful_domains) == 2
        assert len(vr.failed_domains) == 1


class TestKnowledgeValidator:
    """Test KnowledgeValidator"""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        validator = KnowledgeValidator(
            min_confidence=0.7,
            consistency_threshold=0.8
        )
        assert validator.min_confidence == 0.7
        assert validator.consistency_threshold == 0.8
        assert validator.domain_validator is not None
    
    def test_validate_basic_valid_principle(self):
        """Test basic validation of valid principle"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test_1",
            core_pattern=Mock(),
            confidence=0.8,
            applicable_domains=['general'],
            execution_logic=lambda x: x
        )
        
        result = validator.validate(p)
        assert result.is_valid
        assert result.confidence > 0.5
    
    def test_validate_basic_invalid_type(self):
        """Test validation rejects non-Principle"""
        validator = KnowledgeValidator()
        result = validator.validate("not a principle")
        
        assert not result.is_valid
        assert result.confidence == 0.0
        assert len(result.errors) > 0
    
    def test_validate_basic_low_confidence(self):
        """Test validation warns on low confidence"""
        validator = KnowledgeValidator(min_confidence=0.8)
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.5
        )
        
        result = validator.validate(p)
        assert any('Low confidence' in w for w in result.warnings)
    
    def test_validate_consistency_domain_conflict(self):
        """Test consistency validation detects domain conflicts"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            applicable_domains=['medical'],
            contraindicated_domains=['medical']  # Conflict!
        )
        
        result = validator.validate_consistency(p)
        assert not result.is_valid
        assert any('conflict' in e.lower() for e in result.errors)
    
    def test_domain_criticality_calculation(self):
        """Test domain criticality scoring"""
        validator = KnowledgeValidator()
        
        # Safety critical domain
        crit1 = validator._get_domain_criticality(['safety_critical'])
        assert crit1 > 0.9
        
        # General domain
        crit2 = validator._get_domain_criticality(['general'])
        assert crit2 < 0.5
        
        # Unknown domain
        crit3 = validator._get_domain_criticality(['unknown_domain'])
        assert crit3 == 0.5
    
    def test_select_validation_levels_safety_critical(self):
        """Test that safety-critical domains get full validation"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.7,
            applicable_domains=['safety_critical']
        )
        
        levels = validator._select_validation_levels(p, {})
        
        assert ValidationLevel.BASIC in levels
        assert ValidationLevel.CONSISTENCY in levels
        assert ValidationLevel.DOMAIN_SPECIFIC in levels
        assert ValidationLevel.GENERALIZATION in levels
    
    def test_select_validation_levels_low_confidence(self):
        """Test low confidence principles get thorough validation"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.4,
            applicable_domains=['general']
        )
        
        levels = validator._select_validation_levels(p, {})
        
        assert ValidationLevel.BASIC in levels
        assert ValidationLevel.DOMAIN_SPECIFIC in levels
    
    def test_select_validation_levels_high_confidence(self):
        """Test high confidence with time budget"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.95,
            applicable_domains=['general']
        )
        
        context = {'time_budget_ms': 5000}
        levels = validator._select_validation_levels(p, context)
        
        # Should only do basic validation
        assert ValidationLevel.BASIC in levels
        assert ValidationLevel.GENERALIZATION in levels  # Has time budget
    
    def test_cache_cleanup(self):
        """Test expired cache entries are cleaned"""
        validator = KnowledgeValidator()
        validator.cache_ttl = 0.1  # 100ms
        
        # Add cache entry
        validator.validation_cache['test_key'] = Mock()
        validator.cache_timestamps['test_key'] = time.time()
        
        # Wait for expiry
        time.sleep(0.15)
        
        # Cleanup
        validator._cleanup_expired_cache()
        
        assert 'test_key' not in validator.validation_cache


class TestDomainValidator:
    """Test DomainValidator"""
    
    def test_domain_validator_initialization(self):
        """Test domain validator initializes"""
        dv = DomainValidator()
        assert len(dv.domain_registry) > 0
        assert 'general' in dv.domain_registry
        assert 'optimization' in dv.domain_registry
    
    def test_select_diverse_domains(self):
        """Test diverse domain selection"""
        dv = DomainValidator()
        
        candidate = Mock()
        candidate.origin_domain = 'optimization'
        
        domains = dv.select_diverse_domains(candidate, count=5)
        
        assert len(domains) <= 5
        assert 'optimization' in domains  # Should include origin
    
    def test_categorize_domains_by_data(self):
        """Test domain categorization by data availability"""
        dv = DomainValidator()
        categorized = dv.categorize_domains_by_data_availability()
        
        assert DomainCategory.HIGH_DATA in categorized
        assert 'general' in categorized[DomainCategory.HIGH_DATA]
    
    def test_generate_domain_test(self):
        """Test domain test case generation"""
        dv = DomainValidator()
        
        principle = Mock()
        principle.id = 'test_1'
        principle.compute_expected = None
        
        test_case = dv.generate_domain_test(principle, 'optimization')
        
        assert test_case.domain == 'optimization'
        assert 'domain' in test_case.inputs
        assert test_case.timeout > 0
    
    def test_generate_domain_inputs_optimization(self):
        """Test optimization domain inputs"""
        dv = DomainValidator()
        inputs = dv._generate_domain_inputs('optimization', {})
        
        assert inputs['domain'] == 'optimization'
        assert 'objective' in inputs
        assert 'variables' in inputs
    
    def test_generate_domain_inputs_classification(self):
        """Test classification domain inputs"""
        dv = DomainValidator()
        inputs = dv._generate_domain_inputs('classification', {})
        
        assert 'classes' in inputs
        assert 'features' in inputs
    
    def test_calculate_domain_similarity(self):
        """Test domain similarity calculation"""
        dv = DomainValidator()
        
        # Same domain
        sim1 = dv._calculate_domain_similarity('optimization', 'optimization')
        assert sim1 == 1.0
        
        # Different domains with some overlap
        sim2 = dv._calculate_domain_similarity('optimization', 'prediction')
        assert 0.0 <= sim2 <= 1.0
    
    @patch('subprocess.Popen')
    def test_run_sandboxed_test_success(self, mock_popen):
        """Test successful sandboxed test execution"""
        # Mock subprocess result
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            '{"score": 0.9, "output": "test", "performance": {}, "pattern_match": 0.9}',
            ''
        )
        mock_popen.return_value = mock_process
        
        dv = DomainValidator()
        
        principle = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            execution_logic=lambda x: x,
            execution_type="function"
        )
        
        test_case = DomainTestCase(
            domain='general',
            test_id='test_1',
            inputs={'x': 1},
            timeout=10.0
        )
        
        result = dv.run_sandboxed_test(principle, test_case)
        
        assert result['success'] == True
        assert result['score'] == 0.9
    
    @patch('subprocess.Popen')
    def test_run_sandboxed_test_timeout(self, mock_popen):
        """Test sandboxed test timeout handling"""
        import subprocess
        
        mock_process = Mock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired('cmd', 10)
        mock_process.kill = Mock()
        mock_popen.return_value = mock_process
        
        dv = DomainValidator()
        
        principle = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            execution_logic=lambda x: x
        )
        
        test_case = DomainTestCase(
            domain='general',
            test_id='test_1',
            inputs={'x': 1},
            timeout=1.0
        )
        
        result = dv.run_sandboxed_test(principle, test_case)
        
        assert result['success'] == False
        assert 'timeout' in result['error'].lower()


class TestValidationMultilevel:
    """Test multilevel validation orchestration"""
    
    def test_multilevel_validation_basic_only(self):
        """Test multilevel with only basic validation needed"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.95,  # High confidence
            applicable_domains=['general']
        )
        
        context = {'time_budget_ms': 100}  # Low time budget
        results = validator.validate_principle_multilevel(p, context)
        
        assert 'basic' in results
        assert results['basic'].is_valid
    
    def test_multilevel_validation_early_exit(self):
        """Test early exit on critical failure"""
        validator = KnowledgeValidator()
        
        # Create an invalid principle (not a Principle object)
        # This should fail basic validation
        fake_principle = "not a principle"
        
        results = validator.validate_principle_multilevel(fake_principle, {})
        
        # Should get an error result since it's not a Principle
        assert 'error' in results or 'basic' in results
        if 'basic' in results:
            assert not results['basic'].is_valid
    
    def test_multilevel_validation_comprehensive(self):
        """Test comprehensive validation for safety-critical"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.7,
            applicable_domains=['safety_critical'],
            execution_logic=lambda x: {'output': x['input']}
        )
        
        context = {'force_comprehensive': True}
        results = validator.validate_principle_multilevel(p, context)
        
        assert 'basic' in results
        assert 'consistency' in results


class TestBugFixes:
    """Test specific bug fixes"""
    
    def test_cache_race_condition_fix(self):
        """Test that cache cleanup doesn't cause race condition"""
        validator = KnowledgeValidator()
        
        # Add many cache entries
        for i in range(10):
            validator.validation_cache[f'key_{i}'] = Mock()
            validator.cache_timestamps[f'key_{i}'] = time.time()
        
        # Cleanup should not crash
        validator._cleanup_expired_cache()
        
        # Should still be able to access cache
        assert isinstance(validator.validation_cache, dict)
    
    def test_domain_criticality_partial_match(self):
        """Test domain criticality handles partial matches"""
        validator = KnowledgeValidator()
        
        # Domain with safety_critical in name
        crit = validator._get_domain_criticality(['safety_critical_system'])
        assert crit > 0.9  # Should match 'safety_critical'
    
    def test_validation_handles_none_domains(self):
        """Test validation handles None in domains"""
        validator = KnowledgeValidator()
        
        p = Principle(
            id="test",
            core_pattern=Mock(),
            confidence=0.8,
            applicable_domains=[None, 'general']  # None in list!
        )
        
        # Should not crash
        result = validator.validate(p)
        assert isinstance(result, ValidationResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])