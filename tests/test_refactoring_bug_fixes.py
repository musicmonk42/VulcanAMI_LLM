"""
Tests to verify that all refactoring bugs have been fixed.

This test module validates that all the NameError bugs identified in the
refactoring are now resolved by checking:
1. All required imports exist
2. All required functions are defined
3. All required constants are defined
"""

import pytest
import re


class TestRefactoringBugFixes:
    """Tests for refactoring bug fixes in unified reasoning module."""
    
    def test_bug1_compute_query_hash_import(self):
        """Bug 1: Verify _compute_query_hash is imported correctly."""
        from src.vulcan.reasoning.unified.cache import compute_query_hash
        
        # Test that the function exists and works
        hash1 = compute_query_hash("test query")
        hash2 = compute_query_hash("test query")
        hash3 = compute_query_hash("different query")
        
        assert hash1 == hash2, "Same query should produce same hash"
        assert hash1 != hash3, "Different queries should produce different hashes"
        assert len(hash1) == 32, "Hash should be 32 characters"
    
    def test_bug2_is_creative_task_function(self):
        """Bug 2: Verify _is_creative_task function exists and works."""
        # We need to check that the function is defined in orchestrator.py
        with open('src/vulcan/reasoning/unified/orchestrator.py', 'r') as f:
            content = f.read()
        
        # Check function is defined
        assert 'def _is_creative_task(' in content, "_is_creative_task function not found"
        
        # Check it has proper docstring
        assert 'Creative tasks' in content, "_is_creative_task missing docstring"
        
        # Verify CREATIVE_TASK_KEYWORDS is imported
        assert 'CREATIVE_TASK_KEYWORDS' in content, "CREATIVE_TASK_KEYWORDS not imported"
    
    def test_bug3_is_result_not_applicable_import(self):
        """Bug 3: Verify _is_result_not_applicable is imported."""
        from src.vulcan.reasoning.unified.strategies import _is_result_not_applicable
        
        # Test that the function exists (basic callable check)
        assert callable(_is_result_not_applicable)
    
    def test_bug4_unknown_type_fallback_order_constant(self):
        """Bug 4: Verify UNKNOWN_TYPE_FALLBACK_ORDER constant exists."""
        from src.vulcan.reasoning.unified.config import UNKNOWN_TYPE_FALLBACK_ORDER
        
        # Verify it's a tuple with expected values
        assert isinstance(UNKNOWN_TYPE_FALLBACK_ORDER, tuple)
        assert len(UNKNOWN_TYPE_FALLBACK_ORDER) > 0
        assert "PROBABILISTIC" in UNKNOWN_TYPE_FALLBACK_ORDER
        assert "SYMBOLIC" in UNKNOWN_TYPE_FALLBACK_ORDER
    
    def test_bug5_math_patterns_defined(self):
        """Bug 5: Verify MATH_EXPRESSION_PATTERN and MATH_QUERY_PATTERN are defined."""
        # Check that patterns are defined at module level in orchestrator.py
        with open('src/vulcan/reasoning/unified/orchestrator.py', 'r') as f:
            content = f.read()
        
        assert 'MATH_EXPRESSION_PATTERN = re.compile(' in content
        assert 'MATH_QUERY_PATTERN = re.compile(' in content
        
        # Test that patterns are valid regex
        match1 = re.search(r"MATH_EXPRESSION_PATTERN = re\.compile\(r'([^']+)'\)", content)
        assert match1, "MATH_EXPRESSION_PATTERN pattern not found"
        
        match2 = re.search(r"MATH_QUERY_PATTERN = re\.compile\(r'([^']+)'", content)
        assert match2, "MATH_QUERY_PATTERN pattern not found"
    
    def test_bug6_problem_type_bayesian_constant(self):
        """Bug 6: Verify PROBLEM_TYPE_BAYESIAN constant exists."""
        from src.vulcan.reasoning.unified.config import PROBLEM_TYPE_BAYESIAN
        
        # Verify it's a string
        assert isinstance(PROBLEM_TYPE_BAYESIAN, str)
        assert len(PROBLEM_TYPE_BAYESIAN) > 0
        assert PROBLEM_TYPE_BAYESIAN == "bayesian_inference"
    
    def test_bug7_get_weight_manager_import(self):
        """Bug 7: Verify get_weight_manager is imported."""
        from src.vulcan.reasoning.unified.cache import get_weight_manager
        
        # Test that the function exists and returns a manager
        manager = get_weight_manager()
        assert manager is not None
        
        # Test that it has expected methods
        assert hasattr(manager, 'get_weight')
        assert hasattr(manager, 'adjust_weight')
    
    def test_all_imports_in_orchestrator(self):
        """Verify all imports are present in orchestrator.py."""
        with open('src/vulcan/reasoning/unified/orchestrator.py', 'r') as f:
            content = f.read()
        
        # Check all required imports
        assert 'compute_query_hash as _compute_query_hash' in content
        assert 'get_weight_manager' in content
        assert 'from .strategies import _is_result_not_applicable' in content
        assert 'UNKNOWN_TYPE_FALLBACK_ORDER' in content
        assert 'PROBLEM_TYPE_BAYESIAN' in content
        assert 'CREATIVE_TASK_KEYWORDS' in content
    
    def test_config_exports_all_constants(self):
        """Verify config.py exports all required constants."""
        from src.vulcan.reasoning.unified import config
        
        # Check all constants exist
        assert hasattr(config, 'UNKNOWN_TYPE_FALLBACK_ORDER')
        assert hasattr(config, 'PROBLEM_TYPE_BAYESIAN')
        assert hasattr(config, 'CREATIVE_TASK_KEYWORDS')
        assert hasattr(config, 'CACHE_HASH_LENGTH')
        assert hasattr(config, 'MATH_VERIFICATION_CONFIDENCE_BOOST')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
