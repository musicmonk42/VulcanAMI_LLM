"""
Tests for Mathematical Computation Tool

Comprehensive test suite covering:
- Problem classification
- Code generation (template-based)
- Safe execution integration
- Error handling
- Learning capabilities
- Response formatting
"""

import pytest

# Try to import the modules
try:
    from vulcan.reasoning.mathematical_computation import (
        MathematicalComputationTool,
        ProblemType,
        SolutionStrategy,
        ComputationResult,
        ProblemClassification,
        ProblemClassifier,
        CodeTemplates,
        create_mathematical_computation_tool,
        SAFE_EXECUTION_AVAILABLE,
    )
    COMPUTATION_TOOL_AVAILABLE = True
except ImportError:
    COMPUTATION_TOOL_AVAILABLE = False


# ============================================================================
# PROBLEM CLASSIFIER TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestProblemClassifier:
    """Tests for ProblemClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return ProblemClassifier()

    def test_classify_integration(self, classifier):
        """Test classification of integration problems."""
        result = classifier.classify("Integrate x^2 with respect to x")
        
        assert result.problem_type == ProblemType.CALCULUS
        assert result.confidence > 0.3
        assert "integrate" in result.keywords or "integral" in result.keywords

    def test_classify_differentiation(self, classifier):
        """Test classification of differentiation problems."""
        result = classifier.classify("Find the derivative of sin(x)")
        
        assert result.problem_type == ProblemType.CALCULUS
        assert "derivative" in result.keywords or "diff" in result.keywords

    def test_classify_equation_solving(self, classifier):
        """Test classification of equation solving problems."""
        result = classifier.classify("Solve x^2 - 4 = 0")
        
        assert result.problem_type == ProblemType.ALGEBRA
        assert "solve" in result.keywords or "equation" in result.keywords

    def test_classify_matrix_operation(self, classifier):
        """Test classification of matrix problems."""
        result = classifier.classify("Find the determinant of matrix A")
        
        assert result.problem_type == ProblemType.LINEAR_ALGEBRA
        assert "matrix" in result.keywords or "determinant" in result.keywords

    def test_classify_differential_equation(self, classifier):
        """Test classification of differential equation problems."""
        # Use unambiguous differential equation query
        result = classifier.classify("Solve the ODE dy/dx = y")
        
        assert result.problem_type == ProblemType.DIFFERENTIAL_EQUATIONS
        assert any(kw in result.keywords for kw in ["ode", "differential equation"])

    def test_classify_limit(self, classifier):
        """Test classification of limit problems."""
        result = classifier.classify("Find the limit as x approaches 0 of sin(x)/x")
        
        assert result.problem_type == ProblemType.CALCULUS
        assert "limit" in result.keywords

    def test_classify_unknown(self, classifier):
        """Test classification of unrecognized problems."""
        result = classifier.classify("Do something random")
        
        assert result.problem_type == ProblemType.UNKNOWN
        assert result.confidence < 0.5

    def test_extract_variables(self, classifier):
        """Test variable extraction."""
        result = classifier.classify("Integrate with respect to t")
        
        # Should extract 't' as variable
        assert 't' in result.variables or 'x' in result.variables


# ============================================================================
# CODE TEMPLATES TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestCodeTemplates:
    """Tests for CodeTemplates."""

    def test_integration_template(self):
        """Test integration code template."""
        code = CodeTemplates.integration("x**3", "x")
        
        assert "Symbol" in code
        assert "integrate" in code
        assert "result" in code

    def test_definite_integration_template(self):
        """Test definite integration code template."""
        code = CodeTemplates.integration("x**2", "x", ("0", "1"))
        
        assert "Symbol" in code
        assert "integrate" in code
        assert "0" in code
        assert "1" in code

    def test_differentiation_template(self):
        """Test differentiation code template."""
        code = CodeTemplates.differentiation("x**3", "x")
        
        assert "Symbol" in code
        assert "diff" in code
        assert "result" in code

    def test_solve_equation_template(self):
        """Test equation solving code template."""
        code = CodeTemplates.solve_equation("x**2 - 4", "x")
        
        assert "Symbol" in code
        assert "solve" in code
        assert "result" in code

    def test_limit_template(self):
        """Test limit code template."""
        code = CodeTemplates.limit("sin(x)/x", "x", "0")
        
        assert "Symbol" in code
        assert "limit" in code
        assert "result" in code

    def test_series_template(self):
        """Test series expansion code template."""
        code = CodeTemplates.series_expansion("exp(x)", "x", "0", 5)
        
        assert "Symbol" in code
        assert "series" in code
        assert "result" in code

    def test_matrix_template(self):
        """Test matrix operation code template."""
        code = CodeTemplates.matrix_operation("[[1, 2], [3, 4]]", "det")
        
        assert "Matrix" in code
        assert "det" in code
        assert "result" in code

    def test_differential_equation_template(self):
        """Test differential equation code template."""
        code = CodeTemplates.differential_equation("Eq(f(x).diff(x), f(x))", "f", "x")
        
        assert "Symbol" in code
        assert "Function" in code
        assert "dsolve" in code


# ============================================================================
# MATHEMATICAL COMPUTATION TOOL TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestMathematicalComputationTool:
    """Tests for MathematicalComputationTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance for testing."""
        return MathematicalComputationTool(
            llm=None,
            enable_learning=False,
            prefer_templates=True
        )

    def test_tool_initialization(self, tool):
        """Test tool initializes correctly."""
        assert tool.name == "mathematical_computation"
        assert tool.description is not None
        assert tool._classifier is not None
        assert tool._templates is not None

    def test_simple_integration(self, tool):
        """Test simple polynomial integration."""
        result = tool.execute("Integrate x^2 with respect to x")
        
        assert isinstance(result, ComputationResult)
        assert result.code is not None
        assert result.problem_type == ProblemType.CALCULUS
        
        if SAFE_EXECUTION_AVAILABLE:
            assert result.success is True
            assert result.result is not None
            assert 'x**3/3' in result.result or 'x**3' in result.result

    def test_differentiation(self, tool):
        """Test differentiation."""
        result = tool.execute("Differentiate x^3 with respect to x")
        
        assert result.code is not None
        assert 'diff' in result.code.lower()
        
        if SAFE_EXECUTION_AVAILABLE:
            assert result.success is True

    def test_solve_equation(self, tool):
        """Test equation solving."""
        result = tool.execute("Solve x^2 - 4 = 0")
        
        assert result.code is not None
        assert 'solve' in result.code.lower()
        
        if SAFE_EXECUTION_AVAILABLE:
            assert result.success is True

    def test_limit_computation(self, tool):
        """Test limit computation."""
        result = tool.execute("Calculate the limit of sin(x)/x as x approaches 0")
        
        assert result.code is not None
        assert 'limit' in result.code.lower()
        
        if SAFE_EXECUTION_AVAILABLE:
            assert result.success is True
            assert result.result == '1'

    def test_series_expansion(self, tool):
        """Test series expansion."""
        result = tool.execute("Taylor series of e^x")
        
        assert result.code is not None
        assert 'series' in result.code.lower()

    def test_matrix_operation(self, tool):
        """Test matrix operation."""
        result = tool.execute("Compute the determinant of the matrix [[1, 2], [3, 4]]")
        
        assert result.code is not None
        assert 'Matrix' in result.code or 'matrix' in result.code.lower()
        
        if SAFE_EXECUTION_AVAILABLE:
            assert result.success is True
            assert result.result == '-2'

    def test_format_response_success(self, tool):
        """Test response formatting for successful computation."""
        result = tool.execute("Integrate x^2")
        formatted = tool.format_response(result)
        
        assert '**Mathematical Computation**' in formatted
        
        # Only expect success formatting when safe execution is available
        if SAFE_EXECUTION_AVAILABLE:
            assert '**Code:**' in formatted
            if result.success:
                assert '**Result:**' in formatted
        else:
            # When safe execution is unavailable, expect error formatting
            assert '⚠️' in formatted or 'failed' in formatted.lower()

    def test_format_response_failure(self):
        """Test response formatting for failed computation."""
        result = ComputationResult(
            success=False,
            code='',
            error='Test error',
            explanation='Test failed'
        )
        tool = MathematicalComputationTool()
        formatted = tool.format_response(result)
        
        assert '⚠️' in formatted or 'failed' in formatted.lower()
        assert 'Test error' in formatted

    def test_statistics(self, tool):
        """Test statistics tracking."""
        # Execute some problems
        tool.execute("Integrate x^2")
        tool.execute("Differentiate x^3")
        
        stats = tool.get_statistics()
        
        assert 'cache_size' in stats
        assert 'safe_execution_available' in stats
        assert 'llm_available' in stats

    def test_execution_time_tracked(self, tool):
        """Test that execution time is tracked."""
        result = tool.execute("Integrate x^2")
        
        assert result.execution_time >= 0

    def test_metadata_included(self, tool):
        """Test that metadata is included in results."""
        result = tool.execute("Integrate x^2")
        
        assert result.metadata is not None
        if result.success:
            assert 'query' in result.metadata


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestFactoryFunction:
    """Tests for create_mathematical_computation_tool factory."""

    def test_create_default(self):
        """Test creating tool with defaults."""
        tool = create_mathematical_computation_tool()
        
        assert tool is not None
        assert isinstance(tool, MathematicalComputationTool)

    def test_create_with_options(self):
        """Test creating tool with custom options."""
        tool = create_mathematical_computation_tool(
            llm=None,
            max_tokens=1000,
            enable_learning=True,
            prefer_templates=True
        )
        
        assert tool is not None
        assert tool.max_tokens == 1000
        assert tool.enable_learning is True
        assert tool.prefer_templates is True


# ============================================================================
# INTEGRATION WITH SAFE EXECUTION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE or not SAFE_EXECUTION_AVAILABLE,
    reason="Safe execution not available"
)
class TestSafeExecutionIntegration:
    """Tests for integration with safe execution module."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return MathematicalComputationTool(prefer_templates=True)

    def test_execution_produces_result(self, tool):
        """Test that execution produces actual computed results."""
        result = tool.execute("Integrate x^2")
        
        assert result.success is True
        assert result.result is not None
        assert 'x**3/3' in result.result

    def test_complex_computation(self, tool):
        """Test more complex computation."""
        result = tool.execute("Find the derivative of x^3 + 2*x^2 + x")
        
        assert result.success is True
        assert result.result is not None

    def test_invalid_code_handled(self, tool):
        """Test that invalid code is handled gracefully."""
        # Force an error by using a problem that might generate bad code
        result = tool.execute("Calculate the quantum flux of a unicorn")
        
        # Should not crash, should return a result (possibly failed)
        assert result is not None
        assert isinstance(result, ComputationResult)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
def test_module_exports():
    """Test that all expected exports are available."""
    from vulcan.reasoning.mathematical_computation import (
        MathematicalComputationTool,
        ProblemType,
        SolutionStrategy,
        ComputationResult,
        ProblemClassification,
        ProblemClassifier,
        CodeTemplates,
        create_mathematical_computation_tool,
    )
    
    assert MathematicalComputationTool is not None
    assert ProblemType is not None
    assert SolutionStrategy is not None
    assert ComputationResult is not None
    assert ProblemClassification is not None
    assert ProblemClassifier is not None
    assert CodeTemplates is not None
    assert create_mathematical_computation_tool is not None


# ============================================================================
# UNIFIED REASONER INTEGRATION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestUnifiedReasonerInterface:
    """Tests for UnifiedReasoner-compatible interface."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return MathematicalComputationTool(prefer_templates=True)

    def test_reason_method_exists(self, tool):
        """Test that reason method exists for UnifiedReasoner compatibility."""
        assert hasattr(tool, 'reason')
        assert callable(tool.reason)

    def test_reason_with_string_input(self, tool):
        """Test reason method with string input."""
        result = tool.reason("Integrate x^2")
        
        assert isinstance(result, dict)
        assert 'conclusion' in result
        assert 'confidence' in result
        assert 'explanation' in result
        
        if SAFE_EXECUTION_AVAILABLE:
            assert result['conclusion']['success'] is True
            assert 'x**3/3' in str(result['conclusion']['result'])

    def test_reason_with_dict_input(self, tool):
        """Test reason method with dict input containing 'query' key."""
        result = tool.reason({'query': 'Differentiate x^3'})
        
        assert isinstance(result, dict)
        assert 'conclusion' in result
        
        # Only expect success when safe execution is available
        if SAFE_EXECUTION_AVAILABLE:
            assert result['conclusion']['success'] is True

    def test_reason_with_problem_key(self, tool):
        """Test reason method with dict input containing 'problem' key."""
        result = tool.reason({'problem': 'Solve x^2 - 4 = 0'})
        
        assert isinstance(result, dict)
        assert 'conclusion' in result

    def test_reason_returns_formatted_output(self, tool):
        """Test that reason method includes formatted output."""
        result = tool.reason("Integrate x^2")
        
        assert 'formatted_output' in result
        assert '**Mathematical Computation**' in result['formatted_output']

    def test_reason_returns_metadata(self, tool):
        """Test that reason method includes metadata."""
        result = tool.reason("Integrate x^2")
        
        assert 'metadata' in result
        assert 'tool' in result['metadata']
        assert result['metadata']['tool'] == 'mathematical_computation'

    def test_get_capabilities(self, tool):
        """Test get_capabilities method for UnifiedReasoner."""
        caps = tool.get_capabilities()
        
        assert isinstance(caps, dict)
        assert 'name' in caps
        assert 'description' in caps
        assert 'supported_problem_types' in caps
        assert 'supported_strategies' in caps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
