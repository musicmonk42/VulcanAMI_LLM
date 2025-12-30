"""
Tests for safe mathematical code execution and MathematicalComputationTool.

These tests verify:
1. Safe code execution module functionality
2. RestrictedPython sandbox security
3. MathematicalComputationTool integration
4. Code generation and execution pipeline
"""

import pytest

# Try to import the modules
try:
    from vulcan.utils.safe_execution import (
        SafeCodeExecutor,
        execute_math_code,
        is_safe_execution_available,
        reset_executor,
        RESTRICTED_PYTHON_AVAILABLE,
        SYMPY_AVAILABLE,
    )
    SAFE_EXECUTION_MODULE_AVAILABLE = True
except ImportError:
    SAFE_EXECUTION_MODULE_AVAILABLE = False

try:
    from vulcan.reasoning.mathematical_verification import (
        MathematicalComputationTool,
        ComputationResult,
    )
    COMPUTATION_TOOL_AVAILABLE = True
except ImportError:
    COMPUTATION_TOOL_AVAILABLE = False


# ============================================================================
# SAFE EXECUTION TESTS
# ============================================================================


@pytest.fixture(autouse=True)
def reset_executor_before_test():
    """Reset executor before each test to ensure clean state."""
    if SAFE_EXECUTION_MODULE_AVAILABLE:
        reset_executor()


@pytest.mark.skipif(
    not SAFE_EXECUTION_MODULE_AVAILABLE,
    reason="Safe execution module not available"
)
class TestSafeExecution:
    """Tests for the safe_execution module."""

    def test_is_safe_execution_available(self):
        """Test availability check returns correct status."""
        result = is_safe_execution_available()
        # Should be True if both RestrictedPython and SymPy are available
        assert isinstance(result, bool)
        assert result == (RESTRICTED_PYTHON_AVAILABLE and SYMPY_AVAILABLE)

    def test_basic_integration(self):
        """Test basic SymPy integration."""
        result = execute_math_code('''
x = Symbol('x')
result = integrate(x**2, x)
''')
        assert result['success'] is True
        assert result['error'] is None
        assert 'x**3/3' in str(result['result'])

    def test_differentiation(self):
        """Test SymPy differentiation."""
        result = execute_math_code('''
x = Symbol('x')
f = x**3 + 2*x**2 + x
result = diff(f, x)
''')
        assert result['success'] is True
        # Result should be 3*x**2 + 4*x + 1
        result_str = str(result['result'])
        assert '3*x**2' in result_str or 'x**2' in result_str

    def test_solve_equation(self):
        """Test solving algebraic equations."""
        result = execute_math_code('''
x = Symbol('x')
result = solve(x**2 - 4, x)
''')
        assert result['success'] is True
        # Solution should be [-2, 2]
        result_val = result['result']
        assert -2 in result_val or str(-2) in str(result_val)
        assert 2 in result_val or str(2) in str(result_val)

    def test_limit_computation(self):
        """Test computing limits."""
        result = execute_math_code('''
x = Symbol('x')
result = limit(sin(x)/x, x, 0)
''')
        assert result['success'] is True
        assert str(result['result']) == '1'

    def test_matrix_operations(self):
        """Test matrix operations."""
        result = execute_math_code('''
M = Matrix([[1, 2], [3, 4]])
result = M.det()
''')
        assert result['success'] is True
        assert str(result['result']) == '-2'

    def test_series_expansion(self):
        """Test Taylor series expansion."""
        result = execute_math_code('''
x = Symbol('x')
result = series(exp(x), x, 0, 5)
''')
        assert result['success'] is True
        result_str = str(result['result'])
        assert '1 + x' in result_str
        assert 'x**2' in result_str

    def test_definite_integral(self):
        """Test definite integration."""
        result = execute_math_code('''
x = Symbol('x')
k = Symbol('k', real=True, positive=True)
f = exp(-k*x)
result = integrate(f, (x, 0, oo))
''')
        assert result['success'] is True
        assert '1/k' in str(result['result'])

    def test_symbolic_with_assumptions(self):
        """Test symbolic computation with variable assumptions."""
        result = execute_math_code('''
n = Symbol('n', integer=True, positive=True)
result = factorial(n) / factorial(n-1)
''')
        assert result['success'] is True
        assert 'n' in str(result['result'])

    def test_simplification(self):
        """Test expression simplification."""
        result = execute_math_code('''
x = Symbol('x')
expr = (x + 1)**2 - x**2 - 2*x - 1
# Expand first, then simplify
expanded = expand(expr)
result = simplify(expanded)
''')
        assert result['success'] is True
        # Should simplify to 0
        assert str(result['result']) == '0'

    def test_factorization(self):
        """Test polynomial factorization."""
        result = execute_math_code('''
x = Symbol('x')
result = factor(x**2 + 2*x + 1)
''')
        assert result['success'] is True
        assert '(x + 1)**2' in str(result['result'])

    def test_syntax_error_handling(self):
        """Test that syntax errors are caught."""
        result = execute_math_code('''
x = Symbol('x'
result = integrate(x, x)
''')
        assert result['success'] is False, "Should fail on syntax error"
        assert result['error'] is not None, "Error message should be present"
        error_lower = result['error'].lower()
        assert 'syntax' in error_lower or 'compilation' in error_lower, \
            f"Error should mention syntax or compilation, got: {result['error']}"

    def test_undefined_name_error(self):
        """Test that undefined names are caught."""
        result = execute_math_code('''
result = undefined_function(x)
''')
        assert result['success'] is False
        assert result['error'] is not None

    def test_result_variable_required(self):
        """Test that result must be assigned."""
        result = execute_math_code('''
x = Symbol('x')
y = integrate(x**2, x)
# result not assigned
''')
        # Should succeed but result will be None
        assert result['success'] is True
        assert result['result'] is None

    def test_answer_variable_alternative(self):
        """Test that 'answer' variable is also accepted."""
        result = execute_math_code('''
x = Symbol('x')
answer = integrate(x**2, x)
''')
        assert result['success'] is True
        assert result['result'] is not None


@pytest.mark.skipif(
    not SAFE_EXECUTION_MODULE_AVAILABLE,
    reason="Safe execution module not available"
)
class TestSafeExecutionSecurity:
    """Security tests for safe execution sandbox."""

    def test_no_file_access(self):
        """Verify file operations are blocked."""
        result = execute_math_code('''
result = open('/etc/passwd', 'r').read()
''')
        assert result['success'] is False

    def test_no_import(self):
        """Verify imports are blocked."""
        result = execute_math_code('''
import os
result = os.getcwd()
''')
        assert result['success'] is False

    def test_no_eval(self):
        """Verify eval is not available."""
        result = execute_math_code('''
result = eval('1 + 1')
''')
        assert result['success'] is False

    def test_no_exec(self):
        """Verify exec is not available."""
        result = execute_math_code('''
exec('result = 1')
''')
        assert result['success'] is False

    def test_no_subprocess(self):
        """Verify subprocess access is blocked."""
        result = execute_math_code('''
import subprocess
result = subprocess.check_output(['ls'])
''')
        assert result['success'] is False


# ============================================================================
# MATHEMATICAL COMPUTATION TOOL TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="MathematicalComputationTool not available"
)
class TestMathematicalComputationTool:
    """Tests for MathematicalComputationTool."""

    def test_tool_initialization(self):
        """Test tool initializes correctly."""
        tool = MathematicalComputationTool()
        assert tool.name == "mathematical_computation"
        assert tool.description is not None

    def test_integration_execution(self):
        """Test integration query execution."""
        tool = MathematicalComputationTool()
        result = tool.execute("Integrate x^2 with respect to x")
        
        assert isinstance(result, ComputationResult)
        assert result.success is True
        assert result.code is not None
        assert 'integrate' in result.code.lower()
        assert result.result is not None
        assert 'x**3/3' in result.result

    def test_differentiation_execution(self):
        """Test differentiation query execution."""
        tool = MathematicalComputationTool()
        result = tool.execute("Find the derivative of x^3")
        
        assert result.success is True
        assert 'diff' in result.code.lower()

    def test_solve_execution(self):
        """Test equation solving query execution."""
        tool = MathematicalComputationTool()
        result = tool.execute("Solve x^2 - 4 = 0")
        
        assert result.success is True
        assert 'solve' in result.code.lower()

    def test_limit_execution(self):
        """Test limit query execution."""
        tool = MathematicalComputationTool()
        result = tool.execute("Find the limit as x approaches 0 of sin(x)/x")
        
        assert result.success is True
        assert 'limit' in result.code.lower()
        assert result.result == '1'

    def test_series_execution(self):
        """Test series expansion query execution."""
        tool = MathematicalComputationTool()
        result = tool.execute("Taylor series expansion of e^x")
        
        assert result.success is True
        assert 'series' in result.code.lower()

    def test_format_response_success(self):
        """Test response formatting for successful computation."""
        tool = MathematicalComputationTool()
        result = tool.execute("Integrate x^2")
        
        formatted = tool.format_response(result)
        
        assert '**Mathematical Computation**' in formatted
        assert '**Code:**' in formatted
        assert '**Result:**' in formatted
        assert 'x**3/3' in formatted

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

    def test_tool_without_llm(self):
        """Test tool works without LLM using templates."""
        tool = MathematicalComputationTool(llm=None)
        result = tool.execute("Integrate x^2")
        
        # Should still work using template-based code generation
        assert result.success is True
        assert result.code is not None

    def test_result_includes_metadata(self):
        """Test that result includes metadata."""
        tool = MathematicalComputationTool()
        result = tool.execute("Integrate x^2")
        
        assert result.tool == "mathematical_computation"
        assert result.metadata is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.skipif(
    not (SAFE_EXECUTION_MODULE_AVAILABLE and COMPUTATION_TOOL_AVAILABLE),
    reason="Required modules not available"
)
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_complex_derivation(self):
        """Test a more complex mathematical derivation."""
        result = execute_math_code('''
# Define variables
x = Symbol('x', real=True)
a = Symbol('a', real=True, positive=True)

# Gaussian integral (simplified)
f = exp(-a * x**2)

# Integrate from -oo to oo
integral = integrate(f, (x, -oo, oo))

result = simplify(integral)
''')
        assert result['success'] is True
        # Result should involve sqrt(pi/a) or equivalent
        result_str = str(result['result'])
        assert 'sqrt' in result_str or 'pi' in result_str

    def test_differential_equation_simple(self):
        """Test solving a simple differential equation."""
        result = execute_math_code('''
x = Symbol('x')
f = Function('f')

# Define ODE: f'(x) = f(x)
ode = Eq(f(x).diff(x), f(x))

# Solve
solution = dsolve(ode, f(x))

# Convert to string to avoid Eq truth value issues
result = str(solution)
''')
        assert result['success'] is True
        # Solution should be f(x) = C1*exp(x)
        result_str = str(result['result'])
        assert 'exp' in result_str

    def test_linear_algebra_operations(self):
        """Test linear algebra operations."""
        result = execute_math_code('''
# Define a 3x3 matrix
A = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Compute rank (should be 2 since rows are linearly dependent)
result = A.rank()
''')
        assert result['success'] is True
        assert str(result['result']) == '2'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
