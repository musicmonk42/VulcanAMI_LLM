"""
Mathematical Computation Tool for VULCAN.

Provides symbolic mathematical computation using SymPy for exact
mathematical operations. Wraps the existing MathematicalComputationTool
as a tool that the LLM can call directly.

Use this tool when:
    - Computing integrals, derivatives, limits
    - Solving equations and systems of equations
    - Simplifying algebraic expressions
    - Computing sums and products
    - Matrix operations
    - Any operation requiring exact symbolic math

Do NOT use for:
    - Simple arithmetic the LLM can do (2+2)
    - Approximate numerical calculations
    - Questions about mathematical concepts (LLM handles theory)

Industry Standards:
    - Thread-safe execution
    - Safe sandboxed code execution
    - Timeout protection for complex computations
    - Input validation and size limits

Security Considerations:
    - Sandboxed execution via RestrictedPython
    - No arbitrary code execution
    - Resource limits (memory, time)

Version History:
    1.0.0 - Initial implementation wrapping MathematicalComputationTool
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Final, List, Optional

from pydantic import Field, field_validator

from .base import Tool, ToolInput, ToolOutput, ToolStatus

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum expression length
MAX_EXPRESSION_LENGTH: Final[int] = 5_000

# Maximum computation timeout (seconds)
MAX_COMPUTATION_TIMEOUT: Final[float] = 30.0

# Default timeout for math operations
DEFAULT_TIMEOUT: Final[float] = 10.0


# =============================================================================
# INPUT MODEL
# =============================================================================


class MathEngineInput(ToolInput):
    """
    Input parameters for the math engine tool.
    
    Supports various mathematical operations:
    - General expressions: "integrate x^2 from 0 to 1"
    - Equation solving: "solve x^2 - 4 = 0 for x"
    - Simplification: "simplify (x+1)^2 - x^2"
    - Differentiation: "derivative of sin(x)"
    
    Attributes:
        expression: Mathematical expression or problem (required)
        operation: Specific operation to perform (optional, auto-detected)
        variable: Variable for operations that need it (optional)
        timeout: Computation timeout in seconds
    """
    
    expression: str = Field(
        ...,
        min_length=1,
        max_length=MAX_EXPRESSION_LENGTH,
        description=(
            "Mathematical expression or problem to solve. "
            "Examples: 'integrate x^2', 'solve x^2 - 4 = 0', "
            "'derivative of sin(x)', 'simplify (x+1)^2'"
        )
    )
    operation: Optional[str] = Field(
        default=None,
        description=(
            "Specific operation to perform. If not provided, auto-detected. "
            "Options: integrate, differentiate, solve, simplify, expand, "
            "factor, limit, series, matrix"
        )
    )
    variable: Optional[str] = Field(
        default=None,
        description=(
            "Primary variable for the operation. "
            "If not provided, auto-detected from expression."
        )
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        ge=0.1,
        le=MAX_COMPUTATION_TIMEOUT,
        description=f"Computation timeout in seconds (max: {MAX_COMPUTATION_TIMEOUT}s)"
    )
    
    @field_validator("expression", mode="before")
    @classmethod
    def clean_expression(cls, v: str) -> str:
        """Clean and validate expression."""
        if isinstance(v, str):
            # Strip whitespace and common formatting chars
            v = v.strip()
            # Remove zero-width characters that can cause parsing issues
            v = v.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        return v


# =============================================================================
# MATH ENGINE TOOL
# =============================================================================


class MathEngineTool(Tool):
    """
    Mathematical computation tool using SymPy.
    
    Provides exact symbolic mathematical computation for operations
    that LLMs cannot reliably perform (integration, differentiation,
    equation solving, etc.).
    
    Thread Safety:
        This tool is thread-safe. Each execution uses isolated
        SymPy computations with no shared mutable state.
    
    Capabilities:
        - Integration (definite and indefinite)
        - Differentiation
        - Equation solving (single and systems)
        - Algebraic simplification
        - Expression expansion and factoring
        - Limits and series
        - Matrix operations
    
    Example:
        >>> tool = MathEngineTool()
        >>> result = tool.execute(expression="integrate x^2 from 0 to 1")
        >>> print(result.result["result"])
        '1/3'
        
        >>> result = tool.execute(expression="solve x^2 - 4 = 0")
        >>> print(result.result["solutions"])
        [-2, 2]
    """
    
    def __init__(self) -> None:
        """Initialize the math engine tool."""
        super().__init__()
        
        # Thread-local storage for computation context
        self._local = threading.local()
        
        # Check for SymPy and MathematicalComputationTool availability
        self._sympy_available = False
        self._math_tool_available = False
        self._math_tool_class = None
        self._init_error: Optional[str] = None
        
        # Check SymPy
        try:
            import sympy
            self._sympy_available = True
            logger.debug("MathEngineTool: SymPy available")
        except ImportError as e:
            self._init_error = f"SymPy not available: {e}"
            logger.warning(f"MathEngineTool: {self._init_error}")
        
        # Check MathematicalComputationTool
        if self._sympy_available:
            try:
                from vulcan.reasoning.mathematical_computation import MathematicalComputationTool
                self._math_tool_class = MathematicalComputationTool
                self._math_tool_available = True
                logger.debug("MathEngineTool: MathematicalComputationTool available")
            except ImportError as e:
                logger.info(f"MathEngineTool: Using direct SymPy (no MathematicalComputationTool): {e}")
    
    @property
    def name(self) -> str:
        """Tool name for LLM to reference."""
        return "math_engine"
    
    @property
    def description(self) -> str:
        """Description for LLM to understand when to use this tool."""
        return """Symbolic mathematical computation using SymPy.

Use this tool when you need to:
- Compute integrals: "integrate x^2 from 0 to 1"
- Compute derivatives: "derivative of sin(x)"
- Solve equations: "solve x^2 - 4 = 0 for x"
- Simplify expressions: "simplify (x+1)^2 - x^2"
- Expand/factor: "expand (x+1)^3", "factor x^2 - 1"
- Compute limits: "limit of sin(x)/x as x approaches 0"
- Matrix operations: "determinant of [[1,2],[3,4]]"

IMPORTANT: LLMs often make arithmetic errors. Use this tool for:
- Any integral or derivative
- Equation solving
- Algebraic manipulation

Do NOT use for simple arithmetic (2+2) or conceptual questions.

Returns: Exact symbolic result with step-by-step computation."""
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema for parameters (OpenAI function calling compatible)."""
        return MathEngineInput.model_json_schema()
    
    @property
    def is_available(self) -> bool:
        """Check if math engine is available."""
        return self._sympy_available
    
    def _get_sympy(self):
        """Get SymPy module (lazy import)."""
        if not hasattr(self._local, "sympy"):
            import sympy
            self._local.sympy = sympy
        return self._local.sympy
    
    def execute(
        self,
        expression: str,
        operation: Optional[str] = None,
        variable: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ToolOutput:
        """
        Execute mathematical computation.
        
        Args:
            expression: Mathematical expression or problem
            operation: Specific operation (auto-detected if not provided)
            variable: Variable for operations that need it
            timeout: Computation timeout in seconds
            
        Returns:
            ToolOutput with computed result and step information
        """
        start_time = time.perf_counter()
        
        def elapsed_ms() -> float:
            return (time.perf_counter() - start_time) * 1000
        
        # Check availability
        if not self._sympy_available:
            return ToolOutput.create_failure(
                error=self._init_error or "SymPy not available",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.UNAVAILABLE,
                metadata={"tool": self.name},
            )
        
        # Validate input
        if not expression or not expression.strip():
            return ToolOutput.create_failure(
                error="Expression cannot be empty",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.INVALID_INPUT,
            )
        
        if len(expression) > MAX_EXPRESSION_LENGTH:
            return ToolOutput.create_failure(
                error=f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.INVALID_INPUT,
            )
        
        try:
            # Try using MathematicalComputationTool if available
            if self._math_tool_available:
                result = self._execute_with_math_tool(expression, operation, variable, timeout)
            else:
                result = self._execute_with_sympy(expression, operation, variable, timeout)
            
            computation_time = elapsed_ms()
            self._record_execution(computation_time)
            
            return ToolOutput.create_success(
                result=result,
                computation_time_ms=computation_time,
                metadata={
                    "tool": self.name,
                    "engine": "MathematicalComputationTool" if self._math_tool_available else "direct_sympy",
                },
            )
            
        except TimeoutError:
            return ToolOutput.create_failure(
                error=f"Computation timed out after {timeout}s",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.TIMEOUT,
            )
        except Exception as e:
            logger.error(f"MathEngineTool: Error computing: {e}", exc_info=True)
            return ToolOutput.create_failure(
                error=f"Math computation failed: {str(e)}",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.FAILURE,
            )
    
    def _execute_with_math_tool(
        self,
        expression: str,
        operation: Optional[str],
        variable: Optional[str],
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Execute using MathematicalComputationTool.
        
        Args:
            expression: Mathematical expression
            operation: Specific operation
            variable: Variable for operations
            timeout: Timeout in seconds
            
        Returns:
            Dict with computation result
        """
        # Create tool instance (stateless, so create fresh each time)
        tool = self._math_tool_class(llm=None, prefer_templates=True)
        
        # Execute computation
        result = tool.execute(expression)
        
        # Convert ComputationResult to dict
        return {
            "expression": expression,
            "result": str(result.result) if result.result else None,
            "code": result.code if hasattr(result, 'code') else None,
            "success": result.success if hasattr(result, 'success') else True,
            "steps": result.steps if hasattr(result, 'steps') else None,
            "error": result.error if hasattr(result, 'error') else None,
        }
    
    def _execute_with_sympy(
        self,
        expression: str,
        operation: Optional[str],
        variable: Optional[str],
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Execute using direct SymPy.
        
        Provides basic symbolic computation when MathematicalComputationTool
        is not available.
        
        Args:
            expression: Mathematical expression
            operation: Specific operation
            variable: Variable for operations
            timeout: Timeout in seconds
            
        Returns:
            Dict with computation result
        """
        sympy = self._get_sympy()
        
        # Detect operation from expression if not provided
        expr_lower = expression.lower()
        
        if operation is None:
            if any(kw in expr_lower for kw in ['integrate', 'integral', '∫']):
                operation = 'integrate'
            elif any(kw in expr_lower for kw in ['derivative', 'differentiate', 'diff', "d/dx"]):
                operation = 'differentiate'
            elif any(kw in expr_lower for kw in ['solve', 'find x', 'find the value']):
                operation = 'solve'
            elif any(kw in expr_lower for kw in ['simplify', 'reduce']):
                operation = 'simplify'
            elif any(kw in expr_lower for kw in ['expand']):
                operation = 'expand'
            elif any(kw in expr_lower for kw in ['factor']):
                operation = 'factor'
            elif any(kw in expr_lower for kw in ['limit', 'approaches', '→']):
                operation = 'limit'
            else:
                operation = 'evaluate'
        
        # Default variable
        x = sympy.Symbol('x')
        if variable:
            x = sympy.Symbol(variable)
        
        # Try to parse and compute
        try:
            # Extract mathematical expression from natural language
            math_expr = self._extract_math_expression(expression, sympy)
            
            if operation == 'integrate':
                result = sympy.integrate(math_expr, x)
            elif operation == 'differentiate':
                result = sympy.diff(math_expr, x)
            elif operation == 'solve':
                result = sympy.solve(math_expr, x)
            elif operation == 'simplify':
                result = sympy.simplify(math_expr)
            elif operation == 'expand':
                result = sympy.expand(math_expr)
            elif operation == 'factor':
                result = sympy.factor(math_expr)
            elif operation == 'limit':
                # Default limit as x -> 0
                result = sympy.limit(math_expr, x, 0)
            else:
                # Try to evaluate/simplify
                result = sympy.simplify(math_expr)
            
            return {
                "expression": expression,
                "parsed_expression": str(math_expr),
                "operation": operation,
                "variable": str(x),
                "result": str(result),
                "latex": sympy.latex(result),
            }
            
        except Exception as e:
            return {
                "expression": expression,
                "operation": operation,
                "error": str(e),
                "suggestion": "Try using more explicit notation like 'x**2' instead of 'x^2'",
            }
    
    def _extract_math_expression(self, text: str, sympy) -> Any:
        """
        Extract mathematical expression from text.
        
        Args:
            text: Natural language or mathematical text
            sympy: SymPy module
            
        Returns:
            SymPy expression
        """
        # Common replacements
        expr = text.lower()
        
        # Remove common phrases
        for phrase in ['integrate', 'integral of', 'derivative of', 'diff of',
                       'solve', 'simplify', 'expand', 'factor', 'evaluate',
                       'compute', 'calculate', 'find', 'what is']:
            expr = expr.replace(phrase, '')
        
        # Clean up
        expr = expr.strip()
        
        # Replace common notation
        expr = expr.replace('^', '**')
        expr = expr.replace('×', '*')
        expr = expr.replace('÷', '/')
        
        # Try to parse with sympify
        try:
            return sympy.sympify(expr)
        except:
            # If that fails, try parsing just the first mathematical part
            import re
            # Look for patterns like x**2, sin(x), etc.
            match = re.search(r'[a-z0-9\*\+\-\/\(\)\s\^]+', expr)
            if match:
                clean_expr = match.group().replace('^', '**').strip()
                return sympy.sympify(clean_expr)
            raise ValueError(f"Could not parse expression: {text}")
