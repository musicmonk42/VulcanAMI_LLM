"""
invariant_detector.py - Invariant detection and management for World Model
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
Integrated with comprehensive safety validation.
FIXED: API compatibility - check_invariant_violations accepts dict or float, check() method added
FIXED: eval() security vulnerability - uses AST with safe operation whitelist
IMPLEMENTED: Full symbolic expression system with sympy integration and safety checks
FIXED: Circular import with safety_validator using lazy loading
"""

import ast
import logging
import operator
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Initialize logger early before any usage
logger = logging.getLogger(__name__)

# DO NOT import safety validator at module level - use lazy loading
# This prevents circular import: invariant_detector -> safety_validator -> domain_validators -> invariant_detector
EnhancedSafetyValidator = None
SafetyConfig = None
SAFETY_VALIDATOR_AVAILABLE = False


def _lazy_load_safety_validator():
    """Lazy load safety validator to avoid circular imports"""
    global EnhancedSafetyValidator, SafetyConfig, SAFETY_VALIDATOR_AVAILABLE

    if EnhancedSafetyValidator is None:
        try:
            from ..safety.safety_types import SafetyConfig as _SC
            from ..safety.safety_validator import EnhancedSafetyValidator as _ESV

            EnhancedSafetyValidator = _ESV
            SafetyConfig = _SC
            SAFETY_VALIDATOR_AVAILABLE = True
            logger.info("Safety validator loaded successfully via lazy loading")
        except ImportError as e:
            logger.warning(f"safety_validator not available: {e}")
            SAFETY_VALIDATOR_AVAILABLE = False


# Protected imports with fallbacks
try:
    import sympy as sp
    from sympy.core.sympify import SympifyError
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logging.warning("sympy not available, using fallback implementations")

try:
    from scipy import stats
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, using fallback implementations")

logger = logging.getLogger(__name__)


class ExpressionComplexityError(Exception):
    """Raised when expression complexity exceeds limits"""


class ExpressionSafetyError(Exception):
    """Raised when expression contains unsafe operations"""


class SymbolicExpressionSystem:
    """
    Comprehensive symbolic expression system with sympy integration

    Features:
    - Full symbolic parsing and manipulation
    - Safety checks for complexity and operations
    - Variable extraction and substitution
    - Expression simplification and solving
    - Fallback to AST-based evaluation when sympy unavailable
    """

    # Safe operations whitelist
    SAFE_OPERATIONS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    SAFE_COMPARISONS = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    SAFE_FUNCTIONS = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "floor": np.floor,
        "ceil": np.ceil,
        "round": round,
    }

    # Complexity limits
    MAX_EXPRESSION_LENGTH = 10000
    MAX_TREE_DEPTH = 20
    MAX_VARIABLE_COUNT = 100
    MAX_OPERATION_COUNT = 500

    def __init__(self, safety_validator: Optional[Any] = None):
        """
        Initialize symbolic expression system

        Args:
            safety_validator: Optional safety validator for additional checks
        """
        self.safety_validator = safety_validator
        self.use_sympy = SYMPY_AVAILABLE

        # Caching
        self._parse_cache = {}
        self._simplify_cache = {}
        self._cache_lock = threading.Lock()

        # Statistics
        self.complexity_blocks = 0
        self.safety_blocks = 0
        self.parse_errors = 0

        logger.info("SymbolicExpressionSystem initialized (sympy=%s)", self.use_sympy)

    def parse(
        self, expr_str: str, variables: Optional[Set[str]] = None
    ) -> "SymbolicExpression":
        """
        Parse expression string into SymbolicExpression

        Args:
            expr_str: Expression string to parse
            variables: Optional set of expected variable names

        Returns:
            SymbolicExpression object

        Raises:
            ExpressionSafetyError: If expression is unsafe
            ExpressionComplexityError: If expression is too complex
        """

        # Check cache
        cache_key = (expr_str, tuple(sorted(variables)) if variables else None)
        with self._cache_lock:
            if cache_key in self._parse_cache:
                return self._parse_cache[cache_key]

        # EXAMINE: Validate expression safety
        self._validate_expression_safety(expr_str)

        # SELECT & APPLY: Parse with appropriate method
        if self.use_sympy:
            expr = self._parse_with_sympy(expr_str, variables)
        else:
            expr = self._parse_with_ast(expr_str, variables)

        # REMEMBER: Cache result
        with self._cache_lock:
            if len(self._parse_cache) < 10000:  # Limit cache size
                self._parse_cache[cache_key] = expr

        return expr

    def create_symbol(self, name: str) -> Union["sp.Symbol", "SimpleSymbol"]:
        """Create a symbolic variable"""

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid variable name: {name}")

        if self.use_sympy:
            return sp.Symbol(name, real=True)
        else:
            return SimpleSymbol(name)

    def simplify(self, expr: "SymbolicExpression") -> "SymbolicExpression":
        """
        Simplify symbolic expression

        Args:
            expr: Expression to simplify

        Returns:
            Simplified expression
        """

        # Check cache
        expr_str = str(expr.expression)
        with self._cache_lock:
            if expr_str in self._simplify_cache:
                return self._simplify_cache[expr_str]

        if self.use_sympy and hasattr(expr.expression, "simplify"):
            try:
                simplified = expr.expression.simplify()
                result = SymbolicExpression(simplified, expr.variables.copy())

                # Cache result
                with self._cache_lock:
                    if len(self._simplify_cache) < 10000:
                        self._simplify_cache[expr_str] = result

                return result
            except Exception as e:
                logger.debug("Simplification failed: %s", e)

        return expr

    def solve(
        self,
        expr: "SymbolicExpression",
        variable: Union[str, "sp.Symbol", "SimpleSymbol"],
    ) -> List[Any]:
        """
        Solve expression for variable

        Args:
            expr: Expression to solve
            variable: Variable to solve for

        Returns:
            List of solutions
        """

        if not self.use_sympy:
            logger.warning("Solving requires sympy, which is not available")
            return []

        try:
            if isinstance(variable, str):
                variable = sp.Symbol(variable)

            solutions = sp.solve(expr.expression, variable)
            return solutions
        except Exception as e:
            logger.debug("Solve failed: %s", e)
            return []

    def check_equality(
        self, expr1: "SymbolicExpression", expr2: "SymbolicExpression"
    ) -> bool:
        """
        Check if two expressions are mathematically equal

        Args:
            expr1: First expression
            expr2: Second expression

        Returns:
            True if expressions are equal
        """

        if self.use_sympy:
            try:
                diff = sp.simplify(expr1.expression - expr2.expression)
                return diff == 0
            except Exception as e:
                logger.debug("Equality check failed: %s", e)

        # Fallback: string comparison
        return str(expr1.expression) == str(expr2.expression)

    def expand(self, expr: "SymbolicExpression") -> "SymbolicExpression":
        """Expand expression"""

        if self.use_sympy and hasattr(expr.expression, "expand"):
            try:
                expanded = expr.expression.expand()
                return SymbolicExpression(expanded, expr.variables.copy())
            except Exception as e:
                logger.debug("Expansion failed: %s", e)

        return expr

    def factor(self, expr: "SymbolicExpression") -> "SymbolicExpression":
        """Factor expression"""

        if self.use_sympy and hasattr(expr.expression, "factor"):
            try:
                factored = expr.expression.factor()
                return SymbolicExpression(factored, expr.variables.copy())
            except Exception as e:
                logger.debug("Factoring failed: %s", e)

        return expr

    def _validate_expression_safety(self, expr_str: str):
        """
        Validate expression for safety and complexity

        Raises:
            ExpressionSafetyError: If expression is unsafe
            ExpressionComplexityError: If expression is too complex
        """

        # Check length
        if len(expr_str) > self.MAX_EXPRESSION_LENGTH:
            self.complexity_blocks += 1
            raise ExpressionComplexityError(
                f"Expression too long: {len(expr_str)} > {self.MAX_EXPRESSION_LENGTH}"
            )

        # Check for dangerous patterns
        dangerous_patterns = [
            r"__\w+__",  # Dunder methods
            r"import\s",  # Import statements
            r"exec\s*\(",  # Exec calls
            r"eval\s*\(",  # Eval calls
            r"compile\s*\(",  # Compile calls
            r"__builtins__",  # Builtins access
            r"globals\s*\(",  # Globals access
            r"locals\s*\(",  # Locals access
            r"open\s*\(",  # File operations
            r"file\s*\(",  # File operations
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expr_str, re.IGNORECASE):
                self.safety_blocks += 1
                raise ExpressionSafetyError(
                    f"Expression contains unsafe pattern: {pattern}"
                )

        # Parse and check AST complexity
        try:
            tree = ast.parse(expr_str, mode="eval")
            self._check_ast_safety(tree)
        except SyntaxError as e:
            self.parse_errors += 1
            raise ExpressionSafetyError(f"Invalid expression syntax: {e}")

    def _check_ast_safety(self, node: ast.AST, depth: int = 0):
        """
        Recursively check AST node safety

        Args:
            node: AST node to check
            depth: Current tree depth

        Raises:
            ExpressionComplexityError: If tree is too deep
            ExpressionSafetyError: If unsafe operations detected
        """

        # Check depth
        if depth > self.MAX_TREE_DEPTH:
            self.complexity_blocks += 1
            raise ExpressionComplexityError(
                f"Expression tree too deep: {depth} > {self.MAX_TREE_DEPTH}"
            )

        # Check node type
        if isinstance(node, ast.Expression):
            self._check_ast_safety(node.body, depth + 1)

        elif isinstance(node, (ast.Constant, ast.Num)):
            pass  # Safe

        elif isinstance(node, ast.Name):
            pass  # Safe - variable reference

        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.SAFE_OPERATIONS:
                self.safety_blocks += 1
                raise ExpressionSafetyError(
                    f"Unsafe binary operation: {type(node.op).__name__}"
                )
            self._check_ast_safety(node.left, depth + 1)
            self._check_ast_safety(node.right, depth + 1)

        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.SAFE_OPERATIONS:
                self.safety_blocks += 1
                raise ExpressionSafetyError(
                    f"Unsafe unary operation: {type(node.op).__name__}"
                )
            self._check_ast_safety(node.operand, depth + 1)

        elif isinstance(node, ast.Compare):
            self._check_ast_safety(node.left, depth + 1)
            for op in node.ops:
                if type(op) not in self.SAFE_COMPARISONS:
                    self.safety_blocks += 1
                    raise ExpressionSafetyError(
                        f"Unsafe comparison: {type(op).__name__}"
                    )
            for comparator in node.comparators:
                self._check_ast_safety(comparator, depth + 1)

        elif isinstance(node, ast.Call):
            # Check function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in self.SAFE_FUNCTIONS:
                    self.safety_blocks += 1
                    raise ExpressionSafetyError(f"Unsafe function call: {func_name}")
            else:
                self.safety_blocks += 1
                raise ExpressionSafetyError("Complex function calls not allowed")

            # Check arguments
            for arg in node.args:
                self._check_ast_safety(arg, depth + 1)

        elif isinstance(node, (ast.BoolOp, ast.IfExp)):
            # Boolean operations and conditionals
            for child in ast.iter_child_nodes(node):
                self._check_ast_safety(child, depth + 1)

        else:
            self.safety_blocks += 1
            raise ExpressionSafetyError(f"Unsafe AST node type: {type(node).__name__}")

    def _parse_with_sympy(
        self, expr_str: str, variables: Optional[Set[str]] = None
    ) -> "SymbolicExpression":
        """Parse expression using sympy"""

        try:
            # Use sympy parser with safe transformations
            transformations = standard_transformations + (
                implicit_multiplication_application,
            )

            expr = parse_expr(
                expr_str,
                transformations=transformations,
                evaluate=False,  # Don't auto-evaluate
            )

            # Extract variables
            extracted_vars = {str(s) for s in expr.free_symbols}

            # Validate variable count
            if len(extracted_vars) > self.MAX_VARIABLE_COUNT:
                self.complexity_blocks += 1
                raise ExpressionComplexityError(
                    f"Too many variables: {len(extracted_vars)} > {self.MAX_VARIABLE_COUNT}"
                )

            # Check expected variables
            if variables is not None:
                unexpected = extracted_vars - variables
                if unexpected:
                    logger.warning("Unexpected variables in expression: %s", unexpected)

            return SymbolicExpression(expr, extracted_vars)

        except SympifyError as e:
            self.parse_errors += 1
            raise ExpressionSafetyError(f"Sympy parsing failed: {e}")

    def _parse_with_ast(
        self, expr_str: str, variables: Optional[Set[str]] = None
    ) -> "SymbolicExpression":
        """Parse expression using AST fallback"""

        try:
            tree = ast.parse(expr_str, mode="eval")

            # Extract variables
            extractor = VariableExtractor()
            extractor.visit(tree)
            extracted_vars = extractor.variables

            # Validate variable count
            if len(extracted_vars) > self.MAX_VARIABLE_COUNT:
                self.complexity_blocks += 1
                raise ExpressionComplexityError(
                    f"Too many variables: {len(extracted_vars)} > {self.MAX_VARIABLE_COUNT}"
                )

            # Create SimpleExpression
            simple_expr = SimpleExpression(expr_str, extracted_vars, tree)

            return SymbolicExpression(simple_expr, extracted_vars)

        except SyntaxError as e:
            self.parse_errors += 1
            raise ExpressionSafetyError(f"AST parsing failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "using_sympy": self.use_sympy,
            "cache_size": len(self._parse_cache),
            "simplify_cache_size": len(self._simplify_cache),
            "complexity_blocks": self.complexity_blocks,
            "safety_blocks": self.safety_blocks,
            "parse_errors": self.parse_errors,
        }


class VariableExtractor(ast.NodeVisitor):
    """Extract variable names from AST"""

    def __init__(self):
        self.variables = set()

    def visit_Name(self, node):
        # Skip Python keywords and safe functions
        if (
            node.id not in {"True", "False", "None"}
            and node.id not in SymbolicExpressionSystem.SAFE_FUNCTIONS
        ):
            self.variables.add(node.id)
        self.generic_visit(node)


@dataclass
class SymbolicExpression:
    """
    Wrapper for symbolic expressions

    Provides unified interface for both sympy and fallback implementations
    """

    expression: Any  # sp.Expr or SimpleExpression
    variables: Set[str]

    def substitute(self, substitutions: Dict[str, float]) -> Union[float, bool, Any]:
        """
        Substitute values into expression

        Args:
            substitutions: Dictionary mapping variable names to values

        Returns:
            Result of substitution (numeric or boolean)
        """

        if SYMPY_AVAILABLE and isinstance(self.expression, sp.Basic):
            # Sympy substitution
            subs_dict = {sp.Symbol(var): val for var, val in substitutions.items()}
            try:
                result = self.expression.subs(subs_dict)

                # Try to evaluate to float
                if hasattr(result, "evalf"):
                    result = result.evalf()

                # Convert to Python type
                if hasattr(result, "is_Boolean") and result.is_Boolean:
                    return bool(result)
                elif hasattr(result, "is_Number") and result.is_Number:
                    return float(result)
                else:
                    return result
            except Exception as e:
                logger.debug("Sympy substitution failed: %s", e)
                return self.expression

        elif isinstance(self.expression, SimpleExpression):
            # Fallback substitution
            return self.expression.substitute(substitutions)

        else:
            logger.error("Unknown expression type: %s", type(self.expression))
            return self.expression

    def __str__(self) -> str:
        return str(self.expression)

    def __repr__(self) -> str:
        return f"SymbolicExpression({self.expression})"


class SimpleExpression:
    """
    Enhanced fallback expression handler for when sympy is not available

    Uses AST-based evaluation with comprehensive safety checks
    """

    def __init__(
        self, expr_str: str, variables: Set[str], ast_tree: Optional[ast.AST] = None
    ):
        """
        Initialize simple expression

        Args:
            expr_str: Expression string
            variables: Set of variable names in expression
            ast_tree: Pre-parsed AST tree (optional)
        """
        self.expr_str = expr_str
        self.variables = variables
        self.ast_tree = ast_tree if ast_tree else ast.parse(expr_str, mode="eval")

    def substitute(self, substitutions: Dict[str, float]) -> Union[float, bool, Any]:
        """
        Substitute values into expression using safe AST evaluation

        Args:
            substitutions: Dictionary mapping variables to values

        Returns:
            Result of expression evaluation
        """

        try:
            # Create safe evaluator
            evaluator = SafeASTEvaluator(substitutions)
            result = evaluator.eval(self.ast_tree)
            return result

        except Exception as e:
            logger.error(f"Expression evaluation failed for '{self.expr_str}': {e}")
            raise

    def __str__(self) -> str:
        return self.expr_str

    def __repr__(self) -> str:
        return f"SimpleExpression({self.expr_str})"


class SafeASTEvaluator:
    """Safe AST evaluator with operation whitelist"""

    def __init__(self, context: Dict[str, Any]):
        """
        Initialize evaluator

        Args:
            context: Dictionary of variable values
        """
        self.context = context
        self.operation_count = 0

    def eval(self, node: ast.AST) -> Any:
        """
        Evaluate AST node safely

        Args:
            node: AST node to evaluate

        Returns:
            Evaluation result
        """

        self.operation_count += 1
        if self.operation_count > SymbolicExpressionSystem.MAX_OPERATION_COUNT:
            raise ExpressionComplexityError("Too many operations in expression")

        if isinstance(node, ast.Expression):
            return self.eval(node.body)

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n

        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name in self.context:
                return self.context[var_name]
            else:
                raise ValueError(f"Undefined variable: {var_name}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SymbolicExpressionSystem.SAFE_OPERATIONS:
                raise ExpressionSafetyError(f"Unsafe operation: {op_type.__name__}")

            left = self.eval(node.left)
            right = self.eval(node.right)
            op_func = SymbolicExpressionSystem.SAFE_OPERATIONS[op_type]

            return op_func(left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SymbolicExpressionSystem.SAFE_OPERATIONS:
                raise ExpressionSafetyError(f"Unsafe operation: {op_type.__name__}")

            operand = self.eval(node.operand)
            op_func = SymbolicExpressionSystem.SAFE_OPERATIONS[op_type]

            return op_func(operand)

        elif isinstance(node, ast.Compare):
            left = self.eval(node.left)

            for op, comparator in zip(node.ops, node.comparators):
                right = self.eval(comparator)
                op_type = type(op)

                if op_type not in SymbolicExpressionSystem.SAFE_COMPARISONS:
                    raise ExpressionSafetyError(
                        f"Unsafe comparison: {op_type.__name__}"
                    )

                op_func = SymbolicExpressionSystem.SAFE_COMPARISONS[op_type]
                if not op_func(left, right):
                    return False

                left = right

            return True

        elif isinstance(node, ast.Call):
            # Function call
            if not isinstance(node.func, ast.Name):
                raise ExpressionSafetyError("Complex function calls not allowed")

            func_name = node.func.id
            if func_name not in SymbolicExpressionSystem.SAFE_FUNCTIONS:
                raise ExpressionSafetyError(f"Unsafe function: {func_name}")

            func = SymbolicExpressionSystem.SAFE_FUNCTIONS[func_name]
            args = [self.eval(arg) for arg in node.args]

            return func(*args)

        elif isinstance(node, ast.BoolOp):
            # Boolean operation (and/or)
            if isinstance(node.op, ast.And):
                return all(self.eval(value) for value in node.values)
            elif isinstance(node.op, ast.Or):
                return any(self.eval(value) for value in node.values)
            else:
                raise ExpressionSafetyError(
                    f"Unsafe boolean operation: {type(node.op).__name__}"
                )

        elif isinstance(node, ast.IfExp):
            # Ternary expression
            test = self.eval(node.test)
            if test:
                return self.eval(node.body)
            else:
                return self.eval(node.orelse)

        else:
            raise ExpressionSafetyError(f"Unsafe node type: {type(node).__name__}")


class SimpleSymbol:
    """Simple symbol for when sympy is not available"""

    def __init__(self, name: str):
        self.name = str(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Symbol({self.name})"

    def __eq__(self, other) -> bool:
        if isinstance(other, SimpleSymbol):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)

    def __ge__(self, other):
        """Create inequality expression"""
        return SimpleExpression(f"{self.name} >= {other}", {self.name}, None)

    def __le__(self, other):
        """Create inequality expression"""
        return SimpleExpression(f"{self.name} <= {other}", {self.name}, None)

    def __gt__(self, other):
        """Create inequality expression"""
        return SimpleExpression(f"{self.name} > {other}", {self.name}, None)

    def __lt__(self, other):
        """Create inequality expression"""
        return SimpleExpression(f"{self.name} < {other}", {self.name}, None)


def simple_spearmanr(x, y):
    """Simple Spearman rank correlation for when scipy is not available"""
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y) or len(x) < 2:
        return 0.0, 1.0

    # Rank the data
    def rankdata(a):
        """Simple ranking function"""
        n = len(a)
        ranks = np.empty(n)
        sorted_indices = np.argsort(a)

        # Handle ties by averaging ranks
        i = 0
        while i < n:
            j = i
            # Find all equal values
            while j < n - 1 and a[sorted_indices[j]] == a[sorted_indices[j + 1]]:
                j += 1

            # Average rank for tied values
            avg_rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1

        return ranks

    # Get ranks
    rank_x = rankdata(x)
    rank_y = rankdata(y)

    # Calculate Pearson correlation on ranks
    n = len(rank_x)
    mean_x = np.mean(rank_x)
    mean_y = np.mean(rank_y)

    numerator = np.sum((rank_x - mean_x) * (rank_y - mean_y))
    denominator = np.sqrt(
        np.sum((rank_x - mean_x) ** 2) * np.sum((rank_y - mean_y) ** 2)
    )

    if denominator == 0:
        return 0.0, 1.0

    r = numerator / denominator

    # Simplified p-value calculation
    t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
    p_value = 2 * (1 - min(0.999, 0.5 + 0.5 * abs(t_stat) / np.sqrt(n)))

    return r, p_value


# Use fallbacks if libraries not available
if not SYMPY_AVAILABLE:

    class MockSympy:
        Symbol = SimpleSymbol

        @staticmethod
        def sympify(expr):
            if isinstance(expr, str):
                return SimpleExpression(expr, set(), None)
            return expr

    sp = MockSympy()

if not SCIPY_AVAILABLE:

    class MockStats:
        @staticmethod
        def spearmanr(x, y):
            return simple_spearmanr(x, y)

    stats = MockStats()

    def minimize(fun, x0, method=None, bounds=None, options=None):
        """Simple minimization using gradient descent"""
        x = np.asarray(x0)
        learning_rate = 0.01
        max_iter = 100 if options is None else options.get("maxiter", 100)

        best_x = x.copy()
        best_val = fun(x)

        for _ in range(max_iter):
            # Numerical gradient
            eps = 1e-8
            grad = np.zeros_like(x)
            f0 = fun(x)

            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                grad[i] = (fun(x_plus) - f0) / eps

            # Update with gradient descent
            x = x - learning_rate * grad

            # Apply bounds if provided
            if bounds is not None:
                for i, (low, high) in enumerate(bounds):
                    if low is not None:
                        x[i] = max(x[i], low)
                    if high is not None:
                        x[i] = min(x[i], high)

            # Track best solution
            current_val = fun(x)
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()

        # Return optimization result object
        class OptimizeResult:
            def __init__(self, x, fun_val):
                self.x = x
                self.fun = fun_val
                self.success = True

        return OptimizeResult(best_x, best_val)


class InvariantType(Enum):
    """Types of invariants"""

    CONSERVATION = "conservation"  # Sum or product remains constant
    CONSTRAINT = "constraint"  # Variable bounds or relationships
    SYMMETRY = "symmetry"  # Transformation invariance
    PERIODIC = "periodic"  # Periodic behavior
    MONOTONIC = "monotonic"  # Monotonic relationships
    LINEAR = "linear"  # Linear relationships
    NONLINEAR = "nonlinear"  # Nonlinear relationships


@dataclass
class Invariant:
    """Invariant relationship or constraint"""

    type: InvariantType
    expression: str
    variables: List[str]
    confidence: float
    violation_count: int = 0
    domain: str = "global"
    parameters: Dict[str, float] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    validation_count: int = 0
    symbolic_expr: Optional[SymbolicExpression] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "type": self.type.value,
            "expression": self.expression,
            "variables": self.variables,
            "confidence": self.confidence,
            "violation_count": self.violation_count,
            "domain": self.domain,
            "parameters": self.parameters,
            "discovered_at": self.discovered_at,
            "last_validated": self.last_validated,
            "validation_count": self.validation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Invariant":
        """Create from dictionary"""
        data["type"] = InvariantType(data["type"])
        return cls(**data)


class InvariantEvaluator:
    """Evaluates invariants against states - SEPARATED CONCERN"""

    def __init__(self, symbolic_system: SymbolicExpressionSystem):
        """
        Initialize evaluator

        Args:
            symbolic_system: Symbolic expression system for evaluations
        """
        self.symbolic_system = symbolic_system

        self.evaluation_methods = {
            InvariantType.CONSERVATION: self._evaluate_conservation,
            InvariantType.CONSTRAINT: self._evaluate_constraint,
            InvariantType.SYMMETRY: self._evaluate_symmetry,
            InvariantType.PERIODIC: self._evaluate_periodic,
            InvariantType.MONOTONIC: self._evaluate_monotonic,
            InvariantType.LINEAR: self._evaluate_linear,
            InvariantType.NONLINEAR: self._evaluate_symbolic,
        }

    def evaluate(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate if invariant holds for given state"""

        try:
            # SELECT: Choose evaluation method
            evaluator = self.evaluation_methods.get(
                invariant.type, self._evaluate_symbolic
            )

            # APPLY: Evaluate invariant
            return evaluator(invariant, state)

        except Exception as e:
            logger.debug("Failed to evaluate invariant: %s", e)
            return True  # Assume holds if can't evaluate

    def _evaluate_conservation(
        self, invariant: Invariant, state: Dict[str, Any]
    ) -> bool:
        """Evaluate conservation law"""
        total = invariant.parameters.get("conserved_value", 0)
        current_sum = sum(state.get(var, 0) for var in invariant.variables)

        # Allow small tolerance
        tolerance = invariant.parameters.get("tolerance", 0.01)
        return abs(current_sum - total) < tolerance * abs(total + 1e-10)

    def _evaluate_constraint(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate constraint"""
        try:
            # Use symbolic evaluation
            if invariant.symbolic_expr:
                result = invariant.symbolic_expr.substitute(state)

                # Handle boolean and numeric results
                if isinstance(result, bool):
                    return result
                elif isinstance(result, (int, float)):
                    return abs(result) < 0.01  # Near zero means constraint satisfied
                else:
                    logger.debug("Unexpected constraint result type: %s", type(result))
                    return True

            return True
        except Exception as e:
            logger.debug(f"Constraint evaluation failed: {e}")
            return True

    def _evaluate_symmetry(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate symmetry invariant"""
        transform_type = invariant.parameters.get("transform", "reflection")

        if transform_type == "reflection":
            # Check reflection symmetry
            for i in range(len(invariant.variables) // 2):
                var1 = invariant.variables[i]
                var2 = invariant.variables[-(i + 1)]
                if abs(state.get(var1, 0) - state.get(var2, 0)) > 0.01:
                    return False
            return True

        return True

    def _evaluate_periodic(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate periodic invariant"""
        # This would need historical data to properly evaluate
        # For now, just check if value is within expected range
        return True

    def _evaluate_monotonic(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate monotonic relationship"""
        # Would need historical data to verify monotonicity
        return True

    def _evaluate_linear(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate linear relationship"""
        if len(invariant.variables) >= 2:
            x = state.get(invariant.variables[0], 0)
            y = state.get(invariant.variables[1], 0)
            a = invariant.parameters.get("slope", 1.0)
            b = invariant.parameters.get("intercept", 0.0)

            expected_y = a * x + b
            tolerance = invariant.parameters.get("tolerance", 0.1)

            return abs(y - expected_y) < tolerance

        return True

    def _evaluate_symbolic(self, invariant: Invariant, state: Dict[str, Any]) -> bool:
        """Evaluate symbolic expression"""
        if not invariant.symbolic_expr:
            return True

        try:
            result = invariant.symbolic_expr.substitute(state)

            if isinstance(result, bool):
                return result
            elif isinstance(result, (int, float)):
                return abs(result) < 0.01
            else:
                return True
        except Exception as e:
            logger.debug(f"Symbolic evaluation failed: {e}")
            return True


class InvariantValidator:
    """Validates invariants against observations - SEPARATED CONCERN"""

    def __init__(self, evaluator: InvariantEvaluator):
        self.evaluator = evaluator
        self.validation_history = deque(maxlen=1000)

    def validate(
        self, invariant: Invariant, observations: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate an invariant against observations

        Args:
            invariant: The invariant to validate
            observations: List of observations to validate against

        Returns:
            True if invariant holds for observations
        """

        if not observations:
            return True

        # EXAMINE: Count violations
        violations = 0
        total = 0

        for obs in observations:
            # Check if all required variables are present
            if not all(var in obs for var in invariant.variables):
                continue

            total += 1

            # Evaluate invariant
            if not self.evaluator.evaluate(invariant, obs):
                violations += 1

        if total == 0:
            return True  # No applicable observations

        # SELECT: Calculate success rate
        success_rate = 1.0 - (violations / total)

        # APPLY: Update invariant confidence
        alpha = 0.1  # Learning rate
        invariant.confidence = (1 - alpha) * invariant.confidence + alpha * success_rate
        invariant.validation_count += 1
        invariant.last_validated = time.time()

        # REMEMBER: Track validation
        self.validation_history.append(
            {
                "invariant_id": id(invariant),
                "success_rate": success_rate,
                "timestamp": time.time(),
            }
        )

        return success_rate > 0.9  # Allow 10% noise


class InvariantIndexer:
    """Manages invariant indexing and lookup - SEPARATED CONCERN"""

    def __init__(self):
        self.invariants = {}  # id -> Invariant
        self.domain_index = defaultdict(list)  # domain -> [invariant_ids]
        self.variable_index = defaultdict(list)  # variable -> [invariant_ids]
        self.next_id = 1
        self.lock = threading.Lock()

    def add(self, invariant: Invariant) -> str:
        """Add invariant to index"""

        with self.lock:
            # Assign ID
            invariant_id = f"inv_{self.next_id}"
            self.next_id += 1

            # Store invariant
            self.invariants[invariant_id] = invariant

            # Index by domain
            self.domain_index[invariant.domain].append(invariant_id)

            # Index by variables
            for var in invariant.variables:
                self.variable_index[var].append(invariant_id)

        return invariant_id

    def remove(self, invariant_id: str):
        """Remove invariant from index"""

        with self.lock:
            if invariant_id not in self.invariants:
                return

            invariant = self.invariants[invariant_id]

            # Remove from domain index
            domain_invs = self.domain_index[invariant.domain]
            if invariant_id in domain_invs:
                domain_invs.remove(invariant_id)

            # Remove from variable index
            for var in invariant.variables:
                var_invs = self.variable_index[var]
                if invariant_id in var_invs:
                    var_invs.remove(invariant_id)

            # Remove invariant
            del self.invariants[invariant_id]

    def get_by_domain(self, domain: str) -> List[Tuple[str, Invariant]]:
        """Get invariants for a domain"""

        with self.lock:
            invariant_ids = self.domain_index.get(domain, [])
            invariant_ids.extend(self.domain_index.get("global", []))

            results = []
            for inv_id in set(invariant_ids):
                if inv_id in self.invariants:
                    results.append((inv_id, self.invariants[inv_id]))

            return results

    def get_by_variables(self, variables: List[str]) -> List[Tuple[str, Invariant]]:
        """Get invariants for variables"""

        with self.lock:
            relevant_ids = set()
            for var in variables:
                relevant_ids.update(self.variable_index.get(var, []))

            results = []
            for inv_id in relevant_ids:
                if inv_id in self.invariants:
                    results.append((inv_id, self.invariants[inv_id]))

            return results

    def get_all(self) -> Dict[str, Invariant]:
        """Get all invariants"""

        with self.lock:
            return self.invariants.copy()


class InvariantRegistry:
    """Manages discovered invariants - REFACTORED WITH SAFETY AND FIXED API"""

    def __init__(
        self,
        violation_threshold: int = 5,
        confidence_threshold: float = 0.7,
        safety_config: Optional[Dict[str, Any]] = None,
        safety_validator=None,
    ):
        """
        Initialize invariant registry - FIXED: Added safety_validator parameter

        Args:
            violation_threshold: Max violations before removing invariant
            confidence_threshold: Minimum confidence to keep invariant
            safety_config: Optional safety configuration (deprecated, use safety_validator)
            safety_validator: Optional shared safety validator instance (preferred over safety_config)
        """

        # Lazy load safety validator if needed
        _lazy_load_safety_validator()

        # Initialize safety validator - prefer shared instance
        if safety_validator is not None:
            # Use provided shared instance (PREFERRED - prevents duplication)
            self.safety_validator = safety_validator
            logger.info(
                f"{self.__class__.__name__}: Using shared safety validator instance"
            )
        elif SAFETY_VALIDATOR_AVAILABLE:
            # Fallback: try to get singleton, or create new instance
            try:
                from ..safety.safety_validator import initialize_all_safety_components

                self.safety_validator = initialize_all_safety_components(
                    config=safety_config, reuse_existing=True
                )
                logger.info(
                    f"{self.__class__.__name__}: Using singleton safety validator"
                )
            except Exception as e:
                logger.debug(f"Could not get singleton safety validator: {e}")
                # Last resort: create new instance
                if isinstance(safety_config, dict) and safety_config:
                    self.safety_validator = EnhancedSafetyValidator(
                        SafetyConfig.from_dict(safety_config)
                    )
                else:
                    self.safety_validator = EnhancedSafetyValidator()
                logger.warning(
                    f"{self.__class__.__name__}: Created new safety validator instance (may cause duplication)"
                )
        else:
            self.safety_validator = None
            logger.warning(
                f"{self.__class__.__name__}: Safety validator not available - operating without safety checks"
            )

        # Initialize symbolic system
        self.symbolic_system = SymbolicExpressionSystem(self.safety_validator)

        # Components
        self.indexer = InvariantIndexer()
        self.evaluator = InvariantEvaluator(self.symbolic_system)
        self.validator = InvariantValidator(self.evaluator)

        # Configuration
        self.violation_threshold = violation_threshold
        self.confidence_threshold = confidence_threshold

        # Safety tracking
        self.safety_blocks = defaultdict(int)
        self.safety_corrections = defaultdict(int)

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            "InvariantRegistry initialized (refactored with safety and full symbolic system)"
        )

    def register(self, invariant: Invariant) -> str:
        """
        Register a new invariant - WITH SAFETY VALIDATION

        Args:
            invariant: The invariant to register

        Returns:
            ID of registered invariant
        """

        with self.lock:
            # SAFETY: Validate invariant
            if self.safety_validator:
                inv_check = self._validate_invariant_safety(invariant)
                if not inv_check["safe"]:
                    logger.warning("Blocked unsafe invariant: %s", inv_check["reason"])
                    self.safety_blocks["invariant"] += 1
                    return ""

            # EXAMINE: Parse symbolic expression if not already done
            if not invariant.symbolic_expr and invariant.expression:
                try:
                    invariant.symbolic_expr = self.symbolic_system.parse(
                        invariant.expression, set(invariant.variables)
                    )
                except (ExpressionSafetyError, ExpressionComplexityError) as e:
                    logger.warning("Failed to parse invariant expression: %s", e)
                    self.safety_blocks["parse_error"] += 1
                    return ""

            # EXAMINE: Check for duplicates
            duplicate_id = self._find_duplicate(invariant)
            if duplicate_id:
                logger.debug("Duplicate invariant detected, skipping registration")
                return duplicate_id

            # SELECT & APPLY: Add to index
            invariant_id = self.indexer.add(invariant)

            # REMEMBER: Log registration
            logger.info(
                "Registered invariant %s: %s", invariant_id, invariant.expression
            )

        return invariant_id

    def validate_invariant(
        self, invariant: Invariant, observations: List[Dict[str, Any]]
    ) -> bool:
        """Validate an invariant against observations - DELEGATED"""

        return self.validator.validate(invariant, observations)

    def check_invariant_violations(
        self, state: Union[Dict[str, Any], float, int]
    ) -> List[Tuple[str, Invariant]]:
        """
        Check for invariant violations in a state - FIXED: Accepts dict or float

        Args:
            state: Current state to check (dict or numeric value)

        Returns:
            List of (invariant_id, invariant) tuples that are violated
        """

        # FIXED: Normalize input to dict
        if isinstance(state, (int, float)):
            state_dict = {"value": float(state)}
        elif isinstance(state, dict):
            state_dict = state
        else:
            logger.warning(
                f"Unexpected state type: {type(state).__name__}, converting to dict"
            )
            try:
                state_dict = {"value": float(state)}
            except (TypeError, ValueError):
                logger.error(
                    f"Cannot convert state of type {type(state).__name__} to dict"
                )
                return []

        # SAFETY: Validate state first
        if self.safety_validator:
            try:
                if hasattr(self.safety_validator, "validate_state"):
                    state_check = self.safety_validator.validate_state(state_dict)
                    if not state_check.get("safe", True):
                        logger.warning(
                            "Unsafe state provided: %s",
                            state_check.get("reason", "unknown"),
                        )
                        self.safety_blocks["state"] += 1
                        return []
            except Exception as e:
                logger.error("State validation error: %s", e)

        violations = []

        with self.lock:
            # EXAMINE: Get all invariants
            all_invariants = self.indexer.get_all()

            for inv_id, invariant in all_invariants.items():
                # Check if all required variables are in state
                if not all(var in state_dict for var in invariant.variables):
                    continue

                # SELECT & APPLY: Evaluate invariant
                if not self.evaluator.evaluate(invariant, state_dict):
                    invariant.violation_count += 1
                    violations.append((inv_id, invariant))

                    # REMEMBER: Remove if too many violations
                    if invariant.violation_count >= self.violation_threshold:
                        logger.warning(
                            "Removing invariant %s due to excessive violations", inv_id
                        )
                        self.indexer.remove(inv_id)

        return violations

    def get_invariants_for_domain(self, domain: str) -> List[Invariant]:
        """Get all invariants for a specific domain - REFACTORED"""

        with self.lock:
            # EXAMINE: Get invariants from index
            domain_invariants = self.indexer.get_by_domain(domain)

            # SELECT: Filter by confidence
            filtered = [
                inv
                for _, inv in domain_invariants
                if inv.confidence >= self.confidence_threshold
            ]

            return filtered

    def get_invariants_for_variables(self, variables: List[str]) -> List[Invariant]:
        """Get invariants involving specific variables - REFACTORED"""

        with self.lock:
            # EXAMINE: Get invariants from index
            variable_invariants = self.indexer.get_by_variables(variables)

            # SELECT: Filter by confidence
            filtered = [
                inv
                for _, inv in variable_invariants
                if inv.confidence >= self.confidence_threshold
            ]

            return filtered

    def get_invariant_types(self) -> Dict[str, int]:
        """Get count of each invariant type"""

        with self.lock:
            type_counts = defaultdict(int)

            for invariant in self.indexer.invariants.values():
                type_counts[invariant.type.value] += 1

            return dict(type_counts)

    def prune_weak_invariants(self):
        """Remove invariants with low confidence - REFACTORED"""

        with self.lock:
            # EXAMINE: Find weak invariants
            to_remove = []

            for inv_id, invariant in self.indexer.get_all().items():
                if invariant.confidence < self.confidence_threshold:
                    to_remove.append(inv_id)
                elif invariant.violation_count >= self.violation_threshold:
                    to_remove.append(inv_id)

            # APPLY: Remove weak invariants
            for inv_id in to_remove:
                self.indexer.remove(inv_id)

            # REMEMBER: Log pruning
            if to_remove:
                logger.info("Pruned %d weak invariants", len(to_remove))

    def _validate_invariant_safety(self, invariant: Invariant) -> Dict[str, Any]:
        """Validate invariant for safety"""

        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check expression length
        if len(invariant.expression) > 10000:
            violations.append("Expression too long")

        # Check variable count
        if len(invariant.variables) > 100:
            violations.append("Too many variables")

        # Check confidence bounds
        if not (0 <= invariant.confidence <= 1):
            violations.append(f"Invalid confidence: {invariant.confidence}")

        # Check parameters for non-finite values
        for key, value in invariant.parameters.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                violations.append(f"Non-finite parameter {key}: {value}")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _find_duplicate(self, invariant: Invariant) -> Optional[str]:
        """Find duplicate invariant ID if exists"""

        for inv_id, existing in self.indexer.get_all().items():
            if self._is_duplicate(invariant, existing):
                return inv_id

        return None

    def _is_duplicate(self, inv1: Invariant, inv2: Invariant) -> bool:
        """Check if two invariants are duplicates"""

        if inv1.type != inv2.type:
            return False

        if set(inv1.variables) != set(inv2.variables):
            return False

        # Check if expressions are equivalent
        if inv1.expression == inv2.expression:
            return True

        # Use symbolic system to check equivalence
        if inv1.symbolic_expr and inv2.symbolic_expr:
            return self.symbolic_system.check_equality(
                inv1.symbolic_expr, inv2.symbolic_expr
            )

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get invariant registry statistics"""

        stats = {
            "total_invariants": len(self.indexer.invariants),
            "invariant_types": self.get_invariant_types(),
            "validation_history_size": len(self.validator.validation_history),
            "symbolic_system": self.symbolic_system.get_statistics(),
        }

        # Add safety statistics
        if self.safety_validator:
            stats["safety"] = {
                "enabled": True,
                "blocks": dict(self.safety_blocks),
                "corrections": dict(self.safety_corrections),
                "total_blocks": sum(self.safety_blocks.values()),
                "total_corrections": sum(self.safety_corrections.values()),
            }
        else:
            stats["safety"] = {"enabled": False}

        return stats


class ConservationLawDetector:
    """Detects conservation laws - SEPARATED CONCERN"""

    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples

    def detect(self, variables: Dict[str, List[float]]) -> List[Invariant]:
        """Find conservation laws (sums that remain constant)"""

        invariants = []
        var_names = list(variables.keys())

        # Check single variables
        invariants.extend(self._detect_constant_variables(variables))

        # Check pairs
        invariants.extend(self._detect_conserved_pairs(variables, var_names))

        return invariants

    def _detect_constant_variables(
        self, variables: Dict[str, List[float]]
    ) -> List[Invariant]:
        """Detect variables that remain constant"""

        invariants = []

        for var, values in variables.items():
            if len(values) >= self.min_samples:
                std = np.std(values)
                mean = np.mean(values)

                # Check if approximately constant
                if mean != 0 and std / abs(mean) < 0.05:
                    inv = Invariant(
                        type=InvariantType.CONSERVATION,
                        expression=f"{var} = {mean:.3f}",
                        variables=[var],
                        confidence=0.95 * (1 - std / (abs(mean) + 1e-10)),
                        parameters={"conserved_value": mean, "tolerance": 0.05},
                    )
                    invariants.append(inv)

        return invariants

    def _detect_conserved_pairs(
        self, variables: Dict[str, List[float]], var_names: List[str]
    ) -> List[Invariant]:
        """Detect pairs of variables with conserved sum"""

        invariants = []

        for i, var1 in enumerate(var_names):
            for var2 in var_names[i + 1 :]:
                if var1 in variables and var2 in variables:
                    vals1 = np.array(variables[var1])
                    vals2 = np.array(variables[var2])

                    if len(vals1) == len(vals2) and len(vals1) >= self.min_samples:
                        # Check if sum is constant
                        sums = vals1 + vals2
                        std = np.std(sums)
                        mean = np.mean(sums)

                        if mean != 0 and std / abs(mean) < 0.05:
                            inv = Invariant(
                                type=InvariantType.CONSERVATION,
                                expression=f"{var1} + {var2} = {mean:.3f}",
                                variables=[var1, var2],
                                confidence=0.9 * (1 - std / (abs(mean) + 1e-10)),
                                parameters={"conserved_value": mean, "tolerance": 0.05},
                            )
                            invariants.append(inv)

        return invariants


class LinearRelationshipDetector:
    """Detects linear relationships - SEPARATED CONCERN"""

    def __init__(self, min_samples: int = 20, min_correlation: float = 0.95):
        self.min_samples = min_samples
        self.min_correlation = min_correlation

    def detect(self, variables: Dict[str, List[float]]) -> List[Invariant]:
        """Detect linear relationships between variables"""

        invariants = []
        var_names = list(variables.keys())

        for i, var1 in enumerate(var_names):
            for var2 in var_names[i + 1 :]:
                invariant = self._check_linear_relationship(var1, var2, variables)
                if invariant:
                    invariants.append(invariant)

        return invariants

    def _check_linear_relationship(
        self, var1: str, var2: str, variables: Dict[str, List[float]]
    ) -> Optional[Invariant]:
        """Check for linear relationship between two variables"""

        if var1 not in variables or var2 not in variables:
            return None

        vals1 = np.array(variables[var1])
        vals2 = np.array(variables[var2])

        if len(vals1) != len(vals2) or len(vals1) < self.min_samples:
            return None

        # Check for linear correlation
        correlation = np.corrcoef(vals1, vals2)[0, 1]

        if abs(correlation) > self.min_correlation:
            # Fit linear model
            slope, intercept = np.polyfit(vals1, vals2, 1)

            return Invariant(
                type=InvariantType.LINEAR,
                expression=f"{var2} = {slope:.3f} * {var1} + {intercept:.3f}",
                variables=[var1, var2],
                confidence=abs(correlation),
                parameters={
                    "slope": slope,
                    "intercept": intercept,
                    "correlation": correlation,
                    "tolerance": 0.1,
                },
            )

        return None


class InvariantDetector:
    """Detects invariant relationships in data - REFACTORED WITH FULL SYMBOLIC SYSTEM"""

    def __init__(
        self,
        min_confidence: float = 0.8,
        min_samples: int = 20,
        safety_config: Optional[Dict[str, Any]] = None,
        safety_validator=None,
    ):
        """
        Initialize invariant detector - FIXED: Added safety_validator parameter

        Args:
            min_confidence: Minimum confidence for detected invariants
            min_samples: Minimum samples needed for detection
            safety_config: Optional safety configuration (deprecated, use safety_validator)
            safety_validator: Optional shared safety validator instance (preferred over safety_config)
        """
        self.min_confidence = min_confidence
        self.min_samples = min_samples

        # Lazy load safety validator if needed
        _lazy_load_safety_validator()

        # Initialize safety validator - prefer shared instance
        if safety_validator is not None:
            # Use provided shared instance (PREFERRED - prevents duplication)
            self.safety_validator = safety_validator
            logger.info(
                f"{self.__class__.__name__}: Using shared safety validator instance"
            )
        elif SAFETY_VALIDATOR_AVAILABLE:
            # Fallback: try to get singleton, or create new instance
            try:
                from ..safety.safety_validator import initialize_all_safety_components

                self.safety_validator = initialize_all_safety_components(
                    config=safety_config, reuse_existing=True
                )
                logger.info(
                    f"{self.__class__.__name__}: Using singleton safety validator"
                )
            except Exception as e:
                logger.debug(f"Could not get singleton safety validator: {e}")
                # Last resort: create new instance
                if isinstance(safety_config, dict) and safety_config:
                    self.safety_validator = EnhancedSafetyValidator(
                        SafetyConfig.from_dict(safety_config)
                    )
                else:
                    self.safety_validator = EnhancedSafetyValidator()
                logger.warning(
                    f"{self.__class__.__name__}: Created new safety validator instance (may cause duplication)"
                )
        else:
            self.safety_validator = None
            logger.warning(
                f"{self.__class__.__name__}: Safety validator not available - operating without safety checks"
            )

        # Initialize symbolic system
        self.symbolic_system = SymbolicExpressionSystem(self.safety_validator)

        # Specialized detectors
        self.conservation_detector = ConservationLawDetector(min_samples)
        self.linear_detector = LinearRelationshipDetector(min_samples)

        # Safety tracking
        self.safety_blocks = defaultdict(int)
        self.safety_corrections = defaultdict(int)

        # Store recent observations for check() method
        self._recent_observations = deque(maxlen=1000)

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            "InvariantDetector initialized (refactored with full symbolic system)"
        )

    def check(
        self, observations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Router-compatible check method - FIXED: Added for router compatibility

        Args:
            observations: Optional list of observations to check

        Returns:
            Dict with check status
        """

        # Use provided observations or recent observations
        if observations is None:
            observations = list(self._recent_observations)
        else:
            # Store observations for future use
            self._recent_observations.extend(observations)

        if not observations:
            return {
                "status": "no_observations",
                "message": "No observations available for invariant detection",
                "invariants_detected": 0,
            }

        # Detect invariants
        try:
            detected_invariants = self.detect_invariants(observations)

            return {
                "status": "success",
                "invariants_detected": len(detected_invariants),
                "observations_processed": len(observations),
                "min_confidence": self.min_confidence,
            }
        except Exception as e:
            logger.error("Invariant detection failed: %s", e)
            return {"status": "error", "message": str(e), "invariants_detected": 0}

    def detect_invariants(self, observations: List[Dict[str, Any]]) -> List[Invariant]:
        """
        Detect invariants from observations - WITH SAFETY VALIDATION AND FULL SYMBOLIC SYSTEM

        Args:
            observations: List of observations

        Returns:
            List of detected invariants
        """

        if len(observations) < self.min_samples:
            return []

        with self.lock:
            # Store observations
            self._recent_observations.extend(observations)

            # SAFETY: Filter observations
            safe_observations = observations
            if self.safety_validator:
                safe_observations = []
                for obs in observations:
                    try:
                        if hasattr(self.safety_validator, "analyze_observation_safety"):
                            obs_check = (
                                self.safety_validator.analyze_observation_safety(obs)
                            )
                            if obs_check.get("safe", True):
                                safe_observations.append(obs)
                            else:
                                self.safety_blocks["observation"] += 1
                    except Exception as e:
                        logger.error("Observation safety check error: %s", e)
                        safe_observations.append(obs)  # Include if check fails

                if len(safe_observations) < self.min_samples:
                    logger.warning(
                        "Insufficient safe observations for invariant detection"
                    )
                    return []

            # EXAMINE: Extract variables
            variables = self._extract_variables(safe_observations)

            # SAFETY: Validate extracted variables
            if self.safety_validator:
                variables = self._validate_variables_safety(variables)

            # SELECT: Apply detection methods
            invariants = []

            # Conservation laws
            invariants.extend(self.conservation_detector.detect(variables))

            # Linear relationships
            invariants.extend(self.linear_detector.detect(variables))

            # Constraints
            invariants.extend(self._detect_constraints(safe_observations, variables))

            # Other patterns
            invariants.extend(self._detect_other_patterns(safe_observations, variables))

            # APPLY: Parse symbolic expressions and filter by confidence
            filtered = []
            for inv in invariants:
                if inv.confidence >= self.min_confidence:
                    # Parse symbolic expression if not already done
                    if not inv.symbolic_expr and inv.expression:
                        try:
                            inv.symbolic_expr = self.symbolic_system.parse(
                                inv.expression, set(inv.variables)
                            )
                        except (ExpressionSafetyError, ExpressionComplexityError) as e:
                            logger.debug("Failed to parse invariant expression: %s", e)
                            self.safety_blocks["parse_error"] += 1
                            continue

                    filtered.append(inv)

            # REMEMBER: Log detection
            logger.info(
                "Detected %d invariants from %d observations",
                len(filtered),
                len(safe_observations),
            )

        return filtered

    def find_conservation_laws(
        self, variables: Dict[str, List[float]]
    ) -> List[Invariant]:
        """Find conservation laws - DELEGATED"""

        return self.conservation_detector.detect(variables)

    def find_constraints(self, state_space: List[Dict[str, float]]) -> List[Invariant]:
        """Find constraints (bounds and relationships)"""

        invariants = []

        if len(state_space) < self.min_samples:
            return invariants

        # Extract variable bounds
        variables = self._extract_variables(state_space)

        for var, values in variables.items():
            invariants.extend(self._find_variable_bounds(var, values))

        return invariants

    def test_invariant_hypothesis(
        self, hypothesis: str, data: List[Dict[str, Any]]
    ) -> Optional[Invariant]:
        """
        Test a specific invariant hypothesis using symbolic system

        Args:
            hypothesis: Hypothesis expression string
            data: Data to test against

        Returns:
            Invariant if hypothesis is validated, None otherwise
        """

        try:
            # Parse hypothesis using symbolic system
            expr = self.symbolic_system.parse(hypothesis)
            variables = list(expr.variables)

            # Test on data
            violations = 0
            total = 0

            for obs in data:
                if all(var in obs for var in variables):
                    total += 1

                    # Substitute values
                    result = expr.substitute(obs)

                    # Check if true
                    if isinstance(result, bool):
                        if not result:
                            violations += 1
                    elif isinstance(result, (int, float)):
                        if abs(float(result)) > 0.01:
                            violations += 1

            if total == 0:
                return None

            success_rate = 1.0 - (violations / total)

            if success_rate > 0.9:
                return Invariant(
                    type=InvariantType.CONSTRAINT,
                    expression=hypothesis,
                    variables=variables,
                    confidence=success_rate,
                    symbolic_expr=expr,
                )

        except (ExpressionSafetyError, ExpressionComplexityError) as e:
            logger.debug("Failed to test hypothesis %s: %s", hypothesis, e)
        except Exception as e:
            logger.debug("Failed to test hypothesis %s: %s", hypothesis, e)

        return None

    def _validate_variables_safety(
        self, variables: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Validate and filter variables for safety"""

        safe_variables = {}

        for var, values in variables.items():
            # Filter non-finite values
            safe_values = [v for v in values if np.isfinite(v)]

            if len(safe_values) != len(values):
                self.safety_corrections["non_finite_values"] += len(values) - len(
                    safe_values
                )

            # Check for extreme values
            filtered_values = []
            for v in safe_values:
                if abs(v) > 1e6:
                    self.safety_corrections["extreme_values"] += 1
                    filtered_values.append(np.clip(v, -1e6, 1e6))
                else:
                    filtered_values.append(v)

            if len(filtered_values) >= self.min_samples:
                safe_variables[var] = filtered_values

        return safe_variables

    def _extract_variables(
        self, observations: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Extract numeric variables from observations"""

        variables = defaultdict(list)

        for obs in observations:
            for key, value in obs.items():
                if isinstance(value, (int, float)):
                    variables[key].append(float(value))

        return dict(variables)

    def _detect_constraints(
        self, observations: List[Dict[str, Any]], variables: Dict[str, List[float]]
    ) -> List[Invariant]:
        """Detect constraint invariants"""

        return self.find_constraints(observations)

    def _detect_other_patterns(
        self, observations: List[Dict[str, Any]], variables: Dict[str, List[float]]
    ) -> List[Invariant]:
        """Detect other patterns (symmetries, periodicity, etc.)"""

        invariants = []

        # Symmetries
        invariants.extend(self._detect_symmetries(variables))

        # Periodic patterns
        invariants.extend(self._detect_periodic_patterns(variables))

        # Monotonic relationships
        invariants.extend(self._detect_monotonic_relationships(variables))

        return invariants

    def _find_variable_bounds(self, var: str, values: List[float]) -> List[Invariant]:
        """Find bounds for a variable using symbolic system"""

        invariants = []

        if len(values) < self.min_samples:
            return invariants

        min_val = np.min(values)
        max_val = np.max(values)

        # Check for strict bounds
        if np.all(np.array(values) >= min_val * 0.99):
            expr_str = f"{var} >= {min_val:.3f}"
            try:
                symbolic_expr = self.symbolic_system.parse(expr_str, {var})

                inv = Invariant(
                    type=InvariantType.CONSTRAINT,
                    expression=expr_str,
                    variables=[var],
                    confidence=0.95,
                    parameters={"bound_type": "lower", "bound_value": min_val},
                    symbolic_expr=symbolic_expr,
                )
                invariants.append(inv)
            except (ExpressionSafetyError, ExpressionComplexityError) as e:
                logger.debug("Failed to create bound constraint: %s", e)

        if np.all(np.array(values) <= max_val * 1.01):
            expr_str = f"{var} <= {max_val:.3f}"
            try:
                symbolic_expr = self.symbolic_system.parse(expr_str, {var})

                inv = Invariant(
                    type=InvariantType.CONSTRAINT,
                    expression=expr_str,
                    variables=[var],
                    confidence=0.95,
                    parameters={"bound_type": "upper", "bound_value": max_val},
                    symbolic_expr=symbolic_expr,
                )
                invariants.append(inv)
            except (ExpressionSafetyError, ExpressionComplexityError) as e:
                logger.debug("Failed to create bound constraint: %s", e)

        return invariants

    def _detect_symmetries(self, variables: Dict[str, List[float]]) -> List[Invariant]:
        """Detect symmetry relationships"""

        invariants = []
        var_names = list(variables.keys())

        # Check for reflection symmetries
        for i, var1 in enumerate(var_names):
            for var2 in var_names[i + 1 :]:
                if var1 in variables and var2 in variables:
                    vals1 = np.array(variables[var1])
                    vals2 = np.array(variables[var2])

                    if len(vals1) == len(vals2):
                        # Check if values are symmetric
                        diff = np.abs(vals1 - vals2)
                        if np.mean(diff) < 0.01 * (
                            np.mean(np.abs(vals1)) + np.mean(np.abs(vals2))
                        ):
                            inv = Invariant(
                                type=InvariantType.SYMMETRY,
                                expression=f"{var1} ≈ {var2}",
                                variables=[var1, var2],
                                confidence=0.9,
                                parameters={"transform": "reflection"},
                            )
                            invariants.append(inv)

        return invariants

    def _detect_periodic_patterns(
        self, variables: Dict[str, List[float]]
    ) -> List[Invariant]:
        """Detect periodic patterns"""

        invariants = []

        for var, values in variables.items():
            if len(values) >= 2 * self.min_samples:
                # Use FFT to detect periodicity
                fft = np.fft.fft(values)
                power = np.abs(fft) ** 2
                freqs = np.fft.fftfreq(len(values))

                # Find dominant frequency
                idx = np.argmax(power[1 : len(power) // 2]) + 1
                dominant_freq = freqs[idx]

                if dominant_freq > 0 and power[idx] > np.mean(power) * 10:
                    period = 1.0 / dominant_freq

                    inv = Invariant(
                        type=InvariantType.PERIODIC,
                        expression=f"{var} has period {period:.3f}",
                        variables=[var],
                        confidence=0.8,
                        parameters={"period": period, "frequency": dominant_freq},
                    )
                    invariants.append(inv)

        return invariants

    def _detect_monotonic_relationships(
        self, variables: Dict[str, List[float]]
    ) -> List[Invariant]:
        """Detect monotonic relationships"""

        invariants = []
        var_names = list(variables.keys())

        for i, var1 in enumerate(var_names):
            for var2 in var_names[i + 1 :]:
                if var1 in variables and var2 in variables:
                    vals1 = np.array(variables[var1])
                    vals2 = np.array(variables[var2])

                    if len(vals1) == len(vals2) and len(vals1) >= self.min_samples:
                        # Check for monotonic relationship using Spearman correlation
                        spearman_corr, p_value = stats.spearmanr(vals1, vals2)

                        if abs(spearman_corr) > 0.95:
                            direction = (
                                "increasing" if spearman_corr > 0 else "decreasing"
                            )

                            inv = Invariant(
                                type=InvariantType.MONOTONIC,
                                expression=f"{var2} is {direction} with {var1}",
                                variables=[var1, var2],
                                confidence=abs(spearman_corr),
                                parameters={
                                    "direction": direction,
                                    "correlation": spearman_corr,
                                },
                            )
                            invariants.append(inv)

        return invariants

    def get_statistics(self) -> Dict[str, Any]:
        """Get invariant detector statistics"""

        stats = {
            "min_confidence": self.min_confidence,
            "min_samples": self.min_samples,
            "recent_observations_size": len(self._recent_observations),
            "symbolic_system": self.symbolic_system.get_statistics(),
        }

        # Add safety statistics
        if self.safety_validator:
            stats["safety"] = {
                "enabled": True,
                "blocks": dict(self.safety_blocks),
                "corrections": dict(self.safety_corrections),
                "total_blocks": sum(self.safety_blocks.values()),
                "total_corrections": sum(self.safety_corrections.values()),
            }
        else:
            stats["safety"] = {"enabled": False}

        return stats


# =============================================================================
# EXPORTS
# =============================================================================

# Export all classes for external use
__all__ = [
    "InvariantDetector",
    "InvariantRegistry",
    "Invariant",
    "InvariantType",
    "InvariantEvaluator",
    "InvariantValidator",
    "InvariantIndexer",
    "SymbolicExpressionSystem",
    "SymbolicExpression",
    "SimpleExpression",
    "SimpleSymbol",
    "SafeASTEvaluator",
    "ConservationLawDetector",
    "LinearRelationshipDetector",
    "ExpressionComplexityError",
    "ExpressionSafetyError",
]
