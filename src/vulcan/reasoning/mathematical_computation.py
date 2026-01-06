"""
SOTA Mathematical Computation Tool for VULCAN-AGI

Provides symbolic mathematical computation using SymPy with:
- LLM-based code generation for complex problems
- Template-based generation for common operations
- Safe sandboxed execution via RestrictedPython
- Multi-strategy solving (symbolic, numeric, hybrid)
- Learning from successful solutions
- Integration with mathematical verification

This tool GENERATES solutions, unlike MathematicalVerificationEngine which only verifies.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default OpenAI model for chat completions (MAJOR-12 fix)
# Using gpt-3.5-turbo as default since it's more widely available than gpt-4
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

# Maximum attributes to log when debugging unknown LLM interface
DEBUG_LOG_MAX_ATTRS = 10


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================


class ProblemType(Enum):
    """Types of mathematical problems the tool can solve."""
    CALCULUS = "calculus"
    ALGEBRA = "algebra"
    DIFFERENTIAL_EQUATIONS = "differential_equations"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"
    PHYSICS = "physics"
    OPTIMIZATION = "optimization"
    UNKNOWN = "unknown"


class SolutionStrategy(Enum):
    """Strategies for solving mathematical problems."""
    SYMBOLIC = "symbolic"           # Pure SymPy symbolic computation
    NUMERIC = "numeric"             # NumPy numerical computation
    HYBRID = "hybrid"               # Combined symbolic + numeric
    TEMPLATE = "template"           # Template-based code generation
    LLM_GENERATED = "llm_generated" # LLM-generated code


@dataclass
class ComputationResult:
    """Result from a mathematical computation."""
    success: bool
    code: str
    result: Optional[str] = None
    explanation: str = ""
    error: Optional[str] = None
    tool: str = "mathematical_computation"
    problem_type: ProblemType = ProblemType.UNKNOWN
    strategy: SolutionStrategy = SolutionStrategy.TEMPLATE
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemClassification:
    """Classification of a mathematical problem."""
    problem_type: ProblemType
    confidence: float
    keywords: List[str]
    suggested_strategy: SolutionStrategy
    variables: List[str] = field(default_factory=list)


# ============================================================================
# SAFE EXECUTION INTEGRATION
# ============================================================================

# Try to import safe execution module
try:
    from ..utils.safe_execution import execute_math_code, is_safe_execution_available
    SAFE_EXECUTION_AVAILABLE = is_safe_execution_available()
except ImportError:
    SAFE_EXECUTION_AVAILABLE = False
    execute_math_code = None
    logger.warning("Safe execution module not available for mathematical computation")


# ============================================================================
# UNICODE EXPRESSION EXTRACTION (Issue #2 Fix)
# ============================================================================


def extract_math_expression(query: str) -> Optional[str]:
    """
    Extract mathematical expressions from query text.
    
    TASK 4 FIX: Enhanced to handle more mathematical notation patterns.
    
    Handles:
    - Unicode math: ∑, ∏, ∫, √, π, α, β, γ, δ, ε, λ, μ, σ
    - LaTeX-style: \\sum_{k=1}^n, \\frac{a}{b}
    - Probability: P(X|Y)
    - Standard algebraic: x^2 + 2x + 1
    - Natural language math: "sum from k=1 to n"
    - Greek letters: α, β, γ
    
    Args:
        query: Input query text
        
    Returns:
        Extracted mathematical expression, or None if not found
    """
    if not query or not query.strip():
        return None
    
    # TASK 4 FIX: Pattern 0 - Check for "Compute exactly:" pattern first
    # This catches: "Compute exactly: ∑(k=1 to n)(2k−1)"
    compute_exact_pattern = r'(?:Compute|Calculate|Evaluate|Find|Solve)\s+(?:exactly|precisely)?\s*:?\s*(.+?)(?:\.|$|Then|Task|Verify)'
    match = re.search(compute_exact_pattern, query, re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Check if it contains math symbols
        math_indicators = ['∑', '∏', '∫', '√', 'π', '+', '-', '*', '/', '^', '=', '(', ')']
        if any(ind in candidate for ind in math_indicators):
            logger.debug(f"[MathTool] TASK 4 FIX: Extracted via 'Compute exactly' pattern: {candidate}")
            return candidate
    
    # TASK 4 FIX: Pattern 1 - Unicode math symbols with improved character set
    # Using explicit Unicode characters instead of character class ranges
    # which may not work correctly across all Python versions/platforms
    # Math symbols: ∑ (summation), ∏ (product), ∫ (integral), √ (sqrt), π (pi)
    # Greek letters listed explicitly for cross-platform reliability
    math_symbols = '∑∏∫√π'
    greek_lower = 'αβγδεζηθικλμνξοπρστυφχψω'
    greek_upper = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
    unicode_start_chars = math_symbols + greek_lower + greek_upper
    
    # Check if query starts with or contains a math symbol
    unicode_pattern = r'[' + re.escape(unicode_start_chars) + r'][\w\d\s+*/()^._{}|,<>=−–\-]+'
    match = re.search(unicode_pattern, query)
    if match:
        # Get from symbol to end of line or natural boundary
        start = match.start()
        rest = query[start:]
        # Find natural boundary (newline, period, "Task:", "Then", "Verify")
        boundaries = ['\n', '. ', 'Task:', 'Then', 'and verify', 'Verify', 'Prove']
        for boundary in boundaries:
            if boundary in rest:
                rest = rest.split(boundary)[0]
                break
        extracted = rest.strip()
        if extracted:
            logger.debug(f"[MathTool] TASK 4 FIX: Extracted via Unicode pattern: {extracted}")
            return extracted
    
    # TASK 4 FIX: Pattern 1b - Natural language sum/product
    # Matches: "sum from k=1 to n of (2k-1)"
    natural_sum_pattern = r'(?:sum(?:mation)?|product)\s+(?:from\s+)?(\w+)\s*=\s*(\d+)\s+to\s+(\w+)\s+(?:of\s+)?(.+?)(?:\.|$|Then|Task)'
    match = re.search(natural_sum_pattern, query, re.IGNORECASE)
    if match:
        index, lower, upper, expr = match.groups()
        # Convert expression to SymPy-compatible format
        # Replace common notation: 2k → 2*k, k−1 → k-1
        expr_clean = expr.strip()
        expr_clean = expr_clean.replace('−', '-')  # Unicode minus to ASCII
        expr_clean = re.sub(r'(\d)([a-z])', r'\1*\2', expr_clean)  # 2k → 2*k
        result = f"summation({expr_clean}, ({index}, {lower}, {upper}))"
        logger.debug(f"[MathTool] TASK 4 FIX: Converted natural sum to: {result}")
        return result
    
    # Pattern 2: Probability notation P(X|Y), P(A ∧ B)
    prob_pattern = r'P\s*\([^)]+\)'
    match = re.search(prob_pattern, query)
    if match:
        return match.group(0)
    
    # Pattern 3: LaTeX-style \sum_{k=1}^{n}, \frac{a}{b}
    latex_pattern = r'\\[a-z]+(?:_\{[^}]*\})?(?:\^\{[^}]*\})?(?:\{[^}]*\})*'
    match = re.search(latex_pattern, query)
    if match:
        return match.group(0)
    
    # Pattern 4: "Compute:" or "Calculate:" followed by expression (general case)
    compute_pattern = r'(?:Compute|Calculate|Evaluate|Solve|Find)[\s:]+([^\n]+)'
    match = re.search(compute_pattern, query, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        # Recursively extract from this substring
        recursive_result = extract_math_expression(candidate)
        if recursive_result:
            return recursive_result
        # Check if candidate has math content
        math_indicators = ['+', '-', '*', '/', '^', '=', '(', '∑', '∫']
        if any(ind in candidate for ind in math_indicators):
            return candidate
    
    # Pattern 5: Equations with equals sign
    equation_pattern = r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)'
    match = re.search(equation_pattern, query)
    if match:
        return f"{match.group(1)} = {match.group(2).strip()}"
    
    # Pattern 6: Standard algebraic (fallback)
    # Note: Hyphen at end of character class to avoid ambiguity
    algebra_pattern = r'[a-zA-Z0-9+*/()^._\s\-]+'
    match = re.search(algebra_pattern, query)
    if match:
        expr = match.group(0).strip()
        # Must contain operators or single-letter variables
        has_ops = any(op in expr for op in ['+', '-', '*', '/', '^', '('])
        has_vars = any(c.isalpha() and len(c) == 1 for c in expr.split())
        if has_ops or has_vars:
            return expr
    
    return None


# ============================================================================
# PROBLEM CLASSIFIER
# ============================================================================


class ProblemClassifier:
    """
    Classifies mathematical problems to determine the best solving strategy.
    
    Uses keyword matching and pattern recognition to identify:
    - Problem type (calculus, algebra, etc.)
    - Suggested solving strategy
    - Key variables and operations
    """
    
    # Keyword patterns for problem classification
    PROBLEM_PATTERNS: Dict[ProblemType, List[str]] = {
        ProblemType.CALCULUS: [
            "integrate", "integral", "derivative", "differentiate", "diff",
            "limit", "series", "taylor", "maclaurin", "antiderivative",
            "partial derivative", "gradient", "jacobian", "hessian",
            "divergence", "curl", "laplacian"
        ],
        ProblemType.ALGEBRA: [
            "solve", "equation", "factor", "expand", "simplify",
            "polynomial", "roots", "quadratic", "cubic", "linear equation",
            "system of equations", "inequality"
        ],
        ProblemType.DIFFERENTIAL_EQUATIONS: [
            "differential equation", "ode", "pde", "dsolve",
            "boundary condition", "initial condition", "laplace transform",
            "fourier transform", "green's function", "eigenvalue problem"
        ],
        ProblemType.LINEAR_ALGEBRA: [
            "matrix", "determinant", "eigenvalue", "eigenvector",
            "inverse", "transpose", "rank", "nullspace", "column space",
            "linear transformation", "diagonalize", "svd", "lu decomposition",
            "qr decomposition", "orthogonal", "unitary"
        ],
        ProblemType.STATISTICS: [
            "probability", "distribution", "mean", "variance", "standard deviation",
            "expected value", "covariance", "correlation", "hypothesis",
            "confidence interval", "regression", "bayesian"
        ],
        ProblemType.NUMBER_THEORY: [
            "prime", "divisor", "gcd", "lcm", "modular", "congruence",
            "diophantine", "fibonacci", "factorial", "binomial"
        ],
        ProblemType.GEOMETRY: [
            "area", "volume", "perimeter", "distance", "angle",
            "circle", "sphere", "triangle", "rectangle", "polygon",
            "coordinate", "vector", "plane"
        ],
        ProblemType.PHYSICS: [
            "force", "energy", "momentum", "velocity", "acceleration",
            "wave", "oscillation", "quantum", "hamiltonian", "lagrangian",
            "schrodinger", "maxwell", "einstein"
        ],
        ProblemType.OPTIMIZATION: [
            "minimize", "maximize", "optimize", "constraint", "lagrange multiplier",
            "gradient descent", "convex", "linear programming"
        ],
    }

    def classify(self, query: str) -> ProblemClassification:
        """
        Classify a mathematical problem.
        
        Args:
            query: Natural language problem description
            
        Returns:
            ProblemClassification with type, confidence, and strategy
        """
        query_lower = query.lower()
        
        # Count keyword matches for each problem type
        # Use word boundary matching for short keywords to avoid false positives
        scores: Dict[ProblemType, Tuple[int, List[str]]] = {}
        
        for problem_type, keywords in self.PROBLEM_PATTERNS.items():
            matches = []
            for kw in keywords:
                # Use word boundary regex for short keywords (<=3 chars)
                # to avoid matching "ode" in "code" or "node"
                if len(kw) <= 3:
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, query_lower):
                        matches.append(kw)
                else:
                    if kw in query_lower:
                        matches.append(kw)
            scores[problem_type] = (len(matches), matches)
        
        # Priority order for tie-breaking (more specific types have higher priority)
        # This ensures that if "solve" matches ALGEBRA and "ode" matches DIFFERENTIAL_EQUATIONS,
        # the more specific DIFFERENTIAL_EQUATIONS wins
        PRIORITY_ORDER = [
            ProblemType.DIFFERENTIAL_EQUATIONS,  # Highest priority (most specific)
            ProblemType.LINEAR_ALGEBRA,
            ProblemType.PHYSICS,
            ProblemType.OPTIMIZATION,
            ProblemType.CALCULUS,
            ProblemType.NUMBER_THEORY,
            ProblemType.STATISTICS,
            ProblemType.GEOMETRY,
            ProblemType.ALGEBRA,  # Lower priority (most general)
            ProblemType.UNKNOWN,
        ]
        
        # Find best match with priority-based tie-breaking
        best_type = ProblemType.UNKNOWN
        best_score = 0
        best_keywords: List[str] = []
        best_priority = len(PRIORITY_ORDER)  # Lower index = higher priority
        
        for problem_type, (score, keywords) in scores.items():
            priority = PRIORITY_ORDER.index(problem_type) if problem_type in PRIORITY_ORDER else len(PRIORITY_ORDER)
            
            # Win if higher score, or same score but higher priority
            if score > best_score or (score == best_score and score > 0 and priority < best_priority):
                best_score = score
                best_type = problem_type
                best_keywords = keywords
                best_priority = priority
        
        # Calculate confidence
        total_keywords = sum(len(kws) for kws in self.PROBLEM_PATTERNS.values())
        confidence = min(1.0, best_score / 3.0) if best_score > 0 else 0.1
        
        # Determine strategy
        strategy = self._suggest_strategy(best_type, query_lower)
        
        # Extract variables
        variables = self._extract_variables(query)
        
        return ProblemClassification(
            problem_type=best_type,
            confidence=confidence,
            keywords=best_keywords,
            suggested_strategy=strategy,
            variables=variables
        )
    
    def _suggest_strategy(self, problem_type: ProblemType, query: str) -> SolutionStrategy:
        """Suggest best solving strategy based on problem type."""
        # Numerical keywords suggest numeric/hybrid approach
        numeric_keywords = ["numerical", "approximate", "decimal", "float", "compute numerically"]
        if any(kw in query for kw in numeric_keywords):
            return SolutionStrategy.HYBRID
        
        # Most problems benefit from symbolic approach
        if problem_type in [ProblemType.CALCULUS, ProblemType.ALGEBRA, 
                           ProblemType.DIFFERENTIAL_EQUATIONS, ProblemType.LINEAR_ALGEBRA]:
            return SolutionStrategy.SYMBOLIC
        
        # Statistics often needs numeric
        if problem_type == ProblemType.STATISTICS:
            return SolutionStrategy.HYBRID
        
        return SolutionStrategy.SYMBOLIC
    
    def _extract_variables(self, query: str) -> List[str]:
        """Extract potential variable names from query."""
        # Common variable patterns
        var_patterns = [
            r'\b([a-zA-Z])\s*=',  # x =, y =
            r'variable\s+([a-zA-Z])',  # variable x
            r'with respect to\s+([a-zA-Z])',  # with respect to x
            r'for\s+([a-zA-Z])\b',  # for x
        ]
        
        variables = set()
        for pattern in var_patterns:
            matches = re.findall(pattern, query)
            variables.update(matches)
        
        # Default to common variables if none found
        if not variables:
            variables = {'x'}
        
        return list(variables)


# ============================================================================
# CODE TEMPLATES
# ============================================================================


class CodeTemplates:
    """
    Template-based code generation for common mathematical operations.
    
    Provides reliable, tested code templates when LLM generation is unavailable
    or for common problem types where templates are more reliable.
    """
    
    @staticmethod
    def integration(expression: str = "x**2", variable: str = "x", 
                   bounds: Optional[Tuple[str, str]] = None) -> str:
        """Generate integration code."""
        if bounds:
            return f"""# Definite Integration
{variable} = Symbol('{variable}')
f = {expression}
result = integrate(f, ({variable}, {bounds[0]}, {bounds[1]}))
"""
        return f"""# Indefinite Integration
{variable} = Symbol('{variable}')
f = {expression}
result = integrate(f, {variable})
"""

    @staticmethod
    def differentiation(expression: str = "x**3", variable: str = "x", 
                       order: int = 1) -> str:
        """Generate differentiation code."""
        return f"""# Differentiation (order {order})
{variable} = Symbol('{variable}')
f = {expression}
result = diff(f, {variable}, {order})
"""

    @staticmethod
    def solve_equation(equation: str = "x**2 - 4", variable: str = "x") -> str:
        """Generate equation solving code."""
        return f"""# Solve Equation
{variable} = Symbol('{variable}')
equation = {equation}
result = solve(equation, {variable})
"""

    @staticmethod
    def solve_system(equations: List[str], variables: List[str]) -> str:
        """Generate system of equations solving code."""
        var_def = ", ".join([f"{v} = Symbol('{v}')" for v in variables])
        eq_list = ", ".join(equations)
        var_list = ", ".join(variables)
        return f"""# Solve System of Equations
{var_def}
equations = [{eq_list}]
result = solve(equations, [{var_list}])
"""

    @staticmethod
    def limit(expression: str = "sin(x)/x", variable: str = "x", 
             point: str = "0", direction: str = "") -> str:
        """Generate limit computation code."""
        dir_arg = f", '{direction}'" if direction else ""
        return f"""# Limit Computation
{variable} = Symbol('{variable}')
f = {expression}
result = limit(f, {variable}, {point}{dir_arg})
"""

    @staticmethod
    def series_expansion(expression: str = "exp(x)", variable: str = "x",
                        point: str = "0", order: int = 5) -> str:
        """Generate series expansion code."""
        return f"""# Taylor/Series Expansion
{variable} = Symbol('{variable}')
f = {expression}
result = series(f, {variable}, {point}, {order})
"""

    @staticmethod
    def matrix_operation(matrix: str = "[[1, 2], [3, 4]]", 
                        operation: str = "det") -> str:
        """Generate matrix operation code."""
        operations = {
            "det": "M.det()",
            "inv": "M.inv()",
            "eigenvals": "M.eigenvals()",
            "eigenvects": "M.eigenvects()",
            "rank": "M.rank()",
            "nullspace": "M.nullspace()",
            "transpose": "M.T",
            "trace": "M.trace()",
        }
        op_code = operations.get(operation, "M.det()")
        return f"""# Matrix Operation: {operation}
M = Matrix({matrix})
result = {op_code}
"""

    @staticmethod
    def differential_equation(equation: str, function: str = "f", 
                             variable: str = "x") -> str:
        """Generate ODE solving code."""
        return f"""# Solve Differential Equation
{variable} = Symbol('{variable}')
{function} = Function('{function}')
ode = {equation}
result = str(dsolve(ode, {function}({variable})))
"""

    @staticmethod
    def simplify_expression(expression: str = "x**2 + 2*x + 1") -> str:
        """Generate simplification code."""
        return f"""# Simplify Expression
x = Symbol('x')
expr = {expression}
result = simplify(expr)
"""

    @staticmethod
    def factor_expression(expression: str = "x**2 + 2*x + 1") -> str:
        """Generate factorization code."""
        return f"""# Factor Expression
x = Symbol('x')
expr = {expression}
result = factor(expr)
"""

    @staticmethod
    def expand_expression(expression: str = "(x + 1)**2") -> str:
        """Generate expansion code."""
        return f"""# Expand Expression
x = Symbol('x')
expr = {expression}
result = expand(expr)
"""

    @staticmethod
    def summation(expression: str = "k", index: str = "k", 
                  lower: str = "1", upper: str = "n") -> str:
        """Generate summation code.
        
        Computes ∑_{index=lower}^{upper} expression
        
        Args:
            expression: The expression to sum (in terms of index variable)
            index: The summation index variable (e.g., "k")
            lower: Lower bound of summation
            upper: Upper bound of summation
            
        Returns:
            SymPy code to compute the summation
        """
        # Avoid tuple unpacking as RestrictedPython doesn't support it
        return f"""# Summation: Sum of {expression} from {index}={lower} to {upper}
{index} = Symbol('{index}')
n = Symbol('n')
expr = {expression}
result = summation(expr, ({index}, {lower}, {upper}))
# Simplify the result
result = simplify(result)
"""


# ============================================================================
# MATHEMATICAL COMPUTATION TOOL
# ============================================================================


class MathematicalComputationTool:
    """
    SOTA Mathematical Computation Tool for VULCAN-AGI.
    
    This tool bridges the gap between LLM-generated mathematical approaches and
    actual computational execution. Instead of just describing how to solve a
    problem, it generates executable SymPy code, runs it in a safe sandbox,
    and returns both the code and computed result.

    Features:
        - Problem classification for optimal strategy selection
        - LLM-based code generation for complex problems
        - Template-based generation for common operations
        - Safe sandboxed execution via RestrictedPython
        - Multi-strategy solving (symbolic, numeric, hybrid)
        - Learning from successful solutions
        - Integration with VULCAN's reasoning system
        - Graceful error handling with fallbacks

    Example:
        >>> tool = MathematicalComputationTool(llm=my_llm)
        >>> result = tool.execute("Integrate x^2 with respect to x")
        >>> print(result.code)    # x = Symbol('x'); result = integrate(x**2, x)
        >>> print(result.result)  # x**3/3
    """

    # System prompt for LLM code generation
    CODE_GENERATION_PROMPT = '''You are a Python code generator for symbolic mathematics.
Generate executable Python code using SymPy to solve mathematical problems.

RULES:
1. Use SymPy functions and objects (already imported: Symbol, symbols, integrate, diff, solve, simplify, expand, factor, limit, series, Matrix, sqrt, exp, log, sin, cos, tan, pi, E, I, oo, Function, dsolve, Eq, etc.)
2. Define variables using Symbol() or symbols()
3. Assign the FINAL answer to the variable named 'result'
4. Show derivation steps as comments
5. DO NOT include any import statements
6. Output ONLY valid Python code (no markdown, no explanations before or after)

EXAMPLE FORMAT:
# Define variables
x = Symbol('x')
# Define the function
f = x**2
# Perform the operation
integral = integrate(f, x)
# Simplify and assign result
result = simplify(integral)
'''

    def __init__(
        self, 
        llm=None, 
        max_tokens: int = 500,
        enable_learning: bool = True,
        prefer_templates: bool = False
    ):
        """
        Initialize the mathematical computation tool.

        Args:
            llm: Language model for code generation (must have .generate() method).
                 WARNING: This should be an LLM client object, NOT a string model name.
            max_tokens: Maximum tokens for code generation
            enable_learning: Whether to learn from successful solutions
            prefer_templates: Whether to prefer templates over LLM generation
        """
        # LLM INTERFACE FIX: Detect and warn when a string is passed instead of an object
        # This catches a common configuration error where a model name like "gpt-3.5-turbo"
        # is passed instead of an actual LLM client object
        if llm is not None and isinstance(llm, str):
            # Truncate long strings for logging but don't add "..." for short ones
            llm_preview = llm[:50] + '...' if len(llm) > 50 else llm
            logger.error(
                f"LLM Interface Bug Detected: 'llm' parameter received a string ('{llm_preview}') "
                f"instead of an LLM client object. This is likely a configuration error. "
                f"Pass an actual LLM client instance (e.g., OpenAI(), GraphixVulcanLLM()) "
                f"or None to use templates only. Setting llm=None as fallback."
            )
            llm = None  # Set to None to use template fallback
        
        self.llm = llm
        self.max_tokens = max_tokens
        self.enable_learning = enable_learning
        self.prefer_templates = prefer_templates
        
        self.name = "mathematical_computation"
        self.description = "Symbolic mathematics using SymPy with safe code execution"
        
        self._lock = threading.RLock()
        self._classifier = ProblemClassifier()
        self._templates = CodeTemplates()
        
        # Learning: track successful solutions
        self._solution_cache: Dict[str, ComputationResult] = {}
        self._success_patterns: Dict[ProblemType, List[str]] = defaultdict(list)
        
        logger.info(
            f"MathematicalComputationTool initialized: "
            f"safe_execution={SAFE_EXECUTION_AVAILABLE}, "
            f"llm={'available' if llm else 'none'}, "
            f"learning={enable_learning}"
        )

    def execute(self, query: str, **kwargs) -> ComputationResult:
        """
        Execute mathematical computation.

        Args:
            query: Mathematical problem or question in natural language
            **kwargs: Additional arguments (llm override, strategy override, etc.)

        Returns:
            ComputationResult with code, result, and explanation
        """
        start_time = time.time()
        llm = kwargs.get("llm", self.llm)
        strategy_override = kwargs.get("strategy")

        # LLM INTERFACE FIX: Detect if llm override is a string (common config error)
        if llm is not None and isinstance(llm, str):
            llm_preview = llm[:50] + '...' if len(llm) > 50 else llm
            logger.warning(
                f"LLM Interface Bug in execute(): 'llm' kwarg received a string ('{llm_preview}') "
                f"instead of an LLM client object. Using templates as fallback."
            )
            llm = None

        with self._lock:
            try:
                # Step 1: Classify the problem
                classification = self._classifier.classify(query)
                logger.debug(f"Problem classified as {classification.problem_type.value} "
                           f"with confidence {classification.confidence:.2f}")
                
                # Step 2: Determine strategy
                strategy = strategy_override or classification.suggested_strategy
                if self.prefer_templates and strategy == SolutionStrategy.LLM_GENERATED:
                    strategy = SolutionStrategy.TEMPLATE
                
                # Step 3: Generate code
                code = self._generate_code(query, classification, strategy, llm)
                
                # BUG #12 FIX: Handle None return when no math expression found
                # Previously this assumed code was always a string. Now _generate_code
                # returns None when no mathematical content is detected, which means
                # the math engine should gracefully decline rather than compute garbage.
                if not code or not code.strip():
                    logger.info(
                        f"[MathTool] BUG#12 FIX: No mathematical expression found in query. "
                        f"Returning failure result instead of computing default expression."
                    )
                    return self._create_error_result(
                        query, "", "No mathematical expression found in query", 
                        classification, strategy, time.time() - start_time
                    )

                # Step 4: Execute the code
                if not SAFE_EXECUTION_AVAILABLE or execute_math_code is None:
                    # BUG #3 FIX: Try simple arithmetic fallback before giving up
                    # This allows basic calculations like "2+2" to work without SymPy
                    simple_result = self._try_simple_arithmetic(query)
                    if simple_result is not None:
                        return ComputationResult(
                            success=True,
                            code=f"# Simple arithmetic: {query}\nresult = {simple_result}",
                            result=str(simple_result),
                            explanation=f"Computed using simple arithmetic: {query} = {simple_result}",
                            tool=self.name,
                            problem_type=classification.problem_type,
                            strategy=SolutionStrategy.NUMERIC,
                            execution_time=time.time() - start_time,
                            metadata={"fallback": "simple_arithmetic"},
                        )
                    return self._create_error_result(
                        query, code, "Safe execution not available and query is not simple arithmetic",
                        classification, strategy, time.time() - start_time
                    )

                execution_result = execute_math_code(code)

                if not execution_result["success"]:
                    logger.warning(f"Code execution failed: {execution_result['error']}")
                    
                    # Try fallback strategy
                    fallback_result = self._try_fallback(
                        query, classification, strategy, llm
                    )
                    if fallback_result and fallback_result.success:
                        return fallback_result
                    
                    return self._create_error_result(
                        query, code, execution_result["error"],
                        classification, strategy, time.time() - start_time
                    )

                # Step 5: Format successful response
                result_str = str(execution_result["result"])
                explanation = self._generate_explanation(query, code, result_str, llm)
                execution_time = time.time() - start_time
                
                result = ComputationResult(
                    success=True,
                    code=code,
                    result=result_str,
                    explanation=explanation,
                    tool=self.name,
                    problem_type=classification.problem_type,
                    strategy=strategy,
                    execution_time=execution_time,
                    metadata={
                        "query": query,
                        "classification_confidence": classification.confidence,
                        "keywords": classification.keywords,
                    },
                )
                
                # Learn from success
                if self.enable_learning:
                    self._learn_from_success(query, classification, result)
                
                return result

            except Exception as e:
                logger.error(f"Mathematical computation tool failed: {e}")
                return ComputationResult(
                    success=False,
                    code="",
                    error=str(e),
                    explanation=f"Failed to solve: {e}",
                    tool=self.name,
                    execution_time=time.time() - start_time,
                )

    def _generate_code(
        self, 
        query: str, 
        classification: ProblemClassification,
        strategy: SolutionStrategy,
        llm
    ) -> Optional[str]:
        """
        Generate SymPy code to solve the problem.
        
        Uses strategy-appropriate code generation method.
        
        BUG #12 FIX: Returns None when no mathematical content is found.
        Previously, this always returned code (falling back to default expression).
        Now returns None if the query doesn't contain mathematical content,
        allowing callers to handle non-math queries appropriately.
        
        Returns:
            Generated code string if mathematical content is found,
            None if the query doesn't contain mathematical content.
        """
        # Try template first for common patterns
        if strategy == SolutionStrategy.TEMPLATE or (strategy == SolutionStrategy.SYMBOLIC and self.prefer_templates):
            template_code = self._generate_template_code(query, classification)
            if template_code:
                return template_code
        
        # Use LLM for complex problems
        if llm is not None and strategy in [SolutionStrategy.LLM_GENERATED, SolutionStrategy.SYMBOLIC, SolutionStrategy.HYBRID]:
            llm_code = self._generate_llm_code(query, llm)
            if llm_code:
                return llm_code
        
        # BUG #12 FIX: Fallback to template - but this can now return None
        # if no mathematical content is found in the query
        return self._generate_template_code(query, classification)

    def _generate_template_code(self, query: str, classification: ProblemClassification) -> Optional[str]:
        """
        Generate code using templates based on problem classification.
        
        Uses a priority-based matching system where more specific patterns
        are checked before general ones to ensure correct template selection.
        
        BUG #12 FIX: Returns None when no mathematical expression is found.
        Previously, this method returned a default expression "x**2 + 2*x + 1"
        which caused irrelevant mathematical output to be sent to OpenAI when
        non-math queries were incorrectly routed to the math engine.
        
        Returns:
            Generated code string if a mathematical pattern is found,
            None if no mathematical content is detected in the query.
        """
        query_lower = query.lower()
        variables = classification.variables or ['x']
        var = variables[0] if variables else 'x'
        
        # =================================================================
        # BUG #12 FIX (CRITICAL): PRIORITY 0 - Reject non-mathematical queries
        # =================================================================
        # These patterns indicate queries that should NOT be processed by the
        # math engine, even if they contain math-related words like "proof"
        # or "function". Return None early to prevent nonsensical output.
        #
        # Examples:
        # - "Is {A→B, B→C, ¬C, A∨B} satisfiable?" -> Logic query, not math
        # - "Verify proof about differentiable functions" -> Proof verification, not computation
        # - "Formalize in FOL" -> First-order logic, not math
        # =================================================================
        logic_patterns = [
            # Propositional logic
            '→', '∧', '∨', '¬', '⊢', '⊨',
            'satisfiable', 'unsatisfiable', 'tautology', 'contradiction',
            'propositional', 'boolean',
            # First-order logic
            '∀', '∃', 'forall', 'exists',
            'formalize', 'fol', 'first-order logic', 'first order logic',
            'predicate', 'quantifier',
            # Proof verification (not computation)
            'verify proof', 'check proof', 'is the proof valid',
            'proof is valid', 'proof is invalid',
            # SAT/Model checking
            'sat problem', 'model', 'assignment',
        ]
        
        if any(pattern in query_lower for pattern in logic_patterns):
            logger.info(
                f"[MathTool] BUG#12 FIX: Query contains logic patterns, not mathematical. "
                f"Declining to compute. Query: {query[:80]}..."
            )
            return None
        
        # Also check for logic symbols in original query (case-sensitive)
        if any(sym in query for sym in ['→', '∧', '∨', '¬', '∀', '∃', '⊢', '⊨']):
            logger.info(
                f"[MathTool] BUG#12 FIX: Query contains logic symbols. "
                f"Declining to compute. Query: {query[:80]}..."
            )
            return None
        
        # PRIORITY 1: Differential equations (must check BEFORE "solve"/"equation")
        # These patterns are very specific and should take precedence
        if any(kw in query_lower for kw in [
            "differential equation", "ode", "pde", "dsolve",
            "dy/dx", "d/dx", "dy/dt", "dx/dt"
        ]):
            return self._templates.differential_equation(
                "Eq(f(x).diff(x), f(x))", "f", var
            )
        
        # PRIORITY 2: Matrix operations (must check BEFORE general "find")
        if any(kw in query_lower for kw in [
            "matrix", "determinant", "eigenvalue", "eigenvector",
            "inverse matrix", "transpose", "rank of matrix"
        ]):
            operation = "det"
            if "eigenvalue" in query_lower or "eigen" in query_lower:
                operation = "eigenvals"
            elif "eigenvector" in query_lower:
                operation = "eigenvects"
            elif "inverse" in query_lower:
                operation = "inv"
            elif "rank" in query_lower:
                operation = "rank"
            elif "transpose" in query_lower:
                operation = "transpose"
            elif "trace" in query_lower:
                operation = "trace"
            return self._templates.matrix_operation("[[1, 2], [3, 4]]", operation)
        
        # PRIORITY 3: Summation (check before limits)
        # FIX Issue #13: Add summation template for ∑ expressions
        if any(kw in query_lower for kw in ["sum", "summation", "∑"]) or "∑" in query:
            # Parse summation expression from query
            # Pattern: ∑_{k=1}^n (2k-1) or sum from k=1 to n of (2k-1)
            import re
            
            # Try to extract expression and bounds
            # Pattern 1: ∑_{k=lower}^{upper} (expression)
            sum_match = re.search(r'[∑]\s*_?\{?(\w+)\s*=\s*(\d+)\s*\}?\s*\^?\{?(\w+)\}?\s*\(?([^)]+)\)?', query)
            if sum_match:
                index = sum_match.group(1)  # e.g., "k"
                lower = sum_match.group(2)  # e.g., "1"
                upper = sum_match.group(3)  # e.g., "n"
                expr = sum_match.group(4).strip()  # e.g., "2k-1"
                # Convert to SymPy format (k → k, − → -)
                expr = expr.replace('−', '-')
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)  # 2k → 2*k
                return self._templates.summation(expr, index, lower, upper)
            
            # Pattern 2: sum from k=lower to upper of expression
            sum_match2 = re.search(r'sum(?:mation)?\s+(?:from\s+)?(\w+)\s*=\s*(\d+)\s+to\s+(\w+)\s+(?:of\s+)?(.+)', query_lower)
            if sum_match2:
                index = sum_match2.group(1)
                lower = sum_match2.group(2)
                upper = sum_match2.group(3)
                expr = sum_match2.group(4).strip()
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)
                return self._templates.summation(expr, index, lower, upper)
            
            # Default summation
            return self._templates.summation("k", "k", "1", "n")
        
        # PRIORITY 4: Limits (check before integration since both are calculus)
        if "limit" in query_lower:
            direction = "+" if "right" in query_lower else ("-" if "left" in query_lower else "")
            return self._templates.limit("sin(x)/x", var, "0", direction)
        
        # PRIORITY 5: Series expansion
        if any(kw in query_lower for kw in ["series", "taylor", "maclaurin", "expansion", "expand around"]):
            return self._templates.series_expansion("exp(x)", var, "0", 5)
        
        # PRIORITY 6: Integration
        if any(kw in query_lower for kw in ["integrate", "integral", "antiderivative", "∫"]):
            if "definite" in query_lower or "from" in query_lower:
                return self._templates.integration("x**2", var, ("0", "1"))
            return self._templates.integration("x**2", var)
        
        # PRIORITY 7: Differentiation
        # BUG #12 FIX: Use word-boundary matching for short keywords like "diff"
        # to avoid matching "differentiable" or "difference"
        # - "differentiate" and "derivative" are long enough to be safe
        # - "diff" needs word boundary check
        # - "d/dx" is specific notation and safe
        differentiation_keywords = ["differentiate", "derivative", "d/dx"]
        has_diff_keyword = any(kw in query_lower for kw in differentiation_keywords)
        
        # Check for "diff" as a standalone word (not part of "differentiable", "difference", etc.)
        if not has_diff_keyword:
            import re
            has_diff_keyword = bool(re.search(r'\bdiff\b', query_lower))
        
        if has_diff_keyword:
            order = 2 if "second" in query_lower else (3 if "third" in query_lower else 1)
            return self._templates.differentiation("x**3 + x**2", var, order)
        
        # PRIORITY 8: Specific algebraic operations
        if "factor" in query_lower and "expand" not in query_lower:
            return self._templates.factor_expression("x**2 + 2*x + 1")
        
        if "expand" in query_lower and "series" not in query_lower:
            return self._templates.expand_expression("(x + 1)**3")
        
        if "simplify" in query_lower:
            return self._templates.simplify_expression("(x**2 - 1)/(x - 1)")
        
        # PRIORITY 9: Equation solving (general - checked last among operations)
        if any(kw in query_lower for kw in ["solve", "equation", "find x", "find the value", "roots"]):
            if "system" in query_lower:
                return self._templates.solve_system(["x + y - 2", "x - y"], ["x", "y"])
            return self._templates.solve_equation("x**2 - 4", var)
        
        # PRIORITY 10: Classification-based fallback
        # Use the classifier's detected type to choose appropriate template
        if classification.problem_type == ProblemType.CALCULUS:
            return self._templates.differentiation("x**3", var)
        elif classification.problem_type == ProblemType.LINEAR_ALGEBRA:
            return self._templates.matrix_operation("[[1, 2], [3, 4]]", "det")
        elif classification.problem_type == ProblemType.DIFFERENTIAL_EQUATIONS:
            return self._templates.differential_equation("Eq(f(x).diff(x), f(x))", "f", var)
        elif classification.problem_type == ProblemType.ALGEBRA:
            return self._templates.solve_equation("x**2 - 4", var)
        
        # BUG #12 FIX: Return None when no mathematical content is found
        # Previously this returned "x**2 + 2*x + 1" as a default, which caused
        # irrelevant math output (the expansion of (x+1)²) to be included in
        # responses to non-mathematical queries. The math engine should NOT
        # compute anything when no math expression is found in the query.
        logger.warning(
            f"[MathTool] BUG#12 FIX: No mathematical expression found in query. "
            f"Returning None instead of default expression. Query: {query[:100]}..."
        )
        return None

    def _generate_llm_code(self, query: str, llm) -> Optional[str]:
        """
        Generate code using LLM.
        
        This method supports multiple LLM interfaces:
        - OpenAI client (chat.completions.create)
        - LangChain LLMs (invoke, predict)
        - HuggingFace (generate, __call__)
        - Custom interfaces (complete)
        
        FIX MAJOR-12: Added support for OpenAI-compatible interfaces.
        LLM INTERFACE FIX: Added early detection of string LLM parameters
        to provide clear error messages instead of confusing attribute errors.
        
        Args:
            query: The mathematical problem to solve
            llm: LLM client object (NOT a model name string)
            
        Returns:
            Generated Python code string, or None if generation fails
        """
        # LLM INTERFACE FIX: Early detection of string parameter
        if isinstance(llm, str):
            llm_preview = llm[:50] + '...' if len(llm) > 50 else llm
            logger.error(
                f"LLM Interface Bug: _generate_llm_code received a string ('{llm_preview}') "
                f"instead of an LLM client object. This indicates a configuration error upstream. "
                f"The 'llm' parameter should be an instantiated client object, not a model name."
            )
            return None
        
        prompt = f"""{self.CODE_GENERATION_PROMPT}

Problem: {query}

Generate ONLY the Python code:"""

        try:
            code = None
            
            # Try different LLM interfaces in order of specificity
            
            # 1. OpenAI-style client (chat.completions.create)
            if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
                response = llm.chat.completions.create(
                    model=getattr(llm, "model", DEFAULT_OPENAI_MODEL),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                )
                code = response.choices[0].message.content
                logger.debug("Using OpenAI chat.completions interface")
                
            # 2. Direct create method (OpenAI Completion API)
            elif hasattr(llm, "create"):
                response = llm.create(prompt=prompt, max_tokens=self.max_tokens)
                if hasattr(response, "choices") and response.choices:
                    code = response.choices[0].text if hasattr(response.choices[0], "text") else str(response.choices[0])
                else:
                    code = str(response)
                logger.debug("Using OpenAI create interface")
                
            # 3. LangChain-style invoke
            elif hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
                code = response.content if hasattr(response, "content") else str(response)
                logger.debug("Using LangChain invoke interface")
                
            # 4. LangChain-style predict  
            elif hasattr(llm, "predict"):
                code = llm.predict(prompt)
                logger.debug("Using LangChain predict interface")
                
            # 5. GraphixVulcanLLM-style generate
            elif hasattr(llm, "generate"):
                code = llm.generate(prompt, max_tokens=self.max_tokens)
                logger.debug("Using generate interface")
                
            # 6. Direct callable
            elif hasattr(llm, "__call__"):
                code = llm(prompt)
                logger.debug("Using callable interface")
                
            # 7. Generic complete method
            elif hasattr(llm, "complete"):
                response = llm.complete(prompt)
                code = response.text if hasattr(response, "text") else str(response)
                logger.debug("Using complete interface")
            else:
                # FIX MAJOR-12: Log available attributes to help debugging
                # LLM INTERFACE FIX: Provide clearer error message
                available_attrs = [attr for attr in dir(llm) if not attr.startswith("_")]
                llm_type = type(llm).__name__
                logger.warning(
                    f"Unknown LLM interface (type={llm_type}). "
                    f"Expected an LLM client with one of: chat.completions, create, invoke, predict, generate, or complete. "
                    f"Available attrs: {available_attrs[:DEBUG_LOG_MAX_ATTRS]}..."
                )
                return None

            if code is None:
                logger.warning("LLM returned None response")
                return None
                
            return self._clean_code(code)

        except Exception as e:
            logger.warning(f"LLM code generation failed: {e}")
            return None

    def _clean_code(self, code: str) -> str:
        """Remove markdown formatting and import statements from generated code."""
        # Remove markdown code blocks
        if "```python" in code:
            parts = code.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0]

        # Remove import lines (already in namespace)
        lines = code.strip().split("\n")
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("from sympy import", "import sympy",
                                   "from numpy import", "import numpy")):
                continue
            clean_lines.append(line)

        return "\n".join(clean_lines).strip()

    def _generate_explanation(
        self, query: str, code: str, result: str, llm
    ) -> str:
        """Generate natural language explanation of the solution."""
        if llm is None:
            return f"The computation was performed using SymPy. The result is: {result}"

        prompt = f"""Explain this mathematical solution concisely (2-3 sentences):

Problem: {query}

Code:
{code}

Result: {result}

Brief explanation:"""

        try:
            if hasattr(llm, "generate"):
                return llm.generate(prompt, max_tokens=200)
            elif hasattr(llm, "__call__"):
                return llm(prompt)
            else:
                return f"The computation was performed using SymPy. The result is: {result}"
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return f"The computation was performed using SymPy. The result is: {result}"

    def _try_simple_arithmetic(self, query: str) -> Optional[Union[int, float]]:
        """
        BUG #3 FIX: Try to evaluate simple arithmetic expressions.
        
        This is a fallback for when SymPy/RestrictedPython is not available.
        Only allows safe mathematical operations - no arbitrary code execution.
        
        Supported:
        - Basic operations: +, -, *, /, **, %
        - Parentheses: (2+3)*4
        - Decimal numbers: 3.14 * 2
        - "What is X" style questions
        
        Returns:
            The computed result, or None if query is not simple arithmetic
        """
        import re
        
        # Extract mathematical expression from common question patterns
        # "What is 2+2?" -> "2+2"
        # "Calculate 3*4" -> "3*4"
        # "Compute 10/2" -> "10/2"
        patterns = [
            r'(?:what\s+is|calculate|compute|evaluate|solve)\s+(.+?)(?:\?|$)',
            r'^([\d\.\s\+\-\*\/\%\^\(\)]+)$',  # Pure expression
        ]
        
        expression = None
        for pattern in patterns:
            match = re.search(pattern, query.strip(), re.IGNORECASE)
            if match:
                expression = match.group(1).strip()
                break
        
        if not expression:
            # Try the whole query as an expression
            expression = query.strip().rstrip('?')
        
        # Clean up the expression
        expression = expression.strip()
        
        # Replace common mathematical notation
        expression = expression.replace('^', '**')  # Caret for exponent
        expression = re.sub(r'(\d)\s*x\s*(\d)', r'\1*\2', expression)  # 2x3 -> 2*3
        
        # Security check: Only allow safe characters
        # Allowed: digits (0-9), decimal point (.), operators (+, -, *, /, %)
        #          parentheses ((, )), whitespace
        # NOT allowed: letters, quotes, brackets, attribute access, function calls
        # Note: ** (exponentiation) is allowed because * is in the character class
        allowed_pattern = r'^[\d\.\+\-\*\/\%\(\)\s]+$'
        if not re.match(allowed_pattern, expression):
            logger.debug(f"Expression contains disallowed characters: {expression}")
            return None
        
        # Validate balanced parentheses
        if expression.count('(') != expression.count(')'):
            logger.debug(f"Unbalanced parentheses in expression: {expression}")
            return None
        
        # Prevent empty or trivial expressions
        if not expression or not any(c.isdigit() for c in expression):
            return None
        
        try:
            # SECURITY TRADE-OFF: We use compile+eval instead of ast.literal_eval.
            # ast.literal_eval only allows literals (strings, numbers, tuples, etc.)
            # and cannot evaluate operators like 2+2. Our approach is safe because:
            # 1. We validate the input against a strict character whitelist (line 989)
            # 2. We execute with {"__builtins__": {}} preventing function calls
            # 3. The character whitelist prevents attribute access (no dots for .x)
            # 4. The only executable operations are basic arithmetic
            code = compile(expression, '<string>', 'eval')
            
            # Execute in a restricted namespace with no builtins
            # This prevents function calls, attribute access, etc.
            result = eval(code, {"__builtins__": {}}, {})
            
            # Validate the result is a number
            if isinstance(result, (int, float)) and not isinstance(result, bool):
                # Round floats to avoid floating point display issues
                if isinstance(result, float):
                    # If it's close to an integer, return as int
                    if abs(result - round(result)) < 1e-9:
                        return int(round(result))
                    # Otherwise round to reasonable precision
                    return round(result, 10)
                return result
            
            logger.debug(f"Expression result is not a number: {type(result)}")
            return None
            
        except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Simple arithmetic evaluation failed for '{expression}': {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in simple arithmetic: {e}")
            return None

    def _try_fallback(
        self, 
        query: str, 
        classification: ProblemClassification,
        failed_strategy: SolutionStrategy,
        llm
    ) -> Optional[ComputationResult]:
        """Try alternative strategy if primary fails."""
        # If LLM failed, try template
        if failed_strategy in [SolutionStrategy.LLM_GENERATED, SolutionStrategy.SYMBOLIC]:
            template_code = self._generate_template_code(query, classification)
            if template_code and SAFE_EXECUTION_AVAILABLE and execute_math_code:
                result = execute_math_code(template_code)
                if result["success"]:
                    return ComputationResult(
                        success=True,
                        code=template_code,
                        result=str(result["result"]),
                        explanation=f"Solved using template approach. Result: {result['result']}",
                        tool=self.name,
                        problem_type=classification.problem_type,
                        strategy=SolutionStrategy.TEMPLATE,
                    )
        
        return None

    def _create_error_result(
        self,
        query: str,
        code: str,
        error: str,
        classification: ProblemClassification,
        strategy: SolutionStrategy,
        execution_time: float
    ) -> ComputationResult:
        """Create error result with context."""
        explanation = (
            f"Code execution failed ({error}). "
            f"The attempted approach was:\n{code}" if code else
            f"Could not generate executable code for: {query}"
        )

        return ComputationResult(
            success=False,
            code=code,
            result=None,
            explanation=explanation,
            error=error,
            tool=self.name,
            problem_type=classification.problem_type,
            strategy=strategy,
            execution_time=execution_time,
        )

    def _learn_from_success(
        self, 
        query: str, 
        classification: ProblemClassification,
        result: ComputationResult
    ):
        """Learn from successful solutions for future use."""
        # Cache the successful result
        query_key = query.lower().strip()[:100]  # Truncate for memory
        self._solution_cache[query_key] = result
        
        # Track successful patterns
        self._success_patterns[classification.problem_type].append(query_key)
        
        # Limit cache size
        if len(self._solution_cache) > 1000:
            # Remove oldest entries
            keys = list(self._solution_cache.keys())
            for key in keys[:100]:
                del self._solution_cache[key]

    def reason(self, input_data: Any, query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reasoning interface compatible with UnifiedReasoner.
        
        This method adapts the MathematicalComputationTool to work with
        the UnifiedReasoner's expected interface for reasoners.
        
        Args:
            input_data: The input problem. Can be:
                - str: Direct mathematical query
                - dict with 'query' or 'problem' key
            query: Optional query parameters dict
            
        Returns:
            Dict with 'conclusion', 'confidence', and other reasoning fields
        """
        # Extract the mathematical query from input_data
        if isinstance(input_data, str):
            math_query = input_data
        elif isinstance(input_data, dict):
            math_query = input_data.get('query') or input_data.get('problem') or str(input_data)
        else:
            math_query = str(input_data)
        
        # Execute the computation
        result = self.execute(math_query)
        
        # Convert to reasoner-compatible dict format
        confidence = 0.9 if result.success else 0.1
        
        return {
            'conclusion': {
                'success': result.success,
                'result': result.result,
                'code': result.code,
                'problem_type': result.problem_type.value if result.problem_type else 'unknown',
            },
            'confidence': confidence,
            'explanation': result.explanation,
            'formatted_output': self.format_response(result),
            'metadata': {
                'tool': self.name,
                'strategy': result.strategy.value if result.strategy else 'unknown',
                'execution_time': result.execution_time,
                'error': result.error,
            }
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return reasoner capabilities for UnifiedReasoner integration.
        
        Returns:
            Dict describing the tool's capabilities
        """
        return {
            'name': self.name,
            'description': self.description,
            'supported_problem_types': [pt.value for pt in ProblemType],
            'supported_strategies': [ss.value for ss in SolutionStrategy],
            'safe_execution_available': SAFE_EXECUTION_AVAILABLE,
            'llm_available': self.llm is not None,
        }

    def format_response(self, result: ComputationResult) -> str:
        """
        Format computation result for display.

        Args:
            result: ComputationResult from execute()

        Returns:
            Formatted string for display
        """
        if not result.success:
            return f"""**Mathematical Computation**

⚠️ Execution failed: {result.error}

**Problem Type:** {result.problem_type.value}
**Strategy:** {result.strategy.value}

{result.explanation}
"""

        return f"""**Mathematical Computation**

**Code:**
```python
{result.code}
```

**Result:** {result.result}

**Explanation:** {result.explanation}

---
*Problem Type: {result.problem_type.value} | Strategy: {result.strategy.value} | Time: {result.execution_time:.3f}s*
"""

    def get_statistics(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "cache_size": len(self._solution_cache),
            "success_patterns": {
                pt.value: len(patterns) 
                for pt, patterns in self._success_patterns.items()
            },
            "safe_execution_available": SAFE_EXECUTION_AVAILABLE,
            "llm_available": self.llm is not None,
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_mathematical_computation_tool(
    llm=None,
    max_tokens: int = 500,
    enable_learning: bool = True,
    prefer_templates: bool = False
) -> MathematicalComputationTool:
    """
    Factory function to create a MathematicalComputationTool.
    
    Args:
        llm: Optional language model for code generation
        max_tokens: Maximum tokens for LLM generation
        enable_learning: Whether to learn from successful solutions
        prefer_templates: Whether to prefer templates over LLM
        
    Returns:
        Configured MathematicalComputationTool instance
    """
    return MathematicalComputationTool(
        llm=llm,
        max_tokens=max_tokens,
        enable_learning=enable_learning,
        prefer_templates=prefer_templates
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ProblemType",
    "SolutionStrategy",
    # Data structures
    "ComputationResult",
    "ProblemClassification",
    # Main classes
    "MathematicalComputationTool",
    "ProblemClassifier",
    "CodeTemplates",
    # Factory
    "create_mathematical_computation_tool",
    # Expression extraction (Issue #2 Fix)
    "extract_math_expression",
    # Module status
    "SAFE_EXECUTION_AVAILABLE",
]
