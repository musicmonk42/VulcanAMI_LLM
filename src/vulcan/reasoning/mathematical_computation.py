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
# CONSTANTS - Explicit Mathematical Notation Detection (Issue #1 Fix)
# ============================================================================
# Symbols and keywords that indicate explicit mathematical content.
# When present, these bypass logic pattern rejection because they're
# clearly mathematical expressions, not logic problems.

EXPLICIT_MATH_SYMBOLS: Tuple[str, ...] = ('∑', '∫', '∏', '∂', '∇', '√')

EXPLICIT_MATH_KEYWORDS: Tuple[str, ...] = (
    'compute exactly', 'calculate exactly', 'evaluate exactly',
    'compute the sum', 'calculate the sum', 'evaluate the sum',
    'summation', 'sigma notation',
)

# ============================================================================
# ISSUE #3 FIX: Proof Verification Detection Patterns (Compiled for Performance)
# ============================================================================
# Compiled regex patterns for efficient proof verification detection.
# These are compiled once at module load time for optimal performance.
#
# INDUSTRY STANDARD: Pre-compiled regex patterns for efficient proof verification detection.
# These are compiled once at module load time for optimal performance.
#
# PERFORMANCE OPTIMIZATION: Non-greedy quantifiers (.*?) prevent catastrophic backtracking.
# Using non-greedy instead of greedy quantifiers ensures O(n) performance instead of
# potential O(2^n) worst-case with malicious input.

PROOF_VERIFICATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"verify\s+(this\s+)?proof", re.IGNORECASE),
    re.compile(r"check\s+(this\s+)?proof", re.IGNORECASE),
    re.compile(r"proof\s+check", re.IGNORECASE),
    re.compile(r"find\s+the\s+flaw", re.IGNORECASE),
    re.compile(r"(is|are)\s+(this|the)\s+proof", re.IGNORECASE),
    # More specific: requires "proof" or "step" in context, not standalone
    # PERFORMANCE: Non-greedy (.*?) prevents catastrophic backtracking
    re.compile(r"proof.*?step\s+\d+", re.IGNORECASE),  # "proof: step 1"
    re.compile(r"step\s+\d+.*?proof", re.IGNORECASE),  # "step 1 of proof"
)


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
    if not SAFE_EXECUTION_AVAILABLE:
        # Provide more informative message about what's missing
        logger.warning(
            "Safe execution not fully available for mathematical computation. "
            "Complex symbolic math (summations, integrals, etc.) will be limited. "
            "Install RestrictedPython and SymPy for full functionality: "
            "pip install RestrictedPython sympy"
        )
except ImportError:
    SAFE_EXECUTION_AVAILABLE = False
    execute_math_code = None
    logger.warning(
        "Safe execution module not available for mathematical computation. "
        "Install RestrictedPython and SymPy for full functionality: "
        "pip install RestrictedPython sympy"
    )


# ============================================================================
# UNICODE EXPRESSION EXTRACTION (Issue #2 Fix)
# ============================================================================


def extract_math_expression(query: str) -> Optional[str]:
    """
    Extract mathematical expressions from query text.
    
    Note: Enhanced to handle more mathematical notation patterns.
    
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
    
    # BUG #1 FIX: Preprocessing step - collapse multi-line Unicode math expressions
    # Multi-line Unicode expressions like:
    #   ∑_{k=1}^n
    #   (2k−1)
    # Need to be collapsed into single line: ∑_{k=1}^n (2k−1)
    # This handles fragmented math expressions that span multiple lines
    query = query.replace('\n', ' ').replace('\r', ' ')
    # Collapse multiple spaces
    query = re.sub(r'\s+', ' ', query)
    
    # Note: Pattern 0 - Check for "Compute exactly:" pattern first
    # This catches: "Compute exactly: ∑(k=1 to n)(2k−1)"
    compute_exact_pattern = r'(?:Compute|Calculate|Evaluate|Find|Solve)\s+(?:exactly|precisely)?\s*:?\s*(.+?)(?:\.|$|Then|Task|Verify)'
    match = re.search(compute_exact_pattern, query, re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Check if it contains math symbols
        math_indicators = ['∑', '∏', '∫', '√', 'π', '+', '-', '*', '/', '^', '=', '(', ')']
        if any(ind in candidate for ind in math_indicators):
            logger.debug(f"[MathTool] Note: Extracted via 'Compute exactly' pattern: {candidate}")
            return candidate
    
    # Note: Pattern 1 - Unicode math symbols with improved character set
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
            logger.debug(f"[MathTool] Note: Extracted via Unicode pattern: {extracted}")
            return extracted
    
    # Bug #2 FIX: Pattern 1a - ∑(expression) from index=lower to upper
    # Matches: "∑(2k-1) from k=1 to n"
    sum_from_pattern = r'∑\s*\(([^)]+)\)\s+from\s+(\w+)\s*=\s*(\d+)\s+to\s+(\w+)'
    match = re.search(sum_from_pattern, query, re.IGNORECASE)
    if match:
        expr, index, lower, upper = match.groups()
        # Convert expression to SymPy-compatible format
        expr_clean = expr.strip()
        expr_clean = expr_clean.replace('−', '-')  # Unicode minus to ASCII
        expr_clean = re.sub(r'(\d)([a-z])', r'\1*\2', expr_clean)  # 2k → 2*k
        result = f"summation({expr_clean}, ({index}, {lower}, {upper}))"
        logger.debug(f"[MathTool] Bug #2 FIX: Converted ∑(expr) from pattern to: {result}")
        return result
    
    # Note: Pattern 1b - Natural language sum/product
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
        logger.debug(f"[MathTool] Note: Converted natural sum to: {result}")
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
    # FIX (Jan 8 2026): Removed short keywords like "diff" that cause false positives
    # - "diff" was matching "different" in "what makes you different from other AI?"
    # - "differentiate" and "derivative" are sufficient for calculus detection
    # - Word boundary matching only applies to keywords <= 3 chars, so "diff" (4 chars)
    #   was being matched as a substring
    PROBLEM_PATTERNS: Dict[ProblemType, List[str]] = {
        ProblemType.CALCULUS: [
            "integrate", "integral", "derivative", "differentiate",
            # FIX: Removed "diff" - causes false positives like "different"
            # Use "d/dx" or full words "differentiate"/"derivative" instead
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
        # ROOT CAUSE FIX: Use word boundary matching for ALL keywords
        # to avoid false positives like "solve riddle" or "integrate feedback"
        scores: Dict[ProblemType, Tuple[int, List[str]]] = {}
        
        for problem_type, keywords in self.PROBLEM_PATTERNS.items():
            matches = []
            for kw in keywords:
                # ROOT CAUSE FIX: Use word boundary for ALL keywords
                # Previously only used for keywords <= 3 chars
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, query_lower):
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
        
        # ROOT CAUSE FIX: Require at least 2 keyword matches for high confidence
        # A single match like "solve" could be "solve this riddle" (not math)
        if best_score < 2:
            # Check for actual math expressions to confirm it's mathematical
            has_math_expr = self._has_mathematical_expression(query)
            if not has_math_expr and best_score == 1:
                # Single keyword without math expression = low confidence
                best_type = ProblemType.UNKNOWN
                confidence = 0.1
            else:
                confidence = min(1.0, best_score / 3.0) if best_score > 0 else 0.1
        else:
            # Multiple keyword matches = higher confidence
            confidence = min(1.0, best_score / 3.0)
        
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
    
    def _has_mathematical_expression(self, query: str) -> bool:
        """
        ROOT CAUSE FIX: Check if query contains actual mathematical expressions.
        
        This prevents false positives from queries like "solve this riddle"
        where "solve" matches but there's no actual math content.
        """
        # Check for arithmetic expressions: 2+2, 3*4, etc.
        if re.search(r'\d+\s*[+\-*/^]\s*\d+', query):
            return True
        
        # Check for equations: x = 5, y + 2 = 10
        if re.search(r'[a-z]\s*[+\-*/^]?\s*=\s*\d+', query, re.I):
            return True
        
        # Check for mathematical notation
        if any(sym in query for sym in ['∑', '∏', '∫', '√', 'π', '∞', '±']):
            return True
        
        # Check for function calls with numbers: sin(30), log(10)
        if re.search(r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\(\s*[\d.]+', query, re.I):
            return True
        
        # Check for polynomial notation: x^2, x**2
        if re.search(r'[a-z]\s*[\^*]{1,2}\s*\d+', query, re.I):
            return True
        
        return False
    
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
        # BUG #8 FIX: Define common variables in preamble
        # When processing queries with mathematical components (integral ∫u(t)²dt),
        # ensure all variables are defined to avoid "name 'x' is not defined" errors
        if bounds:
            return f"""# Definite Integration
# BUG #8 FIX: Define common variables
x, t, u, E = symbols('x t u E')
E_safe = Symbol('E_safe', positive=True)
{variable} = Symbol('{variable}')
f = {expression}
result = integrate(f, ({variable}, {bounds[0]}, {bounds[1]}))
"""
        return f"""# Indefinite Integration
# BUG #8 FIX: Define common variables
x, t, u, E = symbols('x t u E')
E_safe = Symbol('E_safe', positive=True)
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
                # ISSUE #3 FIX: Detect proof verification requests
                # The math tool is designed for computation, not proof verification
                # Proof verification queries should route to symbolic reasoner instead
                is_proof_verification = self._is_proof_verification_query(query)
                if is_proof_verification:
                    logger.info(
                        f"[MathTool] ISSUE #3 FIX: Detected proof verification request. "
                        f"Math tool is for computation, not proof verification."
                    )
                    return self._create_proof_verification_error(
                        query, time.time() - start_time
                    )
                
                # Step 1: Classify the problem
                classification = self._classifier.classify(query)
                logger.debug(f"Problem classified as {classification.problem_type.value} "
                           f"with confidence {classification.confidence:.2f}")
                
                # Step 2: Determine strategy
                strategy = strategy_override or classification.suggested_strategy
                if self.prefer_templates and strategy == SolutionStrategy.LLM_GENERATED:
                    strategy = SolutionStrategy.TEMPLATE
                
                # Step 3: Generate code
                code = self._generate_code(query, classification, strategy, llm, **kwargs)
                
                # Note: Handle None return when no math expression found
                # Previously this assumed code was always a string. Now _generate_code
                # returns None when no mathematical content is detected, which means
                # the math engine should gracefully decline rather than compute garbage.
                if not code or not code.strip():
                    logger.info(
                        f"[MathTool] Note: No mathematical expression found in query. "
                        f"Returning failure result instead of computing default expression."
                    )
                    return self._create_error_result(
                        query, "", "No mathematical expression found in query", 
                        classification, strategy, time.time() - start_time
                    )

                # Step 4: Execute the code
                if not SAFE_EXECUTION_AVAILABLE or execute_math_code is None:
                    # Note: Try simple arithmetic fallback before giving up
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

                # FIX Issue #2: Implement retry loop with error feedback
                # When code generation fails with syntax errors, retry with error feedback
                MAX_RETRIES = 3
                for attempt in range(MAX_RETRIES):
                    execution_result = execute_math_code(code)
                    
                    if execution_result["success"]:
                        # Success - break out of retry loop
                        break
                    
                    # Execution failed
                    error_msg = execution_result["error"]
                    logger.info(f"Code execution failed (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}")
                    
                    # If not the last attempt and LLM is available, try to correct the code
                    if attempt < MAX_RETRIES - 1 and llm is not None:
                        corrected_code = self._request_code_correction(
                            query, code, error_msg, llm
                        )
                        if corrected_code and corrected_code != code:
                            logger.info(f"Attempting retry with corrected code (attempt {attempt + 2}/{MAX_RETRIES})")
                            code = corrected_code
                            continue
                    
                    # Last attempt or LLM not available - try fallback strategy
                    if attempt == MAX_RETRIES - 1:
                        logger.info("All retry attempts exhausted, trying fallback strategy")
                        fallback_result = self._try_fallback(
                            query, classification, strategy, llm, error_msg
                        )
                        if fallback_result and fallback_result.success:
                            return fallback_result
                        
                        return self._create_error_result(
                            query, code, error_msg,
                            classification, strategy, time.time() - start_time
                        )
                
                # If we get here without breaking, execution_result must have succeeded
                if not execution_result["success"]:
                    # Fallback for unexpected flow
                    return self._create_error_result(
                        query, code, execution_result.get("error", "Unknown error"),
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
        llm,
        **kwargs
    ) -> Optional[str]:
        """
        Generate SymPy code to solve the problem.
        
        Uses strategy-appropriate code generation method.
        
        Note: Returns None when no mathematical content is found.
        Previously, this always returned code (falling back to default expression).
        Now returns None if the query doesn't contain mathematical content,
        allowing callers to handle non-math queries appropriately.
        
        Args:
            query: Mathematical problem description
            classification: Problem classification
            strategy: Solution strategy to use
            llm: Language model for code generation (optional)
            **kwargs: Additional options passed through to code generation
        
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
            # BUG #1 FIX: Pass kwargs to _generate_llm_code to avoid NameError
            # The _generate_llm_code method references kwargs.get('skip_gate_check', False)
            # but kwargs was not in the function signature, causing:
            # "NameError: name 'kwargs' is not defined" in production logs
            # Fixed by adding **kwargs to both _generate_code and _generate_llm_code signatures
            llm_code = self._generate_llm_code(query, llm, **kwargs)
            if llm_code:
                return llm_code
        
        # Note: Fallback to template - but this can now return None
        # if no mathematical content is found in the query
        return self._generate_template_code(query, classification)

    # ============================================================================
    # NEW REQUIREMENT: Expression Parsing Helpers (Industry Standard)
    # ============================================================================
    # These helper methods extract mathematical expressions from natural language
    # queries to avoid hardcoded template defaults. They follow industry standards:
    # - Comprehensive error handling with graceful degradation
    # - Extensive regex patterns for Unicode, LaTeX, and ASCII notation
    # - Type hints and Google-style docstrings
    # - Logging for debugging and monitoring
    # - Security: Safe parsing without eval() or exec()
    # ============================================================================

    def _parse_integral_expression(self, query: str) -> Tuple[Optional[str], Optional[str], Optional[Tuple[str, str]]]:
        """
        Parse integral expression from query with industry-standard robustness.
        
        Extracts the integrand, variable of integration, and bounds (if definite).
        Handles multiple notation formats:
        - Unicode: ∫₀ᵀu(t)²dt, ∫u(t)²dt
        - LaTeX: \\int_0^T u(t)^2 dt, \\int u(t)^2 dt
        - English: "integrate u(t)^2 from 0 to T", "integral of x^2"
        - Mixed: "calculate ∫u(t)² with respect to t from 0 to T"
        
        Args:
            query: The query string containing integral expression
            
        Returns:
            Tuple of (integrand, variable, bounds) where:
            - integrand: Expression to integrate (e.g., "u(t)**2"), or None if not found
            - variable: Integration variable (e.g., "t"), or None if not found
            - bounds: Tuple of (lower, upper) for definite integrals, or None for indefinite
            
        Examples:
            >>> self._parse_integral_expression("∫₀ᵀu(t)²dt")
            ("u(t)**2", "t", ("0", "T"))
            >>> self._parse_integral_expression("integrate x^2 with respect to x")
            ("x**2", "x", None)
            >>> self._parse_integral_expression("no math here")
            (None, None, None)
        
        Industry Standards Applied:
        - Non-greedy quantifiers to prevent catastrophic backtracking
        - Multiple regex patterns for robustness
        - Graceful degradation (returns None instead of raising exceptions)
        - Security: No eval() or exec() - pure string parsing
        - Comprehensive logging for debugging
        """
        integrand = None
        variable = None
        bounds = None
        
        try:
            # Pattern 1: Unicode integral with subscript/superscript bounds
            # Matches: ∫₀ᵀu(t)²dt, ∫₁ⁿk²dk
            # Subscripts: ₀₁₂₃₄₅₆₇₈₉, Superscripts: ⁰¹²³⁴⁵⁶⁷⁸⁹
            unicode_subscript_map = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', 
                                     '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'}
            unicode_superscript_map = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
                                       '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
            
            # Pattern: ∫[bounds]expression d(variable)
            unicode_integral_pattern = re.compile(
                r'∫([₀-₉ᵀᵁᵂⁿᵏᵐ]+)?([⁰-⁹ᵀᵁᵂⁿᵏᵐ]+)?([^d∫]+?)d([a-zA-Z])',
                re.UNICODE
            )
            
            match = unicode_integral_pattern.search(query)
            if match:
                lower_bound_unicode = match.group(1)
                upper_bound_unicode = match.group(2)
                integrand_raw = match.group(3).strip()
                variable = match.group(4)
                
                # Convert Unicode subscripts/superscripts to ASCII
                if lower_bound_unicode and upper_bound_unicode:
                    lower = ''.join(unicode_subscript_map.get(c, c) for c in lower_bound_unicode)
                    upper = ''.join(unicode_superscript_map.get(c, c) for c in upper_bound_unicode)
                    bounds = (lower, upper)
                
                # Normalize integrand: Convert Unicode superscripts to ** notation
                # ² → **2, ³ → **3, etc.
                integrand = integrand_raw
                for unicode_sup, ascii_num in unicode_superscript_map.items():
                    integrand = integrand.replace(unicode_sup, f'**{ascii_num}')
                
                logger.info(
                    f"[MathTool] Parsed Unicode integral: integrand={integrand}, "
                    f"variable={variable}, bounds={bounds}"
                )
                return (integrand, variable, bounds)
            
            # Pattern 2: LaTeX integral notation
            # Matches: \int_0^T u(t)^2 dt, \int_{0}^{T} f(x) dx
            latex_integral_pattern = re.compile(
                r'\\int(?:_\{?([^}^]+?)\}?)?\^?\{?([^}d\s]+?)?\}?\s*([^d\\]+?)\s*d([a-zA-Z])',
                re.IGNORECASE
            )
            
            match = latex_integral_pattern.search(query)
            if match:
                lower = match.group(1)
                upper = match.group(2)
                integrand = match.group(3).strip()
                variable = match.group(4)
                
                if lower and upper:
                    bounds = (lower.strip(), upper.strip())
                
                logger.info(
                    f"[MathTool] Parsed LaTeX integral: integrand={integrand}, "
                    f"variable={variable}, bounds={bounds}"
                )
                return (integrand, variable, bounds)
            
            # Pattern 3: English "integrate ... from ... to ..."
            # Matches: "integrate u(t)^2 from 0 to T", "integrate x^2 with respect to x from a to b"
            english_integral_pattern = re.compile(
                r'integr(?:ate|al)\s+(?:of\s+)?([^\s]+(?:\([^)]+\))?[^\s]*?)\s*'
                r'(?:with\s+respect\s+to\s+|wrt\s+|d)?([a-zA-Z])?'
                r'(?:\s+from\s+([^\s]+)\s+to\s+([^\s]+))?',
                re.IGNORECASE
            )
            
            match = english_integral_pattern.search(query)
            if match:
                integrand = match.group(1).strip()
                variable = match.group(2) if match.group(2) else None
                lower = match.group(3)
                upper = match.group(4)
                
                if lower and upper:
                    bounds = (lower.strip(), upper.strip())
                
                # If variable not found in "with respect to", try to infer from integrand
                if not variable:
                    # Look for common variable names in integrand
                    var_match = re.search(r'([a-zA-Z])\(', integrand)
                    if var_match:
                        variable = var_match.group(1)
                    else:
                        # Find single-letter variables
                        vars_in_expr = re.findall(r'\b([a-z])\b', integrand.lower())
                        if vars_in_expr:
                            variable = vars_in_expr[0]
                
                logger.info(
                    f"[MathTool] Parsed English integral: integrand={integrand}, "
                    f"variable={variable}, bounds={bounds}"
                )
                return (integrand, variable, bounds)
            
            # Pattern 4: Bare integral symbol with expression after it
            # Matches: "∫ x^2", "calculate ∫ sin(x)"
            if '∫' in query:
                # Find text after ∫ up to common delimiters
                integral_pos = query.find('∫')
                after_integral = query[integral_pos + 1:].strip()
                
                # Extract expression (up to "from", "with", or end of meaningful content)
                expr_match = re.match(r'^([^,\.\?!]+?)(?:\s+(?:from|with|for)\b|$)', after_integral)
                if expr_match:
                    integrand = expr_match.group(1).strip()
                    
                    # Try to find variable
                    var_match = re.search(r'd([a-zA-Z])\b', after_integral)
                    if var_match:
                        variable = var_match.group(1)
                    else:
                        # Infer from integrand
                        vars_in_expr = re.findall(r'\b([a-z])\b', integrand.lower())
                        if vars_in_expr:
                            variable = vars_in_expr[0]
                    
                    logger.info(
                        f"[MathTool] Parsed bare integral: integrand={integrand}, variable={variable}"
                    )
                    return (integrand, variable, None)
        
        except Exception as e:
            # Industry standard: Log but don't fail - graceful degradation
            logger.warning(
                f"[MathTool] Exception while parsing integral expression: {e}. "
                f"Falling back to None (will use defaults)."
            )
            return (None, None, None)
        
        # No pattern matched
        logger.debug(f"[MathTool] Could not parse integral expression from query: {query[:100]}...")
        return (None, None, None)

    def _normalize_math_expression(self, expr: str) -> str:
        """
        Normalize mathematical expression for SymPy compatibility.
        
        Converts various notations to SymPy-compatible format:
        - Implicit multiplication: 2x → 2*x
        - Power notation: x^2 → x**2
        - Unicode symbols: π → pi
        - Function call spacing: sin x → sin(x)
        
        Args:
            expr: Raw mathematical expression string
            
        Returns:
            Normalized expression suitable for SymPy
            
        Examples:
            >>> self._normalize_math_expression("2x^2")
            "2*x**2"
            >>> self._normalize_math_expression("u(t)²")
            "u(t)**2"
        
        Industry Standards:
        - Non-destructive: Returns original if normalization fails
        - Idempotent: Normalizing twice gives same result
        - Secure: No eval() or exec()
        """
        try:
            normalized = expr
            
            # Convert ^ to ** for powers (but not in function names)
            normalized = re.sub(r'\^', '**', normalized)
            
            # Add implicit multiplication: 2x → 2*x, but not for function calls like sin(x)
            # Matches digit followed by letter (not in a function call context)
            normalized = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', normalized)
            
            # Add implicit multiplication: )( → )*(
            normalized = re.sub(r'\)\s*\(', ')*(', normalized)
            
            # Replace common Unicode symbols
            unicode_replacements = {
                'π': 'pi',
                '∞': 'oo',
                '√': 'sqrt',
                'ℯ': 'E',
                'Σ': 'Sum',
                '∏': 'Product',
            }
            for unicode_sym, sympy_name in unicode_replacements.items():
                normalized = normalized.replace(unicode_sym, sympy_name)
            
            return normalized
        
        except Exception as e:
            logger.warning(f"[MathTool] Failed to normalize expression '{expr}': {e}. Using original.")
            return expr

    def _sanitize_sympy_expression(self, expr: str) -> str:
        """
        Sanitize expression for safe SymPy execution.
        
        Removes potentially dangerous constructs while preserving mathematical content.
        This is a defense-in-depth measure since RestrictedPython already provides
        sandboxing, but we add extra validation for expressions directly from users.
        
        Args:
            expr: Expression to sanitize
            
        Returns:
            Sanitized expression safe for SymPy
            
        Security Standards:
        - No imports or __builtins__ access
        - No file operations
        - No system calls
        - Whitelist approach: Only allow mathematical operations
        """
        # Remove any import statements
        if 'import' in expr.lower():
            logger.warning(f"[MathTool] Blocked expression with 'import': {expr[:50]}...")
            return "x**2"  # Safe fallback
        
        # Remove dangerous functions
        dangerous_patterns = ['__', 'eval', 'exec', 'compile', 'open', 'file']
        for pattern in dangerous_patterns:
            if pattern in expr.lower():
                logger.warning(f"[MathTool] Blocked expression with dangerous pattern '{pattern}': {expr[:50]}...")
                return "x**2"  # Safe fallback
        
        return expr

    def _generate_template_code(self, query: str, classification: ProblemClassification) -> Optional[str]:
        """
        Generate code using templates based on problem classification.
        
        Uses a priority-based matching system where more specific patterns
        are checked before general ones to ensure correct template selection.
        
        Note: Returns None when no mathematical expression is found.
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
        # FIX (Issue #1): Check for EXPLICIT mathematical expressions FIRST
        # =================================================================
        # Before checking logic patterns to reject queries, we must first check
        # if the query contains explicit mathematical notation that we CAN handle.
        # Queries with ∑, ∫, or explicit "compute/calculate" + math content
        # should proceed to mathematical processing even if they contain some
        # logic-related words like "verify" or "prove".
        #
        # Example that was broken:
        #   "Compute exactly: ∑(2k-1) from k=1 to n, then verify by induction"
        #   - Contains ∑ (summation symbol) → SHOULD process mathematically
        #   - Contains "verify" → Was being rejected as logic pattern
        #
        # Fix: Detect explicit mathematical notation and bypass logic rejection.
        # Uses module-level constants EXPLICIT_MATH_SYMBOLS and EXPLICIT_MATH_KEYWORDS
        # for better performance (avoid recreating lists on every call).
        # =================================================================
        has_explicit_math = (
            any(sym in query for sym in EXPLICIT_MATH_SYMBOLS) or
            any(kw in query_lower for kw in EXPLICIT_MATH_KEYWORDS)
        )
        
        if has_explicit_math:
            logger.info(
                f"[MathTool] Explicit mathematical notation detected (∑, ∫, etc.) - "
                f"proceeding with mathematical processing. Query: {query[:80]}..."
            )
            # Skip logic pattern rejection - this is explicit math
        else:
            # =================================================================
            # FIX Issue #3: Check for MATH PROOF indicators BEFORE rejecting
            # =================================================================
            # Problem: Queries like "Verify this calculus proof" were being rejected
            # as "logic patterns" because they contain "verify proof".
            # 
            # Solution: Check for mathematical proof indicators (calculus, analysis,
            # continuity, limits, derivatives, etc.) BEFORE rejecting as logic.
            # Only reject queries that are PURE logic without math content.
            # =================================================================
            
            math_proof_indicators = [
                # Calculus and analysis
                'differentiable', 'continuous', 'continuity', 'limit', 'lim',
                'derivative', 'integral', 'convergence', 'divergence',
                # Mathematical notation
                'f(x)', 'g(x)', 'x→', 'd/dx', 'dy/dx',
                # Mathematical verification (NOT logic verification)
                'mathematical verification', 'calculus proof', 'analysis proof',
                'limit proof', 'derivative proof', 'integral proof',
                # Mathematical concepts
                'epsilon', 'delta', 'epsilon-delta',
                'cauchy', 'weierstrass', 'riemann',
                # General mathematical context
                'calculus', 'analysis', 'topology', 'measure theory',
            ]
            
            # Check if this is a mathematical proof verification
            has_math_proof_indicators = any(
                indicator in query_lower for indicator in math_proof_indicators
            )
            
            if has_math_proof_indicators:
                logger.info(
                    f"[MathTool] FIX Issue #3: Detected mathematical proof indicators - "
                    f"proceeding with verification despite 'proof' keyword. Query: {query[:80]}..."
                )
                # Continue to mathematical processing - don't reject
            else:
                # =================================================================
                # Note (CRITICAL): PRIORITY 0 - Reject non-mathematical queries
                # =================================================================
                # These patterns indicate queries that should NOT be processed by the
                # math engine, even if they contain math-related words like "proof"
                # or "function". Return None early to prevent nonsensical output.
                #
                # Examples:
                # - "Is {A→B, B→C, ¬C, A∨B} satisfiable?" -> Logic query, not math
                # - "Verify proof about predicate logic" -> Logic verification, not math
                # - "Formalize in FOL" -> First-order logic, not math
                # =================================================================
                # ROOT CAUSE FIX: Expanded logic patterns to catch more cases
                logic_patterns = [
                    # Propositional logic symbols
                    '→', '∧', '∨', '¬', '⊢', '⊨', '⇒', '⇔',
                    # Propositional logic keywords
                    'satisfiable', 'unsatisfiable', 'tautology', 'contradiction',
                    'propositional', 'boolean', 'truth table', 'truth value',
                    # First-order logic symbols
                    '∀', '∃',
                    # First-order logic keywords
                    'forall', 'exists', 'formalize', 'fol', 
                    'first-order logic', 'first order logic',
                    'predicate logic', 'quantifier', 'universal quantifier', 'existential',
                    # FIX Issue #3: Removed overly broad patterns to avoid rejecting math proofs
                    # Removed: 'prove that', 'verify proof', 'check proof', 'theorem', 'axiom'
                    # These are kept ONLY in logic-specific context below
                    'logic proof', 'logical proof', 'proof of logic',
                    'theorem in logic', 'axiom of logic', 'logical axiom',
                    # Logic validation - ROOT CAUSE FIX: Added
                    'logically valid', 'logically invalid', 'logical validity',
                    'logical soundness', 'logical completeness',
                    'entails', 'logically follows',
                    # Model theory - ROOT CAUSE FIX: Added
                    'interpretation', 'model of', 'satisfies',
                    'consistent', 'inconsistent', 'consistency',
                    # SAT/SMT
                    'sat problem', 'sat solver', 'smt',
                    # Self-introspection queries (not math!)
                    'what makes you', 'who are you', 'are you', 'your capabilities',
                    'how do you', 'your reasoning', 'your architecture',
                ]
                
                if any(pattern in query_lower for pattern in logic_patterns):
                    logger.info(
                        f"[MathTool] Note: Query contains logic patterns, not mathematical. "
                        f"Declining to compute. Query: {query[:80]}..."
                    )
                    return None
                
                # Also check for logic symbols in original query (case-sensitive)
                if any(sym in query for sym in ['→', '∧', '∨', '¬', '∀', '∃', '⊢', '⊨', '⇒', '⇔']):
                    logger.info(
                        f"[MathTool] Note: Query contains logic symbols. "
                        f"Declining to compute. Query: {query[:80]}..."
                    )
                    return None
        
        # =================================================================
        # BUG #2 FIX: PRIORITY 0.5 - Simple arithmetic expressions
        # =================================================================
        # Before checking complex patterns, detect simple arithmetic like:
        # - "What is 2+2?" -> 2+2
        # - "3*4" -> 3*4  
        # - "10/2" -> 10/2
        # - "(2+3)*4" -> (2+3)*4
        # These should generate straightforward evaluation code.
        # =================================================================
        simple_result = self._extract_simple_arithmetic(query)
        if simple_result is not None:
            logger.info(
                f"[MathTool] Simple arithmetic detected: {simple_result}"
            )
            return f"result = {simple_result}"
        
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
            # Note: `re` module is imported at top of file
            
            # Try to extract expression and bounds
            # BUG #1 FIX: Updated pattern to handle fragmented multi-line Unicode
            # Handles multi-line expressions like:
            #   ∑_{k=1}^n
            #   (2k−1)
            # Pattern captures: (index, lower, upper, expression)
            sum_pattern = r'∑[\s\n]*_?[\s\n]*\{?[\s\n]*([\w𝑘𝑛]+)[\s\n]*=[\s\n]*(\d+)[\s\n]*\}?[\s\n]*\^?[\s\n]*\{?[\s\n]*([\w𝑛]+)[\s\n]*\}?[\s\n]*\(?([^)\n]+)\)?'
            sum_match = re.search(sum_pattern, query)
            if sum_match:
                index = sum_match.group(1).replace('𝑘', 'k').replace('𝑛', 'n')  # e.g., "k"
                lower = sum_match.group(2)  # e.g., "1"
                upper = sum_match.group(3).replace('𝑛', 'n')  # e.g., "n"
                expr = sum_match.group(4).strip()  # e.g., "2k-1"
                # BUG #1 FIX: Convert Unicode characters to ASCII
                expr = expr.replace('−', '-')  # Unicode minus
                expr = expr.replace('𝑘', 'k')  # Math italic k
                expr = expr.replace('𝑛', 'n')  # Math italic n
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)  # 2k → 2*k
                return self._templates.summation(expr, index, lower, upper)
            
            # Bug #2 FIX Pattern 1.5: ∑(expression) from index=lower to upper
            # Handles queries like: "Compute ∑(2k-1) from k=1 to n"
            # SUMMATION FIX: Normalize Unicode minus before regex to ensure matching
            query_normalized = query.replace('−', '-')  # Unicode minus → ASCII
            sum_match1_5 = re.search(r'∑\s*\(([^)]+)\)\s+from\s+(\w+)\s*=\s*(\d+)\s+to\s+(\w+)', query_normalized, re.IGNORECASE)
            if sum_match1_5:
                expr = sum_match1_5.group(1).strip()  # e.g., "2k-1"
                index = sum_match1_5.group(2)  # e.g., "k"
                lower = sum_match1_5.group(3)  # e.g., "1"
                upper = sum_match1_5.group(4)  # e.g., "n"
                # Convert to SymPy format (2k → 2*k)
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)  # 2k → 2*k
                logger.info(f"[MathTool] SUMMATION FIX: Matched ∑(expr) from {index}={lower} to {upper} pattern: expr={expr}")
                return self._templates.summation(expr, index, lower, upper)
            
            # SUMMATION FIX Pattern 1.6: Compute/Calculate ∑(expression) 
            # Even more flexible pattern for "Compute ∑(2k-1)" style queries
            # Captures expression from ∑(...) anywhere in the query
            sum_match1_6 = re.search(r'∑\s*\(([^)]+)\)', query_normalized)
            if sum_match1_6:
                expr = sum_match1_6.group(1).strip()  # e.g., "2k-1"
                # Convert to SymPy format (2k → 2*k)
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)  # 2k → 2*k
                # Try to extract bounds from "from k=1 to n" or "k=1 to n" anywhere
                bounds_match = re.search(r'(?:from\s+)?(\w+)\s*=\s*(\d+)\s+to\s+(\w+)', query_normalized, re.IGNORECASE)
                if bounds_match:
                    index = bounds_match.group(1)
                    lower = bounds_match.group(2)
                    upper = bounds_match.group(3)
                    logger.info(f"[MathTool] SUMMATION FIX: Pattern 1.6 matched ∑({expr}) with bounds {index}={lower} to {upper}")
                    return self._templates.summation(expr, index, lower, upper)
                else:
                    # Default bounds if not specified
                    logger.info(f"[MathTool] SUMMATION FIX: Pattern 1.6 matched ∑({expr}) with default bounds k=1 to n")
                    return self._templates.summation(expr, "k", "1", "n")
            
            # Pattern 2: sum from k=lower to upper of expression
            sum_match2 = re.search(r'sum(?:mation)?\s+(?:from\s+)?(\w+)\s*=\s*(\d+)\s+to\s+(\w+)\s+(?:of\s+)?(.+)', query_lower)
            if sum_match2:
                index = sum_match2.group(1)
                lower = sum_match2.group(2)
                upper = sum_match2.group(3)
                expr = sum_match2.group(4).strip()
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)
                return self._templates.summation(expr, index, lower, upper)
            
            # SUMMATION FIX: Try to extract expression from query before falling back
            # Look for any expression after ∑ that might be the summand
            # Pattern explanation:
            #   ∑             - Summation symbol
            #   [\s_{}^0-9nk]* - Optional subscript/superscript notation (whitespace, {}, ^, digits, n, k)
            #   \(?           - Optional opening parenthesis
            #   ([0-9]+[a-z][\s\-+*/0-9a-z]*) - Capture: digit(s)+letter followed by math expression
            #   \)?           - Optional closing parenthesis
            # Examples matched: "∑(2k-1)", "∑_{k=1}^n 2k-1", "∑(3n+2)"
            fallback_expr_match = re.search(r'∑[\s_{}^0-9nk]*\(?([0-9]+[a-z][\s\-+*/0-9a-z]*)\)?', query_normalized)
            if fallback_expr_match:
                expr = fallback_expr_match.group(1).strip()
                expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)
                logger.info(f"[MathTool] SUMMATION FIX: Fallback pattern matched expression: {expr}")
                return self._templates.summation(expr, "k", "1", "n")
            
            # Default summation - only if no expression could be extracted
            # MAX_QUERY_LOG_LENGTH: Truncate long queries in log messages for readability
            MAX_QUERY_LOG_LENGTH = 100
            logger.warning(f"[MathTool] SUMMATION: No expression found, using default 'k'. Query: {query[:MAX_QUERY_LOG_LENGTH]}...")
            return self._templates.summation("k", "k", "1", "n")
        
        # PRIORITY 4: Limits (check before integration since both are calculus)
        if "limit" in query_lower:
            direction = "+" if "right" in query_lower else ("-" if "left" in query_lower else "")
            return self._templates.limit("sin(x)/x", var, "0", direction)
        
        # PRIORITY 5: Series expansion
        if any(kw in query_lower for kw in ["series", "taylor", "maclaurin", "expansion", "expand around"]):
            return self._templates.series_expansion("exp(x)", var, "0", 5)
        
        # PRIORITY 6: Integration
        # NEW REQUIREMENT FIX: Parse actual expression from query instead of hardcoded "x**2"
        if any(kw in query_lower for kw in ["integrate", "integral", "antiderivative", "∫"]):
            # Parse the integral expression from the query
            integrand_parsed, var_parsed, bounds_parsed = self._parse_integral_expression(query)
            
            # Use parsed values if available, otherwise fall back to defaults
            integrand = integrand_parsed if integrand_parsed else "x**2"
            integration_var = var_parsed if var_parsed else var
            
            # Normalize and sanitize the expression for SymPy
            if integrand_parsed:
                integrand = self._normalize_math_expression(integrand)
                integrand = self._sanitize_sympy_expression(integrand)
                logger.info(
                    f"[MathTool] NEW REQUIREMENT FIX: Parsed integral expression from query. "
                    f"integrand={integrand}, variable={integration_var}, bounds={bounds_parsed}"
                )
            else:
                logger.warning(
                    f"[MathTool] Could not parse integral expression from query. "
                    f"Falling back to default: integrand=x**2, variable={var}"
                )
            
            # Generate template code with parsed or default values
            if bounds_parsed:
                return self._templates.integration(integrand, integration_var, bounds_parsed)
            elif "definite" in query_lower or "from" in query_lower:
                # Query mentions definite integral but we couldn't parse bounds
                # Use defaults for bounds
                return self._templates.integration(integrand, integration_var, ("0", "1"))
            else:
                # Indefinite integral
                return self._templates.integration(integrand, integration_var)
        
        # PRIORITY 7: Differentiation
        # Note: Use word-boundary matching for short keywords like "diff"
        # to avoid matching "differentiable" or "difference"
        # - "differentiate" and "derivative" are long enough to be safe
        # - "diff" needs word boundary check
        # - "d/dx" is specific notation and safe
        differentiation_keywords = ["differentiate", "derivative", "d/dx"]
        has_diff_keyword = any(kw in query_lower for kw in differentiation_keywords)
        
        # Check for "diff" as a standalone word (not part of "differentiable", "difference", etc.)
        # Note: `re` module is imported at top of file
        if not has_diff_keyword:
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
        
        # Note: Return None when no mathematical content is found
        # Previously this returned "x**2 + 2*x + 1" as a default, which caused
        # irrelevant math output (the expansion of (x+1)²) to be included in
        # responses to non-mathematical queries. The math engine should NOT
        # compute anything when no math expression is found in the query.
        logger.warning(
            f"[MathTool] Note: No mathematical expression found in query. "
            f"Returning None instead of default expression. Query: {query[:100]}..."
        )
        return None

    def _generate_llm_code(self, query: str, llm, **kwargs) -> Optional[str]:
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
        
        ROOT CAUSE FIX: Added pre-flight check to reject non-mathematical queries
        before asking LLM to generate code. Prevents LLM from generating
        garbage code for queries like "What makes you different?"
        
        Args:
            query: The mathematical problem to solve
            llm: LLM client object (NOT a model name string)
            **kwargs: Additional options (e.g., skip_gate_check)
            
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
        
        # ROOT CAUSE FIX: Pre-flight check for mathematical content
        # Prevents LLM from generating code for non-mathematical queries
        # 
        # INDUSTRY-STANDARD FIX: Allow LLM code generation when skip_gate_check is set
        # This happens when the router's LLM classifier has high confidence (≥0.8)
        # that this is a mathematical query. We trust the LLM in this case.
        skip_gate_check = kwargs.get('skip_gate_check', False)
        
        if not skip_gate_check and not self._is_genuinely_mathematical(query):
            logger.info(
                f"[MathTool] LLM code generation declined - query is not mathematical: "
                f"{query[:80]}..."
            )
            return None
        elif skip_gate_check:
            logger.info(
                f"[MathTool] Gate check SKIPPED for LLM code generation - "
                f"trusting router's high-confidence classification"
            )
        
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
        """
        Remove markdown formatting, import statements, and fix common syntax issues
        in generated code.
        
        Bug #2 FIX (Jan 9 2026): Added preprocessing to handle implicit multiplication
        in LLM-generated code. The LLM may generate expressions like "2k-1" which is
        invalid Python syntax (should be "2*k-1").
        """
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

        code = "\n".join(clean_lines).strip()
        
        # =================================================================
        # Bug #2 FIX: Preprocess mathematical expressions for Python syntax
        # =================================================================
        # LLM may generate code with implicit multiplication (e.g., "2k" or "3n")
        # which is invalid Python. Convert to explicit multiplication (e.g., "2*k").
        #
        # The error that prompted this fix:
        #   Query: "Compute ∑(2k-1) from k=1 to n"
        #   LLM generates: expr = 2k-1  (invalid Python)
        #   RestrictedPython: SyntaxError: invalid syntax at statement: '-'
        #
        # This preprocessing catches ALL mathematical notation issues:
        # - 2k → 2*k (number followed by variable)
        # - 2(x+1) → 2*(x+1) (number followed by parenthesis)
        # - Unicode minus − → ASCII minus - (U+2212 to U+002D)
        # - Unicode symbols 𝑘, 𝑛 → ASCII k, n
        # =================================================================
        
        # =================================================================
        # BUG #1 FIX: Comprehensive Unicode normalization
        # =================================================================
        # Unicode normalization for mathematical symbols
        code = code.replace('−', '-')  # Unicode minus → ASCII minus (U+2212 → U+002D)
        code = code.replace('×', '*')  # Unicode multiplication → ASCII asterisk
        code = code.replace('÷', '/')  # Unicode division → ASCII slash
        code = code.replace('–', '-')  # En dash → ASCII minus
        code = code.replace('—', '-')  # Em dash → ASCII minus
        
        # Math italic variables → ASCII variables
        code = code.replace('𝑘', 'k')  # Math italic k → ASCII k
        code = code.replace('𝑛', 'n')  # Math italic n → ASCII n
        code = code.replace('𝑥', 'x')  # Math italic x → ASCII x
        code = code.replace('𝑦', 'y')  # Math italic y → ASCII y
        code = code.replace('𝑧', 'z')  # Math italic z → ASCII z
        code = code.replace('𝑖', 'i')  # Math italic i → ASCII i
        code = code.replace('𝑗', 'j')  # Math italic j → ASCII j
        code = code.replace('𝑡', 't')  # Math italic t → ASCII t
        code = code.replace('𝑢', 'u')  # Math italic u → ASCII u
        code = code.replace('𝑣', 'v')  # Math italic v → ASCII v
        code = code.replace('𝑤', 'w')  # Math italic w → ASCII w
        
        # Add implicit multiplication operator
        # Pattern: digit followed by letter (2k → 2*k)
        code = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', code)
        
        # Pattern: digit followed by opening parenthesis (2(x+1) → 2*(x+1))
        code = re.sub(r'(\d)\(', r'\1*(', code)
        
        # Pattern: closing paren followed by opening paren ()(x+1) → )*(x+1))
        # This handles cases like (k+1)(k+2)
        code = re.sub(r'\)\(', r')*(', code)
        
        return code

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

    def _extract_simple_arithmetic(self, query: str) -> Optional[str]:
        """
        BUG #2 FIX: Extract simple arithmetic expression from query.
        
        Returns the expression string if the query is simple arithmetic,
        None otherwise. This is used by _generate_code to generate
        straightforward evaluation code.
        
        Supported:
        - Basic operations: +, -, *, /, **, %
        - Parentheses: (2+3)*4
        - Decimal numbers: 3.14 * 2
        - "What is X" style questions
        
        Returns:
            Expression string if valid arithmetic, None otherwise
        """
        # Extract mathematical expression from common question patterns
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
        
        # Security check: Only allow safe characters
        allowed_pattern = r'^[\d\.\+\-\*\/\%\(\)\s]+$'
        if not re.match(allowed_pattern, expression):
            return None
        
        # Validate balanced parentheses
        if expression.count('(') != expression.count(')'):
            return None
        
        # Prevent empty or trivial expressions
        if not expression or not any(c.isdigit() for c in expression):
            return None
        
        # Make sure it has an operator (not just a number)
        # Note: ** is checked via '*' being present twice consecutively
        if not any(op in expression for op in ['+', '-', '*', '/', '%']):
            return None
        
        # Validate operator positioning - prevent invalid patterns like '++', '--', '2+', '+3'
        # Allow: leading minus for negative numbers, ** for exponentiation
        expression_no_spaces = expression.replace(' ', '')
        
        # Check for invalid consecutive operators (except ** for exponentiation)
        invalid_patterns = ['++', '+-', '+/', '+%', '-+', '--', '-/', '-%', 
                           '/+', '/-', '//', '/%', '%+', '%-', '%/', '%%',
                           '*+', '*-', '*/', '*%']  # Note: ** is valid (exponent)
        for pattern in invalid_patterns:
            if pattern in expression_no_spaces:
                return None
        
        # Check for trailing operators
        if expression_no_spaces and expression_no_spaces[-1] in '+-*/%':
            return None
        
        # Check for operators at start (except leading minus for negative numbers)
        if expression_no_spaces and expression_no_spaces[0] in '+*/%':
            return None
        
        return expression

    def _try_simple_arithmetic(self, query: str) -> Optional[Union[int, float]]:
        """
        Note: Try to evaluate simple arithmetic expressions.
        
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
        # Note: `re` module is imported at top of file
        
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

    def _is_genuinely_mathematical(self, query: str) -> bool:
        """
        ROOT CAUSE FIX: Check if query is ACTUALLY mathematical.
        
        This prevents generating code for queries like:
        - "What makes you different from other AI systems?"
        - "Can you solve this riddle?"
        - "Integrate this feedback into your response"
        
        Issue #4 FIX: Don't reject mathematical verification queries.
        Queries like "Mathematical Verification - Proof check" ARE mathematical.
        Check for mathematical verification patterns BEFORE rejecting based on "proof".
        
        Returns:
            True if query contains genuine mathematical content,
            False if it's a non-mathematical query that happens to contain math words.
        """
        query_lower = query.lower()
        
        # Issue #4 FIX: Check for mathematical verification patterns FIRST
        # These indicate mathematical proof checking, which IS a mathematical task
        math_verification_patterns = [
            'mathematical verification',
            'proof check',
            'verify.*proof',
            'check.*proof',
            'step 1.*step 2',  # Multi-step verification
            'claim:',          # Mathematical claim format
            'therefore',       # Proof conclusion marker
            'hidden flaw',     # Flaw detection in proofs
        ]
        
        is_math_verification = any(
            re.search(pattern, query_lower) for pattern in math_verification_patterns
        )
        
        if is_math_verification:
            logger.info(
                f"[MathTool] Issue #4 FIX: Detected mathematical verification pattern - "
                f"treating as mathematical despite 'proof' keyword"
            )
            return True
        
        # Check for logic/proof queries (NOT mathematical computation)
        # Issue #4 FIX: Removed "proof" from blacklist when mathematical verification detected
        logic_indicators = [
            # Logic symbols
            '→', '∧', '∨', '¬', '⊢', '⊨', '∀', '∃', '⇒', '⇔',
            # Logic keywords
            'satisfiable', 'tautology', 'valid', 'entails',
            'prove that', 'prove the', 'is the proof',
            'formalize', 'fol', 'first-order',
            # Self-introspection (definitely not math!)
            'what makes you', 'who are you', 'are you', 'your capabilities',
        ]
        if any(ind in query_lower or ind in query for ind in logic_indicators):
            return False
        
        # Check for ACTUAL math expressions
        # Arithmetic: 2+2, 3*4, etc.
        if re.search(r'\d+\s*[+\-*/^]\s*\d+', query):
            return True
        
        # Equations: x = 5, 2x + 3 = 10
        if re.search(r'[a-z]\s*[+\-*/^]?\s*=\s*[\d]', query, re.I):
            return True
        
        # Mathematical notation: ∑, ∫, √, π
        if any(sym in query for sym in ['∑', '∏', '∫', '√', 'π', '∞', '±']):
            return True
        
        # Function calls: sin(30), log(10), sqrt(16)
        if re.search(r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\(\s*[\d.]+', query, re.I):
            return True
        
        # Polynomial notation: x^2, x**2, x²
        if re.search(r'[a-z]\s*[\^*²³]{1,2}\s*\d*', query, re.I):
            return True
        
        # Derivative/integral notation: d/dx, dy/dx, ∫
        if re.search(r'd\s*/\s*d[a-z]', query, re.I):
            return True
        
        # Check for mathematical keywords with actual math content nearby
        math_keywords = ['integrate', 'derivative', 'solve', 'equation', 'limit', 'sum']
        has_math_keyword = any(re.search(r'\b' + kw + r'\b', query_lower) for kw in math_keywords)
        
        if has_math_keyword:
            # Must also have some mathematical content (numbers, variables, operators)
            has_math_content = bool(re.search(r'\d', query) or 
                                   re.search(r'\b[xyz]\b', query_lower) or
                                   re.search(r'[+\-*/^()²³]', query))
            return has_math_content
        
        # BUG FIX #2: Enhanced natural language mathematical query detection
        # These patterns recognize mathematical queries that don't use literal arithmetic syntax
        
        # Bayesian/probability problems: P(X|Y) notation with conditional bar
        # BUG A FIX: Enhanced to detect P(X|Y) conditional probability notation
        if re.search(r'P\s*\([^)]+\|[^)]+\)', query):
            logger.info("[MathTool] BUG A FIX: Detected Bayesian conditional probability P(X|Y) notation")
            return True
        
        # Also check for P(X) without conditional (simpler probability)
        if re.search(r'P\s*\([^)]+\)', query):
            return True
        
        # BUG A FIX: Expanded Bayesian/probability keywords with decimal numbers
        # Added 'bayes theorem', 'bayes rule', and standalone probability terms
        # GATE CHECK EXPANSION: Added more Bayesian/medical terminology
        bayes_keywords = [
            'bayes', "bayes'", "bayes's", 'bayesian', 'bayes theorem', 'bayes rule',
            'sensitivity', 'specificity', 'prevalence', 'posterior', 
            'prior', 'likelihood', 'conditional probability',
            'false positive', 'false negative', 'true positive', 'true negative',
            'base rate', 'predictive value'  # Added per problem statement
        ]
        if any(kw in query_lower for kw in bayes_keywords):
            if re.search(r'\d+\.\d+', query):  # Has decimal numbers
                logger.info("[MathTool] BUG A FIX: Detected Bayesian keyword with numerical data")
                return True
        
        # BUG A FIX: Natural language mathematical commands
        # These indicate explicit mathematical computation requests
        # GATE CHECK EXPANSION: Added induction patterns per problem statement
        natural_math_commands = [
            'compute exactly', 'calculate exactly', 'evaluate exactly',
            'compute the', 'calculate the', 'evaluate the',
            'show steps', 'show all steps', 'show the steps',
            'verify by induction', 'prove by induction', 'proof by induction',
            'induction proof', 'mathematical induction', 'inductive step',  # Added per problem statement
            'closed form', 'closed-form', 'closed form solution',
            'derive the formula', 'find the formula'
        ]
        if any(cmd in query_lower for cmd in natural_math_commands):
            logger.info("[MathTool] BUG A FIX: Detected natural language math command")
            return True
        
        # BUG A FIX: Enhanced mathematical notation detection with unicode handling
        # Matches patterns like "∑(2k-1) from k=1 to n", "∏ ... to ...", or "∫(expression)"
        # Also handles line breaks and fragmentation in notation expressions
        # GATE CHECK EXPANSION: Added Σ (uppercase sigma), ∏ (product), ∫ (integral)
        math_notation_symbols = ['∑', 'Σ', '∏', '∫']
        if any(sym in query for sym in math_notation_symbols):
            # Look for mathematical notation patterns: ∑...to, ∏...from...to, ∫(expression)
            if re.search(r'[∑Σ∏∫].*\bto\b', query_lower) or re.search(r'[∑Σ∏∫].*\bfrom\b.*\bto\b', query_lower):
                logger.info("[MathTool] BUG A FIX: Detected mathematical notation with bounds")
                return True
            # Also accept bare notation symbol with variables/numbers
            if re.search(r'[∑Σ∏∫]\s*[\(\[]?[a-z0-9]', query_lower):
                logger.info("[MathTool] BUG A FIX: Detected mathematical notation expression")
                return True
        
        # Mathematical verification with calculus terms
        # BUG A FIX: Enhanced to catch "verify" with broader mathematical terms
        verification_keywords = ['verify', 'proof check', 'mathematical verification', 'valid', 'invalid', 'check']
        calculus_keywords = ['differentiable', 'continuous', 'limit', 'derivative', 'integral', 'lim']
        if any(kw in query_lower for kw in verification_keywords):
            if any(calc in query_lower for calc in calculus_keywords):
                logger.info("[MathTool] BUG A FIX: Detected verification with calculus terms")
                return True
        
        # Optimization/constraint problems with mathematical notation (e.g., E > E_safe)
        if re.search(r'[A-Z]\s*[<>=]+\s*[A-Z]', query):
            return True
        
        # Optimization keywords with numerical constraints
        optimization_keywords = ['maximize', 'minimize', 'constraint', 'permissible', 'optimal']
        if any(kw in query_lower for kw in optimization_keywords):
            if re.search(r'\d', query):  # Has numbers
                return True
        
        # Calculus limit notation in natural language: lim x→a, lim as x approaches
        if re.search(r'lim.*x\s*→', query_lower) or re.search(r'lim.*as.*x.*approach', query_lower):
            return True
        
        return False

    def _request_code_correction(
        self, 
        query: str, 
        failed_code: str, 
        error_msg: str, 
        llm
    ) -> Optional[str]:
        """
        FIX Issue #2: Request LLM to correct code based on error feedback.
        
        This implements the retry loop error feedback mechanism where the LLM
        is given the original query, the failed code, and the error message,
        then asked to generate corrected code.
        
        Args:
            query: Original mathematical query
            failed_code: The code that failed to execute
            error_msg: The error message from execution
            llm: Language model for code correction
            
        Returns:
            Corrected code string, or None if correction fails
        """
        if llm is None:
            return None
        
        correction_prompt = f"""{self.CODE_GENERATION_PROMPT}

ERROR CORRECTION REQUEST:

Original Problem: {query}

Failed Code:
```python
{failed_code}
```

Error Message:
{error_msg}

Please generate CORRECTED Python code that fixes this error. Common issues to check:
1. Unclosed parentheses: '(' was never closed
2. Syntax errors: invalid syntax at statement
3. Undefined variables: name 'X' is not defined
4. Type errors: unsupported operand type(s)
5. Import errors: module imports are not allowed (use only pre-imported functions)

Generate ONLY the corrected Python code (no markdown, no explanations):"""

        try:
            corrected_code = None
            
            # Try different LLM interfaces
            if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
                response = llm.chat.completions.create(
                    model=getattr(llm, "model", DEFAULT_OPENAI_MODEL),
                    messages=[{"role": "user", "content": correction_prompt}],
                    max_tokens=self.max_tokens,
                )
                corrected_code = response.choices[0].message.content
                logger.debug("Using OpenAI chat.completions interface for correction")
                
            elif hasattr(llm, "invoke"):
                response = llm.invoke(correction_prompt)
                corrected_code = response.content if hasattr(response, "content") else str(response)
                logger.debug("Using LangChain invoke interface for correction")
                
            elif hasattr(llm, "generate"):
                corrected_code = llm.generate(correction_prompt, max_tokens=self.max_tokens)
                logger.debug("Using generate interface for correction")
                
            elif hasattr(llm, "__call__"):
                corrected_code = llm(correction_prompt)
                logger.debug("Using callable interface for correction")
            else:
                logger.warning("Unknown LLM interface for code correction")
                return None
            
            if corrected_code is None:
                logger.warning("LLM returned None response for correction")
                return None
            
            # Clean the corrected code
            return self._clean_code(corrected_code)
            
        except Exception as e:
            logger.warning(f"LLM code correction failed: {e}")
            return None

    def _try_fallback(
        self, 
        query: str, 
        classification: ProblemClassification,
        failed_strategy: SolutionStrategy,
        llm,
        error_msg: Optional[str] = None
    ) -> Optional[ComputationResult]:
        """
        FIX Issue #2: Try alternative strategy if primary fails.
        
        Updated to accept error_msg parameter for better context in fallback attempts.
        
        Args:
            query: Original query
            classification: Problem classification
            failed_strategy: Strategy that failed
            llm: Language model (optional)
            error_msg: Error message from failed execution (optional)
            
        Returns:
            ComputationResult if fallback succeeds, None otherwise
        """
        # If LLM failed, try template
        if failed_strategy in [SolutionStrategy.LLM_GENERATED, SolutionStrategy.SYMBOLIC]:
            template_code = self._generate_template_code(query, classification)
            if template_code and SAFE_EXECUTION_AVAILABLE and execute_math_code:
                result = execute_math_code(template_code)
                if result["success"]:
                    explanation = f"Solved using template approach. Result: {result['result']}"
                    if error_msg:
                        explanation += f"\n(Original error: {error_msg})"
                    
                    return ComputationResult(
                        success=True,
                        code=template_code,
                        result=str(result["result"]),
                        explanation=explanation,
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

    def _is_proof_verification_query(self, query: str) -> bool:
        """
        Detect if query is requesting proof verification.
        
        ISSUE #3 FIX: Math tool is for computation, not proof verification.
        Proof verification should route to symbolic reasoner instead.
        
        INDUSTRY STANDARD: Uses pre-compiled regex patterns for optimal performance.
        Patterns are compiled once at module level, not on every invocation.
        
        Args:
            query: The query string to check
            
        Returns:
            True if query is requesting proof verification, False otherwise
            
        Example proof verification queries:
            - "Verify this proof: ..."
            - "Check if this proof is valid"
            - "Find the flaw in this proof"
            - "Is this derivation correct?"
            - "Proof check: Step 1..."
        """
        query_lower = query.lower()
        
        # Proof verification keywords
        proof_keywords = [
            "verify", "check", "validate", "evaluate", "assess",
            "correct", "incorrect", "valid", "invalid",
            "flaw", "error", "mistake", "bug",
            "derivation", "argument",
        ]
        
        # Proof indicators
        proof_indicators = [
            "proof:", "proof check", "this proof", "the proof",
            "hidden flaw", "subtle error", "what's wrong",
        ]
        
        # Check for proof verification patterns
        has_proof_keyword = any(kw in query_lower for kw in proof_keywords)
        has_proof_indicator = any(ind in query_lower for ind in proof_indicators)
        
        # INDUSTRY STANDARD: Use pre-compiled patterns (module-level PROOF_VERIFICATION_PATTERNS)
        has_strong_pattern = any(
            pattern.search(query) for pattern in PROOF_VERIFICATION_PATTERNS
        )
        
        # Determine if this is proof verification
        is_proof_verification = (
            has_strong_pattern or
            (has_proof_keyword and has_proof_indicator)
        )
        
        if is_proof_verification:
            logger.info(
                f"[MathTool] ISSUE #3 FIX: Detected proof verification query - "
                f"strong_pattern={has_strong_pattern}, "
                f"proof_keyword={has_proof_keyword}, "
                f"proof_indicator={has_proof_indicator}"
            )
        
        return is_proof_verification

    def _create_proof_verification_error(
        self, query: str, execution_time: float
    ) -> ComputationResult:
        """
        Create error result for proof verification requests.
        
        ISSUE #3 FIX: Provide helpful error message suggesting correct routing.
        
        Args:
            query: The proof verification query
            execution_time: Time spent before declining
            
        Returns:
            ComputationResult with helpful error message
        """
        error_message = (
            "This query appears to be requesting proof verification. "
            "The mathematical computation tool is designed for computation "
            "(solving equations, computing integrals, etc.), not proof verification. "
            "\n\nFor proof verification, please use the symbolic reasoner which can: "
            "\n  • Validate logical steps in proofs"
            "\n  • Check derivation correctness"
            "\n  • Identify flaws in arguments"
            "\n  • Verify mathematical reasoning"
            "\n\nTo route to symbolic reasoner, rephrase as a logical reasoning query "
            "or use the tool selector to explicitly request symbolic reasoning."
        )
        
        explanation = (
            f"Query detected as proof verification request. "
            f"This tool handles computation, not proof checking. "
            f"Suggested routing: symbolic reasoner for logical validation."
        )
        
        logger.info(
            f"[MathTool] ISSUE #3 FIX: Declining proof verification request. "
            f"Query: {query[:100]}..."
        )
        
        return ComputationResult(
            success=False,
            code="",
            result=None,
            explanation=explanation,
            error=error_message,
            tool=self.name,
            problem_type=ProblemType.UNKNOWN,
            strategy=SolutionStrategy.SYMBOLIC,  # Suggest symbolic strategy
            execution_time=execution_time,
            metadata={
                "declined_reason": "proof_verification_not_supported",
                "suggested_tool": "symbolic_reasoner",
                "issue_fix": "ISSUE_3_proof_verification_detection",
            },
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
        
        # INDUSTRY-STANDARD FIX: Extract skip_gate_check from query dict and pass to execute
        # This allows the LLM classifier's high-confidence decisions to be honored
        kwargs = {}
        if query and isinstance(query, dict):
            if 'skip_gate_check' in query:
                kwargs['skip_gate_check'] = query['skip_gate_check']
                kwargs['router_confidence'] = query.get('router_confidence', 0.0)
                kwargs['llm_classification'] = query.get('llm_classification', 'unknown')
        
        # Execute the computation
        result = self.execute(math_query, **kwargs)
        
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
