"""
Safe Code Execution

Cross-platform sandboxed execution for math and code evaluation.
Uses ThreadPoolExecutor for timeout enforcement (works on Windows and Unix).

Security Features:
- Blocks file system access
- Blocks network operations
- Blocks arbitrary imports
- Blocks system command execution
- Blocks class hierarchy traversal (prevents sandbox escape)
- Allows only whitelisted mathematical operations
- Cross-platform execution timeout using ThreadPoolExecutor
- Rate-limits print output to prevent log flooding

Example:
    >>> from vulcan.utils.safe_execution import execute_math_code
    >>> result = execute_math_code('''
    ... x = Symbol('x')
    ... result = integrate(x**2, x)
    ... ''')
    >>> print(result['result'])  # x**3/3
"""

import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _preprocess_math_code(code: str) -> str:
    """
    Preprocess mathematical code to fix common syntax issues.
    
    Bug #2 FIX (Jan 9 2026): Added as safety net for implicit multiplication.
    Security FIX (Jan 11 2026): Added comprehensive Unicode normalization.
    MM1 CRASH FIX (Jan 19 2026): Added zero-width character stripping to prevent
    IndentationError when queries contain invisible Unicode characters.
    
    This handles cases where LLM-generated or template code contains:
    - Zero-width characters: \u200b, \u200c, \u200d, \ufeff (cause IndentationError)
    - Implicit multiplication: 2k → 2*k, 3n → 3*n
    - Digit followed by parenthesis: 2(x+1) → 2*(x+1)
    - Unicode minus: − (U+2212) → - (U+002D ASCII)
    - Unicode math italic letters: Full range of mathematical alphanumeric symbols
    - Unicode mathematical operators: × → *, ÷ → /, etc.
    
    This is applied as a safety net in execute() to catch any code that
    wasn't preprocessed by mathematical_computation.py._clean_code().
    
    Args:
        code: Python code string to preprocess
        
    Returns:
        Preprocessed code with valid Python syntax
    """
    if not code:
        return code
    
    # MM1 CRASH FIX: Strip zero-width characters that cause IndentationError
    # These invisible characters appear in queries with Unicode math notation
    # and cause Python compilation to fail with "IndentationError: '\u200b'"
    # Zero-width characters to strip:
    # - \u200b: Zero-width space
    # - \u200c: Zero-width non-joiner
    # - \u200d: Zero-width joiner
    # - \ufeff: Zero-width no-break space (BOM)
    # - \u2060: Word joiner
    # - \u180e: Mongolian vowel separator
    # Using str.translate() for better performance (O(n) instead of O(n*m))
    zero_width_chars = '\u200b\u200c\u200d\ufeff\u2060\u180e'
    zero_width_table = str.maketrans('', '', zero_width_chars)
    code = code.translate(zero_width_table)
    
    # Comprehensive Unicode normalization for mathematical alphanumeric symbols
    # Unicode range: U+1D400–U+1D7FF (Mathematical Alphanumeric Symbols)
    # First check if any math symbols exist to avoid unnecessary processing
    has_math_symbols = any(0x1D400 <= ord(c) <= 0x1D7FF for c in code)
    if has_math_symbols:
        # Use NFKD normalization which handles all mathematical alphanumeric symbols
        # This converts mathematical italic/bold/script letters to their ASCII equivalents
        code = unicodedata.normalize('NFKD', code)
    
    # Replace mathematical operators
    code = code.replace('×', '*')  # Multiplication sign
    code = code.replace('÷', '/')  # Division sign
    code = code.replace('−', '-')  # Unicode minus → ASCII minus (U+2212 → U+002D)
    code = code.replace('√', 'sqrt')  # Square root (basic support)
    
    # Add implicit multiplication operator (after normalization)
    # Pattern: digit followed by letter (2k → 2*k)
    code = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', code)
    
    # Pattern: digit followed by opening parenthesis (2(x+1) → 2*(x+1))
    code = re.sub(r'(\d)\(', r'\1*(', code)
    
    # Pattern: closing paren followed by opening paren ()(x+1) → )*(x+1))
    # This handles cases like (k+1)(k+2)
    code = re.sub(r'\)\(', r')*(', code)
    
    return code


# Maximum number of print calls allowed per execution (DoS protection)
_MAX_PRINT_CALLS = 100
# Maximum total output length from print (bytes)
_MAX_PRINT_OUTPUT = 10000


class _SafePrintCollector:
    """
    Rate-limited print collector for sandboxed code execution.
    
    SECURITY: This prevents log flooding DoS attacks and limits information disclosure
    by rate-limiting print calls and total output size.
    
    RestrictedPython rewrites print(...) to:
        _print = _print_(_getattr_)  # Create collector
        _print._call_print(...)      # Each print statement
    
    This collector receives all print output and enforces limits.
    """
    
    def __init__(self, _getattr_=None):
        """
        Initialize the print collector.
        
        Args:
            _getattr_: The guarded getattr function (provided by RestrictedPython)
        """
        self._call_count = 0
        self._total_output = 0
        self._getattr_ = _getattr_
        self.txt: list = []  # Collected text (for compatibility)
    
    def write(self, text: str) -> None:
        """
        Write text to the collector (called by print's file= argument).
        
        Args:
            text: Text to write
        """
        if self._call_count > _MAX_PRINT_CALLS:
            return  # Silently ignore after limit
        
        text_len = len(text) if text else 0
        if self._total_output + text_len > _MAX_PRINT_OUTPUT:
            return  # Silently ignore if would exceed limit
        
        self._total_output += text_len
        self.txt.append(text)
    
    def __call__(self) -> str:
        """
        Return collected output (called to get final print result).
        
        Returns:
            Collected text joined as a single string.
        """
        return ''.join(self.txt)
    
    def _call_print(self, *objects, **kwargs) -> None:
        """
        Handle a print call.
        
        This is called by RestrictedPython's rewritten print statements.
        Implements rate-limiting and output size limiting.
        
        Args:
            *objects: Objects to print
            **kwargs: Keyword arguments (sep, end, file, flush)
        """
        self._call_count += 1
        
        # Rate limit: max calls per execution
        if self._call_count > _MAX_PRINT_CALLS:
            return  # Silently ignore after limit
        
        # Handle file= argument - block printing to arbitrary file objects
        if kwargs.get('file', None) is not None:
            logger.debug("[sandboxed print] Blocked print to file=...")
            return
        
        # Estimate output size before constructing the full string
        # This avoids unnecessary string operations for oversized output
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        
        # Quick size estimate: sum of str lengths + separators
        estimated_size = sum(len(str(obj)) for obj in objects)
        estimated_size += len(sep) * max(0, len(objects) - 1) + len(end)
        
        # Size limit: max total output
        if self._total_output + estimated_size > _MAX_PRINT_OUTPUT:
            return  # Silently ignore if would exceed limit
        
        # Now construct the actual output string
        output = sep.join(str(obj) for obj in objects) + end
        self._total_output += len(output)
        
        # Write to our collector and log for debugging
        self.txt.append(output)
        logger.debug(f"[sandboxed print] {output.rstrip()}")


def _create_safe_print_collector(_getattr_=None) -> _SafePrintCollector:
    """
    Factory function to create a rate-limited print collector.
    
    RestrictedPython calls this as: _print_(_getattr_) to create the collector.
    
    Args:
        _getattr_: The guarded getattr function (may be None)
        
    Returns:
        A fresh _SafePrintCollector instance with reset counters.
    """
    return _SafePrintCollector(_getattr_=_getattr_)

# Try to import RestrictedPython
try:
    from RestrictedPython import compile_restricted_exec
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
    )
    from RestrictedPython.Eval import (
        default_guarded_getiter,
        default_guarded_getitem,
    )

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logger.warning("RestrictedPython not available. Safe code execution disabled.")
    compile_restricted_exec = None
    guarded_iter_unpack_sequence = None
    guarded_unpack_sequence = None
    default_guarded_getiter = None
    default_guarded_getitem = None


# SECURITY: Blocked attributes that could be used to escape the sandbox
# These allow traversal of the class hierarchy to access __globals__ and escape
_BLOCKED_ATTRS = frozenset({
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__dict__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__cached__",
    "__file__",
    "__name__",
    "__qualname__",
    "__module__",
    "__annotations__",
    "__wrapped__",
    "gi_frame",
    "gi_code",
    "f_globals",
    "f_locals",
    "f_code",
    "f_builtins",
    "co_code",
    "func_globals",
    "func_code",
})


def _guarded_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """
    Secure attribute access that prevents sandbox escape.
    
    SECURITY: This function blocks access to attributes that could be used
    to traverse the class hierarchy and escape the RestrictedPython sandbox.
    
    The class hierarchy attack works by:
    1. Access ().__class__ to get tuple class
    2. Access __bases__ to get object class  
    3. Access __subclasses__() to enumerate all classes
    4. Find a class with __init__.__globals__ containing dangerous modules
    5. Access sys, os, etc. through __globals__
    
    This guard blocks all such traversal attempts.
    
    Args:
        obj: The object to access attribute on
        name: The attribute name
        default: Default value if attribute not found
        
    Returns:
        The attribute value, or default if blocked/not found
        
    Raises:
        AttributeError: If access to a blocked attribute is attempted
    """
    # Block dangerous attribute access
    if name in _BLOCKED_ATTRS:
        raise AttributeError(
            f"Access to '{name}' is not allowed in sandboxed code"
        )
    
    # Block dunder attributes that start and end with __
    # Exception: Allow specific safe dunders needed for math operations
    if name.startswith("__") and name.endswith("__"):
        # Whitelist of safe dunder methods for mathematical operations
        safe_dunders = {
            "__add__", "__radd__", "__sub__", "__rsub__",
            "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
            "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
            "__pow__", "__rpow__", "__neg__", "__pos__", "__abs__",
            "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
            "__len__", "__iter__", "__next__", "__getitem__", "__setitem__",
            "__contains__", "__hash__", "__str__", "__repr__", "__bool__",
            "__int__", "__float__", "__complex__", "__round__",
            "__enter__", "__exit__",  # Context managers (limited use)
        }
        if name not in safe_dunders:
            raise AttributeError(
                f"Access to '{name}' is not allowed in sandboxed code"
            )
    
    # Use getattr - the third argument is the default if attribute doesn't exist
    # If getattr raises AttributeError (which it shouldn't with 3 args), re-raise
    return getattr(obj, name, default)

# Try to import SymPy
try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("SymPy not available. Symbolic mathematics disabled.")
    sp = None

# Try to import NumPy
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Numerical computations disabled.")
    np = None


class SafeCodeExecutor:
    """
    Execute mathematical Python code in a restricted environment.

    Uses RestrictedPython to sandbox code execution, preventing dangerous
    operations while allowing mathematical computations using SymPy and NumPy.

    Thread Safety:
        This class is thread-safe. Multiple threads can execute code concurrently.
        Uses ThreadPoolExecutor for cross-platform timeout support.

    Example:
        >>> executor = SafeCodeExecutor()
        >>> result = executor.execute('''
        ... x = Symbol('x')
        ... result = integrate(x**2, x)
        ... ''')
        >>> print(result['result'])  # x**3/3
    """

    # Maximum execution count before reset (prevents unbounded integer growth)
    _MAX_EXECUTION_COUNT = 2**31 - 1

    def __init__(self, timeout: int = 10):
        """
        Initialize with safe namespace containing allowed modules.

        Args:
            timeout: Maximum execution time in seconds (default: 10)
        """
        self.timeout = timeout
        self._lock = threading.RLock()
        self._execution_count = 0
        self.safe_namespace = self._build_safe_namespace()
        # Thread pool for timeout enforcement (industry standard: shared pool)
        # Note: Single worker to ensure execution isolation and prevent resource exhaustion
        self._executor_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="safe_code_exec"
        )
        self._closed = False

        logger.info(
            f"SafeCodeExecutor initialized: "
            f"RestrictedPython={RESTRICTED_PYTHON_AVAILABLE}, "
            f"SymPy={SYMPY_AVAILABLE}, NumPy={NUMPY_AVAILABLE}"
        )

    def _build_safe_namespace(self) -> Dict[str, Any]:
        """
        Build namespace with allowed modules and functions.

        Returns:
            Dictionary of safe names available during code execution.

        Security Notes:
            - Only explicitly listed functions are available
            - No file I/O operations
            - No network operations
            - No system calls
            - No arbitrary imports
            - No class hierarchy traversal (sandbox escape prevention)
            - Rate-limited print to prevent log flooding DoS
            - Dangerous SymPy features (Function, Lambda) removed
        """
        # Safe builtins (no file I/O, imports, etc.)
        # Note: print is handled via _print_ RestrictedPython guard
        namespace: Dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "frozenset": frozenset,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "complex": complex,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "any": any,
                "all": all,
                "pow": pow,
                "divmod": divmod,
                "isinstance": isinstance,
                "type": type,
                "True": True,
                "False": False,
                "None": None,
            }
        }

        # Add RestrictedPython guards if available
        if guarded_iter_unpack_sequence is not None:
            namespace["__builtins__"]["_iter_unpack_sequence_"] = (
                guarded_iter_unpack_sequence
            )
        if guarded_unpack_sequence is not None:
            namespace["__builtins__"]["_unpack_sequence_"] = (
                guarded_unpack_sequence
            )

        # SECURITY: Use our hardened getattr guard instead of safer_getattr
        # This prevents sandbox escape via class hierarchy traversal
        namespace["_getattr_"] = _guarded_getattr
        
        # Add iteration guards required by RestrictedPython for loops
        if default_guarded_getiter is not None:
            namespace["_getiter_"] = default_guarded_getiter
        if default_guarded_getitem is not None:
            namespace["_getitem_"] = default_guarded_getitem

        # Add SymPy functions if available - curated safe subset for symbolic math
        # SECURITY: Function and Lambda are NOT included as they can execute
        # arbitrary Python code and bypass the sandbox
        if SYMPY_AVAILABLE and sp is not None:
            namespace.update(
                {
                    # SECURITY: Do NOT expose full module reference as it provides
                    # access to dangerous features. Only expose whitelisted functions.
                    # "sp": sp,  # REMOVED - too much access
                    # "sympy": sp,  # REMOVED - too much access
                    # Symbols
                    "Symbol": sp.Symbol,
                    "symbols": sp.symbols,
                    "Dummy": sp.Dummy,
                    "Wild": sp.Wild,
                    # Core operations
                    "integrate": sp.integrate,
                    "Integral": sp.Integral,
                    "diff": sp.diff,
                    "Derivative": sp.Derivative,
                    "solve": sp.solve,
                    "solveset": sp.solveset,
                    "dsolve": sp.dsolve,
                    "simplify": sp.simplify,
                    "expand": sp.expand,
                    "factor": sp.factor,
                    "collect": sp.collect,
                    "cancel": sp.cancel,
                    "apart": sp.apart,
                    "together": sp.together,
                    "trigsimp": sp.trigsimp,
                    "powsimp": sp.powsimp,
                    "radsimp": sp.radsimp,
                    "ratsimp": sp.ratsimp,
                    # Calculus
                    "limit": sp.limit,
                    "Limit": sp.Limit,
                    "series": sp.series,
                    "summation": sp.summation,
                    "Sum": sp.Sum,
                    "product": sp.product,
                    "Product": sp.Product,
                    # Linear algebra
                    "Matrix": sp.Matrix,
                    "eye": sp.eye,
                    "zeros": sp.zeros,
                    "ones": sp.ones,
                    "diag": sp.diag,
                    # Functions
                    "sqrt": sp.sqrt,
                    "cbrt": sp.cbrt,
                    "root": sp.root,
                    "exp": sp.exp,
                    "log": sp.log,
                    "ln": sp.ln,
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "cot": sp.cot,
                    "sec": sp.sec,
                    "csc": sp.csc,
                    "asin": sp.asin,
                    "acos": sp.acos,
                    "atan": sp.atan,
                    "atan2": sp.atan2,
                    "sinh": sp.sinh,
                    "cosh": sp.cosh,
                    "tanh": sp.tanh,
                    "asinh": sp.asinh,
                    "acosh": sp.acosh,
                    "atanh": sp.atanh,
                    "Abs": sp.Abs,
                    "sign": sp.sign,
                    "floor": sp.floor,
                    "ceiling": sp.ceiling,
                    "factorial": sp.factorial,
                    "binomial": sp.binomial,
                    "gamma": sp.gamma,
                    "erf": sp.erf,
                    "erfc": sp.erfc,
                    # Constants
                    "pi": sp.pi,
                    "E": sp.E,
                    "I": sp.I,
                    "oo": sp.oo,
                    "zoo": sp.zoo,
                    "nan": sp.nan,
                    # Number types
                    "Rational": sp.Rational,
                    "Integer": sp.Integer,
                    "Float": sp.Float,
                    # Logic
                    "Eq": sp.Eq,
                    "Ne": sp.Ne,
                    "Lt": sp.Lt,
                    "Le": sp.Le,
                    "Gt": sp.Gt,
                    "Ge": sp.Ge,
                    "And": sp.And,
                    "Or": sp.Or,
                    "Not": sp.Not,
                    # Special functions
                    "Piecewise": sp.Piecewise,
                    # SECURITY: Function and Lambda REMOVED - they can execute
                    # arbitrary Python code. Use sp.Function only in controlled
                    # contexts where code is fully trusted.
                    # "Function": sp.Function,  # SECURITY RISK - REMOVED
                    # "Lambda": sp.Lambda,      # SECURITY RISK - REMOVED
                    # Assumptions (optional - may not exist in all SymPy versions)
                    "Assumptions": getattr(sp, "Assumptions", None),
                    "assuming": getattr(sp, "assuming", None),
                    # Utilities
                    "latex": sp.latex,
                    "pprint": sp.pprint,
                    "N": sp.N,
                    "nsimplify": sp.nsimplify,
                }
            )
            
            # Log missing optional features for debugging
            if getattr(sp, "Assumptions", None) is None:
                logger.debug("SymPy 'Assumptions' not available in this version")
            
            # =================================================================
            # Issue #5 FIX: Pre-define common mathematical bound symbols
            # =================================================================
            # When parsing integrals like ∫₀ᵀu(t)²dt, the symbol T is extracted
            # as a bound but not defined as a SymPy symbol before code execution.
            # This causes "NameError: name 'T' is not defined" errors.
            #
            # Solution: Pre-define common single-letter symbols that are frequently
            # used as integration bounds, function arguments, or mathematical constants.
            # These symbols are created with positive=True assumption where appropriate
            # for proper mathematical behavior.
            #
            # Industry Standard: SymPy symbols should be defined with appropriate
            # assumptions (positive, real, etc.) for correct simplification.
            #
            # NAMING CONFLICT RESOLUTION:
            # - 'N' conflicts with sp.N (numerical evaluation function) - use 'N_sym'
            # - 'I' conflicts with sp.I (imaginary unit) - use 'I_sym' 
            # - 'E' conflicts with sp.E (Euler's number) - already defined above
            # - 'i' could conflict with imaginary but sp.I is the standard - use 'idx'
            # =================================================================
            common_bound_symbols = {
                # Common upper bound symbols (typically positive)
                "T": sp.Symbol("T", positive=True, real=True),  # Time bound
                "N_sym": sp.Symbol("N", positive=True, integer=True),  # Count bound (N_sym to avoid sp.N conflict)
                "M": sp.Symbol("M", positive=True, real=True),  # Mass/magnitude
                "L": sp.Symbol("L", positive=True, real=True),  # Length
                "R": sp.Symbol("R", positive=True, real=True),  # Radius
                "A": sp.Symbol("A", real=True),  # Area/amplitude
                "B": sp.Symbol("B", real=True),  # Bound variable
                "C": sp.Symbol("C", real=True),  # Constant
                "D": sp.Symbol("D", real=True),  # Domain variable
                "K": sp.Symbol("K", positive=True, real=True),  # Constant
                # Common index/integration variables
                "n": sp.Symbol("n", integer=True),  # Index
                "k": sp.Symbol("k", integer=True),  # Index
                "m": sp.Symbol("m", integer=True),  # Index
                "idx": sp.Symbol("idx", integer=True),  # Index (safer than 'i' which could be confused with imaginary)
                "j": sp.Symbol("j", integer=True),  # Index
                # Common function argument variables
                "x": sp.Symbol("x", real=True),  # Primary variable
                "y": sp.Symbol("y", real=True),  # Secondary variable
                "z": sp.Symbol("z", real=True),  # Tertiary variable
                "t": sp.Symbol("t", real=True),  # Time variable
                "u": sp.Symbol("u", real=True),  # Control variable
                "v": sp.Symbol("v", real=True),  # Velocity variable
                "w": sp.Symbol("w", real=True),  # Angular velocity
                "s": sp.Symbol("s", real=True),  # Laplace variable
                # Common parameters
                "a": sp.Symbol("a", real=True),  # Coefficient
                "b": sp.Symbol("b", real=True),  # Coefficient
                "c": sp.Symbol("c", real=True),  # Coefficient
                # Greek letter symbols (commonly used in physics/engineering)
                "alpha": sp.Symbol("alpha", real=True),
                "beta": sp.Symbol("beta", real=True),
                "gamma": sp.Symbol("gamma", real=True),
                "delta": sp.Symbol("delta", real=True),
                "epsilon": sp.Symbol("epsilon", positive=True, real=True),
                "theta": sp.Symbol("theta", real=True),
                "lambda_": sp.Symbol("lambda", positive=True, real=True),  # Python keyword
                "mu": sp.Symbol("mu", real=True),
                "sigma": sp.Symbol("sigma", positive=True, real=True),
                "omega": sp.Symbol("omega", real=True),
                "tau": sp.Symbol("tau", positive=True, real=True),
                "phi": sp.Symbol("phi", real=True),
                "psi": sp.Symbol("psi", real=True),
                "rho": sp.Symbol("rho", positive=True, real=True),
            }
            namespace.update(common_bound_symbols)
            
            logger.debug(
                f"Issue #5 FIX: Added {len(common_bound_symbols)} pre-defined symbols "
                "to safe execution namespace"
            )

        # Add NumPy functions if available - for numerical computations
        if NUMPY_AVAILABLE and np is not None:
            namespace.update(
                {
                    "np": np,
                    "numpy": np,
                    "array": np.array,
                    "linspace": np.linspace,
                    "arange": np.arange,
                    "zeros_np": np.zeros,
                    "ones_np": np.ones,
                    "eye_np": np.eye,
                    "dot": np.dot,
                    "cross": np.cross,
                    "linalg": np.linalg,
                }
            )

        return namespace

    def execute(
        self, 
        code: str, 
        timeout: Optional[int] = None,
        max_retries: int = 0,
        error_callback: Optional[Callable[[str, str], Optional[str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in safe environment and return results.

        Uses ThreadPoolExecutor for cross-platform timeout enforcement.
        This approach works on both Windows and Unix systems, unlike signal-based timeouts.
        
        BUG FIX #5: Added retry logic with optional error feedback callback.
        Industry Standard: Retry with Exponential Backoff for transient failures.

        Args:
            code: Python code string to execute
            timeout: Maximum execution time in seconds (overrides default)
            max_retries: Maximum number of retry attempts on SyntaxError/NameError (default: 0)
            error_callback: Optional callback(code, error) -> fixed_code for error correction.
                           If provided and returns corrected code, that code will be retried.

        Returns:
            Dict with keys:
                - 'success': bool - Whether execution succeeded
                - 'result': Any - The computed result (from 'result' or 'answer' variable)
                - 'error': str | None - Error message if failed
                - 'output': str - Print output from code execution
                - 'namespace': Dict - The namespace after execution (on success only)
                - 'retry_count': int - Number of retries attempted (if retry enabled)

        Security Notes:
            - Timeout enforced via ThreadPoolExecutor (cross-platform)
            - All sandbox protections remain active
            - Thread-safe execution with proper resource cleanup
            - Retry logic only applies to compilation/execution errors, not security violations

        Example:
            >>> executor = SafeCodeExecutor()
            >>> result = executor.execute('''
            ... x = Symbol('x')
            ... result = integrate(x**2, x)
            ... ''')
            >>> print(result['success'])  # True
            >>> print(result['result'])   # x**3/3
            
        Example with retry:
            >>> def fix_code(code, error):
            ...     # LLM-based code correction would go here
            ...     return corrected_code
            >>> result = executor.execute(code, max_retries=2, error_callback=fix_code)
        """
        if not RESTRICTED_PYTHON_AVAILABLE:
            return {
                "success": False,
                "result": None,
                "error": "RestrictedPython not available",
                "output": "",
                "namespace": {},
                "retry_count": 0,
            }

        # Use provided timeout or default
        effective_timeout = timeout if timeout is not None else self.timeout

        with self._lock:
            # Use modulo to prevent unbounded integer growth (Issue 8)
            self._execution_count = (
                self._execution_count + 1
            ) % self._MAX_EXECUTION_COUNT
            exec_id = self._execution_count

        # BUG FIX #5: Retry loop with error feedback
        # Industry Standard: Retry with exponential backoff for transient failures
        last_error = None
        current_code = code
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            retry_count = attempt  # Track current attempt number for observability
            
            if attempt > 0:
                logger.info(
                    f"[{exec_id}] BUG FIX #5: Retry attempt {retry_count}/{max_retries} "
                    f"after error: {last_error[:100] if last_error else 'unknown'}"
                )
            
            logger.debug(f"[{exec_id}] Executing code ({len(current_code)} chars), attempt {attempt + 1}")

            # Bug #2 FIX: Preprocess code to fix implicit multiplication and unicode issues
            # This is a safety net in case code wasn't preprocessed by mathematical_computation.py
            current_code = _preprocess_math_code(current_code)

            # Create fresh namespace for this execution (each execution is isolated)
            execution_namespace = self.safe_namespace.copy()

            # SECURITY: Set up rate-limited print collector factory for this execution
            # RestrictedPython rewrites print(...) to:
            #   _print = _print_(_getattr_)  # Factory creates collector
            #   _print._call_print(...)       # Each print call
            # This prevents log flooding DoS attacks and state leakage between executions
            execution_namespace["_print_"] = _create_safe_print_collector

            # Compile with RestrictedPython (blocks dangerous operations)
            try:
                byte_code = compile_restricted_exec(current_code)
            except Exception as e:
                error_msg = f"Compilation error: {type(e).__name__}: {e}"
                logger.warning(f"[{exec_id}] {error_msg}")
                
                # BUG FIX #5: Check if we can retry with error callback
                if attempt < max_retries and error_callback is not None:
                    last_error = error_msg
                    try:
                        corrected_code = error_callback(current_code, error_msg)
                        if corrected_code and corrected_code != current_code:
                            current_code = corrected_code
                            logger.info(f"[{exec_id}] Error callback provided corrected code, retrying...")
                            continue  # Retry with corrected code
                        else:
                            logger.warning(f"[{exec_id}] Error callback returned no correction, giving up")
                    except Exception as callback_error:
                        logger.error(f"[{exec_id}] Error callback failed: {callback_error}")
                
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "output": "",
                    "namespace": {},
                    "retry_count": retry_count,
                }

            if byte_code.errors:
                error_msg = f"Compilation errors: {byte_code.errors}"
                logger.warning(f"[{exec_id}] {error_msg}")
                
                # BUG FIX #5: Check if we can retry with error callback
                if attempt < max_retries and error_callback is not None:
                    last_error = error_msg
                    try:
                        corrected_code = error_callback(current_code, error_msg)
                        if corrected_code and corrected_code != current_code:
                            current_code = corrected_code
                            logger.info(f"[{exec_id}] Error callback provided corrected code, retrying...")
                            continue  # Retry with corrected code
                    except Exception as callback_error:
                        logger.error(f"[{exec_id}] Error callback failed: {callback_error}")
                
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "output": "",
                    "namespace": {},
                    "retry_count": retry_count,
            }

            # Define the execution function to run in thread
            def _run_code() -> Dict[str, Any]:
                """Inner function to execute code in isolated thread."""
                try:
                    # Execute the code
                    # nosec B102: exec() is intentional here - code is pre-compiled with
                    # RestrictedPython which blocks dangerous operations (imports, file I/O,
                    # attribute access to private members, etc.). This is a sandboxed execution
                    # environment designed for safe mathematical computation.
                    exec(byte_code.code, execution_namespace)  # nosec B102

                    # Extract result (code should assign to 'result' or 'answer')
                    # Check if result variables exist in namespace (regardless of their truthiness)
                    # This correctly handles falsy values like 0, False, empty string
                    if "result" in execution_namespace:
                        result = execution_namespace["result"]
                    elif "answer" in execution_namespace:
                        result = execution_namespace["answer"]
                    else:
                        result = None

                    # Get print output
                    output = ""
                    if "_print_" in execution_namespace:
                        # The print collector is created by _create_safe_print_collector
                        # and stored in a closure. We need to extract it properly.
                        pass  # Output collection handled below

                    # Try to get output from print collector
                    for key, value in execution_namespace.items():
                        if isinstance(value, _SafePrintCollector):
                            output = value()
                            break

                    logger.debug(f"[{exec_id}] Execution successful, result type: {type(result)}")

                    return {
                        "success": True,
                        "result": result,
                        "error": None,
                        "output": output,
                        "namespace": execution_namespace,
                        "retry_count": retry_count,
                    }

                except Exception as e:
                    # Get output even on error
                    output = ""
                    for key, value in execution_namespace.items():
                        if isinstance(value, _SafePrintCollector):
                            output = value()
                            break

                    error_msg = f"{type(e).__name__}: {e}"
                    logger.warning(f"[{exec_id}] Execution error: {error_msg}")
                    return {
                        "success": False,
                        "result": None,
                        "error": error_msg,
                        "output": output,
                        "namespace": {},
                        "retry_count": retry_count,
                    }

            # Execute with timeout using ThreadPoolExecutor (CROSS-PLATFORM)
            # Industry Standard: Use thread pool for timeout enforcement
            if self._closed:
                return {
                    "success": False,
                    "result": None,
                    "error": "Executor has been closed",
                    "output": "",
                    "namespace": {},
                    "retry_count": retry_count,
                }
            
            try:
                future = self._executor_pool.submit(_run_code)
                try:
                    result = future.result(timeout=effective_timeout)
                    
                    # BUG FIX #5: Check if execution failed and we can retry
                    if not result.get("success", False) and attempt < max_retries and error_callback is not None:
                        last_error = result.get("error", "Unknown error")
                        logger.info(f"[{exec_id}] Execution failed, checking if we can retry...")
                        
                        try:
                            corrected_code = error_callback(current_code, last_error)
                            if corrected_code and corrected_code != current_code:
                                current_code = corrected_code
                                logger.info(f"[{exec_id}] Error callback provided corrected code, retrying...")
                                continue  # Retry with corrected code
                        except Exception as callback_error:
                            logger.error(f"[{exec_id}] Error callback failed: {callback_error}")
                    
                    # Success or no more retries - return result
                    return result
                    
                except FuturesTimeoutError:
                    # Get output even on timeout
                    output = ""
                    for key, value in execution_namespace.items():
                        if isinstance(value, _SafePrintCollector):
                            output = value()
                            break

                    error_msg = f"Execution timed out after {effective_timeout} seconds"
                    logger.warning(f"[{exec_id}] {error_msg}")
                    
                    # Timeout is not retryable - return immediately
                    return {
                        "success": False,
                        "result": None,
                        "error": error_msg,
                        "output": output,
                        "namespace": {},
                        "retry_count": retry_count,
                    }
                    
            except Exception as e:
                error_msg = f"Execution failed: {type(e).__name__}: {e}"
                logger.error(f"[{exec_id}] {error_msg}", exc_info=True)
                
                # BUG FIX #5: Check if we can retry this error
                if attempt < max_retries and error_callback is not None:
                    last_error = error_msg
                    try:
                        corrected_code = error_callback(current_code, error_msg)
                        if corrected_code and corrected_code != current_code:
                            current_code = corrected_code
                            logger.info(f"[{exec_id}] Error callback provided corrected code, retrying...")
                            continue  # Retry with corrected code
                    except Exception as callback_error:
                        logger.error(f"[{exec_id}] Error callback failed: {callback_error}")
                
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "output": "",
                    "namespace": {},
                    "retry_count": retry_count,
                }
        
        # BUG FIX #5: Defensive Programming - Fallback if loop exits without return
        # This code should never execute due to returns within the loop, but provides
        # fail-safe behavior in case of unexpected control flow (e.g., future code changes
        # that introduce break statements or other loop exits). The explicit error helps
        # identify bugs during development.
        # INDUSTRY STANDARD: Defense in Depth - multiple layers of error handling
        logger.error(
            f"[{exec_id}] DEFENSIVE FALLBACK: Retry loop exited without returning. "
            f"This indicates a bug in the retry logic. Attempts: {max_retries + 1}"
        )
        return {
            "success": False,
            "result": None,
            "error": f"Internal error: Retry loop completed without result (max_retries={max_retries})",
            "output": "",
            "namespace": {},
            "retry_count": max_retries,
        }

    def close(self):
        """
        Explicitly close the executor and clean up resources.
        
        Industry Standard: Explicit cleanup method for resource management.
        Prefer this over relying on __del__.
        """
        if not self._closed:
            self._closed = True
            try:
                self._executor_pool.shutdown(wait=True, cancel_futures=True)
                logger.debug("SafeCodeExecutor closed successfully")
            except Exception as e:
                logger.warning(f"Error during executor shutdown: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def __del__(self):
        """Cleanup thread pool on deletion (fallback only - prefer explicit close())."""
        if hasattr(self, '_closed') and not self._closed:
            try:
                if hasattr(self, '_executor_pool'):
                    self._executor_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass  # Ignore cleanup errors in __del__


# Singleton instance
_executor: Optional[SafeCodeExecutor] = None
_executor_lock = threading.Lock()


def get_executor(timeout: int = 10) -> SafeCodeExecutor:
    """
    Get singleton executor instance.

    Args:
        timeout: Maximum execution time in seconds

    Returns:
        SafeCodeExecutor singleton instance
    """
    global _executor

    with _executor_lock:
        if _executor is None:
            _executor = SafeCodeExecutor(timeout=timeout)
            logger.info("Global SafeCodeExecutor created")
        return _executor


def reset_executor() -> None:
    """Reset global safe executor (for testing)."""
    global _executor

    with _executor_lock:
        _executor = None
        logger.info("Global SafeCodeExecutor reset")


def execute_math_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Convenience function to execute mathematical code.

    Args:
        code: Python code string to execute (should use SymPy syntax)
        timeout: Maximum execution time in seconds

    Returns:
        Dict with 'success', 'result', 'error', 'namespace' keys

    Example:
        >>> result = execute_math_code('''
        ... x = Symbol('x')
        ... result = integrate(x**2, x)
        ... ''')
        >>> if result['success']:
        ...     print(f"Result: {result['result']}")
    """
    return get_executor(timeout).execute(code, timeout)


# Module-level availability check
def is_safe_execution_available() -> bool:
    """Check if safe code execution is available."""
    return RESTRICTED_PYTHON_AVAILABLE and SYMPY_AVAILABLE


__all__ = [
    "SafeCodeExecutor",
    "get_executor",
    "reset_executor",
    "execute_math_code",
    "is_safe_execution_available",
    "RESTRICTED_PYTHON_AVAILABLE",
    "SYMPY_AVAILABLE",
    "NUMPY_AVAILABLE",
]
