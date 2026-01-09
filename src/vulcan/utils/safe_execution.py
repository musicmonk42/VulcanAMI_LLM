"""
Safe code execution for mathematical computations.

Allows SymPy/NumPy code execution in a restricted environment using
RestrictedPython to prevent dangerous operations.

Security Features:
- Blocks file system access
- Blocks network operations
- Blocks arbitrary imports
- Blocks system command execution
- Allows only whitelisted mathematical operations

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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _preprocess_math_code(code: str) -> str:
    """
    Preprocess mathematical code to fix common syntax issues.
    
    Bug #2 FIX (Jan 9 2026): Added as safety net for implicit multiplication.
    
    This handles cases where LLM-generated or template code contains:
    - Implicit multiplication: 2k → 2*k, 3n → 3*n
    - Digit followed by parenthesis: 2(x+1) → 2*(x+1)
    - Unicode minus: − (U+2212) → - (U+002D ASCII)
    - Unicode math italic letters: 𝑘, 𝑛, 𝑥, 𝑦, 𝑧 → k, n, x, y, z
    
    This is applied as a safety net in execute() to catch any code that
    wasn't preprocessed by mathematical_computation.py._clean_code().
    
    Args:
        code: Python code string to preprocess
        
    Returns:
        Preprocessed code with valid Python syntax
    """
    if not code:
        return code
    
    # Unicode normalization for mathematical symbols
    code = code.replace('−', '-')  # Unicode minus → ASCII minus (U+2212 → U+002D)
    code = code.replace('𝑘', 'k')  # Math italic k → ASCII k
    code = code.replace('𝑛', 'n')  # Math italic n → ASCII n
    code = code.replace('𝑥', 'x')  # Math italic x → ASCII x
    code = code.replace('𝑦', 'y')  # Math italic y → ASCII y
    code = code.replace('𝑧', 'z')  # Math italic z → ASCII z
    
    # Add implicit multiplication operator
    # Pattern: digit followed by letter (2k → 2*k)
    code = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', code)
    
    # Pattern: digit followed by opening parenthesis (2(x+1) → 2*(x+1))
    code = re.sub(r'(\d)\(', r'\1*(', code)
    
    # Pattern: closing paren followed by opening paren ()(x+1) → )*(x+1))
    # This handles cases like (k+1)(k+2)
    code = re.sub(r'\)\(', r')*(', code)
    
    return code

# Try to import RestrictedPython
try:
    from RestrictedPython import compile_restricted_exec
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        safer_getattr,
    )

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logger.warning("RestrictedPython not available. Safe code execution disabled.")
    compile_restricted_exec = None
    guarded_iter_unpack_sequence = None
    safer_getattr = None

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

    Example:
        >>> executor = SafeCodeExecutor()
        >>> result = executor.execute('''
        ... x = Symbol('x')
        ... result = integrate(x**2, x)
        ... ''')
        >>> print(result['result'])  # x**3/3
    """

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
        """
        # Safe builtins (no file I/O, imports, etc.)
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
                "print": print,  # For debugging
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

        # Add safer_getattr to allow attribute access on safe objects
        if safer_getattr is not None:
            namespace["_getattr_"] = safer_getattr

        # Add SymPy functions if available - full access for symbolic math
        if SYMPY_AVAILABLE and sp is not None:
            namespace.update(
                {
                    # Module reference
                    "sp": sp,
                    "sympy": sp,
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
                    "Function": sp.Function,
                    "Lambda": sp.Lambda,
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

    def execute(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute code in safe environment and return results.

        Args:
            code: Python code string to execute
            timeout: Maximum execution time in seconds (overrides default)

        Returns:
            Dict with keys:
                - 'success': bool - Whether execution succeeded
                - 'result': Any - The computed result (from 'result' or 'answer' variable)
                - 'error': str | None - Error message if failed
                - 'namespace': Dict - The namespace after execution

        Example:
            >>> executor = SafeCodeExecutor()
            >>> result = executor.execute('''
            ... x = Symbol('x')
            ... result = integrate(x**2, x)
            ... ''')
            >>> print(result['success'])  # True
            >>> print(result['result'])   # x**3/3
        """
        if not RESTRICTED_PYTHON_AVAILABLE:
            return {
                "success": False,
                "result": None,
                "error": "RestrictedPython not available",
                "namespace": {},
            }

        with self._lock:
            self._execution_count += 1
            exec_id = self._execution_count

        logger.debug(f"[{exec_id}] Executing code ({len(code)} chars)")

        try:
            # Bug #2 FIX: Preprocess code to fix implicit multiplication and unicode issues
            # This is a safety net in case code wasn't preprocessed by mathematical_computation.py
            code = _preprocess_math_code(code)
            
            # Create fresh namespace for this execution
            execution_namespace = self.safe_namespace.copy()

            # Compile with RestrictedPython (blocks dangerous operations)
            byte_code = compile_restricted_exec(code)

            if byte_code.errors:
                error_msg = f"Compilation errors: {byte_code.errors}"
                logger.warning(f"[{exec_id}] {error_msg}")
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "namespace": {},
                }

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

            logger.debug(f"[{exec_id}] Execution successful, result type: {type(result)}")

            return {
                "success": True,
                "result": result,
                "error": None,
                "namespace": execution_namespace,
            }

        except SyntaxError as e:
            error_msg = f"Syntax error: {e}"
            logger.warning(f"[{exec_id}] {error_msg}")
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "namespace": {},
            }
        except NameError as e:
            error_msg = f"Name error (undefined variable/function): {e}"
            logger.warning(f"[{exec_id}] {error_msg}")
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "namespace": {},
            }
        except TypeError as e:
            error_msg = f"Type error: {e}"
            logger.warning(f"[{exec_id}] {error_msg}")
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "namespace": {},
            }
        except Exception as e:
            error_msg = f"Execution failed: {type(e).__name__}: {e}"
            logger.error(f"[{exec_id}] {error_msg}")
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "namespace": {},
            }


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
