"""
Security Fixes Module
Provides safe versions of potentially dangerous operations
"""
import pickle
import io
import logging
from typing import Any, Set, Optional

logger = logging.getLogger(__name__)


class SafeUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe classes to be loaded.
    
    This prevents arbitrary code execution vulnerabilities (CWE-502)
    by maintaining a whitelist of allowed modules and classes.
    """
    
    # Whitelist of safe modules and classes
    SAFE_MODULES = {
        'torch',
        'numpy',
        'builtins',
        'collections',
        'datetime',
        '__builtin__',
    }
    
    SAFE_CLASSES = {
        # PyTorch
        ('torch', 'Tensor'),
        ('torch._utils', '_rebuild_tensor_v2'),
        ('torch', 'FloatStorage'),
        ('torch', 'LongStorage'),
        ('torch', 'IntStorage'),
        ('torch', 'DoubleStorage'),
        
        # NumPy
        ('numpy', 'ndarray'),
        ('numpy', 'dtype'),
        ('numpy.core.multiarray', 'scalar'),
        ('numpy.core.multiarray', '_reconstruct'),
        
        # Python built-ins (safe ones only)
        ('builtins', 'set'),
        ('builtins', 'frozenset'),
        ('builtins', 'dict'),
        ('builtins', 'list'),
        ('builtins', 'tuple'),
        ('builtins', 'int'),
        ('builtins', 'float'),
        ('builtins', 'str'),
        ('builtins', 'bool'),
        ('builtins', 'bytes'),
        ('builtins', 'bytearray'),
        
        # Collections
        ('collections', 'OrderedDict'),
        ('collections', 'defaultdict'),
        ('collections', 'Counter'),
        ('collections', 'deque'),
        
        # Datetime
        ('datetime', 'datetime'),
        ('datetime', 'date'),
        ('datetime', 'time'),
        ('datetime', 'timedelta'),
    }
    
    def __init__(self, file, *, additional_safe_classes: Optional[Set[tuple]] = None):
        """
        Initialize SafeUnpickler.
        
        Args:
            file: File object to unpickle from
            additional_safe_classes: Optional set of (module, class) tuples to allow
        """
        super().__init__(file)
        
        # Allow caller to extend the whitelist if needed
        if additional_safe_classes:
            self.allowed_classes = self.SAFE_CLASSES | additional_safe_classes
        else:
            self.allowed_classes = self.SAFE_CLASSES.copy()
    
    def find_class(self, module: str, name: str):
        """
        Override find_class to restrict which classes can be unpickled.
        
        Args:
            module: Module name
            name: Class name
            
        Returns:
            The class object if allowed
            
        Raises:
            pickle.UnpicklingError: If class is not in whitelist
        """
        # Check if module is in safe list
        if module not in self.SAFE_MODULES:
            logger.warning(f"Blocked unpickling of class from unsafe module: {module}.{name}")
            raise pickle.UnpicklingError(
                f"Global '{module}.{name}' is forbidden. "
                f"Only classes from {self.SAFE_MODULES} are allowed."
            )
        
        # Check if specific class is in whitelist
        if (module, name) not in self.allowed_classes:
            logger.warning(f"Blocked unpickling of non-whitelisted class: {module}.{name}")
            raise pickle.UnpicklingError(
                f"Class '{module}.{name}' is not in the whitelist. "
                f"If this class is safe, add it to SAFE_CLASSES."
            )
        
        # Class is allowed, proceed with normal unpickling
        logger.debug(f"Allowing unpickling of whitelisted class: {module}.{name}")
        return super().find_class(module, name)


def safe_pickle_load(file, *, additional_safe_classes: Optional[Set[tuple]] = None) -> Any:
    """
    Safely load a pickle file with restricted class loading.
    
    This function prevents arbitrary code execution by only allowing
    a whitelist of known-safe classes to be unpickled.
    
    Args:
        file: File object or file-like object to unpickle from
        additional_safe_classes: Optional set of (module, class) tuples to allow
                                 in addition to the default whitelist
    
    Returns:
        The unpickled object
        
    Raises:
        pickle.UnpicklingError: If an unsafe class is encountered
        
    Example:
        >>> with open('model.pkl', 'rb') as f:
        ...     model = safe_pickle_load(f)
        
        >>> # Allow additional classes if needed
        >>> additional = {('mymodule', 'MyClass')}
        >>> with open('data.pkl', 'rb') as f:
        ...     data = safe_pickle_load(f, additional_safe_classes=additional)
    
    Security Notes:
        - This function prevents CWE-502 (Deserialization of Untrusted Data)
        - Always use this instead of pickle.load() for untrusted data
        - Review the SAFE_CLASSES whitelist before adding new classes
        - Consider using safer formats like JSON or msgpack when possible
    """
    try:
        unpickler = SafeUnpickler(file, additional_safe_classes=additional_safe_classes)
        result = unpickler.load()
        logger.info("Successfully loaded pickle file with safety checks")
        return result
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to load pickle file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading pickle file: {e}")
        raise pickle.UnpicklingError(f"Failed to load pickle: {e}") from e


def safe_pickle_loads(data: bytes, *, additional_safe_classes: Optional[Set[tuple]] = None) -> Any:
    """
    Safely load a pickle from bytes with restricted class loading.
    
    Args:
        data: Pickled bytes to load
        additional_safe_classes: Optional set of (module, class) tuples to allow
    
    Returns:
        The unpickled object
        
    Raises:
        pickle.UnpicklingError: If an unsafe class is encountered
    """
    return safe_pickle_load(io.BytesIO(data), additional_safe_classes=additional_safe_classes)


# Convenience function for backward compatibility
def restricted_loads(data: bytes) -> Any:
    """
    Alias for safe_pickle_loads for backward compatibility.
    
    Args:
        data: Pickled bytes to load
        
    Returns:
        The unpickled object
    """
    return safe_pickle_loads(data)


def is_safe_pickle_file(filepath: str, *, max_size_mb: int = 100) -> tuple[bool, str]:
    """
    Check if a pickle file is safe to load by attempting to parse its structure
    without executing any code.
    
    Args:
        filepath: Path to pickle file
        max_size_mb: Maximum file size in MB (default 100MB)
        
    Returns:
        Tuple of (is_safe: bool, reason: str)
        
    Example:
        >>> is_safe, reason = is_safe_pickle_file('model.pkl')
        >>> if is_safe:
        ...     with open('model.pkl', 'rb') as f:
        ...         model = safe_pickle_load(f)
    """
    import os
    
    # Check file exists
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    # Check file size
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
    
    # Try to parse the pickle structure
    try:
        with open(filepath, 'rb') as f:
            # Don't actually load it, just check the structure
            unpickler = SafeUnpickler(f)
            # This will raise UnpicklingError if unsafe classes are found
            unpickler.load()
        return True, "File appears safe"
    except pickle.UnpicklingError as e:
        return False, f"Unsafe classes detected: {e}"
    except Exception as e:
        return False, f"Error parsing file: {e}"


__all__ = [
    'safe_pickle_load',
    'safe_pickle_loads',
    'restricted_loads',
    'is_safe_pickle_file',
    'SafeUnpickler',
]
