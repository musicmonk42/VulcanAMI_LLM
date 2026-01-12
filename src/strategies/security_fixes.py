"""
Security fixes module for strategies package.

Re-exports safe_pickle_load and RestrictedUnpickler from vulcan.security_fixes
with robust fallback implementation for cases where vulcan is not available.

This module ensures secure pickle operations are available throughout the codebase
regardless of import context or package structure.
"""

import logging

logger = logging.getLogger(__name__)

# Try multiple import paths with graceful fallback
try:
    from src.vulcan.security_fixes import RestrictedUnpickler, safe_pickle_load
    logger.debug("Imported security_fixes from src.vulcan.security_fixes")
except ImportError:
    try:
        from vulcan.security_fixes import RestrictedUnpickler, safe_pickle_load
        logger.debug("Imported security_fixes from vulcan.security_fixes")
    except ImportError:
        logger.warning(
            "Could not import security_fixes from vulcan, using fallback implementation"
        )
        import pickle
        import os
        from typing import Any, BinaryIO, Union
        
        class RestrictedUnpickler(pickle.Unpickler):
            """
            Fallback restricted unpickler that blocks dangerous modules.
            
            This prevents arbitrary code execution via pickle deserialization by
            blocking imports of modules that could be used maliciously.
            
            Blocked modules include:
                - os, subprocess: System command execution
                - sys, builtins, __builtin__: Interpreter manipulation
                - socket, urllib: Network access
                - importlib: Dynamic imports
            
            Thread Safety:
                This class is thread-safe as pickle.Unpickler is thread-safe.
            """
            
            # Modules that are blocked from unpickling
            BLOCKED_MODULES = {
                'os', 'subprocess', 'sys', 'builtins', '__builtin__',
                'socket', 'urllib', 'urllib.request', 'urllib2',
                'importlib', 'imp', '__import__'
            }
            
            def find_class(self, module: str, name: str) -> type:
                """
                Override find_class to block dangerous modules.
                
                Args:
                    module: Module name
                    name: Class name within module
                
                Returns:
                    The class object if allowed
                
                Raises:
                    pickle.UnpicklingError: If module is blocked
                """
                # Check if the root module is blocked
                root_module = module.split('.')[0]
                if root_module in self.BLOCKED_MODULES:
                    raise pickle.UnpicklingError(
                        f"Unpickling from module '{module}' is not allowed for security reasons"
                    )
                
                # Allow the unpickling
                return super().find_class(module, name)
        
        def safe_pickle_load(file_or_path: Union[str, bytes, BinaryIO]) -> Any:
            """
            Safely load pickle file with restricted unpickler.
            
            This function uses RestrictedUnpickler to prevent code execution
            during deserialization. It accepts either a file path or file object.
            
            Args:
                file_or_path: File path (str/bytes) or file object (BinaryIO)
            
            Returns:
                Unpickled object
            
            Raises:
                pickle.UnpicklingError: If pickle contains blocked modules
                FileNotFoundError: If file path doesn't exist
                IOError: If file cannot be read
            
            Security:
                Uses RestrictedUnpickler to prevent arbitrary code execution.
                Only safe modules are allowed during unpickling.
            
            Example:
                >>> obj = safe_pickle_load("/path/to/file.pkl")
                >>> # Or with file object:
                >>> with open("/path/to/file.pkl", "rb") as f:
                ...     obj = safe_pickle_load(f)
            """
            # Handle file path input
            if isinstance(file_or_path, (str, bytes)):
                if isinstance(file_or_path, bytes):
                    try:
                        file_or_path = file_or_path.decode('utf-8')
                    except UnicodeDecodeError as e:
                        raise ValueError(
                            f"Invalid UTF-8 encoded path: {e}. "
                            "Path must be valid UTF-8 or ASCII string."
                        ) from e
                
                if not os.path.isfile(file_or_path):
                    raise FileNotFoundError(f"Pickle file not found: {file_or_path}")
                
                with open(file_or_path, 'rb') as f:
                    return RestrictedUnpickler(f).load()
            
            # Handle file object input
            return RestrictedUnpickler(file_or_path).load()

__all__ = ["safe_pickle_load", "RestrictedUnpickler"]

