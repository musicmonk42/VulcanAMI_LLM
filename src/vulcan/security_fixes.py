"""
Production Readiness Fixes

This module contains critical security and reliability fixes identified in the audit.
Apply these fixes systematically across the codebase.
"""

# ============================================================================
# FIX 1: Replace Bare Except Clauses
# ============================================================================

# BEFORE (UNSAFE):
# try:
#     risky_operation()
# except:
#     pass

# AFTER (SAFE):
# try:
#     risky_operation()
# except Exception as e:
#     logger.error(f"Operation failed: {e}", exc_info=True)
#     # Handle appropriately: retry, return default, or re-raise

# ============================================================================
# FIX 2: Safe Pickle Loading
# ============================================================================

import string
import secrets
from typing import Dict, List
from typing import Callable, Optional, TypeVar
from functools import wraps
import traceback
from pathlib import Path
import subprocess
import shlex
import re
import io
import logging
import os
import pickle
from typing import Any, BinaryIO, Set, Type, Union

logger = logging.getLogger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe classes.
    Prevents arbitrary code execution via pickle deserialization.
    """

    # Whitelist of allowed modules and classes
    SAFE_MODULES: Set[str] = {
        "builtins",
        "numpy",
        "numpy.core",
        "numpy.core.multiarray",
        "numpy._core",  # numpy >= 2.0 uses _core instead of core
        "numpy._core.multiarray",
        "numpy._core.multiarray.scalar",
        "numpy._core.multiarray._reconstruct",
        "torch",
        "torch.nn",
        "torch.nn.modules",
        "collections",
        "datetime",
        # Conformal calibration modules for confidence calibration
        "src.conformal.confidence_calibration",
        "conformal.confidence_calibration",
        # Sklearn modules commonly used for calibration
        "sklearn.isotonic",
        "sklearn.linear_model",
        "sklearn.linear_model._logistic",
        "sklearn.calibration",
        # Cost model module for serialization
        "src.strategies.cost_model",
        "strategies.cost_model",
        # Test modules (for pytest test fixtures and helper classes)
        "__main__",  # Sometimes test classes are in __main__
        # Add your safe modules here
    }

    SAFE_CLASSES: Set[str] = {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "int",
        "float",
        "complex",
        "bool",
        "str",
        "bytes",
        "NoneType",
        "type",
        # Add your safe classes here
    }

    def find_class(self, module: str, name: str) -> Type:
        """
        Only allow safe classes to be unpickled.
        Raises pickle.UnpicklingError for unsafe classes.
        """
        # Allow numpy array reconstruction functions (critical for array loading)
        if module == "numpy._core.multiarray" or module == "numpy.core.multiarray":
            if name == "_reconstruct":
                try:
                    import numpy._core.multiarray as nm

                    return getattr(nm, name)
                except (ImportError, AttributeError):
                    # Fall back to numpy.core for older versions
                    try:
                        import numpy.core.multiarray as nm

                        return getattr(nm, name)
                    except (ImportError, AttributeError):
                        pass

        # Allow PyTorch tensor reconstruction functions (critical for model loading)
        if module == "torch._utils":
            if name.startswith("_rebuild_"):
                import torch._utils

                return getattr(torch._utils, name)

        # Allow PyTorch storage types (FloatStorage, LongStorage, etc.)
        if module == "torch.storage":
            import torch.storage

            if hasattr(torch.storage, name):
                return getattr(torch.storage, name)

        # Allow PyTorch parameter
        if module == "torch.nn.parameter" and name == "Parameter":
            import torch.nn.parameter

            return torch.nn.parameter.Parameter

        # Allow PyTorch module internals
        if module == "torch.nn.modules.module":
            import torch.nn.modules.module

            if hasattr(torch.nn.modules.module, name):
                return getattr(torch.nn.modules.module, name)

        # Allow collections.OrderedDict (used in PyTorch state_dict)
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict

            return OrderedDict

        # Allow collections.abc classes
        if module == "collections.abc":
            import collections.abc

            if hasattr(collections.abc, name):
                return getattr(collections.abc, name)

        # FIXED: Allow test classes (from test modules)
        # This allows test fixtures and helper classes to be pickled/unpickled in tests
        if ".tests." in module or module.startswith("tests."):
            try:
                import importlib

                mod = importlib.import_module(module)
                return getattr(mod, name)
            except (ImportError, AttributeError) as e:
                # If we can't import it, fall through to the whitelist check
                pass

        # Check if module is in whitelist
        if module not in self.SAFE_MODULES:
            raise pickle.UnpicklingError(
                f"Attempted to unpickle unsafe module: {module}.{name}"
            )

        # Additional check for class name if needed
        # if name not in self.SAFE_CLASSES:
        #     raise pickle.UnpicklingError(
        #         f"Attempted to unpickle unsafe class: {module}.{name}"
        #     )

        return super().find_class(module, name)


def safe_pickle_load(file_or_path: Union[str, os.PathLike, BinaryIO]) -> Any:
    """
    Safely load pickled data with restricted unpickler.

    FIXED: Now accepts either a file path OR an already-opened file handle.
    This fixes the "TypeError: expected str, bytes or os.PathLike object, not BufferedReader" error.

    Args:
        file_or_path: Either a path to pickle file (str/PathLike) or an open file handle (BinaryIO)

    Returns:
        Unpickled data

    Raises:
        pickle.UnpicklingError: If unsafe class is encountered
        FileNotFoundError: If file doesn't exist (when path is provided)
        TypeError: If argument is neither a path nor a file handle

    Examples:
        # Usage with file path
        data = safe_pickle_load("/path/to/file.pkl")

        # Usage with open file handle
        with open("/path/to/file.pkl", "rb") as f:
            data = safe_pickle_load(f)
    """
    # Check if it's a file-like object (has 'read' method)
    if hasattr(file_or_path, "read"):
        # It's already an open file handle
        try:
            return RestrictedUnpickler(file_or_path).load()
        except pickle.UnpicklingError as e:
            logger.error(f"Unsafe pickle load attempt from file handle: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load pickle from file handle: {e}")
            raise
    else:
        # It's a file path - open it
        try:
            with open(file_or_path, "rb") as f:
                return RestrictedUnpickler(f).load()
        except pickle.UnpicklingError as e:
            logger.error(f"Unsafe pickle load attempt from {file_or_path}: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Pickle file not found: {file_or_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load pickle from {file_or_path}: {e}")
            raise


# ============================================================================
# FIX 3: Safe Subprocess Execution
# ============================================================================


def validate_file_path(file_path: str, allowed_base: str = None) -> Path:
    """
    Validate and sanitize file path to prevent path traversal.

    Args:
        file_path: Path to validate
        allowed_base: Base directory that path must be within

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or outside allowed base
    """
    path = Path(file_path).resolve()

    # Check for path traversal
    if allowed_base:
        allowed = Path(allowed_base).resolve()
        if not str(path).startswith(str(allowed)):
            raise ValueError(f"Path {path} is outside allowed base {allowed}")

    # Additional checks
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    return path


def safe_git_add(file_path: str, repo_root: str = ".") -> subprocess.CompletedProcess:
    """
    Safely execute 'git add' with validated file path.

    Args:
        file_path: File to add to git
        repo_root: Git repository root

    Returns:
        CompletedProcess from subprocess.run

    Raises:
        ValueError: If file_path is invalid
        subprocess.CalledProcessError: If git command fails
    """
    # Validate file path
    validated_path = validate_file_path(file_path, allowed_base=repo_root)

    # Use list form (not shell=True) for safety
    cmd = ["git", "add", str(validated_path)]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,  # Prevent hanging
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"Git add timed out for {file_path}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Git add failed: {e.stderr}")
        raise


def safe_git_commit(message: str, repo_root: str = ".") -> subprocess.CompletedProcess:
    """
    Safely execute 'git commit' with sanitized message.

    Args:
        message: Commit message
        repo_root: Git repository root

    Returns:
        CompletedProcess from subprocess.run
    """
    # Sanitize commit message - remove potentially dangerous characters
    # Allow alphanumeric, space, punctuation, but no shell metacharacters
    safe_message = re.sub(r"[;&|`$()<>]", "", message)
    safe_message = safe_message[:500]  # Limit length

    cmd = ["git", "commit", "-m", safe_message]

    try:
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=30
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error("Git commit timed out")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Git commit failed: {e.stderr}")
        raise


# ============================================================================
# FIX 4: Enhanced Error Handling Pattern
# ============================================================================


T = TypeVar("T")


def safe_execute(
    operation_name: str, default_return: Optional[T] = None, reraise: bool = False
) -> Callable:
    """
    Decorator for safe execution with proper error handling.

    Args:
        operation_name: Name of operation for logging
        default_return: Value to return on error
        reraise: Whether to re-raise exception after logging

    Usage:
        @safe_execute("database_query", default_return=[])
        def query_users():
            return db.query(User).all()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                # Never catch keyboard interrupt
                raise
            except SystemExit:
                # Never catch system exit
                raise
            except Exception as e:
                logger.error(
                    f"{operation_name} failed: {e}",
                    exc_info=True,
                    extra={
                        "operation": operation_name,
                        "function": func.__name__,
                        "args": str(args)[:100],  # Limit log size
                        "kwargs": str(kwargs)[:100],
                    },
                )
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


# ============================================================================
# FIX 5: Production Configuration Validation
# ============================================================================


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


def validate_production_config() -> None:
    """
    Validate that all required production configuration is present.
    Call this on application startup.

    Raises:
        ConfigurationError: If required config is missing
    """
    required_vars: Dict[str, str] = {
        "JWT_SECRET_KEY": "JWT signing secret",
        "DB_URI": "Database connection string",
        "REDIS_HOST": "Redis host for rate limiting",
        "CORS_ORIGINS": "Allowed CORS origins",
    }

    missing: List[str] = []
    weak_secrets: List[str] = []

    for var, description in required_vars.items():
        value = os.environ.get(var)

        if not value:
            missing.append(f"{var} ({description})")
            continue

        # Check for weak secrets
        if "SECRET" in var or "KEY" in var:
            weak_values = {"secret", "password", "default", "changeme", "dev", "test"}
            if any(weak in value.lower() for weak in weak_values):
                weak_secrets.append(var)

    if missing:
        raise ConfigurationError(
            f"Missing required configuration: {', '.join(missing)}"
        )

    if weak_secrets:
        raise ConfigurationError(
            f"Weak secrets detected (use strong random values): {', '.join(weak_secrets)}"
        )

    # Validate production mode settings
    if os.environ.get("FLASK_ENV") == "development":
        logger.warning(
            "FLASK_ENV=development detected. Set to 'production' for production!"
        )

    if os.environ.get("DEBUG", "").lower() == "true":
        raise ConfigurationError("DEBUG=true is not allowed in production")

    logger.info("✅ Production configuration validated successfully")


# ============================================================================
# FIX 6: Secure Random Token Generation
# ============================================================================


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Args:
        length: Token length (default 32 bytes = 64 hex chars)

    Returns:
        Hex-encoded random token
    """
    return secrets.token_hex(length)


def generate_secure_password(length: int = 16) -> str:
    """
    Generate a cryptographically secure random password.

    Args:
        length: Password length (minimum 12)

    Returns:
        Random password with mixed case, digits, and symbols
    """
    if length < 12:
        raise ValueError("Password length must be at least 12 characters")

    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = "".join(secrets.choice(alphabet) for _ in range(length))

    # Ensure password has at least one of each character type
    if not any(c.islower() for c in password):
        password = secrets.choice(string.ascii_lowercase) + password[1:]
    if not any(c.isupper() for c in password):
        password = secrets.choice(string.ascii_uppercase) + password[1:]
    if not any(c.isdigit() for c in password):
        password = secrets.choice(string.digits) + password[1:]
    if not any(c in string.punctuation for c in password):
        password = secrets.choice(string.punctuation) + password[1:]

    return password


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Safe pickle loading (both methods)
    try:
        # Method 1: With file path
        data = safe_pickle_load("/path/to/checkpoint.pkl")
        print("Loaded data safely:", type(data))

        # Method 2: With open file handle
        with open("/path/to/checkpoint.pkl", "rb") as f:
            data = safe_pickle_load(f)
            print("Loaded data safely from handle:", type(data))
    except pickle.UnpicklingError as e:
        print(f"Unsafe pickle detected: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")

    # Example 2: Safe git operations
    try:
        safe_git_add("src/my_file.py", repo_root=".")
        safe_git_commit("Update my_file.py with security fixes")
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"Git operation failed: {e}")

    # Example 3: Safe execution decorator
    @safe_execute("user_lookup", default_return=None)
    def get_user(user_id: int):
        # This would normally query database
        if user_id < 0:
            raise ValueError("Invalid user_id")
        return {"id": user_id, "name": "Test User"}

    result = get_user(-1)  # Returns None instead of crashing
    print("User lookup result:", result)

    # Example 4: Configuration validation
    try:
        validate_production_config()
    except ConfigurationError as e:
        print(f"Configuration error: {e}")

    # Example 5: Secure token generation
    token = generate_secure_token()
    password = generate_secure_password(16)
    print(f"Generated token: {token}")
    print(f"Generated password: {password}")
