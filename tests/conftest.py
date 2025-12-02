# D:\Graphix\tests\conftest.py
# Full, untruncated shim to make short module imports work for tests,
# while your actual source stays organized under src/ with packages.
# INCLUDES DOTENV LOADING for environment variables.

import sys
import os
import pathlib
import importlib
import traceback
from unittest.mock import MagicMock
from dotenv import load_dotenv # <<< --- ADDED DOTENV --- >>>

# CRITICAL FIX: Ensure cryptography and other critical packages are never mocked
# This must happen BEFORE any imports that might use these packages
PROTECTED_MODULES = ['cryptography', 'OpenSSL', 'ssl']
for mod_name in list(sys.modules.keys()):
    if any(mod_name == protected or mod_name.startswith(protected + '.')
           for protected in PROTECTED_MODULES):
        if isinstance(sys.modules[mod_name], MagicMock):
            del sys.modules[mod_name]

# 1) Ensure src/ is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not SRC.exists():
    raise RuntimeError(f"Expected source directory not found: {SRC}")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# <<< --- ADDED DOTENV LOADING --- >>>
print("[conftest] Attempting to load .env file...")
# Assumes .env file is in the project root directory (one level up from tests/)
dotenv_path = ROOT / '.env'
print(f"[conftest] Looking for .env at: {dotenv_path}")
loaded = load_dotenv(dotenv_path=dotenv_path, override=True) # Override existing system vars if needed
if loaded:
    print("[conftest] .env file loaded successfully.")
    # Optional: Verify a specific key (uncomment to debug)
    # print(f"[conftest] OPENAI_API_KEY loaded: {'Exists' if os.getenv('OPENAI_API_KEY') else 'Not Found'}")
else:
    print("[conftest] .env file not found or not loaded.")
# <<< --- END DOTENV LOADING --- >>>

# <<< --- TEST ENVIRONMENT VARIABLES --- >>>
# Set environment variables needed for testing if not already set
if not os.environ.get("GRAPHIX_JWT_SECRET"):
    os.environ["ALLOW_EPHEMERAL_SECRET"] = "true"
# <<< --- END TEST ENVIRONMENT VARIABLES --- >>>

# 2) Build a mapping from simple module name -> dotted module path by scanning src
#    Example: src/unified_runtime/hardware_dispatcher.py
#    - dotted: "unified_runtime.hardware_dispatcher"
#    - alias:  "hardware_dispatcher"
#
# We will import the dotted name and then alias sys.modules["hardware_dispatcher"] = imported_module
# so tests importing the short name keep working.
#
# We explicitly avoid aliasing names that could collide with stdlib/third-party modules
# (not exhaustive, but we cover the big offenders and anything under "cryptography*").
BLOCKLIST_BASENAMES = {
    # common third-party or stdlib module roots to avoid shadowing:
    "cryptography", "asyncio", "concurrent", "multiprocessing", "logging", "json",
    "numpy", "pytest", "unittest", "threading", "collections", "dataclasses",
    "typing", "pathlib", "importlib", "inspect", "time", "datetime", "math",
    "random", "re", "subprocess", "http", "urllib", "email", "hashlib", "hmac",
    "secrets", "ssl", "socket", "select", "selectors", "io", "os", "sys",
    "platform", "statistics", "functools", "itertools", "tempfile", "shutil",
    "pydantic", "pandas", "scipy", "faiss", "sklearn", "requests",
}

# Modules that should not be aliased because they have problematic dependencies
# that might not be available or might cause import errors
SKIP_ALIAS_MODULES = {
    "persistence",  # Uses cryptography which may have import issues
    # Modules that use complex mocking in tests - aliasing causes mock pollution
    "unified_runtime_core",
    "execution_engine",
}

def _is_python_file(p: pathlib.Path) -> bool:
    # Skip __init__.py and test files
    if p.name.startswith("test_") or p.name.endswith("_test.py"):
        return False
    # Skip files in test directories
    if "tests" in p.parts or "test" in p.parts:
        return False
    return p.is_file() and p.suffix == ".py" and p.name not in {"__init__.py"}

def _to_dotted_module(path: pathlib.Path) -> str:
    # Convert a src-relative file path to a dotted module, e.g.:
    #   SRC / "unified_runtime" / "hardware_dispatcher.py" -> "unified_runtime.hardware_dispatcher"
    rel = path.relative_to(SRC)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]  # drop .py
    return ".".join(parts)

def _safe_alias(simple_name: str, dotted: str):
    """
    Import `dotted`, then alias it to `simple_name` in sys.modules if safe.
    """
    # Skip explicitly blocked modules
    if simple_name in SKIP_ALIAS_MODULES:
        return

    # Skip blocklisted names and anything starting with cryptography to avoid breaking 3rd-party imports
    if simple_name in BLOCKLIST_BASENAMES or simple_name.startswith("cryptography"):
        return

    # If the simple name already resolves to a real module (site-packages or stdlib), don't replace it.
    try:
        existing = importlib.util.find_spec(simple_name)
    except Exception:
        existing = None
    if existing is not None:
        # Already importable via normal means; don't alias over it.
        return

    try:
        mod = importlib.import_module(dotted)
    except Exception as e:
        # If import fails (e.g., due to missing dependencies or package layout),
        # don't crash the test session at collection time; just skip alias for this one.
        # You can uncomment the print if you need trace info:
        # print(f"[conftest] Failed to import {dotted} for alias {simple_name}: {e}")
        return

    # Only alias if the module actually imported and the simple name isn't taken
    if simple_name not in sys.modules:
        sys.modules[simple_name] = mod

def _alias_all_src_modules():
    for path in SRC.rglob("*.py"):
        if not _is_python_file(path):
            continue
        dotted = _to_dotted_module(path)
        simple = path.stem
        _safe_alias(simple, dotted)

# Perform the aliasing once per session import
_alias_all_src_modules()

# Optional: expose a helpful debug dump when you need it.
# You can enable this temporarily if you want to see which aliases were created.
# def pytest_sessionstart(session):
#     created = sorted(
#         name for name, mod in sys.modules.items()
#         if isinstance(mod, type(importlib)) and getattr(mod, "__file__", None) and str(SRC) in mod.__file__
#     )
#     print("\n[conftest] Aliases created for short imports:\n  " + "\n  ".join(created) + "\n")

# ============================================================
# FIX: Prevent atexit handlers from blocking test suite exit
# ============================================================

def pytest_sessionstart(session):
    """
    Set environment variable to indicate we're in test mode.
    This allows atexit handlers to skip blocking operations.
    """
    os.environ["PYTEST_RUNNING"] = "1"
    print("[conftest] Test session starting - atexit handlers will be non-blocking")

def pytest_sessionfinish(session, exitstatus):
    """
    Clean up and unregister blocking atexit handlers to prevent freeze after test completion.
    This fixes the issue where tests freeze after running 9000+ tests.
    """
    import atexit as atexit_module
    import threading
    
    print("[conftest] Test session finishing - cleaning up atexit handlers...")
    
    # Get the list of registered atexit handlers
    # We'll clear them to prevent blocking during cleanup
    if hasattr(atexit_module, '_exithandlers'):
        # Python stores atexit handlers in a list of (func, args, kwargs) tuples
        handlers = atexit_module._exithandlers
        original_count = len(handlers)
        
        # Clear all handlers to prevent blocking
        # This is safe because pytest has its own cleanup mechanism
        handlers.clear()
        print(f"[conftest] Cleared {original_count} atexit handlers to prevent freeze")
    
    # Also set a flag to indicate cleanup is done
    os.environ["PYTEST_CLEANUP_DONE"] = "1"
    
    print(f"[conftest] Test session finished with exit status {exitstatus}")