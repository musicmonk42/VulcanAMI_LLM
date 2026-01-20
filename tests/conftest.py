# D:\Graphix\tests\conftest.py
# Full, untruncated shim to make short module imports work for tests,
# while your actual source stays organized under src/ with packages.
# INCLUDES DOTENV LOADING for environment variables.

import importlib
import os
import pathlib
import sys
import time
import traceback
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
from dotenv import load_dotenv  # <<< --- ADDED DOTENV --- >>>

# ============================================================
# CI-Specific Optimizations for Faster Test Execution
# ============================================================
# When running in CI mode, apply optimizations to reduce test overhead:
# - Skip expensive fixture initialization when not needed
# - Use faster timeouts for CI environment
# - Reduce default complexity of test fixtures
#
# Environment variables that enable CI optimizations:
# - CI=true (standard CI indicator)
# - VULCAN_CI_MODE=1 (explicit VULCAN CI mode)
# - VULCAN_FAST_FIXTURES=1 (use minimal fixtures)
#
# Use flexible checks to handle variations like 'True', '1', 'yes', etc.
# Only treat specific truthy values as True, everything else as False
CI_MODE = (
    os.environ.get("CI", "").lower() in ("true", "1", "yes") or
    os.environ.get("VULCAN_CI_MODE", "").lower() in ("1", "true", "yes")
)
FAST_FIXTURES = os.environ.get("VULCAN_FAST_FIXTURES", "").lower() in ("1", "true", "yes")

if CI_MODE:
    # Reduce pytest timeouts in CI for faster failure detection
    # Individual tests can still override with @pytest.mark.timeout(N)
    DEFAULT_TEST_TIMEOUT = 180  # 3 minutes instead of default 300
    
    # Set environment variables for faster test execution
    os.environ.setdefault("VULCAN_SKIP_SLOW_INIT", "1")
    os.environ.setdefault("VULCAN_MINIMAL_FIXTURES", "1")
    
    print(f"[conftest] CI mode enabled - using optimized configuration")
    print(f"[conftest] - Fast fixtures: {FAST_FIXTURES}")
    print(f"[conftest] - Default test timeout: {DEFAULT_TEST_TIMEOUT}s")
else:
    DEFAULT_TEST_TIMEOUT = 300  # 5 minutes for local development

# ============================================================
# CRITICAL FIX: Ensure cryptography and other critical packages are never mocked
# This must happen BEFORE any imports that might use these packages
PROTECTED_MODULES = ["cryptography", "OpenSSL", "ssl"]
for mod_name in list(sys.modules.keys()):
    if any(
        mod_name == protected or mod_name.startswith(protected + ".")
        for protected in PROTECTED_MODULES
    ):
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
dotenv_path = ROOT / ".env"
print(f"[conftest] Looking for .env at: {dotenv_path}")
loaded = load_dotenv(
    dotenv_path=dotenv_path, override=True
)  # Override existing system vars if needed
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
    "cryptography",
    "asyncio",
    "concurrent",
    "multiprocessing",
    "logging",
    "json",
    "numpy",
    "pytest",
    "unittest",
    "threading",
    "collections",
    "dataclasses",
    "typing",
    "pathlib",
    "importlib",
    "inspect",
    "time",
    "datetime",
    "math",
    "random",
    "re",
    "subprocess",
    "http",
    "urllib",
    "email",
    "hashlib",
    "hmac",
    "secrets",
    "ssl",
    "socket",
    "select",
    "selectors",
    "io",
    "os",
    "sys",
    "platform",
    "statistics",
    "functools",
    "itertools",
    "tempfile",
    "shutil",
    "pydantic",
    "pandas",
    "scipy",
    "faiss",
    "sklearn",
    "requests",
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


_module_aliasing_logged = False

def _alias_all_src_modules():
    """
    Alias all source modules for short imports in tests.
    
    IMPORTANT: This is skipped in CI mode to avoid slow test collection.
    The module aliasing imports 627+ modules, each triggering initialization
    code that can cause pytest to hang during collection with -n auto.
    
    In CI mode, tests should use full import paths (e.g., 'from vulcan.xxx import yyy')
    instead of short imports (e.g., 'from xxx import yyy').
    """
    global _module_aliasing_logged
    
    # Skip module aliasing in CI mode to prevent slow collection and worker initialization
    # This is critical for pytest-xdist (-n auto) where each worker imports conftest
    if CI_MODE:
        # Only log once per process to avoid cluttering output in parallel execution
        if not _module_aliasing_logged:
            _module_aliasing_logged = True
            print("[conftest] Skipping module aliasing in CI mode for faster test collection")
        return
    
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
# Note: Test Isolation - Reset State Between Tests
# ============================================================


@pytest.fixture(autouse=True)
def reset_random_state():
    """Ensure each test starts with a fresh random state."""
    np.random.seed(12345)
    yield


@pytest.fixture(autouse=True)
def reset_environment_state(tmp_path, monkeypatch):
    """
    Reset environment variables before each test to prevent state contamination.

    This fixes:
    - CSIU state contamination where _csiu_regs_enabled is False when it should be True
    - Environment variables persisting between tests
    - Database locking by ensuring unique storage paths per test
    """
    # Remove CSIU environment variables to ensure clean state
    # monkeypatch automatically restores original values after test
    env_vars_to_reset = [
        "INTRINSIC_CSIU_OFF",
        "INTRINSIC_CSIU_REGS_OFF",
        "INTRINSIC_CSIU_CALC_OFF",
    ]

    for var in env_vars_to_reset:
        monkeypatch.delenv(var, raising=False)

    # Set unique storage path for this test to prevent database conflicts
    # This ensures each test gets its own isolated database
    test_storage_path = tmp_path / f"test_storage_{uuid.uuid4().hex}"
    test_storage_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VULCAN_STORAGE_PATH", str(test_storage_path))

    yield
    # monkeypatch automatically restores environment on cleanup


@pytest.fixture
def isolated_db_path(tmp_path):
    """
    Create isolated temporary database path for each test.

    This fixes:
    - SQLite database locking issues from concurrent access
    - "database is locked" errors during teardown/setup overlap

    Each test gets its own unique database file to prevent contention.
    """
    db_file = tmp_path / f"test_{uuid.uuid4().hex}.db"
    return str(db_file)


@pytest.fixture
def fresh_pytorch_model():
    """
    Create a fresh PyTorch model in training mode for each test.

    This fixes:
    - PyTorch gradient state contamination
    - "element 0 of tensors does not require grad" errors
    - Models stuck in eval() mode from previous tests

    The model is explicitly set to train() mode and is NOT shared across tests.
    
    In CI mode with FAST_FIXTURES, returns a smaller model for faster initialization.
    """
    try:
        import torch
        import torch.nn as nn

        class FreshTestModel(nn.Module):
            def __init__(self, input_dim=512, hidden_dim=256):
                super().__init__()
                # In CI fast mode, use smaller dimensions
                if FAST_FIXTURES:
                    input_dim = min(input_dim, 128)
                    hidden_dim = min(hidden_dim, 64)
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = FreshTestModel()
        model.train()  # Explicitly set to train mode
        return model
    except ImportError:
        # PyTorch not available, return None
        return None


@pytest.fixture
def fresh_tensors():
    """
    Create fresh PyTorch tensors with gradients enabled for each test.

    This fixes:
    - Tensor gradient state contamination
    - Reused tensors without requires_grad
    - Shared tensor instances across tests

    Each test gets NEW tensors with requires_grad=True.
    """
    try:
        import torch

        def create_tensor(shape, requires_grad=True):
            """Create a new tensor with specified shape and gradient tracking"""
            return torch.randn(*shape, requires_grad=requires_grad)

        return create_tensor
    except ImportError:
        # PyTorch not available, return None
        return None


@pytest.fixture(autouse=True)
def reset_pytorch_state():
    """
    Reset PyTorch state before and after each test.

    This fixes:
    - Models left in eval() mode by previous tests
    - Tensors with .detach() called on shared instances
    - Global gradient state contamination from torch.no_grad() or torch.set_grad_enabled(False)
    - Gradient state left disabled by previous tests

    Note: .eval() affects model behavior but not global gradient state directly.
    However, this fixture ensures a clean state for all PyTorch operations.

    This is applied automatically to all tests.
    """
    try:
        import torch

        # CRITICAL: Explicitly enable gradients before each test
        # This prevents "element 0 of tensors does not require grad" errors
        # when tests run together (test pollution from previous tests)
        torch.set_grad_enabled(True)

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set default dtype to float32 to ensure consistency
        torch.set_default_dtype(torch.float32)

        yield

        # CRITICAL: Re-enable gradients after each test as well
        # This ensures the next test starts with a clean gradient state
        torch.set_grad_enabled(True)

        # Cleanup after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        # PyTorch not available, skip
        yield


# ============================================================
# Note: Prevent atexit handlers from blocking test suite exit
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

    The issue was that atexit handlers registered by safety modules (tool_safety.py,
    safety_validator.py, neural_safety.py) and config.py were being called during Python
    interpreter shutdown. These handlers contained blocking operations like thread.join()
    and executor.shutdown(wait=True), causing the test suite to hang instead of exiting cleanly.

    Solution:
    1. Set PYTEST_RUNNING=1 at session start so handlers know to skip blocking operations
    2. Clear atexit handlers at session finish to prevent them from running at all
    3. Terminate any remaining multiprocessing child processes to prevent hanging
    4. This is safe because pytest has its own cleanup mechanism
    """
    import atexit as atexit_module
    import multiprocessing
    import threading

    print("[conftest] Test session finishing - cleaning up atexit handlers...")

    # Get the list of registered atexit handlers
    # We'll clear them to prevent blocking during cleanup
    if hasattr(atexit_module, "_exithandlers"):
        # Python stores atexit handlers in a list of (func, args, kwargs) tuples
        handlers = atexit_module._exithandlers
        original_count = len(handlers)

        # Clear all handlers to prevent blocking
        # This is safe because pytest has its own cleanup mechanism
        handlers.clear()
        print(f"[conftest] Cleared {original_count} atexit handlers to prevent freeze")
    else:
        print(
            "[conftest] Could not access _exithandlers (Python implementation may vary)"
        )

    # Clean up any remaining multiprocessing child processes
    # This prevents pytest from hanging while waiting for child processes to exit
    print("[conftest] Terminating remaining multiprocessing child processes...")
    active_children = multiprocessing.active_children()
    if active_children:
        print(f"[conftest] Found {len(active_children)} active child processes")
        for child in active_children:
            try:
                print(
                    f"[conftest] Terminating child process {child.pid} ({child.name})"
                )
                child.terminate()
            except Exception as e:
                print(f"[conftest] Error terminating child process {child.pid}: {e}")

        # Give processes a moment to terminate gracefully
        time.sleep(0.5)

        # Force kill any that didn't terminate
        remaining = multiprocessing.active_children()
        if remaining:
            print(f"[conftest] Force killing {len(remaining)} remaining processes")
            for child in remaining:
                try:
                    child.kill()
                    child.join(timeout=1)
                except Exception as e:
                    print(f"[conftest] Error killing child process {child.pid}: {e}")
    else:
        print("[conftest] No active child processes found")

    # Also set a flag to indicate cleanup is done
    os.environ["PYTEST_CLEANUP_DONE"] = "1"

    print(f"[conftest] Test session finished with exit status {exitstatus}")
