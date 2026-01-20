# src/__init__.py
import logging
import os
import sys

# Configure root logger to output to stdout instead of stderr
# This ensures all log messages are correctly classified by cloud platforms
# (stderr is often treated as error-level regardless of actual log level)
if not logging.root.handlers:
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.root.addHandler(_stdout_handler)
    logging.root.setLevel(logging.INFO)

_logger = logging.getLogger(__name__)

# Only install stdio_policy if NOT in test/CI mode
# This prevents issues with pytest worker process initialization
_is_test_mode = (
    os.environ.get("PYTEST_CURRENT_TEST") is not None
    or os.environ.get("PYTEST_RUNNING") == "1"
    or os.environ.get("CI") == "true"
    or os.environ.get("GITHUB_ACTIONS") is not None
    or "pytest" in sys.modules
)

if not _is_test_mode:
    try:
        from .stdio_policy import install

        install(replace_builtins=True, patch_ray=True, patch_colorama=True)
    except Exception as e:
        _logger.debug(f"StdIO policy install failed (may not be needed): {e}")
else:
    _logger.debug("Skipping stdio_policy installation in test/CI mode")
