# src/__init__.py
import logging
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

try:
    from .stdio_policy import install

    install(replace_builtins=True, patch_ray=True, patch_colorama=True)
except Exception as e:
    _logger.debug(f"StdIO policy install failed (may not be needed): {e}")
