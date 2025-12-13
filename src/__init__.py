# src/__init__.py
import logging
_logger = logging.getLogger(__name__)

try:
    from .stdio_policy import install

    install(replace_builtins=True, patch_ray=True, patch_colorama=True)
except Exception as e:
    _logger.debug(f"StdIO policy install failed (may not be needed): {e}")
