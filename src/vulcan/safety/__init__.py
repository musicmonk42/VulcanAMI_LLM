# src/vulcan/safety/__init__.py
import importlib
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Ensure src is in the path for absolute imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

__all__ = [
    "get_safety_validator",
    "SafetyUnavailable",
    "SafetyValidator",
    "GovernanceOrchestrator",
]


class SafetyUnavailable:
    """Fallback when full safety stack isn't available."""

    def validate(self, *a, **k):
        return {"status": "unavailable", "safe": True}

    def clamp(self, *a, **k):
        return a[0] if a else None

    def get_status(self):
        return {"available": False, "reason": "lazy import not resolved"}


# Import SafetyValidator - NO STUBS
try:
    from vulcan.safety.safety_types import SafetyValidator
except ImportError:
    try:
        from src.vulcan.safety.safety_types import SafetyValidator
    except ImportError as e:
        logger.error(f"CRITICAL: SafetyValidator not available: {e}")
        raise ImportError("SafetyValidator is required - no stubs allowed") from e

# Import GovernanceOrchestrator - NO STUBS
try:
    from vulcan.safety.safety_types import GovernanceOrchestrator
except ImportError:
    try:
        from src.vulcan.safety.safety_types import GovernanceOrchestrator
    except ImportError as e:
        logger.error(f"CRITICAL: GovernanceOrchestrator not available: {e}")
        raise ImportError("GovernanceOrchestrator is required - no stubs allowed") from e


def get_safety_validator():
    """Return EnhancedSafetyValidator if importable, else stub."""
    try:
        mod = importlib.import_module("vulcan.safety.safety_validator")
        cls = getattr(mod, "EnhancedSafetyValidator", None)
        return cls() if cls else SafetyUnavailable()
    except Exception as e:
        logger.warning("Safety validator unavailable (lazy): %s", e)
        return SafetyUnavailable()
