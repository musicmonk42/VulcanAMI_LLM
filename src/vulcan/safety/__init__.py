# src/vulcan/safety/__init__.py
import importlib
import logging

logger = logging.getLogger(__name__)

__all__ = ["get_safety_validator", "SafetyUnavailable"]


class SafetyUnavailable:
    """Fallback when full safety stack isn't available."""

    def validate(self, *a, **k):
        return {"status": "unavailable", "safe": True}

    def clamp(self, *a, **k):
        return a[0] if a else None

    def get_status(self):
        return {"available": False, "reason": "lazy import not resolved"}


def get_safety_validator():
    """Return EnhancedSafetyValidator if importable, else stub."""
    try:
        mod = importlib.import_module("vulcan.safety.safety_validator")
        cls = getattr(mod, "EnhancedSafetyValidator", None)
        return cls() if cls else SafetyUnavailable()
    except Exception as e:
        logger.warning("Safety validator unavailable (lazy): %s", e)
        return SafetyUnavailable()
