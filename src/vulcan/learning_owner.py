"""Dependency-light import facade for the canonical learning owner contract.

The contract source lives at ``src/vulcan/learning/owner.py``. This facade loads
that file directly so importing the runtime owner does not execute the historical
``vulcan.learning`` package initializer or import neural/web dependencies.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_OWNER_PATH = Path(__file__).resolve().parent / "learning" / "owner.py"
_SPEC = importlib.util.spec_from_file_location("vulcan._learning_owner_contract", _OWNER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError("learning owner contract is unavailable")
_MODULE = importlib.util.module_from_spec(_SPEC)
import sys as _sys
_sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

LearningCapabilityStatus = _MODULE.LearningCapabilityStatus
LearningOwnerState = _MODULE.LearningOwnerState
LearningOwnerClosedError = _MODULE.LearningOwnerClosedError
LearningOwnerBackpressureError = _MODULE.LearningOwnerBackpressureError
LearningCapabilitySnapshot = _MODULE.LearningCapabilitySnapshot
QueueHealthSnapshot = _MODULE.QueueHealthSnapshot
LearningOwnerStatusSnapshot = _MODULE.LearningOwnerStatusSnapshot
LearningOwner = _MODULE.LearningOwner

__all__ = [
    "LearningCapabilityStatus",
    "LearningOwnerState",
    "LearningOwnerClosedError",
    "LearningOwnerBackpressureError",
    "LearningCapabilitySnapshot",
    "QueueHealthSnapshot",
    "LearningOwnerStatusSnapshot",
    "LearningOwner",
]
