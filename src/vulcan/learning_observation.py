"""Dependency-light facade for the canonical learning observation contract."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys as _sys
_PATH = Path(__file__).resolve().parent / "learning" / "observation.py"
_SPEC = importlib.util.spec_from_file_location("vulcan._learning_observation_contract", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("learning observation contract is unavailable")
_MODULE = importlib.util.module_from_spec(_SPEC)
_sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
for _name in getattr(_MODULE, "__all__", ()): globals()[_name] = getattr(_MODULE, _name)
# explicit exports when __all__ is absent
for _name in [n for n in dir(_MODULE) if not n.startswith("_")]: globals().setdefault(_name, getattr(_MODULE, _name))
