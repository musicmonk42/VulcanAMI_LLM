"""Dependency-light facade for governed learning policy activation."""
from __future__ import annotations
import importlib.util, sys as _sys
from pathlib import Path
_PATH = Path(__file__).resolve().parent / "learning" / "governance.py"
_SPEC = importlib.util.spec_from_file_location("vulcan._learning_governance_contract", _PATH)
if _SPEC is None or _SPEC.loader is None: raise ImportError("learning governance unavailable")
_MODULE = importlib.util.module_from_spec(_SPEC); _sys.modules[_SPEC.name] = _MODULE; _SPEC.loader.exec_module(_MODULE)
for _name in [n for n in dir(_MODULE) if not n.startswith("_")]: globals()[_name] = getattr(_MODULE, _name)
