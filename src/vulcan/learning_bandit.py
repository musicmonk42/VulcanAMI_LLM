"""Dependency-light facade for the shadow learning bandit contract."""
from __future__ import annotations
import importlib.util, sys as _sys
from pathlib import Path
_PATH = Path(__file__).resolve().parent / "learning" / "bandit.py"
_SPEC = importlib.util.spec_from_file_location("vulcan._learning_bandit_contract", _PATH)
if _SPEC is None or _SPEC.loader is None: raise ImportError("learning bandit unavailable")
_MODULE = importlib.util.module_from_spec(_SPEC); _sys.modules[_SPEC.name] = _MODULE; _SPEC.loader.exec_module(_MODULE)
for _name in [n for n in dir(_MODULE) if not n.startswith("_")]: globals()[_name] = getattr(_MODULE, _name)
