from pathlib import Path

__all__ = ["__version__"]

def _read_semver() -> str:
    try:
        p = Path(__file__).resolve().parents[2] / "configs" / "packer" / "semver.txt"
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"

__version__ = _read_semver()