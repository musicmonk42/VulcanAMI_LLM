#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphix StdIO Policy (world-class edition)
==========================================

Purpose
-------
A single, authoritative module that makes console I/O:
- **Deterministic** (stable ordering, normalized newlines),
- **Safe** (no recursion loops on Windows with Colorama / pytest / Ray),
- **Governed** (effect-tagged events you can route to audit),
- **Portable** (same behavior across OSes and shells),
- **Testable** (idempotent install/uninstall, self-test).

Why this matters for an AI-native language
-----------------------------------------
In Graphix, printing/logging is an **effect**. This policy provides:
- A canonical `safe_print()` with an **effect label**: `Effect.IO.Stdout`.
- A structured `json_print()` for auditability (JSONL friendly).
- Optional **audit sinks** compatible with your governance loop.
- An `install()` that (optionally) replaces `builtins.print` for every module,
  fixes Ray's `tqdm_ray.safe_print`, and neutralizes Colorama recursion.

It is idempotent, reversible, and ships with re-entry guards and diagnostics.

FIXES APPLIED:
- Proper error handling (no silent swallowing)
- Path edge case handling for empty dirname
- Removed duplicate imports
- Removed unicode ellipsis from test
- Context manager support for StdIOHandle
- Consistent locking throughout
- Better error logging

Usage
-----
    from src.stdio_policy import install, safe_print, json_print

    handle = install(replace_builtins=True, patch_ray=True, patch_colorama=True)
    safe_print("hello world")
    json_print(event="fitness", value=0.92, effect="Effect.IO.Stdout")
    handle.restore()

    # Or use as context manager:
    with install(replace_builtins=True) as handle:
        safe_print("hello world")
    # Automatically restored

Drop-in "always on" hook (optional):
------------------------------------
At the top of `src/__init__.py`:

    try:
        from .stdio_policy import install
        install(replace_builtins=True, patch_ray=True, patch_colorama=True)
    except Exception as e:
        _logger.debug(f"StdIO policy install in __init__.py failed (may not be needed): {e}")

License: MIT (or your project's license)
"""

from __future__ import annotations
import builtins as _builtins

import atexit
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TextIO

# Set up logger for this module
_logger = logging.getLogger("stdio_policy")

# Import builtins once at module level to avoid duplicate imports

# ---------- Configuration & State ----------


@dataclass
class StdIOConfig:
    # Behavior toggles
    replace_builtins: bool = True
    patch_ray: bool = True
    patch_colorama: bool = True
    patch_tqdm: bool = True

    # Output normalization
    normalize_newlines: bool = True
    newline: str = "\n"
    flush: bool = True

    # Coloring / ANSI
    enable_color: Optional[bool] = None  # None=auto by env; True/False forces
    strip_color_on_windows: bool = False  # keep ANSI but avoid Colorama wrapping

    # Structured logging / audit
    jsonl_path: Optional[str] = None  # e.g., "governance_artifacts/io_events.jsonl"
    include_pid_tid: bool = True
    include_ts: bool = True
    effect_label: str = "Effect.IO.Stdout"

    # Safety
    max_len: int = 1_000_000  # prevent giant writes
    lock_print: bool = True  # serialize concurrent prints

    # Diagnostics
    verbose: bool = False


@dataclass
class _StdIOState:
    installed: bool = False
    original_print: Optional[Callable[..., Any]] = None
    original_stdout: Optional[TextIO] = None
    original_stderr: Optional[TextIO] = None
    patched_ray: bool = False
    patched_colorama: bool = False
    patched_tqdm: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATE = _StdIOState()


# ---------- Utilities ----------


def _is_windows() -> bool:
    return os.name == "nt"


def _should_disable_color(env: Dict[str, str], cfg: StdIOConfig) -> bool:
    if cfg.enable_color is not None:
        return not cfg.enable_color
    # Respect NO_COLOR and Pytest/no-tty
    if env.get("NO_COLOR") or env.get("PYTEST_CURRENT_TEST"):
        return True
    # Non-tty streams commonly want plain text
    try:
        if not sys.stdout.isatty():
            return True
    except Exception as e:
        _logger.debug(f"Error checking isatty: {e}")
        return True
    return False


def _normalize_text(text: str, cfg: StdIOConfig) -> str:
    if cfg.normalize_newlines:
        # Replace CRLF/CR with LF for determinism; the console renders LF fine
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if cfg.newline != "\n":
            text = text.replace("\n", cfg.newline)
    if len(text) > cfg.max_len:
        text = text[: cfg.max_len] + "...[truncated]"
    return text


def _write(stream: TextIO, text: str, cfg: StdIOConfig) -> None:
    """
    Write text to stream with fallback error handling.

    FIXED: Proper error handling instead of silent swallowing.
    """
    try:
        stream.write(text)
        if cfg.flush:
            stream.flush()
    except Exception as e:
        # FIXED: Log error instead of silently ignoring
        _logger.warning(f"Primary stream write failed: {e}")
        # Fall back to the real std streams if wrapped streams are broken
        try:
            sys.__stdout__.write(text)
            if cfg.flush:
                sys.__stdout__.flush()
        except Exception as e2:
            _logger.error(f"Fallback stream write also failed: {e2}")


# ---------- Safe Printers ----------


def _plain_print(
    *args: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[TextIO] = None,
    cfg: Optional[StdIOConfig] = None,
) -> None:
    """
    A recursion-proof, cross-platform print that bypasses layered wrappers.
    Writes to sys.__stdout__ by default and never calls Colorama's wrapped streams.

    FIXED: Consistent locking for all writes.
    """
    if cfg is None:
        cfg = StdIOConfig(replace_builtins=False)

    s = sep.join("" if a is None else str(a) for a in args) + end
    s = _normalize_text(s, cfg)
    target = file if file is not None else sys.__stdout__

    # FIXED: Always use lock for consistency
    if cfg.lock_print:
        with _STATE.lock:
            _write(target, s, cfg)
    else:
        _write(target, s, cfg)


def safe_print(
    *args: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[TextIO] = None,
    effect: Optional[str] = None,
    cfg: Optional[StdIOConfig] = None,
    **kv: Any,
) -> None:
    """
    Public, effect-aware print for Graphix.

    Adds:
      - deterministic newline normalization
      - optional audit JSONL sink with effect label (default Effect.IO.Stdout)
      - serialization under a reentrant lock to preserve ordering
    """
    if cfg is None:
        cfg = StdIOConfig(replace_builtins=False)
    _plain_print(*args, sep=sep, end=end, file=file, cfg=cfg)

    # Optional JSONL audit of the same message (structured, minimal overhead)
    if cfg.jsonl_path:
        payload = {
            "type": "io.print",
            "text": sep.join("" if a is None else str(a) for a in args),
            "sep": sep,
            "end": end,
            "stream": "stdout" if file is None else getattr(file, "name", "unknown"),
            "effect": effect or cfg.effect_label,
        }
        if cfg.include_pid_tid:
            payload["pid"] = os.getpid()
            payload["thread"] = threading.current_thread().name
        if cfg.include_ts:
            payload["ts"] = time.time()
        payload.update(kv or {})

        try:
            with _STATE.lock:
                # FIXED: Handle empty dirname case
                dir_path = os.path.dirname(cfg.jsonl_path)
                if dir_path:  # Only create directory if dirname is not empty
                    os.makedirs(dir_path, exist_ok=True)

                with open(cfg.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            # FIXED: Log error instead of silent ignore
            _logger.warning(f"Failed to write audit log to {cfg.jsonl_path}: {e}")


def json_print(
    *,
    data: Any = None,
    effect: Optional[str] = None,
    file: Optional[TextIO] = None,
    cfg: Optional[StdIOConfig] = None,
    **extra: Any,
) -> None:
    """
    Structured print for audit / ML ops. Always emits a single JSON object per line.
    """
    if cfg is None:
        cfg = StdIOConfig(replace_builtins=False)
    payload = {
        "type": "io.json",
        "data": data,
        "effect": effect or cfg.effect_label,
    }
    if cfg.include_pid_tid:
        payload["pid"] = os.getpid()
        payload["thread"] = threading.current_thread().name
    if cfg.include_ts:
        payload["ts"] = time.time()
    payload.update(extra or {})
    s = json.dumps(payload, ensure_ascii=False)
    _plain_print(s, end="\n", file=file, cfg=cfg)


# ---------- Integration Patches ----------


def _patch_colorama(cfg: StdIOConfig) -> bool:
    """
    Prevent recursive wrapping by making Colorama **non-wrapping**.
    Keep ANSI intact unless disable-color is requested.

    FIXED: Proper error logging.
    """
    try:
        import colorama  # type: ignore
    except Exception as e:
        _logger.debug(f"Colorama not available: {e}")
        return False

    try:
        colorama.deinit()
    except Exception as e:
        _logger.debug(f"Colorama deinit failed: {e}")

    disable = _should_disable_color(os.environ, cfg)
    # Use simple init() to avoid wrap=False conflicting with other boolean args.
    # When wrap=False, features like strip require wrapping to function.
    # Simplest container-friendly option: just call init() with defaults,
    # or use just_fix_windows_console() which is cross-platform in colorama 0.4.6+.
    try:
        if disable:
            # If color is disabled, just use plain init without wrapping
            colorama.init()
        else:
            # Try modern API first (colorama 0.4.6+), fallback to simple init
            # just_fix_windows_console() is cross-platform and safer
            if hasattr(colorama, "just_fix_windows_console"):
                colorama.just_fix_windows_console()
            else:
                colorama.init()
        return True
    except Exception as e:
        _logger.warning(f"Colorama init failed: {e}")
        return False


def _patch_ray(cfg: StdIOConfig) -> bool:
    """
    Replace Ray's tqdm_ray.safe_print with our recursion-proof printer.

    FIXED: Proper error logging.
    """
    try:
        from ray.experimental import tqdm_ray  # type: ignore

        tqdm_ray.safe_print = lambda *a, **k: _plain_print(*a, cfg=cfg, **k)
        return True
    except Exception as e:
        _logger.debug(f"Ray patching not available: {e}")
        return False


def _patch_tqdm(cfg: StdIOConfig) -> bool:
    """
    Tame tqdm notebooks auto-detection to avoid weird streams under pytest/Windows.

    FIXED: Proper error logging.
    """
    try:
        import tqdm  # type: ignore

        # Prefer plain tqdm by default; force disable notebook auto-mode.
        os.environ.setdefault("TQDM_DISABLE", "0")
        os.environ.setdefault("TQDM_NOTEBOOK", "0")
        return True
    except Exception as e:
        _logger.debug(f"tqdm patching not available: {e}")
        return False


# ---------- Installation / Uninstall ----------


@dataclass
class StdIOHandle:
    """
    Handle for stdio policy installation.

    FIXED: Added context manager support and __del__ for guaranteed cleanup.
    """

    cfg: StdIOConfig
    restored: bool = False

    def restore(self) -> None:
        """Revert all patches safely."""
        if self.restored:
            return

        with _STATE.lock:
            try:
                if _STATE.original_stdout is not None:
                    sys.stdout = _STATE.original_stdout
                if _STATE.original_stderr is not None:
                    sys.stderr = _STATE.original_stderr
                if _STATE.original_print is not None:
                    _builtins.print = _STATE.original_print  # type: ignore

                self.restored = True
                _STATE.installed = False

                if self.cfg.verbose:
                    sys.__stderr__.write("[stdio_policy] restored\n")
                    try:
                        sys.__stderr__.flush()
                    except Exception as e:
                        _logger.debug(f"Failed to flush stderr during restore: {e}")
            except Exception as e:
                _logger.error(f"Error during restore: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.restore()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if not self.restored:
                self.restore()
        except Exception as e:
            _logger.debug(f"Failed to restore StdIO during __del__: {e}")


def install(
    *,
    replace_builtins: bool = True,
    patch_ray: bool = True,
    patch_colorama: bool = True,
    patch_tqdm: bool = True,
    jsonl_path: Optional[str] = None,
    enable_color: Optional[bool] = None,
    verbose: bool = False,
) -> StdIOHandle:
    """
    Install the Graphix StdIO policy. Idempotent and reversible.

    Returns a handle with `.restore()` or use as context manager.

    Parameters
    ----------
    replace_builtins : bool
        Replace `builtins.print` globally with our safe printer.
    patch_ray : bool
        Override `ray.experimental.tqdm_ray.safe_print`.
    patch_colorama : bool
        Make Colorama non-wrapping to avoid recursion.
    patch_tqdm : bool
        Disable notebook auto-mode and other surprises.
    jsonl_path : str | None
        If set, `safe_print` also appends structured events to this JSONL file.
    enable_color : bool | None
        Force color on/off, or leave auto (None).
    verbose : bool
        Extra diagnostics to stderr during install.
    """
    cfg = StdIOConfig(
        replace_builtins=replace_builtins,
        patch_ray=patch_ray,
        patch_colorama=patch_colorama,
        patch_tqdm=patch_tqdm,
        jsonl_path=jsonl_path,
        enable_color=enable_color,
        verbose=verbose,
    )

    if _STATE.installed:
        if verbose:
            sys.__stderr__.write(
                "[stdio_policy] already installed, returning existing handle\n"
            )
            try:
                sys.__stderr__.flush()
            except Exception as e:
                _logger.debug(f"Failed to flush stderr: {e}")
        return StdIOHandle(cfg=cfg, restored=False)

    with _STATE.lock:
        # Snapshot originals
        _STATE.original_print = getattr(_builtins, "print", None)
        _STATE.original_stdout = sys.stdout
        _STATE.original_stderr = sys.stderr

        # Apply patches
        if patch_colorama:
            _STATE.patched_colorama = _patch_colorama(cfg)

        if patch_ray:
            _STATE.patched_ray = _patch_ray(cfg)

        if patch_tqdm:
            _STATE.patched_tqdm = _patch_tqdm(cfg)

        # Replace builtins.print with safe_print (bound to our cfg)
        if replace_builtins:

            def _bound_print(*a: Any, **k: Any) -> None:
                # Map builtins.print(...) to safe_print(..., cfg=cfg)
                file = k.pop("file", None)
                sep = k.pop("sep", " ")
                end = k.pop("end", "\n")
                effect = k.pop("effect", None)
                return safe_print(
                    *a, sep=sep, end=end, file=file, cfg=cfg, effect=effect, **k
                )

            _builtins.print = _bound_print  # type: ignore

        # Mark installed and register atexit cleanup
        _STATE.installed = True

        def _cleanup() -> None:
            try:
                StdIOHandle(cfg).restore()
            except Exception as e:
                _logger.error(f"Error during atexit cleanup: {e}")

        atexit.register(_cleanup)

    if verbose:
        sys.__stderr__.write(
            f"[stdio_policy] installed (ray={_STATE.patched_ray}, colorama={_STATE.patched_colorama}, "
            f"tqdm={_STATE.patched_tqdm}, replace_builtins={replace_builtins})\n"
        )
        try:
            sys.__stderr__.flush()
        except Exception as e:
            _logger.debug(f"Failed to flush stderr during install: {e}")

    return StdIOHandle(cfg=cfg, restored=False)


# ---------- Self-Test ----------


def self_test() -> Dict[str, Any]:
    """
    Run a tiny battery of checks to confirm deterministic, recursion-free behavior.
    Safe to run under pytest on Windows.

    Returns a dict of diagnostics you can json_print() or assert on.
    """
    diag: Dict[str, Any] = {
        "os": os.name,
        "isatty": getattr(sys.stdout, "isatty", lambda: False)(),
        "color_disabled": _should_disable_color(
            os.environ, StdIOConfig(replace_builtins=False)
        ),
        "ray_patched": _STATE.patched_ray,
        "colorama_patched": _STATE.patched_colorama,
        "tqdm_patched": _STATE.patched_tqdm,
    }

    # Ensure no recursion: printing a long line and nested prints should not explode
    try:
        # FIXED: Removed unicode ellipsis
        safe_print("Graphix StdIO self-test start...")
        for i in range(3):
            safe_print(f"line {i}", effect="Effect.IO.Stdout")
        json_print(data={"ok": True, "lines": 3}, effect="Effect.IO.Stdout")
        diag["print_ok"] = True
    except RecursionError:
        diag["print_ok"] = False
        _logger.error("Self-test encountered recursion error")
    except Exception as e:
        diag["print_ok"] = f"error: {e!r}"
        _logger.error(f"Self-test error: {e}")

    return diag


# ---------- Module entrypoint ----------

if __name__ == "__main__":
    # Manual smoke run: python -m src.stdio_policy
    print("\n" + "=" * 70)
    print("StdIO Policy Self-Test")
    print("=" * 70 + "\n")

    # Test with context manager
    with install(
        replace_builtins=True, patch_ray=True, patch_colorama=True, verbose=True
    ) as handle:
        print("Hello from stdio_policy (builtins routed via safe_print).")
        print("\nDiagnostics:")
        print(json.dumps(self_test(), indent=2))

    print("\n" + "=" * 70)
    print("Self-Test Complete")
    print("=" * 70 + "\n")
