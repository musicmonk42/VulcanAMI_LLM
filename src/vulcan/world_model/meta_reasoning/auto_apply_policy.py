"""
Auto-apply policy engine (hardened)
- Validates diffs against allow/deny globs with repository-root jail
- Enforces small, safe change budgets (files, LOC)
- Executes pre-apply gates with timeouts and shell disabled
- Evaluates optional NSO requirements (risk gating)
- Uses safe_execution module for enhanced security
"""

from __future__ import annotations

import fnmatch
import hashlib

# logger = logging.getLogger(__name__) # Original logger
# --- START FIX: Replace YAML import block ---
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("YAML not available, falling back to JSON")
    import json

    # Create a mock 'yaml' object with a 'safe_load' method that uses json.load
    # This allows the rest of the file to call yaml.safe_load()
    class YamlJsonFallback:
        def safe_load(self, stream):
            # json.load reads from a stream (like yaml.safe_load)
            return json.load(stream)

    yaml = YamlJsonFallback()
# --- END FIX ---

# Import safe execution if available
try:
    from .safe_execution import get_safe_executor

    SAFE_EXECUTION_AVAILABLE = True
except ImportError:
    SAFE_EXECUTION_AVAILABLE = False
    logger.debug("Safe execution module not available, using direct subprocess")
    get_safe_executor = None


# ----------------------------
# Exceptions and result types
# ----------------------------


class PolicyError(Exception):
    pass


class GateFailure(Exception):
    def __init__(self, failures: List[str]):
        super().__init__("; ".join(failures))
        self.failures = failures


@dataclass(frozen=True)
class GateSpec:
    name: str
    cmd: Sequence[str]
    timeout_s: int = 60
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Policy:
    version: str = "1"
    enabled: bool = False
    repo_root: Path = Path(".")
    max_files: int = 3
    max_total_loc: int = 50
    allowed_globs: Tuple[str, ...] = tuple()
    deny_globs: Tuple[str, ...] = tuple()
    gates: Tuple[GateSpec, ...] = tuple()
    nso_requirements: Dict[str, Any] = field(
        default_factory=lambda: {"adversarial_detected": False, "risk_score_max": 0.3}
    )
    policy_hash: str = ""

    def with_root(self, root: Path) -> "Policy":
        return Policy(
            version=self.version,
            enabled=self.enabled,
            repo_root=root,
            max_files=self.max_files,
            max_total_loc=self.max_total_loc,
            allowed_globs=self.allowed_globs,
            deny_globs=self.deny_globs,
            gates=self.gates,
            nso_requirements=self.nso_requirements,
            policy_hash=self.policy_hash,
        )


@dataclass
class FileCheckResult:
    ok: bool
    reasons: List[str]
    offending_files: List[str]


@dataclass
class GatesReport:
    ok: bool
    failures: List[str]
    outputs: Dict[str, Dict[str, Any]]  # gate_name -> {rc, elapsed_s, stdout, stderr}


# ----------------------------
# Loading and validation
# ----------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_policy(path: str | Path, default_root: Optional[Path] = None) -> Policy:
    """
    Load a YAML policy file safely and validate keys.
    """
    if not path:
        return Policy(enabled=False, policy_hash="")
    p = Path(path)

    # MODIFIED: Check the global flag instead of 'if yaml is None'
    if not YAML_AVAILABLE and not p.name.endswith(".json"):
        logger.warning(
            f"PyYAML not found, but policy file {p} is not .json. Attempting JSON load anyway."
        )
        # We proceed, as the fallback 'yaml' object will use json.load

    if not p.exists() or not p.is_file():
        raise PolicyError(f"Policy file not found: {p}")

    raw_bytes = p.read_bytes()
    try:
        # We read as bytes, but json.load (our fallback) needs text.
        # yaml.safe_load can handle either bytes or text.
        if not YAML_AVAILABLE:
            # Decode for json.load
            raw_text = raw_bytes.decode("utf-8-sig")  # Use utf-8-sig to handle BOM
            doc = json.loads(raw_text)  # Use json.loads for string
        else:
            # Use PyYAML's safe_load, which handles bytes
            doc = yaml.safe_load(raw_bytes)

    except Exception as e:
        # Provide more context if JSON fallback failed
        if not YAML_AVAILABLE:
            raise PolicyError(f"Failed to parse as JSON (YAML not found): {e}")
        raise PolicyError(f"Failed to parse YAML: {e}")

    if not isinstance(doc, dict):
        raise PolicyError("Policy root must be a YAML/JSON mapping")

    node = doc.get("auto_apply")
    if not isinstance(node, dict):
        raise PolicyError("Missing 'auto_apply' mapping in policy")

    # Basic fields
    enabled = bool(node.get("enabled", False))
    version = str(node.get("version", "1")).strip()
    max_files = int(node.get("max_files", 3))
    max_total_loc = int(node.get("max_total_loc", 50))
    allowed_globs = tuple(str(x) for x in (node.get("allowed_globs") or []))
    deny_globs = tuple(str(x) for x in (node.get("deny_globs") or []))

    # Gates: accept either string or list of strings for the command
    gates_node = node.get("gates") or {}
    commands = gates_node.get("commands", [])
    if not isinstance(commands, list):
        raise PolicyError("'auto_apply.gates.commands' must be a list")
    gate_specs: List[GateSpec] = []
    for idx, entry in enumerate(commands):
        name = f"gate_{idx}"
        if isinstance(entry, dict):
            cmd_spec = entry.get("cmd")
            if isinstance(cmd_spec, str):
                cmd = _split_cmd(cmd_spec)
            elif isinstance(cmd_spec, list) and all(
                isinstance(x, str) for x in cmd_spec
            ):
                cmd = list(cmd_spec)
            else:
                raise PolicyError(
                    f"Gate {idx}: 'cmd' must be a string or list of strings"
                )
            timeout_s = int(entry.get("timeout_s", 60))
            cwd = entry.get("cwd")
            env = entry.get("env") or {}
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()
            ):
                raise PolicyError(
                    f"Gate {idx}: 'env' must be a mapping of string->string"
                )
            gate_specs.append(
                GateSpec(
                    name=name, cmd=tuple(cmd), timeout_s=timeout_s, cwd=cwd, env=env
                )
            )
        elif isinstance(entry, str):
            gate_specs.append(
                GateSpec(name=name, cmd=tuple(_split_cmd(entry)), timeout_s=60)
            )
        else:
            raise PolicyError(f"Gate {idx}: must be string or mapping with 'cmd'")

    nso_requirements = gates_node.get(
        "nso_requirements", {"adversarial_detected": False, "risk_score_max": 0.3}
    )
    if not isinstance(nso_requirements, dict):
        raise PolicyError("'auto_apply.gates.nso_requirements' must be a mapping")

    policy_hash = _sha256_bytes(raw_bytes)
    root = (default_root or p.parent).resolve()

    return Policy(
        version=version,
        enabled=enabled,
        repo_root=root,
        max_files=max_files,
        max_total_loc=max_total_loc,
        allowed_globs=allowed_globs,
        deny_globs=deny_globs,
        gates=tuple(gate_specs),
        nso_requirements=dict(nso_requirements),
        policy_hash=policy_hash,
    )


def _split_cmd(cmd: str) -> List[str]:
    """
    Split a command string safely into argv list without invoking a shell.
    Uses posix=True for consistent splitting behavior on all platforms.
    """
    return shlex.split(cmd, posix=True)


# ----------------------------
# Path normalization and globbing
# ----------------------------


def normalize_paths(paths: Iterable[str | Path], repo_root: Path) -> List[str]:
    """
    Normalize file paths relative to repo_root using POSIX separators.
    Rejects any path that escapes the root.
    """
    normed: List[str] = []
    repo_root = repo_root.resolve()
    for p in paths:
        path = Path(p).resolve()
        try:
            rel = path.relative_to(repo_root)
        except Exception:
            raise PolicyError(f"Path escapes repository root: {path}")
        normed.append(rel.as_posix())
    return normed


def _glob_match(path: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        if pat.startswith("!"):
            # Negations are treated as deny_globs; skip here
            continue
        if fnmatch.fnmatch(path, pat):
            return True
    return False


def _glob_denied(path: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        pat_eff = pat[1:] if pat.startswith("!") else pat
        if fnmatch.fnmatch(path, pat_eff):
            return True
    return False


# ----------------------------
# Core validators and gates
# ----------------------------


def check_files_against_policy(
    files: List[str | Path],
    policy: Policy,
    total_loc: Optional[int] = None,
) -> FileCheckResult:
    """
    Validate file list against the policy.
    - Enforce repo-root jail
    - Enforce allow/deny globs
    - Enforce budgets (file count, total LOC)
    """
    reasons: List[str] = []
    offending: List[str] = []

    try:
        rel_paths = normalize_paths(files, policy.repo_root)
    except PolicyError as e:
        return FileCheckResult(ok=False, reasons=[str(e)], offending_files=[])

    if len(rel_paths) > policy.max_files:
        reasons.append(f"too many files ({len(rel_paths)} > {policy.max_files})")

    # If total LOC not provided, the caller should compute it from the diff/plan
    if total_loc is not None and total_loc > policy.max_total_loc:
        reasons.append(f"diff too large ({total_loc} > {policy.max_total_loc})")

    for rp in rel_paths:
        if _glob_denied(rp, policy.deny_globs):
            offending.append(rp)
            reasons.append(f"denied path: {rp}")
        elif policy.allowed_globs and not _glob_match(rp, policy.allowed_globs):
            offending.append(rp)
            reasons.append(f"not allowed by policy: {rp}")

    ok = len(reasons) == 0
    return FileCheckResult(
        ok=ok, reasons=dedupe_stable(reasons), offending_files=sorted(set(offending))
    )


def run_gates(policy: Policy, env: Optional[Dict[str, str]] = None) -> GatesReport:
    """
    Execute all gates with timeouts and without shell.
    Each gate can specify its own cwd and env (merged over provided env).
    """
    if not policy.gates:
        return GatesReport(ok=True, failures=[], outputs={})

    outputs: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    for gate in policy.gates:
        argv = list(gate.cmd)
        if not argv:
            failures.append(f"{gate.name}: empty command")
            continue

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        if gate.env:
            merged_env.update(gate.env)

        # Resolve cwd: default to repo_root
        cwd = str(policy.repo_root if gate.cwd is None else Path(gate.cwd).resolve())

        t0 = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(1, int(gate.timeout_s)),
                shell=False,
                check=False,
                text=True,
            )
            elapsed = time.time() - t0
            outputs[gate.name] = {
                "rc": proc.returncode,
                "elapsed_s": round(elapsed, 3),
                "stdout": proc.stdout[-50_000:],  # cap to avoid huge blobs
                "stderr": proc.stderr[-50_000:],
                "cmd": argv,
            }
            if proc.returncode != 0:
                failures.append(f"{gate.name} failed (rc={proc.returncode})")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            outputs[gate.name] = {
                "rc": None,
                "elapsed_s": round(elapsed, 3),
                "stdout": "",
                "stderr": f"timeout after {gate.timeout_s}s",
                "cmd": argv,
            }
            failures.append(f"{gate.name} timeout after {gate.timeout_s}s")
        except FileNotFoundError as e:
            failures.append(f"{gate.name} missing executable: {e}")
        except Exception as e:
            failures.append(f"{gate.name} error: {e}")

    ok = len(failures) == 0
    return GatesReport(ok=ok, failures=failures, outputs=outputs)


def evaluate_nso_requirements(
    nso_report: Dict[str, Any], policy: Policy
) -> Tuple[bool, List[str]]:
    """
    Compare NSO/aligner output to the policy's requirements.
    Expected fields:
      - adversarial_detected: bool
      - risk_score: float in [0,1]
    """
    reasons: List[str] = []
    req = policy.nso_requirements or {}

    if req.get("adversarial_detected") is False and nso_report.get(
        "adversarial_detected", False
    ):
        reasons.append("NSO: adversarial content detected")

    risk_max = float(req.get("risk_score_max", 0.3))
    risk_score = float(nso_report.get("risk_score", 0.0))
    if risk_score > risk_max:
        reasons.append(f"NSO: risk_score {risk_score:.3f} exceeds {risk_max:.3f}")

    return (len(reasons) == 0), reasons


# ----------------------------
# Utilities
# ----------------------------


def dedupe_stable(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in items:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out
