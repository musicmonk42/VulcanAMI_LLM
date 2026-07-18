"""Governed self-improvement transaction.

Dependency-light transaction owner for Phase 8.  All source mutation is CAS,
policy, approval, audit, and changed-tree verification gated.  The module never
commits, pushes, evaluates plan callables, or executes model supplied commands.
"""
from __future__ import annotations

import ast
import fcntl
import difflib
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "vulcan-improvement-proposal/1"
DISABLED_ENV = "VULCAN_AUTO_APPLY_DISABLED"
_ALLOWED_PROPOSAL_KEYS = frozenset({
    "schema_version", "proposal_id", "objective_type", "target_path",
    "expected_original_sha256", "original_content", "candidate_content",
    "candidate_sha256", "inspected_source_digest", "generator_identity",
    "provider_release_digest", "rationale", "expected_policy_digest",
    "approval_id",
})
_DENY_PREFIXES = (
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    "data", "audit", "logs", "memory", "models", "weights", "secrets", ".github/workflows",
)
_DENY_NAMES = {".env", "auto_apply_policy.yaml"}
_TERMINAL = {"applied", "rejected", "aborted", "verification_failed"}

class TransactionStatus(str, Enum):
    APPLIED_AND_VERIFIED = "applied_and_verified"
    REJECTED_BEFORE_INSTALLATION = "rejected_before_installation"
    VERIFICATION_FAILED_ROLLBACK_SUCCEEDED = "verification_failed_rollback_succeeded"
    VERIFICATION_FAILED_ROLLBACK_FAILED = "verification_failed_rollback_failed"
    EXTERNAL_MUTATION_PREVENTED_ROLLBACK = "external_mutation_prevented_rollback"

    @property
    def verified_success(self) -> bool:
        return self is TransactionStatus.APPLIED_AND_VERIFIED

    @property
    def rollback_succeeded(self) -> bool:
        return self is TransactionStatus.VERIFICATION_FAILED_ROLLBACK_SUCCEEDED


class TransactionError(Exception):
    """Fail-closed transaction validation/application error."""

@dataclass(frozen=True)
class VerificationGate:
    identity: str
    argv: Tuple[str, ...]
    timeout_s: float = 10.0
    output_limit: int = 20000
    env: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class ImprovementPolicy:
    schema_version: str
    enabled: bool
    repo_root: Path
    permitted_objectives: Tuple[str, ...]
    permitted_path_globs: Tuple[str, ...]
    denied_path_globs: Tuple[str, ...]
    max_files: int
    max_candidate_bytes: int
    max_changed_lines: int
    approval_required: bool
    permitted_generators: Mapping[str, Tuple[str, ...]]
    verification_gates: Tuple[VerificationGate, ...]
    timeout_s: float
    output_limit: int
    audit_required: bool
    clean_changed_file_set: bool = True
    allow_hardlinks: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        body = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "digest"}
        dg = _sha256(json.dumps(body, sort_keys=True, default=str, separators=(",", ":")).encode())
        object.__setattr__(self, "digest", dg)

@dataclass(frozen=True)
class InspectionSnapshot:
    repo_root: Path
    files: Mapping[str, str]
    contents: Mapping[str, str]
    digest: str

@dataclass(frozen=True)
class ImprovementProposal:
    schema_version: str
    proposal_id: str
    objective_type: str
    target_path: str
    expected_original_sha256: str
    original_content: str
    candidate_content: str
    candidate_sha256: str
    inspected_source_digest: str
    generator_identity: str
    provider_release_digest: str
    rationale: str = ""
    expected_policy_digest: str = ""
    approval_id: str = ""

    @classmethod
    def from_json(cls, text: str) -> "ImprovementProposal":
        def pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in pairs:
                if k in out:
                    raise TransactionError(f"duplicate proposal key: {k}")
                out[k] = v
            return out
        obj = json.loads(text, object_pairs_hook=pairs_hook)
        return cls.from_mapping(obj)

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "ImprovementProposal":
        if not isinstance(obj, Mapping):
            raise TransactionError("proposal must be a mapping")
        unknown = set(obj) - _ALLOWED_PROPOSAL_KEYS
        if unknown:
            raise TransactionError(f"unknown proposal fields: {sorted(unknown)}")
        required = _ALLOWED_PROPOSAL_KEYS - {"rationale", "approval_id"}
        missing = [k for k in sorted(required) if k not in obj]
        if missing:
            raise TransactionError(f"missing proposal fields: {missing}")
        vals = {k: obj.get(k, "") for k in _ALLOWED_PROPOSAL_KEYS}
        if not all(isinstance(v, str) for v in vals.values()):
            raise TransactionError("proposal fields must be strings")
        return cls(**{k: vals[k] for k in cls.__dataclass_fields__})

    def digest(self) -> str:
        data = {k: getattr(self, k) for k in sorted(self.__dataclass_fields__) if k != "approval_id"}
        return _sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class TrustedApprovalPrincipal:
    principal_id: str
    scopes: Tuple[str, ...]
    expires_at: float

class ApprovalAuthorityPort:
    def is_authorized(self, principal: TrustedApprovalPrincipal, bindings: Mapping[str, str]) -> bool:
        raise NotImplementedError

class ClosedApprovalAuthority(ApprovalAuthorityPort):
    def __init__(self, principals: Mapping[str, TrustedApprovalPrincipal] = None):
        self._principals = dict(principals or {})
    def issue_principal(self, principal_id: str, scopes: Sequence[str], ttl_seconds: float = 3600.0) -> TrustedApprovalPrincipal:
        if not isinstance(principal_id, str) or not principal_id or len(principal_id) > 128:
            raise TransactionError("bad principal")
        p = TrustedApprovalPrincipal(principal_id, tuple(str(s) for s in scopes), time.time()+float(ttl_seconds))
        self._principals[principal_id] = p
        return p
    def is_authorized(self, principal: TrustedApprovalPrincipal, bindings: Mapping[str, str]) -> bool:
        if not isinstance(principal, TrustedApprovalPrincipal): return False
        stored = self._principals.get(principal.principal_id)
        if stored != principal or principal.expires_at < time.time(): return False
        required = str(bindings.get("required_scope", "self_improvement.approve"))
        if required not in principal.scopes: return False
        return all(isinstance(bindings.get(k), str) and bindings.get(k) for k in ("approval_id","proposal_digest","policy_digest","original_source_digest"))

@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    proposal_digest: str
    policy_digest: str
    original_source_digest: str
    approver_identity: str
    approved_at: float
    expires_at: float
    state: str = "approved"
    used: bool = False
    def __post_init__(self):
        import math, re
        for n in ("approval_id","approver_identity"):
            v=getattr(self,n)
            if not isinstance(v,str) or not v or len(v)>128 or any(ord(c)<32 for c in v): raise TransactionError("bad approval identifier")
        for n in ("proposal_digest","policy_digest","original_source_digest"):
            if not isinstance(getattr(self,n),str) or len(getattr(self,n))>128 or any(ord(c)<32 for c in getattr(self,n)): raise TransactionError("bad approval digest")
        if not all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in (self.approved_at,self.expires_at)): raise TransactionError("bad approval timestamp")
        if self.expires_at <= self.approved_at: raise TransactionError("bad approval expiry")
        if self.state not in {"approved","claimed","consumed","rejected","expired","verification_failed","aborted","manual_recovery_required"}: raise TransactionError("bad approval state")

@dataclass
class TransactionResult:
    status: TransactionStatus | str
    state: str
    proposal_digest: str = ""
    target_path: str = ""
    failure_category: str = ""
    gate_results: List[Dict[str, Any]] = field(default_factory=list)
    rollback_digest: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = TransactionStatus(self.status)

    @property
    def verified_success(self) -> bool:
        return self.status.verified_success

    @property
    def status_code(self) -> str:
        return self.status.value

class ApprovalStore:
    def __init__(self, path: Path):
        self.path = Path(path)
    def _check_paths(self) -> None:
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        if self.path.is_symlink() or lock.is_symlink(): raise TransactionError("symlinked approval store or lock")

    def load(self, approval_id: str) -> ApprovalRecord:
        self._check_paths()
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TransactionError("approval store unavailable or corrupt") from exc
        rec = doc.get(approval_id)
        if not isinstance(rec, dict):
            raise TransactionError("approval not found")
        return ApprovalRecord(**rec)
    def _strict_record(self, record: ApprovalRecord) -> None:
        import re
        for n in ("proposal_digest","policy_digest","original_source_digest"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(record,n)): raise TransactionError("bad approval digest")

    def save(self, record: ApprovalRecord) -> None:
        self._check_paths(); self._strict_record(record)
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except Exception as exc:
            raise TransactionError("approval store unavailable or corrupt") from exc
        if record.approval_id in doc:
            raise TransactionError("approval already exists")
        doc[record.approval_id] = record.__dict__.copy()
        _atomic_write(self.path, json.dumps(doc, sort_keys=True).encode(), 0o600)
    def claim(self, approval_id: str, proposal_digest: str) -> ApprovalRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._check_paths()
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        fd = os.open(lock, os.O_CREAT|os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            doc = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            recd = doc.get(approval_id)
            if not isinstance(recd, dict): raise TransactionError("approval not found")
            rec = ApprovalRecord(**recd)
            if rec.used or rec.state != "approved" or rec.proposal_digest != proposal_digest: raise TransactionError("approval already consumed or not claimable")
            recd["state"] = "claimed"; doc[approval_id] = recd
            _atomic_write(self.path, json.dumps(doc, sort_keys=True).encode(), 0o600)
            return ApprovalRecord(**recd)
        finally:
            try: fcntl.flock(fd, fcntl.LOCK_UN)
            finally: os.close(fd)
    def terminalize(self, approval_id: str, state: str) -> None:
        self._check_paths()
        if state not in {"consumed","rejected","expired","verification_failed","aborted","manual_recovery_required"}:
            raise TransactionError("bad terminal approval state")
        doc = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if approval_id not in doc: raise TransactionError("approval not found")
        if doc[approval_id].get("state") in {"consumed","rejected","expired","verification_failed","aborted"}: raise TransactionError("approval already terminal")
        doc[approval_id]["used"] = True; doc[approval_id]["state"] = state
        _atomic_write(self.path, json.dumps(doc, sort_keys=True).encode(), 0o600)
    def mark_used(self, approval_id: str) -> None:
        self.terminalize(approval_id, "consumed")

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _rel_path(path: str) -> Path:
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise TransactionError("invalid target path")
    if any(ord(c) < 32 for c in path):
        raise TransactionError("control character in target path")
    p = Path(path)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise TransactionError("ambiguous or traversing target path")
    if p.suffix != ".py":
        raise TransactionError("target must be an existing Python file")
    return p

def resolve_repo_root(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink():
        raise TransactionError("repository root symlink rejected")
    resolved = root.resolve()
    if str(resolved) in {"/", str(Path.home())} or len(resolved.parts) < 3:
        raise TransactionError("broad repository root rejected")
    if not (resolved / ".git").exists():
        raise TransactionError("repository marker .git missing")
    return resolved

def resolve_target(repo_root: Path, rel: str) -> Path:
    rp = _rel_path(rel)
    cur = repo_root
    for part in rp.parts:
        cur = cur / part
        if cur.is_symlink():
            raise TransactionError("symlink in target path rejected")
    target = (repo_root / rp).resolve()
    if not str(target).startswith(str(repo_root) + os.sep):
        raise TransactionError("target escapes repository root")
    return target

def inspect_repository(repo_root: Path, allow_globs: Sequence[str], *, max_files: int = 100, max_file_bytes: int = 200000, max_total_bytes: int = 1000000) -> InspectionSnapshot:
    root = resolve_repo_root(repo_root)
    files: Dict[str, str] = {}; contents: Dict[str, str] = {}; total = 0
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if _denied(rel) or not _glob_any(rel, allow_globs) or f.is_symlink() or not f.is_file():
            continue
        data = f.read_bytes()
        if len(data) > max_file_bytes: continue
        total += len(data)
        if len(files) >= max_files or total > max_total_bytes: raise TransactionError("inspection bounds exceeded")
        try: text = data.decode("utf-8")
        except UnicodeDecodeError: continue
        digest = _sha256(data); files[rel] = digest; contents[rel] = text
    snap = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return InspectionSnapshot(root, files, contents, _sha256(snap))

def _glob_any(path: str, globs: Sequence[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(path, g) for g in globs if _safe_glob(g))

def _safe_glob(g: str) -> bool:
    return bool(g) and not g.startswith("/") and "\\" not in g and ".." not in Path(g).parts

def _denied(rel: str) -> bool:
    parts = rel.split("/")
    return parts[0] in _DENY_PREFIXES or any(rel == p or rel.startswith(p + "/") for p in _DENY_PREFIXES) or parts[-1] in _DENY_NAMES

def load_governed_policy(path: Path) -> ImprovementPolicy:
    raw = Path(path).read_bytes(); doc = json.loads(raw.decode("utf-8")) if path.suffix == ".json" else __import__("yaml").safe_load(raw)
    node = doc.get("auto_apply", doc)
    root = resolve_repo_root(Path(node["repository_root"]))
    gates = tuple(VerificationGate(str(g["id"]), tuple(g["argv"]), float(g.get("timeout_s", node.get("timeout_s", 10))), int(g.get("output_limit", node.get("output_limit", 20000))), dict(g.get("env", {}))) for g in node.get("verification_gates", []))
    digest = _sha256(json.dumps(node, sort_keys=True, default=str).encode())
    return ImprovementPolicy(str(node.get("schema_version", "auto-apply-policy/2")), bool(node.get("enabled", False)), root, tuple(node.get("permitted_objective_types", [])), tuple(node.get("permitted_path_globs", [])), tuple(node.get("denied_path_globs", [])), int(node.get("max_files", 1)), int(node.get("max_candidate_bytes", 50000)), int(node.get("max_changed_lines", 200)), bool(node.get("approval_required", True)), {str(k): tuple(v) for k, v in dict(node.get("permitted_generators", {})).items()}, gates, float(node.get("timeout_s", 10)), int(node.get("output_limit", 20000)), bool(node.get("audit_required", True)), bool(node.get("clean_changed_file_set", True)), bool(node.get("allow_hardlinks", False)), digest)

class GovernedSelfImprovementTransaction:
    def __init__(self, policy: Optional[ImprovementPolicy], audit_owner: Any, approval_store: Optional[ApprovalStore] = None, gate_runner: Any = None):
        self.policy = policy; self.audit = audit_owner; self.approval_store = approval_store; self.gate_runner = gate_runner or self._run_gate; self._unresolved = False
    def readiness(self) -> Tuple[bool, str]:
        if self._unresolved: return False, "unresolved improvement transaction"
        if os.environ.get(DISABLED_ENV, "1") != "0": return False, "disabled by default"
        if not self.policy or not self.policy.enabled: return False, "policy disabled or missing"
        if not self.audit: return False, "audit owner missing"
        if not self.policy.verification_gates: return False, "verification gates missing"
        return True, "ready"
    def apply(self, proposal: ImprovementProposal, snapshot: InspectionSnapshot, actor: str = "offline-owner") -> TransactionResult:
        pd = proposal.digest()
        try:
            ok, reason = self.readiness()
            if not ok: raise TransactionError(reason)
            policy = self.policy; assert policy is not None
            repo = resolve_repo_root(policy.repo_root); target = resolve_target(repo, proposal.target_path)
            self._validate(policy, proposal, snapshot, target)
            self._audit("improvement.proposed", proposal, pd, actor, state="proposed")
            approval = self._require_approval(policy, proposal, pd)
            self._audit("improvement.approved", proposal, pd, actor, approver=approval.approver_identity, state="approved")
            st = target.stat(); original = target.read_bytes(); original_digest = _sha256(original)
            if original_digest != proposal.expected_original_sha256: raise TransactionError("stale original digest")
            if not policy.allow_hardlinks and getattr(st, "st_nlink", 1) > 1: raise TransactionError("unexpected hardlink")
            cand = proposal.candidate_content.encode("utf-8")
            ast.parse(proposal.candidate_content)
            self._audit("improvement.apply_prepared", proposal, pd, actor, state="applying")
            self._unresolved = True
            safe_mode = stat.S_IMODE(st.st_mode) & 0o777 & ~0o6000
            _atomic_write(target, cand, safe_mode)
            installed_digest = _sha256(target.read_bytes())
            if installed_digest != proposal.candidate_sha256: raise TransactionError("install digest mismatch")
            self._audit("improvement.candidate_installed", proposal, pd, actor, installed_digest=installed_digest)
            gate_results = []
            for gate in policy.verification_gates:
                gr = self.gate_runner(gate, repo, policy)
                gate_results.append(gr); self._audit("improvement.gate_completed", proposal, pd, actor, gate=gr)
                if not gr.get("ok") or _sha256(target.read_bytes()) != proposal.candidate_sha256:
                    raise GateFailure("gate failed or mutated candidate", gate_results)
            self._unresolved = False
            if self.approval_store and proposal.approval_id: self.approval_store.mark_used(proposal.approval_id)
            self._audit("improvement.applied", proposal, pd, actor, result_digest=proposal.candidate_sha256)
            return TransactionResult(TransactionStatus.APPLIED_AND_VERIFIED, "applied", pd, proposal.target_path, gate_results=gate_results)
        except GateFailure as exc:
            return self._rollback(proposal, pd, target, original, safe_mode, exc.gate_results, str(exc))
        except Exception as exc:
            try: self._audit("improvement.aborted", proposal, pd, actor, failure_category=str(exc)[:200], state="aborted")
            except Exception: pass
            return TransactionResult(TransactionStatus.REJECTED_BEFORE_INSTALLATION, "rejected", pd, proposal.target_path, str(exc)[:200])
    def _validate(self, policy: ImprovementPolicy, p: ImprovementProposal, snapshot: InspectionSnapshot, target: Path) -> None:
        if p.schema_version != SCHEMA_VERSION: raise TransactionError("bad schema")
        if p.objective_type not in policy.permitted_objectives: raise TransactionError("unknown objective")
        rel = p.target_path
        if rel not in snapshot.files or p.inspected_source_digest != snapshot.digest: raise TransactionError("target was not inspected")
        if _denied(rel) or _glob_any(rel, policy.denied_path_globs) or not _glob_any(rel, policy.permitted_path_globs): raise TransactionError("protected or unpermitted path")
        if not target.exists() or not target.is_file(): raise TransactionError("new file or directory rejected")
        if p.expected_original_sha256 != snapshot.files[rel] or _sha256(p.original_content.encode()) != p.expected_original_sha256: raise TransactionError("missing or mismatched original")
        cand = p.candidate_content.encode("utf-8")
        if not cand or len(cand) > policy.max_candidate_bytes or _sha256(cand) != p.candidate_sha256: raise TransactionError("candidate digest/size invalid")
        if p.candidate_content == p.original_content: raise TransactionError("identical candidate")
        if "```" in p.candidate_content or "..." in p.candidate_content or "TODO" == p.candidate_content.strip(): raise TransactionError("unbounded model wrapper or placeholder")
        ast.parse(p.candidate_content)
        changed = sum(1 for l in difflib.unified_diff(p.original_content.splitlines(), p.candidate_content.splitlines(), lineterm="") if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
        if policy.max_files != 1 or changed > policy.max_changed_lines: raise TransactionError("change bounds exceeded")
        if p.generator_identity not in policy.permitted_generators or p.provider_release_digest not in policy.permitted_generators[p.generator_identity]: raise TransactionError("unknown provider/release")
        if p.expected_policy_digest != policy.digest: raise TransactionError("policy digest mismatch")
    def _require_approval(self, policy: ImprovementPolicy, p: ImprovementProposal, pd: str) -> ApprovalRecord:
        if not policy.approval_required: return ApprovalRecord("not-required", pd, policy.digest, p.expected_original_sha256, "policy", time.time(), time.time()+1)
        if not self.approval_store or not p.approval_id: raise TransactionError("approval required")
        rec = self.approval_store.claim(p.approval_id, pd) if hasattr(self.approval_store, "claim") else self.approval_store.load(p.approval_id)
        if rec.used or rec.state not in {"approved","claimed"} or rec.expires_at < time.time() or rec.proposal_digest != pd or rec.policy_digest != policy.digest or rec.original_source_digest != p.expected_original_sha256:
            raise TransactionError("approval binding invalid")
        return rec
    def _run_gate(self, gate: VerificationGate, repo: Path, policy: ImprovementPolicy) -> Dict[str, Any]:
        if not gate.argv or gate.argv[0] == "git" and any(a in {"commit", "push"} for a in gate.argv): raise TransactionError("forbidden gate command")
        start = time.time(); env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(repo / "src")}; env.update(gate.env)
        try:
            proc = subprocess.run(list(gate.argv), cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=gate.timeout_s, shell=False, check=False)
            out = (proc.stdout + proc.stderr)[:gate.output_limit]
            return {"identity": gate.identity, "argv_digest": _sha256(json.dumps(list(gate.argv)).encode()), "exit_status": proc.returncode, "timeout": False, "output_digest": _sha256(out), "started_at": start, "ended_at": time.time(), "ok": proc.returncode == 0}
        except subprocess.TimeoutExpired as exc:
            return {"identity": gate.identity, "argv_digest": _sha256(json.dumps(list(gate.argv)).encode()), "exit_status": None, "timeout": True, "output_digest": "", "started_at": start, "ended_at": time.time(), "ok": False}
    def _rollback(self, p: ImprovementProposal, pd: str, target: Path, original: bytes, mode: int, gate_results: List[Dict[str, Any]], why: str) -> TransactionResult:
        try:
            if _sha256(target.read_bytes()) != p.candidate_sha256:
                self._audit("improvement.manual_recovery_required", p, pd, "system", failure_category="external mutation prevented rollback")
                self._unresolved = False
                self.approval_store.terminalize(p.approval_id, "verification_failed") if self.approval_store and p.approval_id else None
                return TransactionResult(TransactionStatus.EXTERNAL_MUTATION_PREVENTED_ROLLBACK, "verification_failed", pd, p.target_path, why, gate_results)
            _atomic_write(target, original, mode)
            rd = _sha256(target.read_bytes())
            if rd != p.expected_original_sha256: raise TransactionError("rollback digest mismatch")
            self._audit("improvement.rollback_completed", p, pd, "system", rollback_digest=rd)
            self._unresolved = False
            self.approval_store.terminalize(p.approval_id, "verification_failed") if self.approval_store and p.approval_id else None
            return TransactionResult(TransactionStatus.VERIFICATION_FAILED_ROLLBACK_SUCCEEDED, "verification_failed", pd, p.target_path, why, gate_results, rd)
        except Exception as exc:
            self._unresolved = False
            try: self._audit("improvement.manual_recovery_required", p, pd, "system", failure_category=str(exc)[:200])
            except Exception: pass
            self.approval_store.terminalize(p.approval_id, "verification_failed") if self.approval_store and p.approval_id else None
            return TransactionResult(TransactionStatus.VERIFICATION_FAILED_ROLLBACK_FAILED, "verification_failed", pd, p.target_path, str(exc)[:200], gate_results)
    def _audit(self, event: str, p: ImprovementProposal, pd: str, actor: str, **meta: Any) -> None:
        if not self.audit: raise TransactionError("audit owner unavailable")
        payload = {"event": event, "proposal_digest": pd, "policy_digest": self.policy.digest if self.policy else "", "original_digest": p.expected_original_sha256, "candidate_digest": p.candidate_sha256, "target_path": p.target_path, "objective_type": p.objective_type, "actor": actor, "timestamp": time.time()}
        payload.update(meta)
        self.audit.record_event(event, payload)

class GateFailure(Exception):
    def __init__(self, msg: str, gate_results: List[Dict[str, Any]]):
        super().__init__(msg); self.gate_results = gate_results

def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp_path, mode & 0o777 & ~0o6000)
        os.replace(tmp_path, path)
        try:
            dfd = os.open(str(path.parent), os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
        except Exception: pass
    finally:
        try:
            if tmp_path.exists(): tmp_path.unlink()
        except Exception: pass
