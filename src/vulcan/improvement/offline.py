"""Offline, human-authorized improvement packaging and deployment.

Nothing in this module is imported by :mod:`vulcan.runtime`.  It is an operator
surface with its own filesystem/process capability and must run outside a
serving image.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .proposal import ImprovementProposal
from vulcan.world_model.meta_reasoning.governed_transaction import (
    GovernedSelfImprovementTransaction, ImprovementPolicy, TransactionError,
    load_governed_policy, inspect_repository,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class Approval:
    schema_version: str
    approval_id: str
    proposal_digest: str
    policy_digest: str
    source_digest: str
    approver: str
    issued_at: int
    expires_at: int
    state: str = "approved"
    package_id: str = ""
    signature: str = ""


class ApprovalAuthority:
    """The sole offline issuer, verifier, and durable approval-store schema."""

    SCHEMA = "vulcan-offline-improvement-approval/1"

    def __init__(self, root: Path, secret: bytes):
        if len(secret) < 32:
            raise TransactionError("approval signing key must contain at least 32 bytes")
        self.root, self.secret = Path(root), bytes(secret)
        self.root.mkdir(parents=True, exist_ok=True)

    def _signature(self, approval: Approval) -> str:
        body = {k: v for k, v in asdict(approval).items() if k != "signature"}
        return hmac.new(self.secret, _canonical(body), hashlib.sha256).hexdigest()

    def _path(self, approval_id: str) -> Path:
        if (not approval_id.startswith("approval-") or len(approval_id) != 41
                or any(character not in "0123456789abcdef" for character in approval_id[9:])):
            raise TransactionError("invalid approval id")
        return self.root / f"{approval_id}.json"

    def issue(self, proposal: ImprovementProposal, policy: ImprovementPolicy, approver: str, *, ttl: int = 3600) -> Approval:
        if not approver or len(approver) > 128:
            raise TransactionError("invalid human approver")
        now = int(time.time())
        approval = Approval(self.SCHEMA, "approval-" + os.urandom(16).hex(), proposal.digest(), policy.digest,
                            proposal.expected_original_sha256, approver, now, now + ttl)
        approval = replace(approval, signature=self._signature(approval))
        path = self._path(approval.approval_id)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(asdict(approval))); stream.flush(); os.fsync(stream.fileno())
        return approval

    def verify(self, approval_id: str, proposal: ImprovementProposal, policy: ImprovementPolicy) -> Approval:
        path = self._path(approval_id)
        if path.is_symlink():
            raise TransactionError("symlinked approval")
        try: approval = Approval(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc: raise TransactionError("approval unavailable or invalid") from exc
        if (approval.schema_version != self.SCHEMA or not hmac.compare_digest(approval.signature, self._signature(approval))
                or approval.expires_at < int(time.time()) or approval.proposal_digest != proposal.digest()
                or approval.policy_digest != policy.digest or approval.source_digest != proposal.expected_original_sha256
                or approval.state != "approved" or approval.package_id):
            raise TransactionError("approval verification failed")
        return approval

    def consume(self, approval_id: str, proposal: ImprovementProposal, policy: ImprovementPolicy, package_id: str) -> None:
        """Atomically consume an approval so concurrent/replayed packaging fails."""
        lock_path = self.root / ".approval.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            approval = self.verify(approval_id, proposal, policy)
            consumed = replace(approval, state="consumed", package_id=package_id, signature="")
            consumed = replace(consumed, signature=self._signature(consumed))
            path = self._path(approval_id)
            temporary = self.root / f".{approval_id}.{os.urandom(8).hex()}.tmp"
            temporary.write_bytes(_canonical(asdict(consumed)))
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


@dataclass(frozen=True)
class DeploymentPackage:
    schema_version: str
    package_id: str
    proposal_digest: str
    approval_id: str
    policy_digest: str
    target_path: str
    original_digest: str
    candidate_digest: str
    candidate_content: str
    gate_results: tuple[dict[str, Any], ...]
    signature: str = ""


class OfflineImprovementOperator:
    """Review in isolation, package, deploy, and rollback with audit evidence."""

    def __init__(self, repo: Path, policy: ImprovementPolicy, authority: ApprovalAuthority, audit_path: Path, package_key: bytes):
        self.repo, self.policy, self.authority = Path(repo).resolve(), policy, authority
        self.audit_path, self.package_key = Path(audit_path), bytes(package_key)
        if len(package_key) < 32: raise TransactionError("package signing key must contain at least 32 bytes")
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _audit(self, event: str, **data: Any) -> None:
        record = {"schema_version":"vulcan-improvement-audit/1", "event":event, "at":int(time.time()), **data}
        with self.audit_path.open("ab") as stream:
            stream.write(_canonical(record) + b"\n"); stream.flush(); os.fsync(stream.fileno())

    def _sign(self, package: DeploymentPackage) -> str:
        return hmac.new(self.package_key, _canonical({k:v for k,v in asdict(package).items() if k != "signature"}), hashlib.sha256).hexdigest()

    def review_and_package(self, proposal: ImprovementProposal, approval_id: str, destination: Path) -> DeploymentPackage:
        approval = self.authority.verify(approval_id, proposal, self.policy)
        admitted = inspect_repository(self.repo, self.policy.permitted_path_globs)
        if admitted.digest != proposal.inspected_source_digest:
            raise TransactionError("proposal source snapshot digest changed")
        self._audit("improvement.reviewed", proposal_digest=proposal.digest(), approval_id=approval_id, approver=approval.approver)
        with tempfile.TemporaryDirectory(prefix="vulcan-improvement-") as temp:
            worktree = Path(temp) / "worktree"
            try:
                subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=self.repo,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
                isolated_policy = replace(self.policy, repo_root=worktree, digest=self.policy.digest)
                snapshot = inspect_repository(worktree, isolated_policy.permitted_path_globs)
                if snapshot.files != admitted.files:
                    raise TransactionError("isolated worktree does not match admitted source")
                # Adapter for the retired transaction's approval record; authorization
                # has already been verified by the sole authority above.
                isolated_policy = replace(isolated_policy, approval_required=False)
                sandbox_proposal = replace(proposal, inspected_source_digest=snapshot.digest,
                                           expected_policy_digest=isolated_policy.digest)
                result = GovernedSelfImprovementTransaction(isolated_policy, _AuditAdapter(self), offline_authorized=True).apply(sandbox_proposal, snapshot, "offline-operator")
                if not result.verified_success: raise TransactionError(result.failure_category or "verification failed")
            finally:
                if worktree.exists():
                    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=self.repo,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
        package = DeploymentPackage("vulcan-deployment-package/1", "package-"+os.urandom(16).hex(), proposal.digest(), approval_id,
                                    self.policy.digest, proposal.target_path, proposal.expected_original_sha256,
                                    proposal.candidate_sha256, proposal.candidate_content, tuple(result.gate_results))
        package = replace(package, signature=self._sign(package))
        destination = Path(destination)
        temporary = destination.with_name(f".{destination.name}.{package.package_id}.tmp")
        temporary.write_bytes(_canonical(asdict(package)))
        os.chmod(temporary, 0o600)
        try:
            self.authority.consume(approval_id, proposal, self.policy, package.package_id)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._audit("improvement.package_signed", proposal_digest=proposal.digest(), package_id=package.package_id)
        return package

    def install(self, package_path: Path, deployment_root: Path) -> str:
        package = DeploymentPackage(**json.loads(Path(package_path).read_text(encoding="utf-8")))
        if not hmac.compare_digest(package.signature, self._sign(package)): raise TransactionError("package signature invalid")
        root = Path(deployment_root).resolve(); target = (root / package.target_path).resolve()
        if root not in target.parents or not target.is_file(): raise TransactionError("deployment target escapes root")
        current = target.read_bytes()
        if hashlib.sha256(current).hexdigest() != package.original_digest: raise TransactionError("deployment source digest changed")
        backup = target.with_suffix(target.suffix + f".{package.package_id}.rollback")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(current); stream.flush(); os.fsync(stream.fileno())
        candidate = target.with_name(f".{target.name}.{package.package_id}.install")
        descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), target.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(package.candidate_content.encode()); stream.flush(); os.fsync(stream.fileno())
        os.replace(candidate, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != package.candidate_digest: raise TransactionError("installed digest mismatch")
        self._audit("improvement.deployed", proposal_digest=package.proposal_digest, package_id=package.package_id, candidate_digest=package.candidate_digest)
        return package.package_id

    def rollback(self, package_path: Path, deployment_root: Path) -> None:
        package = DeploymentPackage(**json.loads(Path(package_path).read_text(encoding="utf-8")))
        if not hmac.compare_digest(package.signature, self._sign(package)): raise TransactionError("package signature invalid")
        root = Path(deployment_root).resolve(); target = (root / package.target_path).resolve()
        if root not in target.parents: raise TransactionError("rollback target escapes root")
        backup = target.with_suffix(target.suffix + f".{package.package_id}.rollback")
        if hashlib.sha256(target.read_bytes()).hexdigest() != package.candidate_digest or hashlib.sha256(backup.read_bytes()).hexdigest() != package.original_digest:
            raise TransactionError("rollback state mismatch")
        os.replace(backup, target)
        self._audit("improvement.rolled_back", proposal_digest=package.proposal_digest, package_id=package.package_id, restored_digest=package.original_digest)


class _AuditAdapter:
    def __init__(self, operator: OfflineImprovementOperator): self.operator = operator
    def record_event(self, event: str, data: dict[str, Any]) -> None:
        self.operator._audit(event, **{k:v for k,v in data.items() if k not in {"event", "rationale", "original_content", "candidate_content"}})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vulcan-improvement-operator")
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True); parser.add_argument("--key-file", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    approve = sub.add_parser("approve"); approve.add_argument("proposal", type=Path); approve.add_argument("--approver", required=True)
    package = sub.add_parser("package"); package.add_argument("proposal", type=Path); package.add_argument("approval_id"); package.add_argument("output", type=Path)
    for name in ("install", "rollback"):
        command=sub.add_parser(name); command.add_argument("package", type=Path); command.add_argument("deployment_root", type=Path)
    args=parser.parse_args(argv); key=args.key_file.read_bytes(); policy=load_governed_policy(args.policy)
    authority=ApprovalAuthority(args.state/"approvals", key); operator=OfflineImprovementOperator(args.repo, policy, authority, args.state/"audit.jsonl", key)
    if args.command == "approve":
        print(authority.issue(ImprovementProposal.from_json(args.proposal.read_text()), policy, args.approver).approval_id)
    elif args.command == "package": operator.review_and_package(ImprovementProposal.from_json(args.proposal.read_text()), args.approval_id, args.output)
    elif args.command == "install": operator.install(args.package, args.deployment_root)
    else: operator.rollback(args.package, args.deployment_root)
    return 0
