from __future__ import annotations

import hashlib
import json
import os
import py_compile
import stat
import subprocess
import time
from pathlib import Path

import pytest

import importlib.util

GT_PATH = Path(__file__).resolve().parents[1] / "src/vulcan/world_model/meta_reasoning/governed_transaction.py"
spec = importlib.util.spec_from_file_location("governed_transaction", GT_PATH)
gt = importlib.util.module_from_spec(spec)
import sys
sys.modules["governed_transaction"] = gt
spec.loader.exec_module(gt)
ApprovalRecord = gt.ApprovalRecord
ApprovalStore = gt.ApprovalStore
GovernedSelfImprovementTransaction = gt.GovernedSelfImprovementTransaction
ImprovementPolicy = gt.ImprovementPolicy
ImprovementProposal = gt.ImprovementProposal
VerificationGate = gt.VerificationGate
inspect_repository = gt.inspect_repository


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class Audit:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail
    def record_event(self, event, payload):
        if self.fail:
            raise RuntimeError("audit down")
        assert "secret" not in json.dumps(payload).lower()
        assert "candidate_content" not in payload and "original_content" not in payload
        self.events.append((event, payload))


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src/pkg").mkdir(parents=True)
    target = root / "src/pkg/mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    return root


def policy(repo, *, gates=None, approval=False, max_bytes=10000, max_lines=20):
    return ImprovementPolicy(
        schema_version="auto-apply-policy/2",
        enabled=True,
        repo_root=repo,
        permitted_objectives=("bugfix",),
        permitted_path_globs=("src/**/*.py",),
        denied_path_globs=(".git/**", "configs/auto_apply_policy.yaml"),
        max_files=1,
        max_candidate_bytes=max_bytes,
        max_changed_lines=max_lines,
        approval_required=approval,
        permitted_generators={"human": ("release-1",)},
        verification_gates=tuple(gates if gates is not None else [VerificationGate("compile", ("python", "-m", "py_compile", "src/pkg/mod.py"), 5)]),
        timeout_s=5,
        output_limit=2000,
        audit_required=True,
        digest="policy-digest",
    )


def proposal(repo, pol, *, path="src/pkg/mod.py", original="VALUE = 1\n", candidate="VALUE = 2\n", approval_id=""):
    snap = inspect_repository(repo, ("src/**/*.py",))
    p = ImprovementProposal(
        schema_version=gt.SCHEMA_VERSION,
        proposal_id="p1",
        objective_type="bugfix",
        target_path=path,
        expected_original_sha256=sha(original),
        original_content=original,
        candidate_content=candidate,
        candidate_sha256=sha(candidate),
        inspected_source_digest=snap.digest,
        generator_identity="human",
        provider_release_digest="release-1",
        rationale="bounded metadata only",
        expected_policy_digest=pol.digest,
        approval_id=approval_id,
    )
    return p, snap


def run_tx(repo, pol=None, audit=None, p=None, snap=None, store=None, runner=None):
    os.environ[gt.DISABLED_ENV] = "0"
    pol = pol or policy(repo)
    if p is None:
        p, snap = proposal(repo, pol)
    return GovernedSelfImprovementTransaction(pol, audit if audit is not None else Audit(), store, runner).apply(p, snap)


def test_disabled_missing_policy_and_audit_reject_before_mutation(repo, monkeypatch):
    pol = policy(repo); p, snap = proposal(repo, pol)
    monkeypatch.setenv(gt.DISABLED_ENV, "1")
    r = GovernedSelfImprovementTransaction(pol, Audit()).apply(p, snap)
    assert r.status == "rejected_before_installation"
    monkeypatch.setenv(gt.DISABLED_ENV, "0")
    assert GovernedSelfImprovementTransaction(None, Audit()).apply(p, snap).status == "rejected_before_installation"
    assert GovernedSelfImprovementTransaction(pol, None).apply(p, snap).status == "rejected_before_installation"
    assert (repo / "src/pkg/mod.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize("bad_path", ["/x.py", "../x.py", "src//x.py", "src\\x.py", "src/pkg/t.txt"])
def test_path_traversal_absolute_ambiguity_non_python_rejected(repo, bad_path):
    pol = policy(repo); p, snap = proposal(repo, pol, path=bad_path)
    r = run_tx(repo, pol, p=p, snap=snap)
    assert r.status == "rejected_before_installation"


def test_root_component_symlink_new_file_and_protected_paths_rejected(repo, tmp_path):
    linkroot = tmp_path / "linkroot"; linkroot.symlink_to(repo)
    with pytest.raises(gt.TransactionError): gt.resolve_repo_root(linkroot)
    (repo / "src/link").symlink_to(repo / "src/pkg")
    pol = policy(repo); p, snap = proposal(repo, pol, path="src/link/mod.py")
    assert run_tx(repo, pol, p=p, snap=snap).status == "rejected_before_installation"
    p2, snap2 = proposal(repo, pol, path="src/pkg/new.py")
    assert run_tx(repo, pol, p=p2, snap=snap2).status == "rejected_before_installation"
    (repo / "configs").mkdir(); (repo / "configs/auto_apply_policy.yaml").write_text("x=1\n")
    p3, snap3 = proposal(repo, pol, path="configs/auto_apply_policy.yaml")
    assert run_tx(repo, pol, p=p3, snap=snap3).status == "rejected_before_installation"


@pytest.mark.parametrize("mut", ["stale", "digest", "noop", "syntax", "oversize", "loc", "objective", "provider", "release", "policy_digest", "uninspected"])
def test_proposal_policy_bounds_rejections(repo, mut):
    pol = policy(repo, max_bytes=5 if mut == "oversize" else 10000, max_lines=0 if mut == "loc" else 20)
    p, snap = proposal(repo, pol, candidate="VALUE = 2\n")
    if mut == "stale": (repo / "src/pkg/mod.py").write_text("VALUE = 3\n")
    if mut == "digest": p = ImprovementProposal(**{**p.__dict__, "candidate_sha256": "0" * 64})
    if mut == "noop": p = ImprovementProposal(**{**p.__dict__, "candidate_content": p.original_content, "candidate_sha256": p.expected_original_sha256})
    if mut == "syntax": p = ImprovementProposal(**{**p.__dict__, "candidate_content": "def bad(:\n", "candidate_sha256": sha("def bad(:\n")})
    if mut == "objective": p = ImprovementProposal(**{**p.__dict__, "objective_type": "deploy"})
    if mut == "provider": p = ImprovementProposal(**{**p.__dict__, "generator_identity": "llm"})
    if mut == "release": p = ImprovementProposal(**{**p.__dict__, "provider_release_digest": "evil"})
    if mut == "policy_digest": p = ImprovementProposal(**{**p.__dict__, "expected_policy_digest": "old"})
    if mut == "uninspected": p = ImprovementProposal(**{**p.__dict__, "target_path": "src/pkg/other.py"})
    r = run_tx(repo, pol, p=p, snap=snap)
    assert r.status == "rejected_before_installation"


def test_duplicate_unknown_json_and_plan_callable_not_executed(repo):
    text = '{"schema_version":"x","schema_version":"y"}'
    with pytest.raises(gt.TransactionError): ImprovementProposal.from_json(text)
    pol = policy(repo); p, snap = proposal(repo, pol)
    with pytest.raises(gt.TransactionError): ImprovementProposal.from_mapping({**p.__dict__, "apply": lambda: (_ for _ in ()).throw(AssertionError())})
    assert run_tx(repo, pol, p=p, snap=snap).status == "applied_and_verified"


def test_approval_lifecycle_binds_expires_reuse_and_resumes(repo, tmp_path):
    pol = policy(repo, approval=True); p, snap = proposal(repo, pol, approval_id="a1")
    store_path = tmp_path / "approvals.json"
    rec = ApprovalRecord("a1", p.digest(), pol.digest, p.expected_original_sha256, "alice", time.time(), time.time()+60)
    store_path.write_text(json.dumps({"a1": rec.__dict__}), encoding="utf-8")
    assert run_tx(repo, pol, p=p, snap=snap, store=ApprovalStore(store_path)).status == "applied_and_verified"
    (repo / "src/pkg/mod.py").write_text("VALUE = 1\n")
    assert run_tx(repo, pol, p=p, snap=snap, store=ApprovalStore(store_path)).status == "rejected_before_installation"
    rec2 = ApprovalRecord("a2", "bad", pol.digest, p.expected_original_sha256, "alice", time.time(), time.time()+60)
    store_path.write_text(json.dumps({"a2": rec2.__dict__}), encoding="utf-8")
    p2 = ImprovementProposal(**{**p.__dict__, "approval_id": "a2"})
    assert run_tx(repo, pol, p=p2, snap=snap, store=ApprovalStore(store_path)).status == "rejected_before_installation"
    rec3 = ApprovalRecord("a3", p.digest(), pol.digest, p.expected_original_sha256, "alice", time.time()-100, time.time()-1)
    store_path.write_text(json.dumps({"a3": rec3.__dict__}), encoding="utf-8")
    p3 = ImprovementProposal(**{**p.__dict__, "approval_id": "a3"})
    assert run_tx(repo, pol, p=p3, snap=snap, store=ApprovalStore(store_path)).status == "rejected_before_installation"


def test_gate_shell_false_metacharacters_candidate_installed_success_and_compile(repo):
    seen = {}
    def runner(gate, root, pol):
        seen["argv"] = gate.argv; seen["content"] = (root / "src/pkg/mod.py").read_text(); seen["shell"] = False
        return {"identity": gate.identity, "ok": True, "exit_status": 0, "timeout": False, "argv_digest": sha(json.dumps(gate.argv)), "output_digest": sha("")}
    pol = policy(repo, gates=[VerificationGate("inert", ("python", "-c", "print('a|b;$(x)')"), 5)])
    r = run_tx(repo, pol, runner=runner)
    assert r.status == "applied_and_verified" and seen["content"] == "VALUE = 2\n" and seen["shell"] is False
    py_compile.compile(str(repo / "src/pkg/mod.py"), doraise=True)


def test_failed_gate_timeout_and_gate_mutation_restore_original(repo):
    pol = policy(repo, gates=[VerificationGate("fail", ("python", "-c", "raise SystemExit(7)"), 5)])
    assert run_tx(repo, pol).status == "verification_failed_rollback_succeeded"
    assert (repo / "src/pkg/mod.py").read_text() == "VALUE = 1\n"
    pol2 = policy(repo, gates=[VerificationGate("timeout", ("python", "-c", "import time; time.sleep(2)"), 0.1)])
    assert run_tx(repo, pol2).status == "verification_failed_rollback_succeeded"
    def mutate(gate, root, pol):
        (root / "src/pkg/mod.py").write_text("VALUE = 99\n")
        return {"identity": gate.identity, "ok": True, "exit_status": 0, "timeout": False}
    assert run_tx(repo, policy(repo), runner=mutate).status == "external_mutation_prevented_rollback"
    assert (repo / "src/pkg/mod.py").read_text() == "VALUE = 99\n"


def test_rollback_failure_manual_recovery_and_mode_safe(repo, monkeypatch):
    os.chmod(repo / "src/pkg/mod.py", 0o6755)
    pol = policy(repo)
    calls = {"n": 0}; real = gt._atomic_write
    def flaky(path, data, mode):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk")
        return real(path, data, mode)
    monkeypatch.setattr(gt, "_atomic_write", flaky)
    r = run_tx(repo, pol, runner=lambda g, r, p: {"identity": "fail", "ok": False})
    assert r.status == "verification_failed_rollback_failed"
    monkeypatch.setattr(gt, "_atomic_write", real)
    (repo / "src/pkg/mod.py").write_text("VALUE = 1\n")
    r2 = run_tx(repo, pol)
    assert r2.status == "applied_and_verified"
    assert stat.S_IMODE((repo / "src/pkg/mod.py").stat().st_mode) & 0o6000 == 0


def test_audit_unavailable_before_install_no_mutation_and_readiness_unresolved(repo):
    pol = policy(repo); p, snap = proposal(repo, pol)
    r = GovernedSelfImprovementTransaction(pol, Audit(fail=True)).apply(p, snap)
    assert r.status == "rejected_before_installation"
    assert (repo / "src/pkg/mod.py").read_text() == "VALUE = 1\n"
    tx = GovernedSelfImprovementTransaction(pol, Audit()); os.environ[gt.DISABLED_ENV] = "0"; tx._unresolved = True
    assert tx.readiness()[0] is False


def test_legacy_direct_write_and_static_reachability_fail_closed(repo):
    app_path = Path(__file__).resolve().parents[1] / "src/vulcan/world_model/self_improvement_apply.py"
    assert "legacy self-improvement application is disabled" in app_path.read_text()
    app = app_path.read_text()
    tree = __import__("ast").parse(app)
    calls = [n for n in __import__("ast").walk(tree) if isinstance(n, __import__("ast").Call)]
    forbidden = []
    for c in calls:
        f = c.func
        if isinstance(f, __import__("ast").Subscript): forbidden.append("subscript-call")
        if isinstance(f, __import__("ast").Attribute) and f.attr in {"commit", "push"}: forbidden.append(f.attr)
    assert forbidden == []
    governed = (Path(__file__).resolve().parents[1] / "src/vulcan/world_model/meta_reasoning/governed_transaction.py").read_text()
    assert "shell=False" in governed
    gtree = __import__("ast").parse(governed)
    for node in __import__("ast").walk(gtree):
        if isinstance(node, __import__("ast").Call) and isinstance(node.func, __import__("ast").Attribute):
            assert node.func.attr not in {"commit", "push"}
        if isinstance(node, __import__("ast").Call) and isinstance(node.func, __import__("ast").Subscript):
            raise AssertionError("plan-supplied callable call is reachable")


def test_success_leaves_git_worktree_uncommitted(repo):
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "a"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    assert run_tx(repo).status == "applied_and_verified"
    status = subprocess.run(["git", "status", "--short"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout
    assert "src/pkg/mod.py" in status
