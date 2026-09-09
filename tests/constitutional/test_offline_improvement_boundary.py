from __future__ import annotations

import ast
import hashlib
import importlib
import sys
import json
import subprocess
from pathlib import Path

import pytest

from vulcan.improvement.proposal import ImprovementProposal, ImprovementProposalStore, ProposalError, SCHEMA_VERSION
from vulcan.runtime.state_authorities import DisabledCSIUPolicyAuthority
from vulcan.improvement.offline import ApprovalAuthority, OfflineImprovementOperator
from vulcan.world_model.meta_reasoning.governed_transaction import ImprovementPolicy, TransactionError, VerificationGate, inspect_repository


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def proposal() -> ImprovementProposal:
    original, candidate = "VALUE = 1\n", "VALUE = 2\n"
    return ImprovementProposal(SCHEMA_VERSION, "proposal-1", "fix_known_bugs", "src/example.py",
                               _digest(original), original, candidate, _digest(candidate), "a"*64,
                               "test-generator", "release-1", "", "b"*64)


def test_serving_import_closure_has_no_install_or_process_capability() -> None:
    for name in list(sys.modules):
        if name.startswith("vulcan.improvement.offline") or name.endswith("governed_transaction"):
            sys.modules.pop(name)
    importlib.reload(importlib.import_module("vulcan.runtime.container"))
    assert "vulcan.improvement.offline" not in sys.modules
    assert "vulcan.world_model.meta_reasoning.governed_transaction" not in sys.modules
    for source in ("src/vulcan/improvement/proposal.py", "src/vulcan/runtime/app.py", "src/vulcan/runtime/container.py"):
        tree = ast.parse(Path(source).read_text())
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert imports.isdisjoint({"subprocess", "shutil", "tempfile"})
    dockerfile = Path("Dockerfile").read_text()
    for capability in ("improvement/offline.py", "governed_transaction.py", "self_improvement_drive.py"):
        assert f"/app/src/vulcan/{capability}" in dockerfile or capability in dockerfile
    assert "/usr/local/bin/vulcan-improvement-operator" in dockerfile


def test_serving_proposal_store_is_immutable_and_rejects_escape(tmp_path: Path) -> None:
    store = ImprovementProposalStore(tmp_path)
    assert store.emit(proposal()) == proposal().digest()
    persisted = (tmp_path/"proposal-1.json").read_text()
    assert "rationale" not in persisted and "approval_id" not in persisted
    with pytest.raises(ProposalError, match="already emitted"):
        store.emit(proposal())
    escaped = proposal().__class__(**{**proposal().__dict__, "target_path":"../escape.py"})
    with pytest.raises(ProposalError, match="escapes"):
        store.emit(escaped)
    malicious_id = proposal().__class__(**{**proposal().__dict__, "proposal_id":"../../escape"})
    with pytest.raises(ProposalError, match="proposal id"):
        store.emit(malicious_id)
    reasoning = proposal().__class__(**{**proposal().__dict__, "proposal_id":"proposal--2", "rationale":"private chain of thought"})
    with pytest.raises(ProposalError, match="free-form rationale"):
        store.emit(reasoning)


def test_disabled_csiu_is_a_distinct_policy_authority() -> None:
    authority = DisabledCSIUPolicyAuthority()
    assert authority.kind == "csiu"
    assert authority.owner == "constitutional:disabled-csiu-policy"
    assert "improvement" not in authority.owner


def test_offline_human_approved_package_install_and_rollback(tmp_path: Path) -> None:
    repo=tmp_path/"repo"; (repo/"src/pkg").mkdir(parents=True); target=repo/"src/pkg/example.py"
    target.write_text("VALUE = 1\n")
    subprocess.run(["git","init","-q"],cwd=repo,check=True)
    subprocess.run(["git","config","user.email","test@example.invalid"],cwd=repo,check=True)
    subprocess.run(["git","config","user.name","Test"],cwd=repo,check=True)
    subprocess.run(["git","add","."],cwd=repo,check=True); subprocess.run(["git","commit","-qm","base"],cwd=repo,check=True)
    policy=ImprovementPolicy("policy/1",True,repo,("fix_known_bugs",),("src/**/*.py",),(),1,1000,20,True,
                             {"test-generator":("release-1",)},(VerificationGate("compile",(sys.executable,"-m","compileall","-q","src")),),30,2000,True)
    snapshot=inspect_repository(repo,policy.permitted_path_globs); p=proposal()
    p=p.__class__(**{**p.__dict__,"target_path":"src/pkg/example.py","inspected_source_digest":snapshot.digest,"expected_policy_digest":policy.digest})
    key=b"operator-test-key-that-is-at-least-32-bytes"
    authority=ApprovalAuthority(tmp_path/"state/approvals",key)
    approval=authority.issue(p,policy,"human-reviewer")
    operator=OfflineImprovementOperator(repo,policy,authority,tmp_path/"state/audit.jsonl",key)
    package_path=tmp_path/"package.json"; package=operator.review_and_package(p,approval.approval_id,package_path)
    assert package.proposal_digest == p.digest()
    replayed = tmp_path/"replayed.json"
    restarted = OfflineImprovementOperator(repo,policy,ApprovalAuthority(tmp_path/"state/approvals",key),tmp_path/"state/audit.jsonl",key)
    with pytest.raises(TransactionError, match="approval verification failed"):
        restarted.review_and_package(p,approval.approval_id,replayed)
    assert not replayed.exists()
    operator.install(package_path,repo); assert target.read_text() == "VALUE = 2\n"
    operator.rollback(package_path,repo); assert target.read_text() == "VALUE = 1\n"
    events=[json.loads(line)["event"] for line in (tmp_path/"state/audit.jsonl").read_text().splitlines()]
    assert {"improvement.reviewed","improvement.package_signed","improvement.deployed","improvement.rolled_back"} <= set(events)
