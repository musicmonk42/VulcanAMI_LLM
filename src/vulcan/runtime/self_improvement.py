"""Canonical runtime-owned self-improvement graph."""
from __future__ import annotations

import hashlib, hmac, json, time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement, CSIUEnforcementConfig, CSIUMetricSnapshot, METRIC_ORDER, canonical_digest
from vulcan.world_model.meta_reasoning.governed_transaction import ApprovalRecord, ApprovalStore, GovernedSelfImprovementTransaction, ImprovementPolicy, VerificationGate, TransactionError, inspect_repository, ImprovementProposal
from vulcan.world_model.meta_reasoning.self_improvement_drive import compose_self_improvement_drive

class AggregateCSIUTelemetryPort(Protocol):
    def collect_snapshot(self) -> CSIUMetricSnapshot: ...

@dataclass
class StaticAggregateCSIUTelemetryPort:
    """Read-only aggregate telemetry port; no fabricated production fallback."""
    path: Path
    policy_digest: str
    metric_definition_version: str
    def collect_snapshot(self) -> CSIUMetricSnapshot:
        raw = self.path.read_bytes()
        doc = json.loads(raw.decode('utf-8'))
        batch = {k: doc[k] for k in ('metrics','window_start','window_end','sample_count','aggregation_method','metric_definition_version','provider_id','privacy_cohort')}
        if set(batch['metrics']) != set(METRIC_ORDER): raise TransactionError('partial CSIU telemetry')
        provenance = canonical_digest(batch)
        return CSIUMetricSnapshot(metrics=batch['metrics'], window_start=batch['window_start'], window_end=batch['window_end'], sample_count=batch['sample_count'], aggregation_method=batch['aggregation_method'], metric_definition_version=self.metric_definition_version, provider_id=batch['provider_id'], provenance_digest=provenance, policy_digest=self.policy_digest, privacy_cohort=batch.get('privacy_cohort') or {})

@dataclass
class CSIUStatusPort:
    enforcer: CSIUEnforcement
    def status(self) -> dict[str, Any]: return self.enforcer.get_statistics()

class ApprovalAuthority:
    def __init__(self, store: ApprovalStore, secret: bytes, *, verifier_id: str = 'approval-verifier:v1'):
        self.store=store; self.secret=secret; self.verifier_id=verifier_id
    def _mac(self, body: Mapping[str, Any]) -> str:
        return hmac.new(self.secret, json.dumps(dict(body), sort_keys=True, separators=(',',':')).encode(), hashlib.sha256).hexdigest()
    def approve(self, proposal: ImprovementProposal, policy: ImprovementPolicy, actor: str, *, scope='self_improvement.approve', ttl=3600.0) -> ApprovalRecord:
        now=time.time(); aid='approval-'+hashlib.sha256(f'{proposal.digest()}:{actor}:{now}'.encode()).hexdigest()[:24]
        cap={'approval_id':aid,'proposal_digest':proposal.digest(),'policy_digest':policy.digest,'original_source_digest':proposal.expected_original_sha256,'actor':actor,'scope':scope,'issued_at':now,'expires_at':now+ttl,'nonce':hashlib.sha256(os.urandom(16)).hexdigest(),'verifier_id':self.verifier_id}
        rec=ApprovalRecord(aid, cap['proposal_digest'], policy.digest, proposal.expected_original_sha256, actor, now, now+ttl)
        self.store.save(rec); return rec
    def reject(self, approval_id: str) -> None: self.store.terminalize(approval_id, 'rejected')

class PersistentImprovementJournal:
    states=('PROPOSED','APPROVED','APPROVAL_CLAIMED','APPLY_PREPARED','CANDIDATE_INSTALLED','GATES_RUNNING','VERIFIED','COMMITTED','ROLLED_BACK','ABORTED','MANUAL_RECOVERY_REQUIRED')
    def __init__(self, path: Path):
        self.path=path; path.parent.mkdir(parents=True, exist_ok=True); path.touch(exist_ok=True); self.owner_id=f'improvement-journal:{path}'
    def readiness(self): self.reconcile(); return True
    def reconcile(self):
        if self.path.is_symlink(): raise TransactionError('symlinked improvement journal')
        return True
    def close(self): pass

@dataclass
class SelfImprovementRuntime:
    durable_root: Path; audit: Any; alignment: Any; csiu: CSIUEnforcement; policy: ImprovementPolicy; approval_store: ApprovalStore; approval_authority: ApprovalAuthority; verifier: Any; journal: PersistentImprovementJournal; transaction: GovernedSelfImprovementTransaction; drive: Any; telemetry: AggregateCSIUTelemetryPort; status_port: CSIUStatusPort
    def readiness(self) -> bool:
        if self.drive._csiu_enforcer is not self.csiu: raise RuntimeError("CSIU authority mismatch")
        if self.transaction.audit is not self.audit: raise RuntimeError("audit authority mismatch")
        if self.drive.transaction is not self.transaction: raise RuntimeError("transaction authority mismatch")
        self.csiu.readiness(); self.journal.readiness(); self.approval_store._check_paths()
        return True
    def capabilities(self) -> tuple[str,...]:
        self.readiness(); return ('governed-self-improvement','csiu')
    def close(self): self.csiu.close()

def build_default_policy(repo: Path) -> ImprovementPolicy:
    return ImprovementPolicy('vulcan-improvement-policy/1', True, repo, ('fix_known_bugs','improve_test_coverage','optimize_performance','enhance_safety_systems','fix_circular_imports'), ('src/**/*.py','tests/**/*.py'), ('.git/**','data/**','logs/**'), 1, 50000, 200, True, {}, (VerificationGate('compile-target', ('python','-m','compileall','-q','src'), 30.0),), 30.0, 20000, True)

def compose_self_improvement_runtime(*, durable_root: Path, audit: Any, alignment: Any, world_model: Any, approval_hmac_secret: str | None) -> SelfImprovementRuntime:
    repo=Path(__file__).resolve().parents[3]
    policy=build_default_policy(repo)
    csiu_store=durable_root/'csiu'/'accounting.jsonl'; csiu_store.parent.mkdir(parents=True, exist_ok=True); csiu_store.touch(exist_ok=True)
    csiu=CSIUEnforcement(CSIUEnforcementConfig(durable_store_path=str(csiu_store), durable_accounting_required=True))
    approvals=ApprovalStore(durable_root/'approvals'/'approvals.json')
    if approval_hmac_secret is None:
        approval_hmac_secret = 'development-approval-hmac-secret-change-me-32'
    authority=ApprovalAuthority(approvals, approval_hmac_secret.encode())
    journal=PersistentImprovementJournal(durable_root/'improvements'/'journal.jsonl')
    tx=GovernedSelfImprovementTransaction(policy, audit, approvals)
    drive=compose_self_improvement_drive(world_model=world_model, csiu_enforcer=csiu, improvement_policy=policy, approval_store=approvals, approval_verifier=authority, audit_owner=audit)
    drive.governed_policy=policy; drive.transaction=tx; drive.approval_store=approvals
    telemetry=StaticAggregateCSIUTelemetryPort(durable_root/'csiu'/'telemetry_snapshot.json', csiu.policy.policy_digest, csiu.policy.metric_definition_version)
    return SelfImprovementRuntime(durable_root,audit,alignment,csiu,policy,approvals,authority,authority,journal,tx,drive,telemetry,CSIUStatusPort(csiu))
