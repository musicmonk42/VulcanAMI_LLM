from __future__ import annotations
from types import SimpleNamespace
import pytest
from vulcan.memory.composition import MemoryRuntimeConfig, compose_governed_memory
from vulcan.memory.governed import MemoryActorContext, MemoryKind, MemoryReadRequest, MemoryReason, MemoryWriteProposal, SQLiteMemoryRepository
from vulcan.runtime.audit import CanonicalAudit
from vulcan.runtime.container import RuntimeContainer
from vulcan.runtime.settings import RuntimeSettings, VulcanEnvironment, durable_root_paths, OpaqueSecret, SecretSource
class World:
    def readiness(self): return True
class Safety:
    def readiness(self): return True
    def validate(self,*args,**kwargs): return True
    def validate_response(self,*args,**kwargs): return True
def settings(tmp_path):
    root=(tmp_path/'durable').resolve(); root.mkdir(exist_ok=True)
    return RuntimeSettings(environment=VulcanEnvironment.production,jwt_issuer='vulcan',jwt_audience='vulcan-runtime',jwt_secret=OpaqueSecret(SecretSource.direct,'A'*40+'1!bcdefgh','VULCAN_JWT_SECRET'),durable_root=root,durable_paths=durable_root_paths(root),approval_hmac_secret=OpaqueSecret(SecretSource.direct,'B'*40+'1!cdefghi','VULCAN_APPROVAL_HMAC_SECRET'),memory_enabled=True,memory_sqlite_path=root/'memory'/'memory.sqlite')
def deployment(): return SimpleNamespace(collective=SimpleNamespace(deps=SimpleNamespace(world_model=World(), safety_validator=Safety(), continual=None)))
def actor(): return MemoryActorContext('tenant','subject','actor', request_id='req')
def proposal(key='idem'): return MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE,'profile','response_style','concise',key)
class Owner:
    owner_id='owner:test'
    capability=SimpleNamespace(value='shadow')
    def readiness(self): return True
    def close(self): return None
    def capabilities(self): return ()

@pytest.mark.asyncio
async def test_runtime_memory_audits_closes_restarts_and_borrows_audit(monkeypatch, tmp_path):
    import vulcan.runtime.container as container
    monkeypatch.setattr(container, 'compose_self_improvement_runtime', lambda **kwargs: SimpleNamespace(drive=Owner(), capabilities=lambda: (), close=lambda: None))
    monkeypatch.setattr(container, 'ShadowLinUCBToolBandit', lambda: Owner())
    monkeypatch.setattr(container, 'LearningOwner', lambda **kwargs: Owner())
    monkeypatch.setattr(container, 'EnhancedSafetyResponseAdapter', lambda safety: Owner())
    monkeypatch.setattr(container, 'SafetyResponseFinalizer', lambda response_safety: Owner())
    c=RuntimeContainer.new(deployment=deployment(), settings=settings(tmp_path)); audit=c.audit; memory=c.memory
    res=memory.remember(actor(), proposal()); assert res.reason is MemoryReason.COMMITTED
    assert memory.retrieve(MemoryReadRequest(actor(),'profile','response_style',1))[0].value == 'concise'
    memory.close(); assert audit is not None; audit.readiness()
    await c.close()
    c2=RuntimeContainer.new(deployment=deployment(), settings=settings(tmp_path))
    try: assert c2.memory.retrieve(MemoryReadRequest(actor(),'profile','response_style',1))[0].value == 'concise'
    finally: await c2.close()
def test_memory_enabled_requires_audit(tmp_path):
    cfg=MemoryRuntimeConfig(True, tmp_path/'m.sqlite', tmp_path)
    with pytest.raises(RuntimeError, match='audit'): compose_governed_memory(cfg)
class WrongAudit:
    owner_id='memory:pretender'
    def readiness(self): return True
    def append(self,event_type,data): return None
def test_wrong_audit_owner_rejected(tmp_path):
    cfg=MemoryRuntimeConfig(True, tmp_path/'m.sqlite', tmp_path)
    with pytest.raises(RuntimeError, match='canonical audit owner'): compose_governed_memory(cfg, audit=WrongAudit())
def test_duplicate_composition_and_close_ordering_fail_closed(tmp_path):
    audit=CanonicalAudit(tmp_path/'audit/events.jsonl'); cfg=MemoryRuntimeConfig(True, tmp_path/'m.sqlite', tmp_path); first=compose_governed_memory(cfg, audit=audit)
    try:
        with pytest.raises(RuntimeError, match='already owned'): compose_governed_memory(cfg, audit=audit)
        audit.close()
        with pytest.raises(RuntimeError, match='audit'): first.readiness()
    finally: first.close()
def test_repository_does_not_close_borrowed_audit(tmp_path):
    audit=CanonicalAudit(tmp_path/'audit/events.jsonl'); repo=SQLiteMemoryRepository(str(tmp_path/'m.sqlite'), durable_root=str(tmp_path), audit=audit)
    repo.close(); audit.readiness(); audit.close()
def test_disabled_memory_is_null_port(tmp_path):
    mem=compose_governed_memory(MemoryRuntimeConfig(False), audit=None)
    assert mem.capabilities() == (); assert mem.remember(actor(), proposal()).reason is MemoryReason.MEMORY_DISABLED
