import base64, hashlib, hmac, json, time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vulcan.runtime.auth import AuthConfig, AuthError, AuthorizationError, authenticate_bearer
from vulcan.memory.governed import SQLiteMemoryRepository, MemoryActorContext, MemoryWriteProposal, MemoryKind, MemoryReadRequest, MemoryReason

SECRET='s'*32

def b64(o):
    raw=json.dumps(o, separators=(',', ':'), allow_nan=False).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode()
def tok(payload, header=None, secret=SECRET):
    h=b64(header or {'alg':'HS256','typ':'JWT'}); p=b64(payload); sig=base64.urlsafe_b64encode(hmac.new(secret.encode(),f'{h}.{p}'.encode(),hashlib.sha256).digest()).rstrip(b'=').decode(); return f'Bearer {h}.{p}.{sig}'
def cfg(): return AuthConfig(SECRET,'iss','aud')
def good(**kw):
    p={'sub':'u1','tenant':'t1','iss':'iss','aud':'aud','scope':'reason:write memory:read','exp':2000,'iat':900,'nbf':900}; p.update(kw); return p

def test_auth_valid_and_principal_has_no_raw_token():
    p=authenticate_bearer(tok(good()), cfg(), clock=lambda:1000)
    assert (p.subject,p.tenant)==('u1','t1') and 'reason:write' in p.scopes and not hasattr(p,'token')

def test_auth_rejects_weak_secret_missing_bearer_structure_padding_and_bad_sig():
    with pytest.raises(AuthError): AuthConfig('short','iss','aud')
    with pytest.raises(AuthError): authenticate_bearer(tok(good())[7:], cfg(), clock=lambda:1000)
    with pytest.raises(AuthError): authenticate_bearer('Bearer a.b', cfg(), clock=lambda:1000)
    bad=tok(good()).replace('.', '=.', 1)
    with pytest.raises(AuthError): authenticate_bearer(bad, cfg(), clock=lambda:1000)
    with pytest.raises(AuthError): authenticate_bearer(tok(good(), secret='x'*32), cfg(), clock=lambda:1000)

def test_auth_rejects_duplicate_keys_algorithms_key_selection_times_and_mismatch():
    h='eyJhbGciOiJIUzI1NiIsImFsZyI6IkhTMjU2IiwidHlwIjoiSldUIn0'; p=b64(good()); sig=base64.urlsafe_b64encode(hmac.new(SECRET.encode(),f'{h}.{p}'.encode(),hashlib.sha256).digest()).rstrip(b'=').decode()
    with pytest.raises(AuthError): authenticate_bearer(f'Bearer {h}.{p}.{sig}', cfg(), clock=lambda:1000)
    for header in ({'alg':'none','typ':'JWT'},{'alg':'RS256','typ':'JWT'},{'alg':'HS256','typ':'JWT','kid':'k'}):
        with pytest.raises(AuthError): authenticate_bearer(tok(good(), header), cfg(), clock=lambda:1000)
    for pl in (good(exp=999), good(exp=True), good(nbf=2000), good(iat=2000), good(iss='bad'), good(aud='bad'), good(sub=''), good(tenant=''), good(scope=['a','a'])):
        with pytest.raises(AuthError): authenticate_bearer(tok(pl), cfg(), clock=lambda:1000)
    raw='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.'+base64.urlsafe_b64encode(b'{"sub":"u1","tenant":"t1","iss":"iss","aud":"aud","scope":"reason:write","exp":NaN}').rstrip(b'=').decode()
    sig=base64.urlsafe_b64encode(hmac.new(SECRET.encode(), raw[7:].encode() if raw.startswith('Bearer ') else raw.encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
    with pytest.raises(AuthError): authenticate_bearer('Bearer '+raw+'.'+sig, cfg(), clock=lambda:1000)
    p=authenticate_bearer(tok(good(nbf=1050, iat=1050)), cfg(), clock=lambda:1000); assert p.subject=='u1'
    with pytest.raises(AuthorizationError): p.require('domains:write')

class FailingAudit:
    owner_id = "audit:test"
    def __init__(self): self.events=[]; self.fail=False; self.fail_after=False
    def readiness(self):
        if self.fail: raise RuntimeError('audit down')
        return True
    def append(self, et, payload):
        if self.fail: raise RuntimeError('audit down')
        self.events.append((et,payload['operation_id']))
        if self.fail_after: raise RuntimeError('ack failed')

def actor(): return MemoryActorContext('tenant','subject','subject',request_id='req')
def proposal(k='idem'): return MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE,'profile','response_style','concise',k)

def test_outbox_pending_excluded_retry_reopen_and_no_duplicate_event(tmp_path):
    audit=FailingAudit(); db=tmp_path/'m.db'; repo=SQLiteMemoryRepository(str(db), durable_root=str(tmp_path), audit=audit)
    audit.fail=True
    with pytest.raises(RuntimeError): repo.commit(actor(), proposal('same'))
    assert repo.read(MemoryReadRequest(actor(),'profile','response_style',1)) == ()
    assert repo._db.execute('select count(*) from memory_revisions').fetchone()[0] == 1
    assert repo._db.execute('select count(*) from memory_audit_outbox where audit_complete=0').fetchone()[0] == 1
    audit.fail=False
    res=repo.commit(actor(), proposal('same'))
    assert res.reason is MemoryReason.COMMITTED and res.record.revision == 1
    assert [e[0] for e in audit.events].count('memory.write_committed') == 1
    repo.close()
    repo2=SQLiteMemoryRepository(str(db), durable_root=str(tmp_path), audit=audit)
    assert len(repo2.read(MemoryReadRequest(actor(),'profile','response_style',1))) == 1
    assert repo2._db.execute('select count(*) from memory_revisions').fetchone()[0] == 1
    repo2.close()

def test_outbox_audit_success_ack_failure_is_idempotent(tmp_path):
    audit=FailingAudit(); audit.fail_after=True; db=tmp_path/'m.db'; repo=SQLiteMemoryRepository(str(db), durable_root=str(tmp_path), audit=audit)
    with pytest.raises(RuntimeError): repo.commit(actor(), proposal('ack'))
    assert repo.read(MemoryReadRequest(actor(),'profile','response_style',1)) == ()
    audit.fail_after=False
    res=repo.commit(actor(), proposal('ack'))
    assert res.reason is MemoryReason.COMMITTED
    assert repo._db.execute('select count(*) from memory_revisions').fetchone()[0] == 1
    assert [e[0] for e in audit.events].count('memory.write_committed') == 1
    repo.close()

def test_asyncio_fallback_plugin_paths_are_mutually_exclusive(pytestconfig):
    import tests.conftest as c
    class PM:
        def __init__(self, present): self.present=present
        def hasplugin(self, name): return self.present and name == 'asyncio'
    class Cfg:
        def __init__(self, present): self.pluginmanager=PM(present)
    assert c._compatible_async_plugin_present(Cfg(True)) is True
    assert c._compatible_async_plugin_present(Cfg(False)) is False
