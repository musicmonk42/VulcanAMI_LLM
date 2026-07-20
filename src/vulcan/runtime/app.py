"""Canonical statically-composed authenticated ASGI application."""
from __future__ import annotations
import asyncio, json
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .auth import AuthConfig, AuthError, AuthorizationError, authenticate_bearer
from .case import CognitiveCase
from .composition import compose_runtime
from .settings import SettingsError, load_runtime_settings
from .kernel import KernelRequest
from .semantic import Utterance
from vulcan.memory.governed import MemoryActorContext, MemoryKind, MemoryReadRequest, MemoryWriteProposal, MemoryReason
from .errors import StartupErrorCategory, StartupFailure
from .route_manifest import generate_route_manifest

MAX_BODY=16_384
ABSENT_ETAG='"absent"'


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready=False; app.state.runtime=None; app.state.auth_config=None; app.state.runtime_settings=None; runtime=None
    try:
        try:
            settings=load_runtime_settings()
        except SettingsError as exc:
            app.state.startup_error = StartupFailure(StartupErrorCategory.SETTINGS_INVALID, "runtime settings invalid", exc)
            raise app.state.startup_error from exc
        app.state.runtime_settings=settings
        app.state.auth_config=settings.auth_config()
        runtime=await asyncio.to_thread(compose_runtime, settings)
        await runtime.deep_integrity(); app.state.runtime=runtime; app.state.ready=True
        yield
    except BaseException as exc:
        app.state.ready=False; app.state.runtime=None
        if isinstance(exc, StartupFailure):
            app.state.startup_error = exc
        else:
            app.state.startup_error = StartupFailure(StartupErrorCategory.RUNTIME_UNHEALTHY, "runtime startup failed", exc)
        if runtime is not None:
            try:
                await runtime.close()
            except BaseException as close_exc:
                exc.add_note(f"runtime close failed during startup cleanup: {type(close_exc).__name__}")
        raise
    finally:
        app.state.ready=False
        active=getattr(app.state,'runtime',None); app.state.runtime=None
        if active is not None: await active.close()

def _json_loads_no_dupes(raw:bytes):
    def pairs(p):
        d={}
        for k,v in p:
            if k in d: raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in()).throw(ValueError('non-finite')))
async def _body(request:Request):
    ct=request.headers.get('content-type','').split(';',1)[0].strip().lower()
    if ct!='application/json': raise HTTPException(415, 'unsupported content type')
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>MAX_BODY: raise HTTPException(413, 'request body too large')
    try: return _json_loads_no_dupes(bytes(raw))
    except Exception: raise HTTPException(400, 'malformed JSON') from None

def _principal(request:Request, scope:str):
    try:
        p=authenticate_bearer(request.headers.get('authorization'), request.app.state.auth_config)
        p.require(scope); return p
    except AuthorizationError: raise HTTPException(403,'forbidden') from None
    except AuthError: raise HTTPException(401,'authentication required', headers={'WWW-Authenticate':'Bearer'}) from None
async def _runtime(request:Request):
    rt=getattr(request.app.state,'runtime',None)
    if not getattr(request.app.state,'ready',False) or rt is None or rt.closed: raise HTTPException(503,'runtime not ready')
    try: await rt.admission()
    except Exception:
        raise HTTPException(503,'runtime not ready') from None
    return rt

async def _ready_runtime(request:Request):
    rt=getattr(request.app.state,'runtime',None)
    if not getattr(request.app.state,'ready',False) or rt is None or rt.closed: raise HTTPException(503,'runtime not ready')
    try: await rt.shallow_readiness()
    except Exception:
        raise HTTPException(503,'runtime not ready') from None
    return rt

async def _integrity_runtime(request:Request):
    rt=getattr(request.app.state,'runtime',None)
    if not getattr(request.app.state,'ready',False) or rt is None or rt.closed: raise HTTPException(503,'runtime not ready')
    try: await rt.deep_integrity()
    except Exception:
        raise HTTPException(503,'runtime integrity check failed') from None
    return rt

def _actor(p, request_id): return MemoryActorContext(p.tenant,p.subject,p.subject,request_id=request_id)

class ReasonRequest(BaseModel):
    message: StrictStr = Field(..., min_length=1, max_length=2048)
    conversation_id: StrictStr|None = Field(default=None, max_length=128)
    model_config=ConfigDict(extra='forbid', strict=True)
class MemoryWriteBody(BaseModel):
    key: StrictStr=Field(..., min_length=1, max_length=64); value: StrictStr=Field(..., min_length=1, max_length=64); idempotency_key: StrictStr=Field(..., min_length=1, max_length=128)
    model_config=ConfigDict(extra='forbid', strict=True)
class MemoryCorrectBody(MemoryWriteBody):
    base_revision: StrictInt=Field(..., ge=1, le=1_000_000)
class BundleBody(BaseModel):
    bundle: dict
    model_config=ConfigDict(extra='forbid', strict=True)

def create_app()->FastAPI:
    app=FastAPI(title='VULCAN canonical runtime', version='7.0', lifespan=lifespan)
    app.state.route_manifest=generate_route_manifest(); app.state.ready=False; app.state.runtime=None; app.state.startup_error=None
    @app.middleware('http')
    async def bounds(request, call_next):
        resp=await call_next(request); resp.headers.setdefault('Cache-Control','no-store'); resp.headers.setdefault('X-Content-Type-Options','nosniff'); return resp
    @app.get('/health/live')
    async def live(): return {'status':'alive'}
    @app.get('/health/ready')
    async def ready(request:Request):
        try:
            rt=await _ready_runtime(request)
            state = rt.health.state.value if rt.health is not None else 'ready'
            return {'status':'ready' if state == 'ready' else state}
        except HTTPException:
            err=getattr(request.app.state,'startup_error',None); code=err.public_code if isinstance(err,StartupFailure) else 'runtime_not_ready'
            return JSONResponse(status_code=503, content={'status':'not_ready','code':code})
    @app.get('/health/integrity')
    async def integrity(request:Request):
        _principal(request,'operator:read')
        try:
            rt=await _integrity_runtime(request)
            snap = rt.health.snapshot() if rt.health is not None else None
            last = None if snap is None or snap.last_integrity is None else {'ok':snap.last_integrity.ok,'category':None if snap.last_integrity.category is None else snap.last_integrity.category.value,'checked_at':snap.last_integrity.checked_at.isoformat(),'last_success_at':None if snap.last_integrity.last_success_at is None else snap.last_integrity.last_success_at.isoformat()}
            return {'status':'passed','runtime_id':rt.runtime_id,'state':rt.health.state.value if rt.health is not None else 'ready','last_integrity':last}
        except HTTPException:
            return JSONResponse(status_code=503, content={'status':'failed','code':'runtime_integrity_failed'})
    @app.get('/v1/capabilities')
    async def capabilities():
        from vulcan.runtime.capabilities import public_capability_response
        return public_capability_response()
    async def chat(request:Request):
        _principal(request,'reason:write'); rt=await _runtime(request); data=await _body(request); body=ReasonRequest.model_validate(data)
        utterance=Utterance.from_text(body.message); case=CognitiveCase.create(request_id='req-'+uuid4().hex, conversation_id=body.conversation_id, input_digest=utterance.digest)
        result=await rt.kernel.handle(KernelRequest(utterance, body.conversation_id), case); out=result.transport(case_id=case.case_id,runtime_id=rt.runtime_id,snapshot_id=case.state_snapshot_id); out['status']=result.status.value; return out
    for _p in ('/v1/chat','/v1/chat/orchestrated','/vulcan/v1/chat'):
        app.post(_p)(chat)
    @app.post('/v1/admin/domains')
    async def domains(request:Request):
        p=_principal(request,'domains:write'); rt=await _runtime(request); etag=request.headers.get('if-match')
        if etag is None: raise HTTPException(412,'If-Match required')
        data=await _body(request); body=BundleBody.model_validate(data); snap=await asyncio.to_thread(rt.domain_registry.load_bundle, json.dumps(body.bundle).encode(), expected_previous_digest=None if etag==ABSENT_ETAG else etag.strip('"'))
        return {'committed_snapshot_id':snap,'committed_audit_identity':'domain.activation_committed','actor':p.subject}
    @app.post('/v1/admin/alignment')
    async def alignment(request:Request):
        p=_principal(request,'alignment:write'); rt=await _runtime(request); etag=request.headers.get('if-match')
        if etag is None: raise HTTPException(412,'If-Match required')
        data=await _body(request); pol=await asyncio.to_thread(rt.alignment.update, data, expected_previous_digest=etag.strip('"'), actor_id=p.subject)
        return {'policy_id':pol.policy_id,'revision':pol.revision,'policy_digest':pol.policy_digest,'committed_audit_identity':'alignment.activation_committed'}
    @app.get('/v1/audit/cases/{case_id}')
    async def audit_case(case_id:str, request:Request):
        _principal(request,'audit:read'); rt=await _runtime(request)
        if not case_id.startswith('case-') or len(case_id)>128: raise HTTPException(404,'case not found')
        rows=rt.audit.case_events(case_id) if hasattr(rt.audit,'case_events') else []
        return {'case_id':case_id,'events':rows[:64]}
    @app.get('/v1/admin/improvements')
    async def improvements(request:Request):
        _principal(request,'self_improvement:read'); rt=await _runtime(request)
        return {'runtime_id':rt.runtime_id,'owner_id':getattr(rt.self_improvement.journal,'owner_id',''),'pending':list(getattr(rt.self_improvement.drive.state,'pending_approvals',[]))[:64]}
    @app.get('/v1/admin/improvements/{proposal_id}')
    async def improvement_detail(proposal_id:str, request:Request):
        _principal(request,'self_improvement:read'); rt=await _runtime(request)
        rows=[r for r in getattr(rt.self_improvement.drive.state,'pending_approvals',[]) if r.get('id')==proposal_id or r.get('proposal_id')==proposal_id]
        if not rows: raise HTTPException(404,'proposal not found')
        return rows[0]
    @app.post('/v1/admin/improvements/{proposal_id}/approve')
    async def improvement_approve(proposal_id:str, request:Request):
        p=_principal(request,'self_improvement:approve'); rt=await _runtime(request)
        etag=request.headers.get('if-match')
        if etag is None: raise HTTPException(412,'If-Match required')
        data=await _body(request); prop=data.get('proposal')
        if not isinstance(prop,dict): raise HTTPException(400,'proposal required')
        from vulcan.world_model.meta_reasoning.governed_transaction import ImprovementProposal
        proposal=ImprovementProposal.from_mapping(prop)
        rec=await asyncio.to_thread(rt.self_improvement.approval_authority.approve, proposal, rt.self_improvement.policy, p.subject)
        return {'approval_id':rec.approval_id,'proposal_digest':rec.proposal_digest,'state':rec.state,'verifier_id':rt.self_improvement.approval_authority.verifier_id}
    @app.post('/v1/admin/improvements/{proposal_id}/reject')
    async def improvement_reject(proposal_id:str, request:Request):
        _principal(request,'self_improvement:approve'); rt=await _runtime(request); data=await _body(request)
        approval_id=str(data.get('approval_id') or proposal_id)
        await asyncio.to_thread(rt.self_improvement.approval_authority.reject, approval_id)
        return {'approval_id':approval_id,'state':'rejected'}
    @app.post('/v1/admin/improvements/{proposal_id}/resume')
    async def improvement_resume(proposal_id:str, request:Request):
        p=_principal(request,'self_improvement:approve'); rt=await _runtime(request); data=await _body(request)
        from vulcan.world_model.meta_reasoning.governed_transaction import ImprovementProposal, inspect_repository
        prop=data.get('proposal')
        if not isinstance(prop,dict): raise HTTPException(400,'proposal required')
        proposal=ImprovementProposal.from_mapping(prop)
        snapshot=inspect_repository(rt.self_improvement.policy.repo_root, rt.self_improvement.policy.permitted_path_globs)
        res=await asyncio.to_thread(rt.self_improvement.transaction.apply, proposal, snapshot, p.subject)
        return {'status':res.status_code,'state':res.state,'proposal_digest':res.proposal_digest}
    @app.get('/v1/admin/improvements/{proposal_id}/status')
    async def improvement_status(proposal_id:str, request:Request):
        _principal(request,'self_improvement:read'); rt=await _runtime(request)
        return {'proposal_id':proposal_id,'csiu':rt.self_improvement.status_port.status(),'journal_owner':rt.self_improvement.journal.owner_id}
    @app.get('/v1/audit/improvements/{proposal_digest}')
    async def audit_improvement(proposal_digest:str, request:Request):
        _principal(request,'audit:read'); rt=await _runtime(request)
        return {'proposal_digest':proposal_digest,'events':[e.__dict__ for e in rt.audit.events_for_proposal(proposal_digest)][:128]}
    @app.post('/v1/memory/preferences')
    async def mem_create(request:Request):
        p=_principal(request,'memory:write'); rt=await _runtime(request); body=MemoryWriteBody.model_validate(await _body(request)); res=await asyncio.to_thread(rt.memory.remember,_actor(p,'req-'+uuid4().hex),MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE,'profile',body.key,body.value,body.idempotency_key))
        if res.reason is not MemoryReason.COMMITTED: raise HTTPException(409,res.reason.value)
        return {'record_id':res.record.record_id,'revision':res.record.revision,'key':res.record.key,'value':res.record.value}
    @app.get('/v1/memory/preferences/{key}')
    async def mem_read(key:str, request:Request):
        p=_principal(request,'memory:read'); rt=await _runtime(request); rows=await asyncio.to_thread(rt.memory.retrieve,MemoryReadRequest(_actor(p,'req-'+uuid4().hex),'profile',key,1))
        if not rows: raise HTTPException(404,'memory not found')
        r=rows[0]; return {'record_id':r.record_id,'revision':r.revision,'key':r.key,'value':r.value}
    @app.patch('/v1/memory/preferences/{record_id}')
    async def mem_correct(record_id:str, request:Request):
        p=_principal(request,'memory:write'); rt=await _runtime(request); body=MemoryCorrectBody.model_validate(await _body(request)); res=await asyncio.to_thread(rt.memory.correct,_actor(p,'req-'+uuid4().hex),record_id,body.base_revision,MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE,'profile',body.key,body.value,body.idempotency_key))
        if res.reason is not MemoryReason.COMMITTED: raise HTTPException(409,res.reason.value)
        return {'record_id':res.record.record_id,'revision':res.record.revision,'key':res.record.key,'value':res.record.value}
    @app.delete('/v1/memory/preferences/{record_id}')
    async def mem_forget(record_id:str, request:Request):
        p=_principal(request,'memory:forget'); rt=await _runtime(request); rev=int(request.headers.get('if-match','0').strip('"') or '0')
        if rev<1: raise HTTPException(412,'If-Match required')
        res=await asyncio.to_thread(rt.memory.forget,_actor(p,'req-'+uuid4().hex),record_id,rev,request.headers.get('idempotency-key') or 'forget-'+uuid4().hex)
        if res.reason is not MemoryReason.COMMITTED: raise HTTPException(409,res.reason.value)
        return {'record_id':record_id,'revision':res.record.revision,'state':'forgotten'}
    return app
app=create_app()
