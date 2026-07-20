"""Canonical statically-composed authenticated ASGI application."""
from __future__ import annotations
import asyncio, json, re
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from .auth import AuthConfig, AuthError, AuthorizationError, authenticate_bearer
from .case import CognitiveCase
from .composition import compose_runtime
from .settings import SettingsError, load_runtime_settings
from .kernel import KernelRequest
from .semantic import Utterance
from vulcan.memory.governed import MemoryActorContext, MemoryKind, MemoryReadRequest, MemoryWriteProposal, MemoryReason
from .api_models import ApprovalRejectBody, BundleBody, MemoryCorrectBody, MemoryWriteBody, ProposalBody, ReasonRequest
from .errors import ApiContractError, ApiErrorCategory, StartupErrorCategory, StartupFailure
from .route_manifest import generate_route_manifest

MAX_BODY=16_384
ABSENT_ETAG='"absent"'
_ETAG=re.compile(r'^"(?:absent|[0-9a-f]{64})"$')
_ID=re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')


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
    if ct!='application/json': raise ApiContractError(415, ApiErrorCategory.CONTENT_TYPE_UNSUPPORTED, 'unsupported content type')
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>MAX_BODY: raise ApiContractError(413, ApiErrorCategory.BODY_TOO_LARGE, 'request body too large')
    try: return _json_loads_no_dupes(bytes(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError): raise ApiContractError(400, ApiErrorCategory.MALFORMED_JSON, 'malformed JSON') from None

def _principal(request:Request, scope:str):
    try:
        p=authenticate_bearer(request.headers.get('authorization'), request.app.state.auth_config)
        p.require(scope); return p
    except AuthorizationError: raise ApiContractError(403, ApiErrorCategory.FORBIDDEN, 'forbidden') from None
    except AuthError: raise ApiContractError(401, ApiErrorCategory.AUTHENTICATION_REQUIRED, 'authentication required') from None
async def _runtime(request:Request):
    rt=getattr(request.app.state,'runtime',None)
    if not getattr(request.app.state,'ready',False) or rt is None or rt.closed: raise ApiContractError(503, ApiErrorCategory.RUNTIME_NOT_READY, 'runtime not ready')
    try: await rt.admission()
    except Exception:
        raise ApiContractError(503, ApiErrorCategory.RUNTIME_NOT_READY, 'runtime not ready') from None
    return rt

async def _ready_runtime(request:Request):
    rt=getattr(request.app.state,'runtime',None)
    if not getattr(request.app.state,'ready',False) or rt is None or rt.closed: raise ApiContractError(503, ApiErrorCategory.RUNTIME_NOT_READY, 'runtime not ready')
    try: await rt.shallow_readiness()
    except Exception:
        raise ApiContractError(503, ApiErrorCategory.RUNTIME_NOT_READY, 'runtime not ready') from None
    return rt

async def _integrity_runtime(request:Request):
    rt=getattr(request.app.state,'runtime',None)
    if not getattr(request.app.state,'ready',False) or rt is None or rt.closed: raise ApiContractError(503, ApiErrorCategory.RUNTIME_NOT_READY, 'runtime not ready')
    try: await rt.deep_integrity()
    except Exception:
        raise ApiContractError(503, ApiErrorCategory.RUNTIME_NOT_READY, 'runtime integrity check failed') from None
    return rt


def _request_id(request: Request) -> str:
    value = request.headers.get('x-request-id') or 'req-' + uuid4().hex
    if not _ID.fullmatch(value):
        raise ApiContractError(400, ApiErrorCategory.SCHEMA_INVALID, 'invalid request id')
    return value

def _if_match(request: Request, *, allow_absent: bool = False) -> str | None:
    value = request.headers.get('if-match')
    if value is None:
        raise ApiContractError(412, ApiErrorCategory.ETAG_REQUIRED, 'If-Match required')
    if not _ETAG.fullmatch(value) or (value == ABSENT_ETAG and not allow_absent):
        raise ApiContractError(400, ApiErrorCategory.ETAG_MALFORMED, 'malformed If-Match')
    return None if value == ABSENT_ETAG else value.strip('"')

def _idempotency_key(request: Request) -> str:
    value = request.headers.get('idempotency-key')
    if not value or not _ID.fullmatch(value):
        raise ApiContractError(400, ApiErrorCategory.SCHEMA_INVALID, 'Idempotency-Key required')
    return value

def _event_dict(event):
    return {'schema_version':event.schema_version,'sequence':event.sequence,'event_type':event.event_type,'timestamp':event.timestamp,'previous_hash':event.previous_hash,'data':event.data,'event_hash':event.event_hash}

def _actor(p, request_id): return MemoryActorContext(p.tenant,p.subject,p.subject,request_id=request_id)

def create_app()->FastAPI:
    app=FastAPI(title='VULCAN canonical runtime', version='7.0', lifespan=lifespan)
    app.state.ready=False; app.state.runtime=None; app.state.startup_error=None

    @app.exception_handler(ApiContractError)
    async def api_contract_error(request: Request, exc: ApiContractError):
        headers = {'WWW-Authenticate':'Bearer'} if exc.category is ApiErrorCategory.AUTHENTICATION_REQUIRED else None
        return JSONResponse(status_code=exc.status_code, content={'error': {'code': exc.category.value, 'message': exc.message, 'request_id': request.headers.get('x-request-id')}}, headers=headers)
    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={'error': {'code': ApiErrorCategory.SCHEMA_INVALID.value, 'message': 'schema validation failed', 'request_id': request.headers.get('x-request-id')}})
    @app.exception_handler(ValidationError)
    async def pydantic_validation_error(request: Request, exc: ValidationError):
        return JSONResponse(status_code=400, content={'error': {'code': ApiErrorCategory.SCHEMA_INVALID.value, 'message': 'schema validation failed', 'request_id': request.headers.get('x-request-id')}})
    @app.middleware('http')
    async def bounds(request, call_next):
        resp=await call_next(request); resp.headers.setdefault('Cache-Control','no-store'); resp.headers.setdefault('Pragma','no-cache'); resp.headers.setdefault('X-Content-Type-Options','nosniff'); resp.headers.setdefault('Referrer-Policy','no-referrer'); resp.headers.setdefault('X-Request-ID', request.headers.get('x-request-id') or 'req-'+uuid4().hex); return resp
    @app.get('/health/live')
    async def live(): return {'status':'alive'}
    @app.get('/health/ready')
    async def ready(request:Request):
        try:
            rt=await _ready_runtime(request)
            state = rt.health.state.value if rt.health is not None else 'ready'
            return {'status':'ready' if state == 'ready' else state}
        except (HTTPException, ApiContractError):
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
        except (HTTPException, ApiContractError):
            return JSONResponse(status_code=503, content={'status':'failed','code':'runtime_integrity_failed'})
    @app.get('/v1/capabilities')
    async def capabilities():
        from vulcan.runtime.capabilities import public_capability_response
        return public_capability_response()
    async def chat(request:Request):
        _principal(request,'reason:write'); rt=await _runtime(request); data=await _body(request); body=ReasonRequest.model_validate(data)
        utterance=Utterance.from_text(body.message); case=CognitiveCase.create(request_id=_request_id(request), conversation_id=body.conversation_id, input_digest=utterance.digest)
        result=await rt.kernel.handle(KernelRequest(utterance, body.conversation_id), case); out=result.transport(case_id=case.case_id,runtime_id=rt.runtime_id,snapshot_id=case.state_snapshot_id); out['status']=result.status.value; return out
    for _p in ('/v1/chat','/v1/chat/orchestrated','/vulcan/v1/chat'):
        app.post(_p)(chat)
    @app.post('/v1/admin/domains')
    async def domains(request:Request):
        p=_principal(request,'domains:write'); rt=await _runtime(request); expected=_if_match(request, allow_absent=True); _idempotency_key(request)
        data=await _body(request); body=BundleBody.model_validate(data); snap=await asyncio.to_thread(rt.domain_registry.load_bundle, json.dumps(body.bundle, sort_keys=True, separators=(',',':')).encode(), expected_previous_digest=expected, actor_id=p.subject, approval_provenance={"request_id": _request_id(request)})
        return {'committed_snapshot_id':snap,'committed_audit_identity':'domain.activation_committed','actor':p.subject}
    @app.post('/v1/admin/alignment')
    async def alignment(request:Request):
        p=_principal(request,'alignment:write'); rt=await _runtime(request); expected=_if_match(request); _idempotency_key(request)
        data=await _body(request); pol=await asyncio.to_thread(rt.alignment.update, data, expected_previous_digest=expected, actor_id=p.subject)
        return {'policy_id':pol.policy_id,'revision':pol.revision,'policy_digest':pol.policy_digest,'committed_audit_identity':'alignment.activation_committed'}
    @app.get('/v1/audit/cases/{case_id}')
    async def audit_case(case_id:str, request:Request):
        _principal(request,'audit:read'); rt=await _runtime(request)
        if not case_id.startswith('case-') or len(case_id)>128: raise HTTPException(404,'case not found')
        rows=rt.audit.events_for_case(case_id)
        if not rows: raise ApiContractError(404, ApiErrorCategory.NOT_FOUND, 'case not found')
        return {'case_id':case_id,'events':[_event_dict(e) for e in rows[:64]]}
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
        _if_match(request); _idempotency_key(request)
        data=ProposalBody.model_validate(await _body(request)); prop=data.proposal
        if prop.get('proposal_id') not in {proposal_id, None}: raise ApiContractError(409, ApiErrorCategory.CONFLICT, 'proposal id mismatch')
        from vulcan.world_model.meta_reasoning.governed_transaction import ImprovementProposal
        proposal=ImprovementProposal.from_mapping(prop)
        rec=await asyncio.to_thread(rt.self_improvement.approval_authority.approve, proposal, rt.self_improvement.policy, p.subject)
        return {'approval_id':rec.approval_id,'proposal_digest':rec.proposal_digest,'state':rec.state,'verifier_id':rt.self_improvement.approval_authority.verifier_id}
    @app.post('/v1/admin/improvements/{proposal_id}/reject')
    async def improvement_reject(proposal_id:str, request:Request):
        _principal(request,'self_improvement:approve'); rt=await _runtime(request); _if_match(request); _idempotency_key(request); data=ApprovalRejectBody.model_validate(await _body(request))
        approval_id=str(data.approval_id or proposal_id)
        await asyncio.to_thread(rt.self_improvement.approval_authority.reject, approval_id)
        return {'approval_id':approval_id,'state':'rejected'}
    @app.post('/v1/admin/improvements/{proposal_id}/resume')
    async def improvement_resume(proposal_id:str, request:Request):
        p=_principal(request,'self_improvement:approve'); rt=await _runtime(request); _if_match(request); _idempotency_key(request); data=ProposalBody.model_validate(await _body(request))
        from vulcan.world_model.meta_reasoning.governed_transaction import ImprovementProposal, inspect_repository
        prop=data.proposal
        if prop.get('proposal_id') not in {proposal_id, None}: raise ApiContractError(409, ApiErrorCategory.CONFLICT, 'proposal id mismatch')
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
        if res.reason is not MemoryReason.COMMITTED: raise ApiContractError(409, ApiErrorCategory.CONFLICT, res.reason.value)
        return {'record_id':res.record.record_id,'revision':res.record.revision,'key':res.record.key,'value':res.record.value}
    @app.get('/v1/memory/preferences/{key}')
    async def mem_read(key:str, request:Request):
        p=_principal(request,'memory:read'); rt=await _runtime(request); rows=await asyncio.to_thread(rt.memory.retrieve,MemoryReadRequest(_actor(p,'req-'+uuid4().hex),'profile',key,1))
        if not rows: raise HTTPException(404,'memory not found')
        r=rows[0]; return {'record_id':r.record_id,'revision':r.revision,'key':r.key,'value':r.value}
    @app.patch('/v1/memory/preferences/{record_id}')
    async def mem_correct(record_id:str, request:Request):
        p=_principal(request,'memory:write'); rt=await _runtime(request); body=MemoryCorrectBody.model_validate(await _body(request)); res=await asyncio.to_thread(rt.memory.correct,_actor(p,'req-'+uuid4().hex),record_id,body.base_revision,MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE,'profile',body.key,body.value,body.idempotency_key))
        if res.reason is not MemoryReason.COMMITTED: raise ApiContractError(409, ApiErrorCategory.CONFLICT, res.reason.value)
        return {'record_id':res.record.record_id,'revision':res.record.revision,'key':res.record.key,'value':res.record.value}
    @app.delete('/v1/memory/preferences/{record_id}')
    async def mem_forget(record_id:str, request:Request):
        p=_principal(request,'memory:forget'); rt=await _runtime(request); raw_rev=request.headers.get('if-match')
        if raw_rev is None or not re.fullmatch(r'"[1-9][0-9]{0,6}"', raw_rev): raise ApiContractError(400, ApiErrorCategory.ETAG_MALFORMED, 'malformed If-Match')
        rev=int(raw_rev.strip('"'))
        res=await asyncio.to_thread(rt.memory.forget,_actor(p,'req-'+uuid4().hex),record_id,rev,_idempotency_key(request))
        if res.reason is not MemoryReason.COMMITTED: raise ApiContractError(409, ApiErrorCategory.CONFLICT, res.reason.value)
        return {'record_id':record_id,'revision':res.record.revision,'state':'forgotten'}
    app.state.route_manifest=generate_route_manifest(app)
    return app
app=create_app()
