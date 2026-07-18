"""Canonical statically-composed authenticated ASGI application."""
from __future__ import annotations
import asyncio, json, os
from contextlib import asynccontextmanager
from uuid import uuid4
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception:  # dependency-light route-manifest diagnostics
    class HTTPException(Exception):
        def __init__(self, status_code:int, detail:str='', headers=None): self.status_code=status_code; self.detail=detail; self.headers=headers or {}; super().__init__(detail)
    class JSONResponse(dict):
        def __init__(self, status_code:int=200, content=None, headers=None): super().__init__(content or {}); self.status_code=status_code; self.headers=headers or {}
    class Request: pass
    class _State: pass
    class _Route:
        def __init__(self,path,endpoint): self.path=path; self.endpoint=endpoint
    class FastAPI:
        def __init__(self, *a, **k): self.routes=[]; self.state=_State()
        def get(self,path):
            def deco(fn): self.routes.append(_Route(path,fn)); return fn
            return deco
        def post(self,path):
            def deco(fn): self.routes.append(_Route(path,fn)); return fn
            return deco
        def patch(self,path):
            def deco(fn): self.routes.append(_Route(path,fn)); return fn
            return deco
        def delete(self,path):
            def deco(fn): self.routes.append(_Route(path,fn)); return fn
            return deco
        def middleware(self, _kind):
            def deco(fn): return fn
            return deco
try:
    from pydantic import BaseModel, ConfigDict, Field, StrictStr, StrictInt
except Exception:  # pragma: no cover
    BaseModel=object; ConfigDict=dict; Field=lambda *a,**k: None; StrictStr=str; StrictInt=int
from .auth import AuthConfig, AuthError, AuthorizationError, authenticate_bearer
from .case import CognitiveCase
from .composition import compose_runtime
from .kernel import KernelRequest
from .semantic import Utterance
from vulcan.memory.governed import MemoryActorContext, MemoryKind, MemoryReadRequest, MemoryWriteProposal, MemoryReason

MAX_BODY=16_384
ABSENT_ETAG='"absent"'

def _auth_config_from_env()->AuthConfig:
    return AuthConfig(secret=os.getenv('VULCAN_JWT_SECRET') or os.getenv('GRAPHIX_JWT_SECRET') or os.getenv('JWT_SECRET') or '', issuer=os.getenv('VULCAN_JWT_ISSUER','vulcan'), audience=os.getenv('VULCAN_JWT_AUDIENCE','vulcan-runtime'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready=False; app.state.runtime=None; app.state.auth_config=None; runtime=None
    try:
        app.state.auth_config=_auth_config_from_env()
        runtime=await asyncio.to_thread(compose_runtime)
        await runtime.readiness(); app.state.runtime=runtime; app.state.ready=True
        yield
    except BaseException:
        app.state.ready=False; app.state.runtime=None
        if runtime is not None:
            try: await runtime.close()
            except BaseException: pass
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
    try: await rt.readiness()
    except Exception:
        request.app.state.ready=False; raise HTTPException(503,'runtime not ready') from None
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

def generate_route_manifest():
    return tuple({'path':p,'method':m,'classification':'public' if p.startswith('/health/') else 'protected'} for p,m in [('/health/live','GET'),('/health/ready','GET'),('/v1/chat','POST'),('/v1/admin/domains','POST'),('/v1/admin/alignment','POST'),('/v1/audit/cases/{case_id}','GET'),('/v1/memory/preferences','POST'),('/v1/memory/preferences/{key}','GET'),('/v1/memory/preferences/{record_id}','PATCH'),('/v1/memory/preferences/{record_id}','DELETE')])

def create_app()->FastAPI:
    app=FastAPI(title='VULCAN canonical runtime', version='7.0', lifespan=lifespan)
    app.state.route_manifest=generate_route_manifest(); app.state.ready=False; app.state.runtime=None
    @app.middleware('http')
    async def bounds(request, call_next):
        resp=await call_next(request); resp.headers.setdefault('Cache-Control','no-store'); resp.headers.setdefault('X-Content-Type-Options','nosniff'); return resp
    @app.get('/health/live')
    async def live(): return {'status':'alive'}
    @app.get('/health/ready')
    async def ready(request:Request):
        try: rt=await _runtime(request); return {'status':'ready','runtime_id':rt.runtime_id,'capabilities':list(rt.capabilities())}
        except HTTPException: return JSONResponse(status_code=503, content={'status':'not_ready'})
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
