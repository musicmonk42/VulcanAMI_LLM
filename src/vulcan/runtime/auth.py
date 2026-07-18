"""Dependency-light strict HS256 bearer authentication for canonical runtime."""
from __future__ import annotations
import base64, binascii, hashlib, hmac, json, math, re, time, unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

_B64=re.compile(r"^[A-Za-z0-9_-]+$")
_CTL=re.compile(r"[\x00-\x1f\x7f]")
KEY_FIELDS={"jku","jwk","kid","x5u","x5c","x5t","x5t#S256","crit"}
MAX_HEADER=8192; MAX_JWT=6144; MAX_SEG=2048; MAX_JSON=4096; MAX_CLAIMS=32; MAX_LIST=16; MAX_TEXT=256
class AuthError(ValueError): pass
class AuthorizationError(AuthError): pass
@dataclass(frozen=True)
class AuthConfig:
    secret: str; issuer: str; audience: str; skew_seconds:int=60; require_typ:bool=True; tenant_claim:str="tenant"
    def __post_init__(self):
        if len(self.secret.encode('utf-8'))<32: raise AuthError("strong JWT secret required")
        for v in (self.issuer,self.audience,self.tenant_claim): _text(v,MAX_TEXT,"config")
        if not (0<=self.skew_seconds<=300): raise AuthError("invalid auth skew")
@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject:str; tenant:str; issuer:str; audience:tuple[str,...]; scopes:frozenset[str]
    def require(self, scope:str)->None:
        if scope not in self.scopes: raise AuthorizationError("missing required scope")

def _text(v,max_len,name):
    if not isinstance(v,str) or not v or len(v)>max_len: raise AuthError(f"invalid {name}")
    n=unicodedata.normalize('NFC',v)
    if n!=v or _CTL.search(n) or any(0xD800<=ord(c)<=0xDFFF for c in n): raise AuthError(f"invalid {name}")
    return n

def _b64(seg):
    if not isinstance(seg,str) or not seg or len(seg)>MAX_SEG or '=' in seg or not _B64.fullmatch(seg): raise AuthError("invalid JWT encoding")
    try: raw=base64.urlsafe_b64decode(seg+'='*((4-len(seg)%4)%4))
    except (binascii.Error, ValueError): raise AuthError("invalid JWT encoding") from None
    if not raw or len(raw)>MAX_JSON: raise AuthError("decoded JWT segment too large")
    if base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')!=seg: raise AuthError("noncanonical JWT encoding")
    return raw

def _loads(raw):
    def pairs(p):
        if len(p)>MAX_CLAIMS: raise AuthError("too many claims")
        d={}
        for k,v in p:
            _text(k,64,"claim name")
            if k in d: raise AuthError("duplicate JSON key")
            d[k]=v
        return d
    try: obj=json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(AuthError("non-finite number")))
    except (UnicodeDecodeError,json.JSONDecodeError) as e: raise AuthError("invalid JWT JSON") from e
    if not isinstance(obj,dict): raise AuthError("JWT object required")
    return obj

def _finite_time(o,name):
    if type(o) not in (int,float) or not math.isfinite(o): raise AuthError(f"invalid {name}")
    return float(o)

def _aud(v):
    if isinstance(v,str): return (_text(v,MAX_TEXT,"audience"),)
    if isinstance(v,list) and 1<=len(v)<=MAX_LIST:
        vals=tuple(_text(x,MAX_TEXT,"audience") for x in v)
        if len(set(vals))!=len(vals): raise AuthError("duplicate audience")
        return vals
    raise AuthError("invalid audience")

def _scopes(v):
    if isinstance(v,str): vals=tuple(x for x in v.split(' ') if x)
    elif isinstance(v,list): vals=tuple(v)
    else: raise AuthError("invalid scopes")
    if not vals or len(vals)>MAX_LIST: raise AuthError("invalid scopes")
    vals=tuple(_text(x,64,"scope") for x in vals)
    if len(set(vals))!=len(vals): raise AuthError("duplicate scopes")
    return frozenset(vals)

def authenticate_bearer(header:str|None, config:AuthConfig, *, clock:Callable[[],float]|None=None)->AuthenticatedPrincipal:
    if not isinstance(header,str) or not header.startswith('Bearer ') or len(header)>MAX_HEADER: raise AuthError("missing bearer token")
    token=header[7:]
    if len(token)>MAX_JWT or token.count('.')!=2: raise AuthError("malformed JWT")
    hseg,pseg,sseg=token.split('.')
    header_obj,payload=_loads(_b64(hseg)),_loads(_b64(pseg))
    if set(header_obj)&KEY_FIELDS: raise AuthError("JWT key selection is forbidden")
    if header_obj.get('alg')!='HS256': raise AuthError("unsupported JWT algorithm")
    if config.require_typ and header_obj.get('typ')!='JWT': raise AuthError("invalid JWT typ")
    if set(header_obj)-{'alg','typ'}: raise AuthError("unsupported JWT header")
    expected=base64.urlsafe_b64encode(hmac.new(config.secret.encode(), f'{hseg}.{pseg}'.encode('ascii'), hashlib.sha256).digest()).rstrip(b'=').decode('ascii')
    if not hmac.compare_digest(expected,sseg): raise AuthError("invalid JWT signature")
    now=(clock or time.time)(); exp=_finite_time(payload.get('exp'), 'exp')
    if exp<=now: raise AuthError("expired JWT")
    if 'nbf' in payload and _finite_time(payload['nbf'],'nbf')>now+config.skew_seconds: raise AuthError("JWT not yet valid")
    if 'iat' in payload and _finite_time(payload['iat'],'iat')>now+config.skew_seconds: raise AuthError("JWT issued in future")
    sub=_text(payload.get('sub'),128,'subject'); tenant=_text(payload.get(config.tenant_claim),128,'tenant')
    if payload.get('iss')!=config.issuer: raise AuthError("issuer mismatch")
    audiences=_aud(payload.get('aud'))
    if config.audience not in audiences: raise AuthError("audience mismatch")
    scopes=_scopes(payload.get('scope', payload.get('scopes')))
    return AuthenticatedPrincipal(sub, tenant, config.issuer, audiences, scopes)
