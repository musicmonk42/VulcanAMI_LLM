"""One immutable authority for canonical runtime process settings."""
from __future__ import annotations

import json, os, re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

_CTL = re.compile(r"[\x00-\x1f\x7f]")
_TRUE={"1","true","yes","on"}; _FALSE={"0","false","no","off"}

class SettingsError(ValueError): pass
class VulcanEnvironment(str, Enum): development="development"; test="test"; production="production"
class LanguageMode(str, Enum): disabled="disabled"; deterministic_only="deterministic_only"; transformer_proposal="transformer_proposal"
class MemoryBackend(str, Enum): disabled="disabled"; sqlite="sqlite"
class SecretSource(str, Enum): direct="direct"; file="file"

@dataclass(frozen=True, repr=False)
class OpaqueSecret:
    source: SecretSource
    value: str = field(repr=False)
    env_name: str = ""
    def reveal(self) -> str: return self.value
    def __repr__(self)->str: return "OpaqueSecret(**redacted**)"
    def to_public(self)->dict[str,str]: return {"source": self.source.value, "env_name": self.env_name, "redacted": "true"}

@dataclass(frozen=True)
class RuntimeSettings:
    environment: VulcanEnvironment
    jwt_issuer: str
    jwt_audience: str
    jwt_secret: OpaqueSecret = field(repr=False)
    durable_root: Path
    language_mode: LanguageMode = LanguageMode.deterministic_only
    language_release_path: Path|None = None
    openai_enabled: bool = False
    anthropic_enabled: bool = False
    memory_enabled: bool = True
    memory_backend: MemoryBackend = MemoryBackend.sqlite
    memory_sqlite_path: Path|None = None
    audit_enabled: bool = True
    csiu_enabled: bool = True
    learning_enabled: bool = True
    self_improvement_enabled: bool = False
    approval_hmac_secret: OpaqueSecret|None = field(default=None, repr=False)
    replicas: int = 1
    request_timeout_seconds: float = 30.0
    public_diagnostics: bool = False
    deprecation_warnings: tuple[str,...] = ()

    def auth_config(self):
        from .auth import AuthConfig
        return AuthConfig(secret=self.jwt_secret.reveal(), issuer=self.jwt_issuer, audience=self.jwt_audience)
    def public_dict(self)->dict[str,object]: return _public(self)
    def schema(self)->dict[str,object]: return generate_settings_schema()

ALIASES={
 "VULCAN_ENV": ("VULCAN_ENV",),
 "VULCAN_JWT_SECRET": ("VULCAN_JWT_SECRET","GRAPHIX_JWT_SECRET","JWT_SECRET_KEY","JWT_SECRET"),
 "VULCAN_JWT_ISSUER": ("VULCAN_JWT_ISSUER",),
 "VULCAN_JWT_AUDIENCE": ("VULCAN_JWT_AUDIENCE",),
 "VULCAN_RUNTIME_DURABLE_ROOT": ("VULCAN_RUNTIME_DURABLE_ROOT",),
 "VULCAN_LANGUAGE_MODE": ("VULCAN_LANGUAGE_MODE",),
 "VULCAN_LANGUAGE_RELEASE_PATH": ("VULCAN_LANGUAGE_RELEASE_PATH","VULCAN_TEXT_MODEL_REVISION"),
 "OPENAI_API_KEY": ("OPENAI_API_KEY",),
 "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY",),
 "VULCAN_MEMORY_ENABLED": ("VULCAN_MEMORY_ENABLED",),
 "VULCAN_MEMORY_BACKEND": ("VULCAN_MEMORY_BACKEND",),
 "VULCAN_MEMORY_SQLITE_PATH": ("VULCAN_MEMORY_SQLITE_PATH",),
 "VULCAN_AUDIT_ENABLED": ("VULCAN_AUDIT_ENABLED",),
 "VULCAN_CSIU_ENABLED": ("VULCAN_CSIU_ENABLED","INTRINSIC_CSIU_OFF"),
 "VULCAN_LEARNING_ENABLED": ("VULCAN_LEARNING_ENABLED",),
 "VULCAN_ENABLE_SELF_IMPROVEMENT": ("VULCAN_ENABLE_SELF_IMPROVEMENT","ENABLE_SELF_IMPROVEMENT"),
 "VULCAN_APPROVAL_HMAC_SECRET": ("VULCAN_APPROVAL_HMAC_SECRET",),
 "VULCAN_RUNTIME_REPLICAS": ("VULCAN_RUNTIME_REPLICAS",),
 "VULCAN_REQUEST_TIMEOUT_SECONDS": ("VULCAN_REQUEST_TIMEOUT_SECONDS","HYBRID_EXECUTOR_TIMEOUT"),
 "VULCAN_PUBLIC_DIAGNOSTICS": ("VULCAN_PUBLIC_DIAGNOSTICS",),
}
DEPRECATED={"GRAPHIX_JWT_SECRET":"VULCAN_JWT_SECRET","JWT_SECRET_KEY":"VULCAN_JWT_SECRET","JWT_SECRET":"VULCAN_JWT_SECRET","ENABLE_SELF_IMPROVEMENT":"VULCAN_ENABLE_SELF_IMPROVEMENT","HYBRID_EXECUTOR_TIMEOUT":"VULCAN_REQUEST_TIMEOUT_SECONDS","INTRINSIC_CSIU_OFF":"VULCAN_CSIU_ENABLED","VULCAN_TEXT_MODEL_REVISION":"VULCAN_LANGUAGE_RELEASE_PATH"}

def _select(env:Mapping[str,str], canonical:str, warnings:list[str])->str|None:
    found=[(n,env[n]) for n in ALIASES[canonical] if env.get(n) not in (None,"")]
    if not found: return None
    vals={v for _,v in found}
    if len(vals)>1: raise SettingsError(f"conflicting values for {canonical}")
    for n,_ in found:
        if n in DEPRECATED: warnings.append(f"deprecated environment variable {n}; use {DEPRECATED[n]}")
    return found[0][1]

def _text(v:str|None, name:str, default:str|None=None)->str:
    v = default if v is None else v
    if not isinstance(v,str) or not v or len(v)>256 or _CTL.search(v): raise SettingsError(f"invalid {name}")
    return v

def _bool(v:str|None, name:str, default:bool)->bool:
    if v is None: return default
    lv=v.lower()
    if lv in _TRUE: return True
    if lv in _FALSE: return False
    if name=="VULCAN_CSIU_ENABLED" and lv in _TRUE|_FALSE: return not (lv in _TRUE)
    raise SettingsError(f"invalid boolean for {name}")

def _int(v:str|None, name:str, default:int, lo:int, hi:int)->int:
    if v is None: return default
    try: x=int(v)
    except ValueError: raise SettingsError(f"invalid integer for {name}") from None
    if not lo<=x<=hi: raise SettingsError(f"out-of-range integer for {name}")
    return x

def _float(v:str|None, name:str, default:float, lo:float, hi:float)->float:
    if v is None: return default
    try: x=float(v)
    except ValueError: raise SettingsError(f"invalid float for {name}") from None
    if not lo<=x<=hi: raise SettingsError(f"out-of-range float for {name}")
    return x

def _path(v:str|None, name:str, required:bool)->Path|None:
    if v is None:
        if required: raise SettingsError(f"{name} is required")
        return None
    p=Path(v)
    if not p.is_absolute(): raise SettingsError(f"{name} must be absolute")
    r=p.resolve(strict=False); repo=Path(__file__).resolve().parents[3]
    if r in {Path('/'),Path('/tmp'),Path.home(),repo.resolve()} or str(r).startswith(str(repo.resolve())+os.sep): raise SettingsError(f"unsafe {name}")
    cur=Path('/')
    for part in r.parts[1:]:
        cur=cur/part
        if cur.exists() and cur.is_symlink(): raise SettingsError(f"symlinked {name}")
    return r

def _secret(value:str|None, name:str, *, required:bool)->OpaqueSecret|None:
    if value is None:
        if required: raise SettingsError(f"{name} is required")
        return None
    if len(value.encode())<32 or value.lower() in {"changeme","secret","password","dev-secret-change-me"}: raise SettingsError(f"weak secret for {name}")
    return OpaqueSecret(SecretSource.direct, value, name)

def load_runtime_settings(env:Mapping[str,str]|None=None)->RuntimeSettings:
    env=dict(os.environ if env is None else env); warnings=[]
    get=lambda n: _select(env,n,warnings)
    environment=VulcanEnvironment(_text(get("VULCAN_ENV"),"VULCAN_ENV","development"))
    durable=_path(get("VULCAN_RUNTIME_DURABLE_ROOT"),"VULCAN_RUNTIME_DURABLE_ROOT", True)
    lang=LanguageMode(_text(get("VULCAN_LANGUAGE_MODE"),"VULCAN_LANGUAGE_MODE","deterministic_only"))
    release=_path(get("VULCAN_LANGUAGE_RELEASE_PATH"),"VULCAN_LANGUAGE_RELEASE_PATH", lang is LanguageMode.transformer_proposal)
    jwt=_secret(get("VULCAN_JWT_SECRET"),"VULCAN_JWT_SECRET", required=True)
    approval=_secret(get("VULCAN_APPROVAL_HMAC_SECRET"),"VULCAN_APPROVAL_HMAC_SECRET", required=environment is VulcanEnvironment.production)
    replicas=_int(get("VULCAN_RUNTIME_REPLICAS"),"VULCAN_RUNTIME_REPLICAS",1,1,1)
    self_imp=_bool(get("VULCAN_ENABLE_SELF_IMPROVEMENT"),"VULCAN_ENABLE_SELF_IMPROVEMENT",False)
    csiu_val=get("VULCAN_CSIU_ENABLED"); csiu = (not _bool(csiu_val,"VULCAN_CSIU_ENABLED",False)) if "INTRINSIC_CSIU_OFF" in env and "VULCAN_CSIU_ENABLED" not in env else _bool(csiu_val,"VULCAN_CSIU_ENABLED",True)
    if environment is VulcanEnvironment.production and (not csiu or not _bool(get("VULCAN_AUDIT_ENABLED"),"VULCAN_AUDIT_ENABLED",True)):
        raise SettingsError("production requires audit and CSIU")
    if self_imp and approval is None: raise SettingsError("self-improvement requires approval HMAC secret")
    mem_enabled=_bool(get("VULCAN_MEMORY_ENABLED"),"VULCAN_MEMORY_ENABLED",True)
    mem_backend=MemoryBackend(_text(get("VULCAN_MEMORY_BACKEND"),"VULCAN_MEMORY_BACKEND","sqlite" if mem_enabled else "disabled"))
    mem_path=_path(get("VULCAN_MEMORY_SQLITE_PATH") or (str(durable/"memory"/"memory.sqlite") if mem_enabled else None),"VULCAN_MEMORY_SQLITE_PATH", mem_enabled)
    return RuntimeSettings(environment,_text(get("VULCAN_JWT_ISSUER"),"VULCAN_JWT_ISSUER","vulcan"),_text(get("VULCAN_JWT_AUDIENCE"),"VULCAN_JWT_AUDIENCE","vulcan-runtime"),jwt,durable,lang,release,get("OPENAI_API_KEY") is not None,get("ANTHROPIC_API_KEY") is not None,mem_enabled,mem_backend,mem_path,_bool(get("VULCAN_AUDIT_ENABLED"),"VULCAN_AUDIT_ENABLED",True),csiu,_bool(get("VULCAN_LEARNING_ENABLED"),"VULCAN_LEARNING_ENABLED",True),self_imp,approval,replicas,_float(get("VULCAN_REQUEST_TIMEOUT_SECONDS"),"VULCAN_REQUEST_TIMEOUT_SECONDS",30.0,0.1,300.0),_bool(get("VULCAN_PUBLIC_DIAGNOSTICS"),"VULCAN_PUBLIC_DIAGNOSTICS",False),tuple(warnings[:16]))

def _public(v):
    if isinstance(v,OpaqueSecret): return v.to_public()
    if isinstance(v,Enum): return v.value
    if isinstance(v,Path): return str(v)
    if is_dataclass(v): return {f.name:_public(getattr(v,f.name)) for f in fields(v)}
    if isinstance(v,tuple): return [_public(x) for x in v]
    return v

def generate_settings_schema()->dict[str,object]:
    return {"schema_version":"vulcan-runtime-settings/1","environment_variables":{k:{"aliases":list(v),"deprecated":[a for a in v if a in DEPRECATED]} for k,v in ALIASES.items()},"fields":[f.name for f in fields(RuntimeSettings)],"secret_fields":["jwt_secret","approval_hmac_secret"]}

def generate_environment_reference()->str:
    lines=["# Vulcan runtime environment reference", json.dumps(generate_settings_schema(), sort_keys=True, indent=2)]
    return "\n".join(lines)+"\n"
