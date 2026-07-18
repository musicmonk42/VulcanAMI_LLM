"""Fail-closed verification for local transformer span language releases.

Approval fields and hashes prove only local byte integrity and promotion state
within this verifier. They do not prove publisher identity unless paired with an
external signature, and they do not prove general reasoning correctness. This
repository intentionally bundles no promoted neural release; deterministic mode
therefore remains the safe default whenever an approved release is absent.
"""
from __future__ import annotations
import hashlib,json,math,os,re,stat,unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from .adapter import RUNTIME_ABI

_MANIFEST_SCHEMA="local-language-release/1"; _PROVIDER_SCHEMA="local-gpt-span-provider/1"; _EVAL_SCHEMA="local-language-evaluation/1"
_SHA256=re.compile(r"^[0-9a-f]{64}$"); _IDENTIFIER=re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_REQUIRED_ARTIFACTS=frozenset({"weights","tokenizer","config","evaluation_report"})
_ARTIFACT_FILES={"weights":"weights.safetensors","tokenizer":"tokenizer.json","config":"config.json","evaluation_report":"evaluation.json"}
MAX_MANIFEST=262_144; MAX_ARTIFACT=64*1024*1024; MAX_JSON=4*1024*1024
REQ_CATS=frozenset({"valid_operation_span_accuracy","malformed_output_rejection","fact_answer_injection_rejection","domain_hint_rejection","unicode_span_correctness","context_overflow_handling","deterministic_repeated_generation","fallback_behavior"})
class ReleaseVerificationError(ValueError): pass
class ReleaseRole(str,Enum): INPUT="input-language-adapter"; OUTPUT="output-language-adapter"
@dataclass(frozen=True)
class Artifact: name:str; path:str; sha256:str; byte_size:int
@dataclass(frozen=True)
class Evaluation: report_sha256:str; passed:bool; categories:tuple[str,...]
@dataclass(frozen=True)
class ProviderConfig:
    architecture:str; vocabulary_size:int; embedding_width:int; layer_count:int; attention_head_count:int; feed_forward_width:int; approved_context_length:int; max_generated_proposal_tokens:int; special_token_ids:dict[str,int]; dropout:float; generation_method:str
@dataclass(frozen=True)
class LocalLanguageRelease:
    release_id:str; version:str; role:ReleaseRole; runtime_abi:str; provider_implementation:str; provider_config_artifact:str; tokenizer_artifact:str; weights_artifact:str; evaluation_report_artifact:str; release_created:str; manifest_digest:str; promotion_state:str; approval_id:str; artifacts:tuple[Artifact,...]; evaluation:Evaluation; provider_config:ProviderConfig|None=None

def _pairs(p):
    d={}
    for k,v in p:
        if k in d: raise ReleaseVerificationError(f"duplicate JSON key: {k}")
        d[k]=v
    return d

def _bad_const(x): raise ReleaseVerificationError("non-finite JSON number")
def _read(path:Path,limit:int)->bytes:
    st1=path.lstat()
    if stat.S_ISLNK(st1.st_mode) or not stat.S_ISREG(st1.st_mode): raise ReleaseVerificationError("artifact is not a regular release file")
    if st1.st_size<0 or st1.st_size>limit: raise ReleaseVerificationError("artifact size bound")
    data=path.read_bytes()
    st2=path.lstat()
    if (st1.st_mode,st1.st_size)!=(st2.st_mode,st2.st_size): raise ReleaseVerificationError("artifact changed during verification")
    if len(data)!=st1.st_size: raise ReleaseVerificationError("artifact read race")
    return data

def _load_json(path:Path,limit:int):
    try: return json.loads(_read(path,limit).decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_const)
    except ReleaseVerificationError: raise
    except Exception as exc: raise ReleaseVerificationError("invalid bounded JSON artifact") from exc

def _mapping(v,label,keys):
    if not isinstance(v,dict) or set(v)!=set(keys): raise ReleaseVerificationError(f"invalid {label} schema")
    return v

def _str(v,label,pat=_IDENTIFIER):
    if not isinstance(v,str) or not pat.fullmatch(v): raise ReleaseVerificationError(f"invalid {label}")
    return v

def _num(v,label,lo,hi,integer=True):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)): raise ReleaseVerificationError(f"invalid {label}")
    if integer and type(v) is not int: raise ReleaseVerificationError(f"invalid {label}")
    if not lo<=float(v)<=hi: raise ReleaseVerificationError(f"invalid {label}")
    return int(v) if integer else float(v)

def _safe_path(root:Path,relative:str, *, exact_filename:bool=True)->Path:
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts: raise ReleaseVerificationError("artifact path escapes release root")
    if exact_filename and ("/" in relative or "\\" in relative): raise ReleaseVerificationError("artifact filename must not contain separators")
    try:
        base=root.resolve(strict=True); resolved=(base/relative).resolve(strict=True)
    except OSError as exc: raise ReleaseVerificationError("artifact path missing") from exc
    if base != resolved.parent and base not in resolved.parents: raise ReleaseVerificationError("artifact path escapes release root")
    return resolved

def _digest_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def _manifest_claim_digest(raw:Any)->str:
    clone=json.loads(json.dumps(raw,ensure_ascii=False,sort_keys=True,separators=(",",":")))
    if isinstance(clone,dict):
        clone["canonical_manifest_digest"]="0"*64
        if isinstance(clone.get("approval"),dict): clone["approval"]["manifest_digest"]="0"*64
    return hashlib.sha256(json.dumps(clone,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def _timestamp(v):
    if not isinstance(v,str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",v): raise ReleaseVerificationError("invalid timestamp")
    return v

def validate_provider_config(raw:Any)->ProviderConfig:
    keys={"schema_version","model_architecture","vocabulary_size","embedding_width","layer_count","attention_head_count","feed_forward_width","approved_context_length","max_generated_proposal_tokens","special_token_ids","dropout","generation_method"}
    r=_mapping(raw,"provider config",keys)
    if r["schema_version"]!=_PROVIDER_SCHEMA: raise ReleaseVerificationError("unsupported provider schema")
    arch=_str(r["model_architecture"],"architecture")
    vocab=_num(r["vocabulary_size"],"vocabulary_size",32,65536); emb=_num(r["embedding_width"],"embedding_width",8,4096); heads=_num(r["attention_head_count"],"attention_head_count",1,128)
    if emb%heads: raise ReleaseVerificationError("embedding width must divide by attention heads")
    layers=_num(r["layer_count"],"layer_count",1,96); ff=_num(r["feed_forward_width"],"feed_forward_width",8,16384); ctx=_num(r["approved_context_length"],"context",64,32768); maxg=_num(r["max_generated_proposal_tokens"],"max generated",8,2048)
    specials=_mapping(r["special_token_ids"],"special_token_ids",{"bos","eos","pad","unk"})
    spec={k:_num(v,k,0,vocab-1) for k,v in specials.items()}
    if len(set(spec.values()))!=4: raise ReleaseVerificationError("special-token collision")
    dropout=_num(r["dropout"],"dropout",0,0,integer=False)
    if r["generation_method"]!="greedy": raise ReleaseVerificationError("serving generation must be deterministic greedy")
    return ProviderConfig(arch,vocab,emb,layers,heads,ff,ctx,maxg,spec,dropout,"greedy")

def validate_evaluation_report(raw:Any, release_id:str)->Evaluation:
    r=_mapping(raw,"evaluation",{"schema_version","release_id","runtime_abi","dataset_digest","categories","metrics","thresholds","passed","evaluated_at","evaluator"})
    if r["schema_version"]!=_EVAL_SCHEMA or r["release_id"]!=release_id or r["runtime_abi"]!=RUNTIME_ABI: raise ReleaseVerificationError("evaluation identity mismatch")
    _str(r["dataset_digest"],"dataset digest",_SHA256); _timestamp(r["evaluated_at"]); _str(r["evaluator"],"evaluator")
    cats=r["categories"]
    if not isinstance(cats,list) or set(cats)!=REQ_CATS or len(cats)!=len(set(cats)): raise ReleaseVerificationError("evaluation categories incomplete")
    metrics=_mapping(r["metrics"],"metrics",REQ_CATS); thresholds=_mapping(r["thresholds"],"thresholds",REQ_CATS)
    for c in REQ_CATS:
        if _num(metrics[c],c,0,1,integer=False) < _num(thresholds[c],c,0,1,integer=False):
            if r["passed"] is True: raise ReleaseVerificationError("evaluation metric/status inconsistency")
            raise ReleaseVerificationError("evaluation failed")
    if r["passed"] is not True: raise ReleaseVerificationError("evaluation failed")
    return Evaluation("", True, tuple(sorted(cats)))

def _parse_manifest(raw:Any)->LocalLanguageRelease:
    keys={"schema_version","release_id","version","role","runtime_abi","provider_implementation","provider_config_artifact","tokenizer_artifact","weights_artifact","evaluation_report_artifact","release_created","canonical_manifest_digest","approval","evaluation","artifacts"}
    m=_mapping(raw,"release manifest",keys)
    if m["schema_version"]!=_MANIFEST_SCHEMA: raise ReleaseVerificationError("unsupported manifest schema")
    rid=_str(m["release_id"],"release_id"); ver=_str(m["version"],"version")
    if m["role"]!=ReleaseRole.INPUT.value: raise ReleaseVerificationError("unsupported language-adapter role")
    if m["runtime_abi"]!=RUNTIME_ABI: raise ReleaseVerificationError("unsupported runtime ABI")
    impl=_str(m["provider_implementation"],"provider implementation")
    approval=_mapping(m["approval"],"approval",{"state","approval_id","approved_by","approved_at","manifest_digest"})
    if approval["state"]!="approved" or not _IDENTIFIER.fullmatch(str(approval["approval_id"])): raise ReleaseVerificationError("release is not explicitly approved")
    _timestamp(approval["approved_at"]); _str(approval["approved_by"],"approved_by"); _str(approval["manifest_digest"],"approval digest",_SHA256)
    evmeta=_mapping(m["evaluation"],"evaluation metadata",{"passed","report_sha256"})
    if evmeta["passed"] is not True or not _SHA256.fullmatch(str(evmeta["report_sha256"])): raise ReleaseVerificationError("evaluation metadata not passing")
    artifacts=[]; items=m["artifacts"]
    if not isinstance(items,list): raise ReleaseVerificationError("invalid artifacts")
    for it in items:
        a=_mapping(it,"artifact",{"name","path","sha256","byte_size"}); name=_str(a["name"],"artifact name")
        path=a["path"]; sha=_str(a["sha256"],"artifact digest",_SHA256); size=_num(a["byte_size"],"artifact size",1,MAX_ARTIFACT)
        artifacts.append(Artifact(name,path,sha,size))
    if {a.name for a in artifacts}!=_REQUIRED_ARTIFACTS or len(artifacts)!=4 or len({a.path for a in artifacts})!=4: raise ReleaseVerificationError("release must bind fixed artifacts exactly once")
    amap={a.name:a for a in artifacts}
    for name,fn in _ARTIFACT_FILES.items():
        if amap[name].path != fn: raise ReleaseVerificationError("unexpected artifact filename")
    if m["provider_config_artifact"]!=amap["config"].path or m["tokenizer_artifact"]!=amap["tokenizer"].path or m["weights_artifact"]!=amap["weights"].path or m["evaluation_report_artifact"]!=amap["evaluation_report"].path: raise ReleaseVerificationError("artifact selector mismatch")
    return LocalLanguageRelease(rid,ver,ReleaseRole.INPUT,RUNTIME_ABI,impl,m["provider_config_artifact"],m["tokenizer_artifact"],m["weights_artifact"],m["evaluation_report_artifact"],_timestamp(m["release_created"]),str(m["canonical_manifest_digest"]),approval["state"],approval["approval_id"],tuple(artifacts),Evaluation(evmeta["report_sha256"],True,()))

def verify_release(release_root: str|Path)->LocalLanguageRelease:
    root=Path(release_root)
    if not root.is_absolute(): root=root.resolve()
    if not root.is_dir() or root.is_symlink(): raise ReleaseVerificationError("release root is not a directory")
    allowed=set(_ARTIFACT_FILES.values())|{"manifest.json"}
    for child in root.iterdir():
        if child.name not in allowed: raise ReleaseVerificationError(f"unexpected file in release root: {child.name}")
    manifest_path=_safe_path(root,"manifest.json")
    manifest_bytes=_read(manifest_path,MAX_MANIFEST)
    try: raw=json.loads(manifest_bytes.decode(),object_pairs_hook=_pairs,parse_constant=_bad_const)
    except ReleaseVerificationError: raise
    except Exception as exc: raise ReleaseVerificationError("unreadable release manifest") from exc
    rel=_parse_manifest(raw)
    manifest_digest=_manifest_claim_digest(raw)
    if rel.manifest_digest!=manifest_digest: raise ReleaseVerificationError("canonical manifest digest mismatch")
    amap={a.name:a for a in rel.artifacts}
    artifact_bytes={}
    for a in rel.artifacts:
        path=_safe_path(root,a.path); data=_read(path,MAX_ARTIFACT)
        if len(data)!=a.byte_size or _digest_bytes(data)!=a.sha256: raise ReleaseVerificationError(f"artifact digest or byte-size mismatch: {a.name}")
        artifact_bytes[a.name]=data
    cfg=validate_provider_config(json.loads(artifact_bytes["config"].decode(),object_pairs_hook=_pairs,parse_constant=_bad_const))
    eval_report=validate_evaluation_report(json.loads(artifact_bytes["evaluation_report"].decode(),object_pairs_hook=_pairs,parse_constant=_bad_const), rel.release_id)
    from .tokenizer import validate_tokenizer_contract
    validate_tokenizer_contract(json.loads(artifact_bytes["tokenizer"].decode(),object_pairs_hook=_pairs,parse_constant=_bad_const), cfg)
    return LocalLanguageRelease(**{**rel.__dict__,"provider_config":cfg,"evaluation":Evaluation(amap["evaluation_report"].sha256,eval_report.passed,eval_report.categories)})
