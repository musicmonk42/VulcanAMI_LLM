"""Verified transformer span adapter for the canonical language input port."""
from __future__ import annotations
import asyncio, json, math, threading
from dataclasses import dataclass
from typing import Any, Protocol
from vulcan.runtime.semantic import InterpretationProposal, ProposedCandidate, SCHEMA_VERSION, SourceSpan, Utterance

PROPOSAL_SCHEMA="transformer-span-proposal/1"
RUNTIME_ABI="vulcan-transformer-span/1"
ALLOWED_OPERATIONS=frozenset({"arithmetic","lookup","memory_read","memory_write","memory_forget","unsupported"})
ARGS={"arithmetic":frozenset({"expression"}),"lookup":frozenset({"key"}),"memory_read":frozenset({"request_span"}),"memory_write":frozenset({"request_span"}),"memory_forget":frozenset({"request_span"}),"unsupported":frozenset({"request_span"})}
FORBIDDEN=frozenset({"answer","response","prose","markdown","evidence","citation","citations","domain_hint","domain","graphix_plan","plan","code","tool","tools","sql","path","parameters","facts","value","values","memory_identity","policy"})

class SpanProposalError(ValueError): pass
class AdapterClosedError(RuntimeError): pass

def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise SpanProposalError(f"duplicate JSON key: {k}")
        d[k]=v
    return d

def _bad_const(x): raise SpanProposalError("non-finite JSON number")

def _loads_exact(raw: str) -> Any:
    if not isinstance(raw,str) or not raw or len(raw)>16384: raise SpanProposalError("invalid proposal bytes")
    dec=json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_bad_const)
    try: obj,end=dec.raw_decode(raw)
    except json.JSONDecodeError as exc: raise SpanProposalError("invalid proposal JSON") from exc
    if end != len(raw): raise SpanProposalError("proposal has trailing data")
    # Enforce canonical/minimal JSON so prose stripping and alternate NaN spellings never pass.
    if json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False) != raw:
        raise SpanProposalError("noncanonical proposal JSON")
    return obj

def _span(v: Any, text_len:int) -> SourceSpan:
    if not isinstance(v,dict) or set(v)!={"start","end"}: raise SpanProposalError("invalid span schema")
    s,e=v["start"],v["end"]
    if type(s) is not int or type(e) is not int or s<0 or e<=s or e>text_len: raise SpanProposalError("invalid source span")
    return SourceSpan(s,e)

def parse_transformer_span_proposal(raw: str, utterance: Utterance) -> InterpretationProposal:
    obj=_loads_exact(raw)
    if not isinstance(obj,dict) or set(obj)!={"schema_version","candidates"}: raise SpanProposalError("invalid proposal schema")
    if obj["schema_version"]!=PROPOSAL_SCHEMA: raise SpanProposalError("unsupported proposal schema")
    cands=obj["candidates"]
    if not isinstance(cands,list) or not 1<=len(cands)<=4: raise SpanProposalError("invalid candidates")
    out=[]
    occupied=[]
    for c in cands:
        if not isinstance(c,dict) or set(c)!={"operation","span","argument_spans","confidence"}: raise SpanProposalError("invalid candidate schema")
        if set(c) & FORBIDDEN: raise SpanProposalError("forbidden proposal field")
        op=c["operation"]
        if op not in ALLOWED_OPERATIONS: raise SpanProposalError("invalid operation")
        conf=c["confidence"]
        if isinstance(conf,bool) or not isinstance(conf,(int,float)) or not math.isfinite(float(conf)) or not (0.0<=float(conf)<=1.0): raise SpanProposalError("invalid confidence")
        span=_span(c["span"], len(utterance.text))
        args=c["argument_spans"]
        allowed=ARGS[op]
        if not isinstance(args,dict) or set(args)!=allowed: raise SpanProposalError("invalid operation arguments")
        spans=[span]
        for name,val in args.items():
            if name in FORBIDDEN: raise SpanProposalError("forbidden argument")
            if not isinstance(val,dict): raise SpanProposalError("literal argument value rejected")
            spans.append(_span(val,len(utterance.text)))
        # no overlapping argument spans; candidate span may contain them
        argsp=spans[1:]
        for i,a in enumerate(argsp):
            for b in argsp[i+1:]:
                if max(a.start,b.start) < min(a.end,b.end): raise SpanProposalError("overlapping argument spans")
        expr_span=next(iter(args.values()))
        expr=utterance.text[expr_span["start"]:expr_span["end"]].strip()
        out.append(ProposedCandidate(op, expr, _span(expr_span,len(utterance.text)), float(conf)))
        occupied.append((span.start,span.end))
    out.sort(key=lambda x: x.diagnostic_confidence or 0.0, reverse=True)
    return InterpretationProposal(SCHEMA_VERSION, tuple(out), "verified-local-transformer-span/1")

class RawSpanProvider(Protocol):
    def generate(self, prompt: str, *, max_tokens:int) -> str: ...

PROMPT_CONTRACT='Return exactly canonical JSON schema transformer-span-proposal/1 with operation and spans only. Request:'

@dataclass(frozen=True)
class VerifiedAdapterMetadata:
    release_digest: str; runtime_abi: str=RUNTIME_ABI; mode: str="transformer_proposal"

class VerifiedLocalSpanCompletion:
    def __init__(self, *, provider: RawSpanProvider, tokenizer: Any, metadata: VerifiedAdapterMetadata, context_length:int, max_generated_tokens:int, timeout_seconds:float=2.0, concurrency_safe:bool=False):
        self._provider=provider; self._tokenizer=tokenizer; self.metadata=metadata; self.context_length=context_length; self.max_generated_tokens=max_generated_tokens; self.timeout_seconds=timeout_seconds; self._closed=False; self._lock=threading.Lock() if not concurrency_safe else None
    def readiness(self):
        if self._closed: raise AdapterClosedError("adapter closed")
        return {"mode":"transformer_proposal","runtime_abi":self.metadata.runtime_abi,"release_digest":self.metadata.release_digest}
    def _enc_len(self, text:str)->int:
        enc=getattr(self._tokenizer,"encode",None)
        return len(enc(text)) if callable(enc) else len(text)
    async def propose(self, utterance: Utterance) -> InterpretationProposal:
        if self._closed: raise AdapterClosedError("adapter closed")
        prompt=f"{PROMPT_CONTRACT}\n{utterance.text}\n"
        if self._enc_len(PROMPT_CONTRACT)+self._enc_len(utterance.text)+2+self.max_generated_tokens>self.context_length: raise SpanProposalError("context overflow")
        def call():
            if self._lock:
                with self._lock: return self._provider.generate(prompt,max_tokens=self.max_generated_tokens)
            return self._provider.generate(prompt,max_tokens=self.max_generated_tokens)
        raw=await asyncio.wait_for(asyncio.to_thread(call), timeout=self.timeout_seconds)
        if not isinstance(raw,str): raise SpanProposalError("provider returned non-text")
        return parse_transformer_span_proposal(raw, utterance)
    def close(self):
        if self._closed: return
        self._closed=True
        close=getattr(self._provider,"close",None)
        if close: close()

def build_verified_adapter(*, release_root: str, provider_factory, tokenizer_loader=None, timeout_seconds: float = 2.0) -> VerifiedLocalSpanCompletion:
    """Verify release, construct provider, verify again, closing provider on mutation."""
    from pathlib import Path
    from .release import verify_release
    from .tokenizer import load_tokenizer_contract
    root=Path(release_root)
    first=verify_release(root)
    provider=None
    try:
        provider=provider_factory(first)
        tokenizer=(tokenizer_loader or load_tokenizer_contract)(root / first.tokenizer_artifact)
        second=verify_release(root)
        if second.manifest_digest != first.manifest_digest:
            raise SpanProposalError("release changed during adapter construction")
        cfg=second.provider_config
        return VerifiedLocalSpanCompletion(provider=provider, tokenizer=tokenizer, metadata=VerifiedAdapterMetadata(second.manifest_digest), context_length=cfg.approved_context_length if cfg else 0, max_generated_tokens=cfg.max_generated_proposal_tokens if cfg else 0, timeout_seconds=timeout_seconds)
    except Exception:
        if provider is not None:
            close=getattr(provider,"close",None)
            if close: close()
        raise
