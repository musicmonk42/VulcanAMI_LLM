"""Strict immutable tokenizer and exact decoder contract for span proposals."""
from __future__ import annotations
import json, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .release import ReleaseVerificationError
TOKENIZER_SCHEMA="local-tokenizer/1"; MAX_VOCABULARY=65536
REQ_GRAMMAR=set('{}[],:"-0123456789')|{"schema_version","transformer-span-proposal/1","candidates","operation","span","argument_spans","confidence","start","end","arithmetic","lookup","memory_read","memory_write","memory_forget","unsupported","expression","key","request_span"}
@dataclass(frozen=True)
class ImmutableTokenizerContract:
    normalization:str; vocabulary:tuple[str,...]; special_tokens:dict[str,int]; max_length:int
    def encode(self,text:str)->tuple[int,...]:
        # deterministic longest-token fallback adequate for contract diagnostics
        ids=[]; i=0; pairs=sorted(((t,n) for n,t in enumerate(self.vocabulary)), key=lambda x:len(x[0]), reverse=True)
        while i<len(text):
            for tok,tid in pairs:
                if text.startswith(tok,i): ids.append(tid); i+=len(tok); break
            else: raise ReleaseVerificationError("tokenizer cannot encode input")
        return tuple(ids)
    def decode(self,ids)->str:
        return "".join(self.vocabulary[i] for i in ids)

def _pairs(p):
    d={}
    for k,v in p:
        if k in d: raise ReleaseVerificationError("duplicate tokenizer JSON key")
        d[k]=v
    return d

def validate_tokenizer_contract(raw:Any, provider_config:Any|None=None)->ImmutableTokenizerContract:
    expected={"schema_version","normalization","vocabulary","special_tokens","max_length"}
    if not isinstance(raw,dict) or set(raw)!=expected or raw["schema_version"]!=TOKENIZER_SCHEMA: raise ReleaseVerificationError("invalid tokenizer contract schema")
    vocab=raw["vocabulary"]
    if not isinstance(vocab,list) or not 4<=len(vocab)<=MAX_VOCABULARY: raise ReleaseVerificationError("invalid immutable vocabulary")
    if len(set(vocab))!=len(vocab): raise ReleaseVerificationError("duplicate tokenizer tokens")
    for i,tok in enumerate(vocab):
        if not isinstance(tok,str) or tok=="" or unicodedata.normalize("NFC",tok)!=tok: raise ReleaseVerificationError("tokenizer controls/non-NFC tokens rejected")
        if any(ord(c)==0 or (ord(c)<32 and c not in "\t\n\r") for c in tok): raise ReleaseVerificationError("tokenizer controls/non-NFC tokens rejected")
    specials=raw["special_tokens"]
    if not isinstance(specials,dict) or set(specials)!={"bos","eos","pad","unk"}: raise ReleaseVerificationError("invalid special token map")
    spec={}
    for name,val in specials.items():
        if type(val) is not int or not 0<=val<len(vocab): raise ReleaseVerificationError("invalid special token id")
        spec[name]=val
    if len(set(spec.values()))!=4: raise ReleaseVerificationError("special-token collision")
    if provider_config is not None and getattr(provider_config,"special_token_ids",spec)!=spec: raise ReleaseVerificationError("special-token mismatch")
    if raw["normalization"]!="NFC" or type(raw["max_length"]) is not int or not 0<raw["max_length"]<=10000: raise ReleaseVerificationError("invalid tokenizer bounds")
    missing=[t for t in REQ_GRAMMAR if t not in vocab]
    if missing: raise ReleaseVerificationError("tokenizer missing proposal grammar token")
    return ImmutableTokenizerContract("NFC",tuple(vocab),spec,raw["max_length"])

def load_tokenizer_contract(path: str|Path)->ImmutableTokenizerContract:
    try: raw=json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except Exception as exc: raise ReleaseVerificationError("unreadable tokenizer contract") from exc
    return validate_tokenizer_contract(raw)

def decode_generated_suffix(token_ids, tokenizer:ImmutableTokenizerContract, *, max_tokens:int, require_eos:bool=True)->str:
    ids=list(token_ids)
    if len(ids)>max_tokens: raise ReleaseVerificationError("excessive generated tokens")
    spec=tokenizer.special_tokens; eos=spec["eos"]
    if spec["unk"] in ids or spec["bos"] in ids or spec["pad"] in ids: raise ReleaseVerificationError("UNK/BOS/PAD misuse rejected")
    if require_eos:
        if not ids or eos not in ids: raise ReleaseVerificationError("missing EOS")
        if ids[-1]!=eos or ids.count(eos)!=1: raise ReleaseVerificationError("early EOS and data after EOS rejected")
        ids=ids[:-1]
    try: return tokenizer.decode(ids)
    except Exception as exc: raise ReleaseVerificationError("undecodable token IDs") from exc
