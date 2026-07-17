"""Framework-independent, closed semantic contracts for the canonical runtime."""
from __future__ import annotations
import ast, hashlib, html, json, math, operator, unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

SCHEMA_VERSION = "semantic-ingress/2"; LEDGER_VERSION = "semantic-ledger/1"; RESPONSE_IR_VERSION = "response-ir/2"
MAX_UTTERANCE_CHARS = 10_000; MAX_CANDIDATES = 4; MAX_REFERENCE = 512; MAX_TRACE = 1024
class EpistemicStatus(str, Enum):
    PROVEN="proven"; COMPUTED="computed"; OBSERVED="observed"; RETRIEVED="retrieved"; ASSUMED="assumed"; HYPOTHESIS="hypothesis"; CONTESTED="contested"; UNKNOWN="unknown"; ERROR="error"
class EvidenceKind(str, Enum):
    SOURCE_DOCUMENT="source_document"; SOURCE_EXCERPT="source_excerpt"; OBSERVATION="observation"; TOOL_OUTPUT="tool_output"; RETRIEVED_RECORD="retrieved_record"; FORMAL_PREMISE="formal_premise"; USER_ASSERTION="user_assertion"; POLICY_FACT="policy_fact"; STATE_SNAPSHOT="state_snapshot"
class ExecutionStatus(str, Enum): SUCCESS="success"; PARTIAL="partial"; NOT_APPLICABLE="not_applicable"; UNKNOWN="unknown"; ERROR="error"; CANCELLED="cancelled"
class ResponseMode(str, Enum): STRICT="strict"; CLARIFICATION="clarification"; PARTIAL="partial"; UNKNOWN="unknown"; ERROR="error"
@dataclass(frozen=True)
class Utterance:
    text:str; digest:str; locale:str="und"; normalization:str="NFC"
    @classmethod
    def from_text(cls,text:str,locale:str="und")->"Utterance":
        if not isinstance(text,str) or not text or len(text)>MAX_UTTERANCE_CHARS: raise ValueError("utterance must be bounded")
        text=unicodedata.normalize("NFC",text); return cls(text,hashlib.sha256(text.encode()).hexdigest(),locale)
@dataclass(frozen=True)
class SourceSpan:
    start:int; end:int; unit:str="unicode-codepoint"
    def resolve(self,u:Utterance)->str:
        if self.unit!="unicode-codepoint" or self.start<0 or self.end<=self.start or self.end>len(u.text): raise ValueError("invalid source span")
        return u.text[self.start:self.end]
@dataclass(frozen=True)
class ProposedCandidate: operation:str; expression:str; span:SourceSpan; diagnostic_confidence:float|None=None
@dataclass(frozen=True)
class InterpretationProposal: schema_version:str; candidates:tuple[ProposedCandidate,...]; parser_identity:str
class LanguageInputPort(Protocol):
    async def propose(self, utterance:Utterance)->InterpretationProposal: ...
@dataclass(frozen=True)
class InterpretationBundle: schema_version:str; ontology_version:str; input_digest:str; candidates:tuple[ProposedCandidate,...]; diagnostics:tuple[str,...]
@dataclass(frozen=True)
class AcceptedInterpretation: candidate_index:int; operation:str; expression:str; assumptions:tuple[str,...]=()
@dataclass(frozen=True)
class ClarificationRequest: field:str; question:str
@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id:str; kind:EvidenceKind; content_digest:str; reference:str; origin:str; acquisition_method:str; case_id:str; schema_version:str=LEDGER_VERSION; state_snapshot_id:str|None=None; observed_at:datetime|None=None; valid_until:datetime|None=None; scope:str="request"; locale:str="und"; privacy_class:str="request-confidential"; source_integrity:str="not-applicable"; trust_policy:str="not-evaluated"; citation:str|None=None; supporting_span:SourceSpan|None=None; contradicts:tuple[str,...]=(); supersedes:tuple[str,...]=(); limitations:tuple[str,...]=(); adapter_identity:str="kernel"; adapter_version:str="1"
@dataclass(frozen=True)
class Derivation:
    derivation_id:str; method:str; method_version:str; inputs:tuple[str,...]; output_claim_id:str; assumptions:tuple[str,...]; parameters:tuple[tuple[str,str],...]; deterministic:bool; status:ExecutionStatus; trace_digest:str; error_detail:str|None=None
@dataclass(frozen=True)
class Proposition: subject:str; predicate:str; object:str; expression:str|None=None; units:str|None=None; negated:bool=False; modality:str="assertive"; quantifier:str="specific"
@dataclass(frozen=True)
class Claim:
    claim_id:str; proposition:Proposition; status:EpistemicStatus; evidence_ids:tuple[str,...]=(); derivation_ids:tuple[str,...]=(); assumptions:tuple[str,...]=(); contradictions:tuple[str,...]=(); citation_ids:tuple[str,...]=(); scope:str="request"; temporal_validity:str|None=None; uncertainty:str|None=None; caveat:str|None=None
@dataclass(frozen=True)
class ResponseIR:
    schema_version:str; response_id:str; case_id:str; accepted_interpretation_id:str|None; state_snapshot_id:str|None; mode:ResponseMode; required_claim_ids:tuple[str,...]; optional_claim_ids:tuple[str,...]=(); locale:str="und"; style:str="strict"; max_chars:int=4000; literals:tuple[str,...]=()
@dataclass(frozen=True)
class RenderArtifact: text:str; renderer:str; renderer_version:str; ir_digest:str; claim_ids:tuple[str,...]; citation_ids:tuple[str,...]; locale:str; diagnostics:tuple[str,...]=()
@dataclass(frozen=True)
class UntrustedRenderDraft: text:str
class LanguageOutputPort(Protocol):
    async def render(self, response_ir:ResponseIR, style:str, locale:str, max_chars:int)->UntrustedRenderDraft: ...
class DeterministicLanguageInput:
    async def propose(self,u:Utterance)->InterpretationProposal:
        return InterpretationProposal(SCHEMA_VERSION,(ProposedCandidate("arithmetic",u.text.strip(),SourceSpan(0,len(u.text))),),"deterministic-arithmetic/1")
def validate_proposal(u:Utterance,p:InterpretationProposal)->InterpretationBundle:
    if p.schema_version!=SCHEMA_VERSION or not p.candidates or len(p.candidates)>MAX_CANDIDATES: raise ValueError("unsupported interpretation proposal")
    for c in p.candidates:
        if c.operation!="arithmetic" or c.span.resolve(u)!=u.text or (c.diagnostic_confidence is not None and not math.isfinite(c.diagnostic_confidence)): raise ValueError("invalid proposal")
    return InterpretationBundle(SCHEMA_VERSION,"formal-arithmetic/1",u.digest,p.candidates,())
def accept(b:InterpretationBundle)->AcceptedInterpretation|ClarificationRequest:
    return AcceptedInterpretation(0,b.candidates[0].operation,b.candidates[0].expression) if len(b.candidates)==1 else ClarificationRequest("interpretation","Please choose one unambiguous supported request.")
_BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.Mod:operator.mod}; _UN={ast.UAdd:operator.pos,ast.USub:operator.neg}
def _evaluate(n:ast.AST)->int|float:
    if isinstance(n,ast.Constant) and type(n.value) in (int,float) and math.isfinite(n.value): return n.value
    if isinstance(n,ast.UnaryOp) and type(n.op) in _UN:return _UN[type(n.op)](_evaluate(n.operand))
    if isinstance(n,ast.BinOp) and type(n.op) in _BIN:
        v=_BIN[type(n.op)](_evaluate(n.left),_evaluate(n.right))
        if not math.isfinite(v) or abs(v)>10**100: raise ValueError("bounds")
        return v
    raise ValueError("unsupported")
def execute(a:AcceptedInterpretation)->tuple[Claim,Derivation]:
    try:
        value=str(_evaluate(ast.parse(a.expression,mode="eval").body)); cid="claim-1"; d=Derivation("derivation-1","restricted-python-ast","1",(),cid,a.assumptions,(("precision","exact-integer-or-ieee-float"),),True,ExecutionStatus.SUCCESS,canonical_digest(a.expression))
        return Claim(cid,Proposition(a.expression,"evaluates_to",value,expression=a.expression),EpistemicStatus.COMPUTED,derivation_ids=(d.derivation_id,),assumptions=a.assumptions),d
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):
        return Claim("claim-1",Proposition("request","support","unsupported"),EpistemicStatus.UNKNOWN,caveat="No factual result was inferred."),Derivation("derivation-1","restricted-python-ast","1",(),"claim-1",(),(),True,ExecutionStatus.NOT_APPLICABLE,"", "unsupported syntax")
def validate_ledger(evidence:tuple[EvidenceArtifact,...], derivations:tuple[Derivation,...], claims:tuple[Claim,...])->None:
    eids={e.artifact_id for e in evidence}; dids={d.derivation_id for d in derivations}; cids={c.claim_id for c in claims}
    if len(eids)!=len(evidence) or len(dids)!=len(derivations) or len(cids)!=len(claims): raise ValueError("duplicate ledger id")
    for c in claims:
        if not set(c.evidence_ids)<=eids or not set(c.derivation_ids)<=dids or not set(c.contradictions)<=cids: raise ValueError("dangling ledger reference")
        if c.status is EpistemicStatus.COMPUTED and not any(d.status is ExecutionStatus.SUCCESS and d.output_claim_id==c.claim_id for d in derivations): raise ValueError("computed claim lacks successful derivation")
        if c.status is EpistemicStatus.PROVEN and (not c.evidence_ids or not any(d.status is ExecutionStatus.SUCCESS and d.output_claim_id==c.claim_id for d in derivations)): raise ValueError("proven claim lacks proof inputs")
        if c.status is EpistemicStatus.OBSERVED and not any(e.kind is EvidenceKind.OBSERVATION for e in evidence if e.artifact_id in c.evidence_ids): raise ValueError("observed claim lacks observation")
        if c.status is EpistemicStatus.RETRIEVED and not c.evidence_ids: raise ValueError("retrieved claim lacks artifact")
        if c.status is EpistemicStatus.ASSUMED and not c.assumptions: raise ValueError("assumption not visible")
def render_strict(ir:ResponseIR, claims:tuple[Claim,...])->RenderArtifact:
    byid={c.claim_id:c for c in claims}; validate_ledger((),(),tuple(c for c in claims if not c.derivation_ids)) if False else None
    if not set(ir.required_claim_ids)<=set(byid): raise ValueError("IR references unknown claim")
    parts=[]
    for cid in ir.required_claim_ids:
        c=byid[cid]; p=c.proposition
        if c.status is EpistemicStatus.COMPUTED: text=f"The computed result is {p.object}."
        elif c.status is EpistemicStatus.UNKNOWN: text="This request is not supported by the deterministic interpreter."
        else: text=f"{c.status.value.capitalize()}: {p.subject} {p.predicate} {p.object}."
        if c.caveat:text+=f" Caveat: {c.caveat}"
        parts.append(html.escape(text,quote=False))
    text="\n".join(parts)
    if len(text)>ir.max_chars: raise ValueError("render bound")
    return RenderArtifact(text,"strict-template","1",canonical_digest(ir),ir.required_claim_ids,(),ir.locale)
def canonical_digest(value:object)->str:return hashlib.sha256(json.dumps(value,default=str,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
