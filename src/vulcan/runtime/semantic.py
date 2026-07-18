"""Closed semantic contracts for the canonical runtime.

Language ports may suggest spans only.  The server reconstructs executable
arithmetic from those spans before a request can reach the evaluator.
"""
from __future__ import annotations
import ast
import hashlib
import html
import json
import math
import operator
import re
import unicodedata
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
from typing import Any, Protocol
from uuid import uuid4

SCHEMA_VERSION = "semantic-ingress/2"
LEDGER_VERSION = "semantic-ledger/2"
RESPONSE_IR_VERSION = "response-ir/3"
MAX_UTTERANCE_CHARS = 10_000
MAX_CANDIDATES = 4
MAX_REFERENCE = 512
MAX_TRACE = 1024
MAX_AST_NODES = 64
MAX_AST_DEPTH = 16
MAX_INTEGER_BITS = 256
MAX_EXPONENT = 32

class EpistemicStatus(str, Enum):
    PROVEN="proven"; COMPUTED="computed"; OBSERVED="observed"; RETRIEVED="retrieved"; ASSUMED="assumed"; HYPOTHESIS="hypothesis"; CONTESTED="contested"; UNKNOWN="unknown"; ERROR="error"
class EvidenceKind(str, Enum):
    SOURCE_DOCUMENT="source_document"; SOURCE_EXCERPT="source_excerpt"; OBSERVATION="observation"; TOOL_OUTPUT="tool_output"; RETRIEVED_RECORD="retrieved_record"; FORMAL_PREMISE="formal_premise"; USER_ASSERTION="user_assertion"; POLICY_FACT="policy_fact"; STATE_SNAPSHOT="state_snapshot"
class ExecutionStatus(str, Enum): SUCCESS="success"; PARTIAL="partial"; NOT_APPLICABLE="not_applicable"; UNKNOWN="unknown"; ERROR="error"; CANCELLED="cancelled"
class ResponseMode(str, Enum): STRICT="strict"; CLARIFICATION="clarification"; PARTIAL="partial"; UNKNOWN="unknown"; ERROR="error"; ACTION_CONFIRMATION="action_confirmation"

@dataclass(frozen=True)
class GraphixPlan:
    schema_version: str; plan_id: str; operation: str; request_digest: str; state_snapshot_id: str; domain_snapshot_id: str; operands: tuple[tuple[str, str], ...]; parameters: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True)
class CompiledGraphixPlan:
    plan: GraphixPlan; plan_digest: str

@dataclass(frozen=True)
class Utterance:
    text: str
    digest: str
    locale: str = "und"
    normalization: str = "NFC"
    @classmethod
    def from_text(cls, text: str, locale: str = "und") -> "Utterance":
        if not isinstance(text, str) or not text or len(text) > MAX_UTTERANCE_CHARS:
            raise ValueError("utterance must be a non-empty bounded string")
        normalized = unicodedata.normalize("NFC", text)
        if not re.fullmatch(r"[A-Za-z0-9_-]{2,35}", locale):
            raise ValueError("invalid locale")
        return cls(normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest(), locale)

@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    unit: str = "unicode-codepoint"
    def resolve(self, utterance: Utterance) -> str:
        if self.unit != "unicode-codepoint" or self.start < 0 or self.end <= self.start or self.end > len(utterance.text):
            raise ValueError("invalid source span")
        return utterance.text[self.start:self.end]

@dataclass(frozen=True)
class ProposedCandidate:
    operation: str
    expression: str
    span: SourceSpan
    diagnostic_confidence: float | None = None

@dataclass(frozen=True)
class InterpretationProposal:
    schema_version: str
    candidates: tuple[ProposedCandidate, ...]
    parser_identity: str

class LanguageInputPort(Protocol):
    async def propose(self, utterance: Utterance) -> InterpretationProposal: ...

class DomainLookupPort(Protocol):
    domain_snapshot_id: str
    def lookup_exact(self, key: str) -> object: ...

@dataclass(frozen=True)
class InterpretationBundle:
    schema_version: str
    ontology_version: str
    input_digest: str
    candidates: tuple[ProposedCandidate, ...]
    diagnostics: tuple[str, ...]
    parser_identity: str = "unknown"

@dataclass(frozen=True)
class AcceptedInterpretation:
    candidate_index: int
    operation: str
    expression: str
    interpretation_id: str = ""
    assumptions: tuple[str, ...] = ()

@dataclass(frozen=True)
class ClarificationRequest:
    field: str
    question: str
    clarification_id: str = ""

@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str; kind: EvidenceKind; content_digest: str; reference: str; origin: str; acquisition_method: str; case_id: str
    schema_version: str = LEDGER_VERSION; state_snapshot_id: str | None = None; observed_at: datetime | None = None; valid_until: datetime | None = None
    scope: str = "request"; locale: str = "und"; privacy_class: str = "request-confidential"; source_integrity: str = "not-applicable"; trust_policy: str = "not-evaluated"; citation: str | None = None; supporting_span: SourceSpan | None = None; contradicts: tuple[str, ...] = (); supersedes: tuple[str, ...] = (); limitations: tuple[str, ...] = (); adapter_identity: str = "kernel"; adapter_version: str = "1"
@dataclass(frozen=True)
class Derivation:
    derivation_id: str; method: str; method_version: str; inputs: tuple[str, ...]; output_claim_id: str; assumptions: tuple[str, ...]; parameters: tuple[tuple[str,str], ...]; deterministic: bool; status: ExecutionStatus; trace_digest: str; error_detail: str | None = None
@dataclass(frozen=True)
class Proposition:
    subject: str; predicate: str; object: str; expression: str | None = None; units: str | None = None; negated: bool = False; modality: str = "assertive"; quantifier: str = "specific"
@dataclass(frozen=True)
class Claim:
    claim_id: str; proposition: Proposition; status: EpistemicStatus; evidence_ids: tuple[str, ...] = (); derivation_ids: tuple[str, ...] = (); assumptions: tuple[str, ...] = (); contradictions: tuple[str, ...] = (); citation_ids: tuple[str, ...] = (); scope: str = "request"; temporal_validity: str | None = None; uncertainty: str | None = None; caveat: str | None = None
    @property
    def derivation_id(self) -> str | None:  # temporary compatibility for callers that read singular result derivations
        return self.derivation_ids[0] if len(self.derivation_ids) == 1 else None
@dataclass(frozen=True)
class ResponseIR:
    schema_version: str; response_id: str; case_id: str; accepted_interpretation_id: str | None; state_snapshot_id: str | None; mode: ResponseMode; required_claim_ids: tuple[str, ...]; optional_claim_ids: tuple[str, ...] = (); locale: str = "und"; style: str = "strict"; max_chars: int = 4000; literals: tuple[str, ...] = ()
@dataclass(frozen=True)
class RenderArtifact:
    text: str; renderer: str; renderer_version: str; ir_digest: str; claim_ids: tuple[str, ...]; citation_ids: tuple[str, ...]; locale: str; diagnostics: tuple[str, ...] = ()
_ARITHMETIC = re.compile(r"^[\s0-9+*/%().-]+$")
class DeterministicLanguageInput:
    """Permanent bounded parser baseline; no model, files, or network."""
    identity = "deterministic-parser/2"

    async def propose(self, utterance: Utterance) -> InterpretationProposal:
        text = utterance.text.strip()
        offset = utterance.text.index(text)
        operation = "arithmetic" if _ARITHMETIC.fullmatch(text) else "unsupported"
        return InterpretationProposal(SCHEMA_VERSION, (ProposedCandidate(operation, text, SourceSpan(offset, offset + len(text))),), self.identity)

    def close(self) -> None:
        return None

def _valid_text(value: str, maximum: int = MAX_REFERENCE) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= maximum

def validate_proposal(utterance: Utterance, proposal: InterpretationProposal) -> InterpretationBundle:
    if proposal.schema_version != SCHEMA_VERSION or not _valid_text(proposal.parser_identity) or not proposal.candidates or len(proposal.candidates) > MAX_CANDIDATES:
        raise ValueError("unsupported interpretation proposal")
    validated: list[ProposedCandidate] = []
    for candidate in proposal.candidates:
        if candidate.operation not in {"arithmetic", "lookup", "memory_read", "memory_write", "memory_forget", "unsupported"} or not _valid_text(candidate.expression):
            raise ValueError("unsupported proposal operation")
        if candidate.diagnostic_confidence is not None and (not isinstance(candidate.diagnostic_confidence, float) or not math.isfinite(candidate.diagnostic_confidence)):
            raise ValueError("invalid proposal confidence")
        source = candidate.span.resolve(utterance).strip()
        # Crucially the expression is reconstructed from the span; a provider cannot substitute tokens.
        if candidate.expression != source:
            raise ValueError("proposal expression is not grounded to its source span")
        if candidate.operation == "arithmetic" and not _ARITHMETIC.fullmatch(source):
            raise ValueError("arithmetic proposal contains unsupported tokens")
        validated.append(candidate)
    return InterpretationBundle(SCHEMA_VERSION, "formal-arithmetic/2", utterance.digest, tuple(validated), (), proposal.parser_identity)

def accept(bundle: InterpretationBundle) -> AcceptedInterpretation | ClarificationRequest:
    if len(bundle.candidates) != 1:
        return ClarificationRequest("interpretation", "Please provide one supported, unambiguous request.", f"clarification-{uuid4().hex}")
    candidate = bundle.candidates[0]
    if candidate.operation == "unsupported":
        return ClarificationRequest("interpretation", "Please provide one supported, unambiguous request.", f"clarification-{uuid4().hex}")
    return AcceptedInterpretation(0, candidate.operation, candidate.expression, f"accepted-{uuid4().hex}")

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
def _budget(node: ast.AST, depth: int = 1) -> int:
    if depth > MAX_AST_DEPTH: raise ValueError("expression nesting exceeds budget")
    count = 1
    for child in ast.iter_child_nodes(node): count += _budget(child, depth + 1)
    if count > MAX_AST_NODES: raise ValueError("expression node budget exceeded")
    return count
def _evaluate(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        if node.value.bit_length() > MAX_INTEGER_BITS: raise ValueError("integer exceeds bit budget")
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN: return _UN[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > MAX_EXPONENT: raise ValueError("exponent exceeds budget")
        try: value = _BIN[type(node.op)](left, right)
        except ZeroDivisionError: raise ValueError("division by zero") from None
        if value.numerator.bit_length() > MAX_INTEGER_BITS or value.denominator.bit_length() > MAX_INTEGER_BITS: raise ValueError("result exceeds bit budget")
        return value
    raise ValueError("unsupported expression")
def _format_fraction(value: Fraction) -> str: return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
def execute(accepted: AcceptedInterpretation, *, case_id: str = "case") -> tuple[Claim, Derivation]:
    cid, did = f"claim-{uuid4().hex}", f"derivation-{uuid4().hex}"
    try:
        if accepted.operation != "arithmetic": raise ValueError("unsupported operation")
        parsed = ast.parse(accepted.expression, mode="eval")
        _budget(parsed)
        value = _format_fraction(_evaluate(parsed.body))
        derivation = Derivation(did, "bounded-arithmetic-ast", "2", (), cid, accepted.assumptions, (("precision", "exact-rational"),), True, ExecutionStatus.SUCCESS, canonical_digest(accepted.expression))
        return Claim(cid, Proposition(accepted.expression, "evaluates_to", value, expression=accepted.expression), EpistemicStatus.COMPUTED, derivation_ids=(did,), assumptions=accepted.assumptions), derivation
    except (SyntaxError, ValueError, OverflowError):
        derivation = Derivation(did, "bounded-arithmetic-ast", "2", (), cid, (), (), True, ExecutionStatus.NOT_APPLICABLE, "", "unsupported or bounded expression")
        return Claim(cid, Proposition("request", "support", "unsupported"), EpistemicStatus.UNKNOWN, derivation_ids=(did,), caveat="No factual result was inferred."), derivation

def build_graphix_plan(accepted: AcceptedInterpretation, *, request_digest: str, state_snapshot_id: str, domain_snapshot_id: str = "domain:none") -> GraphixPlan:
    if accepted.operation == "arithmetic":
        operands = (("expression", accepted.expression),)
    elif accepted.operation == "lookup":
        operands = (("key", accepted.expression),)
    elif accepted.operation in {"memory_read", "memory_write", "memory_forget"}:
        operands = (("request_span", accepted.expression),)
    else:
        raise ValueError("unsupported operation")
    return GraphixPlan("graphix-plan/1", f"plan-{uuid4().hex}", accepted.operation, request_digest, state_snapshot_id, domain_snapshot_id, operands)

def compile_graphix_plan(plan: GraphixPlan, *, request_digest: str, state_snapshot_id: str, domain_snapshot_id: str) -> CompiledGraphixPlan:
    _validate_graphix_plan(plan, request_digest=request_digest, state_snapshot_id=state_snapshot_id, domain_snapshot_id=domain_snapshot_id)
    return CompiledGraphixPlan(plan, canonical_digest(plan))

def _operand(plan: GraphixPlan, name: str) -> str:
    matches = [v for k, v in plan.operands if k == name]
    if len(matches) != 1: raise ValueError("invalid operands")
    return matches[0]

def _validate_graphix_plan(plan: GraphixPlan, *, request_digest: str, state_snapshot_id: str, domain_snapshot_id: str) -> None:
    if plan.schema_version != "graphix-plan/1" or not re.fullmatch(r"plan-[0-9a-f]{32}", plan.plan_id): raise ValueError("invalid plan identity")
    if plan.request_digest != request_digest or plan.state_snapshot_id != state_snapshot_id or plan.domain_snapshot_id != domain_snapshot_id: raise ValueError("plan snapshot mismatch")
    if plan.operation not in {"arithmetic", "lookup", "memory_read", "memory_write", "memory_forget"}: raise ValueError("operation is not closed")
    if plan.parameters: raise ValueError("plan parameters are closed")
    keys = [k for k, _ in plan.operands]
    if len(keys) != len(set(keys)) or not keys: raise ValueError("invalid operands")
    if any(k in {"domain_hint", "domain_filter", "evidence_domain", "answer", "evidence", "code"} for k, _ in plan.operands): raise ValueError("evidence selection override rejected")
    for key, value in plan.operands:
        if not _valid_text(key, 64) or not _valid_text(value, MAX_REFERENCE): raise ValueError("invalid operand text")
    if plan.operation == "arithmetic" and set(keys) != {"expression"}: raise ValueError("invalid arithmetic operands")
    if plan.operation == "lookup" and set(keys) != {"key"}: raise ValueError("invalid lookup operands")

def execute_graphix_plan(compiled: CompiledGraphixPlan, *, request_digest: str, state_snapshot_id: str, domain_snapshot_id: str, case_id: str = "case", domain: DomainLookupPort | None = None) -> tuple[Claim, Derivation, tuple[EvidenceArtifact, ...]]:
    if canonical_digest(compiled.plan) != compiled.plan_digest: raise ValueError("compiled plan was mutated")
    _validate_graphix_plan(compiled.plan, request_digest=request_digest, state_snapshot_id=state_snapshot_id, domain_snapshot_id=domain_snapshot_id)
    accepted = AcceptedInterpretation(0, compiled.plan.operation, _operand(compiled.plan, "expression") if compiled.plan.operation == "arithmetic" else _operand(compiled.plan, "key"), compiled.plan.plan_id)
    if compiled.plan.operation == "arithmetic":
        claim, derivation = execute(accepted, case_id=case_id)
        return claim, derivation, ()
    if compiled.plan.operation == "lookup":
        if domain is None or getattr(domain, "domain_snapshot_id", None) != domain_snapshot_id: raise ValueError("domain snapshot mismatch")
        result = domain.lookup_exact(accepted.expression)
        eid_base, cid, did = f"evidence-{uuid4().hex}", f"claim-{uuid4().hex}", f"derivation-{uuid4().hex}"
        if hasattr(result, "status"):
            status = getattr(result, "status")
            if getattr(result, "domain_snapshot_id", None) != domain_snapshot_id: raise ValueError("domain result snapshot mismatch")
            if status != "retrieved":
                epistemic = EpistemicStatus.CONTESTED if status == "contested" else EpistemicStatus.UNKNOWN
                der = Derivation(did, "exact-domain-lookup", "2", (), cid, (), (("key", accepted.expression), ("domain_status", status)), True, ExecutionStatus.NOT_APPLICABLE, canonical_digest((accepted.expression, status)), "domain data unavailable")
                cl = Claim(cid, Proposition(accepted.expression, "lookup_value", "unavailable"), epistemic, (), (did,), caveat=f"Domain lookup abstained: {status}.")
                return cl, der, ()
            value = getattr(result, "value")
            if not _valid_text(value): raise ValueError("lookup miss")
            evidence_artifacts = []
            for i, support in enumerate(getattr(result, "evidence", ())):
                evidence_artifacts.append(EvidenceArtifact(f"{eid_base}-{i}", EvidenceKind.RETRIEVED_RECORD, support.content_digest, support.uri, f"domain:{support.domain}:{support.revision}:{support.fact_id}:{support.evidence_id}", support.acquisition_method, case_id, state_snapshot_id=state_snapshot_id, observed_at=support.acquired_at, valid_until=support.valid_until, source_integrity="digest-verified", trust_policy="domain-registry", citation=support.uri, limitations=("evidence-truncated",) if getattr(result, "truncated", False) else ()))
            eids = tuple(e.artifact_id for e in evidence_artifacts)
            der = Derivation(did, "exact-domain-lookup", "2", eids, cid, (), (("key", accepted.expression), ("domain_snapshot_id", domain_snapshot_id)), True, ExecutionStatus.SUCCESS, canonical_digest((accepted.expression, value, eids)))
            cl = Claim(cid, Proposition(accepted.expression, "lookup_value", value), EpistemicStatus.RETRIEVED, eids, (did,), citation_ids=eids, temporal_validity="snapshot-bound")
            return cl, der, tuple(evidence_artifacts)
        value, citation = result
        if not _valid_text(value): raise ValueError("lookup miss")
        eid = eid_base
        ev = EvidenceArtifact(eid, EvidenceKind.RETRIEVED_RECORD, canonical_digest(value), citation or f"domain:{domain_snapshot_id}:{accepted.expression}", "injected-domain-port", "exact-key", case_id, state_snapshot_id=state_snapshot_id, observed_at=datetime.now(timezone.utc), source_integrity="digest-verified", trust_policy="domain-port", citation=citation)
        der = Derivation(did, "exact-domain-lookup", "1", (eid,), cid, (), (("key", accepted.expression),), True, ExecutionStatus.SUCCESS, canonical_digest((accepted.expression, value)))
        cl = Claim(cid, Proposition(accepted.expression, "lookup_value", value), EpistemicStatus.RETRIEVED, (eid,), (did,), citation_ids=(eid,), temporal_validity="snapshot-bound")
        return cl, der, (ev,)
    raise ValueError("unsupported canonical operation")

def _bounded_id(value: str) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 96 and re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is not None

def validate_ledger(evidence: tuple[EvidenceArtifact, ...], derivations: tuple[Derivation, ...], claims: tuple[Claim, ...], *, case_id: str | None = None) -> None:
    eids, dids, cids = {e.artifact_id for e in evidence}, {d.derivation_id for d in derivations}, {c.claim_id for c in claims}
    all_ids = [*(e.artifact_id for e in evidence), *(d.derivation_id for d in derivations), *(c.claim_id for c in claims)]
    if len(set(all_ids)) != len(all_ids) or not all(_bounded_id(i) for i in all_ids): raise ValueError("duplicate or invalid ledger id")
    if len(eids) != len(evidence) or len(dids) != len(derivations) or len(cids) != len(claims): raise ValueError("duplicate ledger id")
    for e in evidence:
        if case_id is not None and e.case_id != case_id: raise ValueError("cross-case evidence")
        if e.schema_version != LEDGER_VERSION or not re.fullmatch(r"[0-9a-f]{64}", e.content_digest): raise ValueError("invalid evidence integrity")
        if not _valid_text(e.reference) or not _valid_text(e.origin) or not _valid_text(e.acquisition_method): raise ValueError("invalid evidence provenance")
        if e.citation is not None and not _valid_text(e.citation): raise ValueError("invalid citation")
        if e.observed_at is not None and e.observed_at.tzinfo is None: raise ValueError("temporal metadata must be timezone aware")
        if e.valid_until is not None and (e.valid_until.tzinfo is None or (e.observed_at and e.valid_until < e.observed_at)): raise ValueError("invalid temporal metadata")
    for derivation in derivations:
        if derivation.output_claim_id not in cids or len(derivation.trace_digest) > MAX_TRACE: raise ValueError("invalid derivation reference")
        if len(set(derivation.inputs)) != len(derivation.inputs) or derivation.derivation_id in derivation.inputs or derivation.output_claim_id in derivation.inputs: raise ValueError("invalid derivation inputs")
        if not set(derivation.inputs) <= (eids | cids): raise ValueError("dangling derivation input")
    referenced_derivations = {d for c in claims for d in c.derivation_ids}
    if referenced_derivations != dids: raise ValueError("dangling or unused derivation")
    for claim in claims:
        if len(set(claim.evidence_ids)) != len(claim.evidence_ids) or len(set(claim.derivation_ids)) != len(claim.derivation_ids): raise ValueError("duplicate claim reference")
        if not set(claim.evidence_ids) <= eids or not set(claim.derivation_ids) <= dids or not set(claim.contradictions) <= cids or not set(claim.citation_ids) <= set(claim.evidence_ids): raise ValueError("dangling ledger reference")
        if any(d.output_claim_id != claim.claim_id for d in derivations if d.derivation_id in claim.derivation_ids): raise ValueError("claim derivation output mismatch")
        successful = [d for d in derivations if d.derivation_id in claim.derivation_ids and d.status is ExecutionStatus.SUCCESS]
        evs = [e for e in evidence if e.artifact_id in claim.evidence_ids]
        if claim.status is EpistemicStatus.COMPUTED and not successful: raise ValueError("computed claim lacks successful derivation")
        if claim.status is EpistemicStatus.PROVEN and (not successful or not any(e.kind is EvidenceKind.FORMAL_PREMISE for e in evs)): raise ValueError("proven claim lacks formal evidence")
        if claim.status is EpistemicStatus.OBSERVED and not any(e.kind is EvidenceKind.OBSERVATION for e in evs): raise ValueError("observed claim lacks observation")
        if claim.status is EpistemicStatus.RETRIEVED and not any(e.kind is EvidenceKind.RETRIEVED_RECORD for e in evs): raise ValueError("retrieved claim lacks retrieved evidence")
        if claim.status is EpistemicStatus.ASSUMED and not claim.assumptions: raise ValueError("assumption not visible")

def render_strict(ir: ResponseIR, claims: tuple[Claim, ...], derivations: tuple[Derivation, ...] = (), evidence: tuple[EvidenceArtifact, ...] = ()) -> RenderArtifact:
    if ir.schema_version != RESPONSE_IR_VERSION or ir.literals or ir.max_chars <= 0: raise ValueError("invalid response IR")
    validate_ledger(evidence, derivations, claims, case_id=ir.case_id)
    by_id = {claim.claim_id: claim for claim in claims}
    if not set(ir.required_claim_ids) <= set(by_id) or set(ir.required_claim_ids) & set(ir.optional_claim_ids): raise ValueError("IR references unknown or duplicate claim")
    parts: list[str] = []
    for claim_id in ir.required_claim_ids:
        claim = by_id[claim_id]; proposition = claim.proposition
        if claim.status is EpistemicStatus.COMPUTED: text = f"The computed result is {proposition.object}."
        elif claim.status is EpistemicStatus.UNKNOWN: text = "This request is not supported by the deterministic interpreter."
        elif claim.status is EpistemicStatus.ERROR: text = "The deterministic interpreter could not complete this request."
        else: text = f"{claim.status.value.capitalize()}: {proposition.subject} {proposition.predicate} {proposition.object}."
        if claim.caveat: text += f" Caveat: {claim.caveat}"
        parts.append(html.escape(text, quote=False))
    text = "\n".join(parts)
    if len(text) > ir.max_chars: raise ValueError("render bound")
    return RenderArtifact(text, "strict-template", "2", canonical_digest(ir), ir.required_claim_ids, (), ir.locale)

def _canonical(value: Any) -> Any:
    if is_dataclass(value): return _canonical(asdict(value))
    if isinstance(value, Enum): return value.value
    if isinstance(value, datetime): return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, tuple): return [_canonical(v) for v in value]
    if isinstance(value, dict): return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (str, int, bool)) or value is None: return value
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")
def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
