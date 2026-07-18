"""Framework-independent typed semantic orchestration boundary."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vulcan.memory.governed import GovernedMemoryPort
from .case import CognitiveCase, CognitiveCaseStatus
from .finalization import ResponseFinalizerPort
from .semantic import (ClarificationRequest, DeterministicLanguageInput, LanguageInputPort, RESPONSE_IR_VERSION, ResponseIR, ResponseMode, Utterance, accept, build_graphix_plan, compile_graphix_plan, execute, execute_graphix_plan, render_strict, validate_proposal)
from .output import DeterministicLanguageOutput, LanguageOutputPort, SemanticFirewall, project
@dataclass(frozen=True)
class KernelRequest:
    utterance: Utterance
    conversation_id: str | None
@dataclass(frozen=True)
class KernelResult:
    response: str
    response_ir: ResponseIR
    status: CognitiveCaseStatus
    finalization: str
    def transport(self, *, case_id: str, runtime_id: str, snapshot_id: str | None) -> dict[str, object]:
        return {"response": self.response, "metadata": {"case_id":case_id,"runtime_id":runtime_id,"state_snapshot_id":snapshot_id,"semantic_schema_version":self.response_ir.schema_version,"finalized":True,"finalization_safety_decision":self.finalization}}
class CognitiveKernel:
    def __init__(self, *, state_authority: Any, finalizer: ResponseFinalizerPort, language_input: LanguageInputPort | None = None, language_output: LanguageOutputPort | None = None, memory: "GovernedMemoryPort | None" = None, audit: Any = None, alignment: Any = None) -> None:
        # The kernel owns the only memory port exposed to the production path.
        # It deliberately does not turn retrieved text into executable semantics.
        self._state_authority=state_authority; self._finalizer=finalizer; self._language_input=language_input or DeterministicLanguageInput(); self._language_output=language_output or DeterministicLanguageOutput(); self._memory=memory; self._audit=audit; self._alignment=alignment; self.calls=0
    def capabilities(self) -> tuple[str, ...]:
        """Capabilities implemented by this composed kernel, not marketing text."""
        caps=["bounded-arithmetic"]
        mem_caps=getattr(self._memory, "capabilities", None)
        if callable(mem_caps):
            try: caps.extend(mem_caps())
            except Exception: pass
        return tuple(caps)
    async def handle(self, request: KernelRequest, case: CognitiveCase) -> KernelResult:
        if case.terminal_status is not CognitiveCaseStatus.OPEN: raise RuntimeError("kernel received a closed cognitive case")
        if request.utterance.digest != case.input_hash or request.conversation_id != case.conversation_id: raise ValueError("request/case correlation mismatch")
        self.calls += 1; case.state_snapshot_id=self._snapshot_id(); case.record("semantic_ingress")
        alignment_lease = self._alignment.lease() if self._alignment is not None else None
        policy = getattr(alignment_lease, "policy", None)
        if self._audit: self._audit.append("case.started", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"conversation_id":case.conversation_id or "","state_snapshot_id":case.state_snapshot_id})
        try:
            try:
                proposal=await self._language_input.propose(request.utterance)
                bundle=validate_proposal(request.utterance,proposal)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A proposer can never turn an error into a provider answer.
                case.record("input_proposal_unavailable")
                bundle=validate_proposal(request.utterance, await DeterministicLanguageInput().propose(request.utterance))
            case.interpretation=bundle; selection=accept(bundle)
            if self._audit: self._audit.append("case.interpreted", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"parser_identity":bundle.parser_identity,"candidate_count":len(bundle.candidates)})
            if isinstance(selection, ClarificationRequest):
                case.clarification=selection
                # A clarification is an explicit unknown claim, not an evaluation side effect.
                claim, derivation=execute(type("Unsupported", (), {"operation":"unsupported", "expression":"", "assumptions":()})())
                if self._audit: self._audit.append("case.plan_compiled", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"operation":"unsupported","plan_digest":"0"*64,"plan_shape":{"operands":0},"domain_snapshot_id":"domain:none","alignment_policy_digest":getattr(policy,"policy_digest","")})
                case.append_ledger(claim=claim,derivation=derivation)
                if self._audit: self._audit.append("case.ledger_committed", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"claim_digests":[claim.claim_id],"derivation_digests":[derivation.derivation_id],"evidence_ids":[],"evidence":[]})
                decision = self._alignment.decide(case.claims, case.evidence, case.derivations, policy) if self._alignment is not None else type("D",(),{"accepted": False, "reason_codes": ("unknown_abstain",), "policy_digest":"", "policy_revision":0})()
                if self._audit: self._audit.append("case.alignment_decided", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"accepted":False,"reason_codes":list(decision.reason_codes),"policy_digest":decision.policy_digest,"policy_revision":decision.policy_revision})
                mode=ResponseMode.CLARIFICATION; status=CognitiveCaseStatus.ABSTAINED; accepted_id=None
            else:
                case.accepted_interpretation=selection
                domain_port=getattr(self._state_authority,"domain",None)
                lease_cm=domain_port.lease() if hasattr(domain_port,"lease") else None
                leased_domain=lease_cm if lease_cm is not None else domain_port
                try:
                    domain_snapshot_id=getattr(leased_domain,"domain_snapshot_id","domain:none")
                    plan=build_graphix_plan(selection, request_digest=request.utterance.digest, state_snapshot_id=case.state_snapshot_id or "", domain_snapshot_id=domain_snapshot_id)
                    compiled=compile_graphix_plan(plan, request_digest=request.utterance.digest, state_snapshot_id=case.state_snapshot_id or "", domain_snapshot_id=domain_snapshot_id)
                    if self._audit: self._audit.append("case.plan_compiled", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"operation":compiled.plan.operation,"plan_digest":compiled.plan_digest,"plan_shape":{"operands":len(compiled.plan.operands)},"domain_snapshot_id":domain_snapshot_id,"alignment_policy_digest":getattr(policy,"policy_digest","")})
                    if leased_domain is not None and getattr(leased_domain,"domain_snapshot_id",None) != compiled.plan.domain_snapshot_id: raise ValueError("plan snapshot mismatch")
                    claim,derivation,evidence=execute_graphix_plan(compiled, request_digest=request.utterance.digest, state_snapshot_id=case.state_snapshot_id or "", domain_snapshot_id=domain_snapshot_id, case_id=case.case_id, domain=leased_domain)
                finally:
                    if lease_cm is not None: lease_cm.close()
                case.append_ledger(claim=claim,derivation=derivation,evidence=evidence)
                if self._audit: self._audit.append("case.ledger_committed", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"claim_digests":[c.claim_id for c in case.claims],"derivation_digests":[d.derivation_id for d in case.derivations],"evidence_ids":[e.artifact_id for e in case.evidence],"evidence":[{"evidence_id":e.artifact_id,"origin":e.origin,"content_digest":e.content_digest,"citation":e.citation or "","source_integrity":e.source_integrity,"valid_until":e.valid_until.isoformat().replace("+00:00","Z") if e.valid_until else ""} for e in case.evidence]})
                decision = self._alignment.decide(case.claims, case.evidence, case.derivations, policy) if self._alignment is not None else type("D",(),{"accepted": claim.status.value in {"computed","retrieved"}, "reason_codes": ("passed",), "policy_digest":"", "policy_revision":0})()
                if self._audit: self._audit.append("case.alignment_decided", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"accepted":decision.accepted,"reason_codes":list(decision.reason_codes),"policy_digest":decision.policy_digest,"policy_revision":decision.policy_revision})
                mode=ResponseMode.STRICT if decision.accepted and claim.status.value in {"computed","retrieved","proven"} else ResponseMode.UNKNOWN; status=CognitiveCaseStatus.SUCCESS if mode is ResponseMode.STRICT else CognitiveCaseStatus.ABSTAINED; accepted_id=selection.interpretation_id
            response_ir=ResponseIR(RESPONSE_IR_VERSION,f"response-{case.case_id}",case.case_id,accepted_id,case.state_snapshot_id,mode,(claim.claim_id,))
            case.response_ir=response_ir
            # The adapter sees only the projection; firewall rejection is always strict fallback.
            projection=project(response_ir, case.claims)
            try:
                draft=await self._language_output.render(projection)
                if SemanticFirewall().validate(projection, draft).accepted:
                    case.record("output_draft_validated")
                else:
                    case.record("output_draft_rejected")
            except asyncio.CancelledError:
                raise
            except Exception:
                # Provider/adapter failures are diagnostics only; strict rendering remains authoritative.
                case.record("output_draft_unavailable")
            artifact=render_strict(response_ir,case.claims,case.derivations,case.evidence); case.render_artifact=artifact; case.record("strict_rendered")
            finalization=await self._finalizer.finalize(artifact); case.record_finalization(finalization.decision.value)
            if self._audit: self._audit.append("case.finalized", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"finalization":finalization.decision.value,"rendered_response_digest":artifact.ir_digest})
            case.close(status)
            if self._audit: self._audit.append("case.completed" if status is CognitiveCaseStatus.SUCCESS else "case.abstained", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"status":status.value,"rendered_response_digest":artifact.ir_digest})
            return KernelResult(finalization.public_text,response_ir,status,finalization.decision.value)
        except asyncio.CancelledError:
            if case.terminal_status is CognitiveCaseStatus.OPEN: case.close(CognitiveCaseStatus.CANCELLED,"cancelled")
            raise
        except Exception as exc:
            if case.terminal_status is CognitiveCaseStatus.OPEN:
                if self._audit:
                    try: self._audit.append("case.failed", {"case_id":case.case_id,"request_id":case.request_id,"request_digest":case.input_hash,"category":type(exc).__name__})
                    except Exception: pass
                case.close(CognitiveCaseStatus.FAILED,type(exc).__name__)
            raise
        finally:
            if alignment_lease is not None: alignment_lease.close()
    def _snapshot_id(self)->str:
        candidate=getattr(self._state_authority,"version",None) or getattr(self._state_authority,"snapshot_id",None)
        if candidate is None: return "world-state:unversioned"
        return f"world-state:{candidate}"
