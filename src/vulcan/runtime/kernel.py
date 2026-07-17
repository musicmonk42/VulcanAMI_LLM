"""Framework-independent typed semantic orchestration boundary."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any
from .case import CognitiveCase, CognitiveCaseStatus
from .finalization import ResponseFinalizerPort
from .semantic import (ClarificationRequest, DeterministicLanguageInput, LanguageInputPort, RESPONSE_IR_VERSION, ResponseIR, ResponseMode, Utterance, accept, execute, render_strict, validate_proposal)
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
    def __init__(self, *, state_authority: Any, finalizer: ResponseFinalizerPort, language_input: LanguageInputPort | None = None, memory: Any | None = None) -> None:
        # The kernel owns the only memory port exposed to the production path.
        # It deliberately does not turn retrieved text into executable semantics.
        self._state_authority=state_authority; self._finalizer=finalizer; self._language_input=language_input or DeterministicLanguageInput(); self._memory=memory; self.calls=0
    async def handle(self, request: KernelRequest, case: CognitiveCase) -> KernelResult:
        if case.terminal_status is not CognitiveCaseStatus.OPEN: raise RuntimeError("kernel received a closed cognitive case")
        if request.utterance.digest != case.input_hash or request.conversation_id != case.conversation_id: raise ValueError("request/case correlation mismatch")
        self.calls += 1; case.state_snapshot_id=self._snapshot_id(); case.record("semantic_ingress")
        try:
            proposal=await self._language_input.propose(request.utterance); bundle=validate_proposal(request.utterance,proposal); case.interpretation=bundle; selection=accept(bundle)
            if isinstance(selection, ClarificationRequest):
                case.clarification=selection
                # A clarification is an explicit unknown claim, not an evaluation side effect.
                claim, derivation=execute(type("Unsupported", (), {"operation":"unsupported", "expression":"", "assumptions":()})())
                case.append_ledger(claim=claim,derivation=derivation); mode=ResponseMode.CLARIFICATION; status=CognitiveCaseStatus.ABSTAINED; accepted_id=None
            else:
                case.accepted_interpretation=selection; claim,derivation=execute(selection,case_id=case.case_id); case.append_ledger(claim=claim,derivation=derivation)
                mode=ResponseMode.STRICT if claim.status.value=="computed" else ResponseMode.UNKNOWN; status=CognitiveCaseStatus.SUCCESS if mode is ResponseMode.STRICT else CognitiveCaseStatus.ABSTAINED; accepted_id=selection.interpretation_id
            response_ir=ResponseIR(RESPONSE_IR_VERSION,f"response-{case.case_id}",case.case_id,accepted_id,case.state_snapshot_id,mode,(claim.claim_id,))
            case.response_ir=response_ir; artifact=render_strict(response_ir,case.claims,case.derivations,case.evidence); case.render_artifact=artifact; case.record("strict_rendered")
            finalization=await self._finalizer.finalize(artifact); case.record_finalization(finalization.decision.value); case.close(status)
            return KernelResult(finalization.public_text,response_ir,status,finalization.decision.value)
        except asyncio.CancelledError:
            if case.terminal_status is CognitiveCaseStatus.OPEN: case.close(CognitiveCaseStatus.CANCELLED,"cancelled")
            raise
        except Exception as exc:
            if case.terminal_status is CognitiveCaseStatus.OPEN: case.close(CognitiveCaseStatus.FAILED,type(exc).__name__)
            raise
    def _snapshot_id(self)->str:
        candidate=getattr(self._state_authority,"version",None) or getattr(self._state_authority,"snapshot_id",None)
        if candidate is None: return "world-state:unversioned"
        return f"world-state:{candidate}"
