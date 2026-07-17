"""Framework-independent typed semantic orchestration boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .case import CognitiveCase, CognitiveCaseStatus
from .semantic import (AcceptedInterpretation, ClarificationRequest, DeterministicLanguageInput,
                       InterpretationBundle, LanguageInputPort, ResponseIR, ResponseMode, Utterance, accept,
                       execute, render_strict, validate_ledger, validate_proposal)

@dataclass(frozen=True)
class KernelRequest:
    utterance: Utterance
    conversation_id: str | None

@dataclass(frozen=True)
class KernelResult:
    response: str
    response_ir: ResponseIR
    status: CognitiveCaseStatus

    def transport(self, *, case_id: str, runtime_id: str, snapshot_id: str | None) -> dict[str, object]:
        return {"response": self.response, "metadata": {"case_id": case_id, "runtime_id": runtime_id,
                "state_snapshot_id": snapshot_id, "semantic_schema_version": self.response_ir.schema_version}}

class CognitiveKernel:
    """Owns acceptance, computation, claims, response planning, and strict rendering."""
    def __init__(self, *, state_authority: Any, language_input: LanguageInputPort | None = None) -> None:
        self._state_authority = state_authority
        self._language_input = language_input or DeterministicLanguageInput()
        self.calls = 0

    async def handle(self, request: KernelRequest, case: CognitiveCase) -> KernelResult:
        if case.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("kernel received a closed cognitive case")
        self.calls += 1
        case.state_snapshot_id = self._snapshot_id()
        case.record("semantic_ingress")
        try:
            proposal = await self._language_input.propose(request.utterance)
            bundle = validate_proposal(request.utterance, proposal)
            case.interpretation = bundle
            selection = accept(bundle)
            if isinstance(selection, ClarificationRequest):
                case.clarification = selection
                claim, derivation = execute(AcceptedInterpretation(0, "arithmetic", "invalid"))
                case.append_ledger(claim=claim, derivation=derivation)
                response_ir = ResponseIR("response-ir/2", "response-1", case.case_id, None, case.state_snapshot_id, ResponseMode.CLARIFICATION, (claim.claim_id,))
                case.response_ir = response_ir
                case.close(CognitiveCaseStatus.ABSTAINED)
                return KernelResult(selection.question, response_ir, CognitiveCaseStatus.ABSTAINED)
            case.accepted_interpretation = selection
            claim, derivation = execute(selection)
            case.append_ledger(claim=claim, derivation=derivation)
            validate_ledger(tuple(case.evidence), tuple(case.derivations), tuple(case.claims))
            mode = ResponseMode.STRICT if claim.status.value == "computed" else ResponseMode.UNKNOWN
            response_ir = ResponseIR("response-ir/2", "response-1", case.case_id, "accepted-0", case.state_snapshot_id, mode, (claim.claim_id,))
            case.response_ir = response_ir
            status = CognitiveCaseStatus.SUCCESS if claim.status.value == "computed" else CognitiveCaseStatus.ABSTAINED
            rendered = render_strict(response_ir, tuple(case.claims)).text
            case.record("strict_rendered")
            case.close(status)
            return KernelResult(rendered, response_ir, status)
        except BaseException as exc:
            status = CognitiveCaseStatus.CANCELLED if type(exc).__name__ == "CancelledError" else CognitiveCaseStatus.FAILED
            case.close(status, type(exc).__name__)
            raise

    def _snapshot_id(self) -> str:
        candidate = getattr(self._state_authority, "version", None) or getattr(self._state_authority, "snapshot_id", None)
        return f"world-state:{candidate if candidate is not None else id(self._state_authority)}"
