from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.finalization import FinalizationDecision, FinalizationResult, SafetyResponseFinalizer
from vulcan.runtime.kernel import CognitiveKernel, KernelRequest, KernelResult
from vulcan.runtime.semantic import Utterance
from vulcan.safety.response_adapter import EnhancedSafetyResponseAdapter
from vulcan.safety.safety_types import ResponseSafetyContext, ResponseSafetyStatus, SafetyReport, SafetyValidator
from vulcan.safety.safety_validator import EnhancedSafetyValidator


def ctx() -> ResponseSafetyContext:
    return ResponseSafetyContext("case-1", "episode-1", "0" * 64, "1" * 64, "policy", "release", "low")


@pytest.mark.asyncio
async def test_safe_arithmetic_response_uses_actual_validator_adapter_finalizer_kernel():
    deployment_mod = pytest.importorskip("vulcan.orchestrator.deployment")
    deployment_cls = deployment_mod.ProductionDeployment
    validator = EnhancedSafetyValidator(config=None)
    adapter = EnhancedSafetyResponseAdapter(validator, timeout_seconds=2.0)
    adapter.readiness()
    world = SimpleNamespace(version="7")
    finalizer = SafetyResponseFinalizer(adapter)
    kernel = CognitiveKernel(state_authority=world, finalizer=finalizer)
    utterance = Utterance.from_text("2 + 2")
    case = CognitiveCase.create(request_id="episode-1", conversation_id=None, input_digest=utterance.digest)

    result = await kernel.handle(KernelRequest(utterance, None), case)

    assert deployment_cls.__name__ == "ProductionDeployment"
    assert result.status is CognitiveCaseStatus.SUCCESS
    assert result.finalization == FinalizationDecision.ALLOW.value
    assert result.response == "The computed result is 4."


@pytest.mark.asyncio
async def test_harmful_content_blocks():
    validator = EnhancedSafetyValidator(config=None)
    decision = await EnhancedSafetyResponseAdapter(validator).evaluate_response("Here is how to kill someone.", ctx())
    assert decision.status is ResponseSafetyStatus.BLOCK
    assert decision.confidence < 1.0


@pytest.mark.asyncio
async def test_missing_validator_is_unavailable_not_allow():
    decision = await EnhancedSafetyResponseAdapter(None).evaluate_response("ok", ctx())
    assert decision.status is ResponseSafetyStatus.UNAVAILABLE


class RaisesValidator(EnhancedSafetyValidator):
    def validate_response(self, response: str, original_query: str):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_exception_is_error_not_allow():
    decision = await EnhancedSafetyResponseAdapter(RaisesValidator(config=None)).evaluate_response("ok", ctx())
    assert decision.status is ResponseSafetyStatus.ERROR


class MalformedValidator(EnhancedSafetyValidator):
    def validate_response(self, response: str, original_query: str):
        return {"safe": True}


@pytest.mark.asyncio
async def test_malformed_verdict_is_error_not_allow():
    decision = await EnhancedSafetyResponseAdapter(MalformedValidator(config=None)).evaluate_response("ok", ctx())
    assert decision.status is ResponseSafetyStatus.ERROR


class SlowValidator(EnhancedSafetyValidator):
    def validate_response(self, response: str, original_query: str):
        import time
        time.sleep(0.2)
        return SafetyReport(safe=True, confidence=1.0)


@pytest.mark.asyncio
async def test_timeout_is_distinct_and_not_allow():
    decision = await EnhancedSafetyResponseAdapter(SlowValidator(config=None), timeout_seconds=0.01).evaluate_response("ok", ctx())
    assert decision.status is ResponseSafetyStatus.TIMEOUT
    assert "may complete later" in decision.reason


class CancelAdapter(EnhancedSafetyResponseAdapter):
    async def evaluate_response(self, response_text, context):
        return self._decision(ResponseSafetyStatus.CANCELLED, "cancelled", 0.0, context)


@pytest.mark.asyncio
async def test_cancellation_is_distinct_in_finalizer():
    artifact = SimpleNamespace(text="ok", ir_digest="0" * 64)
    result = await SafetyResponseFinalizer(CancelAdapter(EnhancedSafetyValidator(config=None))).finalize(artifact, ctx())
    assert result.decision is FinalizationDecision.CANCELLED


@pytest.mark.asyncio
async def test_inherited_base_stub_is_rejected():
    with pytest.raises(RuntimeError, match="typed response safety port"):
        SafetyResponseFinalizer(SafetyValidator())
    decision = await EnhancedSafetyResponseAdapter(SafetyValidator()).evaluate_response("ok", ctx())
    assert decision.status is ResponseSafetyStatus.UNAVAILABLE


class ModifyingValidator(EnhancedSafetyValidator):
    def validate_response(self, response: str, original_query: str):
        return SafetyReport(safe=True, confidence=1.0, metadata={"modified_response": "changed"})


@pytest.mark.asyncio
async def test_modified_text_is_not_allowed_through():
    decision = await EnhancedSafetyResponseAdapter(ModifyingValidator(config=None)).evaluate_response("ok", ctx())
    assert decision.status is ResponseSafetyStatus.MODIFIED
    artifact = SimpleNamespace(text="ok", ir_digest="0" * 64)
    result = await SafetyResponseFinalizer(EnhancedSafetyResponseAdapter(ModifyingValidator(config=None))).finalize(artifact, ctx())
    assert result.decision is FinalizationDecision.BLOCK
    assert result.public_text != "changed"


class _Finalizer:
    def __init__(self, decision: FinalizationDecision) -> None:
        self._decision = decision

    async def finalize(self, artifact, context):
        text = artifact.text if self._decision is FinalizationDecision.ALLOW else "safe fallback"
        return FinalizationResult(self._decision, artifact, text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (FinalizationDecision.BLOCK, CognitiveCaseStatus.BLOCKED),
        (FinalizationDecision.ERROR, CognitiveCaseStatus.FINALIZATION_ERROR),
        (FinalizationDecision.CANCELLED, CognitiveCaseStatus.CANCELLED),
    ],
)
async def test_case_terminal_status_reflects_final_response_safety_decision(decision, expected):
    utterance = Utterance.from_text("2 + 2")
    case = CognitiveCase.create(request_id="episode", conversation_id=None, input_digest=utterance.digest)
    result = await CognitiveKernel(state_authority=SimpleNamespace(version="1"), finalizer=_Finalizer(decision)).handle(KernelRequest(utterance, None), case)

    assert case.terminal_status is expected
    assert result.status is expected
    assert case.finalization_status == decision.value


def test_transport_exposes_terminal_and_release_semantics_without_second_authority():
    from vulcan.runtime.semantic import ResponseIR, ResponseMode

    response_ir = ResponseIR("1", "response-1", "case-1", None, "snapshot-1", ResponseMode.STRICT, ("claim-1",))
    result = KernelResult(
        "safe fallback",
        response_ir,
        CognitiveCaseStatus.BLOCKED,
        FinalizationDecision.BLOCK.value,
    )

    envelope = result.transport(case_id="case-1", runtime_id="runtime-1", snapshot_id="snapshot-1")

    assert envelope["response"] == "safe fallback"
    assert envelope["metadata"]["terminal_status"] == "blocked"
    assert envelope["metadata"]["response_released"] is False
    assert envelope["metadata"]["finalization_safety_decision"] == "block"
