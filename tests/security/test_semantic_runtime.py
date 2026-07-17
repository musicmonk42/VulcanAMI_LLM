import pytest
from types import SimpleNamespace
from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.kernel import CognitiveKernel, KernelRequest
from vulcan.runtime.semantic import (DeterministicLanguageInput, InterpretationProposal,
    ProposedCandidate, SourceSpan, Utterance, validate_proposal)
from vulcan.runtime.finalization import FinalizationDecision, FinalizationResult

class _Finalizer:
    async def finalize(self, artifact):
        return FinalizationResult(FinalizationDecision.ALLOW, artifact, artifact.text)

@pytest.mark.asyncio
async def test_kernel_computes_only_through_typed_ingress_and_records_claims():
    utterance = Utterance.from_text("2 + 3 * 4")
    case = CognitiveCase.create(request_id="r", conversation_id=None, input_digest=utterance.digest)
    result = await CognitiveKernel(state_authority=SimpleNamespace(version="1"), finalizer=_Finalizer()).handle(KernelRequest(utterance, None), case)
    assert result.response == "The computed result is 14."
    assert case.claims[0].derivation_id == case.derivations[0].derivation_id
    assert case.terminal_status is CognitiveCaseStatus.SUCCESS
    assert utterance.text not in repr(case)

def test_provider_fields_and_bad_unicode_spans_fail_closed():
    utterance = Utterance.from_text("é + 1")
    proposal = InterpretationProposal("semantic-ingress/2", (ProposedCandidate("arithmetic", "é + 1", SourceSpan(0, 99)),), "provider")
    with pytest.raises(ValueError): validate_proposal(utterance, proposal)

@pytest.mark.asyncio
async def test_unsupported_input_is_an_unknown_not_a_provider_answer():
    utterance = Utterance.from_text("tell me a secret")
    case = CognitiveCase.create(request_id="r", conversation_id=None, input_digest=utterance.digest)
    result = await CognitiveKernel(state_authority=object(), finalizer=_Finalizer(), language_input=DeterministicLanguageInput()).handle(KernelRequest(utterance, None), case)
    assert result.status is CognitiveCaseStatus.ABSTAINED
    assert "not supported" in result.response
