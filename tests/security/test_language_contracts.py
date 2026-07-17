"""Canonical language-port identity and deterministic fallback gates."""
import asyncio
from types import SimpleNamespace

import pytest

from vulcan.runtime.case import CognitiveCase
from vulcan.runtime.kernel import CognitiveKernel, KernelRequest
from vulcan.runtime.output import LanguageOutputPort, ResponseIRProjection, UntrustedRenderDraft
from vulcan.runtime.semantic import LanguageInputPort, Utterance


class _Finalizer:
    async def finalize(self, artifact):
        return SimpleNamespace(decision=SimpleNamespace(value="allow"), public_text=artifact.text)


class _FailingOutput:
    async def render(self, _projection: ResponseIRProjection) -> UntrustedRenderDraft:
        raise RuntimeError("unavailable")

    def close(self) -> None:
        pass


def test_language_port_contracts_have_one_canonical_module_identity():
    assert LanguageInputPort.__module__ == "vulcan.runtime.semantic"
    assert LanguageOutputPort.__module__ == "vulcan.runtime.output"
    assert not hasattr(__import__("vulcan.runtime.semantic", fromlist=["x"]), "LanguageOutputPort")


@pytest.mark.asyncio
async def test_failed_output_adapter_falls_back_to_strict_renderer():
    utterance = Utterance.from_text("2 + 2")
    case = CognitiveCase.create(request_id="r", conversation_id=None, input_digest=utterance.digest)
    result = await CognitiveKernel(state_authority=object(), finalizer=_Finalizer(), language_output=_FailingOutput()).handle(KernelRequest(utterance, None), case)
    assert result.response == "The computed result is 4."
    assert [event.stage for event in case.events].count("output_draft_unavailable") == 1
    assert case.render_artifact.renderer == "strict-template"
