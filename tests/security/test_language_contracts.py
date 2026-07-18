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

import json
from vulcan.local_language import SpanProposalError, VerifiedAdapterMetadata, VerifiedLocalSpanCompletion, parse_transformer_span_proposal


def _proposal(operation="arithmetic", s=0, e=5, conf=0.9):
    obj={"schema_version":"transformer-span-proposal/1","candidates":[{"operation":operation,"span":{"start":s,"end":e},"argument_spans":{"expression":{"start":s,"end":e}},"confidence":conf}]}
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"))


def test_valid_exact_span_proposal_reconstructs_operand_server_side():
    u=Utterance.from_text("2 + 2")
    p=parse_transformer_span_proposal(_proposal(), u)
    assert p.candidates[0].expression == "2 + 2"


@pytest.mark.parametrize("raw", [
    '{"schema_version":"transformer-span-proposal/1","schema_version":"transformer-span-proposal/1","candidates":[]}',
    '{"candidates":[],"schema_version":"transformer-span-proposal/1"} prose',
    '{"candidates":[{"argument_spans":{"expression":{"end":5,"start":0}},"confidence":NaN,"operation":"arithmetic","span":{"end":5,"start":0}}],"schema_version":"transformer-span-proposal/1"}',
])
def test_strict_transformer_json_rejects_duplicates_trailing_and_nan(raw):
    with pytest.raises(SpanProposalError): parse_transformer_span_proposal(raw, Utterance.from_text("2 + 2"))


@pytest.mark.parametrize("field", ["answer","domain_hint","graphix_plan","code","evidence"])
def test_transformer_forbidden_fields_rejected(field):
    obj=json.loads(_proposal())
    obj["candidates"][0][field]="bad"
    raw=json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"))
    with pytest.raises(SpanProposalError): parse_transformer_span_proposal(raw, Utterance.from_text("2 + 2"))


@pytest.mark.parametrize("span", [{"start":5,"end":1},{"start":0,"end":99},{"start":0,"end":0}])
def test_transformer_invalid_spans_rejected(span):
    obj=json.loads(_proposal()); obj["candidates"][0]["argument_spans"]["expression"]=span
    raw=json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"))
    with pytest.raises(SpanProposalError): parse_transformer_span_proposal(raw, Utterance.from_text("2 + 2"))


class _Provider:
    def __init__(self, raw=None, fail=False): self.raw=raw; self.fail=fail; self.closed=False
    def generate(self, prompt, *, max_tokens):
        if self.fail: raise RuntimeError("boom")
        return self.raw
    def close(self): self.closed=True
class _Tok:
    def encode(self, text): return list(text)

@pytest.mark.asyncio
async def test_verified_adapter_close_and_context_overflow():
    adapter=VerifiedLocalSpanCompletion(provider=_Provider(_proposal()), tokenizer=_Tok(), metadata=VerifiedAdapterMetadata("0"*64), context_length=10, max_generated_tokens=10)
    with pytest.raises(SpanProposalError): await adapter.propose(Utterance.from_text("2 + 2"))
    adapter.close()
    with pytest.raises(RuntimeError): await adapter.propose(Utterance.from_text("2 + 2"))

@pytest.mark.asyncio
async def test_malformed_proposal_falls_back_deterministically():
    utterance=Utterance.from_text("2 + 2")
    case=CognitiveCase.create(request_id="r2", conversation_id=None, input_digest=utterance.digest)
    adapter=VerifiedLocalSpanCompletion(provider=_Provider('{"bad":true}'), tokenizer=_Tok(), metadata=VerifiedAdapterMetadata("0"*64), context_length=10000, max_generated_tokens=10)
    result=await CognitiveKernel(state_authority=object(), finalizer=_Finalizer(), language_input=adapter).handle(KernelRequest(utterance,None),case)
    assert result.response == "The computed result is 4."
    assert [event.stage for event in case.events].count("input_proposal_unavailable") == 1
