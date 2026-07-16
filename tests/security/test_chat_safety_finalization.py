import pytest
from types import SimpleNamespace

from src.vulcan.endpoints import unified_chat as chat


class Validator:
    def __init__(self, result=True, raises=False):
        self.calls = []
        self.result = result
        self.raises = raises

    def validate_action(self, payload):
        self.calls.append(payload)
        if self.raises:
            raise RuntimeError("boom")
        return self.result


@pytest.mark.asyncio
async def test_mandatory_safety_uses_safety_validator_name():
    validator = Validator((True, "ok"))
    deps = SimpleNamespace(safety_validator=validator, safety=None)
    decision = await chat._run_mandatory_safety(deps, {"type": "user_query", "content": "hello"})
    assert decision["decision"] is chat.SafetyDecision.ALLOW
    assert validator.calls


@pytest.mark.asyncio
async def test_missing_or_throwing_safety_fails_closed():
    missing = await chat._run_mandatory_safety(SimpleNamespace(), {"type": "response", "content": "x"})
    assert missing["decision"] is chat.SafetyDecision.ERROR
    throwing = await chat._run_mandatory_safety(SimpleNamespace(safety_validator=Validator(raises=True)), {"type": "response", "content": "x"})
    assert throwing["decision"] is chat.SafetyDecision.ERROR


@pytest.mark.asyncio
async def test_finalization_filters_blocked_output_exactly_once():
    deps = SimpleNamespace(safety_validator=Validator((False, "blocked")))
    metadata = {}
    result = await chat._finalize_chat_response(deps, {"response": "unsafe"}, metadata)
    assert result["response"] != "unsafe"
    assert result["metadata"]["finalized"] is True
    with pytest.raises(RuntimeError):
        await chat._finalize_chat_response(deps, {"response": "again"}, metadata)
