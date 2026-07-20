from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from vulcan.graphix.codec import dumps_envelope, extension_digest, loads_envelope, verify_envelope_digest
from vulcan.graphix.core import DigestMismatchError, ExtensionCollisionError, ForbiddenExecutableSemanticsError, GraphixEnvelope, PrincipalRelease, SourceKind, SourceReference, UnknownFieldError
from vulcan.graphix.registry import DialectRegistration, DialectRegistry

PAYLOAD = b'{"claim":"bounded"}'
D = "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()
ZERO = "sha256:" + "0" * 64

def envelope() -> GraphixEnvelope:
    ext_value = {"label": "display only", "score": 1}
    return GraphixEnvelope(
        dialect="core.test",
        schema_version=1,
        node_artifact_id="node:abc123",
        episode_id="episode:abc123",
        content_digest=D,
        proposer=PrincipalRelease("principal:governor", "release-2026.07.20"),
        authority_level="UNTRUSTED_PROPOSAL",
        source_references=(SourceReference(SourceKind.ARTIFACT, "artifact:abc123", D),),
        snapshot_bundle_digest=ZERO,
        epistemic_status="PROPOSED",
        privacy_class="INTERNAL",
        purpose="contract test",
        consent_references=("consent:abc123",),
        valid_from=datetime(2026, 7, 20, tzinfo=timezone.utc),
        valid_until=None,
        extensions=(),
    )

def test_canonical_round_trip_and_digest_verification():
    e = envelope()
    encoded = dumps_envelope(e)
    assert encoded == dumps_envelope(loads_envelope(encoded))
    verify_envelope_digest(e, PAYLOAD)
    assert b'"node_artifact_id"' in encoded
    assert b'"content_digest"' in encoded

def test_duplicate_keys_and_unknown_fields_fail_closed():
    e = json.loads(dumps_envelope(envelope()))
    e["unexpected"] = True
    with pytest.raises(UnknownFieldError):
        loads_envelope(json.dumps(e))
    dup = b'{"dialect":"core.test","dialect":"evil"}'
    with pytest.raises(UnknownFieldError):
        loads_envelope(dup)

def test_digest_tamper_fails_closed():
    with pytest.raises(DigestMismatchError):
        verify_envelope_digest(envelope(), b"tampered")
    e = json.loads(dumps_envelope(envelope()))
    e["content_digest"] = "sha256:" + "f" * 64
    loaded = loads_envelope(json.dumps(e))
    with pytest.raises(DigestMismatchError):
        verify_envelope_digest(loaded, PAYLOAD)

def test_malicious_payload_cannot_smuggle_executable_semantics():
    e = json.loads(dumps_envelope(envelope()))
    e["extensions"] = [{"namespace":"org.example.display","schema_version":1,"digest":ZERO,"value":{"shell_command":"rm -rf /"}}]
    with pytest.raises(ForbiddenExecutableSemanticsError):
        loads_envelope(json.dumps(e))

def test_extension_collision_and_bounds():
    with pytest.raises(ExtensionCollisionError):
        from vulcan.graphix.core import ExtensionDeclaration
        ExtensionDeclaration("org.example.authority", 1, ZERO, {"label":"x"})
    e = json.loads(dumps_envelope(envelope()))
    e["extensions"] = [{"namespace":"org.example.display","schema_version":1,"digest":ZERO,"value":{"x": i}} for i in range(17)]
    with pytest.raises(ValueError):
        loads_envelope(json.dumps(e))

def test_unicode_and_number_bounds_fail_closed():
    e = json.loads(dumps_envelope(envelope()))
    e["purpose"] = "bad\u0001"
    with pytest.raises(ValueError):
        loads_envelope(json.dumps(e))
    e = json.loads(dumps_envelope(envelope()))
    e["extensions"] = [{"namespace":"org.example.display","schema_version":1,"digest":ZERO,"value":{"n":2**60}}]
    with pytest.raises(ValueError):
        loads_envelope(json.dumps(e))

def test_registry_unknown_version_startup_only_and_migration():
    reg = DialectRegistry(release_id="release-2026.07.20")
    with pytest.raises(ValueError):
        reg.require_supported(envelope())
    def migrate(e: GraphixEnvelope) -> GraphixEnvelope:
        return replace(e, schema_version=2)
    reg.register(DialectRegistration("core.test", 1, "release-2026.07.20", frozenset({2}), {2: migrate}))
    reg.freeze()
    assert reg.require_supported(envelope()).dialect == "core.test"
    assert reg.migrate_to(envelope(), 2).schema_version == 2
    with pytest.raises(ValueError):
        reg.register(DialectRegistration("core.other", 1, "release-2026.07.20"))

def test_extension_digest_is_canonical():
    assert extension_digest({"b": 2, "a": 1}) == extension_digest({"a": 1, "b": 2})


def test_schema_declares_closed_required_envelope_fields():
    schema = json.loads(Path("schemas/graphix/core-v1.json").read_text())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(json.loads(dumps_envelope(envelope())).keys())
    assert schema["properties"]["authority_level"]["enum"] == ["UNTRUSTED_PROPOSAL","VALIDATED_CANDIDATE","COMMITTED_BELIEF","AUTHORIZED_PLAN","EXECUTED_EFFECT"]
