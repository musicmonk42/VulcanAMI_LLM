"""Constitutional identity/credential separation for the Phase-A request path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vulcan.microkernel.episode import (
    ACTOR_BINDING_DIGEST_ALGORITHM,
    ACTOR_BINDING_SCHEMA,
    ActorBinding,
    canonical_digest,
)
from vulcan.runtime.api import VerifiedAuthenticationContext
from vulcan.runtime.auth import AuthenticatedPrincipal, CredentialProvenance

NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


def principal(**changes: object) -> AuthenticatedPrincipal:
    values = {
        "subject": "alice",
        "tenant": "tenant-a",
        "issuer": "https://issuer.example",
        "audience": ("vulcan",),
        "scopes": frozenset({"reason:write"}),
        "jti": "token_0123456789abcdef",
        "key_version": "key-1",
        "authenticated_at": NOW,
    }
    values.update(changes)
    return AuthenticatedPrincipal._from_verified_adapter(**values)  # type: ignore[arg-type]


def test_golden_schema_and_canonical_identity_vector() -> None:
    raw = Path("config/actor-binding-schema.json").read_bytes()
    document = json.loads(raw)
    emitted = Path("config/actor-binding-schema.sha256").read_text().split()[0]
    assert emitted == hashlib.sha256(raw).hexdigest()
    vector = document["golden_vectors"][0]
    assert document["schema"] == ACTOR_BINDING_SCHEMA
    assert document["digest_algorithm"] == ACTOR_BINDING_DIGEST_ALGORITHM
    assert canonical_digest(vector["identity"]) == vector["sha256"]
    assert (
        json.dumps(vector["identity"], sort_keys=True, separators=(",", ":"))
        == vector["canonical_utf8"]
    )


def test_actor_is_stable_across_credentials_and_separates_security_domains() -> None:
    first = principal()
    rotated = principal(
        key_version="key-2",
        jti="token_abcdef0123456789",
        scopes=frozenset({"audit:read"}),
        authenticated_at=NOW + timedelta(minutes=1),
        adapter_release="vulcan.hs256-auth/2",
    )
    assert first.actor == rotated.actor
    assert first.actor != principal(tenant="tenant-b").actor
    assert first.actor != principal(issuer="https://other.example").actor
    assert first.actor.classification == "AUTHENTICATED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key_id", "key-2"),
        ("token_id", "token_abcdef0123456789"),
        ("scopes", frozenset({"audit:read"})),
        ("authenticated_at", NOW + timedelta(seconds=1)),
        ("method", "mTLS"),
        ("adapter_release", "vulcan.hs256-auth/2"),
    ],
)
def test_every_credential_field_is_distinct_from_actor(
    field: str, value: object
) -> None:
    authenticated = principal()
    provenance = CredentialProvenance.from_principal(authenticated)
    assert replace(provenance, **{field: value}) != provenance
    assert authenticated.actor == principal().actor


def test_actor_is_immutable_and_legacy_never_upgrades() -> None:
    actor = principal().actor
    with pytest.raises(FrozenInstanceError):
        actor.tenant = "other"  # type: ignore[misc]
    legacy = ActorBinding("legacy", "a" * 64, "LegacySource")
    assert legacy.classification == "LEGACY_UNVERIFIED"
    assert set(legacy.to_json()) == {"actor_id", "authority", "principal_digest"}
    with pytest.raises(ValueError, match="cannot claim"):
        ActorBinding(
            "legacy",
            "a" * 64,
            "LegacySource",
            tenant="invented-tenant",
        )
    with pytest.raises(ValueError, match="identifier mismatch"):
        replace(actor, actor_id="actor:" + "b" * 64)


def test_context_requires_verified_principal_and_hides_credential_material() -> None:
    with pytest.raises(TypeError, match="adapter-created"):
        VerifiedAuthenticationContext(
            principal().actor, None, frozenset({"reason:write"})
        )
    with pytest.raises(TypeError, match="only by an adapter"):
        AuthenticatedPrincipal(
            "alice",
            "tenant-a",
            "https://issuer.example",
            ("vulcan",),
            frozenset({"reason:write"}),
            "token_0123456789abcdef",
            "key-1",
            NOW,
        )
    with pytest.raises(TypeError, match="verified authentication"):
        VerifiedAuthenticationContext.from_verified_principal(object())  # type: ignore[arg-type]
    authenticated = principal()
    context = VerifiedAuthenticationContext.from_verified_principal(authenticated)
    assert "token_0123456789abcdef" not in repr(context)
    assert "key-1" not in repr(context)
    assert "token_0123456789abcdef" not in repr(authenticated)


def test_only_authentication_adapter_constructs_authenticated_binding() -> None:
    references = []
    for path in Path("src").rglob("*.py"):
        if "_from_verified_identity" in path.read_text(encoding="utf-8"):
            references.append(path.as_posix())
    assert sorted(references) == [
        "src/vulcan/microkernel/episode.py",
        "src/vulcan/runtime/auth.py",
    ]
