"""Constitutional tests for the capability-minimized Phase-A facade."""

from __future__ import annotations

import inspect
import json
import sqlite3
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from vulcan.microkernel.episode import canonical_digest
from vulcan.runtime.api import (
    CommandEnvelope,
    CommandKind,
    ExecutionBudget,
    QueryEnvelope,
    QueryKind,
    RuntimeAPI,
    VerifiedAuthenticationContext,
    _validate_phase_a_settings,
    request_digest,
)
from vulcan.runtime.auth import AuthenticatedPrincipal, AuthorizationError
from vulcan.runtime.route_manifest import (
    PHASE_A_ROUTE_REGISTRY,
    generate_route_manifest,
)
from vulcan.runtime.settings import (
    MemoryBackend,
    OpaqueSecret,
    RuntimeSettings,
    SecretSource,
    VulcanEnvironment,
    durable_root_paths,
)


def authentication(
    *scopes: str, tenant: str = "tenant", issuer: str = "issuer"
) -> VerifiedAuthenticationContext:
    scopes = scopes or ("public:read",)
    principal = AuthenticatedPrincipal._from_verified_adapter(
        "subject",
        tenant,
        issuer,
        ("audience",),
        frozenset(scopes),
        "0123456789abcdef",
        "v1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return VerifiedAuthenticationContext.from_verified_principal(principal)


def command(**changes) -> CommandEnvelope:
    payload = {"message": "2 + 2", "conversation_id": None}
    values = {
        "kind": CommandKind.CHAT,
        "request_digest": request_digest(CommandKind.CHAT, payload),
        "authentication": authentication("reason:write"),
        "request_id": "request-transport-1",
        "idempotency_key": "request-1",
        "deadline": datetime.now(timezone.utc) + timedelta(seconds=5),
        "budget": ExecutionBudget(64, 4096),
        "payload": payload,
    }
    values.update(changes)
    return CommandEnvelope(**values)


def test_runtime_api_has_only_two_public_operations() -> None:
    public_operations = {
        name
        for name, member in inspect.getmembers(RuntimeAPI, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_operations == {"execute", "query"}


def test_asgi_adapter_has_no_retired_route_or_subsystem_imports() -> None:
    source = Path("src/vulcan/runtime/app.py").read_text(encoding="utf-8")
    for retired_route in ("/v1/admin/", "/v1/memory/", "/v1/audit/improvements"):
        assert retired_route not in source
    for forbidden_import in (
        "vulcan.memory",
        "vulcan.learning",
        "vulcan.improvement",
        "vulcan.research",
        "vulcan.npt",
    ):
        assert forbidden_import not in source
    assert "openapi_url=None" in source
    assert "docs_url=None" in source
    assert "redoc_url=None" in source


def test_envelopes_are_frozen_and_deeply_reject_mutable_payloads() -> None:
    envelope = command()
    with pytest.raises(FrozenInstanceError):
        envelope.idempotency_key = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        envelope.payload["message"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable primitives"):
        command(payload={"message": ["2 + 2"]})
    with pytest.raises(TypeError, match="immutable primitives"):
        command(payload={"callback": lambda: None})


def test_envelope_requires_verified_context_digest_key_deadline_and_budget() -> None:
    with pytest.raises(TypeError, match="verified authentication"):
        command(authentication=object())
    with pytest.raises(ValueError, match="request digest"):
        command(request_digest="not-a-digest")
    with pytest.raises(ValueError, match="does not bind"):
        command(request_digest="a" * 64)
    with pytest.raises(ValueError, match="idempotency"):
        command(idempotency_key="contains spaces")
    with pytest.raises(ValueError, match="UTC deadline"):
        command(deadline=datetime.now())
    with pytest.raises(ValueError, match="execution budget"):
        ExecutionBudget(0, 1)


def test_manifest_requires_composed_app_or_typed_registry() -> None:
    with pytest.raises(RuntimeError, match="application or typed route registry"):
        generate_route_manifest()
    manifest = generate_route_manifest(registry=PHASE_A_ROUTE_REGISTRY)
    assert {(row["method"], row["path"]) for row in manifest} == set(
        PHASE_A_ROUTE_REGISTRY
    )
    assert len(manifest) == 6


def test_query_envelope_has_same_mandatory_boundary() -> None:
    envelope = QueryEnvelope(
        QueryKind.READINESS,
        request_digest(QueryKind.READINESS, {}),
        authentication(),
        "query-transport-1",
        "query-1",
        datetime.now(timezone.utc) + timedelta(seconds=5),
        ExecutionBudget(1, 1024),
        {},
    )
    assert envelope.kind is QueryKind.READINESS


@pytest.mark.parametrize(
    "field",
    [
        "memory_enabled",
        "learning_enabled",
        "csiu_enabled",
        "self_improvement_enabled",
        "openai_enabled",
        "anthropic_enabled",
    ],
)
def test_phase_a_rejects_forbidden_owner_settings(field: str) -> None:
    settings = SimpleNamespace(
        memory_enabled=False,
        learning_enabled=False,
        csiu_enabled=False,
        self_improvement_enabled=False,
        openai_enabled=False,
        anthropic_enabled=False,
        language_mode=SimpleNamespace(value="deterministic_only"),
    )
    setattr(settings, field, True)
    with pytest.raises(ValueError, match="rejects configured owners"):
        _validate_phase_a_settings(settings)


def test_phase_a_rejects_transformer_mode() -> None:
    settings = SimpleNamespace(
        memory_enabled=False,
        learning_enabled=False,
        csiu_enabled=False,
        self_improvement_enabled=False,
        openai_enabled=False,
        anthropic_enabled=False,
        language_mode=SimpleNamespace(value="transformer_proposal"),
    )
    with pytest.raises(ValueError, match="transformer"):
        _validate_phase_a_settings(settings)


@pytest.mark.asyncio
async def test_reduced_graph_executes_arithmetic_without_forbidden_packages(
    tmp_path,
) -> None:
    imported_before = frozenset(sys.modules)
    from vulcan.runtime.phase_a_composition import compose_phase_a_runtime

    root = tmp_path / "phase-a"
    root.mkdir(mode=0o700)
    settings = RuntimeSettings(
        VulcanEnvironment.development,
        "vulcan",
        "vulcan-runtime",
        OpaqueSecret(
            SecretSource.direct,
            "Phase-A-Test-Secret-0123456789!abcdef",
            "VULCAN_JWT_SECRET",
        ),
        root,
        durable_root_paths(root),
        memory_enabled=False,
        memory_backend=MemoryBackend.disabled,
        memory_sqlite_path=None,
        csiu_enabled=False,
        learning_enabled=False,
    )
    runtime = compose_phase_a_runtime(settings)
    api = RuntimeAPI(runtime)
    try:
        result = await api.execute(command())
        assert result["response"] == "The computed result is 4."
        assert result["status"] == "success"
        episode = runtime.episode_store.load(result["metadata"]["case_id"])
        lineage = runtime.lineage_store.load("branch-primary")
        assert episode.state.is_terminal
        assert lineage.head_digest == episode.digest
        assert lineage.active_episode_ids == ()
        assert lineage.past_episode_ids == (episode.episode_id,)
        assert episode.request.request_id == "request-transport-1"
        assert episode.actor == command().authentication._actor_binding()
        assert episode.actor.classification == "AUTHENTICATED"
        assert all(
            transition.authority == episode.actor.principal_digest
            for transition in episode.transitions
        )
        serialized = episode.canonical_json()
        assert "0123456789abcdef" not in serialized
        assert "key_version" not in serialized
        journal_path = root / "constitutional" / "constitutional.sqlite3"
        assert journal_path.is_file()
        assert not (root / "episodes" / "episodes.sqlite3").exists()
        assert not (root / "epistemic" / "epistemic.sqlite3").exists()
        connection = sqlite3.connect(journal_path)
        try:
            terminal = connection.execute(
                "SELECT t.commit_seq,m.status FROM terminal_results t "
                "JOIN lineage_membership m ON m.episode_id=t.episode_id "
                "WHERE t.episode_id=?",
                (episode.episode_id,),
            ).fetchone()
            assert terminal is not None and terminal[1] == "past"
            admission = connection.execute(
                "SELECT c.commit_seq,m.admitted_commit_seq FROM commands c "
                "JOIN episodes e ON e.command_id=c.command_id "
                "JOIN lineage_membership m ON m.episode_id=e.episode_id "
                "WHERE e.episode_id=?",
                (episode.episode_id,),
            ).fetchone()
            assert admission is not None and admission[0] == admission[1]
            assert (
                connection.execute(
                    "SELECT count(*) FROM transactional_outbox WHERE commit_seq=?",
                    (admission[0],),
                ).fetchone()[0]
                >= 1
            )
            epistemic = connection.execute(
                "SELECT ec.commit_seq,et.commit_seq FROM epistemic_commits ec "
                "JOIN episode_transitions et ON et.episode_id=ec.episode_id "
                "AND et.to_state='epistemically_committed' "
                "WHERE ec.episode_id=?",
                (episode.episode_id,),
            ).fetchone()
            assert epistemic is not None and epistemic[0] == epistemic[1]
            terminal_transitions = connection.execute(
                "SELECT count(*) FROM episode_transitions "
                "WHERE episode_id=? AND commit_seq=?",
                (episode.episode_id, terminal[0]),
            ).fetchone()[0]
            assert terminal_transitions == 3
            assert (
                connection.execute(
                    "SELECT count(*) FROM transactional_outbox WHERE commit_seq=?",
                    (terminal[0],),
                ).fetchone()[0]
                == 6
            )
            receipts = connection.execute(
                "SELECT a.content FROM artifacts a "
                "WHERE a.kind='transition-receipt.v1' ORDER BY a.artifact_digest"
            ).fetchall()
            assert len(receipts) == len(episode.transitions)
            for row in receipts:
                receipt = json.loads(row[0])
                assert receipt["actor_digest"] == canonical_digest(
                    episode.actor.to_json()
                )
                assert receipt["episode_id"] == episode.episode_id
                assert receipt["issued_at_epoch"] <= receipt["expires_at_epoch"]
                assert receipt["schema_version"] == "vulcan-transition-receipt/1"
            warrants = connection.execute(
                "SELECT a.content FROM artifacts a "
                "WHERE a.kind='epistemic-warrant-receipt.v1'"
            ).fetchall()
            assert len(warrants) == 1
            warrant = json.loads(warrants[0][0])
            assert warrant["status"] == "COMPUTED"
            assert warrant["episode"] == episode.episode_id
            assert warrant["candidate"].startswith("sha256:")
        finally:
            connection.close()
        audit_payload = {"episode_id": episode.episode_id}
        cross_tenant = QueryEnvelope(
            QueryKind.EPISODE_AUDIT,
            request_digest(QueryKind.EPISODE_AUDIT, audit_payload),
            authentication("audit:read", tenant="other-tenant"),
            "cross-tenant-read",
            "audit-cross-tenant",
            datetime.now(timezone.utc) + timedelta(seconds=5),
            ExecutionBudget(1, 4096),
            audit_payload,
        )
        with pytest.raises(AuthorizationError, match="cross-tenant"):
            await api.query(cross_tenant)
        forbidden = (
            "vulcan.memory",
            "vulcan.learning",
            "vulcan.improvement",
            "vulcan.research",
            "vulcan.npt",
        )
        newly_imported = set(sys.modules).difference(imported_before)
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in newly_imported
            for prefix in forbidden
        )
        with pytest.raises(RuntimeError, match="output budget"):
            await api.execute(command(budget=ExecutionBudget(64, 1)))
        await runtime.deep_integrity()
    finally:
        await runtime.close()
    restarted = compose_phase_a_runtime(settings)
    try:
        await restarted.deep_integrity()
        replayed = await RuntimeAPI(restarted).execute(command())
        assert replayed["response"] == "The computed result is 4."
        assert replayed["metadata"]["case_id"] == result["metadata"]["case_id"]
        connection = sqlite3.connect(root / "constitutional" / "constitutional.sqlite3")
        try:
            assert (
                connection.execute("SELECT count(*) FROM commands").fetchone()[0] == 1
            )
        finally:
            connection.close()
    finally:
        await restarted.close()


def test_reduced_composition_closes_durable_owners_after_startup_failure(
    tmp_path, monkeypatch
) -> None:
    from vulcan.microkernel.constitutional_journal import ConstitutionalDatabase
    from vulcan.runtime import phase_a_composition

    root = tmp_path / "failed-start"
    root.mkdir(mode=0o700)
    settings = RuntimeSettings(
        VulcanEnvironment.development,
        "vulcan",
        "vulcan-runtime",
        OpaqueSecret(
            SecretSource.direct,
            "Phase-A-Test-Secret-0123456789!abcdef",
            "VULCAN_JWT_SECRET",
        ),
        root,
        durable_root_paths(root),
        memory_enabled=False,
        memory_backend=MemoryBackend.disabled,
        memory_sqlite_path=None,
        csiu_enabled=False,
        learning_enabled=False,
    )
    monkeypatch.setattr(
        phase_a_composition,
        "load_capability_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("capability failure")),
    )
    with pytest.raises(RuntimeError, match="capability failure"):
        phase_a_composition.compose_phase_a_runtime(settings)

    reopened = ConstitutionalDatabase(
        root / "constitutional" / "constitutional.sqlite3"
    )
    reopened.close()
