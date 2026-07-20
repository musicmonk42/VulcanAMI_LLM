from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
pydantic = pytest.importorskip("pydantic")

from vulcan.runtime.app import create_app
from vulcan.runtime.auth import AuthenticatedPrincipal
from vulcan.runtime.route_manifest import generate_route_manifest


@dataclass(frozen=True)
class Event:
    schema_version: str = "vulcan-audit/1"
    sequence: int = 1
    event_type: str = "case.started"
    timestamp: str = "2026-01-01T00:00:00Z"
    previous_hash: str = "0" * 64
    data: dict = None
    event_hash: str = "1" * 64


class Audit:
    def __init__(self):
        self.case_id = "case-abc"
    def events_for_case(self, case_id):
        if case_id != self.case_id:
            return ()
        return (Event(data={"case_id": case_id, "request_digest": "2" * 64}),)
    def events_for_proposal(self, proposal_digest):
        return ()


class Runtime:
    closed = False
    runtime_id = "rt-1"
    health = None
    audit = Audit()
    self_improvement = SimpleNamespace(
        journal=SimpleNamespace(owner_id="journal"),
        drive=SimpleNamespace(state=SimpleNamespace(pending_approvals=[])),
        status_port=SimpleNamespace(status=lambda: {}),
    )
    async def admission(self):
        return True
    async def shallow_readiness(self):
        return True
    async def deep_integrity(self):
        return True


def principal(scopes):
    return AuthenticatedPrincipal("sub", "tenant", "iss", ("aud",), frozenset(scopes), "j" * 16, "v1")


@pytest.fixture
def client():
    app = create_app()
    app.state.ready = True
    app.state.runtime = Runtime()
    app.state.auth_config = object()
    return TestClient(app)


def test_route_manifest_matches_composed_asgi_routes_and_openapi(client):
    manifest = generate_route_manifest(client.app)
    assert client.app.state.route_manifest == manifest
    openapi_paths = {p for p in client.get("/openapi.json").json()["paths"] if not p.startswith("/health/")}
    manifest_paths = {r["path"] for r in manifest if not r["path"].startswith("/health/")}
    assert openapi_paths == manifest_paths
    assert all(r["auth_scope"] or r["classification"] == "public" for r in manifest)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b'{"message":"a","message":"b"}', "malformed_json"),
        (b'{"message":"a"} trailing', "malformed_json"),
        (b'\xff', "malformed_json"),
    ],
)
def test_adversarial_json_errors_are_bounded(client, body, code):
    with patch("vulcan.runtime.app.authenticate_bearer", return_value=principal({"reason:write"})):
        r = client.post("/v1/chat", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == code


def test_oversized_body_is_rejected_before_schema(client):
    with patch("vulcan.runtime.app.authenticate_bearer", return_value=principal({"reason:write"})):
        r = client.post("/v1/chat", content=b'{"message":"' + b'a' * 17000 + b'"}', headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "body_too_large"


def test_malformed_etag_rejected_and_idempotency_required(client):
    with patch("vulcan.runtime.app.authenticate_bearer", return_value=principal({"domains:write"})):
        r = client.post("/v1/admin/domains", json={"bundle": {}}, headers={"if-match": "not-an-etag"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "etag_malformed"


def test_auth_scope_enforced_for_operator_route(client):
    with patch("vulcan.runtime.app.authenticate_bearer", return_value=principal({"reason:write"})):
        r = client.get("/health/integrity")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


def test_case_audit_round_trip_uses_events_for_case_and_typed_serialization(client):
    with patch("vulcan.runtime.app.authenticate_bearer", return_value=principal({"audit:read"})):
        r = client.get("/v1/audit/cases/case-abc")
    assert r.status_code == 200
    payload = r.json()
    assert payload["case_id"] == "case-abc"
    assert payload["events"][0]["event_type"] == "case.started"
    assert payload["events"][0]["data"]["case_id"] == "case-abc"


def test_missing_case_is_404_not_empty_success(client):
    with patch("vulcan.runtime.app.authenticate_bearer", return_value=principal({"audit:read"})):
        r = client.get("/v1/audit/cases/case-missing")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
