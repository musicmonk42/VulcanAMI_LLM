import os
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "a" * 40)
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("VULCAN_ENV", "production")
os.environ.setdefault("VULCAN_SAFETY_LEVEL", "strict")

from fastapi.testclient import TestClient

import src.full_platform as fp


def test_route_manifest_has_explicit_disposition():
    manifest = fp.generate_route_manifest()
    assert manifest
    for route in manifest:
        assert route["classification"] in {"public", "protected"}
        assert route["authorization"]


def test_protected_routes_reject_anonymous():
    client = TestClient(fp.app)
    protected = [r for r in fp.generate_route_manifest() if r["authentication_required"]]
    assert protected
    for route in protected[:25]:
        response = client.request(route["method"], route["path"])
        assert response.status_code in {401, 405}


def test_mutation_routes_require_authorization():
    token = fp.JWTAuth.create_access_token({"sub": "user", "roles": ["viewer"]})
    client = TestClient(fp.app)
    response = client.post("/api/arena/feedback", headers={"Authorization": f"Bearer {token}"}, json={})
    assert response.status_code in {403, 405}


def test_unsafe_production_config_rejected():
    with patch.dict(os.environ, {"VULCAN_ENV": "production", "VULCAN_SAFETY_LEVEL": "minimal"}):
        with patch.object(fp.settings, "auth_method", fp.AuthMethod.JWT):
            try:
                fp.validate_production_invariants()
            except RuntimeError as exc:
                assert "safety" in str(exc).lower()
            else:
                raise AssertionError("unsafe production config was accepted")


def test_readiness_not_ready_before_mandatory_startup_complete():
    client = TestClient(fp.app)
    with patch.object(fp, "_services_init_complete", False), patch.object(fp, "_services_init_failed", False):
        response = client.get("/health/ready")
    assert response.status_code == 503
