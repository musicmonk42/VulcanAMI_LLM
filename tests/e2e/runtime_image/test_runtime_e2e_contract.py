from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_runtime_workflow_is_required_named_check_and_builds_exact_image_without_push() -> None:
    workflow = read(".github/workflows/runtime-e2e.yml")
    assert "name: runtime-e2e" in workflow
    assert "exact built image runtime qualification" in workflow
    assert "scripts/e2e/run_runtime_qualification.sh" in workflow
    assert "PYTHON_BIN='python -O'" in workflow
    assert "docker push" not in workflow


def test_qualification_harness_fails_closed_on_startup_degraded_fallback_and_block_success() -> None:
    script = read("scripts/e2e/run_runtime_qualification.sh")
    assert "REQUIRE_HASHES=1" in script
    assert "docker volume create" in script
    assert "/health/ready" in script
    assert '"fallback" not in json.dumps(d).lower()' in script
    assert 'd.get("status") not in {"fallback","degraded"}' in script
    assert '[ "$code" != 200 ]' in script
    assert '[ "$code" = 401 ]' in script
    assert "audit-after-restart" in script


def test_mint_test_jwt_uses_scoped_hs256_with_canonical_claims() -> None:
    script = read("scripts/e2e/mint_test_jwt.py")
    for token in ['"alg": "HS256"', '"typ": "JWT"', '"iss"', '"aud"', '"tenant"', '"scope"', '"jti"']:
        assert token in script
    assert "sort_keys=True" in script
