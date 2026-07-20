from __future__ import annotations

from pathlib import Path

import pytest

from vulcan.runtime.settings import RuntimeSettings, SettingsError, generate_settings_schema, load_runtime_settings

SECRET = "AbCdEfGhIjKlMnOpQrStUvWxYz7890+/safe"
APPROVAL = "YnOpQrStUvWxAbCdEfGhIjKlM7890+/approval"

def env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    base = {
        "VULCAN_ENV": "development",
        "VULCAN_JWT_SECRET": SECRET,
        "VULCAN_RUNTIME_DURABLE_ROOT": str(tmp_path / "runtime"),
        "VULCAN_MEMORY_SQLITE_PATH": str(tmp_path / "runtime" / "memory" / "memory.sqlite"),
    }
    base.update(overrides)
    return base

@pytest.mark.parametrize("alias", ["GRAPHIX_JWT_SECRET", "JWT_SECRET_KEY", "JWT_SECRET"])
def test_deprecated_jwt_aliases_are_bounded_and_redacted(tmp_path: Path, alias: str) -> None:
    data = env(tmp_path)
    data.pop("VULCAN_JWT_SECRET")
    data[alias] = SECRET
    settings = load_runtime_settings(data)
    assert settings.jwt_secret.reveal() == SECRET
    assert alias in settings.deprecation_warnings[0]
    assert SECRET not in repr(settings)
    assert SECRET not in str(settings.public_dict())

@pytest.mark.parametrize("canonical,alias", [("VULCAN_ENABLE_SELF_IMPROVEMENT", "ENABLE_SELF_IMPROVEMENT"), ("VULCAN_REQUEST_TIMEOUT_SECONDS", "HYBRID_EXECUTOR_TIMEOUT")])
def test_deprecated_nonsecret_aliases_match_canonical_value(tmp_path: Path, canonical: str, alias: str) -> None:
    value = "1" if canonical == "VULCAN_ENABLE_SELF_IMPROVEMENT" else "1.0"
    settings = load_runtime_settings(env(tmp_path, **{canonical: value, alias: value, "VULCAN_APPROVAL_HMAC_SECRET": APPROVAL}))
    assert alias in " ".join(settings.deprecation_warnings)

@pytest.mark.parametrize("bad", ["", "short", "dev-secret-change-me"])
def test_weak_jwt_secret_rejected_without_leaking_value(tmp_path: Path, bad: str) -> None:
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings(env(tmp_path, VULCAN_JWT_SECRET=bad))
    if bad:
        assert bad not in str(exc.value)
    assert "weak secret" in str(exc.value) or "required" in str(exc.value)

def test_conflicting_aliases_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="conflicting values for VULCAN_JWT_SECRET"):
        load_runtime_settings(env(tmp_path, GRAPHIX_JWT_SECRET="z" * 40))

def test_production_requires_approval_secret_and_csiu(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="required"):
        load_runtime_settings(env(tmp_path, VULCAN_ENV="production"))
    with pytest.raises(SettingsError, match="production requires"):
        load_runtime_settings(env(tmp_path, VULCAN_ENV="production", VULCAN_APPROVAL_HMAC_SECRET=APPROVAL, VULCAN_CSIU_ENABLED="0"))
    settings = load_runtime_settings(env(tmp_path, VULCAN_ENV="production", VULCAN_APPROVAL_HMAC_SECRET=APPROVAL))
    assert settings.environment.value == "production"

@pytest.mark.parametrize("key,value", [("VULCAN_LANGUAGE_MODE", "provider"), ("VULCAN_MEMORY_BACKEND", "redis"), ("VULCAN_RUNTIME_REPLICAS", "2"), ("VULCAN_REQUEST_TIMEOUT_SECONDS", "999")])
def test_invalid_enums_topology_and_bounds_rejected(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises((SettingsError, ValueError)):
        load_runtime_settings(env(tmp_path, **{key: value}))

def test_transformer_mode_requires_absolute_release(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="VULCAN_LANGUAGE_RELEASE_PATH"):
        load_runtime_settings(env(tmp_path, VULCAN_LANGUAGE_MODE="transformer_proposal", VULCAN_LANGUAGE_RELEASE_PATH="relative"))

def test_schema_docs_entrypoint_helm_and_fields_share_names() -> None:
    schema = generate_settings_schema()
    canonical = set(schema["environment_variables"])
    entrypoint = Path("entrypoint.sh").read_text()
    helm = Path("helm/vulcanami/templates/deployment.yaml").read_text()
    example = Path(".env.example").read_text()
    fields = set(schema["fields"])
    assert set(RuntimeSettings.__dataclass_fields__) == fields
    for name in ["VULCAN_ENV", "VULCAN_JWT_SECRET", "VULCAN_RUNTIME_DURABLE_ROOT", "VULCAN_MEMORY_ENABLED", "VULCAN_CSIU_ENABLED"]:
        assert name in canonical
        assert name in entrypoint or name in helm
        assert name in example
