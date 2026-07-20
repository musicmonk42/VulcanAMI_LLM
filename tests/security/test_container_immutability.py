from pathlib import Path


def test_dockerfile_freezes_source_and_config_paths():
    dockerfile = Path("Dockerfile").read_text()
    assert "chmod -R a-w /app/src /app/configs /app/config /app/models" in dockerfile
    assert "chown -R graphix:graphix /var/lib/vulcan /app/data /tmp/vulcan-cache" in dockerfile
    assert "ENV VULCAN_ENV=production" in dockerfile


def test_entrypoint_refuses_limited_no_auth_mode():
    entrypoint = Path("entrypoint.sh").read_text()
    assert "exit 78" in entrypoint
    assert "Production serving refuses to downgrade" in entrypoint
