from pathlib import Path


def test_dockerfile_freezes_source_and_config_paths():
    dockerfile = Path("Dockerfile").read_text()
    assert "chmod -R a-w /usr/local/lib/python3.11/site-packages /app/config /app/docs /app/tests" in dockerfile
    assert "install -d -o vulcan -g vulcan -m 0700 /var/lib/vulcan" in dockerfile
    assert "VULCAN_ENV=production" in dockerfile


def test_entrypoint_refuses_limited_no_auth_mode():
    entrypoint = Path("entrypoint.sh").read_text()
    assert "exit 78" in entrypoint
    assert "Production serving refuses to downgrade" in entrypoint
