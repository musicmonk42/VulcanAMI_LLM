import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DIGEST = "sha256:" + "1" * 64


def render() -> str:
    if shutil.which("helm") is None:
        pytest.skip("helm is required for chart render conformance")
    return subprocess.check_output([
        "helm", "template", "runtime-e2e", "helm/vulcanami",
        "--set", "image.tag=e2e", "--set", f"image.digest={DIGEST}",
    ], cwd=ROOT, text=True)


def test_helm_runtime_conformance_selected_image_security_and_storage() -> None:
    manifest = render()
    required = [
        "replicas: 1",
        f"image: \"ghcr.io/musicmonk42/vulcanami_llm-api:e2e@{DIGEST}\"",
        "uvicorn vulcan.runtime.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1",
        "path: /health/live",
        "path: /health/ready",
        "runAsUser: 1001",
        "runAsNonRoot: true",
        "readOnlyRootFilesystem: true",
        "allowPrivilegeEscalation: false",
        "mountPath: /var/lib/vulcan",
        "mountPath: /tmp",
        "kind: PersistentVolumeClaim",
        "name: runtime-e2e-vulcanami-secrets",
    ]
    for needle in required:
        assert needle in manifest


def test_helm_rejects_multiwriter_production_render() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm is required for chart render conformance")
    proc = subprocess.run([
        "helm", "template", "runtime-e2e", "helm/vulcanami", "--set", "replicaCount=2",
    ], cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "requires replicaCount=1" in proc.stderr
