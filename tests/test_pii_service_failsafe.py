"""Tests for PII service failsafe — endpoint must not return fake results."""
import pytest

def test_detect_endpoint_returns_501():
    """The /detect endpoint must return 501 when no real detector is configured."""
    # Import the FastAPI app from pii_service
    from src.pii_service import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.post("/detect", json={"text": "My SSN is 123-45-6789"})
    assert response.status_code == 501, (
        f"PII /detect must return 501 Not Implemented, got {response.status_code}"
    )

def test_detect_never_returns_high_confidence_false():
    """No stub should return pii_found=False with confidence > 0.5."""
    from src.pii_service import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.post("/detect", json={"text": "test"})
    # If 501, that's correct — no fake results
    if response.status_code == 501:
        return
    data = response.json()
    if not data.get("pii_found", True):
        assert data.get("confidence", 0) < 0.5, (
            "Must not claim high confidence when no PII detection was performed"
        )
