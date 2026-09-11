from __future__ import annotations

import pytest

from vulcan.graphix.epistemic import ClaimStatus
from vulcan.graphix.runtime import ResponseMode
from vulcan.runtime.output import validate_response_coordinate


@pytest.mark.parametrize(
    ("mode", "status", "value", "citations"),
    [
        (ResponseMode.STRICT, ClaimStatus.COMPUTED, "4", ()),
        (ResponseMode.STRICT, ClaimStatus.RETRIEVED, "Paris", ("source:one",)),
        (ResponseMode.UNKNOWN, ClaimStatus.UNKNOWN, None, ()),
        (ResponseMode.CLARIFICATION, ClaimStatus.UNKNOWN, None, ()),
        (ResponseMode.DENIED, ClaimStatus.UNKNOWN, None, ()),
        (ResponseMode.ERROR, ClaimStatus.ERROR, None, ()),
        (
            ResponseMode.CONTESTED,
            ClaimStatus.CONTESTED,
            None,
            ("source:one", "source:two"),
        ),
    ],
)
def test_exhaustive_allowed_response_coordinates(mode, status, value, citations):
    validate_response_coordinate(mode, status, value, citations)


@pytest.mark.parametrize(
    ("mode", "status", "value", "citations"),
    [
        (ResponseMode.STRICT, ClaimStatus.UNKNOWN, None, ()),
        (ResponseMode.UNKNOWN, ClaimStatus.UNKNOWN, "secret-a", ()),
        (ResponseMode.DENIED, ClaimStatus.UNKNOWN, "secret-b", ()),
        (ResponseMode.ERROR, ClaimStatus.ERROR, "internal-error", ()),
        (ResponseMode.CONTESTED, ClaimStatus.CONTESTED, "winner", ("a", "b")),
        (ResponseMode.CONTESTED, ClaimStatus.CONTESTED, None, ("only-one",)),
        (ResponseMode.UNKNOWN, ClaimStatus.UNKNOWN, None, ("citation-as-truth",)),
    ],
)
def test_value_leaks_and_coordinate_substitutions_fail_closed(
    mode, status, value, citations
):
    with pytest.raises(ValueError):
        validate_response_coordinate(mode, status, value, citations)


def test_denied_secret_values_have_identical_public_coordinate() -> None:
    public_a = (ResponseMode.DENIED.value, ClaimStatus.UNKNOWN.value, None, ())
    public_b = (ResponseMode.DENIED.value, ClaimStatus.UNKNOWN.value, None, ())
    assert public_a == public_b
