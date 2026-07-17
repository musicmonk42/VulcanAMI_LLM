from vulcan.runtime.output import DraftSegment, ResponseIRProjection, ProjectedClaim, SemanticFirewall, UntrustedRenderDraft
from vulcan.runtime.semantic import EpistemicStatus, ResponseMode


def _projection():
    return ResponseIRProjection("r", "und", 100, ResponseMode.STRICT, ("claim-a",), (ProjectedClaim("claim-a", "computed", "4", EpistemicStatus.COMPUTED, "exact", ("evidence-a",)),))


def test_firewall_accepts_only_ordered_known_references():
    draft = UntrustedRenderDraft("untrusted-render/1", "test-adapter", (DraftSegment("claim", "claim-a"), DraftSegment("caveat", "claim-a"), DraftSegment("citation", "evidence-a")))
    assert SemanticFirewall().validate(_projection(), draft).accepted


def test_firewall_rejects_added_or_mutated_claims_or_missing_caveat():
    draft = UntrustedRenderDraft("untrusted-render/1", "test-adapter", (DraftSegment("claim", "claim-b"), DraftSegment("text", "The result is 5")))
    assert not SemanticFirewall().validate(_projection(), draft).accepted
