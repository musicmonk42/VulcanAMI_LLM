from vulcan.runtime.output import DraftSegment, ResponseIRProjection, ProjectedClaim, SemanticFirewall, UntrustedRenderDraft

def _projection():
    return ResponseIRProjection("r", "und", 100, ("claim-a",), (ProjectedClaim("claim-a", "4", "computed", None, ("evidence-a",)),))

def test_firewall_accepts_only_ordered_known_references():
    result = SemanticFirewall().validate(_projection(), UntrustedRenderDraft("untrusted-render/1", (DraftSegment("claim", "claim-a"), DraftSegment("citation", "evidence-a"))))
    assert result.accepted

def test_firewall_rejects_added_or_mutated_claims():
    result = SemanticFirewall().validate(_projection(), UntrustedRenderDraft("untrusted-render/1", (DraftSegment("claim", "claim-b"), DraftSegment("text", "The result is 5"))))
    assert not result.accepted
