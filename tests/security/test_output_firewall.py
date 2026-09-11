from vulcan.runtime.output import DraftSegment, ResponseIRProjection, ProjectedClaim, SemanticFirewall, UntrustedRenderDraft
from vulcan.graphix.runtime import EpistemicStatus, ResponseMode


def _projection():
    return ResponseIRProjection("r", "und", 100, ResponseMode.STRICT, ("claim-a",), (ProjectedClaim("claim-a", "computed", "4", EpistemicStatus.COMPUTED, "exact", ("evidence-a",)),))


def test_firewall_accepts_only_ordered_known_references():
    draft = UntrustedRenderDraft("untrusted-render/1", "test-adapter", (DraftSegment("claim", "claim-a"), DraftSegment("caveat", "claim-a"), DraftSegment("citation", "evidence-a")))
    assert SemanticFirewall().validate(_projection(), draft).accepted


def test_firewall_rejects_added_or_mutated_claims_or_missing_caveat():
    draft = UntrustedRenderDraft("untrusted-render/1", "test-adapter", (DraftSegment("claim", "claim-b"), DraftSegment("text", "The result is 5")))
    assert not SemanticFirewall().validate(_projection(), draft).accepted


def test_transport_never_releases_text_other_than_the_authorized_digest():
    from hashlib import sha256
    from vulcan.runtime.case import CognitiveCaseStatus
    from vulcan.runtime.kernel import KernelResult
    from vulcan.graphix.runtime import ResponseIR

    ir = ResponseIR("3", "r", "case", None, "s", ResponseMode.STRICT, ("claim-a",))
    result = KernelResult("The computed result is 5.", ir, CognitiveCaseStatus.SUCCESS, "allow", sha256(b"The computed result is 4.").hexdigest())
    envelope = result.transport(case_id="case", runtime_id="runtime", snapshot_id="s")
    assert envelope["response"] is None
    assert envelope["metadata"]["response_released"] is False
