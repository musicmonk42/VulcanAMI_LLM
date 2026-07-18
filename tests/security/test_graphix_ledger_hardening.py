from dataclasses import replace
from datetime import datetime, timezone
import pytest

from vulcan.runtime.case import CognitiveCase
from vulcan.runtime.domain_registry import DomainEvidenceSupport, DomainLookupResult
from vulcan.runtime.semantic import (
    AcceptedInterpretation, Claim, CompiledGraphixPlan, Derivation, EpistemicStatus,
    EvidenceArtifact, EvidenceKind, ExecutionStatus, GraphixPlan, InterpretationProposal,
    ProposedCandidate, Proposition, RESPONSE_IR_VERSION, ResponseIR, ResponseMode,
    SourceSpan, Utterance, build_graphix_plan, canonical_digest, compile_graphix_plan,
    execute_graphix_plan, render_strict, validate_ledger, validate_proposal,
)


def _ledger(case_id="case"):
    e = EvidenceArtifact("e1", EvidenceKind.RETRIEVED_RECORD, canonical_digest("v"), "ref", "origin", "exact", case_id, observed_at=datetime.now(timezone.utc), source_integrity="digest-verified", trust_policy="test", citation="ref")
    d = Derivation("d1", "exact-domain-lookup", "1", ("e1",), "c1", (), (), True, ExecutionStatus.SUCCESS, canonical_digest("trace"))
    c = Claim("c1", Proposition("k", "lookup_value", "v"), EpistemicStatus.RETRIEVED, ("e1",), ("d1",), citation_ids=("e1",), temporal_validity="snapshot-bound")
    return e, d, c


def test_cross_case_evidence_rejected():
    e, d, c = _ledger("other")
    with pytest.raises(ValueError): validate_ledger((e,), (d,), (c,), case_id="case")


def test_dangling_unused_derivation_rejected():
    e, d, c = _ledger()
    unused = replace(d, derivation_id="d2")
    with pytest.raises(ValueError): validate_ledger((e,), (d, unused), (c,), case_id="case")


def test_derivation_self_reference_rejected():
    e, d, c = _ledger()
    d = replace(d, inputs=("d1",))
    with pytest.raises(ValueError): validate_ledger((e,), (d,), (c,), case_id="case")


def test_citation_borrowed_from_another_claim_rejected():
    e, d, c = _ledger()
    e2 = replace(e, artifact_id="e2")
    c = replace(c, evidence_ids=("e1",), citation_ids=("e2",))
    with pytest.raises(ValueError): validate_ledger((e, e2), (d,), (c,), case_id="case")


def test_plan_mutation_after_compilation_rejected():
    acc = AcceptedInterpretation(0, "arithmetic", "2+2")
    plan = build_graphix_plan(acc, request_digest="r", state_snapshot_id="s")
    compiled = compile_graphix_plan(plan, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:none")
    mutated = CompiledGraphixPlan(replace(plan, operands=(("expression", "2+3"),)), compiled.plan_digest)
    with pytest.raises(ValueError): execute_graphix_plan(mutated, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:none")


def test_valid_digest_domain_filter_plan_rejected():
    plan = GraphixPlan("graphix-plan/1", "plan-" + "a" * 32, "lookup", "r", "s", "d", (("key", "alpha"), ("domain_hint", "private")))
    with pytest.raises(ValueError): compile_graphix_plan(plan, request_digest="r", state_snapshot_id="s", domain_snapshot_id="d")


def test_transformer_supplied_factual_values_rejected():
    utterance = Utterance.from_text("2+2")
    proposal = InterpretationProposal("semantic-ingress/2", (ProposedCandidate("arithmetic", "4", SourceSpan(0, 3)),), "bad")
    with pytest.raises(ValueError): validate_proposal(utterance, proposal)


def test_response_values_absent_from_ledger_rejected():
    e, d, c = _ledger()
    ir = ResponseIR(RESPONSE_IR_VERSION, "r", "case", None, "s", ResponseMode.STRICT, ("c1",), literals=("laundered",))
    with pytest.raises(ValueError): render_strict(ir, (c,), (d,), (e,))


def test_arithmetic_precedence_and_resource_syntax_violations():
    acc = AcceptedInterpretation(0, "arithmetic", "2 + 3 * 4")
    plan = build_graphix_plan(acc, request_digest="r", state_snapshot_id="s")
    comp = compile_graphix_plan(plan, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:none")
    claim, derivation, evidence = execute_graphix_plan(comp, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:none")
    assert claim.proposition.object == "14"
    assert evidence == ()
    bad = build_graphix_plan(AcceptedInterpretation(0, "arithmetic", "2 ** 999"), request_digest="r", state_snapshot_id="s")
    claim, _, _ = execute_graphix_plan(compile_graphix_plan(bad, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:none"), request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:none")
    assert claim.status is EpistemicStatus.UNKNOWN


def test_lookup_plan_executes_through_injected_domain_and_strict_rendering():
    from datetime import datetime, timezone
    class Domain:
        domain_snapshot_id = "domain:1"
        def lookup_exact(self, key):
            support = DomainEvidenceSupport("geo", 1, "fact-1", "ev-1", "cities:paris", canonical_digest("Paris"), datetime.now(timezone.utc), None, "exact-key", "test", ())
            return DomainLookupResult("retrieved", key, "Paris", self.domain_snapshot_id, (support,), total_evidence=1)
    acc = AcceptedInterpretation(0, "lookup", "france.capital")
    plan = build_graphix_plan(acc, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:1")
    claim, derivation, evidence = execute_graphix_plan(compile_graphix_plan(plan, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:1"), request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:1", case_id="case", domain=Domain())
    ir = ResponseIR(RESPONSE_IR_VERSION, "r", "case", None, "s", ResponseMode.STRICT, (claim.claim_id,))
    artifact = render_strict(ir, (claim,), (derivation,), evidence)
    assert "Paris" in artifact.text
    assert "cities:paris" not in artifact.text


def test_tuple_only_lookup_provider_cannot_produce_positive_canonical_claim():
    class LegacyDomain:
        domain_snapshot_id = "domain:legacy"
        def lookup_exact(self, key): return ("Paris", "cities:paris")
    acc = AcceptedInterpretation(0, "lookup", "france.capital")
    plan = build_graphix_plan(acc, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:legacy")
    compiled = compile_graphix_plan(plan, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:legacy")
    import pytest
    with pytest.raises(ValueError, match="typed DomainLookupResult"):
        execute_graphix_plan(compiled, request_digest="r", state_snapshot_id="s", domain_snapshot_id="domain:legacy", case_id="case", domain=LegacyDomain())
