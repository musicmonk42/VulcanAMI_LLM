"""Framework-independent typed semantic orchestration boundary."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vulcan.memory.governed import GovernedMemoryPort

from vulcan.constitution.primitives import AuthorityLevel
from vulcan.microkernel.authority import EvidenceRecord, promote_authority
from vulcan.microkernel.episode import (
    ArtifactRef,
    SnapshotBundleRef,
)
from vulcan.microkernel.episode import canonical_digest as episode_digest
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.principals import Principal, PrincipalKind
from vulcan.microkernel.transactions import (
    CommandAuthority,
    ConstitutionalTransactionService,
    PublicationAuthorization,
    TerminalOutcome,
)
from vulcan.safety.safety_types import ResponseSafetyContext

from .case import CognitiveCase, CognitiveCaseStatus
from .epistemic_adapter import (
    adapt_runtime_semantic_candidate,
    verify_runtime_semantic_projection,
)
from .finalization import FinalizationDecision, ResponseFinalizerPort
from .output import (
    DeterministicLanguageOutput,
    LanguageOutputPort,
    SemanticFirewall,
    bind_publication,
    project_committed,
    render_projection,
)
from .semantic import (
    RESPONSE_IR_VERSION,
    ClarificationRequest,
    DeterministicLanguageInput,
    LanguageInputPort,
    ResponseIR,
    ResponseMode,
    Utterance,
    accept,
    build_graphix_plan,
    canonical_digest,
    compile_graphix_plan,
    execute,
    execute_graphix_plan,
    validate_proposal,
)


@dataclass(frozen=True)
class KernelRequest:
    utterance: Utterance
    conversation_id: str | None


@dataclass(frozen=True)
class KernelResult:
    response: str
    response_ir: ResponseIR
    status: CognitiveCaseStatus
    finalization: str
    authorized_text_digest: str | None = None
    publication_authorization_digest: str | None = None

    def transport(
        self, *, case_id: str, runtime_id: str, snapshot_id: str | None
    ) -> dict[str, object]:
        exact = hashlib.sha256(self.response.encode("utf-8")).hexdigest()
        released = (
            self.finalization == FinalizationDecision.ALLOW.value
            and self.authorized_text_digest == exact
            and self.publication_authorization_digest is not None
        )
        return {
            "response": self.response if released else None,
            "metadata": {
                "case_id": case_id,
                "runtime_id": runtime_id,
                "state_snapshot_id": snapshot_id,
                "semantic_schema_version": self.response_ir.schema_version,
                "terminal_status": self.status.value,
                "response_released": released,
                "finalized": True,
                "finalization_safety_decision": self.finalization,
            },
        }


class CognitiveKernel:
    CAPABILITY_OWNER = "RuntimeContainer.CognitiveKernel"
    CAPABILITY_RELEASE_DIGEST = (
        "e324762e004e2ac30d13758b6d9b5a80ccefb06c5d4a5f8d373a20d782bf426c"
    )

    def __init__(
        self,
        *,
        state_authority: Any,
        finalizer: ResponseFinalizerPort,
        language_input: LanguageInputPort | None = None,
        language_output: LanguageOutputPort | None = None,
        memory: "GovernedMemoryPort | None" = None,
        audit: Any = None,
        alignment: Any = None,
        domain_lookup: Any = None,
    ) -> None:
        # The kernel owns the only memory port exposed to the production path.
        # It deliberately does not turn retrieved text into executable semantics.
        self._state_authority = state_authority
        self._finalizer = finalizer
        self._language_input = language_input or DeterministicLanguageInput()
        self._language_output = language_output or DeterministicLanguageOutput()
        self._memory = memory
        self._audit = audit
        self._alignment = alignment
        self._domain_lookup = domain_lookup
        self.calls = 0
        self._transactions: ConstitutionalTransactionService | None = None
        self._kernel_principal: Principal | None = None
        self._direct_compatibility_store: EpisodeStore | None = None

    def bind_transaction_service(
        self, service: ConstitutionalTransactionService, principal: Principal
    ) -> None:
        if self._transactions is not None and self._transactions is not service:
            raise RuntimeError("transaction service already bound")
        if not principal.is_kernel:
            raise TypeError("transaction authority must be SYSTEM_KERNEL")
        self._transactions = service
        self._kernel_principal = principal

    def disable_legacy_case_audit(self) -> None:
        """Retire mutable ``case.*`` lifecycle writes on the composed path.

        The audit owner remains available to other authority owners through the
        container; only this legacy lifecycle projection is disconnected.
        """
        self._audit = None

    def _command(
        self,
        case: CognitiveCase,
        policy_digest: str,
        validation_digest: str,
        level: AuthorityLevel,
    ) -> CommandAuthority:
        if (
            self._transactions is None
            or self._kernel_principal is None
            or case.episode is None
            or case.episode.snapshot_bundle is None
        ):
            raise RuntimeError("constitutional transaction service is not bound")
        policy = (
            policy_digest
            if len(policy_digest) == 64
            else episode_digest({"policy": policy_digest or "default-deny"})
        )
        evidence = EvidenceRecord(
            self._kernel_principal.identity_digest,
            validation_digest,
            policy,
            datetime.now(timezone.utc),
        )
        grant = promote_authority(
            current=AuthorityLevel.UNTRUSTED_PROPOSAL,
            target=level,
            principal=self._kernel_principal,
            evidence=evidence,
        )
        return CommandAuthority(
            self._kernel_principal,
            grant,
            evidence,
            validation_digest,
            policy,
            case.episode.snapshot_bundle.state_digest,
            case.episode.digest,
        )

    def _apply(self, case: CognitiveCase, episode) -> None:
        case.project_episode(episode)

    def _bind_direct_compatibility(self, case: CognitiveCase) -> None:
        """Give non-composed legacy callers a durable, isolated command path."""
        if self._transactions is not None and self._direct_compatibility_store is None:
            return
        if case.episode is None:
            raise RuntimeError("authoritative episode is unavailable")
        snapshot_digest = episode_digest(
            {"compatibility_snapshot": case.state_snapshot_id or self._snapshot_id()}
        )
        case.state_snapshot_id = snapshot_digest
        case.episode = case.episode.bind_snapshot_bundle_for_migration(
            SnapshotBundleRef(f"snapshot-compat-{case.case_id}", snapshot_digest)
        )
        if self._direct_compatibility_store is None:
            path = tempfile.NamedTemporaryFile(
                prefix="vulcan-direct-kernel-", suffix=".sqlite3", delete=False
            ).name
            self._direct_compatibility_store = EpisodeStore(path)
            principal = Principal(
                PrincipalKind.SYSTEM_KERNEL,
                "direct-kernel-compatibility",
                hashlib.sha256(b"direct-kernel-compatibility-v1").hexdigest(),
            )
            self.bind_transaction_service(
                ConstitutionalTransactionService(
                    self._direct_compatibility_store,
                    EpistemicStore(path + ".epistemic"),
                ),
                principal,
            )
        self._direct_compatibility_store.create(case.episode)

    def capabilities(self) -> tuple[str, ...]:
        """Capabilities implemented by this composed kernel, not marketing text."""
        caps = ["bounded-arithmetic"]
        mem_caps = getattr(self._memory, "capabilities", None)
        if callable(mem_caps):
            caps.extend(mem_caps())
        return tuple(caps)

    async def handle(self, request: KernelRequest, case: CognitiveCase) -> KernelResult:
        if case.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("kernel received a closed cognitive case")
        if (
            request.utterance.digest != case.input_hash
            or request.conversation_id != case.conversation_id
        ):
            raise ValueError("request/case correlation mismatch")
        self.calls += 1
        case.state_snapshot_id = case.state_snapshot_id or self._snapshot_id()
        self._bind_direct_compatibility(case)
        case.record("semantic_ingress")
        alignment_lease = (
            self._alignment.lease() if self._alignment is not None else None
        )
        alignment_lease_closed = False
        terminal_commit_started = False
        policy = getattr(alignment_lease, "policy", None)
        policy_digest = getattr(policy, "policy_digest", "")

        def close_alignment_lease() -> None:
            nonlocal alignment_lease_closed
            if alignment_lease is not None and not alignment_lease_closed:
                alignment_lease.close()
                alignment_lease_closed = True

        if self._audit:
            self._audit.append(
                "case.started",
                {
                    "case_id": case.case_id,
                    "request_id": case.request_id,
                    "request_digest": case.input_hash,
                    "conversation_id": case.conversation_id or "",
                    "state_snapshot_id": case.state_snapshot_id,
                },
            )
        try:
            try:
                proposal = await self._language_input.propose(request.utterance)
                bundle = validate_proposal(request.utterance, proposal)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A proposer can never turn an error into a provider answer.
                case.record("input_proposal_unavailable")
                bundle = validate_proposal(
                    request.utterance,
                    await DeterministicLanguageInput().propose(request.utterance),
                )
            case.interpretation = bundle
            selection = accept(bundle)
            policy_digest = getattr(policy, "policy_digest", "")
            validation_digest = canonical_digest(bundle)
            self._apply(
                case,
                self._transactions.record_interpretation(
                    case.case_id,
                    self._command(
                        case,
                        policy_digest,
                        validation_digest,
                        AuthorityLevel.VALIDATED_CANDIDATE,
                    ),
                    {
                        "parser_identity": bundle.parser_identity,
                        "candidate_count": str(len(bundle.candidates)),
                        "ontology_version": bundle.ontology_version,
                    },
                ),
            )
            self._apply(
                case,
                self._transactions.ground_candidate(
                    case.case_id,
                    self._command(
                        case,
                        policy_digest,
                        validation_digest,
                        AuthorityLevel.VALIDATED_CANDIDATE,
                    ),
                ),
            )
            if self._audit:
                self._audit.append(
                    "case.interpreted",
                    {
                        "case_id": case.case_id,
                        "request_id": case.request_id,
                        "request_digest": case.input_hash,
                        "parser_identity": bundle.parser_identity,
                        "candidate_count": len(bundle.candidates),
                    },
                )
            if isinstance(selection, ClarificationRequest):
                case.clarification = selection
                # A clarification is an explicit unknown claim, not an evaluation side effect.
                claim, derivation = execute(
                    type(
                        "Unsupported",
                        (),
                        {
                            "operation": "unsupported",
                            "expression": "",
                            "assumptions": (),
                        },
                    )()
                )
                plan_digest = "0" * 64
                self._apply(
                    case,
                    self._transactions.open_deliberation(
                        case.case_id,
                        self._command(
                            case,
                            policy_digest,
                            validation_digest,
                            AuthorityLevel.VALIDATED_CANDIDATE,
                        ),
                        (
                            ArtifactRef(
                                f"plan:{case.case_id}",
                                plan_digest,
                                "semantic-plan.compat.v1",
                            ),
                        ),
                    ),
                )
                if self._audit:
                    self._audit.append(
                        "case.plan_compiled",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "operation": "unsupported",
                            "plan_digest": plan_digest,
                            "plan_shape": {"operands": 0},
                            "domain_snapshot_id": "domain:none",
                            "alignment_policy_digest": getattr(
                                policy, "policy_digest", ""
                            ),
                        },
                    )
                proposed_claims = (claim,)
                proposed_derivations = (derivation,)
                proposed_evidence = ()
                epistemic_auth = self._command(
                    case,
                    policy_digest,
                    canonical_digest(
                        {
                            "claims": [
                                canonical_digest(item) for item in proposed_claims
                            ],
                            "derivations": [
                                canonical_digest(item) for item in proposed_derivations
                            ],
                            "evidence": [
                                canonical_digest(item) for item in proposed_evidence
                            ],
                        }
                    ),
                    AuthorityLevel.COMMITTED_BELIEF,
                )
                prior = self._transactions.epistemic_head(case.case_id)
                candidate = adapt_runtime_semantic_candidate(
                    episode_id=case.case_id,
                    case_id=case.case_id,
                    snapshot_digest=case.state_snapshot_id or "",
                    claims=proposed_claims,
                    evidence=proposed_evidence,
                    derivations=proposed_derivations,
                    authority=epistemic_auth,
                    prior_commit_digest=None if prior is None else prior.commit_digest,
                )
                self._apply(
                    case,
                    self._transactions.commit_epistemic_candidate(
                        case.case_id,
                        epistemic_auth,
                        candidate,
                    ),
                )
                case.project_committed_ledger(
                    episode=case.episode,
                    claim=claim,
                    derivation=derivation,
                    evidence=proposed_evidence,
                )
                verify_runtime_semantic_projection(
                    self._transactions.epistemic_head(case.case_id),
                    claims=proposed_claims,
                    evidence=proposed_evidence,
                    derivations=proposed_derivations,
                )
                if self._audit:
                    self._audit.append(
                        "case.ledger_committed",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "claim_digests": [claim.claim_id],
                            "derivation_digests": [derivation.derivation_id],
                            "evidence_ids": [],
                            "evidence": [],
                        },
                    )
                decision = (
                    self._alignment.decide(
                        case.claims, case.evidence, case.derivations, policy
                    )
                    if self._alignment is not None
                    else type(
                        "D",
                        (),
                        {
                            "accepted": False,
                            "reason_codes": ("unknown_abstain",),
                            "policy_digest": "",
                            "policy_revision": 0,
                        },
                    )()
                )
                if self._audit:
                    self._audit.append(
                        "case.alignment_decided",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "accepted": False,
                            "reason_codes": list(decision.reason_codes),
                            "policy_digest": decision.policy_digest,
                            "policy_revision": decision.policy_revision,
                        },
                    )
                mode = ResponseMode.CLARIFICATION
                status = CognitiveCaseStatus.ABSTAINED
                accepted_id = None
            else:
                case.accepted_interpretation = selection
                domain_port = self._domain_lookup
                lease_cm = (
                    domain_port.lease() if hasattr(domain_port, "lease") else None
                )
                leased_domain = lease_cm if lease_cm is not None else domain_port
                try:
                    domain_snapshot_id = getattr(
                        leased_domain, "domain_snapshot_id", "domain:none"
                    )
                    plan = build_graphix_plan(
                        selection,
                        request_digest=request.utterance.digest,
                        state_snapshot_id=case.state_snapshot_id or "",
                        domain_snapshot_id=domain_snapshot_id,
                    )
                    compiled = compile_graphix_plan(
                        plan,
                        request_digest=request.utterance.digest,
                        state_snapshot_id=case.state_snapshot_id or "",
                        domain_snapshot_id=domain_snapshot_id,
                    )
                    self._apply(
                        case,
                        self._transactions.open_deliberation(
                            case.case_id,
                            self._command(
                                case,
                                policy_digest,
                                validation_digest,
                                AuthorityLevel.VALIDATED_CANDIDATE,
                            ),
                            (
                                ArtifactRef(
                                    f"plan:{case.case_id}",
                                    compiled.plan_digest,
                                    "semantic-plan.compat.v1",
                                ),
                            ),
                        ),
                    )
                    if self._audit:
                        self._audit.append(
                            "case.plan_compiled",
                            {
                                "case_id": case.case_id,
                                "request_id": case.request_id,
                                "request_digest": case.input_hash,
                                "operation": compiled.plan.operation,
                                "plan_digest": compiled.plan_digest,
                                "plan_shape": {"operands": len(compiled.plan.operands)},
                                "domain_snapshot_id": domain_snapshot_id,
                                "alignment_policy_digest": getattr(
                                    policy, "policy_digest", ""
                                ),
                            },
                        )
                    if (
                        leased_domain is not None
                        and getattr(leased_domain, "domain_snapshot_id", None)
                        != compiled.plan.domain_snapshot_id
                    ):
                        raise ValueError("plan snapshot mismatch")
                    claim, derivation, evidence = execute_graphix_plan(
                        compiled,
                        request_digest=request.utterance.digest,
                        state_snapshot_id=case.state_snapshot_id or "",
                        domain_snapshot_id=domain_snapshot_id,
                        case_id=case.case_id,
                        domain=leased_domain,
                    )
                finally:
                    if lease_cm is not None:
                        lease_cm.close()
                proposed_claims = (claim,)
                proposed_derivations = (derivation,)
                proposed_evidence = evidence
                ledger_digest = canonical_digest(
                    {
                        "claims": [canonical_digest(item) for item in proposed_claims],
                        "derivations": [
                            canonical_digest(item) for item in proposed_derivations
                        ],
                        "evidence": [
                            canonical_digest(item) for item in proposed_evidence
                        ],
                    }
                )
                epistemic_auth = self._command(
                    case,
                    policy_digest,
                    ledger_digest,
                    AuthorityLevel.COMMITTED_BELIEF,
                )
                prior = self._transactions.epistemic_head(case.case_id)
                candidate = adapt_runtime_semantic_candidate(
                    episode_id=case.case_id,
                    case_id=case.case_id,
                    snapshot_digest=case.state_snapshot_id or "",
                    claims=proposed_claims,
                    evidence=proposed_evidence,
                    derivations=proposed_derivations,
                    authority=epistemic_auth,
                    prior_commit_digest=None if prior is None else prior.commit_digest,
                )
                self._apply(
                    case,
                    self._transactions.commit_epistemic_candidate(
                        case.case_id,
                        epistemic_auth,
                        candidate,
                    ),
                )
                case.project_committed_ledger(
                    episode=case.episode,
                    claim=claim,
                    derivation=derivation,
                    evidence=proposed_evidence,
                )
                verify_runtime_semantic_projection(
                    self._transactions.epistemic_head(case.case_id),
                    claims=proposed_claims,
                    evidence=proposed_evidence,
                    derivations=proposed_derivations,
                )
                if self._audit:
                    self._audit.append(
                        "case.ledger_committed",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "claim_digests": [c.claim_id for c in case.claims],
                            "derivation_digests": [
                                d.derivation_id for d in case.derivations
                            ],
                            "evidence_ids": [e.artifact_id for e in case.evidence],
                            "evidence": [
                                {
                                    "evidence_id": e.artifact_id,
                                    "origin": e.origin,
                                    "content_digest": e.content_digest,
                                    "citation": e.citation or "",
                                    "source_integrity": e.source_integrity,
                                    "valid_until": (
                                        e.valid_until.isoformat().replace("+00:00", "Z")
                                        if e.valid_until
                                        else ""
                                    ),
                                }
                                for e in case.evidence
                            ],
                        },
                    )
                decision = (
                    self._alignment.decide(
                        case.claims, case.evidence, case.derivations, policy
                    )
                    if self._alignment is not None
                    else type(
                        "D",
                        (),
                        {
                            "accepted": claim.status.value in {"computed", "retrieved"},
                            "reason_codes": ("passed",),
                            "policy_digest": "",
                            "policy_revision": 0,
                        },
                    )()
                )
                if self._audit:
                    self._audit.append(
                        "case.alignment_decided",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "accepted": decision.accepted,
                            "reason_codes": list(decision.reason_codes),
                            "policy_digest": decision.policy_digest,
                            "policy_revision": decision.policy_revision,
                        },
                    )
                mode = (
                    ResponseMode.STRICT
                    if decision.accepted
                    and claim.status.value in {"computed", "retrieved", "proven"}
                    else ResponseMode.UNKNOWN
                )
                status = (
                    CognitiveCaseStatus.SUCCESS
                    if mode is ResponseMode.STRICT
                    else CognitiveCaseStatus.ABSTAINED
                )
                accepted_id = selection.interpretation_id
            response_ir = ResponseIR(
                RESPONSE_IR_VERSION,
                f"response-{case.case_id}",
                case.case_id,
                accepted_id,
                case.state_snapshot_id,
                mode,
                (claim.claim_id,),
            )
            case.response_ir = response_ir
            # The adapter sees only the projection; firewall rejection is always strict fallback.
            epistemic_head = self._transactions.epistemic_head(case.case_id)
            if epistemic_head is None:
                raise RuntimeError("durable epistemic head is unavailable")
            durable_episode_head = self._transactions.episode_head(case.case_id)
            if (
                case.episode is None
                or durable_episode_head.digest != case.episode.digest
            ):
                raise RuntimeError("case projection differs from durable episode head")
            projection = project_committed(
                response_ir, durable_episode_head, epistemic_head
            )
            try:
                draft = await self._language_output.render(projection)
                if SemanticFirewall().validate(projection, draft).accepted:
                    case.record("output_draft_validated")
                else:
                    case.record("output_draft_rejected")
            except asyncio.CancelledError:
                raise
            except Exception:
                # Provider/adapter failures are diagnostics only; strict rendering remains authoritative.
                case.record("output_draft_unavailable")
            current_epistemic_head = self._transactions.epistemic_head(case.case_id)
            if (
                current_epistemic_head is None
                or current_epistemic_head.commit_digest
                != projection.epistemic_head_digest
                or self._transactions.episode_head(case.case_id).digest
                != projection.episode_head_digest
            ):
                raise RuntimeError("durable head changed after response projection")
            artifact = render_projection(projection)
            case.render_artifact = artifact
            case.record("strict_rendered")
            final_context = ResponseSafetyContext(
                case_id=case.case_id,
                episode_id=case.request_id,
                response_ir_digest=artifact.ir_digest,
                rendered_text_digest=hashlib.sha256(
                    artifact.text.encode("utf-8")
                ).hexdigest(),
                policy_identity=getattr(policy, "policy_digest", "") or "unknown",
                policy_release=str(
                    getattr(policy, "revision", "")
                    or getattr(policy, "policy_revision", "")
                    or "unknown"
                ),
                actor_risk="unknown",
            )
            finalize = self._finalizer.finalize
            if len(inspect.signature(finalize).parameters) == 1:
                finalization = await finalize(artifact)
            else:
                finalization = await finalize(artifact, final_context)
            if finalization.artifact != artifact:
                raise RuntimeError("finalizer changed the committed render artifact")
            if (
                finalization.decision is FinalizationDecision.ALLOW
                and finalization.public_text != artifact.text
            ):
                raise RuntimeError(
                    "finalizer allowed text outside the committed render"
                )
            case.record_finalization(finalization.decision.value)
            if finalization.decision is FinalizationDecision.BLOCK:
                status = CognitiveCaseStatus.BLOCKED
            elif finalization.decision is FinalizationDecision.ERROR:
                status = CognitiveCaseStatus.FINALIZATION_ERROR
            elif finalization.decision is FinalizationDecision.CANCELLED:
                status = CognitiveCaseStatus.CANCELLED
            close_alignment_lease()
            terminal_commit_started = True
            publication_digest = None
            if finalization.decision is FinalizationDecision.ALLOW and status in {
                CognitiveCaseStatus.SUCCESS,
                CognitiveCaseStatus.ABSTAINED,
            }:
                if case.episode is None:
                    raise RuntimeError("authoritative episode is unavailable")
                response_ref = ArtifactRef(
                    response_ir.response_id,
                    hashlib.sha256(
                        finalization.public_text.encode("utf-8")
                    ).hexdigest(),
                    "published-response.v1",
                )
                bound_policy = (
                    policy_digest
                    if len(policy_digest) == 64
                    else episode_digest({"policy": policy_digest or "default-deny"})
                )
                epistemic_head = self._transactions.epistemic_head(case.case_id)
                if epistemic_head is None:
                    raise RuntimeError("durable epistemic head is unavailable")
                authorization = PublicationAuthorization(
                    epistemic_head.commit_digest.removeprefix("sha256:"),
                    canonical_digest(
                        {
                            "accepted": decision.accepted,
                            "reasons": list(decision.reason_codes),
                            "policy_digest": decision.policy_digest,
                            "policy_revision": decision.policy_revision,
                        }
                    ),
                    bound_policy,
                    int(decision.policy_revision),
                    canonical_digest(
                        {
                            "decision": finalization.decision.value,
                            "artifact": artifact.ir_digest,
                        }
                    ),
                    hashlib.sha256(
                        finalization.public_text.encode("utf-8")
                    ).hexdigest(),
                    canonical_digest({"privacy": case.privacy_classification}),
                    canonical_digest(
                        {"conversation_bound": case.conversation_id is not None}
                    ),
                    self._kernel_principal.identity_digest,
                    self._kernel_principal.release_digest,
                )
                projection = bind_publication(
                    projection, authorization, finalization.public_text
                )
                publication_digest = projection.publication_authorization_digest
                self._apply(
                    case,
                    self._transactions.authorize_response_publication(
                        case.case_id,
                        self._command(
                            case,
                            bound_policy,
                            authorization.finalizer_decision_digest,
                            AuthorityLevel.AUTHORIZED_PLAN,
                        ),
                        authorization=authorization,
                        response=response_ref,
                    ),
                )
                self._apply(
                    case,
                    self._transactions.communicate(
                        case.case_id,
                        self._command(
                            case,
                            bound_policy,
                            authorization.rendered_text_digest,
                            AuthorityLevel.AUTHORIZED_PLAN,
                        ),
                    ),
                )
                consolidation = ArtifactRef(
                    f"consolidation:{case.case_id}",
                    canonical_digest(
                        {
                            "episode": case.episode.digest,
                            "response": response_ref.digest,
                        }
                    ),
                    "episode-consolidation.v1",
                )
                self._apply(
                    case,
                    self._transactions.consolidate(
                        case.case_id,
                        self._command(
                            case,
                            bound_policy,
                            consolidation.digest,
                            AuthorityLevel.COMMITTED_BELIEF,
                        ),
                        consolidation,
                    ),
                )
            else:
                outcomes = {
                    CognitiveCaseStatus.ABSTAINED: TerminalOutcome.ABSTENTION,
                    CognitiveCaseStatus.BLOCKED: TerminalOutcome.BLOCK,
                    CognitiveCaseStatus.FINALIZATION_ERROR: TerminalOutcome.FAILURE,
                    CognitiveCaseStatus.CANCELLED: TerminalOutcome.CANCELLATION,
                    CognitiveCaseStatus.FAILED: TerminalOutcome.FAILURE,
                }
                self._apply(
                    case,
                    self._transactions.terminalize(
                        case.case_id,
                        self._command(
                            case,
                            policy_digest,
                            canonical_digest({"terminal": status.value}),
                            AuthorityLevel.VALIDATED_CANDIDATE,
                        ),
                        outcomes[status],
                    ),
                )
            case.mirror_terminal(status)
            if self._audit:
                self._audit.append(
                    "case.finalized",
                    {
                        "case_id": case.case_id,
                        "request_id": case.request_id,
                        "request_digest": case.input_hash,
                        "finalization": finalization.decision.value,
                        "terminal_status": status.value,
                        "response_ir_digest": artifact.ir_digest,
                    },
                )
                event_type = (
                    "case.completed"
                    if status is CognitiveCaseStatus.SUCCESS
                    else f"case.{status.value}"
                )
                self._audit.append(
                    event_type,
                    {
                        "case_id": case.case_id,
                        "request_id": case.request_id,
                        "request_digest": case.input_hash,
                        "status": status.value,
                        "response_ir_digest": artifact.ir_digest,
                    },
                )
            return KernelResult(
                finalization.public_text,
                response_ir,
                status,
                finalization.decision.value,
                (
                    hashlib.sha256(finalization.public_text.encode("utf-8")).hexdigest()
                    if finalization.decision is FinalizationDecision.ALLOW
                    else None
                ),
                publication_digest,
            )
        except asyncio.CancelledError:
            close_alignment_lease()
            if case.terminal_status is CognitiveCaseStatus.OPEN:
                self._apply(
                    case,
                    self._transactions.terminalize(
                        case.case_id,
                        self._command(
                            case,
                            policy_digest,
                            canonical_digest({"terminal": "cancelled"}),
                            AuthorityLevel.VALIDATED_CANDIDATE,
                        ),
                        TerminalOutcome.CANCELLATION,
                    ),
                )
                case.mirror_terminal(CognitiveCaseStatus.CANCELLED, "cancelled")
                if self._audit:
                    self._audit.append(
                        "case.cancelled",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "status": CognitiveCaseStatus.CANCELLED.value,
                        },
                    )
            raise
        except Exception as exc:
            close_alignment_lease()
            if terminal_commit_started:
                raise
            if case.terminal_status is CognitiveCaseStatus.OPEN:
                self._apply(
                    case,
                    self._transactions.terminalize(
                        case.case_id,
                        self._command(
                            case,
                            policy_digest,
                            canonical_digest(
                                {"terminal": "failed", "category": type(exc).__name__}
                            ),
                            AuthorityLevel.VALIDATED_CANDIDATE,
                        ),
                        TerminalOutcome.FAILURE,
                    ),
                )
                case.mirror_terminal(CognitiveCaseStatus.FAILED, type(exc).__name__)
                if self._audit:
                    self._audit.append(
                        "case.failed",
                        {
                            "case_id": case.case_id,
                            "request_id": case.request_id,
                            "request_digest": case.input_hash,
                            "category": type(exc).__name__,
                        },
                    )
            raise
        finally:
            close_alignment_lease()

    def _snapshot_id(self) -> str:
        candidate = getattr(self._state_authority, "version", None) or getattr(
            self._state_authority, "snapshot_id", None
        )
        if candidate is None:
            return "world-state:unversioned"
        return f"world-state:{candidate}"
