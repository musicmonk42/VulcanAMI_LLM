"""Atomic production transactions over the constitutional journal."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256

from vulcan.constitution.primitives import AuthorityLevel, canonical_json
from vulcan.graphix.epistemic import EpistemicCommit, dumps_commit

from ._transition_permits import LiveTransitionPermit, MutationPort, TransitionEdge
from .authority import AuthorityError
from .constitutional_journal import ConstitutionalJournal, JournalEvent, SuccessorError
from .episode import ActorBinding, ArtifactRef, CognitiveEpisode, canonical_digest
from .episode_store import episode_from_document
from .journal_stores import (
    JournalEpisodeStore,
    JournalEpistemicStore,
    JournalLineageStore,
    emit_transition,
)
from .state_machine import EpisodeState
from .transactions import (
    CommandAuthority,
    ConstitutionalTransactionService,
    PublicationAuthorization,
)


class JournalConstitutionalTransactionService(ConstitutionalTransactionService):
    """The sole serving write owner; repositories only consume its UoW."""

    def __init__(
        self,
        episodes: JournalEpisodeStore,
        epistemic: JournalEpistemicStore,
        lineage: JournalLineageStore,
        *,
        branch_id: str,
        qualified_release_digest: str,
        verifier_digest: str,
    ) -> None:
        if not (
            episodes.database is epistemic.database
            and episodes.database is lineage.database
        ):
            raise TypeError("journal repositories must share one database owner")
        self._store = episodes
        self._epistemic_store = epistemic
        self._lineage_store = lineage
        self._database = episodes.database
        self._journal = ConstitutionalJournal()
        self._branch_id = branch_id
        self._mutation_port = MutationPort(qualified_release_digest)
        self._verifier_digest = verifier_digest
        self._pending_publication: dict[
            str,
            tuple[
                list[tuple[CognitiveEpisode, CognitiveEpisode, dict[str, object]]],
                bytes,
                str,
            ],
        ] = {}

    def _issue_transition_permit(
        self,
        *,
        episode_id: str,
        edge: TransitionEdge,
        policy_digest: str,
        validation_digest: str,
        snapshot_digest: str,
        expected_prior_episode_digest: str,
    ) -> LiveTransitionPermit:
        rows = self._database.read(
            "SELECT actor_digest FROM episodes WHERE episode_id=?", (episode_id,)
        )
        if not rows:
            raise AuthorityError("admitted episode is missing")
        return self._mutation_port.issue(
            edge=edge,
            actor_digest=rows[0]["actor_digest"],
            episode_id=episode_id,
            policy_digest=policy_digest,
            snapshot_digest=snapshot_digest,
            validation_digest=validation_digest,
            verifier_digest=self._verifier_digest,
            expected_prior_episode_digest=expected_prior_episode_digest,
        )

    def _consume_permit(self, permit, *, edge, head):
        rows = self._database.read(
            "SELECT actor_digest FROM episodes WHERE episode_id=?", (head.episode_id,)
        )
        if not rows:
            raise AuthorityError("admitted command binding is missing")
        actor = rows[0]["actor_digest"]
        try:
            return self._mutation_port.consume(
                permit,
                edge=edge,
                actor_digest=actor,
                constitution_digest=self._mutation_port.constitution_digest,
                episode_id=head.episode_id,
                snapshot_digest=head.snapshot_bundle.state_digest,
                verifier_digest=self._verifier_digest,
                expected_prior_episode_digest=head.digest,
            )
        except PermissionError as exc:
            raise AuthorityError(str(exc)) from exc

    def _binding(self, uow, episode_id: str) -> tuple[str, str]:
        rows = uow.query(
            "SELECT e.actor_digest,c.credential_provenance_digest FROM episodes e "
            "JOIN commands c ON c.command_id=e.command_id WHERE e.episode_id=?",
            (episode_id,),
        )
        if not rows:
            raise AuthorityError("admitted command binding is missing")
        return rows[0]["actor_digest"], rows[0]["credential_provenance_digest"]

    def lineage_head_digest(self) -> str:
        rows = self._database.read(
            "SELECT head_digest FROM lineage_branches WHERE branch_id=?",
            (self._branch_id,),
        )
        return "0" * 64 if not rows else rows[0]["head_digest"]

    def replay(
        self,
        *,
        actor: ActorBinding,
        request_digest: str,
        idempotency_key: str,
    ) -> tuple[CognitiveEpisode, str | None, str] | None:
        actor_digest = canonical_digest(actor.to_json())
        rows = self._database.read(
            "SELECT ed.document,a.content,c.request_digest FROM commands c "
            "JOIN episodes e ON e.command_id=c.command_id "
            "JOIN episode_documents ed ON ed.episode_id=e.episode_id "
            "JOIN terminal_results t ON t.episode_id=e.episode_id "
            "JOIN artifacts a ON a.artifact_digest=t.result_digest "
            "WHERE c.actor_digest=? AND c.operation='chat' AND c.idempotency_key=?",
            (actor_digest, idempotency_key),
        )
        if not rows:
            return None
        if rows[0]["request_digest"] != request_digest:
            raise AuthorityError("idempotency key is bound to different command facts")
        try:
            result = json.loads(bytes(rows[0]["content"]))
            response = result["response"]
            status = result["status"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthorityError("terminal result artifact is invalid") from exc
        if response is not None and not isinstance(response, str):
            raise AuthorityError("terminal result response is invalid")
        if not isinstance(status, str):
            raise AuthorityError("terminal result status is invalid")
        return episode_from_document(rows[0]["document"]), response, status

    def admit_episode(
        self,
        *,
        episode: CognitiveEpisode,
        actor: ActorBinding,
        credential_provenance_digest: str,
        request_digest: str,
        request_id: str,
        idempotency_key: str,
        context_bytes: bytes,
    ) -> CognitiveEpisode:
        with self._database.transaction() as uow:
            actor_digest = self._journal.bind_actor(uow, actor)
            admission = self._mutation_port.issue(
                edge=TransitionEdge.ADMISSION,
                actor_digest=actor_digest,
                episode_id=episode.episode_id,
                policy_digest="0" * 64,
                snapshot_digest=episode.snapshot_bundle.state_digest,
                validation_digest=request_digest,
                verifier_digest=self._verifier_digest,
                expected_prior_episode_digest="0" * 64,
            )
            admission_facts = self._mutation_port.consume(
                admission,
                edge=TransitionEdge.ADMISSION,
                actor_digest=actor_digest,
                constitution_digest=self._mutation_port.constitution_digest,
                episode_id=episode.episode_id,
                snapshot_digest=episode.snapshot_bundle.state_digest,
                verifier_digest=self._verifier_digest,
                expected_prior_episode_digest="0" * 64,
            )
            command_id = f"command:{episode.episode_id}"
            outcome = self._journal.bind_command(
                uow,
                command_id=command_id,
                actor_digest=actor_digest,
                credential_provenance_digest=credential_provenance_digest,
                request_id=request_id,
                request_digest=request_digest,
                operation="chat",
                idempotency_key=idempotency_key,
            )
            if outcome.startswith("replay:"):
                raise AuthorityError("idempotent command is already in progress")
            context_digest = self._journal.put_artifact(
                uow, kind="admitted-context.v1", content=context_bytes
            )
            self._journal.create_episode(
                uow,
                episode_id=episode.episode_id,
                command_id=command_id,
                actor_digest=actor_digest,
                context_digest=context_digest,
            )
            self._store.write_genesis(uow, episode)
            expected = self.lineage_head_digest()
            self._journal.advance_lineage(
                uow,
                branch_id=self._branch_id,
                episode_id=episode.episode_id,
                expected_head_digest=expected,
                new_head_digest=episode.digest,
            )
            emit_transition(
                uow,
                event_type="episode.admitted",
                actor_digest=actor_digest,
                credential_provenance_digest=credential_provenance_digest,
                episode=episode,
            )
            committed_at = datetime.now(timezone.utc)
            receipt_document = {
                "actor_digest": actor_digest,
                "artifact_digests": [context_digest],
                "command_id": command_id,
                "committed_at": committed_at.isoformat(),
                "commit_seq": uow.commit_seq,
                "constitution_digest": admission_facts["constitution_digest"],
                "credential_provenance_digest": credential_provenance_digest,
                "episode_id": episode.episode_id,
                "expires_at_epoch": admission_facts["expires_at_epoch"],
                "issued_at_epoch": admission_facts["issued_at_epoch"],
                "nonce_digest": admission_facts["nonce_digest"],
                "operation": admission_facts["edge"],
                "policy_digest": admission_facts["policy_digest"],
                "predecessor_digest": "0" * 64,
                "qualified_release_digest": admission_facts["qualified_release_digest"],
                "resulting_head_digest": episode.digest,
                "schema_version": "vulcan-transition-receipt/1",
                "snapshot_digest": admission_facts["snapshot_digest"],
                "trust_root_digest": admission_facts["verifier_digest"],
                "validation_digest": admission_facts["validation_digest"],
                "verifier_digest": admission_facts["verifier_digest"],
            }
            receipt_artifact = self._journal.put_artifact(
                uow,
                kind="transition-receipt.v1",
                content=canonical_json(receipt_document),
            )
            uow.emit(
                JournalEvent(
                    "transition.receipt.recorded",
                    actor_digest,
                    credential_provenance_digest,
                    {
                        "episode_id": episode.episode_id,
                        "receipt_artifact_digest": receipt_artifact,
                        "resulting_head_digest": episode.digest,
                    },
                    committed_at,
                )
            )
        return episode

    def epistemic_head(self, episode_id: str) -> EpistemicCommit | None:
        return self._epistemic_store.head(episode_id)

    def authorize_response_publication(
        self,
        episode_id: str,
        auth: CommandAuthority,
        *,
        authorization: PublicationAuthorization,
        response: ArtifactRef,
        response_text: str | None = None,
        response_status: str | None = None,
    ) -> CognitiveEpisode:
        if (
            response_text is None
            or response_status is None
            or sha256(response_text.encode("utf-8")).hexdigest() != response.digest
        ):
            raise AuthorityError("exact publication response bytes are required")
        self._pending_publication[episode_id] = (
            [],
            response_text.encode("utf-8"),
            response_status,
        )
        try:
            return super().authorize_response_publication(
                episode_id, auth, authorization=authorization, response=response
            )
        except BaseException:
            self._pending_publication.pop(episode_id, None)
            raise

    def communicate(self, episode_id: str, auth: CommandAuthority) -> CognitiveEpisode:
        pending = self._pending_publication.get(episode_id)
        if pending is None or not pending[0]:
            raise AuthorityError("communication requires staged publication evidence")
        head = pending[0][-1][1]
        if head.authorization is None or head.response is None:
            raise AuthorityError("communication requires publication evidence")
        return self._advance(
            episode_id,
            auth,
            EpisodeState.COMMUNICATED,
            reason="authorized response communicated",
            minimum=AuthorityLevel.AUTHORIZED_PLAN,
        )

    def commit_epistemic_candidate(
        self,
        episode_id: str,
        auth: CommandAuthority,
        candidate: EpistemicCommit,
    ) -> CognitiveEpisode:
        head = self._store.load(episode_id)
        snapshot = "sha256:" + auth.snapshot_digest
        if (
            candidate.episode_id != episode_id
            or candidate.case_id != episode_id
            or candidate.snapshot_digest != snapshot
            or candidate.authority_principal_id
            != f"principal:{auth.principal.identity_digest[:32]}"
            or candidate.authority_release_digest
            != f"sha256:{auth.principal.release_digest}"
            or candidate.validation_digest != f"sha256:{auth.validation_digest}"
            or candidate.policy_digest != f"sha256:{auth.policy_digest}"
            or candidate.authority_evidence_digest
            != f"sha256:{auth.grant.evidence_digest}"
            or head.digest != auth.expected_prior_episode_digest
            or head.snapshot_bundle is None
            or head.snapshot_bundle.state_digest != auth.snapshot_digest
        ):
            raise AuthorityError("epistemic candidate authority binding mismatch")
        claims = tuple(
            ArtifactRef(
                item.claim_id,
                candidate.commit_digest.removeprefix("sha256:"),
                "graphix-epistemic-claim.v1",
            )
            for item in candidate.claims
        )
        evidence = tuple(
            ArtifactRef(
                item.evidence_id,
                item.content_digest.removeprefix("sha256:"),
                "graphix-epistemic-evidence.v1",
            )
            for item in candidate.evidence
        )
        derivations = tuple(
            ArtifactRef(
                item.derivation_id,
                candidate.commit_digest.removeprefix("sha256:"),
                "graphix-epistemic-derivation.v1",
            )
            for item in candidate.derivations
        )
        successor = self._successor(
            head,
            auth,
            EpisodeState.EPISTEMICALLY_COMMITTED,
            reason="durable Graphix Epistemic head committed",
            claims=claims,
            evidence=evidence,
            derivations=derivations,
        )
        digest = candidate.commit_digest.removeprefix("sha256:")
        prior = (
            None
            if candidate.prior_commit_digest is None
            else candidate.prior_commit_digest.removeprefix("sha256:")
        )
        with self._database.transaction() as uow:
            permit_facts = self._consume_permit(
                auth, edge=TransitionEdge.EPISTEMIC_COMMIT, head=head
            )
            actor_digest, credential = self._binding(uow, episode_id)
            artifact = self._journal.put_artifact(
                uow, kind="epistemic-commit.v1", content=dumps_commit(candidate)
            )
            self._journal.commit_epistemic(
                uow,
                episode_id=episode_id,
                epistemic_digest=digest,
                artifact_digest=artifact,
                prior_digest=prior,
                actor_digest=actor_digest,
                credential_provenance_digest=credential,
            )
            self._epistemic_store.write_document(uow, digest, candidate)
            self._persist_transition(
                uow,
                head,
                successor,
                actor_digest,
                credential,
                permit_facts=permit_facts,
                extra_artifact_digests=(artifact,),
            )
        return successor

    def _successor(
        self,
        head: CognitiveEpisode,
        auth: CommandAuthority,
        target: EpisodeState,
        *,
        reason: str,
        **updates: object,
    ) -> CognitiveEpisode:
        if head.digest != auth.expected_prior_episode_digest:
            raise AuthorityError("expected prior episode digest is stale")
        if (
            head.snapshot_bundle is None
            or head.snapshot_bundle.state_digest != auth.snapshot_digest
        ):
            raise AuthorityError("command snapshot digest is not admitted")
        evidence = ArtifactRef(
            f"command-evidence:{head.episode_id}:{len(head.transitions)}",
            canonical_digest(
                {
                    "grant": auth.grant.evidence_digest,
                    "policy": auth.policy_digest,
                    "principal": auth.principal.identity_digest,
                    "release": auth.principal.release_digest,
                    "snapshot": auth.snapshot_digest,
                    "validation": auth.validation_digest,
                }
            ),
            "constitutional-command-evidence.v1",
        )
        policy = ArtifactRef(
            f"policy-reference:{head.episode_id}:{len(head.transitions)}",
            auth.policy_digest,
            "constitutional-policy-reference.v1",
        )
        return head.transition(
            target,
            reason=reason,
            # The kernel authorizes the promotion, but it must not replace the
            # admitted workload actor as the causal actor of the mutation.
            authority=head.actor.principal_digest,
            snapshot_ids=(auth.snapshot_digest,),
            evidence_refs=(evidence, policy),
            **updates,
        )

    def _persist_transition(
        self,
        uow,
        head,
        successor,
        actor_digest,
        credential,
        *,
        terminal_content: bytes | None = None,
        permit_facts: dict[str, object] | None = None,
        extra_artifact_digests: tuple[str, ...] = (),
    ):
        transition = successor.transitions[-1]
        self._journal.append_transition(
            uow,
            episode_id=head.episode_id,
            from_state=head.state.value,
            to_state=successor.state.value,
            transition_digest=successor.digest,
            actor_digest=actor_digest,
            credential_provenance_digest=credential,
        )
        self._store.advance(uow, head.digest, successor)
        lineage_changed = uow._execute(
            "UPDATE lineage_branches SET head_digest=? "
            "WHERE branch_id=? AND head_episode_id=? AND head_digest=?",
            (successor.digest, self._branch_id, successor.episode_id, head.digest),
        ).rowcount
        if lineage_changed != 1:
            raise SuccessorError("lineage and episode head compare-and-swap diverged")
        persisted_artifacts = [
            self._journal.put_artifact(
                uow,
                kind="episode-transition-document.v1",
                content=successor.canonical_json().encode("utf-8"),
            ),
            *extra_artifact_digests,
        ]
        if successor.state.is_terminal:
            result = self._journal.put_artifact(
                uow,
                kind="terminal-result.v2",
                content=(
                    terminal_content
                    if terminal_content is not None
                    else canonical_json(
                        {
                            "response": None,
                            "status": {
                                EpisodeState.BLOCKED: "blocked",
                                EpisodeState.FAILED: "failed",
                                EpisodeState.CANCELLED: "cancelled",
                            }.get(successor.state, successor.state.value),
                        }
                    )
                ),
            )
            self._journal.record_terminal(
                uow,
                episode_id=successor.episode_id,
                result_digest=result,
                actor_digest=actor_digest,
                credential_provenance_digest=credential,
                terminal_state=successor.state.value,
            )
            persisted_artifacts.append(result)
            uow._execute(
                "UPDATE lineage_membership SET status='past' "
                "WHERE branch_id=? AND episode_id=? AND status='active'",
                (self._branch_id, successor.episode_id),
            )
        emit_transition(
            uow,
            event_type="episode.transitioned",
            actor_digest=actor_digest,
            credential_provenance_digest=credential,
            episode=successor,
        )
        if permit_facts is not None:
            command = uow.query(
                "SELECT command_id FROM episodes WHERE episode_id=?",
                (successor.episode_id,),
            )[0]["command_id"]
            receipt = {
                "actor_digest": actor_digest,
                "artifact_digests": sorted(persisted_artifacts),
                "command_id": command,
                "committed_at": successor.transitions[-1].at.isoformat(),
                "commit_seq": uow.commit_seq,
                "constitution_digest": permit_facts["constitution_digest"],
                "credential_provenance_digest": credential,
                "episode_id": successor.episode_id,
                "expires_at_epoch": permit_facts["expires_at_epoch"],
                "issued_at_epoch": permit_facts["issued_at_epoch"],
                "nonce_digest": permit_facts["nonce_digest"],
                "operation": permit_facts["edge"],
                "policy_digest": permit_facts["policy_digest"],
                "predecessor_digest": head.digest,
                "qualified_release_digest": permit_facts["qualified_release_digest"],
                "resulting_head_digest": successor.digest,
                "schema_version": "vulcan-transition-receipt/1",
                "snapshot_digest": permit_facts["snapshot_digest"],
                "trust_root_digest": permit_facts["verifier_digest"],
                "validation_digest": permit_facts["validation_digest"],
                "verifier_digest": permit_facts["verifier_digest"],
            }
            artifact = self._journal.put_artifact(
                uow,
                kind="transition-receipt.v1",
                content=canonical_json(receipt),
            )
            uow.emit(
                JournalEvent(
                    "transition.receipt.recorded",
                    actor_digest,
                    credential,
                    {
                        "episode_id": successor.episode_id,
                        "receipt_artifact_digest": artifact,
                        "resulting_head_digest": successor.digest,
                    },
                    successor.transitions[-1].at,
                )
            )

    def _advance(self, episode_id, auth, target, *, reason, minimum, **updates):
        pending = self._pending_publication.get(episode_id)
        head = (
            pending[0][-1][1]
            if pending and pending[0]
            else self._store.load(episode_id)
        )
        successor = self._successor(head, auth, target, reason=reason, **updates)
        edge = (
            TransitionEdge.PUBLICATION
            if target
            in {
                EpisodeState.NORMATIVELY_AUTHORIZED,
                EpisodeState.COMMUNICATED,
                EpisodeState.CONSOLIDATED,
            }
            else TransitionEdge.VALIDATION
        )
        permit_facts = self._consume_permit(auth, edge=edge, head=head)
        if pending is not None and target in {
            EpisodeState.NORMATIVELY_AUTHORIZED,
            EpisodeState.COMMUNICATED,
        }:
            pending[0].append((head, successor, permit_facts))
            return successor
        with self._database.transaction() as uow:
            actor_digest, credential = self._binding(uow, episode_id)
            if pending is not None:
                pending[0].append((head, successor, permit_facts))
                for index, (prior, current, transition_permit) in enumerate(pending[0]):
                    self._persist_transition(
                        uow,
                        prior,
                        current,
                        actor_digest,
                        credential,
                        permit_facts=transition_permit,
                        terminal_content=(
                            canonical_json(
                                {
                                    "response": pending[1].decode("utf-8"),
                                    "status": pending[2],
                                }
                            )
                            if index == len(pending[0]) - 1
                            else None
                        ),
                    )
            else:
                self._persist_transition(
                    uow,
                    head,
                    successor,
                    actor_digest,
                    credential,
                    permit_facts=permit_facts,
                )
        self._pending_publication.pop(episode_id, None)
        return successor
