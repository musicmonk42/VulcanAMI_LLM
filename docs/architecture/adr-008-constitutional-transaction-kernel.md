# ADR 008: Constitutional Transaction Kernel

## Status

Accepted as the target architecture. This PR implements the first bounded migration slice.

## Context

Vulcan currently contains several overlapping descriptions of a cognitive event: the mutable `CognitiveCase`, the immutable `CognitiveEpisode`, the runtime semantic ledger, Graphix Epistemic, the audit lifecycle, and static capability evidence. Each contains useful work, but more than one can presently appear authoritative.

That ambiguity blocks trustworthy expansion. Persistent agency, tools, learning, self-improvement, and NPT-directed research cannot safely be added while the system lacks one answer to:

- which state was admitted;
- which interpretation was accepted;
- which facts were committed;
- which policy authorized publication or action;
- which effect occurred;
- which lifecycle state was reached.

## Decision

Vulcan will converge on a **Constitutional Transaction Kernel (CTK)**. CTK is the completed form of the existing microkernel architecture, not a replacement repository.

`CognitiveEpisode` is the root aggregate for one bounded cognitive transaction. Every live request must:

1. establish one canonical `case-*`/episode identity;
2. bind one bounded `SnapshotBundle` before semantic interpretation;
3. advance through explicit typed episode transitions;
4. bind interpretation, plans, claims, evidence, derivations, authorization, response, effects, and consolidation by digest;
5. terminate only in an authoritative terminal episode state;
6. release all state leases after completion.

The target live path is:

```text
authenticated ingress
  -> episode and snapshot admission
  -> untrusted language/semantic proposal
  -> validation and grounding
  -> epistemic commitment
  -> normative authorization
  -> response or effect execution
  -> observation
  -> communication
  -> consolidation
```

## Authority model

The authority lattice remains:

```text
UNTRUSTED_PROPOSAL
  -> VALIDATED_CANDIDATE
  -> COMMITTED_BELIEF
  -> AUTHORIZED_PLAN
  -> EXECUTED_EFFECT
```

Only the microkernel may perform promotion. Cognitive organs propose; the kernel commits.

## This PR

The atomic-admission migration slice changes the transaction boundary. Formerly, the runtime created a snapshot-free episode/case projection and the wrapper later mutated its digest by binding a bundle. Now `EpisodeAdmissionService` owns ID generation, bundle admission and validation, snapshot-bound genesis construction, and only then compatibility projection. A failed admission has no visible case and releases acquired leases exactly once.

This migration slice:

- gives the case and episode one canonical `case-*` identity;
- binds the composed runtime's nine-authority snapshot admission function to the kernel;
- preserves direct-kernel tests as an explicitly uncomposed compatibility path while production composition uses the real snapshot admission function;
- advances successful requests through the complete episode lifecycle to `CONSOLIDATED`;
- records abstentions and failures as authoritative terminal episode states;
- binds compatibility-ledger artifacts into the episode by digest;
- releases snapshot leases on every completed handling path;
- composes the legacy semantic kernel as a delegate without copying its resource ownership, and rejects direct production bypass with an unadmitted case;
- requires typed, evidence-bound snapshot rebase commands rather than magic reason strings.

## Deliberate compatibility boundary

The current `runtime.semantic` ledger and Graphix-like plan remain migration projections. This PR binds validated ledger and response artifacts, but it does not yet bind the compiled compatibility plan into the episode or rename the compatibility ledger as the final Graphix Epistemic authority. The next convergence waves will repair full semantic digest coverage, bind canonical plan artifacts, establish one durable epistemic head, derive audit from episode transitions, and remove the duplicate runtime-semantic contracts.

## Consequences

## Live capability authority slice

The old capability boundary combined a static evidence registry used by the
public endpoint with an unrelated kernel/learning string list used as the
capability snapshot. Either view could change without changing the other. The
new sole transaction boundary is `CapabilityManifestAuthority`: it intersects
verified artifact evidence with M3 production reachability, an exact live owner
and release, active configuration, a content-bound state digest, readiness, and
constitutional permission. The public endpoint and episode snapshot port now
consume the same digest-bound attestation and fail closed when the runtime or
owner is unavailable.

`CapabilityRegistry` is retained as the named static-evidence input adapter and
is removed when governed release attestations store that evidence directly.
`RuntimeContainer.capabilities()` is retained as a projection of canonical
public capability identifiers and is removed when all internal callers consume
typed attestations. Neither adapter owns or promotes capability truth. This
slice is M3 on the canonical runtime path but remains below M4 pending exact
built-artifact and restart qualification.

Positive consequences:

- successful public responses can no longer coexist with an episode left at `PERCEIVED`;
- audit retrieval and returned case identity use the same canonical format;
- future persistence work has an immutable episode digest chain to store;
- state admission becomes visible in the live cognitive path;
- later lineage and reafference work can compose episodes rather than bypass them.

Costs and risks:

- compatibility ledger collections remain temporarily mutable;
- production snapshot providers still require explicit per-authority implementations in a later PR;
- audit retains a duplicate lifecycle map until its planned convergence;
- the current Graphix compiler and Graphix Epistemic store are not yet the live path.

## Durable episode authority slice

## Constitutional transaction service slice

The old promotion boundary was the mutable `CognitiveCase`: it constructed
string-labelled transitions and a compatibility response authorization, then
persisted each resulting episode. The new command boundary is
`ConstitutionalTransactionService`. Its commands accept a typed principal and
current grant, validation/evidence and policy digests, the admitted snapshot
digest, and the expected durable episode head. The service re-reads that head
and advances it by compare-and-swap. Only a `SYSTEM_KERNEL` principal can pass
this boundary.

Response publication is distinct from external action. A publication
authorization binds the committed epistemic head, alignment decision and policy
revision, finalizer decision, exact rendered-text digest, privacy and consent
contexts, and the kernel principal/release. Publication advances from
`NORMATIVELY_AUTHORIZED` to `COMMUNICATED`; it does not manufacture an
`EXECUTED_EFFECT`. The latter requires a typed effect authorization and an
execution receipt before observation.

`CognitiveCase`, its runtime-semantic lists, and the direct `CognitiveEpisode`
transition API remain named compatibility adapters. `CognitiveCase` now only
projects already committed heads; it contains no lifecycle promotion or
persistence logic. Remove it when the semantic runtime consumes episodes and a
durable Graphix Epistemic head directly. The
`response-authorization.compat.v1` response-as-effect branch remains solely to
replay pre-migration documents. The composed request path now submits every live
transition through the transaction service, making this slice M3; exact built
artifact and crash/restart qualification are still required for M4.

`CognitiveKernel._bind_direct_compatibility` is the named adapter for legacy
unit callers that instantiate the delegate without production composition. It
creates an isolated filesystem-backed episode store and synthetic snapshot
reference, then uses the same transaction service. It is unreachable from the
composed runtime and is removed when all direct callers use
`EpisodeAdmissionService`.

The old episode transaction boundary ended at assignment of an immutable Python
object to the mutable `CognitiveCase`; process loss erased the authoritative
head, and legacy audit finalization could precede that assignment. The new sole
authority boundary is the microkernel `EpisodeStore`: snapshot-bound genesis and
every subsequent episode transition use a per-episode SQLite compare-and-swap
head, and the immutable transition, new head, and audit-outbox item commit in one
database transaction. The composed handler verifies the required durable head
after an explicit cancellation point and before returning a transportable
result. Terminal compatibility audit events are appended only after that
durable terminal commit.

`CognitiveCase` remains the named working-state adapter, while
`ConstitutionalCognitiveKernel` remains the production admission/transport
adapter. Remove both when the semantic kernel consumes durable episodes
directly. The legacy `CanonicalAudit` lifecycle remains a compatibility
projection until Wave 1.5 makes it consume episode outbox records; undelivered
rows are retained and startup reconciliation delivers them when that sink is
configured. This slice does not make the compatibility semantic ledger or
Graphix Epistemic durable authority.

## Graphix semantic-integrity repair

The old Graphix representation boundary accepted an extension digest without
checking it against its declared schema context, and Epistemic commit digests
omitted multiple meaning-bearing fields. The repaired boundary validates Core
extension digests at object construction and decode, and defines one complete,
canonical Epistemic commit document whose supplied digest is verified on load.
This is an M2 representation and validation change only: the microkernel remains
the sole authority promoter, and the canonical runtime and durable authority
boundaries do not change.

`project_semantic_claim` remains the named legacy claim adapter until Wave 2.2
retires the runtime semantic claim contract. `AuthoritativeClaimLedger` is now
a noncanonical in-memory test adapter; the durable production authority is the
per-episode store described below.

## Faithful state-authority admission

The old state-admission boundary reflectively inspected arbitrary legacy objects,
guessed fields, synthesized fallback digests from `repr`, and reused legacy world,
learning, and self-improvement objects as unrelated authorities. The new boundary
is the explicit nine-member `StateAuthoritySet`. Each member serializes a declared
state document together with its owner, independent revision, release, schema,
and validity window; composition probes and releases all nine before admitting
traffic. Missing or failing readers abort startup.

The legacy world, self, social, and normative projections are explicitly disabled
until their owners expose complete constitutional state. Domain Registry,
Governed Memory, Cognitive Kernel capability truth, the dedicated proposal-only
CSIU policy configuration, and Alignment Registry have named owners. A disabled
authority is itself versioned content-bound state, not a digest of `None`.
`vulcan.testing.snapshots.AttributeSnapshotProvider` is the sole compatibility
adapter and is forbidden in production composition; remove it when migration and
direct-kernel tests construct `StateAuthoritySet` instances.

## Offline improvement authority slice

The old boundary made `SelfImprovementRuntime` a mandatory serving owner.  It
owned CSIU accounting, a second approval implementation, mutable pending state,
and `GovernedSelfImprovementTransaction`; authenticated serving routes could
approve and install candidate source into the running checkout.  The new
transaction boundary ends at an immutable, `UNTRUSTED_PROPOSAL` outbox in the
serving process.  No serving composition or readiness path imports or owns a
source installer, approval issuer, command runner, worktree manager, deployment
packager, or rollback capability.
The canonical serving image also deletes the offline operator and quarantined
legacy apply/drive modules after installation, and startup rejects a legacy
deployment configuration that attempts to enable self-improvement.

Privileged promotion is now an offline operator transaction.  One
`ApprovalAuthority` issues, verifies, and stores the signed, proposal/policy/
source-bound human approval schema.  The operator revalidates source identity
and containment, applies the candidate in a detached isolated git worktree,
runs bounded configured gates, signs a content-bound deployment package, and
installs or rolls it back against compare-and-swap digests outside serving.
Audit evidence records proposal, review, gates, package, deployment, and
rollback without persisting prompts, rationale, provider text, or reasoning
traces.

`vulcan.runtime.self_improvement` is a named import tombstone and is removed
when downstream users import the proposal contract directly.
`GovernedSelfImprovementTransaction` is retained solely as the isolated
single-file application/gate compatibility adapter; remove it when the offline
operator owns a native multi-file patch transaction.  Legacy research drives
remain noncanonical and are not production-importable.  The CSIU snapshot is
owned by the explicit `DisabledCSIUPolicyAuthority`, not by an improvement
runtime; replace it only when a real independently versioned CSIU policy owner
is admitted.

## Durable per-episode epistemic authority slice

The old epistemic boundary was the mutable request-local lists on
`CognitiveCase`, with `AuthoritativeClaimLedger` available as an in-memory
Graphix helper. Episode transitions could therefore project claim references
without a durable Graphix Epistemic commit. The new sole epistemic boundary is
the SQLite-backed `EpistemicStore`, invoked only by
`ConstitutionalTransactionService`: canonical commit bytes, the per-episode
compare-and-swap head, scoped claim/evidence indexes, and an idempotent audit
outbox are committed DB-first. There is intentionally no global cognition head
and no cross-episode total-order chain.

Each commit binds the episode/case identity, admitted snapshot, kernel principal
and release, validation evidence, and policy digest. Startup revalidates every
canonical digest, per-episode chain, head, semantic reference, and cross-episode
provenance link before rebuilding derived indexes. Cross-episode reuse retains
the source episode's evidence content, snapshot, and provenance; it does not
promote that evidence into shared world knowledge.

`adapt_runtime_semantic_candidate` is the named compatibility adapter retained
for `runtime.semantic` claims, evidence, and derivations. It is proposal-only
and is removed when the runtime emits Graphix Epistemic candidates directly.
`CognitiveCase` remains a request-local projection for alignment and rendering;
remove it when those consumers query committed Graphix artifacts. The misnamed
`AuthoritativeClaimLedger` remains noncanonical for legacy dialect tests only
and is removed when those callers use temporary durable stores. The epistemic
DB and episode DB remain distinct owners: a committed epistemic head is
projected into the episode before policy or rendering, and failures withhold
output. Promotion into durable world/domain knowledge remains a future
transaction.

The epistemic audit effect uses the outbox event ID as its idempotency key.
`CanonicalAudit.append_epistemic_commit` returns the existing identical event
both before and after restart and rejects the same ID with different data. The
publication authorization binds the exact current epistemic commit digest,
rather than the enclosing episode digest. Compatibility semantic objects are
rechecked against full object digests embedded in the committed Graphix
candidate before alignment or rendering can observe them.
The former artifact-reference-only transaction method fails closed whenever a
durable epistemic store is composed, preventing an alternate production commit
route.

## Rejected alternatives

## Episode-derived audit slice

The old audit boundary accepted `case.*` calls from mutable orchestration and
independently decided whether that case lifecycle was valid using `_TRANS`.
Consequently an audit state could be appended before, or without, the
corresponding durable episode commit.  The new boundary is the EpisodeStore
transaction: each committed genesis or CAS advancement creates one canonical
`episode.transitioned` outbox artifact, and `CanonicalAudit` idempotently
projects that artifact without promoting lifecycle state.

Every projected event binds the episode and transition identities, from/to
state, prior and resulting episode digests, admitted snapshot digest,
authority/policy/evidence references, and transaction ID.  Audit replay checks
the transition artifact's own canonical digest plus the same prior-digest and
microkernel state-machine chain. Policy references are explicit digest-bound
artifacts in the authoritative episode transition rather than inferred audit
metadata. The composed kernel
no longer emits `case.*`; those schemas, `_TRANS`, and `events_for_case` remain
the named historical-read adapter and are removed after retained v1/v2 audit
archives have completed their governed retention or migration period.

The outbox delivery contract is at-least-once and the audit effect is exactly
once by transition digest.  A crash after episode commit leaves an undelivered
row for startup reconciliation; a crash after append but before marking that
row delivered retries safely against the identical audit event.

EpisodeStore schema v2 migrates v1 outbox payloads from their immutable episode
transition rows and marks them pending for canonical redelivery. This migration
does not alter episode heads or transition history.

### Start a new framework

Rejected because it would duplicate authority, memory, audit, Graphix, alignment, and governance while leaving Vulcan unfinished.

### Add NPT modules directly to the legacy runtime

Rejected because naming a module `center`, `self`, or `consciousness` does not establish the required causal organization and would make the theory circular.

### Keep `CognitiveCase` and `CognitiveEpisode` as co-equal records

Rejected because two lifecycle authorities cannot provide deterministic recovery, audit, or identity.

## Committed response projection slice

The old response boundary rendered `ResponseIR` from mutable `CognitiveCase`
claim, evidence, and derivation lists. The finalizer authorized a text digest,
but the episode's response reference identified only the ResponseIR digest. The
new boundary constructs `ResponseProjection` from matching durable episode and
Graphix Epistemic heads, the admitted snapshot, and accepted interpretation and
plan references. The output adapter can return only ordered committed claim and
citation references; the deterministic renderer realizes values carried by that
typed projection. Both heads are checked again after the asynchronous adapter
call.

Publication now records the exact final public-text digest as the episode
response artifact, and communication fails closed without both that artifact and
publication authorization. Transport independently matches its bytes to the
durable response evidence. `CognitiveCase` lists are the named
`runtime.semantic` compatibility projection and are populated only after the
durable commit; direct append is rejected. `mirror_terminal` similarly mirrors
the durable terminal episode and replaces the former `close()` lifecycle seam.
Remove these projections when the semantic runtime consumes Graphix Epistemic
and `CognitiveEpisode` values directly. Successful episodes durably pass
`COMMUNICATED` before optional durable consolidation; after a lost HTTP return,
the committed communication remains authoritative and its exact response digest
is recoverable without authorizing different text.

An allowed abstention is public language and therefore follows the same
`NORMATIVELY_AUTHORIZED -> COMMUNICATED -> CONSOLIDATED` lifecycle as an allowed
strict answer; `ABSTAINED` remains its compatibility result classification, not
a competing episode terminal state. Blocked, errored, and cancelled results do
not carry response publication evidence and their transport projection contains
no response body. A conforming finalizer must return the exact render artifact it
evaluated and may allow only that artifact's text.
