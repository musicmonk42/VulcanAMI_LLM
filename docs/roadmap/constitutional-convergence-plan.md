# Constitutional convergence plan

## Goal

Transform Vulcan from overlapping architectural generations into one constitutionally transacted cognitive runtime, then build a persistent causal agent above that foundation.

The plan uses a strangler migration inside the existing repository. It does not create a second framework.

## Maturity vocabulary

Every capability should be tracked as:

- **M0 — Declared:** documented intent only;
- **M1 — Implemented:** source exists in isolation;
- **M2 — Tested:** unit and adversarial evidence exists;
- **M3 — Canonical:** wired into the live production path;
- **M4 — Qualified:** exact built image and restart behavior verified;
- **M5 — Empirical:** held-out experimental predictions validated.

Existence in source is never equivalent to M3.

## Current slice

Constitutional primitives are **M2 — Tested**: Graphix and the microkernel now
share one dependency-light authority lattice, digest value object, identifier
constructors, UTC representation, and strict canonical JSON implementation.
Legacy microkernel snapshot and episode bare-hex fields remain named projection
boundaries and are removed when their persisted v1 schemas migrate to the
canonical `sha256:<hex>` wire form. This representation-only slice does not
change which runtime component may promote authority.

Graphix semantic integrity is **M2 — Tested**: Core extension declarations bind
their namespace and schema version as well as their canonical value, and
Graphix Epistemic commit digests cover propositions, qualifiers, full citations,
uncertainty, limitations, assumptions, counterexamples, contradictions,
authority, snapshot, and prior commit. Strict canonical round-trip and tamper
tests pass without relying on assertions. Graphix remains below M3 because the
compatibility semantic ledger, not the durable Graphix Epistemic head, is still
on the composed request path.

Repository assurance is now **M2 — Tested locally**: the Python 3.11 constitutional gate, workflow lint, real type/format checks, authoritative-episode integration, and optimized-Python parity are executable from one documented dependency set. It remains below M4 until the committed workflows run successfully on configured GitHub runners and the exact built artifact passes restart qualification.

The preceding episode-foundation PR advances authoritative cognitive episodes and snapshot admission toward M3 for the bounded deterministic request path. It does not yet make the new Graphix compiler, Graphix Epistemic ledger, or independent nine-authority ports canonical.

One Graphix path is **M3 — Canonical for bounded arithmetic and typed lookup**:
the production cognitive kernel imports the canonical Graphix runtime rather
than `runtime.semantic`; five code-owned dialects describe interpretation
proposal, validated candidate, arithmetic/domain plan candidate, epistemic
commit candidate, and response projection. Only the registered, non-dynamic
`arithmetic` and `lookup` operations can compile. Graphix output remains a
`VALIDATED_CANDIDATE`, and the constitutional transaction service remains the
only commitment boundary. `runtime.semantic` is now a compatibility re-export
for tests and research callers and is removed when those imports migrate.
Every success and abstention validates a live interpretation envelope; supported
operations also require a validated plan artifact before execution, and every
durable claim records the producing Graphix artifact digest.
Exact built-artifact restart qualification remains below M4.

Atomic episode admission is **M3 — Canonical for the composed request path**: every supported production handler asks the composed kernel admission service to create a snapshot-bound genesis episode before the compatibility case becomes visible. Admission and cancellation release leases, direct unadmitted bypass fails closed, and compatibility ledger mutation rolls back when an episode transition fails. The legacy semantic ledger and migration-only post-genesis bind adapters keep overall episode authority below complete Wave 1 convergence, and exact artifact/restart qualification remains below M4.

Durable episode authority is **M3 — Canonical for the composed request path**:
the admitted genesis and every live transition now advance one SQLite
compare-and-swap head with an immutable transition and audit-outbox row in the
same transaction. Startup verifies and replays all heads, competing writers fail
closed, and transport is withheld unless the durable terminal head matches the
request aggregate. Failpoint, restart, replay, cancellation, privacy, tamper,
and concurrency tests provide M2 evidence. It remains below M4 until the exact
built artifact passes restart/crash qualification. Audit derivation is now the
completed Wave 1.5 slice described below.

Episode-derived audit is **M3 — Canonical for the composed request path**: the
EpisodeStore transaction is now the sole lifecycle commit boundary and its
transactional outbox drives idempotent `episode.transitioned` audit effects.
Audit replay exposes and verifies the prior/resulting episode digest chain;
legacy `case.*` lifecycle validation remains historical-read compatibility only.
Crash-window, duplicate, ordering, tamper, migration, and reconciliation tests
provide M2 evidence.  This remains below M4 until the exact built artifact and
restart behavior are qualified.

The constitutional transaction service is **M3 — Canonical for the composed request path**: typed commands bind
kernel principal/release, current grant, validation/evidence, policy, snapshot,
and expected episode-head digests and commit through durable CAS. Adversarial
principal, stale replay, restart, cancellation, and concurrent-writer tests fail
closed. Publication binds exact output and is communication rather than an
executed external effect. The compatibility case is projection-only and every
composed live transition is submitted by the kernel to the service. This remains
below M4 until the exact built artifact passes crash/restart qualification.

Committed response projection is **M3 — Canonical for the composed request
path**: rendering consumes matching durable episode and Graphix Epistemic heads,
not mutable case collections. Publication and transport bind the exact final
public-text digest, direct case-ledger mutation is rejected, and terminal case
status is projection-only. Allowed abstentions also traverse `COMMUNICATED`,
while unauthorized envelopes expose no response body. Head/snapshot drift,
invented references, text swaps,
cancellation, restart, and concurrent-writer behavior have local M2 evidence.
The retained runtime-semantic object projection is removed with Wave 2.2; exact
built-artifact crash/restart qualification remains below M4.

Durable per-episode epistemic authority is **M3 — Canonical for the composed
request path**: `EpistemicStore` now persists canonical Graphix Epistemic
commits, one CAS head per episode, scoped claim/evidence indexes, and an audit
outbox before the episode projects committed artifacts. Startup verifies commit
digests, chains, heads, references, and provenance and rebuilds only derived
indexes. Concurrent-writer, independent-episode, crash-window, restart, replay,
cross-snapshot reuse, contestation/supersession, corruption, and audit-retry
tests provide M2 evidence. `runtime.semantic` remains behind a named candidate
adapter, so full one-Graphix-path convergence is incomplete; exact artifact and
restart qualification remain below M4.

Persistent causal lineage is **M3 — Canonical for the composed request path**:
one durable CAS head per branch records lineage, branch, and process-instance
identity; monotonic ticks; prior/current state digests; nine authority snapshot
refs; commitments; active/past episodes; and pending effects. Episode genesis
binds the exact admitted lineage head, and terminal transport records completion.
Restart, suspension, resume, fork, clone, merge, deterministic replay, stale
writers, tamper, and atomic admission crash rollback have local M2 evidence.
Episode genesis and lineage admission share one constitutional SQLite
transaction. The `lineage-free-direct-kernel` adapter remains for isolated
callers, and exact built-artifact restart qualification remains below M4. These
are causal facts only, not identity or consciousness claims.

## Wave 0 — Recover executable truth

### 0.1 Repository assurance — M2 (local evidence; hosted qualification pending)

- protect `main`;
- establish a reproducible minimal constitutional test command;
- use a supported Python version in every required job;
- replace misleading check names with real type, format, and integration checks;
- prohibit unhashed dependency fallback in required evidence jobs;
- quarantine legacy cloud and scalability workflows until they exercise the canonical runtime.

**Exit gate:** the microkernel, Graphix, runtime, persistence, security, and built-image suites pass repeatedly from a clean checkout.

### 0.2 Executable architecture inventory — M2 (deterministic local gate)

Generate a matrix of component owner, authority ceiling, runtime reachability, state authority, snapshot implementation, persistence, audit, tests, and maturity.

The repository now has a machine-readable component truth map and documentation-status manifest, with deterministic JSON/Markdown generation. The local gate fails closed on ambiguous ownership, unknown reachability, missing evidence, documentation drift, and public claims below M3. This is documentation assurance only: it changes no runtime authority and remains below M4 until the exact built artifact and restart behavior are qualified.

**Exit gate:** no unexplained production-reachable component and no public capability below M3.

## Wave 1 — Unify the constitutional core

### 1.1 Canonical primitives

Create one constitutional package for digest types, IDs, authority levels, epistemic statuses, canonical time, and canonical serialization. Adapt old wire formats explicitly.

### 1.2 Graphix Epistemic digest repair — M2 (integrity contracts tested; canonical wiring pending)

Make commit digests cover complete propositions, qualifiers, citations, uncertainty, limitations, assumptions, counterexamples, contradictions, derivation rules, authority, snapshot, and prior commit.

**Exit gate:** changing any meaning-bearing field changes the commit digest, and canonical round trips preserve semantic equality.

### 1.3 Explicit state-authority ports — M3 (canonical admission; artifact qualification pending)

Replace reflective production snapshot fallback with explicit `lease_snapshot()` implementations for world, self, social, normative, domain, memory, capability, CSIU, and alignment state.

The composed request path now owns exactly one `StateAuthoritySet`; startup
exercises all nine explicit readers, and episode admission binds independently
versioned, content-derived references. Unimplemented world, self, social, and
normative authorities are explicit disabled schemas rather than aliases of the
legacy world model. Reflective adaptation remains test/migration-only. Isolation,
pinning, expiry, release, restart, failure, and concurrency have local M2 evidence.
Gate C remains incomplete until the exact built runtime and restart behavior pass
qualification; this source and local-test result must not be reported as M4.

**Exit gate:** startup fails closed when an authority cannot produce a faithful content-bound snapshot.

### 1.4 Episode authority — M3 (typed command path canonical; artifact qualification pending)

Complete the migration begun in this PR. Remove independent lifecycle and commitment authority from `CognitiveCase`.

**Exit gate:** no response is released unless its episode reached `COMMUNICATED`; no successful episode finishes without `CONSOLIDATED`.

### 1.5 Audit derivation — M3 (canonical projection; artifact qualification pending)

Make canonical audit consume validated episode transition artifacts rather than enforcing a separate case state machine.

**Exit gate:** replaying audit reconstructs the same episode digest, and audit cannot report a state the episode never entered.

### 1.6 Durable epistemic head — M3 (canonical request path; artifact qualification pending)

Replace the mutable request ledger and in-memory authoritative ledger with a durable Graphix Epistemic commit chain using DB-first persistence and an idempotent outbox.

**Exit gate:** only claims in the committed head may be rendered or used for policy.

### 1.7 Live capability attestation — M3 (canonical runtime path; artifact qualification pending)

Intersect static evidence with live owner identity, release, settings, state digest, and readiness.

`CapabilityManifestAuthority` now owns the public projection and the explicit
capability snapshot port. Its digest-bound attestations fail closed unless
static artifacts, M3 reachability, matching live owner/release, active mode,
current state digest, readiness, and constitutional permission intersect. The
former learning-owner list substitution is retained nowhere. Static
`CapabilityRegistry` is an evidence-input compatibility adapter and may be
removed after evidence is stored directly in governed release attestations.
Exact built-artifact and restart qualification remain below M4.

**Exit gate:** the public endpoint cannot advertise an absent, unhealthy, disabled, or differently released capability.

## Wave 2 — Remove legacy authority seams

### 2.1 Typed production composition — M3 (canonical source path; artifact qualification pending)

Construct the canonical owners directly rather than wrapping `ProductionDeployment` and injecting owners with `setattr`.

The canonical `compose_runtime()` path now uses one explicit composition
specification and typed owner inputs. It constructs each owner once, rejects
missing or duplicate ownership, records reverse close order, closes partial
graphs without masking the startup failure, and never imports or instantiates
`ProductionDeployment`. `LegacyWorldReadOnlyAdapter` exposes no legacy reasoning
or mutation method. The retained
`RuntimeContainer.new(deployment=...)` adapter serves tests/research only and is
removed when those callers migrate to typed inputs. Gate E remains incomplete
until Wave 2.3 import closure and exact built-image restart qualification pass.

### 2.2 One Graphix path — M3 (bounded production path; artifact qualification pending)

Port deterministic arithmetic and typed lookup onto the registered Graphix pipeline; retire duplicate runtime semantic plans, claims, evidence, statuses, and response contracts.

The bounded production kernel now uses `vulcan.graphix.runtime`, with a frozen
code-owned operation allowlist and explicit provenance-link contracts. The old
`vulcan.runtime.semantic` module is compatibility-only and owns no definitions
or authority. Its removal condition is migration of remaining test, output,
case-projection, alignment, and research imports to canonical Graphix types.

### 2.3 Production import closure — M3 (qualification-ready; hosted artifact pending)

The audit import boundary is **M2 — Tested locally**: shared audit values now live in an acyclic contract leaf, the segmented writer is persistence-owned, and the runtime module is a removal-bound compatibility re-export. Fresh-process normal and optimized-Python tests cover import order, application construction, and lifespan cleanup without pytest fixture preloading. Exact built-artifact evidence is still required for M4.

The installed `vulcan` wheel is the sole canonical identity and setuptools uses the standard src layout. Runtime, server, constitutional-test, research, cloud, distributed, and development dependency sets are separated. A machine-enforced transitive allowlist/denylist proves canonical serving does not import legacy orchestration/deployment, retired semantic authority, Arena, cloud/distributed, experimental-memory, or NPT research packages. The exact non-root Linux image qualifies readiness, bounded arithmetic, durable episode/epistemic audit retrieval, same-volume restart, source immutability, and absence of legacy routes. CI publishes artifact and input digests. Gate E is qualification-ready for canonical Linux/Docker serving but remains pending until the committed workflow produces passing exact-image evidence; Windows serving is not claimed.

The serving image also removes denied package trees and offline installation
code physically, and bundles only the read-only evidence files needed for live
capability verification. Import closure includes every parent package
initializer, preventing package-level side effects from bypassing the graph.

The Phase-A application façade is **M2 — Tested locally**: its immutable envelopes expose only command/query dispatch, the ASGI surface is reduced to six routes with documentation disabled, and the reduced composition admits only deterministic arithmetic plus constitutional episode, epistemic, lineage, capability, safety, and audit owners. Exact wheel/image restart qualification remains required for M4.

The minimal Phase-A wheel is **M2 — reproducibly built and locally verified**: a reviewed positive inclusion manifest admits the 59-module serving closure and ten content-bound release inputs; two clean builds produce identical wheel bytes; full RECORD, member digest, excluded-import, undeclared-import, mutation, and outside-checkout controls pass. Installed lifespan qualification remains below M4 until the locked runtime dependencies are available in the qualification environment.

Authenticated actor binding is **M3 — canonical for the Phase-A request path**:
the versioned `vulcan.actor-binding/1` identity binds only tenant, issuer, and
subject, while key ID, token ID, scopes, authentication time, method, and adapter
release remain nonpersistent credential provenance. Request ID and idempotency key
remain envelope transaction facts. Canonical admission has no default actor,
cross-tenant/issuer audit reads fail closed, and legacy actor documents remain
`LEGACY_UNVERIFIED` without inferred provenance or automatic upgrade. The golden
serialization and classification rules are emitted in
`config/actor-binding-schema.json`. Exact installed-artifact and key-rotation
qualification remain required for M4.

The constitutional journal is **M3 — canonical on the Phase-A serving path**:
`ConstitutionalDatabase` owns one SQLite file, connection lifecycle, and caller-owned
units of work. Its schema covers actor-bound commands/idempotency, episodes,
transitions, artifacts, epistemic heads, lineage membership, admitted contexts,
terminal results, a transactional outbox, serialized commit sequences, ordinals,
and receipt chaining. It deliberately contains no Phase-C memory, effect,
reafference, or autobiography tables and performs no production dual writes.
Persisted command facts are independently content-bound, operation-scoped replay
returns the original command identity across request IDs and credential rotation,
and deterministic transaction failpoints prove prior-or-complete restart state.
The stable schema fingerprint is emitted in
`config/constitutional-journal-schema.json`. Phase-A now composes journal-backed
episode, epistemic, and lineage repositories over the caller-owned unit of work.
Admission, epistemic commitment, and terminal publication/lineage completion use
the three indivisible boundaries required by A06. The legacy store implementations
remain compatibility-only for non-Phase-A callers and are removed when those callers
migrate; they are neither opened nor dual-written by the canonical serving graph.
The A07 offline migrator validates and exclusively locks legacy episode, epistemic,
and lineage sources, stops with a reconciliation report on disagreement, imports
deterministically through caller-owned journal transactions, verifies and fsyncs a
temporary database, atomically installs it, and preserves sources read-only. Audit
is now an idempotent journal-receipt projector: JSONL deletion/delivery failure does
not gate readiness, and repeated rebuilds are byte-identical. A parent/child harness
proves named SQL, artifact, head, terminal, outbox, and commit crash boundaries under
normal and optimized Python; an unreached failpoint is `NOT_EXECUTED`. This is M3
source qualification, not installed-artifact M4 evidence. The A08 serving path now separates unpicklable process-local edge permits from receipt-chain-bound durable transition receipts. Principal kinds and compatibility grants are descriptive only on that path; process isolation remains mandatory for untrusted code. Stable external cryptographic signing is not claimed: receipts use an unsigned content digest plus the constitutional journal hash chain.

### 2.4 Offline self-improvement — M3 serving boundary / M2 operator evidence

Keep proposal generation and governed review, but move source installation out of the serving process. Use one approval issuer/verifier/store contract.

`SelfImprovementRuntime` and its approval, CSIU, transaction, and installation
graph are no longer composed or required for serving readiness.  Serving can
only append immutable untrusted proposals and audit their metadata.  Static
import-capability evidence proves that the canonical runtime imports no command,
worktree, package, or install primitive.  CSIU admission uses the distinct
`DisabledCSIUPolicyAuthority` until a real policy owner exists.
The canonical serving-image stage physically removes the offline CLI and legacy
source-application modules, while composition rejects legacy configurations
that request in-process self-improvement.

The offline operator validates proposal/source/policy bindings, requires one
signed human approval schema, applies and tests in a detached worktree, signs a
deployment package, and performs digest-checked install/rollback with audit
evidence.  The serving non-modification boundary is M3; the offline workflow is
M2 and remains below M4 until the exact operator and deployment artifacts pass
crash/restart qualification.

The `vulcan.runtime.self_improvement` import tombstone is removed after
downstream migration. `GovernedSelfImprovementTransaction` is removed after its
isolated single-file apply/gate behavior moves into the offline operator.

**Wave exit gate:** the minimal canonical runtime starts, reasons, persists, restarts, and verifies audit without the legacy deployment graph or full research dependency set.

## Wave 3 — Build the continuous causal agent

### 3.1 Persistent lineage — M3 (canonical request path; qualification pending)

Add lineage, branch, instance, monotonic tick, prior/current state digests, current snapshots, active commitments, episodes, and pending effects under one compare-and-swap head.

### 3.2 Effect protocol — M2 (durable sandbox protocol; canonical wiring pending)

Add authorized policy, durable effect intent, scoped capability, idempotent attempt, and execution receipt.

Canonical contracts, a kernel-only transaction service, durable SQLite
intent/attempt/receipt records, single-use replay-protected capabilities,
transactional audit outbox, deterministic reversible sandbox, compensation,
crash ambiguity, idempotent-target behavior, and operator reconciliation now
have local adversarial and optimized-Python evidence. The
`deterministic-effect-sandbox` adapter is removed when governed effect ports
provide equivalent declarations and qualification. This remains below M3 until
the composed request path authorizes an effect and atomically projects its
intent and receipt into episode and lineage heads; exact-artifact restart
qualification remains below M4.

### 3.3 Reafference and causal autobiography — M2 (durable bounded protocols; canonical wiring pending)

Record expected effect, observation, prediction error, self-caused probability, violated assumptions, world/self updates, and autobiographical consolidation.

Canonical `Observation` and `ReafferenceAssessment` contracts now bind typed
environment changes to expected effects and receipts. A deterministic
closed-loop/yoked adapter produces matched observations with different current
causal control, and deterministic assessment records confounders, delays,
external actors, violated predictions, uncertainty, ownership-related updates,
and Brier calibration telemetry. Typed adapter and assessor validators prevent
provider-authored artifacts from entering the chain, while the effect store
verifies the exact successful receipt/intent/expectation evidence. The
microkernel-only transaction service persists candidates before committing
updates and advances lineage only after receipt validation, observation,
assessment, and update commitment. Idempotent authority ports resume at every
durable crash boundary; schema, calibration, tamper, replay, cancellation, and
concurrency behavior has local evidence. This remains
below M3 because it is not composed with the canonical request/effect path and
its update, lineage, and reafference stores do not yet share one atomic commit;
Gate F remains incomplete pending that wiring and exact-artifact qualification.

The bounded autobiographical-memory protocol now appends content-addressed
causal episodes under a kernel-only DB-first transaction with an idempotent
audit outbox. Observation, inference, prediction, and outcome remain distinct;
correction and tombstone revisions preserve history; and causal retrieval is
explicitly tenant/person/purpose scoped and uses policy, model, and causal edges
rather than textual similarity. Replay, poisoning, correction, restart,
authorized document erasure, retention, cancellation/crash windows,
isolation, chain-authority drift, outbox recovery, and concurrent-writer tests
provide local M2 evidence. A mandatory chain-authority port prevents memory
from silently contradicting durable constitutional history.
`autobiography-outbox-callback` remains until canonical audit supplies a typed
dispatcher. This remains below M3 until the canonical effect/reafference path
commits autobiography and lineage under one transaction.

### 3.4 Center, boundary, and valuation in shadow mode — M2 (intervention-sensitive local evidence; canonical wiring pending)

Estimate operational center, controllability boundary, ownership, temporal currentness, and bounded multidimensional valuation without granting those estimators authority.

Production-denied immutable shadow estimators now operationalize five center variables, seven
boundary categories, and seven independently bounded valuation dimensions.
All outputs bind digest-only provenance and calibration/Brier error estimates;
full-rank center interventions and boundary/valuation perturbations produce
distinguishable downstream predictions. Humans fail closed to autonomous
external-agent classification regardless of apparent control signals. Shadow
plan ordering uses nondominated Pareto fronts and exposes neither an
authorization operation nor a scalar reward.
The implementation is M2 only: it is not composed into the canonical request
path and has no authority, memory mutation, effect, or public subjecthood claim.
The research module is removed or replaced when the external instrumentor
provides held-out causal-identification and calibration evidence outside the
authority plane.

### 3.5 External NPT instrumentor — M2 (synthetic qualification; negative result)

Measure typed closure outside the authority plane and compare held-out intervention predictions against rival models.

The production-denied research package now preserves the seven-dimensional
closure vector, provides ground-truth and blind instrumentors, qualifies
multivariate lagged recovery on a known cyclic synthetic graph, and runs the
corrected preregistered closed-loop/yoked-replay comparison against five named
rivals. Dataset digest,
seed, disjoint intervention-family split, prior-state model features, complexity
penalties, limitations, corrective history, and exact per-outcome scores
are reproducibly reported. `SyntheticCyclicWorld.v2` is removed when qualified
Gate F telemetry provides equivalent matched-history intervention records.

The corrected v2 design invalidates v1 because it permitted false-positive
recovery and same-time leakage. V2 recovers the declared graph with perfect
precision and recall, but its checked flagship result remains deliberately
negative: NPT-specific variables did not add held-out predictive value beyond
every rival. Gate G and M5 remain closed, and the theory must be revised before
stronger claims. Gate F is still incomplete, so no Vulcan telemetry was
analysed and this M2 slice does not override the prerequisite.

**Wave exit gate:** a true closed-loop action and an observation-matched replay produce different, predicted causal and ownership updates, and NPT-specific variables add held-out predictive value.

## Hard gates

| Gate | Requirement | Permits |
|---|---|---|
| A | Reproducibly green constitutional evidence | core convergence |
| B | One authoritative episode lifecycle | durable epistemic authority |
| C | Nine faithful state snapshots | cross-authority decisions |
| D | Microkernel-only promotion and effect authorization | legacy removal |
| E | Canonical runtime independent of legacy deployment | persistent lineage |
| F | Intent/action/receipt/reafference survive restart | center and boundary research |
| G | NPT variables outperform rivals on held-out interventions | stronger scientific claims |

## Anti-goals

- no big-bang rewrite;
- no second AI repository;
- no pure-LLM authority;
- no LLM-authored executable semantics;
- no serving-process source modification;
- no unrestricted self-preservation objective;
- no distributed subject design before single-host semantics are proven;
- no artificial suffering experiments;
- no Boolean consciousness claim;
- no new major subsystem before its prerequisite gate passes.
