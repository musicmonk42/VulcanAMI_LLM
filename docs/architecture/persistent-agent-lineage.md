# Persistent causal lineage

## Authority and transaction boundary

The old top-level continuity boundary ended at each request-scoped
`CognitiveEpisode`; process restarts and copies had no durable causal identity.
The new continuity authority is one immutable `LineageState` event stream and
one compare-and-swap head per branch. The cognitive microkernel's
`LineageTransactionService` is the only writer. An episode remains the sole
authority for its bounded cognitive transaction: lineage records its admitted
and terminal episode digests but cannot promote a proposal, commit a belief,
authorize a plan, or publish an effect.

Admission reads the current branch head, embeds that exact digest in episode
genesis, durably creates the episode, and advances the lineage head with the
nine authority snapshot references and genesis episode reference. A stale head
fails closed. Transport is withheld until a terminal episode is moved from the
branch's active set into its past set. Episode genesis, its immutable episode
transition/outbox records, and lineage admission commit in one SQLite
transaction. A crash cannot expose only one side of admission.

## Neutral lifecycle semantics

- **Restart:** a newly started process receives a new `instance-*` identifier
  and appends a `restart` event to the same branch. Active episode facts remain.
- **Suspension:** `suspend` advances the branch to a state that rejects episode
  admission. It is not deletion or a claim about process death.
- **Resume:** only a suspended branch may resume, with a new instance identity.
- **Fork:** a source head is explicit; a new CAS head begins under new branch
  and instance identities while retaining copied causal state.
- **Clone:** has the same copy behavior but is separately recorded. It must have
  new branch and instance identities and cannot impersonate its source.
- **Merge:** advances only the named target and records the exact source branch
  and digest. It unions references; it does not erase either branch or assert
  that their identities are identical.

An event stream deterministically reproduces its exact branch head. Tick is
monotonic within a branch; a fork or clone begins its own tick sequence. User
and conversation identifiers remain episode request metadata and are never
accepted as lineage, branch, or instance identifiers.

## Compatibility and removal

`EpisodeAdmissionService(lineage=None)` is the named
`lineage-free-direct-kernel` compatibility adapter for isolated legacy tests and
research callers. It owns no continuity authority and is removed when every
direct kernel caller supplies a durable lineage service. Production composition
always supplies the lineage store, branch, and kernel authority.

These records expose causal-lineage facts only. They do not encode personal
identity, subjecthood, or consciousness.
